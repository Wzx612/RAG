"""Phase 4B: dense retrieval against the real database and real embedder.

The integration tests use the Phase 3 BGE model and the imported 150-chunk
corpus. They skip rather than fail when the corpus or database is absent, the
same convention as ``test_database.py``.

Four conclusions are deliberately kept apart, because they fail for different
reasons and a passing one does not imply another:

1. retrieval logic is correct (real query -> real chunks -> valid metadata);
2. ``DenseConfig.ef_search`` reaches the live pgvector GUC;
3. the query can use the HNSW index (verified with a diagnostic plan);
4. at the current 150-chunk scale, ``ef_search`` does not change the natural
   plan or the top-10 result set (a planner/scale observation, not a rule).

Stubs are used only to exercise failure paths that must not load the model or
touch the database; every retrieval assertion uses real embeddings and real
PostgreSQL.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
import pytest
from pgvector import Vector
from psycopg_pool import PoolTimeout
from pydantic import ValidationError
from src.db import (
    EF_SEARCH_DEFAULT,
    connection,
    current_ef_search,
    ef_search,
    get_pool,
)
from src.ingestion import IngestionError
from src.ingestion.embedder import SentenceTransformerEmbedder
from src.retrieval import RetrievalError
from src.retrieval.dense import DENSE_SEARCH_SQL, TOP_K_MAX, DenseRetriever
from src.schema.retrieval import DenseConfig, SearchHit

DATA = Path(__file__).resolve().parents[1] / "data" / "eval_queries.json"


# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def gold_query() -> dict[str, Any]:
    """Q-001 and its gold doc ids, read (never written) from the eval set."""
    queries = json.loads(DATA.read_text(encoding="utf-8"))
    return queries[0]


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    """The Phase 3 model, loaded once for the whole module."""
    return SentenceTransformerEmbedder()


@pytest.fixture
def retriever(embedder: SentenceTransformerEmbedder) -> DenseRetriever:
    """A default-shaped retriever: Top-10, ef_search 40."""
    return DenseRetriever(DenseConfig(top_n=10, ef_search=40), embedder=embedder)


@pytest.fixture
def populated_corpus(conn: psycopg.Connection) -> None:
    """Skip integration tests when Phase 3 has not imported the corpus.

    Named distinctly from the session-scoped ``corpus`` fixture in
    ``conftest.py``, which is the JSON documents rather than the database rows.
    """
    row = conn.execute("SELECT count(*) FROM chunks").fetchone()
    if row is None or row[0] == 0:
        pytest.skip("corpus not imported -- run `make ingest`")


# ------------------------------------------------------------------- test doubles


class _StubEmbedder:
    """A finite 1024-dim provider for paths that must not load the real model.

    It is never used for a retrieval assertion -- only to reach error handling
    without an 8-second model load. The vector is deliberately not random: the
    point is a shape-valid input, not a plausible embedding.
    """

    dimension = 1024

    def embed_documents(
        self,
        texts: Sequence[str],
        on_batch: Any = None,
    ) -> list[list[float]]:
        """Return one fixed-width vector per input."""
        return [[1.0] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Return a fixed-width query vector."""
        return [1.0] * self.dimension


class _WrongDimensionEmbedder(_StubEmbedder):
    """A provider that violates the 1024-dimension contract."""

    def embed_query(self, text: str) -> list[float]:
        """Return three components where 1024 are required."""
        return [1.0, 2.0, 3.0]


class _FailingEmbedder(_StubEmbedder):
    """A provider whose backend is down."""

    def embed_query(self, text: str) -> list[float]:
        """Fail the way the Phase 3 backend does after its retries."""
        raise IngestionError("embedding backend unavailable")


class _RecordingHandler(logging.Handler):
    """Collect records from the retriever's logger without changing propagation."""

    def __init__(self) -> None:
        """Start with an empty record list."""
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Store one record."""
        self.records.append(record)


@contextmanager
def _unused_connection() -> Iterator[Any]:
    """Yield a placeholder; a monkeypatched ``_search`` must ignore it."""
    yield object()


@contextmanager
def _must_not_connect() -> Iterator[Any]:
    """Fail the test if validation reaches the database."""
    pytest.fail("database must not be touched")
    yield  # pragma: no cover


@contextmanager
def _unreachable_connection() -> Iterator[Any]:
    """Fail on entry the way ``get_pool()`` does when PostgreSQL is down."""
    raise ConnectionError("cannot reach PostgreSQL at test:5432")
    yield  # pragma: no cover


@contextmanager
def _exhausted_pool() -> Iterator[Any]:
    """Simulate a pool borrow timeout."""
    raise PoolTimeout("could not get a connection from the pool")
    yield  # pragma: no cover


@contextmanager
def _forced_seqscan_off() -> Iterator[psycopg.Connection]:
    """Diagnostic planner setup: force HNSW for one transaction only.

    This is never the production default. It exists because at 150 chunks the
    natural plan is a sequential scan, which ignores ``ef_search`` entirely.
    """
    with connection() as conn, conn.transaction():
        conn.execute("SET LOCAL enable_seqscan = off", prepare=False)
        yield conn


def _stored_vector(conn: psycopg.Connection) -> Vector:
    """Use a persisted embedding for plan-only checks (no model load)."""
    row = conn.execute("SELECT embedding FROM chunks ORDER BY chunk_id LIMIT 1").fetchone()
    assert row is not None
    return Vector(row[0].to_list())


def _explain(
    conn: psycopg.Connection,
    sql: str,
    params: tuple[Any, ...],
) -> str:
    """Return one EXPLAIN (ANALYZE, BUFFERS) plan as text."""
    rows = conn.execute(
        "EXPLAIN (ANALYZE, BUFFERS) " + sql,
        params,
        prepare=False,
    ).fetchall()
    return "\n".join(row[0] for row in rows)


def _assert_pool_ef_default() -> set[int]:
    """Assert every connection the pool hands out carries no swept ef_search."""
    pool = get_pool()
    seen: set[int] = set()
    for _ in range(pool.max_size * 2 + 2):
        with connection() as conn:
            pid = conn.execute("SELECT pg_backend_pid()").fetchone()[0]
            seen.add(pid)
            leaked = current_ef_search(conn)
            assert leaked == EF_SEARCH_DEFAULT, (
                f"connection {pid} came back with ef_search={leaked}, expected {EF_SEARCH_DEFAULT}"
            )
    return seen


# ------------------------------------------------------ A. real retrieval result


def test_a_real_query_returns_real_chunk_hits(
    populated_corpus: None,
    embedder: SentenceTransformerEmbedder,
    retriever: DenseRetriever,
    gold_query: dict[str, Any],
) -> None:
    """The whole path: BGE query -> pgvector -> chunks JOIN documents."""
    hits = retriever.retrieve(gold_query["query"])

    assert len(hits) == 10
    assert [hit.rank for hit in hits] == list(range(1, 11))
    assert {hit.source for hit in hits} == {"dense"}
    assert {"DOC-0001", "DOC-0005"} <= {hit.doc_id for hit in hits}

    for hit in hits:
        assert isinstance(hit, SearchHit)
        assert hit.chunk_id and hit.doc_id
        assert hit.content
        assert hit.title and hit.category and hit.card_tier
        assert hit.chunk_strategy == "document"
        assert hit.chunk_index == 0
        assert hit.distance is not None
        assert math.isfinite(hit.distance)
        assert 0.0 <= hit.distance <= 2.0
        assert hit.score == pytest.approx(1.0 - hit.distance)

    distances = [hit.distance for hit in hits]
    assert distances == sorted(distances), "hits are not ordered by ascending distance"


def test_metadata_matches_the_persisted_chunk_and_document(
    populated_corpus: None,
    retriever: DenseRetriever,
    gold_query: dict[str, Any],
    conn: psycopg.Connection,
) -> None:
    """Every returned field is read from the real tables, not synthesised."""
    hits = retriever.retrieve(gold_query["query"], top_k=5)
    assert len(hits) == 5

    for hit in hits:
        assert hit.chunk_strategy is not None
        row = conn.execute(
            """
            SELECT c.doc_id, c.chunk_strategy, c.chunk_index, c.content,
                   d.title, d.category, d.card_tier, d.effective_year, d.policy_code
            FROM chunks AS c
            JOIN documents AS d ON d.doc_id = c.doc_id
            WHERE c.chunk_strategy = %s AND c.chunk_id = %s
            """,
            (hit.chunk_strategy, hit.chunk_id),
        ).fetchone()
        assert row is not None
        assert row == (
            hit.doc_id,
            hit.chunk_strategy,
            hit.chunk_index,
            hit.content,
            hit.title,
            hit.category,
            hit.card_tier,
            hit.effective_year,
            hit.policy_code,
        )


# -------------------------------------------------- B. data model and Top-K shape


def test_top_k_is_exact_and_chunk_identities_are_unique(
    populated_corpus: None,
    retriever: DenseRetriever,
    gold_query: dict[str, Any],
) -> None:
    """The join must not inflate or duplicate the chunk-level Top-K."""
    for limit in (1, 5, 10):
        hits = retriever.retrieve(gold_query["query"], top_k=limit)
        identities = [(hit.chunk_strategy, hit.chunk_id) for hit in hits]
        assert len(hits) == limit
        assert len(set(identities)) == limit
        assert [hit.rank for hit in hits] == list(range(1, limit + 1))


def test_config_top_n_is_the_default_and_the_argument_overrides_it(
    populated_corpus: None,
    embedder: SentenceTransformerEmbedder,
    gold_query: dict[str, Any],
) -> None:
    """DenseConfig.top_n is real Top-K; retrieve(top_k=...) overrides it."""
    retriever = DenseRetriever(DenseConfig(top_n=5, ef_search=40), embedder=embedder)
    assert len(retriever.retrieve(gold_query["query"])) == 5
    assert len(retriever.retrieve(gold_query["query"], top_k=7)) == 7


def test_every_returned_doc_id_exists_in_documents(
    populated_corpus: None,
    retriever: DenseRetriever,
    gold_query: dict[str, Any],
    conn: psycopg.Connection,
) -> None:
    """The FK-backed JOIN means no hit can point at a missing document."""
    hits = retriever.retrieve(gold_query["query"], top_k=20)
    doc_ids = {hit.doc_id for hit in hits}
    rows = conn.execute(
        "SELECT doc_id FROM documents WHERE doc_id = ANY(%s)",
        (list(doc_ids),),
    ).fetchall()
    assert {row[0] for row in rows} == doc_ids


def test_database_embedding_dimension_matches_the_embedder(
    populated_corpus: None,
    embedder: SentenceTransformerEmbedder,
    conn: psycopg.Connection,
) -> None:
    """The live column width is the same 1024 the query provider emits."""
    row = conn.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        WHERE a.attrelid = 'chunks'::regclass
          AND a.attname = 'embedding' AND a.attnum > 0 AND NOT a.attisdropped
        """
    ).fetchone()
    assert row is not None
    assert row[0] == f"vector({embedder.dimension})"
    assert embedder.dimension == 1024


# ------------------------------------------- C. configuration and ef_search scope


def test_ef_search_reaches_the_live_guc_and_the_hnsw_scan(
    populated_corpus: None,
    conn: psycopg.Connection,
) -> None:
    """Read the GUC back inside the query transaction, then count HNSW rows.

    Forcing seqscan off is diagnostic only; it makes the HNSW scan observable at
    this corpus size. If the configuration never reached the GUC, the read-back
    would be 40 and the row cap would not be 7.
    """
    vector = _stored_vector(conn)
    retriever = DenseRetriever(DenseConfig(top_n=50, ef_search=7))

    with conn.transaction():
        conn.execute("SET LOCAL enable_seqscan = off", prepare=False)
        rows, actual_ef = retriever._search(conn, vector.to_list(), 50)

    assert actual_ef == 7
    assert len(rows) == 7
    assert current_ef_search(conn) == EF_SEARCH_DEFAULT


def test_public_retrieve_consumes_the_configured_ef_search(
    populated_corpus: None,
    embedder: SentenceTransformerEmbedder,
    gold_query: dict[str, Any],
) -> None:
    """End-to-end public path under a forced HNSW plan: ef=7 caps Top-K at 7."""
    retriever = DenseRetriever(
        DenseConfig(top_n=50, ef_search=7),
        embedder=embedder,
        connection_factory=_forced_seqscan_off,
    )
    hits = retriever.retrieve(gold_query["query"])
    assert len(hits) == 7
    assert [hit.rank for hit in hits] == list(range(1, 8))


def test_ef_search_does_not_leak_into_the_connection_pool(
    populated_corpus: None,
    embedder: SentenceTransformerEmbedder,
    gold_query: dict[str, Any],
) -> None:
    """A swept value is transaction-local; the next borrower sees 40."""
    retriever = DenseRetriever(DenseConfig(top_n=5, ef_search=137), embedder=embedder)
    assert len(retriever.retrieve(gold_query["query"])) == 5

    seen = _assert_pool_ef_default()
    assert seen, "the pool handed out no connections to inspect"


def test_different_retriever_configs_do_not_interfere(
    populated_corpus: None,
    embedder: SentenceTransformerEmbedder,
    gold_query: dict[str, Any],
) -> None:
    """Two retrievers with different ef_search values can run back to back."""
    low = DenseRetriever(DenseConfig(top_n=5, ef_search=7), embedder=embedder)
    high = DenseRetriever(DenseConfig(top_n=5, ef_search=40), embedder=embedder)

    assert len(low.retrieve(gold_query["query"])) == 5
    assert len(high.retrieve(gold_query["query"])) == 5
    _assert_pool_ef_default()


# ------------------------------------------------------------ D. failure paths


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_is_rejected_before_embedding(query: str) -> None:
    """Empty/whitespace queries never reach the model or the database."""
    retriever = DenseRetriever(
        embedder=_StubEmbedder(),
        connection_factory=_must_not_connect,
    )
    with pytest.raises(ValueError, match="non-empty"):
        retriever.retrieve(query)


def test_non_string_query_is_rejected() -> None:
    """A non-string query is a programming error, reported before any work."""
    retriever = DenseRetriever(
        embedder=_StubEmbedder(),
        connection_factory=_must_not_connect,
    )
    with pytest.raises(ValueError, match="must be a str"):
        retriever.retrieve(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, -1, TOP_K_MAX + 1, True, 3.0, "3"])
def test_invalid_top_k_is_rejected(top_k: Any) -> None:
    """Top-K must be an int in range; bool is not a count."""
    retriever = DenseRetriever(
        embedder=_StubEmbedder(),
        connection_factory=_must_not_connect,
    )
    with pytest.raises(ValueError, match="top_k must be"):
        retriever.retrieve("anything", top_k=top_k)


def test_embedding_failure_is_reported_and_does_not_touch_the_database() -> None:
    """The embedding stage fails loudly, with its cause preserved."""
    retriever = DenseRetriever(
        embedder=_FailingEmbedder(),
        connection_factory=_must_not_connect,
    )
    with pytest.raises(RetrievalError, match="query embedding failed") as excinfo:
        retriever.retrieve("anything")
    assert isinstance(excinfo.value.__cause__, IngestionError)


def test_dimension_mismatch_is_reported_before_the_database() -> None:
    """A wrong-width query vector is rejected, never padded or truncated."""
    retriever = DenseRetriever(
        embedder=_WrongDimensionEmbedder(),
        connection_factory=_must_not_connect,
    )
    with pytest.raises(RetrievalError, match="dimension 3, expected 1024") as excinfo:
        retriever.retrieve("anything")
    assert isinstance(excinfo.value.__cause__, IngestionError)


def test_database_connection_failure_is_reported() -> None:
    """A dead database becomes RetrievalError, not an empty list."""
    retriever = DenseRetriever(
        embedder=_StubEmbedder(),
        connection_factory=_unreachable_connection,
    )
    with pytest.raises(RetrievalError, match="database connection failed") as excinfo:
        retriever.retrieve("anything")
    assert isinstance(excinfo.value.__cause__, ConnectionError)


def test_pool_exhaustion_is_reported() -> None:
    """Waiting for a pooled connection has its own, clearer failure message."""
    retriever = DenseRetriever(
        embedder=_StubEmbedder(),
        connection_factory=_exhausted_pool,
    )
    with pytest.raises(RetrievalError, match="waiting for a database connection") as excinfo:
        retriever.retrieve("anything")
    assert isinstance(excinfo.value.__cause__, PoolTimeout)


def test_query_timeout_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server-side cancellation is surfaced as a timeout, not swallowed.

    Simulated: a real statement_timeout is a server/connection setting and this
    phase does not add one. The production handler is exercised.
    """
    retriever = DenseRetriever(
        embedder=_StubEmbedder(),
        connection_factory=_unused_connection,
    )

    def _cancelled(
        conn: psycopg.Connection,
        vector: Sequence[float],
        top_k: int,
    ) -> tuple[list[tuple[Any, ...]], int]:
        raise psycopg.errors.QueryCanceled("canceling statement due to statement timeout")

    monkeypatch.setattr(retriever, "_search", _cancelled)
    with pytest.raises(RetrievalError, match="cancelled or timed out") as excinfo:
        retriever.retrieve("anything")
    assert isinstance(excinfo.value.__cause__, psycopg.errors.QueryCanceled)


def test_database_exception_reports_and_rolls_back_the_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real failing query inside ef_search rolls back and leaves the pool clean."""
    retriever = DenseRetriever(embedder=_StubEmbedder())

    def _failing_query(
        conn: psycopg.Connection,
        vector: Sequence[float],
        top_k: int,
    ) -> tuple[list[tuple[Any, ...]], int]:
        with ef_search(conn, 7):
            conn.execute("SELECT 1/0", prepare=False)
        raise AssertionError("unreachable")

    monkeypatch.setattr(retriever, "_search", _failing_query)
    with pytest.raises(RetrievalError, match="database error") as excinfo:
        retriever.retrieve("anything")

    assert isinstance(excinfo.value.__cause__, psycopg.errors.DivisionByZero)
    _assert_pool_ef_default()


def test_no_rows_returns_an_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty result is a valid result, not an exception.

    Simulated: the imported corpus has rows, and this phase must not delete any
    to produce an empty database.
    """
    retriever = DenseRetriever(
        embedder=_StubEmbedder(),
        connection_factory=_unused_connection,
    )
    monkeypatch.setattr(
        retriever,
        "_search",
        lambda conn, vector, top_k: ([], 40),
    )
    assert retriever.retrieve("anything") == []


def test_non_finite_distance_is_reported_and_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NaN distance is rejected, never ranked or converted into a score."""
    nan_row = (
        "chunk-x",
        "DOC-0001",
        "document",
        0,
        "content",
        float("nan"),
        "title",
        "category",
        "platinum",
        2024,
        "PC-1",
    )
    retriever = DenseRetriever(
        embedder=_StubEmbedder(),
        connection_factory=_unused_connection,
    )
    monkeypatch.setattr(
        retriever,
        "_search",
        lambda conn, vector, top_k: ([nan_row], 40),
    )

    handler = _RecordingHandler()
    target = logging.getLogger("apex.src.retrieval.dense")
    target.addHandler(handler)
    try:
        with pytest.raises(RetrievalError, match="non-finite"):
            retriever.retrieve("anything")
    finally:
        target.removeHandler(handler)

    assert any("rejected database rows" in record.getMessage() for record in handler.records)


# ------------------------------------------------------------- config validation


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_n", 0),
        ("top_n", 1001),
        ("m", 1),
        ("m", 101),
        ("ef_construction", 3),
        ("ef_construction", 1001),
        ("ef_search", 0),
        ("ef_search", 1001),
        ("top_n", True),
        ("ef_search", False),
    ],
)
def test_dense_config_rejects_out_of_range_or_bool_values(
    field: str,
    value: Any,
) -> None:
    """Bad values fail at config construction with the field named."""
    with pytest.raises(ValidationError, match=field):
        DenseConfig(**{field: value})


def test_dense_config_defaults_are_unchanged() -> None:
    """The accepted Phase 1 defaults stay exactly as they were."""
    config = DenseConfig()
    assert config.top_n == 50
    assert config.m == 16
    assert config.ef_construction == 64
    assert config.ef_search == 40
    assert config.metric == "cosine"


def test_ef_search_below_top_n_is_legal_for_the_parameter_sweep() -> None:
    """Experiment 4 must be able to ask for ef<top_k; runtime warns instead."""
    config = DenseConfig(top_n=50, ef_search=10)
    assert config.ef_search == 10


def test_dense_config_validates_assignment() -> None:
    """Mutating a config cannot smuggle an invalid value past construction."""
    config = DenseConfig()
    with pytest.raises(ValidationError):
        config.ef_search = 0


# ---------------------------------------------------------------- E. query plans


def test_natural_plan_at_150_chunks_is_a_seq_scan(
    populated_corpus: None,
    conn: psycopg.Connection,
) -> None:
    """Record the current planner choice; this is not an HNSW failure."""
    count = conn.execute("SELECT count(*) FROM chunks").fetchone()
    assert count is not None
    if count[0] != 150:
        pytest.skip(f"planner expectation pinned to 150 chunks; found {count[0]}")

    vector = _stored_vector(conn)
    previous = conn.prepare_threshold
    conn.prepare_threshold = None
    try:
        with ef_search(conn, 40):
            plan = _explain(conn, DENSE_SEARCH_SQL, (vector, 10))
    finally:
        conn.prepare_threshold = previous

    assert "Seq Scan on chunks" in plan, plan
    assert "chunks_embedding_hnsw_idx" not in plan, plan


def test_forced_plan_uses_hnsw_and_returns_top_k(
    populated_corpus: None,
    conn: psycopg.Connection,
) -> None:
    """Diagnosis only: seqscan off proves the inner ANN stage can use HNSW."""
    vector = _stored_vector(conn)
    previous = conn.prepare_threshold
    conn.prepare_threshold = None
    try:
        with conn.transaction():
            conn.execute("SET LOCAL enable_seqscan = off", prepare=False)
            with ef_search(conn, 40):
                plan = _explain(conn, DENSE_SEARCH_SQL, (vector, 10))
    finally:
        conn.prepare_threshold = previous

    hnsw_lines = [line for line in plan.splitlines() if "chunks_embedding_hnsw_idx" in line]
    assert hnsw_lines, plan
    assert "rows=10" in hnsw_lines[0], plan
    assert "Seq Scan on chunks" not in plan, plan


def test_natural_plan_ignores_ef_search_at_current_scale(
    populated_corpus: None,
    embedder: SentenceTransformerEmbedder,
    gold_query: dict[str, Any],
    conn: psycopg.Connection,
) -> None:
    """At 150 chunks ef=7 cannot cap a Seq Scan: it still returns Top-50."""
    retriever = DenseRetriever(DenseConfig(top_n=50, ef_search=7), embedder=embedder)
    hits = retriever.retrieve(gold_query["query"])
    assert len(hits) == 50


def test_ef_search_has_no_observable_effect_on_top10_at_current_scale(
    populated_corpus: None,
    embedder: SentenceTransformerEmbedder,
    gold_query: dict[str, Any],
    conn: psycopg.Connection,
) -> None:
    """Current-corpus observation, not a guarantee: ef=10 already reaches top-10."""
    vector = embedder.embed_query(gold_query["query"])
    low = DenseRetriever(DenseConfig(top_n=10, ef_search=10))
    high = DenseRetriever(DenseConfig(top_n=10, ef_search=50))

    with conn.transaction():
        conn.execute("SET LOCAL enable_seqscan = off", prepare=False)
        rows_low, ef_low = low._search(conn, vector, 10)
        rows_high, ef_high = high._search(conn, vector, 10)

    assert ef_low == 10
    assert ef_high == 50
    assert len(rows_low) == len(rows_high) == 10
    assert {row[0] for row in rows_low} == {row[0] for row in rows_high}


# ------------------------------------------------------------ observability


def test_logs_record_counts_and_timing_but_not_the_query(
    populated_corpus: None,
    retriever: DenseRetriever,
    gold_query: dict[str, Any],
) -> None:
    """Stage, latency, count and ef_search are logged; query text is not."""
    handler = _RecordingHandler()
    target = logging.getLogger("apex.src.retrieval.dense")
    target.addHandler(handler)
    try:
        retriever.retrieve(gold_query["query"])
    finally:
        target.removeHandler(handler)

    messages = [record.getMessage() for record in handler.records]
    assert any("hits=10" in message for message in messages)
    assert any("ef_search=40" in message for message in messages)
    assert any("embed_ms=" in message and "db_ms=" in message for message in messages)
    assert all(gold_query["query"] not in message for message in messages)


def test_truncation_warning_is_emitted_once_for_a_config(
    populated_corpus: None,
    embedder: SentenceTransformerEmbedder,
    gold_query: dict[str, Any],
) -> None:
    """ef<top_k under HNSW is visible as a warning, not a silent short list."""
    retriever = DenseRetriever(
        DenseConfig(top_n=50, ef_search=7),
        embedder=embedder,
        connection_factory=_forced_seqscan_off,
    )
    handler = _RecordingHandler()
    target = logging.getLogger("apex.src.retrieval.dense")
    target.addHandler(handler)
    try:
        first = retriever.retrieve(gold_query["query"])
        second = retriever.retrieve(gold_query["query"])
    finally:
        target.removeHandler(handler)

    assert len(first) == len(second) == 7
    warnings = [
        record.getMessage()
        for record in handler.records
        if "limits the HNSW scan" in record.getMessage()
    ]
    assert len(warnings) == 1, warnings
