"""Phase 5A: sparse retrieval against the real database.

The integration tests use the imported 150-chunk corpus and skip when
PostgreSQL or the corpus is absent. Stubs are used only for failure paths that
must not touch the database.

What is asserted separately:

1. the production SQL returns real chunks with real metadata and scores;
2. Top-K, uniqueness and ordering are exact;
3. OR-mode natural queries retrieve the labelled docs; AND-mode is the strict
   control that demonstrates why OR is the default;
4. Chinese and mixed queries exercise the CJK substring fallback honestly;
5. invalid input and database failures become ``ValueError`` / ``RetrievalError``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg_pool import PoolTimeout
from pydantic import ValidationError
from src.evaluation.metrics import recall_at_k
from src.retrieval import RetrievalError
from src.retrieval.sparse import (
    TOP_K_MAX,
    SparseRetriever,
    _cjk_bigrams,
    _fts_query_text,
    _rows_to_hits,
)
from src.schema.retrieval import SearchHit, SparseConfig

DATA = Path(__file__).resolve().parents[1] / "data" / "eval_queries.json"


# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def eval_queries() -> list[dict[str, Any]]:
    """The labelled queries, read but never written."""
    return json.loads(DATA.read_text(encoding="utf-8"))


@pytest.fixture
def populated_corpus(conn: psycopg.Connection) -> None:
    """Skip integration tests when Phase 3 has not imported the corpus."""
    row = conn.execute("SELECT count(*) FROM chunks").fetchone()
    if row is None or row[0] == 0:
        pytest.skip("corpus not imported -- run `make ingest`")


@pytest.fixture
def retriever() -> SparseRetriever:
    """A default-shaped retriever: Top-10, OR mode, english FTS."""
    return SparseRetriever(SparseConfig(top_n=10))


# ------------------------------------------------------------------ test doubles


@contextmanager
def _must_not_connect() -> Iterator[Any]:
    """Fail the test if validation reaches the database."""
    pytest.fail("database must not be touched")
    yield  # pragma: no cover


@contextmanager
def _unreachable_connection() -> Iterator[Any]:
    """Fail on entry the way the pool does when PostgreSQL is down."""
    raise ConnectionError("cannot reach PostgreSQL at test:5432")
    yield  # pragma: no cover


@contextmanager
def _exhausted_pool() -> Iterator[Any]:
    """Simulate a pool borrow timeout."""
    raise PoolTimeout("could not get a connection from the pool")
    yield  # pragma: no cover


class _FakeResult:
    """The narrow result surface ``fetchall`` uses."""

    def __init__(self, rows: Sequence[Sequence[Any]]) -> None:
        """Store the rows."""
        self._rows = rows

    def fetchall(self) -> list[Sequence[Any]]:
        """Return the stored rows."""
        return list(self._rows)


class _FailingConnection:
    """A connection whose ``execute`` raises a chosen exception."""

    def __init__(self, exc: BaseException) -> None:
        """Store the exception to raise."""
        self._exc = exc

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Raise the configured exception."""
        del args, kwargs
        raise self._exc


class _RowsConnection:
    """A connection returning fixed rows from ``execute``."""

    def __init__(self, rows: Sequence[Sequence[Any]]) -> None:
        """Store the rows."""
        self._rows = rows

    def execute(self, *args: Any, **kwargs: Any) -> _FakeResult:
        """Return the configured rows."""
        del args, kwargs
        return _FakeResult(self._rows)


@contextmanager
def _connection_that(exc: BaseException) -> Iterator[Any]:
    """Yield a connection whose query raises ``exc``."""
    yield _FailingConnection(exc)


@contextmanager
def _connection_with_rows(rows: Sequence[Sequence[Any]]) -> Iterator[_RowsConnection]:
    """Yield a connection returning fixed rows."""
    yield _RowsConnection(rows)


# --------------------------------------------------------------- config and text


def test_sparse_config_defaults_and_validation() -> None:
    """Defaults match the generated tsv; invalid values are rejected."""
    config = SparseConfig()
    assert config.top_n == 50
    assert config.language == "english"
    assert config.query_mode == "or"
    assert config.rank_weights == (0.1, 0.2, 0.4, 1.0)

    for bad_top_n in (0, -1, TOP_K_MAX + 1, True):
        with pytest.raises(ValidationError):
            SparseConfig(top_n=bad_top_n)
    for bad_language in ("simple", "chinese", ""):
        with pytest.raises(ValidationError):
            SparseConfig(language=bad_language)  # type: ignore[arg-type]
    for bad_mode in ("xor", "", "AND"):
        with pytest.raises(ValidationError):
            SparseConfig(query_mode=bad_mode)  # type: ignore[arg-type]
    for bad_weights in (
        (-0.1, 0.2, 0.4, 1.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.1, 0.2, 0.4),
        (math.inf, 0.2, 0.4, 1.0),
        (math.nan, 0.2, 0.4, 1.0),
    ):
        with pytest.raises(ValidationError):
            SparseConfig(rank_weights=bad_weights)  # type: ignore[arg-type]

    config.top_n = 7
    assert config.top_n == 7
    with pytest.raises(ValidationError):
        config.top_n = 0


def test_fts_query_text_or_mode_preserves_explicit_syntax() -> None:
    """Plain terms become OR; caller operators are not rewritten."""
    assert _fts_query_text("balance transfer fee", "or") == "balance or transfer or fee"
    assert _fts_query_text("What is the fee?", "or") == "What or is or the or fee?"
    assert _fts_query_text('"balance transfer" fee', "or") == '"balance transfer" fee'
    assert _fts_query_text("balance -fee", "or") == "balance -fee"
    assert _fts_query_text("balance or fee", "or") == "balance or fee"
    assert _fts_query_text("Global Entry or TSA PreCheck", "or") == (
        "Global or Entry or TSA or PreCheck"
    )
    assert _fts_query_text("balance transfer fee", "and") == "balance transfer fee"
    with pytest.raises(ValueError):
        _fts_query_text("balance", "xor")  # type: ignore[arg-type]


def test_cjk_bigrams_are_overlapping_and_deduplicated() -> None:
    """CJK runs produce overlapping bigrams; one-char runs stay one term."""
    assert _cjk_bigrams("余额转账") == ["余额", "额转", "转账"]
    assert _cjk_bigrams("信用") == ["信用"]
    assert _cjk_bigrams("信用 信用") == ["信用"]
    assert _cjk_bigrams("a余额，转账b") == ["余额", "转账"]
    assert _cjk_bigrams("balance transfer") == []


# ------------------------------------------------------ A. real retrieval results


def test_real_query_returns_real_chunk_hits(
    populated_corpus: None,
    retriever: SparseRetriever,
    eval_queries: list[dict[str, Any]],
) -> None:
    """The whole path: natural query -> FTS -> chunks JOIN documents."""
    query = eval_queries[0]["query"]
    hits = retriever.retrieve(query)

    assert len(hits) == 10
    assert [hit.rank for hit in hits] == list(range(1, 11))
    assert {hit.source for hit in hits} == {"sparse"}
    assert {"DOC-0001", "DOC-0005"} <= {hit.doc_id for hit in hits}

    for hit in hits:
        assert isinstance(hit, SearchHit)
        assert hit.chunk_id and hit.doc_id
        assert hit.content
        assert hit.title and hit.category and hit.card_tier
        assert hit.chunk_strategy == "document"
        assert hit.chunk_index == 0
        assert hit.distance is None
        assert math.isfinite(hit.score)
        assert hit.score > 0

    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True), "hits are not ordered by descending score"


def test_semantic_recall_uses_the_human_labels(
    populated_corpus: None,
    retriever: SparseRetriever,
    eval_queries: list[dict[str, Any]],
) -> None:
    """Q-001/Q-002/Q-003 gold docs are present in the sparse Top-10."""
    for entry in eval_queries[:3]:
        hits = retriever.retrieve(entry["query"])
        doc_ids = list(dict.fromkeys(hit.doc_id for hit in hits))
        recall = recall_at_k(doc_ids, entry["relevant_doc_ids"], 10)
        assert recall == 1.0, f"{entry['query_id']} recall={recall}, docs={doc_ids}"


def test_metadata_matches_the_persisted_chunk_and_document(
    populated_corpus: None,
    retriever: SparseRetriever,
    eval_queries: list[dict[str, Any]],
    conn: psycopg.Connection,
) -> None:
    """Every returned field is read from the real tables, not synthesised."""
    hits = retriever.retrieve(eval_queries[0]["query"], top_k=5)
    assert len(hits) == 5

    for hit in hits:
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


def test_top_k_is_exact_and_chunk_identities_are_unique(
    populated_corpus: None,
    retriever: SparseRetriever,
    eval_queries: list[dict[str, Any]],
) -> None:
    """The join must not inflate or duplicate the chunk-level Top-K."""
    for limit in (1, 5, 10):
        hits = retriever.retrieve(eval_queries[0]["query"], top_k=limit)
        identities = [(hit.chunk_strategy, hit.chunk_id) for hit in hits]
        assert len(hits) == limit
        assert len(set(identities)) == limit
        assert [hit.rank for hit in hits] == list(range(1, limit + 1))


def test_configured_top_n_is_used_when_top_k_is_omitted(
    populated_corpus: None,
) -> None:
    """``SparseConfig.top_n`` is the default; ``top_k`` overrides it."""
    retriever = SparseRetriever(SparseConfig(top_n=3))
    assert len(retriever.retrieve("balance transfer fee")) == 3
    assert len(retriever.retrieve("balance transfer fee", top_k=7)) == 7


def test_no_match_returns_an_empty_list(
    populated_corpus: None,
    retriever: SparseRetriever,
) -> None:
    """A query with no matching lexeme is an empty result, not an error."""
    assert retriever.retrieve("zzzznonexistentterm") == []
    assert retriever.retrieve("the and of") == []


def test_and_mode_is_the_strict_control(
    populated_corpus: None,
    eval_queries: list[dict[str, Any]],
) -> None:
    """Bare AND terms return nothing for a natural question; OR does not."""
    query = eval_queries[1]["query"]
    strict = SparseRetriever(SparseConfig(top_n=10, query_mode="and"))
    assert strict.retrieve(query) == []
    assert SparseRetriever(SparseConfig(top_n=10, query_mode="or")).retrieve(query)


def test_chinese_query_uses_the_cjk_fallback_and_returns_no_english_rows(
    populated_corpus: None,
    retriever: SparseRetriever,
) -> None:
    """The English corpus has no Chinese text; the fallback is not a translation layer."""
    assert retriever.retrieve("余额转账手续费") == []
    assert retriever.retrieve("balance transfer fee 余额") == []


def test_sql_metacharacters_are_bound_not_executed(
    populated_corpus: None,
    retriever: SparseRetriever,
    conn: psycopg.Connection,
) -> None:
    """A hostile-looking query does not escape its parameter or alter the table."""
    before = conn.execute("SELECT count(*) FROM chunks").fetchone()
    assert before is not None
    hostile = retriever.retrieve("'; DROP TABLE chunks; --")
    placeholders = retriever.retrieve("%s $1 :name")
    assert isinstance(hostile, list)
    assert isinstance(placeholders, list)
    after = conn.execute("SELECT count(*) FROM chunks").fetchone()
    assert after is not None
    assert after[0] == before[0]


# ------------------------------------------------------------- B. invalid input


def test_invalid_queries_and_top_k_do_not_touch_the_database() -> None:
    """Validation raises before the connection factory is entered."""
    retriever = SparseRetriever(connection_factory=_must_not_connect)
    for bad_query in ("", "   ", "\n\t"):
        with pytest.raises(ValueError):
            retriever.retrieve(bad_query)
    with pytest.raises(ValueError):
        retriever.retrieve(123)  # type: ignore[arg-type]
    for bad_top_k in (0, -1, TOP_K_MAX + 1, True, 1.5):
        with pytest.raises(ValueError):
            retriever.retrieve("balance", top_k=bad_top_k)  # type: ignore[arg-type]


# ---------------------------------------------------------- C. database failures


def test_unreachable_database_becomes_retrieval_error() -> None:
    """A connection failure is wrapped with a stage-specific message."""
    retriever = SparseRetriever(connection_factory=_unreachable_connection)
    with pytest.raises(RetrievalError, match="database connection failed"):
        retriever.retrieve("balance")


def test_pool_timeout_becomes_retrieval_error() -> None:
    """A pool-borrow timeout is reported without leaking pool internals."""
    retriever = SparseRetriever(connection_factory=_exhausted_pool)
    with pytest.raises(RetrievalError, match="timed out waiting for a database connection"):
        retriever.retrieve("balance")


def test_query_cancel_becomes_retrieval_error() -> None:
    """A statement timeout/cancel is wrapped as a retrieval timeout."""
    error = psycopg.errors.QueryCanceled("canceling statement due to statement timeout")
    retriever = SparseRetriever(connection_factory=lambda: _connection_that(error))
    with pytest.raises(RetrievalError, match="cancelled or timed out"):
        retriever.retrieve("balance")


def test_psycopg_error_becomes_retrieval_error() -> None:
    """A generic database error keeps the backend message and cause."""
    error = psycopg.Error("relation chunks does not exist")
    retriever = SparseRetriever(connection_factory=lambda: _connection_that(error))
    with pytest.raises(RetrievalError, match="sparse retrieval database error"):
        retriever.retrieve("balance")


def test_non_finite_scores_are_rejected() -> None:
    """A corrupt score cannot become a SearchHit."""
    row = ("chunk", "DOC-0001", "document", 0, "content", math.inf, "t", "c", "tier", 2024, "P")
    with pytest.raises(RetrievalError, match="invalid sparse score"):
        _rows_to_hits([row])


def test_rows_are_mapped_to_sparse_hits_in_column_order() -> None:
    """The SELECT column order and the hit fields stay in sync."""
    row = ("chunk", "DOC-0001", "document", 0, "content", 0.5, "title", "cat", "tier", 2024, "P")
    hits = _rows_to_hits([row])
    assert len(hits) == 1
    assert hits[0].chunk_id == "chunk"
    assert hits[0].doc_id == "DOC-0001"
    assert hits[0].score == 0.5
    assert hits[0].rank == 1
    assert hits[0].distance is None
    assert hits[0].source == "sparse"
    assert hits[0].title == "title"
