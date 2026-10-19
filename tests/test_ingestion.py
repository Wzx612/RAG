"""Phase 3: the write path -- JSON -> Document -> Chunk -> vector -> PostgreSQL.

Most of this file runs without a model: the loader reads the real corpus, and
the embedding pipeline is exercised through a stub injected into
``SentenceTransformerEmbedder``. One test at the end does load the real
``BAAI/bge-large-en-v1.5``, because "the model produces 1024 finite numbers" is
the one claim a stub cannot make. It downloads the weights on first run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import pytest
from src.config import Settings
from src.ingestion import IngestionError
from src.ingestion.chunker import chunk_document, chunk_documents
from src.ingestion.embedder import EmbeddingProvider, SentenceTransformerEmbedder, validate_vectors
from src.ingestion.loader import (
    extract_policy_code,
    load_documents,
    source_path,
    validate_records,
)
from src.schema.retrieval import ChunkingConfig

EXPECTED_DOCUMENTS = 150
EXPECTED_POLICY_FAMILIES = 78
"""Phase 1 counted these; the corpus is templated and 150 documents share 78 rules."""


# ------------------------------------------------------------------- the loader


def test_the_corpus_file_is_where_settings_say_it_is() -> None:
    """A moved or renamed corpus fails here rather than deep inside the pipeline."""
    assert source_path().is_file()
    assert source_path().name == "kb_documents.json"


def test_the_corpus_parses_and_holds_exactly_150_documents() -> None:
    """The brief fixes the corpus size, so a size change must be deliberate."""
    assert len(load_documents()) == EXPECTED_DOCUMENTS


def test_every_document_was_validated(corpus_report: Any) -> None:
    """No record is dropped: valid + invalid accounts for the whole file."""
    assert corpus_report.total == EXPECTED_DOCUMENTS
    assert len(corpus_report.documents) == EXPECTED_DOCUMENTS
    assert corpus_report.problems == []
    assert corpus_report.ok


def test_doc_ids_are_unique(corpus: list[Any]) -> None:
    """doc_id is the upsert key, so a collision would silently merge two documents."""
    doc_ids = [document.doc_id for document in corpus]
    assert len(set(doc_ids)) == len(doc_ids) == EXPECTED_DOCUMENTS


def test_no_document_has_empty_content(corpus: list[Any]) -> None:
    """An empty content would embed to a meaningless vector and still pass NOT NULL."""
    assert all(document.content.strip() for document in corpus)


def test_policy_code_is_extracted_for_the_whole_corpus(corpus: list[Any]) -> None:
    """The family token is what makes the near-duplicate documents groupable."""
    codes = {document.policy_code for document in corpus}
    assert None not in codes
    assert len(codes) == EXPECTED_POLICY_FAMILIES


def test_effective_year_is_an_int_even_though_the_json_has_strings(corpus: list[Any]) -> None:
    """The column is SMALLINT; the JSON writes "2024". The loader bridges that."""
    assert all(isinstance(document.effective_year, int) for document in corpus)
    assert {document.effective_year for document in corpus} == {2024, 2025}


def test_extract_policy_code_returns_none_when_absent() -> None:
    """Absent is a legitimate outcome -- the column is nullable, not defaulted."""
    assert extract_policy_code("no code here") is None
    assert extract_policy_code("Policy Code: BT-007-APEX-2025. Details: ...") == "BT-007"


# ------------------------------------------------------------ rejected corpora


def _record(doc_id: str = "DOC-0001", **overrides: Any) -> dict[str, Any]:
    """A minimal valid record, so each test varies exactly one thing."""
    record = {
        "doc_id": doc_id,
        "title": "A title",
        "category": "Balance Transfers",
        "card_tier": "Apex Platinum",
        "effective_year": "2024",
        "content": "Policy Code: BT-001-APEX-2024. Details: something.",
    }
    record.update(overrides)
    return record


def test_a_duplicate_doc_id_is_rejected() -> None:
    """Two records with one key would collapse into a single row on upsert."""
    documents, problems = validate_records([_record("DOC-0001"), _record("DOC-0001")])

    assert len(documents) == 1
    assert len(problems) == 1
    assert "duplicate doc_id" in problems[0].issue


def test_an_empty_content_is_reported_not_dropped_silently() -> None:
    """The brief is explicit: report the document, never repair or delete it."""
    documents, problems = validate_records([_record(content="   ")])

    assert documents == []
    assert len(problems) == 1
    assert problems[0].issue == "content is empty"


def test_a_missing_field_names_the_field() -> None:
    """A pydantic failure must be traceable to a document and a column."""
    broken = _record()
    del broken["card_tier"]

    documents, problems = validate_records([broken])

    assert documents == []
    assert "card_tier" in problems[0].issue


def test_a_corpus_of_the_wrong_size_is_rejected(tmp_path: Path) -> None:
    """Not 150 means the wrong file, so no amount of per-record validation helps."""
    path = tmp_path / "kb_documents.json"
    path.write_text(json.dumps([_record("DOC-0001")]), encoding="utf-8")

    with pytest.raises(IngestionError, match="holds 1 records, expected 150"):
        load_documents(path)


def test_a_missing_corpus_is_reported_with_its_path(tmp_path: Path) -> None:
    """The error has to say which file, or the fix is guesswork."""
    with pytest.raises(IngestionError, match="cannot read knowledge base"):
        load_documents(tmp_path / "absent.json")


def test_a_json_object_instead_of_an_array_is_rejected(tmp_path: Path) -> None:
    """The schema is a list; accepting an object would hide a malformed export."""
    path = tmp_path / "kb_documents.json"
    path.write_text(json.dumps({"documents": []}), encoding="utf-8")

    with pytest.raises(IngestionError, match="must hold a JSON array"):
        load_documents(path)


# ------------------------------------------------------------------- the chunker


def test_the_document_strategy_produces_one_chunk_per_document(corpus: list[Any]) -> None:
    """The baseline every other strategy is measured against."""
    chunks = chunk_documents(corpus, ChunkingConfig(strategy="document"))

    assert len(chunks) == len(corpus) == EXPECTED_DOCUMENTS
    assert {chunk.doc_id for chunk in chunks} == {document.doc_id for document in corpus}
    assert all(chunk.chunk_index == 0 for chunk in chunks)
    assert all(chunk.chunk_strategy == "document" for chunk in chunks)


def test_embedded_text_is_the_content_when_no_header_is_requested(corpus: list[Any]) -> None:
    """Phase 1 chose plain content as the embedding input; anything else is an experiment."""
    chunk = chunk_document(corpus[0], ChunkingConfig(strategy="document"))

    assert chunk[0].embedded_text == corpus[0].content == chunk[0].content


def test_the_context_header_prepends_the_card_tier(corpus: list[Any]) -> None:
    """The header exists to fight tier confusion, so it must lead with the tier."""
    config = ChunkingConfig(strategy="document", context_header=True)
    chunk = chunk_document(corpus[0], config)[0]

    assert chunk.embedded_text.startswith(corpus[0].card_tier)
    assert chunk.embedded_text.endswith(corpus[0].content)
    assert chunk.content == corpus[0].content, "the header must not leak into displayed content"


@pytest.mark.parametrize("strategy", ["fixed_window", "contextual"])
def test_strategies_owned_by_phase_5_are_not_implemented_here(strategy: str) -> None:
    """Experiment 5 compares all three; doing it early would guess at its parameters."""
    with pytest.raises(IngestionError, match="implemented in Phase 5"):
        chunk_document(_document_stub(), ChunkingConfig(strategy=strategy))  # type: ignore[arg-type]


def _document_stub() -> Any:
    """A single document, for the tests that only exercise the strategy switch."""
    from src.schema.document import Document

    return Document(
        doc_id="DOC-0001",
        title="t",
        category="c",
        card_tier="Apex Platinum",
        effective_year=2024,
        content="Policy Code: BT-001-APEX-2024. Details.",
        policy_code="BT-001",
    )


# ------------------------------------------------------------------- the vectors


def test_validate_vectors_accepts_well_formed_vectors() -> None:
    """The happy path, stated once so the rejections below mean something."""
    validate_vectors([[0.0, 1.0], [0.5, 0.5]], 2)


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([0.0] * 3, "dimension 3, expected 2"),
        ([], "dimension 0, expected 2"),
    ],
)
def test_validate_vectors_rejects_the_wrong_width(vector: list[float], message: str) -> None:
    """Padding or truncating would produce a plausible index and a wrong Recall@K."""
    with pytest.raises(IngestionError, match=message):
        validate_vectors([vector], 2)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "0.5"])
def test_validate_vectors_rejects_non_finite_components(bad: Any) -> None:
    """PostgreSQL stores NaN without complaint and then orders it unpredictably."""
    with pytest.raises(IngestionError, match="not a finite number"):
        validate_vectors([[0.1, bad]], 2)


def test_validate_vectors_rejects_a_bare_scalar() -> None:
    """A 1-D batch must not read as a batch of scalars."""
    with pytest.raises(IngestionError, match="expected a vector of floats"):
        validate_vectors([0.1], 2)  # type: ignore[list-item]


# --------------------------------------------------------------- the embedder


class _StubModel:
    """Stands in for SentenceTransformer: deterministic, no weights, no download."""

    def __init__(self, dimension: int = 8, fill: float = 0.5) -> None:
        self.dimension = dimension
        self.fill = fill
        self.calls: list[list[str]] = []

    def encode(
        self,
        texts: Sequence[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Return a full batch of identical vectors, recording the batch shape."""
        self.calls.append(list(texts))
        return np.full((len(texts), self.dimension), self.fill, dtype=np.float32)


class _FlakyModel(_StubModel):
    """Fails a fixed number of times, then behaves -- a transient backend fault."""

    def __init__(self, failures: int, dimension: int = 8) -> None:
        super().__init__(dimension)
        self.remaining = failures

    def encode(self, texts: Sequence[str], **kwargs: Any) -> np.ndarray:
        """Raise until the failure budget is spent."""
        if self.remaining > 0:
            self.remaining -= 1
            raise RuntimeError("backend unavailable")
        return super().encode(texts, **kwargs)


def test_the_real_embedder_satisfies_the_provider_protocol() -> None:
    """The Protocol is the seam Phase 4 will import; it has to actually match."""
    assert isinstance(SentenceTransformerEmbedder(Settings()), EmbeddingProvider)


def test_documents_are_embedded_in_batches_of_the_configured_size() -> None:
    """Batching is what keeps peak memory bounded on a CPU-only host."""
    model = _StubModel(dimension=8)
    settings = Settings(embedding_dim=8, embedding_batch_size=3)
    progress: list[tuple[int, int]] = []

    vectors = SentenceTransformerEmbedder(settings, model=model).embed_documents(
        ["text"] * 7, on_batch=lambda done, total: progress.append((done, total))
    )

    assert [len(call) for call in model.calls] == [3, 3, 1]
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert len(vectors) == 7


def test_embedding_an_empty_batch_is_a_no_op() -> None:
    """Zero documents is a legitimate input and must not reach the model."""
    model = _StubModel(dimension=8)
    embedder = SentenceTransformerEmbedder(Settings(embedding_dim=8), model=model)

    assert embedder.embed_documents([]) == []
    assert model.calls == []


def test_a_transient_failure_is_retried_until_it_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two failures with max_retries=3 must still produce vectors."""
    monkeypatch.setattr("src.ingestion.embedder.time.sleep", lambda _seconds: None)
    settings = Settings(embedding_dim=8, embedding_batch_size=4, embedding_max_retries=3)
    embedder = SentenceTransformerEmbedder(settings, model=_FlakyModel(failures=2, dimension=8))

    assert len(embedder.embed_documents(["a", "b"])) == 2


def test_exhausted_retries_fail_rather_than_return_a_partial_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrying forever would hang the import; returning short would corrupt it."""
    monkeypatch.setattr("src.ingestion.embedder.time.sleep", lambda _seconds: None)
    settings = Settings(embedding_dim=8, embedding_batch_size=4, embedding_max_retries=3)
    embedder = SentenceTransformerEmbedder(settings, model=_FlakyModel(failures=99, dimension=8))

    with pytest.raises(IngestionError, match="failed after 3 attempts"):
        embedder.embed_documents(["a", "b"])


def test_a_model_of_the_wrong_width_is_rejected_before_anything_is_written() -> None:
    """The failure the whole dimension check exists for: model vs VECTOR(n) column."""
    settings = Settings(embedding_dim=1024, embedding_batch_size=4)
    embedder = SentenceTransformerEmbedder(settings, model=_StubModel(dimension=8))

    with pytest.raises(IngestionError, match="dimension 8, expected 1024"):
        embedder.embed_documents(["some text"])


def test_the_real_model_emits_1024_finite_dimensions() -> None:
    """The one claim a stub cannot make, so the real weights are loaded here.

    Downloads ``BAAI/bge-large-en-v1.5`` on a cold cache (~1.3 GB); every
    container run shares the ``hf_cache`` volume, so it happens once.
    """
    settings = Settings()
    embedder = SentenceTransformerEmbedder(settings)

    vectors = embedder.embed_documents(["A balance transfer costs 3% or $10."])
    query = embedder.embed_query("What is the balance transfer fee?")

    assert len(vectors) == 1
    assert len(vectors[0]) == settings.embedding_dim == 1024
    assert len(query) == settings.embedding_dim
    assert all(np.isfinite(vectors[0]))
    assert any(component != 0.0 for component in vectors[0])


def test_query_and_document_embeddings_differ_by_the_instruction_prefix() -> None:
    """BGE wants the prefix on queries only; embedding both the same way is a silent loss."""
    settings = Settings()
    embedder = SentenceTransformerEmbedder(settings)
    text = "What is the balance transfer fee?"

    document_side = embedder.embed_documents([text])[0]
    query_side = embedder.embed_query(text)

    assert document_side != query_side


# ------------------------------------------------------- what the database holds


def _imported_chunks(conn: psycopg.Connection) -> int:
    """Skip rather than fail when `make ingest` has not been run yet."""
    count = conn.execute("SELECT count(*) FROM chunks").fetchone()
    total = count[0] if count else 0
    if total == 0:
        pytest.skip("corpus not imported yet -- run `make ingest`")
    return total


def test_every_stored_embedding_has_the_configured_dimension(conn: psycopg.Connection) -> None:
    """Measured from the stored vectors, not from the column definition."""
    _imported_chunks(conn)
    expected = Settings().embedding_dim

    row = conn.execute(
        "SELECT min(vector_dims(embedding)), max(vector_dims(embedding)) FROM chunks"
    ).fetchone()

    assert row == (expected, expected)


def test_no_stored_embedding_is_a_zero_vector(conn: psycopg.Connection) -> None:
    """A zero vector passes NOT NULL and every dimension check while meaning nothing."""
    _imported_chunks(conn)

    zeros = conn.execute("SELECT count(*) FROM chunks WHERE embedding <#> embedding = 0").fetchone()

    assert zeros == (0,)


def test_the_document_strategy_covers_the_whole_corpus_exactly_once(
    conn: psycopg.Connection,
) -> None:
    """The import contract: one chunk per document, none missing, none duplicated."""
    _imported_chunks(conn)

    row = conn.execute(
        """
        SELECT count(*), count(DISTINCT doc_id), count(DISTINCT chunk_id)
        FROM chunks WHERE chunk_strategy = 'document'
        """
    ).fetchone()
    documents = conn.execute("SELECT count(*) FROM documents").fetchone()

    assert row == (EXPECTED_DOCUMENTS, EXPECTED_DOCUMENTS, EXPECTED_DOCUMENTS)
    assert documents == (EXPECTED_DOCUMENTS,)
