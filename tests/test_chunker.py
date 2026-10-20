"""Phase 2: strategy dispatch and configuration.

These tests connect the pure chunkers to the existing ingestion interface.
They do not touch PostgreSQL, embeddings or a model: they assert strategy
selection, API compatibility, configuration bounds and the span -> ``Chunk``
mapping, including the stable multi-chunk ids and chunk indices.
"""

from __future__ import annotations

from typing import Any

import pytest
from src.ingestion import IngestionError
from src.ingestion.chunker import (
    chunk_document,
    chunk_documents,
    chunk_spans,
    context_header,
)
from src.schema.document import Document
from src.schema.retrieval import ChunkingConfig

FIXED_TEXT = "0123456789" * 10
STRUCTURED_TEXT = (
    "Policy Code: BT-001-APEX-2024. Applicable Product: Apex Platinum. "
    "Policy Effective Year: 2024. Category: Balance Transfers. Details: "
    "First sentence about fees. Second sentence about eligibility. "
    "Third sentence about reporting."
)


def _document(content: str, doc_id: str = "DOC-0001") -> Document:
    """Build a valid document without reading the corpus."""
    return Document(
        doc_id=doc_id,
        title="A card policy",
        category="Balance Transfers",
        card_tier="Apex Platinum",
        effective_year=2024,
        content=content,
        policy_code="BT-001",
    )


def _fixed_config(**overrides: Any) -> ChunkingConfig:
    """A fixed-window config that produces several chunks from ``FIXED_TEXT``."""
    values: dict[str, Any] = {
        "strategy": "fixed_window",
        "chunk_size": 30,
        "chunk_overlap": 10,
        "min_tail": 5,
    }
    values.update(overrides)
    return ChunkingConfig(**values)


def _structured_config(**overrides: Any) -> ChunkingConfig:
    """A structured config with one sentence per chunk."""
    values: dict[str, Any] = {
        "strategy": "structured",
        "structured_target_chars": 10,
        "structured_max_chars": 80,
        "structured_min_chars": 0,
    }
    values.update(overrides)
    return ChunkingConfig(**values)


# --------------------------------------------------------------- configuration


def test_chunking_config_defaults_match_the_phase_one_values() -> None:
    """Defaults are character-based and identical to the pure chunkers'."""
    config = ChunkingConfig()

    assert config.strategy == "document"
    assert (config.chunk_size, config.chunk_overlap, config.min_tail) == (256, 64, 48)
    assert (
        config.structured_target_chars,
        config.structured_max_chars,
        config.structured_min_chars,
    ) == (220, 360, 60)
    assert config.context_header is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": True},
        {"chunk_overlap": -1},
        {"chunk_overlap": 256},
        {"min_tail": -1},
        {"min_tail": 256},
        {"structured_target_chars": 0},
        {"structured_target_chars": 400},
        {"structured_min_chars": -1},
        {"structured_min_chars": 300},
    ],
)
def test_chunking_config_rejects_invalid_parameters(kwargs: dict[str, object]) -> None:
    """Invalid lengths fail at config construction, before any chunk is built."""
    with pytest.raises(ValueError):
        ChunkingConfig(**kwargs)  # type: ignore[arg-type]


def test_unknown_strategy_is_rejected() -> None:
    """The strategy name is a closed set; ``semantic`` is not an available arm."""
    with pytest.raises(ValueError):
        ChunkingConfig(strategy="semantic")  # type: ignore[arg-type]


# ------------------------------------------------------------- API compatibility


def test_chunk_document_keeps_its_optional_config_signature() -> None:
    """Calling without a config still means the document baseline."""
    document = _document("Short policy.")

    assert chunk_document(document) == chunk_document(document, ChunkingConfig())


def test_chunk_documents_keeps_its_optional_config_signature() -> None:
    """The batch helper still accepts a bare sequence of documents."""
    documents = [_document("First policy."), _document("Second policy.", "DOC-0002")]

    assert chunk_documents(documents) == [
        chunk for document in documents for chunk in chunk_document(document)
    ]


def test_chunk_documents_of_an_empty_corpus_is_empty() -> None:
    """No documents is a valid input and does not fabricate a chunk."""
    assert chunk_documents([], ChunkingConfig(strategy="fixed_window")) == []


def test_the_document_strategy_still_maps_one_document_to_one_chunk() -> None:
    """The baseline id, index and content are exactly the original contract."""
    document = _document("The original content.")
    chunk = chunk_document(document, ChunkingConfig(strategy="document"))[0]
    span = chunk_spans(document, ChunkingConfig(strategy="document"))[0]

    assert chunk.chunk_id == document.doc_id
    assert chunk.chunk_index == 0
    assert chunk.chunk_strategy == "document"
    assert chunk.content == document.content
    assert chunk.embedded_text == document.content
    assert (span.char_start, span.char_end) == (0, len(document.content))


# ------------------------------------------------------------------- dispatch


def test_fixed_window_strategy_dispatches_to_the_window_chunker() -> None:
    """The fixed-window arm produces overlapping character windows."""
    document = _document(FIXED_TEXT)
    config = _fixed_config()

    chunks = chunk_document(document, config)
    spans = chunk_spans(document, config)

    assert len(chunks) == len(spans) == 5
    assert all(chunk.chunk_strategy == "fixed_window" for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3, 4]
    assert [span.chunk_index for span in spans] == [0, 1, 2, 3, 4]


def test_structured_strategy_dispatches_to_the_sentence_chunker() -> None:
    """The structured arm keeps whole body sentences and their section path."""
    document = _document(STRUCTURED_TEXT)
    config = _structured_config()

    chunks = chunk_document(document, config)
    spans = chunk_spans(document, config)

    assert len(chunks) == len(spans) == 3
    assert all(span.section_path == ("Details",) for span in spans)
    assert [chunk.content for chunk in chunks] == [
        "First sentence about fees.",
        "Second sentence about eligibility.",
        "Third sentence about reporting.",
    ]
    assert all(chunk.chunk_strategy == "structured" for chunk in chunks)


def test_contextual_is_rejected_instead_of_selecting_a_strategy_implicitly() -> None:
    """The retired name stays parseable but fails with an explicit migration path."""
    config = ChunkingConfig(strategy="contextual")

    with pytest.raises(IngestionError) as excinfo:
        chunk_document(_document("Short policy."), config)

    message = str(excinfo.value)
    assert "contextual" in message
    assert "context_header" in message


# ------------------------------------------------------- offsets and identities


def test_multi_chunk_content_maps_to_exact_source_offsets() -> None:
    """Every emitted chunk is exactly its span, not a reflow of the document."""
    document = _document(FIXED_TEXT)
    config = _fixed_config()

    spans = chunk_spans(document, config)
    chunks = chunk_document(document, config)

    for span, chunk in zip(spans, chunks, strict=True):
        assert document.content[span.char_start : span.char_end] == chunk.content
        assert chunk.content == span.content
        assert chunk.chunk_index == span.chunk_index


def test_structured_metadata_and_offsets_map_to_chunks() -> None:
    """Sentence indices and section paths stay in the span view; content maps exactly."""
    document = _document(STRUCTURED_TEXT)
    config = _structured_config()

    spans = chunk_spans(document, config)
    chunks = chunk_document(document, config)

    for span in spans:
        assert document.content[span.char_start : span.char_end] == span.content
        assert span.sentence_start is not None
        assert span.sentence_end is not None
        assert span.section_path == ("Details",)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_multi_chunk_ids_are_unique_and_stable_across_runs() -> None:
    """Ids derive from doc_id and chunk_index, so reruns upsert the same rows."""
    document = _document(FIXED_TEXT)
    config = _fixed_config()

    first = chunk_document(document, config)
    second = chunk_document(document, config)

    assert first == second
    assert [chunk.chunk_id for chunk in first] == [
        f"{document.doc_id}#{index}" for index in range(len(first))
    ]
    assert len({chunk.chunk_id for chunk in first}) == len(first)


def test_batch_chunking_does_not_mix_strategies() -> None:
    """One import carries one strategy, so the column identifies the arm."""
    documents = [_document(FIXED_TEXT), _document(FIXED_TEXT, "DOC-0002")]

    chunks = chunk_documents(documents, _fixed_config())

    assert {chunk.chunk_strategy for chunk in chunks} == {"fixed_window"}
    assert {chunk.doc_id for chunk in chunks} == {"DOC-0001", "DOC-0002"}


# -------------------------------------------------------------- context header


@pytest.mark.parametrize(
    ("strategy", "text", "config_factory"),
    [
        ("document", "Short policy.", lambda: ChunkingConfig(strategy="document")),
        ("fixed_window", FIXED_TEXT, _fixed_config),
        ("structured", STRUCTURED_TEXT, _structured_config),
    ],
)
def test_context_header_changes_only_embedded_text(
    strategy: str,
    text: str,
    config_factory: Any,
) -> None:
    """The header never changes the strategy, spans, indices or displayed content."""
    document = _document(text)
    plain = config_factory()
    headed = plain.model_copy(update={"context_header": True})

    without_header = chunk_document(document, plain)
    with_header = chunk_document(document, headed)

    assert [chunk.chunk_strategy for chunk in with_header] == [strategy] * len(with_header)
    assert [(chunk.chunk_id, chunk.chunk_index, chunk.content) for chunk in with_header] == [
        (chunk.chunk_id, chunk.chunk_index, chunk.content) for chunk in without_header
    ]
    assert chunk_spans(document, headed) == chunk_spans(document, plain)
    assert all(chunk.embedded_text == chunk.content for chunk in without_header)
    assert all(
        chunk.embedded_text == f"{context_header(document)}\n\n{chunk.content}"
        for chunk in with_header
    )
