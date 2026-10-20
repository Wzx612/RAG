"""Chunking strategy dispatch.

Document -> list[Chunk]. The baseline ``document`` strategy keeps the original
one-chunk-per-document behaviour. ``fixed_window`` and ``structured`` delegate
to the pure chunkers in :mod:`src.ingestion.fixed_window` and
:mod:`src.ingestion.structured`.

No embedding, no retrieval and no database access happen here. In particular,
character offsets stay in :class:`~src.ingestion.spans.ChunkSpan`; the current
``Chunk`` model and schema have no offset columns, so nothing is persisted that
the database cannot store.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.ingestion import IngestionError
from src.ingestion.fixed_window import FixedWindowParams, chunk_fixed_window
from src.ingestion.spans import ChunkSpan
from src.ingestion.structured import StructuredParams, chunk_structured
from src.schema.document import Chunk, Document
from src.schema.retrieval import ChunkingConfig

BASELINE_STRATEGY = "document"
SUPPORTED_STRATEGIES = (BASELINE_STRATEGY, "fixed_window", "structured")
DEPRECATED_CONTEXTUAL_STRATEGY = "contextual"


def context_header(document: Document) -> str:
    """Return the metadata preamble used by metadata-prefixed chunkings.

    ``card_tier`` leads because 20 of the 30 evaluation queries name a card, and
    the corpus's dominant failure mode is two documents that differ only by which
    card a rule applies to.
    """
    parts = [document.card_tier, document.category]
    if document.policy_code:
        parts.append(document.policy_code)
    return " | ".join(parts)


def chunk_document(document: Document, config: ChunkingConfig | None = None) -> list[Chunk]:
    """Split one document into retrievable chunks under ``config.strategy``.

    The signature and the ``document`` strategy's output are unchanged from the
    original baseline: one chunk whose ``chunk_id`` is the ``doc_id`` and whose
    ``chunk_index`` is 0. ``context_header`` only changes ``embedded_text`` for
    every strategy; it never changes the strategy or the source spans.

    Args:
        document: Source document.
        config: Strategy and parameters. Defaults to the ``document`` strategy.

    Returns:
        One or more chunks. Multi-chunk strategies use the stable id
        ``f"{doc_id}#{chunk_index}"`` and contiguous zero-based indices.

    Raises:
        IngestionError: The strategy is retired (``contextual``) or chunking
            produced no usable content.
        ValueError: The strategy parameters are invalid.
    """
    config = config or ChunkingConfig()

    if config.strategy == BASELINE_STRATEGY:
        return [_baseline_chunk(document, config)]

    spans = _strategy_spans(document, config)
    return _map_spans(document, spans, config)


def chunk_documents(
    documents: Sequence[Document], config: ChunkingConfig | None = None
) -> list[Chunk]:
    """Chunk a whole corpus under one strategy.

    The strategy is fixed for the batch on purpose: a single import must never
    mix strategies, or the ``chunk_strategy`` column stops identifying the
    experiment arm it exists to identify.
    """
    config = config or ChunkingConfig()
    return [chunk for document in documents for chunk in chunk_document(document, config)]


def chunk_spans(
    document: Document, config: ChunkingConfig | None = None
) -> list[ChunkSpan]:
    """Return the exact source spans a strategy selects for one document.

    This is the offset-preserving view of :func:`chunk_document`. The spans
    carry ``char_start``, ``char_end``, sentence indices and the structured
    section path; the current ``Chunk`` model does not, so tests and diagnostics
    can use this function without changing the model or the database schema.

    ``context_header`` is deliberately ignored: it affects ``embedded_text``
    only and must not alter which source text is selected.
    """
    config = config or ChunkingConfig()

    if not isinstance(document, Document):
        raise TypeError(f"document must be a Document, got {type(document).__name__}")

    if config.strategy == BASELINE_STRATEGY:
        if not document.content:
            raise IngestionError(f"document {document.doc_id!r} has empty content")
        content = document.content
        return [
            ChunkSpan(
                chunk_index=0,
                content=content,
                char_start=0,
                char_end=len(content),
            )
        ]

    return _strategy_spans(document, config)


def _baseline_chunk(document: Document, config: ChunkingConfig) -> Chunk:
    """Preserve the original one-chunk-per-document mapping exactly."""
    embedded_text = document.content
    if config.context_header:
        embedded_text = f"{context_header(document)}\n\n{embedded_text}"

    return Chunk(
        chunk_id=document.doc_id,
        doc_id=document.doc_id,
        chunk_strategy=BASELINE_STRATEGY,
        chunk_index=0,
        content=document.content,
        embedded_text=embedded_text,
        category=document.category,
        card_tier=document.card_tier,
        effective_year=document.effective_year,
        policy_code=document.policy_code,
    )


def _strategy_spans(document: Document, config: ChunkingConfig) -> list[ChunkSpan]:
    """Dispatch the non-baseline strategies to their pure chunkers."""
    if config.strategy == "fixed_window":
        return chunk_fixed_window(
            document.content,
            FixedWindowParams(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                min_tail=config.min_tail,
            ),
        )

    if config.strategy == "structured":
        return chunk_structured(
            document.content,
            StructuredParams(
                target_chars=config.structured_target_chars,
                max_chars=config.structured_max_chars,
                min_chars=config.structured_min_chars,
            ),
        )

    if config.strategy == DEPRECATED_CONTEXTUAL_STRATEGY:
        raise IngestionError(
            "chunking strategy 'contextual' is retired and does not select an algorithm; "
            "use strategy='document' with context_header=True, or choose 'fixed_window' "
            "or 'structured'. No strategy is selected implicitly."
        )

    raise IngestionError(f"unsupported chunking strategy {config.strategy!r}")


def _map_spans(
    document: Document, spans: Sequence[ChunkSpan], config: ChunkingConfig
) -> list[Chunk]:
    """Map exact spans to the project's ``Chunk`` model.

    ``content`` is the span itself, not the whole document. The context header,
    when requested, is prepended to ``embedded_text`` only. Chunk ids are
    derived from ``doc_id`` and the span index so that repeated runs produce
    identical ids.
    """
    chunks: list[Chunk] = []
    for expected_index, span in enumerate(spans):
        if span.chunk_index != expected_index:
            raise IngestionError(
                f"document {document.doc_id!r} produced non-contiguous chunk index "
                f"{span.chunk_index!r} at position {expected_index}"
            )

        embedded_text = span.content
        if config.context_header:
            embedded_text = f"{context_header(document)}\n\n{embedded_text}"

        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}#{span.chunk_index}",
                doc_id=document.doc_id,
                chunk_strategy=config.strategy,
                chunk_index=span.chunk_index,
                content=span.content,
                embedded_text=embedded_text,
                category=document.category,
                card_tier=document.card_tier,
                effective_year=document.effective_year,
                policy_code=document.policy_code,
            )
        )
    return chunks
