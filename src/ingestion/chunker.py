"""Chunking strategies.

Document -> list[Chunk]. Strategies: document (1 doc = 1 chunk, the baseline),
fixed_window, contextual (metadata-prefixed). No embedding, no retrieval.

Implemented in: Phase 3 (ingestion)
"""

from __future__ import annotations

from collections.abc import Sequence

from src.ingestion import IngestionError
from src.schema.document import Chunk, Document
from src.schema.retrieval import ChunkingConfig

BASELINE_STRATEGY = "document"

# Phase 5 owns these: Experiment 5 compares the three strategies side by side,
# which is the point at which their parameters stop being guesses.
DEFERRED_STRATEGIES = ("fixed_window", "contextual")


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

    Only the ``document`` strategy is implemented at this stage: it is the
    baseline every other strategy is measured against, and it is what the
    Phase 3 import writes. The remaining two arrive with Experiment 5.

    Raises:
        IngestionError: if the strategy is not implemented yet.
    """
    config = config or ChunkingConfig()

    if config.strategy != BASELINE_STRATEGY:
        raise IngestionError(
            f"chunking strategy {config.strategy!r} is implemented in Phase 5; "
            f"this stage supports {BASELINE_STRATEGY!r} only"
        )

    embedded_text = document.content
    if config.context_header:
        embedded_text = f"{context_header(document)}\n\n{embedded_text}"

    return [
        Chunk(
            chunk_id=document.doc_id,
            doc_id=document.doc_id,
            chunk_strategy=config.strategy,
            chunk_index=0,
            content=document.content,
            embedded_text=embedded_text,
            category=document.category,
            card_tier=document.card_tier,
            effective_year=document.effective_year,
            policy_code=document.policy_code,
        )
    ]


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
