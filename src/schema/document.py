"""Domain models for the knowledge base.

Pure data shapes -- no behaviour. `policy_code` is derived by the loader
(see `src.ingestion.loader`) and carried here so that downstream analysis can
group the near-duplicate documents that share one policy rule across card tiers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """One entry of ``data/kb_documents.json``, exactly as shipped."""

    doc_id: str = Field(..., description="Stable identifier, e.g. 'DOC-0001'.")
    title: str
    category: str = Field(..., description="One of the 10 policy categories.")
    card_tier: str = Field(
        ...,
        description="Apex Platinum / Preferred Travel / Everyday Freedom / Business Select.",
    )
    effective_year: int
    content: str

    policy_code: str | None = Field(
        default=None,
        description="Extracted 'Policy Code' token, e.g. 'BT-001'. Populated by the loader.",
    )


class Chunk(BaseModel):
    """A retrievable unit produced by a chunking strategy.

    ``content`` is what we show a human or feed to the reranker; ``embedded_text``
    is what we actually vectorise. The two differ when the chunking strategy
    prepends a context header (card tier, category, policy code), which is the
    single most useful lever against tier-confusion in this corpus.
    """

    chunk_id: str
    doc_id: str = Field(
        ...,
        description="Owning document. Gold labels are doc-level, so every hit must carry this.",
    )
    chunk_strategy: str
    chunk_index: int
    content: str
    embedded_text: str
    category: str
    card_tier: str
    effective_year: int
    policy_code: str | None = None
