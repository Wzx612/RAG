"""Request and response models for the read-only answer and search API.

The API is a transport layer over :class:`~src.rag.pipeline.RAGPipeline`; these
models decide what crosses the boundary. The answer models deliberately drop
``context_text`` and ``raw_response``: the caller gets the answer, its citations
and the evidence that was actually sent to the model, not the full prompt or the
raw provider payload. The search models expose ranked chunks only; they never
carry a generated answer or a provider key.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.schema.rag import (
    Citation,
    CitationStatus,
    Evidence,
    LLMUsage,
    RAGErrorInfo,
    RAGStatus,
)

MAX_QUERY_CHARS = 2000
"""A knowledge-base question is far shorter; this bounds request size."""


class AnswerRequest(BaseModel):
    """One question. Sizes such as Top-K are server configuration, not input."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description="Customer-service question in natural language.",
    )
    query_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional caller label copied into the response.",
    )


class SearchRequest(BaseModel):
    """One retrieval query. Generation settings are not part of this surface."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARS,
        description="Customer-service query in natural language.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description=(
            "Final result count. Omit to use the service default; the value must "
            "not exceed the configured candidate pool."
        ),
    )

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        """Reject whitespace-only queries before retrieval runs."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped


class SearchResult(BaseModel):
    """One ranked knowledge-base chunk, without prompts or provider payloads."""

    chunk_id: str
    doc_id: str
    rank: int = Field(ge=1, description="1-based rank in the returned list.")
    score: float = Field(description="Stage score; higher is better.")
    source: str = Field(description="Producing stage: dense, sparse, fused or reranked.")
    rerank_score: float | None = Field(
        default=None,
        description="Cross-encoder score when reranking ran; not comparable with cosine.",
    )
    title: str | None = None
    category: str | None = None
    card_tier: str | None = None
    effective_year: int | None = None
    policy_code: str | None = None
    content: str | None = Field(
        default=None,
        description="Chunk text returned by the retrieval stage.",
    )


class SearchResponse(BaseModel):
    """Ranked retrieval results for one query."""

    query: str
    top_k: int = Field(ge=1, description="Effective final Top-K.")
    hit_count: int = Field(ge=0)
    retriever: str
    results: list[SearchResult] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Process liveness; does not touch the database or a model."""

    status: Literal["ok"] = "ok"
    service: str = "apex-rag-api"


class ReadinessCheck(BaseModel):
    """One readiness probe and its observed detail."""

    name: str
    ok: bool
    detail: str


class ReadinessResponse(BaseModel):
    """Database readiness. Models stay lazy; their failure surfaces on /answer."""

    status: Literal["ready", "degraded"]
    checks: list[ReadinessCheck]


class AnswerResponse(BaseModel):
    """A structured answer, its citations and the evidence it was given."""

    model_config = ConfigDict(from_attributes=True)

    query_id: str | None = None
    status: RAGStatus
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    citation_status: CitationStatus = "no_citations"
    insufficient_evidence: bool = False
    evidence: list[Evidence] = Field(default_factory=list)
    prompt_version: str
    provider: str | None = None
    model: str | None = None
    usage: LLMUsage | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    provider_attempted: bool = False
    error: RAGErrorInfo | None = None
