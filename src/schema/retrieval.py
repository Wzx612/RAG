"""Retrieval, fusion, rerank and chunking configuration models.

Pure data shapes -- no behaviour. These objects are what make the five required
experiments declarative: an experiment *is* a ``RetrievalConfig``, not a branch
inside a function.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

ChunkStrategy = Literal["document", "fixed_window", "contextual"]
FusionMethod = Literal["rrf", "weighted"]
QueryMode = Literal["or", "and"]


class SearchHit(BaseModel):
    """A single ranked result. Every retriever returns these.

    Dense hits carry both spellings of the cosine metric: ``distance`` is the
    raw pgvector value (lower is better) and ``score`` is similarity
    ``1 - distance`` (higher is better). They are kept separate because later
    stages redefine ``score``, while ``distance`` must stay comparable.
    """

    chunk_id: str
    doc_id: str
    score: float = Field(
        ...,
        description="Stage score. For dense: cosine similarity (1 - cosine distance).",
    )
    distance: float | None = Field(
        default=None,
        description="Raw pgvector distance. For dense: cosine distance; lower is better.",
    )
    rank: int = Field(..., ge=1, description="1-based rank within the producing stage.")
    source: Literal["dense", "sparse", "fused", "reranked"]
    dense_rank: int | None = Field(
        default=None,
        ge=1,
        description="1-based rank in the dense arm; set on fused hits only.",
    )
    sparse_rank: int | None = Field(
        default=None,
        ge=1,
        description="1-based rank in the sparse arm; set on fused hits only.",
    )
    dense_score: float | None = Field(
        default=None,
        description="Dense arm score before fusion; set on fused hits only.",
    )
    sparse_score: float | None = Field(
        default=None,
        description="Sparse arm score before fusion; set on fused hits only.",
    )
    rrf_score: float | None = Field(
        default=None,
        description="RRF contribution sum; set on RRF-fused hits only.",
    )
    candidate_sources: tuple[Literal["dense", "sparse"], ...] = Field(
        default=(),
        description="Which retrieval arms contributed this candidate, in dense/sparse order.",
    )
    rerank_score: float | None = Field(
        default=None,
        description=(
            "Cross-encoder relevance score. Higher is better. Set on reranked hits "
            "only; it is not comparable with dense cosine similarity or ts_rank_cd."
        ),
    )
    pre_rerank_rank: int | None = Field(
        default=None,
        ge=1,
        description="1-based rank of this candidate before reranking; set on reranked hits only.",
    )
    pre_rerank_score: float | None = Field(
        default=None,
        description=(
            "The candidate's stage score before reranking (RRF score for fused hits); "
            "set on reranked hits only."
        ),
    )
    content: str | None = None
    chunk_strategy: str | None = Field(
        default=None,
        description="Chunking strategy. With chunk_id it identifies the chunks row.",
    )
    chunk_index: int | None = None
    title: str | None = None
    category: str | None = None
    card_tier: str | None = None
    effective_year: int | None = None
    policy_code: str | None = None


class ChunkingConfig(BaseModel):
    """Knobs for the chunking comparison experiment (Experiment 5)."""

    strategy: ChunkStrategy = "document"
    chunk_size: int = Field(default=256, description="Tokens, fixed_window only.")
    chunk_overlap: int = Field(default=32, description="Tokens, fixed_window only.")
    context_header: bool = Field(
        default=False,
        description="Prefix embedded_text with card tier / category / policy code.",
    )


class DenseConfig(BaseModel):
    """pgvector ANN search parameters (Experiment 4 tunes m / ef_construction / ef_search).

    ``top_n`` and ``ef_search`` are per-query knobs. ``m`` and
    ``ef_construction`` size the index when it is built: changing them here
    does not alter an existing index and requires a rebuild.

    ``ef_search`` is also the upper bound on how many rows pgvector's HNSW scan
    can return. When ``ef_search < top_n`` the retriever returns at most
    ``ef_search`` hits; that is legal (Experiment 4 sweeps ef_search) but the
    retriever warns instead of pretending Top-K was satisfied.
    """

    model_config = ConfigDict(validate_assignment=True)

    top_n: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Requested Top-K. The retrieve(top_k=...) argument overrides it.",
    )
    m: int = Field(
        default=16,
        ge=2,
        le=100,
        description="HNSW connections per layer; build-time only (pgvector range 2..100).",
    )
    ef_construction: int = Field(
        default=64,
        ge=4,
        le=1000,
        description="HNSW build candidate list; build-time only (pgvector range 4..1000).",
    )
    ef_search: int = Field(
        default=40,
        ge=1,
        le=1000,
        description="HNSW query candidate list; HNSW returns at most this many rows.",
    )
    metric: Literal["cosine"] = "cosine"

    @field_validator("top_n", "m", "ef_construction", "ef_search", mode="before")
    @classmethod
    def _reject_bool(cls, value: object, info: ValidationInfo) -> object:
        """Reject ``True``/``False``: they are ints in Python but not counts."""
        if isinstance(value, bool):
            raise ValueError(f"{info.field_name} must be an int, not a bool")
        return value


class SparseConfig(BaseModel):
    """PostgreSQL full-text search parameters.

    ``language`` is constrained to ``english`` because ``chunks.tsv`` is a
    generated column built with that exact configuration. Parsing the query
    with a different configuration would compare unlike token streams and
    silently return the wrong rows, so the mismatch is rejected here.
    """

    model_config = ConfigDict(validate_assignment=True)

    top_n: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Requested Top-K. retrieve(top_k=...) overrides it.",
    )
    language: Literal["english"] = Field(
        default="english",
        description="Text search configuration; must match the generated chunks.tsv.",
    )
    query_mode: QueryMode = Field(
        default="or",
        description=(
            "Plain-text query semantics. 'or' is recall-oriented and is the default; "
            "'and' requires every non-stopword term. Quoted phrases and '-term' are "
            "left untouched; a bare 'or' is treated as prose and removed."
        ),
    )
    rank_weights: tuple[float, float, float, float] = Field(
        default=(0.1, 0.2, 0.4, 1.0),
        description="ts_rank_cd weights for {D, C, B, A}; title and card_tier are indexed as A.",
    )

    @field_validator("top_n", mode="before")
    @classmethod
    def _reject_bool_top_n(cls, value: object) -> object:
        """Reject ``True``/``False``: they are ints in Python but not counts."""
        if isinstance(value, bool):
            raise ValueError("top_n must be an int, not a bool")
        return value

    @field_validator("rank_weights")
    @classmethod
    def _validate_rank_weights(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Require four finite, non-negative weights with at least one positive."""
        weights = tuple(float(weight) for weight in value)
        if len(weights) != 4:
            raise ValueError("rank_weights must contain exactly 4 values (D, C, B, A)")
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("rank_weights must be finite and non-negative")
        if not any(weight > 0 for weight in weights):
            raise ValueError("rank_weights must contain at least one positive weight")
        return (weights[0], weights[1], weights[2], weights[3])


class FusionConfig(BaseModel):
    """How to merge the dense and sparse ranked lists."""

    method: FusionMethod = "rrf"
    rrf_k: int = Field(default=60, description="Cormack et al. 2009 default.")
    alpha: float = Field(
        default=0.5,
        description="Weighted fusion only: dense weight; sparse gets 1 - alpha.",
    )
    candidate_k: int = Field(
        default=50,
        description="How many merged candidates to keep for the next stage.",
    )


class RerankConfig(BaseModel):
    """Cross-encoder second stage.

    ``activation`` selects the score transform: ``identity`` keeps the model's
    raw logits (the BGE reranker card treats them as unbounded relevance scores;
    only their relative order is meaningful), while ``sigmoid`` maps them to
    ``(0, 1)`` without changing that order. The benchmark records which one ran.
    """

    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = False
    model: str = "BAAI/bge-reranker-base"
    revision: str | None = Field(
        default=None,
        description="Optional pinned model revision. Recorded in evaluation output.",
    )
    candidates: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Pairs scored per query; dominates latency.",
    )
    top_k: int = Field(default=10, ge=1, le=1000)
    batch_size: int = Field(
        default=16,
        ge=1,
        le=1000,
        description="Pairs per forward pass. Bounds peak memory on a CPU-only host.",
    )
    max_length: int = Field(
        default=512,
        ge=8,
        le=8192,
        description="Tokenizer truncation length for each query/passage pair.",
    )
    device: str = Field(default="cpu", min_length=1)
    activation: Literal["identity", "sigmoid"] = "identity"
    local_files_only: bool = Field(
        default=False,
        description="Refuse network access; load the model from the local HF cache only.",
    )
    cache_folder: str | None = Field(
        default=None,
        description="Optional Hugging Face cache root override.",
    )

    @model_validator(mode="after")
    def _top_k_within_candidates(self) -> RerankConfig:
        """Reject a final Top-K larger than the pool that will be reranked."""
        if self.top_k > self.candidates:
            raise ValueError(
                f"top_k ({self.top_k}) must not exceed candidates ({self.candidates})"
            )
        return self


class RetrievalConfig(BaseModel):
    """A complete, named retrieval strategy -- i.e. one experiment arm."""

    name: str
    dense: DenseConfig = Field(default_factory=DenseConfig)
    sparse: SparseConfig = Field(default_factory=SparseConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    use_dense: bool = True
    use_sparse: bool = True
