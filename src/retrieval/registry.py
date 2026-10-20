"""Evaluation arm registry.

The production wiring lives in :mod:`src.rag.factory`; this module enumerates the
*experiment* arms instead. Each arm is a named, declarative recipe that turns one
:class:`ArmContext` (embedder, optional reranker, Top-K, candidate pool, RRF
constant, chunk strategy) into a :class:`~src.retrieval.base.Retriever`.

Arms:

=============  ==========================================================
``dense``      pgvector cosine ANN only.
``sparse_or``  PostgreSQL FTS with recall-oriented OR semantics.
``sparse_and`` PostgreSQL FTS with strict AND semantics (control arm).
``union``      Deterministic interleaved dense/sparse candidate union
               baseline -- coverage, not scoring.
``rrf``        Dense + sparse fused with standard RRF.
``rrf_rerank`` The ``rrf`` pool reranked by a cross-encoder.
=============  ==========================================================

The registry performs no retrieval itself and holds no global state: every call
builds a fresh retriever from the context it is given. An unknown arm, a missing
reranker or an impossible Top-K/candidate-pool combination raises ``ValueError``
before any database or model work happens.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.retrieval.base import Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import (
    COUNT_MAX,
    RRF_K_DEFAULT,
    RRFFusion,
    interleaved_union,
)
from src.retrieval.pipeline import HybridRetriever, RerankedRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.sparse import SparseRetriever
from src.schema.retrieval import (
    DenseConfig,
    QueryMode,
    RerankConfig,
    RetrievalStrategy,
    SearchHit,
    SparseConfig,
)

DEFAULT_TOP_K = 10
"""Final list length for every arm unless the context overrides it."""

DEFAULT_CANDIDATE_K = 10
"""Per-arm candidate pool. Kept small by default because the historical corpus
has only 150 chunks and the rerank pool dominates CPU latency."""

DEFAULT_EF_SEARCH = 40
"""HNSW query-time candidate list used by the dense arm."""

ARM_NAMES: tuple[str, ...] = (
    "dense",
    "sparse_or",
    "sparse_and",
    "union",
    "rrf",
    "rrf_rerank",
)


@dataclass(frozen=True)
class ArmContext:
    """Everything an arm needs that is not a retrieval parameter.

    ``embedder`` is injected rather than created here, so tests can use a fixed
    vector and the benchmark can embed each query once. ``reranker`` is only
    required by ``rrf_rerank``.
    """

    embedder: Any = None
    reranker: Reranker | None = None
    top_k: int = DEFAULT_TOP_K
    candidate_k: int = DEFAULT_CANDIDATE_K
    rrf_k: int = RRF_K_DEFAULT
    ef_search: int = DEFAULT_EF_SEARCH
    chunk_strategy: RetrievalStrategy = "document"

    def __post_init__(self) -> None:
        """Validate the counts and the one cross-field constraint."""
        _validate_count(self.top_k, "top_k", maximum=COUNT_MAX)
        _validate_count(self.candidate_k, "candidate_k", maximum=COUNT_MAX)
        _validate_count(self.rrf_k, "rrf_k", maximum=1_000_000)
        _validate_count(self.ef_search, "ef_search", maximum=1000)
        if self.top_k > self.candidate_k:
            raise ValueError(
                f"top_k ({self.top_k}) must not exceed candidate_k ({self.candidate_k})"
            )
        if self.chunk_strategy not in ("document", "fixed_window", "structured"):
            raise ValueError(
                f"chunk_strategy must be document, fixed_window or structured, "
                f"got {self.chunk_strategy!r}"
            )


@dataclass(frozen=True)
class ArmSpec:
    """One registered arm: its label, its requirements and its builder."""

    name: str
    description: str
    needs_reranker: bool
    build: Callable[[ArmContext], Retriever]
    parameters: Callable[[ArmContext], dict[str, Any]]


def _dense_config(context: ArmContext, top_n: int) -> DenseConfig:
    """Build a dense config scoped to the arm's chunk strategy."""
    return DenseConfig(
        chunk_strategy=context.chunk_strategy,
        top_n=top_n,
        ef_search=context.ef_search,
    )


def _sparse_config(context: ArmContext, top_n: int, mode: QueryMode) -> SparseConfig:
    """Build a sparse config scoped to the arm's chunk strategy and mode."""
    return SparseConfig(
        chunk_strategy=context.chunk_strategy,
        top_n=top_n,
        query_mode=mode,
    )


def _dense(context: ArmContext) -> Retriever:
    """Dense ANN arm."""
    return DenseRetriever(
        _dense_config(context, context.top_k),
        embedder=context.embedder,
    )


def _sparse_or(context: ArmContext) -> Retriever:
    """Sparse OR arm."""
    return SparseRetriever(_sparse_config(context, context.top_k, "or"))


def _sparse_and(context: ArmContext) -> Retriever:
    """Sparse AND control arm."""
    return SparseRetriever(_sparse_config(context, context.top_k, "and"))


def _rrf_base(context: ArmContext) -> HybridRetriever:
    """Dense + sparse OR fused with RRF over ``candidate_k`` per arm."""
    return HybridRetriever(
        DenseRetriever(
            _dense_config(context, context.candidate_k), embedder=context.embedder
        ),
        SparseRetriever(_sparse_config(context, context.candidate_k, "or")),
        fusion=RRFFusion(k=context.rrf_k, candidate_k=context.candidate_k),
        default_top_k=context.top_k,
    )


def _union(context: ArmContext) -> Retriever:
    """Deterministic candidate-union baseline."""
    return _UnionRetriever(
        DenseRetriever(
            _dense_config(context, context.candidate_k), embedder=context.embedder
        ),
        SparseRetriever(_sparse_config(context, context.candidate_k, "or")),
        candidate_k=context.candidate_k,
        top_k=context.top_k,
    )


def _rrf(context: ArmContext) -> Retriever:
    """RRF arm."""
    return _rrf_base(context)


def _rrf_rerank(context: ArmContext) -> Retriever:
    """RRF pool reranked by a cross-encoder."""
    if context.reranker is None:
        raise ValueError("arm 'rrf_rerank' requires a reranker instance")
    return RerankedRetriever(
        _rrf_base(context),
        context.reranker,
        RerankConfig(
            enabled=True,
            candidates=context.candidate_k,
            top_k=context.top_k,
        ),
        default_top_k=context.top_k,
    )


class _UnionRetriever:
    """Adapter exposing :func:`interleaved_union` as a retriever stage."""

    name = "union"

    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        *,
        candidate_k: int,
        top_k: int,
    ) -> None:
        """Bind both arms and the union sizes."""
        self._dense = dense
        self._sparse = sparse
        self._candidate_k = candidate_k
        self._top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return the interleaved union of both arms' candidate pools."""
        resolved_top_k = self._top_k if top_k is None else top_k
        dense_hits = self._dense.retrieve(query, top_k=self._candidate_k)
        sparse_hits = self._sparse.retrieve(query, top_k=self._candidate_k)
        return interleaved_union(
            dense_hits,
            sparse_hits,
            candidate_k=self._candidate_k,
            top_k=resolved_top_k,
        )


ARMS: Mapping[str, ArmSpec] = {
    "dense": ArmSpec(
        name="dense",
        description="pgvector cosine ANN only",
        needs_reranker=False,
        build=_dense,
        parameters=lambda context: {
            "top_k": context.top_k,
            "ef_search": context.ef_search,
            "chunk_strategy": context.chunk_strategy,
        },
    ),
    "sparse_or": ArmSpec(
        name="sparse_or",
        description="PostgreSQL FTS, OR semantics",
        needs_reranker=False,
        build=_sparse_or,
        parameters=lambda context: {
            "top_k": context.top_k,
            "query_mode": "or",
            "chunk_strategy": context.chunk_strategy,
        },
    ),
    "sparse_and": ArmSpec(
        name="sparse_and",
        description="PostgreSQL FTS, AND semantics (control)",
        needs_reranker=False,
        build=_sparse_and,
        parameters=lambda context: {
            "top_k": context.top_k,
            "query_mode": "and",
            "chunk_strategy": context.chunk_strategy,
        },
    ),
    "union": ArmSpec(
        name="union",
        description="Interleaved dense/sparse candidate union (baseline)",
        needs_reranker=False,
        build=_union,
        parameters=lambda context: {
            "top_k": context.top_k,
            "candidate_k": context.candidate_k,
            "chunk_strategy": context.chunk_strategy,
        },
    ),
    "rrf": ArmSpec(
        name="rrf",
        description="Dense + sparse OR fused with reciprocal rank fusion",
        needs_reranker=False,
        build=_rrf,
        parameters=lambda context: {
            "top_k": context.top_k,
            "candidate_k": context.candidate_k,
            "rrf_k": context.rrf_k,
            "chunk_strategy": context.chunk_strategy,
        },
    ),
    "rrf_rerank": ArmSpec(
        name="rrf_rerank",
        description="RRF candidate pool reranked by a cross-encoder",
        needs_reranker=True,
        build=_rrf_rerank,
        parameters=lambda context: {
            "top_k": context.top_k,
            "candidate_k": context.candidate_k,
            "rrf_k": context.rrf_k,
            "chunk_strategy": context.chunk_strategy,
            "rerank": True,
        },
    ),
}


def arm_spec(name: str) -> ArmSpec:
    """Return the registered spec for ``name``.

    Raises:
        ValueError: ``name`` is not a registered arm.
    """
    try:
        return ARMS[name]
    except KeyError:
        raise ValueError(
            f"unknown arm {name!r}; registered arms: {', '.join(ARM_NAMES)}"
        ) from None


def build_arm(name: str, context: ArmContext) -> Retriever:
    """Build one arm for the given context.

    Raises:
        ValueError: unknown arm, missing reranker, or invalid context.
    """
    return arm_spec(name).build(context)


def arm_parameters(name: str, context: ArmContext) -> dict[str, Any]:
    """Return the parameters the report should record for this arm."""
    return dict(arm_spec(name).parameters(context))


def _validate_count(value: object, name: str, *, maximum: int) -> int:
    """Return ``value`` as a positive int, rejecting bools and out-of-range values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}, got {value}")
    return value
