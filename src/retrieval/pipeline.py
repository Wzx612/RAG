"""Retrieval pipelines.

Composes the existing dense and sparse stages with rank fusion:

    HybridRetriever = DenseRetriever + SparseRetriever + RRFFusion

The pipeline performs no retrieval logic of its own and no reranking. It asks
both arms for ``candidate_k`` hits, passes those lists to :class:`RRFFusion`,
and returns the fused Top-K. If either arm fails, the pipeline raises
:class:`~src.retrieval.RetrievalError` naming the failed stage; it never
silently degrades to the other arm, because a hybrid result that quietly
changed strategy would invalidate the evaluation's provenance.

Implemented in: Phase 6B (hybrid retrieval).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.retrieval import RetrievalError
from src.retrieval.base import Retriever
from src.retrieval.fusion import (
    COUNT_MAX,
    RRF_K_DEFAULT,
    TOP_K_DEFAULT,
    RRFFusion,
)
from src.retrieval.reranker import Reranker, RerankError
from src.schema.retrieval import RerankConfig, SearchHit


@dataclass(frozen=True)
class HybridResult:
    """Both arms' candidates plus the fused list and stage timings.

    ``dense_hits`` and ``sparse_hits`` are exactly the candidate lists passed
    to fusion after the per-arm ``candidate_k`` truncation. Timings are
    wall-clock milliseconds for sequential stages; they are diagnostic, not a
    latency benchmark.
    """

    query: str
    dense_hits: list[SearchHit]
    sparse_hits: list[SearchHit]
    fused_hits: list[SearchHit]
    top_k: int
    candidate_k: int
    rrf_k: int
    dense_ms: float
    sparse_ms: float
    fusion_ms: float
    total_ms: float


class HybridRetriever:
    """Dense + sparse retrieval fused with standard RRF.

    ``retrieve`` satisfies the :class:`~src.retrieval.base.Retriever` protocol,
    so the hybrid stage can be evaluated and composed like either single arm.
    ``retrieve_with_details`` additionally exposes both candidate lists and
    per-stage timings for the evaluation report.
    """

    name = "hybrid"

    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        fusion: RRFFusion | None = None,
        default_top_k: int = TOP_K_DEFAULT,
    ) -> None:
        """Bind both arms and the fusion stage.

        Args:
            dense: Dense retriever; must expose ``retrieve(query, top_k)``.
            sparse: Sparse retriever with the same interface.
            fusion: RRF stage. Defaults to ``RRFFusion(k=60, candidate_k=50)``.
            default_top_k: Final list length when ``retrieve`` omits ``top_k``.

        Raises:
            ValueError: if ``default_top_k`` is invalid or exceeds the fusion
                candidate pool.
        """
        self._dense = dense
        self._sparse = sparse
        self._fusion = fusion if fusion is not None else RRFFusion(k=RRF_K_DEFAULT)
        self._default_top_k = _validate_top_k(default_top_k, self._fusion.candidate_k)

    @property
    def fusion(self) -> RRFFusion:
        """The fusion stage this pipeline uses, including its recorded ``k``."""
        return self._fusion

    @property
    def default_top_k(self) -> int:
        """The final Top-K used when the caller omits ``top_k``."""
        return self._default_top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return fused hits, ranked best first.

        Args:
            query: Non-empty natural-language query text.
            top_k: Final list length; ``None`` uses ``default_top_k``.

        Returns:
            At most ``top_k`` fused hits with dense/sparse provenance.

        Raises:
            ValueError: invalid query or ``top_k`` before either arm runs.
            RetrievalError: either arm failed; the message names the stage.
        """
        return self.retrieve_with_details(query, top_k=top_k).fused_hits

    def retrieve_with_details(
        self,
        query: str,
        top_k: int | None = None,
    ) -> HybridResult:
        """Run both arms, fuse them, and return candidates plus timings.

        Args:
            query: Non-empty natural-language query text.
            top_k: Final list length; ``None`` uses ``default_top_k``.

        Returns:
            A :class:`HybridResult` with both candidate lists, fused hits,
            resolved parameters and stage timings.

        Raises:
            ValueError: invalid query or ``top_k`` before either arm runs.
            RetrievalError: either arm failed; the message names the stage.
        """
        resolved_query = _validate_query(query)
        resolved_top_k = _validate_top_k(
            self._default_top_k if top_k is None else top_k,
            self._fusion.candidate_k,
        )
        candidate_k = self._fusion.candidate_k

        started = time.perf_counter()
        dense_started = time.perf_counter()
        try:
            dense_hits = self._dense.retrieve(resolved_query, top_k=candidate_k)
        except RetrievalError as exc:
            raise RetrievalError(f"hybrid dense stage failed: {exc}") from exc
        dense_ms = (time.perf_counter() - dense_started) * 1000.0

        sparse_started = time.perf_counter()
        try:
            sparse_hits = self._sparse.retrieve(resolved_query, top_k=candidate_k)
        except RetrievalError as exc:
            raise RetrievalError(f"hybrid sparse stage failed: {exc}") from exc
        sparse_ms = (time.perf_counter() - sparse_started) * 1000.0

        dense_candidates = list(dense_hits[:candidate_k])
        sparse_candidates = list(sparse_hits[:candidate_k])

        fusion_started = time.perf_counter()
        fused_hits = self._fusion.fuse(
            dense_candidates,
            sparse_candidates,
            top_k=resolved_top_k,
        )
        fusion_ms = (time.perf_counter() - fusion_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0

        return HybridResult(
            query=resolved_query,
            dense_hits=dense_candidates,
            sparse_hits=sparse_candidates,
            fused_hits=fused_hits,
            top_k=resolved_top_k,
            candidate_k=candidate_k,
            rrf_k=self._fusion.k,
            dense_ms=dense_ms,
            sparse_ms=sparse_ms,
            fusion_ms=fusion_ms,
            total_ms=total_ms,
        )


@dataclass(frozen=True)
class RerankedResult:
    """The candidate pool, the reranked list, and the provenance of both.

    ``base_hits`` is exactly the list handed to the reranker after the
    ``candidate_k`` truncation, in its original order. ``hits`` is the reranked
    Top-K. ``fallback_used`` is always ``False``: this implementation has no
    silent fallback, so a failure raises instead of returning ``base_hits`` as
    though it had been reranked.
    """

    query: str
    base_hits: list[SearchHit]
    hits: list[SearchHit]
    candidate_k: int
    top_k: int
    base_ms: float
    rerank_ms: float
    total_ms: float
    base_name: str
    model_id: str | None
    model_revision: str | None
    device: str | None
    batch_size: int | None
    max_length: int | None
    activation: str | None
    model_loaded: bool | None
    empty_candidates: bool
    truncated: bool
    fallback_used: bool


class RerankedRetriever:
    """A base retriever followed by a cross-encoder reranker.

    The class performs no retrieval or scoring of its own. It requests
    ``config.candidates`` hits from the wrapped retriever, passes that exact
    pool to the reranker, and returns the reranked Top-K. It satisfies the same
    :class:`~src.retrieval.base.Retriever` protocol as the dense, sparse and
    hybrid stages.

    The model is owned by the injected reranker and is reused for every query.
    This class does not load or unload an embedding model; callers that compose
    it with a dense/hybrid retriever must account for both models' memory.
    """

    name = "reranked"

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker,
        config: RerankConfig | None = None,
        default_top_k: int | None = None,
    ) -> None:
        """Bind the base retriever, reranker and final Top-K.

        Args:
            base: Any retriever implementing the project protocol.
            reranker: Scoring stage; one instance is reused across queries.
            config: Candidate-pool and output settings. Defaults to the
                reranker's own config when it exposes one.
            default_top_k: Final list length when ``retrieve`` omits ``top_k``.
                Defaults to ``config.top_k``.

        Raises:
            ValueError: if ``default_top_k`` is invalid or exceeds the candidate
                pool.
        """
        self._base = base
        self._reranker = reranker
        own_config = getattr(reranker, "config", None)
        self._config = (
            config
            if config is not None
            else own_config
            if isinstance(own_config, RerankConfig)
            else RerankConfig()
        )
        self._default_top_k = _validate_top_k(
            self._config.top_k if default_top_k is None else default_top_k,
            self._config.candidates,
        )

    @property
    def config(self) -> RerankConfig:
        """The effective candidate-pool and Top-K settings."""
        return self._config

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return reranked hits, ranked best first."""
        return self.retrieve_with_details(query, top_k=top_k).hits

    def retrieve_with_details(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RerankedResult:
        """Run the base retriever, then rerank its candidate pool.

        Args:
            query: Non-empty natural-language query.
            top_k: Final list length; ``None`` uses ``default_top_k``.

        Returns:
            A :class:`RerankedResult` with the exact base pool, reranked hits,
            timings, model provenance and empty/truncation flags.

        Raises:
            ValueError: invalid query or ``top_k`` before retrieval.
            RerankError: the base retriever or the reranker failed. A base
                failure is categorised as ``base_retrieval``; no fallback is
                returned.
        """
        resolved_query = _validate_query(query)
        resolved_top_k = _validate_top_k(
            self._default_top_k if top_k is None else top_k,
            self._config.candidates,
        )

        started = time.perf_counter()
        base_started = time.perf_counter()
        try:
            base_hits = list(self._base.retrieve(resolved_query, top_k=self._config.candidates))
        except RerankError:
            raise
        except RetrievalError as exc:
            raise RerankError(
                f"reranked base retrieval failed: {exc}",
                stage="base_retrieval",
            ) from exc
        base_ms = (time.perf_counter() - base_started) * 1000.0
        candidates = base_hits[: self._config.candidates]

        base_name = str(getattr(self._base, "name", "unknown"))
        model_id = getattr(self._reranker, "model_id", None)
        model_revision = getattr(self._reranker, "revision", None)
        device = getattr(self._reranker, "device", None)
        batch_size = getattr(self._reranker, "batch_size", None)
        max_length = getattr(self._reranker, "max_length", None)
        activation = getattr(self._reranker, "activation", None)
        if not candidates:
            return RerankedResult(
                query=resolved_query,
                base_hits=[],
                hits=[],
                candidate_k=self._config.candidates,
                top_k=resolved_top_k,
                base_ms=base_ms,
                rerank_ms=0.0,
                total_ms=(time.perf_counter() - started) * 1000.0,
                base_name=base_name,
                model_id=model_id,
                model_revision=model_revision,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                activation=activation,
                model_loaded=bool(getattr(self._reranker, "is_loaded", None)),
                empty_candidates=True,
                truncated=False,
                fallback_used=False,
            )

        rerank_started = time.perf_counter()
        hits = self._reranker.rerank(resolved_query, candidates, top_k=resolved_top_k)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0

        return RerankedResult(
            query=resolved_query,
            base_hits=candidates,
            hits=hits,
            candidate_k=self._config.candidates,
            top_k=resolved_top_k,
            base_ms=base_ms,
            rerank_ms=rerank_ms,
            total_ms=total_ms,
            base_name=base_name,
            model_id=model_id,
            model_revision=model_revision,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            activation=activation,
            model_loaded=bool(getattr(self._reranker, "is_loaded", None)),
            empty_candidates=False,
            truncated=len(hits) < len(candidates),
            fallback_used=False,
        )


def _validate_query(query: object) -> str:
    """Return the stripped query, or raise before either arm runs."""
    if not isinstance(query, str):
        raise ValueError(f"query must be a str, got {type(query).__name__}")
    stripped = query.strip()
    if not stripped:
        raise ValueError("query must be a non-empty string")
    return stripped


def _validate_top_k(top_k: object, candidate_k: int) -> int:
    """Return a validated final Top-K bounded by the candidate pool."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError(f"top_k must be an int, got {type(top_k).__name__}")
    if not 1 <= top_k <= COUNT_MAX:
        raise ValueError(f"top_k must be between 1 and {COUNT_MAX}, got {top_k}")
    if top_k > candidate_k:
        raise ValueError(f"top_k ({top_k}) must not exceed candidate_k ({candidate_k})")
    return top_k
