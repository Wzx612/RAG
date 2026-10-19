"""Rank fusion.

Merges two ranked lists -- dense and sparse -- into one. No retrieval and no
reranking: every input hit is already ranked by its own stage.

:class:`RRFFusion` implements standard Reciprocal Rank Fusion (Cormack et al.,
2009)::

    RRF(d) = sum over arms r of 1 / (k + rank_r(d))

``rank_r(d)`` is the 1-based position of document ``d`` in arm ``r`` after that
arm's candidate list is truncated to ``candidate_k``. A candidate present in
only one arm contributes only that arm's term; a candidate in both arms
contributes both terms. Raw dense/sparse scores are never added to each other
-- they have different units and meanings.

Identity is chunk-level: ``(chunk_strategy, chunk_id)``. The same chunk found
by both arms is one candidate; two chunks of the same document are two
candidates. The current corpus has one chunk per document, so chunk-level and
document-level candidate sets coincide here, but fusion does not rely on that.

:func:`interleaved_union` is the candidate-union baseline, not a fusion method.
It gives the two arms no score arithmetic: it takes ``dense[0]``,
``sparse[0]``, ``dense[1]``, ``sparse[1]``, ..., skips identities already
emitted, then truncates. Its order is fully reproducible and is the only
meaning its rank has. The ``score`` field is a documented placeholder (``0.0``)
because the union has no fused score.

Failure and edge behaviour:

* duplicate identity **within one arm** raises ``ValueError`` (a retriever
  contract violation, not something to silently deduplicate);
* the same identity **across arms** merges, recording both ranks and scores;
* two empty lists return ``[]``; one empty list returns the other arm's
  contributions, down-weighted only by the RRF formula itself;
* ``top_k`` may exceed an arm's length; there is no padding;
* invalid ``k``/``candidate_k``/``top_k`` raise ``ValueError`` before any work.

Implemented in: Phase 6B (hybrid retrieval).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from src.schema.retrieval import SearchHit

RRF_K_DEFAULT = 60
"""Cormack et al. default, used unless the caller records another value."""

CANDIDATE_K_DEFAULT = 50
"""Default per-arm candidate pool before fusion."""

TOP_K_DEFAULT = 10
"""Default final list length."""

COUNT_MAX = 1000
"""Largest accepted ``candidate_k``/``top_k``, matching the retrievers' Top-K cap."""

RRF_K_MAX = 1_000_000
"""Sanity bound for ``k``; values this large make all contributions negligible."""

ChunkKey = tuple[str, str]
"""Fusion identity: ``(chunk_strategy, chunk_id)``."""


@dataclass
class _Candidate:
    """Accumulator for one chunk identity while both arms are merged."""

    key: ChunkKey
    first_ordinal: int
    base: SearchHit
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float = 0.0
    sources: list[Literal["dense", "sparse"]] = field(default_factory=list)


class RRFFusion:
    """Standard reciprocal rank fusion over dense and sparse ranked lists.

    ``fuse`` truncates each arm to ``candidate_k``, sums
    ``1 / (k + rank)`` per chunk identity, orders by the fused score and returns
    at most ``top_k`` hits. Ties are broken deterministically by the best rank
    seen in either arm, then by ``chunk_strategy`` and ``chunk_id``; the raw
    dense and sparse scores are never compared with one another.

    Attributes:
        k: RRF smoothing constant, fixed at construction and recorded in
            evaluation output.
        candidate_k: Per-arm candidate pool size used when ``fuse`` is called
            through :class:`~src.retrieval.pipeline.HybridRetriever`.
    """

    name = "rrf"

    def __init__(
        self,
        k: int = RRF_K_DEFAULT,
        candidate_k: int = CANDIDATE_K_DEFAULT,
    ) -> None:
        """Validate and store the RRF parameters.

        Args:
            k: Positive RRF constant; larger values flatten rank differences.
            candidate_k: Per-arm candidate pool, 1..1000.

        Raises:
            ValueError: if either value is not an int in its accepted range.
        """
        self._k = _validate_count(k, "k", maximum=RRF_K_MAX)
        self._candidate_k = _validate_count(candidate_k, "candidate_k", maximum=COUNT_MAX)

    @property
    def k(self) -> int:
        """The configured RRF constant."""
        return self._k

    @property
    def candidate_k(self) -> int:
        """The configured per-arm candidate pool size."""
        return self._candidate_k

    def fuse(
        self,
        dense_hits: Sequence[SearchHit],
        sparse_hits: Sequence[SearchHit],
        top_k: int | None = None,
    ) -> list[SearchHit]:
        """Fuse two ranked lists into one.

        Args:
            dense_hits: Dense hits ordered best first. Only the first
                ``candidate_k`` are considered.
            sparse_hits: Sparse hits ordered best first. Only the first
                ``candidate_k`` are considered.
            top_k: Maximum final hits; must not exceed ``candidate_k``.
                ``None`` uses ``min(10, candidate_k)``.

        Returns:
            Fused hits with ``source="fused"``, ``rank`` 1..N, ``score`` equal
            to ``rrf_score``, and dense/sparse provenance where applicable.

        Raises:
            ValueError: invalid ``top_k``, ``top_k > candidate_k``, duplicate
                identity within an arm, non-finite arm score, wrong arm source,
                or conflicting metadata for one identity across arms.
        """
        resolved_top_k = min(TOP_K_DEFAULT, self._candidate_k) if top_k is None else top_k
        _validate_top_k(resolved_top_k, self._candidate_k)
        dense = _candidate_slice(dense_hits, self._candidate_k, "dense")
        sparse = _candidate_slice(sparse_hits, self._candidate_k, "sparse")
        candidates = _accumulate(dense, sparse, self._k)
        return _finish(candidates, resolved_top_k)


def reciprocal_rank_fusion(
    dense_hits: Sequence[SearchHit],
    sparse_hits: Sequence[SearchHit],
    *,
    k: int = RRF_K_DEFAULT,
    candidate_k: int = CANDIDATE_K_DEFAULT,
    top_k: int = TOP_K_DEFAULT,
) -> list[SearchHit]:
    """Functional wrapper around :class:`RRFFusion` with explicit parameters."""
    fusion = RRFFusion(k=k, candidate_k=candidate_k)
    return fusion.fuse(dense_hits, sparse_hits, top_k=top_k)


def interleaved_union(
    dense_hits: Sequence[SearchHit],
    sparse_hits: Sequence[SearchHit],
    *,
    candidate_k: int = CANDIDATE_K_DEFAULT,
    top_k: int | None = None,
) -> list[SearchHit]:
    """Build the deterministic candidate-union baseline.

    Order is ``dense[0]``, ``sparse[0]``, ``dense[1]``, ``sparse[1]``, ... over
    the first ``candidate_k`` hits of each arm, with repeated identities emitted
    once at their first position. This is a baseline for candidate coverage,
    not a scoring model: ``score`` is ``0.0`` and ``rrf_score`` is ``None``.

    Args:
        dense_hits: Dense hits ordered best first.
        sparse_hits: Sparse hits ordered best first.
        candidate_k: Per-arm candidate pool, 1..1000.
        top_k: Maximum final hits; must not exceed ``candidate_k``.
            ``None`` uses ``min(10, candidate_k)``.

    Returns:
        Union hits in interleaved first-seen order, with provenance.

    Raises:
        ValueError: same validation errors as :meth:`RRFFusion.fuse`.
    """
    candidate_k = _validate_count(candidate_k, "candidate_k", maximum=COUNT_MAX)
    resolved_top_k = min(TOP_K_DEFAULT, candidate_k) if top_k is None else top_k
    _validate_top_k(resolved_top_k, candidate_k)
    dense = _candidate_slice(dense_hits, candidate_k, "dense")
    sparse = _candidate_slice(sparse_hits, candidate_k, "sparse")

    records: dict[ChunkKey, _Candidate] = {}
    ordinal = 0
    for index in range(max(len(dense), len(sparse))):
        if index < len(dense):
            candidate = _record_arm_hit(records, dense[index], index + 1, "dense", ordinal)
            candidate.dense_rank = index + 1
            candidate.dense_score = dense[index].score
            ordinal += 1
        if index < len(sparse):
            candidate = _record_arm_hit(records, sparse[index], index + 1, "sparse", ordinal)
            candidate.sparse_rank = index + 1
            candidate.sparse_score = sparse[index].score
            ordinal += 1

    ordered = sorted(records.values(), key=lambda item: item.first_ordinal)[:resolved_top_k]
    return [
        _fused_hit(candidate, rank=rank, score=0.0, rrf_score=None)
        for rank, candidate in enumerate(ordered, start=1)
    ]


def _validate_count(value: object, name: str, *, maximum: int) -> int:
    """Return ``value`` as a positive int, rejecting bools and out-of-range values."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}, got {value}")
    return value


def _validate_top_k(top_k: object, candidate_k: int) -> int:
    """Validate final Top-K and its relationship to the candidate pool."""
    validated = _validate_count(top_k, "top_k", maximum=COUNT_MAX)
    if validated > candidate_k:
        raise ValueError(f"top_k ({validated}) must not exceed candidate_k ({candidate_k})")
    return validated


def _candidate_slice(
    hits: Sequence[SearchHit],
    candidate_k: int,
    expected_source: Literal["dense", "sparse"],
) -> list[SearchHit]:
    """Validate one arm's list and return its candidate-truncated prefix."""
    selected = list(hits[:candidate_k])
    seen: set[ChunkKey] = set()
    for position, hit in enumerate(selected, start=1):
        if not isinstance(hit, SearchHit):
            raise ValueError(f"{expected_source} arm item {position} is not a SearchHit")
        if hit.source != expected_source:
            raise ValueError(
                f"{expected_source} arm item {position} has source {hit.source!r}; "
                f"expected {expected_source!r}"
            )
        key = _chunk_key(hit)
        if key in seen:
            raise ValueError(f"{expected_source} arm contains duplicate chunk identity {key!r}")
        seen.add(key)
        if not math.isfinite(hit.score):
            raise ValueError(f"{expected_source} arm has non-finite score for {key!r}")
    return selected


def _chunk_key(hit: SearchHit) -> ChunkKey:
    """Return the validated chunk identity for ``hit``."""
    if not isinstance(hit.chunk_strategy, str) or not hit.chunk_strategy:
        raise ValueError(f"hit {hit.chunk_id!r} has no chunk_strategy")
    if not isinstance(hit.chunk_id, str) or not hit.chunk_id:
        raise ValueError("hit has an empty chunk_id")
    return (hit.chunk_strategy, hit.chunk_id)


def _accumulate(
    dense: Sequence[SearchHit],
    sparse: Sequence[SearchHit],
    k: int,
) -> dict[ChunkKey, _Candidate]:
    """Accumulate both arms' RRF contributions by chunk identity."""
    candidates: dict[ChunkKey, _Candidate] = {}
    ordinal = 0
    for rank, hit in enumerate(dense, start=1):
        candidate = _record_arm_hit(candidates, hit, rank, "dense", ordinal)
        candidate.dense_rank = rank
        candidate.dense_score = hit.score
        candidate.rrf_score += 1.0 / (k + rank)
        ordinal += 1
    for rank, hit in enumerate(sparse, start=1):
        candidate = _record_arm_hit(candidates, hit, rank, "sparse", ordinal)
        candidate.sparse_rank = rank
        candidate.sparse_score = hit.score
        candidate.rrf_score += 1.0 / (k + rank)
        ordinal += 1
    return candidates


def _record_arm_hit(
    candidates: dict[ChunkKey, _Candidate],
    hit: SearchHit,
    rank: int,
    arm: Literal["dense", "sparse"],
    ordinal: int,
) -> _Candidate:
    """Add or merge one arm hit, validating cross-arm metadata agreement."""
    key = _chunk_key(hit)
    existing = candidates.get(key)
    if existing is None:
        candidate = _Candidate(key=key, first_ordinal=ordinal, base=hit)
        candidates[key] = candidate
    else:
        _assert_same_metadata(existing.base, hit)
        candidate = existing
    if arm not in candidate.sources:
        candidate.sources.append(arm)
    return candidate


def _assert_same_metadata(left: SearchHit, right: SearchHit) -> None:
    """Reject two arms that describe one chunk identity differently."""
    for name in (
        "doc_id",
        "chunk_index",
        "content",
        "title",
        "category",
        "card_tier",
        "effective_year",
        "policy_code",
    ):
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if left_value is not None and right_value is not None and left_value != right_value:
            raise ValueError(
                f"conflicting {name} for chunk {left.chunk_strategy}/{left.chunk_id}: "
                f"{left_value!r} != {right_value!r}"
            )


def _fused_hit(
    candidate: _Candidate,
    *,
    rank: int,
    score: float,
    rrf_score: float | None,
) -> SearchHit:
    """Materialise one fused hit from its accumulator and its base metadata."""
    base = candidate.base
    return SearchHit(
        chunk_id=base.chunk_id,
        doc_id=base.doc_id,
        score=score,
        distance=None,
        rank=rank,
        source="fused",
        dense_rank=candidate.dense_rank,
        sparse_rank=candidate.sparse_rank,
        dense_score=candidate.dense_score,
        sparse_score=candidate.sparse_score,
        rrf_score=rrf_score,
        candidate_sources=tuple(
            source for source in ("dense", "sparse") if source in candidate.sources
        ),
        content=base.content,
        chunk_strategy=base.chunk_strategy,
        chunk_index=base.chunk_index,
        title=base.title,
        category=base.category,
        card_tier=base.card_tier,
        effective_year=base.effective_year,
        policy_code=base.policy_code,
    )


def _finish(candidates: dict[ChunkKey, _Candidate], top_k: int) -> list[SearchHit]:
    """Order accumulated candidates and materialise fused hits."""
    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            -item.rrf_score,
            min(rank for rank in (item.dense_rank, item.sparse_rank) if rank is not None),
            item.key[0],
            item.key[1],
        ),
    )[:top_k]
    return [
        _fused_hit(candidate, rank=rank, score=candidate.rrf_score, rrf_score=candidate.rrf_score)
        for rank, candidate in enumerate(ordered, start=1)
    ]
