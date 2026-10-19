"""Phase 6B: reciprocal rank fusion and the candidate-union baseline.

These are pure unit tests: no database, no model, no filesystem. They pin the
FORMULA and the identity/ordering rules that the evaluation depends on.
"""

from __future__ import annotations

import math
from typing import Literal

import pytest
from src.retrieval.fusion import (
    RRFFusion,
    interleaved_union,
    reciprocal_rank_fusion,
)
from src.schema.retrieval import SearchHit


def _hit(
    chunk_id: str,
    *,
    source: Literal["dense", "sparse"] = "dense",
    rank: int = 1,
    score: float = 1.0,
    strategy: str = "document",
    doc_id: str | None = None,
    content: str | None = None,
) -> SearchHit:
    """Build a valid arm hit with stable test metadata."""
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=doc_id or f"DOC-{chunk_id}",
        score=score,
        distance=1.0 - score if source == "dense" else None,
        rank=rank,
        source=source,
        content=content or f"content-{chunk_id}",
        chunk_strategy=strategy,
        chunk_index=0,
        title=f"title-{chunk_id}",
        category="fees",
        card_tier="standard",
        effective_year=2025,
        policy_code="P-1",
    )


def _ids(hits: list[SearchHit]) -> list[str]:
    """Chunk ids in returned order."""
    return [hit.chunk_id for hit in hits]


# --------------------------------------------------------------------- formula


def test_rrf_formula_overlapping_and_single_arm_candidates() -> None:
    """Overlap sums both terms; single-arm candidates keep only their own."""
    dense = [_hit("A", rank=1, score=0.9), _hit("B", rank=2, score=0.8)]
    sparse = [
        _hit("B", source="sparse", rank=1, score=5.0),
        _hit("C", source="sparse", rank=2, score=4.0),
    ]

    fused = reciprocal_rank_fusion(dense, sparse, k=60, candidate_k=10, top_k=10)

    assert _ids(fused) == ["B", "A", "C"]
    assert [hit.rank for hit in fused] == [1, 2, 3]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)
    assert all(hit.rrf_score == hit.score for hit in fused)

    both = fused[0]
    assert both.dense_rank == 2
    assert both.sparse_rank == 1
    assert both.dense_score == pytest.approx(0.8)
    assert both.sparse_score == pytest.approx(5.0)
    assert both.candidate_sources == ("dense", "sparse")
    assert both.source == "fused"
    assert both.distance is None
    assert both.content == "content-B"
    assert both.doc_id == "DOC-B"

    assert fused[1].dense_rank == 1
    assert fused[1].sparse_rank is None
    assert fused[1].candidate_sources == ("dense",)
    assert fused[2].dense_rank is None
    assert fused[2].sparse_rank == 2
    assert fused[2].candidate_sources == ("sparse",)


def test_rank_starts_at_one_not_zero() -> None:
    """A single first-place candidate scores 1/(k+1), not 1/k."""
    fused = reciprocal_rank_fusion([_hit("A")], [], k=60, candidate_k=5, top_k=5)
    assert fused[0].dense_rank == 1
    assert fused[0].score == pytest.approx(1 / 61)


def test_raw_scores_are_never_added_to_the_fused_score() -> None:
    """Dense similarity and sparse rank are not commensurate and stay separate."""
    dense = [_hit("A", score=0.9)]
    sparse = [_hit("A", source="sparse", score=7.5)]
    fused = reciprocal_rank_fusion(dense, sparse, k=60, candidate_k=5, top_k=5)
    assert fused[0].score == pytest.approx(2 / 61)
    assert fused[0].score != pytest.approx(0.9 + 7.5)
    assert fused[0].dense_score == pytest.approx(0.9)
    assert fused[0].sparse_score == pytest.approx(7.5)


def test_rrf_constant_is_configurable_and_recorded() -> None:
    """The class exposes the exact k later recorded in evaluation output."""
    fusion = RRFFusion(k=12, candidate_k=7)
    assert fusion.k == 12
    assert fusion.candidate_k == 7
    fused = fusion.fuse([_hit("A")], [])
    assert fused[0].score == pytest.approx(1 / 13)


# ------------------------------------------------------------- arm edge cases


def test_one_empty_arm_returns_the_other_arms_candidates() -> None:
    """Absence from one arm is not failure; it simply contributes no term."""
    dense_only = reciprocal_rank_fusion(
        [_hit("A", rank=1, score=0.9), _hit("B", rank=2, score=0.8)],
        [],
        k=60,
        candidate_k=10,
        top_k=10,
    )
    assert _ids(dense_only) == ["A", "B"]
    assert dense_only[0].candidate_sources == ("dense",)

    sparse_only = reciprocal_rank_fusion(
        [],
        [_hit("C", source="sparse", score=2.5)],
        k=60,
        candidate_k=10,
        top_k=10,
    )
    assert _ids(sparse_only) == ["C"]
    assert sparse_only[0].candidate_sources == ("sparse",)
    assert sparse_only[0].sparse_score == pytest.approx(2.5)


def test_both_arms_empty_returns_empty() -> None:
    """No candidates is an empty fused result, not an exception."""
    assert reciprocal_rank_fusion([], [], candidate_k=10, top_k=10) == []
    assert interleaved_union([], [], candidate_k=10, top_k=10) == []


def test_duplicate_identity_within_one_arm_is_rejected() -> None:
    """A repeated chunk in one ranked list is an arm contract violation."""
    duplicate = [_hit("A", rank=1), _hit("A", rank=2, score=0.5)]
    with pytest.raises(ValueError, match="duplicate chunk identity"):
        reciprocal_rank_fusion(duplicate, [], candidate_k=10, top_k=10)
    sparse_duplicate = [
        _hit("A", source="sparse", rank=1),
        _hit("A", source="sparse", rank=2, score=0.5),
    ]
    with pytest.raises(ValueError, match="duplicate chunk identity"):
        interleaved_union([], sparse_duplicate, candidate_k=10, top_k=10)


def test_same_identity_across_arms_is_one_candidate() -> None:
    """A chunk found by both arms appears once and records both ranks."""
    dense = [_hit("A", rank=1, score=0.9)]
    sparse = [_hit("A", source="sparse", rank=1, score=3.0)]
    fused = reciprocal_rank_fusion(dense, sparse, candidate_k=10, top_k=10)
    assert len(fused) == 1
    assert fused[0].candidate_sources == ("dense", "sparse")
    assert fused[0].dense_rank == 1
    assert fused[0].sparse_rank == 1


def test_identity_is_chunk_level_not_document_level() -> None:
    """Two chunks of one document remain two fusion candidates."""
    dense = [_hit("chunk-1", doc_id="DOC-1")]
    sparse = [
        _hit(
            "chunk-2",
            source="sparse",
            doc_id="DOC-1",
            strategy="fixed_window",
        )
    ]
    fused = reciprocal_rank_fusion(dense, sparse, candidate_k=10, top_k=10)
    assert len(fused) == 2
    assert {hit.chunk_id for hit in fused} == {"chunk-1", "chunk-2"}
    assert {hit.doc_id for hit in fused} == {"DOC-1"}


def test_conflicting_metadata_for_one_identity_is_rejected() -> None:
    """The same chunk identity cannot describe two different documents."""
    dense = [_hit("A", doc_id="DOC-1")]
    sparse = [_hit("A", source="sparse", doc_id="DOC-2")]
    with pytest.raises(ValueError, match="conflicting doc_id"):
        reciprocal_rank_fusion(dense, sparse, candidate_k=10, top_k=10)


def test_wrong_arm_source_is_rejected() -> None:
    """Passing sparse hits as the dense arm would mislabel provenance."""
    with pytest.raises(ValueError, match="has source 'sparse'"):
        reciprocal_rank_fusion([_hit("A", source="sparse")], [], candidate_k=10, top_k=10)


def test_non_finite_arm_score_is_rejected() -> None:
    """NaN/inf arm scores cannot enter provenance."""
    with pytest.raises(ValueError, match="non-finite score"):
        reciprocal_rank_fusion([_hit("A", score=math.nan)], [], candidate_k=10, top_k=10)
    with pytest.raises(ValueError, match="non-finite score"):
        interleaved_union([_hit("A", score=math.inf)], [], candidate_k=10, top_k=10)


def test_hit_without_chunk_strategy_is_rejected() -> None:
    """Chunk identity requires the strategy, not just the id."""
    hit = SearchHit(chunk_id="A", doc_id="DOC-A", score=1.0, rank=1, source="dense")
    with pytest.raises(ValueError, match="no chunk_strategy"):
        reciprocal_rank_fusion([hit], [], candidate_k=10, top_k=10)


# ------------------------------------------------------------------ truncation


def test_candidate_pool_truncates_both_arms_before_fusion() -> None:
    """Only the first candidate_k hits of each arm enter the fusion."""
    dense = [_hit("A", rank=1), _hit("B", rank=2), _hit("C", rank=3)]
    sparse = [
        _hit("D", source="sparse", rank=1),
        _hit("E", source="sparse", rank=2),
        _hit("F", source="sparse", rank=3),
    ]
    fused = reciprocal_rank_fusion(dense, sparse, k=60, candidate_k=2, top_k=2)
    assert _ids(fused) == ["A", "D"]
    assert {hit.chunk_id for hit in fused} == {"A", "D"}


def test_final_top_k_truncates_after_fusion() -> None:
    """Final Top-K is applied after scoring the whole candidate pool."""
    dense = [_hit("A", rank=1), _hit("B", rank=2), _hit("C", rank=3)]
    fused = reciprocal_rank_fusion(dense, [], k=60, candidate_k=3, top_k=2)
    assert _ids(fused) == ["A", "B"]
    assert [hit.rank for hit in fused] == [1, 2]


def test_equal_scores_tie_break_deterministically() -> None:
    """Equal fused scores fall back to best arm rank, then chunk identity."""
    dense = [_hit("B", rank=1), _hit("A", rank=2)]
    sparse = [_hit("A", source="sparse", rank=1), _hit("B", source="sparse", rank=2)]
    first = reciprocal_rank_fusion(dense, sparse, k=60, candidate_k=10, top_k=10)
    second = reciprocal_rank_fusion(dense, sparse, k=60, candidate_k=10, top_k=10)
    assert _ids(first) == ["A", "B"]
    assert _ids(second) == _ids(first)
    assert first[0].score == pytest.approx(first[1].score)


# ----------------------------------------------------------------- validation


@pytest.mark.parametrize("bad_k", [0, -1, True, 1.5, "60"])
def test_invalid_rrf_k_is_rejected(bad_k: object) -> None:
    """K must be a positive int and is validated at construction."""
    with pytest.raises(ValueError, match="k must be"):
        RRFFusion(k=bad_k)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_candidate_k", [0, -1, True, 1001, 2.5])
def test_invalid_candidate_k_is_rejected(bad_candidate_k: object) -> None:
    """candidate_k is bounded like a retriever Top-K."""
    with pytest.raises(ValueError, match="candidate_k must be"):
        reciprocal_rank_fusion(
            [],
            [],
            candidate_k=bad_candidate_k,  # type: ignore[arg-type]
            top_k=1,
        )


@pytest.mark.parametrize("bad_top_k", [0, -1, True, 11, 1.5])
def test_invalid_top_k_is_rejected(bad_top_k: object) -> None:
    """top_k must be a positive int no larger than the candidate pool."""
    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion(
            [],
            [],
            candidate_k=10,
            top_k=bad_top_k,  # type: ignore[arg-type]
        )


# --------------------------------------------------------- candidate union


def test_interleaved_union_has_explicit_reproducible_order() -> None:
    """Union order is dense[0], sparse[0], dense[1], sparse[1], ... after dedup."""
    dense = [_hit("A", rank=1), _hit("B", rank=2)]
    sparse = [
        _hit("B", source="sparse", rank=1),
        _hit("C", source="sparse", rank=2),
    ]
    union = interleaved_union(dense, sparse, candidate_k=10, top_k=10)

    assert _ids(union) == ["A", "B", "C"]
    assert [hit.rank for hit in union] == [1, 2, 3]
    assert all(hit.source == "fused" for hit in union)
    assert all(hit.score == 0.0 for hit in union)
    assert all(hit.rrf_score is None for hit in union)
    assert union[1].candidate_sources == ("dense", "sparse")
    assert union[1].dense_rank == 2
    assert union[1].sparse_rank == 1


def test_interleaved_union_truncates_candidates_and_final_list() -> None:
    """Both truncations apply to the union baseline too."""
    dense = [_hit("A", rank=1), _hit("B", rank=2)]
    sparse = [
        _hit("C", source="sparse", rank=1),
        _hit("D", source="sparse", rank=2),
    ]
    union = interleaved_union(dense, sparse, candidate_k=2, top_k=2)
    assert _ids(union) == ["A", "C"]


def test_interleaved_union_with_one_empty_arm() -> None:
    """A missing arm leaves the other arm's order intact."""
    dense = [_hit("A", rank=1), _hit("B", rank=2)]
    assert _ids(interleaved_union(dense, [], candidate_k=10, top_k=10)) == ["A", "B"]
    sparse = [_hit("C", source="sparse", rank=1)]
    assert _ids(interleaved_union([], sparse, candidate_k=10, top_k=10)) == ["C"]


def test_interleaved_union_rejects_invalid_top_k() -> None:
    """Union validates its parameters exactly like RRF."""
    with pytest.raises(ValueError, match="must not exceed candidate_k"):
        interleaved_union([], [], candidate_k=2, top_k=3)
