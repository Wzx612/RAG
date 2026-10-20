"""Phase 4C: metric correctness for the dense benchmark.

These are pure functions, so the tests use no database and no model. The
nearest-rank percentile rule is asserted with exact values rather than ranges
so a later interpolation change cannot pass silently.
"""

from __future__ import annotations

import math

import pytest
from src.evaluation.metrics import (
    LABELS_EXHAUSTIVE,
    METRIC_VERSION,
    RELEVANCE_SCHEME,
    dedupe_in_order,
    first_relevant_rank,
    hit_at_k,
    latency_summary,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

# --------------------------------------------------------------------- recall


def test_recall_at_k_perfect_with_smaller_gold_set() -> None:
    """All labelled docs retrieved -> 1.0, even though K is larger than gold."""
    assert recall_at_k(["d1", "d2", "d3"], ["d1", "d3"], k=10) == 1.0


def test_recall_at_k_partial() -> None:
    """One of two labelled docs present -> 0.5."""
    assert recall_at_k(["d1", "d2"], ["d1", "d3"], k=10) == pytest.approx(0.5)


def test_recall_at_k_respects_cutoff() -> None:
    """A gold doc below the cutoff does not count."""
    assert recall_at_k(["d1", "x", "d2"], ["d2"], k=2) == 0.0
    assert recall_at_k(["d1", "x", "d2"], ["d2"], k=3) == 1.0


def test_recall_at_k_denominator_is_reference_size_not_k() -> None:
    """ANN Recall@K uses the exact Top-K as its reference, not K itself."""
    exact = [f"c{i}" for i in range(10)]
    ann = exact[:8]
    assert recall_at_k(ann, exact, k=10) == pytest.approx(0.8)


def test_recall_at_k_deduplicates_both_sides() -> None:
    """Repeated ids cannot inflate recall above 1.0."""
    assert recall_at_k(["a", "a", "b"], ["a", "b", "b"], k=10) == 1.0


def test_recall_at_k_empty_retrieved_is_zero() -> None:
    """No results means no recall, not a division error."""
    assert recall_at_k([], ["a"], k=10) == 0.0


def test_recall_at_k_rejects_empty_reference() -> None:
    """A query without gold ids has no defined recall."""
    with pytest.raises(ValueError, match="at least one id"):
        recall_at_k(["a"], [], k=10)


@pytest.mark.parametrize("k", [0, -1])
def test_recall_at_k_rejects_non_positive_k(k: int) -> None:
    """K must be a positive cutoff."""
    with pytest.raises(ValueError, match="k must be positive"):
        recall_at_k(["a"], ["a"], k)


@pytest.mark.parametrize("k", [True, 1.5, "10"])
def test_recall_at_k_rejects_non_int_k(k: object) -> None:
    """Booleans and floats are not cutoffs."""
    with pytest.raises(ValueError, match="k must be an int"):
        recall_at_k(["a"], ["a"], k)  # type: ignore[arg-type]


# ---------------------------------------------------------------- precision


def test_metric_metadata_is_single_sourced() -> None:
    """The label contract is declared once and is not a graded-relevance claim."""
    assert LABELS_EXHAUSTIVE is False
    assert RELEVANCE_SCHEME == "binary_known_gold"
    assert METRIC_VERSION == 2


def test_precision_at_k_uses_a_fixed_k_denominator() -> None:
    """Two gold hits in K=10 score 0.2, even though gold has only two ids."""
    assert precision_at_k(["d1", "d2", "d3"], ["d1", "d3"], k=10) == pytest.approx(0.2)


def test_precision_at_k_partial_hit() -> None:
    """One gold hit in K=10 scores 0.1."""
    assert precision_at_k(["d1", "x"], ["d1", "d3"], k=10) == pytest.approx(0.1)


def test_precision_at_k_respects_cutoff() -> None:
    """A gold doc below the cutoff does not count, but the denominator stays K."""
    assert precision_at_k(["d1", "x", "d2"], ["d2"], k=2) == 0.0
    assert precision_at_k(["d1", "x", "d2"], ["d2"], k=3) == pytest.approx(1 / 3)


def test_precision_at_k_deduplicates_before_cutoff() -> None:
    """A repeated id occupies one effective rank, so the second rank is 'b'."""
    assert precision_at_k(["a", "a", "b"], ["b"], k=2) == pytest.approx(0.5)


def test_precision_at_k_count_is_shorter_than_k() -> None:
    """Fewer than K results still divides by K, not by the returned count."""
    assert precision_at_k(["a"], ["a", "b"], k=10) == pytest.approx(0.1)
    assert precision_at_k(["a"], ["a"], k=10) == pytest.approx(0.1)


def test_precision_at_k_empty_ranking_is_zero() -> None:
    """No returned documents means zero precision, not a division error."""
    assert precision_at_k([], ["a"], k=10) == 0.0


def test_precision_at_k_k_equals_one() -> None:
    """K=1 is a valid cutoff and only the first effective rank counts."""
    assert precision_at_k(["a", "b"], ["a"], k=1) == 1.0
    assert precision_at_k(["x", "a"], ["a"], k=1) == 0.0


def test_precision_at_k_rejects_empty_gold() -> None:
    """A query without a known gold set has no defined precision."""
    with pytest.raises(ValueError, match="at least one id"):
        precision_at_k(["a"], [], k=10)


@pytest.mark.parametrize("k", [0, -1])
def test_precision_at_k_rejects_non_positive_k(k: int) -> None:
    """K must be a positive cutoff."""
    with pytest.raises(ValueError, match="k must be positive"):
        precision_at_k(["a"], ["a"], k)


@pytest.mark.parametrize("k", [True, 1.5, "10"])
def test_precision_at_k_rejects_non_int_k(k: object) -> None:
    """Booleans, floats and strings are not cutoffs."""
    with pytest.raises(ValueError, match="k must be an int"):
        precision_at_k(["a"], ["a"], k)  # type: ignore[arg-type]


# --------------------------------------------------------------------- ndcg


def test_ndcg_at_k_perfect_ranking_is_one() -> None:
    """Both gold docs are at the top; unevaluated tail positions do not hurt."""
    assert ndcg_at_k(["a", "b", "x"], ["a", "b"], k=10) == 1.0


def test_ndcg_at_k_single_gold_at_rank_two() -> None:
    """DCG = 1/log2(3) and IDCG = 1 for one gold doc."""
    assert ndcg_at_k(["x", "a"], ["a"], k=10) == pytest.approx(0.6309297535714574)


def test_ndcg_at_k_penalises_gold_below_non_gold() -> None:
    """The ideal order scores 1.0; gold pushed down scores 1.5/IDCG."""
    ideal_idcg = 1.0 + 1.0 / math.log2(3.0)
    assert ndcg_at_k(["a", "b", "x"], ["a", "b"], k=3) == 1.0
    assert ndcg_at_k(["x", "b", "a"], ["a", "b"], k=3) == pytest.approx(
        1.1309297535714575 / ideal_idcg
    )


def test_ndcg_at_k_k_equals_one() -> None:
    """K=1 is valid and only the first effective rank contributes."""
    assert ndcg_at_k(["a", "b"], ["a"], k=1) == 1.0
    assert ndcg_at_k(["x", "a"], ["a"], k=1) == 0.0


def test_ndcg_at_k_respects_cutoff() -> None:
    """Gold below K contributes nothing; at K=3 it gains 1/log2(4)."""
    assert ndcg_at_k(["x", "y", "b"], ["b"], k=2) == 0.0
    assert ndcg_at_k(["x", "y", "b"], ["b"], k=3) == pytest.approx(0.5)


def test_ndcg_at_k_deduplicates_before_cutoff() -> None:
    """Repeated ids collapse first, so ['a','a','b'] is the ideal ['a','b']."""
    assert ndcg_at_k(["a", "a", "b"], ["a", "b"], k=3) == 1.0


def test_ndcg_at_k_reports_partial_ideal_when_fewer_results_than_gold() -> None:
    """IDCG still uses both known gold docs, so returning one cannot score 1.0."""
    assert ndcg_at_k(["a"], ["a", "b"], k=10) == pytest.approx(0.6131471927654584)


def test_ndcg_at_k_ideal_is_capped_by_k() -> None:
    """With K smaller than the gold set, IDCG only counts the best K gains."""
    assert ndcg_at_k(["a", "b"], ["a", "b", "c"], k=2) == 1.0


def test_ndcg_at_k_empty_ranking_is_zero() -> None:
    """An empty ranking has zero DCG, not a zero-division error."""
    assert ndcg_at_k([], ["a"], k=10) == 0.0


def test_ndcg_at_k_rejects_empty_gold() -> None:
    """A query without a known gold set has no defined ideal DCG."""
    with pytest.raises(ValueError, match="at least one id"):
        ndcg_at_k(["a"], [], k=10)


@pytest.mark.parametrize("k", [0, -1])
def test_ndcg_at_k_rejects_non_positive_k(k: int) -> None:
    """K must be a positive cutoff."""
    with pytest.raises(ValueError, match="k must be positive"):
        ndcg_at_k(["a"], ["a"], k)


@pytest.mark.parametrize("k", [True, 1.5, "10"])
def test_ndcg_at_k_rejects_non_int_k(k: object) -> None:
    """Booleans, floats and strings are not cutoffs."""
    with pytest.raises(ValueError, match="k must be an int"):
        ndcg_at_k(["a"], ["a"], k)  # type: ignore[arg-type]


def test_ranking_metrics_do_not_mutate_inputs() -> None:
    """Deduplication and truncation operate on copies."""
    retrieved = ["a", "a", "b"]
    gold = ["a", "b", "b"]
    precision_at_k(retrieved, gold, k=2)
    ndcg_at_k(retrieved, gold, k=2)
    assert retrieved == ["a", "a", "b"]
    assert gold == ["a", "b", "b"]


# ----------------------------------------------------------------- percentile


def test_percentile_nearest_rank_odd_count() -> None:
    """P50 of five samples is the third observed value."""
    assert percentile([5, 1, 4, 2, 3], 50) == 3.0


def test_percentile_nearest_rank_even_count() -> None:
    """P50 of four samples uses ceil(2), not an average of the middle pair."""
    assert percentile([4, 1, 3, 2], 50) == 2.0


def test_percentile_p95_uses_observed_sample() -> None:
    """P95 of 20 samples is the 19th value under nearest rank."""
    samples = [float(value) for value in range(1, 21)]
    assert percentile(samples, 95) == 19.0


def test_percentile_extremes() -> None:
    """P0 and P100 are the minimum and maximum."""
    samples = [3.5, 1.25, 9.0]
    assert percentile(samples, 0) == 1.25
    assert percentile(samples, 100) == 9.0


def test_percentile_p99_clamps_to_max() -> None:
    """ceil(0.99 * 20) is 20, so P99 is the largest sample, not index 21."""
    samples = [float(value) for value in range(1, 21)]
    assert percentile(samples, 99) == 20.0


def test_percentile_does_not_mutate_input() -> None:
    """Sorting happens on a copy."""
    samples = [3.0, 1.0, 2.0]
    percentile(samples, 50)
    assert samples == [3.0, 1.0, 2.0]


@pytest.mark.parametrize("p", [-0.1, 100.1, True, "95"])
def test_percentile_rejects_invalid_p(p: object) -> None:
    """The percentile must be a number in [0, 100]."""
    with pytest.raises(ValueError, match="p must be"):
        percentile([1.0], p)  # type: ignore[arg-type]


def test_percentile_rejects_empty_samples() -> None:
    """No samples means no percentile."""
    with pytest.raises(ValueError, match="must not be empty"):
        percentile([], 50)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_percentile_rejects_non_finite_samples(bad: float) -> None:
    """NaN or infinity cannot be a latency."""
    with pytest.raises(ValueError, match="finite"):
        percentile([1.0, bad], 50)


# ------------------------------------------------------- shared latency summary


def test_latency_summary_reports_count_and_all_percentiles() -> None:
    """The shared summary carries n, min, P50/P95/P99, max and mean."""
    summary = latency_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["n"] == 5
    assert summary["min"] == 1.0
    assert summary["p50"] == 3.0
    assert summary["p95"] == 5.0
    assert summary["p99"] == 5.0
    assert summary["max"] == 5.0
    assert summary["mean"] == pytest.approx(3.0)


def test_latency_summary_uses_the_same_nearest_rank_rule() -> None:
    """P99 of 100 samples is the 99th observed value, not an interpolation."""
    samples = [float(value) for value in range(1, 101)]
    summary = latency_summary(samples)
    assert summary["p50"] == 50.0
    assert summary["p95"] == 95.0
    assert summary["p99"] == 99.0


def test_latency_summary_empty_is_none_not_zero() -> None:
    """An unmeasured latency must not be reported as a fast one."""
    summary = latency_summary([])
    assert summary == {
        "n": 0,
        "min": None,
        "p50": None,
        "p95": None,
        "p99": None,
        "max": None,
        "mean": None,
    }


def test_latency_summary_does_not_mutate_input() -> None:
    """Ordering is restored on the caller's list."""
    samples = [3.0, 1.0, 2.0]
    latency_summary(samples)
    assert samples == [3.0, 1.0, 2.0]


# -------------------------------------------------------------- rank and MRR


def test_dedupe_in_order_keeps_first_rank() -> None:
    """Document-level ranking collapses repeats at their best position."""
    assert dedupe_in_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_first_relevant_rank_counts_effective_ranks() -> None:
    """A duplicate before the gold id does not push its rank back."""
    assert first_relevant_rank(["a", "a", "b"], ["b"], k=10) == 2


def test_first_relevant_rank_respects_cutoff() -> None:
    """A gold id below the cutoff is a miss, not rank 3."""
    assert first_relevant_rank(["x", "y", "b"], ["b"], k=2) is None
    assert first_relevant_rank(["x", "y", "b"], ["b"], k=3) == 3


def test_first_relevant_rank_empty_ranking_is_none() -> None:
    """No results means no rank."""
    assert first_relevant_rank([], ["b"], k=10) is None


def test_reciprocal_rank_at_k_matches_one_over_rank() -> None:
    """RR is exactly 1/rank for a hit and 0.0 for a miss."""
    assert reciprocal_rank_at_k(["x", "b"], ["b"], k=10) == pytest.approx(0.5)
    assert reciprocal_rank_at_k(["x", "y"], ["b"], k=10) == 0.0


def test_mrr_at_k_means_per_query_reciprocal_ranks() -> None:
    """MRR averages the per-query terms; a miss contributes zero."""
    rankings = [(["a"], ["a"]), (["x", "b"], ["b"]), (["x", "y"], ["b"])]
    assert mrr_at_k(rankings, k=10) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_mrr_at_k_empty_rankings_is_zero() -> None:
    """No evaluated queries gives 0.0, which callers report as no sample."""
    assert mrr_at_k([], k=10) == 0.0


def test_mrr_at_k_rejects_query_without_gold() -> None:
    """Every pair needs at least one relevant id."""
    with pytest.raises(ValueError, match="at least one id"):
        mrr_at_k([(["a"], [])], k=10)


@pytest.mark.parametrize("k", [0, -1])
def test_rank_metrics_reject_non_positive_k(k: int) -> None:
    """The cutoff rule is shared with recall/precision/nDCG."""
    with pytest.raises(ValueError, match="k must be positive"):
        first_relevant_rank(["a"], ["a"], k)


def test_hit_at_k_is_true_only_for_a_hit_inside_the_cutoff() -> None:
    """Hit@K and first_relevant_rank cannot disagree."""
    assert hit_at_k(["x", "a"], ["a"], k=2) is True
    assert hit_at_k(["x", "a"], ["a"], k=1) is False
    assert hit_at_k([], ["a"], k=10) is False
