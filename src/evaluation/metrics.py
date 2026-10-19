"""Retrieval metrics.

Pure functions: ``recall_at_k``, ``precision_at_k``, ``ndcg_at_k`` and
``percentile`` are implemented here. MRR and Hit@1 remain inline in the hybrid
benchmark because they depend on its document-level arm record shape.

Golden results in, floats out. No I/O, no DB.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

LABELS_EXHAUSTIVE = False
"""The gold labels are a known-relevant set, not a complete relevance judgement."""

RELEVANCE_SCHEME = "binary_known_gold"
"""Only known gold document ids are relevant; other ids are unlabelled."""

METRIC_VERSION = 2
"""Metric contract version for the P@K / binary nDCG@K extension."""


def _validate_cutoff(k: int) -> None:
    """Validate the 1-based cutoff shared by the ranking metrics."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError(f"k must be an int, got {type(k).__name__}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Return set-based Recall@K of ``retrieved`` against ``relevant``.

    The denominator is the number of distinct ids in ``relevant`` -- not ``k``.
    That is the standard definition, and it keeps the two uses in this project
    separate but consistent:

    * ANN Recall@K: ``relevant`` is the exact vector Top-K; its size is at most
      ``k``, so a perfect ANN result scores 1.0.
    * Semantic Recall@K: ``relevant`` is the human-labelled gold doc set, whose
      size is 1--3 here; returning all gold docs scores 1.0 even though K=10.

    Duplicate ids in either argument are collapsed before the intersection, so
    a repeated result can never inflate recall above 1.0.

    Args:
        retrieved: Ranked ids, best first. Only the first ``k`` are considered.
        relevant: Reference ids (exact Top-K or labelled gold docs).
        k: Cutoff, 1-based. Entries after position ``k`` are ignored.

    Returns:
        ``|top_k(retrieved) ∩ relevant| / |relevant|`` in ``[0, 1]``.

    Raises:
        ValueError: ``k`` is not a positive int, or ``relevant`` is empty.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError(f"k must be an int, got {type(k).__name__}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    reference = list(dict.fromkeys(relevant))
    if not reference:
        raise ValueError("relevant must contain at least one id")

    top_k = set(list(retrieved)[:k])
    return len(top_k.intersection(reference)) / len(reference)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Return Precision@K against the known gold ids.

    The denominator is always ``k``. Returning fewer than ``k`` documents, or
    having fewer than ``k`` gold documents, does not shrink it. Duplicate
    retrieved ids are collapsed in rank order *before* the cutoff, so a repeated
    id occupies one effective rank and cannot inflate precision.

    Only ids in ``relevant`` are treated as relevant. With
    ``LABELS_EXHAUSTIVE = False``, a non-gold hit is unlabelled rather than a
    confirmed true negative; this metric is therefore supplementary and
    label-limited, not a complete relevance judgement.

    Args:
        retrieved: Ranked ids, best first.
        relevant: Known gold ids. Must be non-empty.
        k: Positive cutoff.

    Returns:
        ``|dedup(retrieved)[:k] ∩ dedup(relevant)| / k`` in ``[0, 1]``.

    Raises:
        ValueError: ``k`` is not a positive int, or ``relevant`` is empty.
    """
    _validate_cutoff(k)
    gold = set(dict.fromkeys(relevant))
    if not gold:
        raise ValueError("relevant must contain at least one id")

    ranked = list(dict.fromkeys(retrieved))[:k]
    return sum(1 for doc_id in ranked if doc_id in gold) / k


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Return binary nDCG@K against the known gold ids.

    Relevance is binary: a known gold id scores 1, every other id scores 0.
    Non-gold ids are not asserted to be irrelevant -- they are simply
    unlabelled, which is why the result carries ``LABELS_EXHAUSTIVE = False``
    and ``RELEVANCE_SCHEME = "binary_known_gold"`` at the output layer.

    Duplicate retrieved ids are collapsed in rank order *before* the cutoff,
    matching the document-level ranking used by the benchmarks. The ideal DCG
    uses the same ``k`` cutoff and the full known gold set:
    ``sum(1 / log2(i + 1) for i in 1..min(|gold|, k))``.

    Args:
        retrieved: Ranked ids, best first.
        relevant: Known gold ids. Must be non-empty.
        k: Positive cutoff.

    Returns:
        ``DCG@K / IDCG@K`` in ``[0, 1]``; ``0.0`` when ``DCG@K`` is zero.

    Raises:
        ValueError: ``k`` is not a positive int, or ``relevant`` is empty.
    """
    _validate_cutoff(k)
    gold = set(dict.fromkeys(relevant))
    if not gold:
        raise ValueError("relevant must contain at least one id")

    ranked = list(dict.fromkeys(retrieved))[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked, start=1)
        if doc_id in gold
    )
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def percentile(samples: Sequence[float], p: float) -> float:
    """Return the nearest-rank percentile of ``samples``.

    Nearest rank is used deliberately: it returns an actually observed latency,
    not an interpolated value that no request produced. For ``n`` samples the
    rank is ``ceil(p / 100 * n)``, clamped to ``[1, n]``; P50/P95/P99 use the
    same rule.

    Args:
        samples: Non-empty finite durations, in any order. Units are preserved.
        p: Percentile in ``[0, 100]``. ``0`` returns the minimum and ``100`` the
            maximum.

    Returns:
        The observed sample at the nearest rank.

    Raises:
        ValueError: ``samples`` is empty, contains a non-finite value, or ``p``
            is outside ``[0, 100]``.
    """
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise ValueError(f"p must be a number, got {type(p).__name__}")
    if not 0.0 <= float(p) <= 100.0:
        raise ValueError(f"p must be between 0 and 100, got {p}")

    values = [float(value) for value in samples]
    if not values:
        raise ValueError("samples must not be empty")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("samples must contain only finite values")

    values.sort()
    if p == 0:
        return values[0]
    rank = math.ceil((float(p) / 100.0) * len(values))
    rank = min(max(rank, 1), len(values))
    return values[rank - 1]
