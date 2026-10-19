"""Phase 6B hybrid-retrieval benchmark.

Runs five arms over the same labelled queries, chunk identities, candidate
pool and final Top-K:

1. ``dense`` -- :class:`~src.retrieval.dense.DenseRetriever`;
2. ``sparse_or`` -- :class:`~src.retrieval.sparse.SparseRetriever` in OR mode;
3. ``sparse_and`` -- the same sparse path in strict AND mode;
4. ``candidate_union`` -- deterministic interleaved dense/sparse union;
5. ``rrf`` -- standard Reciprocal Rank Fusion of the dense and sparse OR arms.

The module executes retrieval and records; it does not rerank, tune parameters,
or alter labels. ``run_hybrid_benchmark`` accepts injected retrievers so the
whole arm/metric path can be tested without a database or embedding model.

Metric conventions (also written into the JSON ``definitions``):

* candidate identity is chunk-level ``(chunk_strategy, chunk_id)``;
* quality metrics are document-level: ``doc_id`` is deduplicated in rank order;
* Recall@K uses the human ``relevant_doc_ids``; its denominator is the number
  of distinct gold docs, not K;
* MRR uses the first gold document at doc rank ``r <= K`` and contributes
  ``1/r``; a query with no gold in its Top-K contributes ``0``;
* MRR/Recall means are over evaluated queries; PASS requires every arm to be
  evaluated for every query.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from src.config import PROJECT_ROOT, get_settings
from src.evaluation.dense_benchmark import (
    BenchmarkError,
    EvalQuery,
    load_eval_queries,
)
from src.evaluation.metrics import (
    LABELS_EXHAUSTIVE,
    METRIC_VERSION,
    RELEVANCE_SCHEME,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
)
from src.retrieval.base import Retriever
from src.retrieval.fusion import (
    CANDIDATE_K_DEFAULT,
    COUNT_MAX,
    RRF_K_DEFAULT,
    TOP_K_DEFAULT,
    RRFFusion,
    interleaved_union,
)
from src.schema.retrieval import DenseConfig, SearchHit, SparseConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

ARM_NAMES = ("dense", "sparse_or", "sparse_and", "candidate_union", "rrf")
"""The five evaluation arms, in report order."""

FOCUS_MISSES: tuple[tuple[str, str], ...] = (
    ("Q-004", "DOC-0017"),
    ("Q-005", "DOC-0021"),
    ("Q-009", "DOC-0039"),
    ("Q-009", "DOC-0040"),
    ("Q-014", "DOC-0067"),
)
"""Gold documents absent from dense Top-10 in the Phase 4C artifact."""

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "hybrid_benchmark.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "hybrid_benchmark.md"

ProgressFn = Callable[[str], None]

ArmName = Literal["dense", "sparse_or", "sparse_and", "candidate_union", "rrf"]


@dataclass(frozen=True)
class HybridBenchmarkConfig:
    """Fixed parameters shared by every arm in one run."""

    top_k: int = TOP_K_DEFAULT
    candidate_k: int = CANDIDATE_K_DEFAULT
    rrf_k: int = RRF_K_DEFAULT
    repetitions: int = 5
    warmup_passes: int = 1


# ------------------------------------------------------------------ validation


def build_config(
    top_k: int = TOP_K_DEFAULT,
    candidate_k: int = CANDIDATE_K_DEFAULT,
    rrf_k: int = RRF_K_DEFAULT,
    repetitions: int = 5,
    warmup_passes: int = 1,
) -> HybridBenchmarkConfig:
    """Validate benchmark parameters through the production config models.

    Args:
        top_k: Final list length evaluated by the metrics.
        candidate_k: Per-arm candidate pool. Must be at least ``top_k``.
        rrf_k: Standard RRF smoothing constant.
        repetitions: Timed passes per query.
        warmup_passes: Untimed passes per query.

    Returns:
        A validated configuration.

    Raises:
        BenchmarkError: if any parameter is outside its accepted range.
    """
    if (
        isinstance(candidate_k, bool)
        or not isinstance(candidate_k, int)
        or not 1 <= candidate_k <= COUNT_MAX
    ):
        raise BenchmarkError(f"candidate_k must be an int between 1 and {COUNT_MAX}")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 1:
        raise BenchmarkError("rrf_k must be a positive int")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= candidate_k:
        raise BenchmarkError(f"top_k must be an int between 1 and candidate_k, got {top_k!r}")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise BenchmarkError("repetitions must be an int of at least 1")
    if isinstance(warmup_passes, bool) or not isinstance(warmup_passes, int) or warmup_passes < 0:
        raise BenchmarkError("warmup_passes must be a non-negative int")
    try:
        DenseConfig(top_n=candidate_k)
        SparseConfig(top_n=candidate_k)
        fusion = RRFFusion(k=rrf_k, candidate_k=candidate_k)
    except ValueError as exc:  # pydantic ValidationError subclasses ValueError
        raise BenchmarkError(f"invalid benchmark configuration: {exc}") from exc
    return HybridBenchmarkConfig(
        top_k=top_k,
        candidate_k=candidate_k,
        rrf_k=fusion.k,
        repetitions=repetitions,
        warmup_passes=warmup_passes,
    )


# ------------------------------------------------------------------- execution


def _default_dense(candidate_k: int) -> Retriever:
    """Construct the production dense retriever lazily."""
    from src.retrieval.dense import DenseRetriever

    return DenseRetriever(DenseConfig(top_n=candidate_k))


def _default_sparse(candidate_k: int, query_mode: Literal["or", "and"]) -> Retriever:
    """Construct a production sparse retriever in the requested mode."""
    from src.retrieval.sparse import SparseRetriever

    return SparseRetriever(SparseConfig(top_n=candidate_k, query_mode=query_mode))


@dataclass
class _Attempt:
    """One arm call: hits, elapsed time, or an isolated error."""

    hits: list[SearchHit] | None
    elapsed_ms: float
    error: BaseException | None


def _attempt(
    retriever: Retriever,
    query: str,
    top_k: int,
) -> _Attempt:
    """Run one retriever, converting any exception into an isolated record."""
    started = time.perf_counter()
    try:
        hits = list(retriever.retrieve(query, top_k=top_k))
    except Exception as exc:  # noqa: BLE001 -- one arm must not abort the run
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return _Attempt(hits=None, elapsed_ms=elapsed_ms, error=exc)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return _Attempt(hits=hits, elapsed_ms=elapsed_ms, error=None)


def _error_record(
    exc: BaseException,
    *,
    stage: str,
    arm: str,
    query_id: str,
    pass_number: int,
) -> dict[str, Any]:
    """Serialise an isolated failure without a traceback or query vector."""
    return {
        "stage": stage,
        "arm": arm,
        "query_id": query_id,
        "pass_number": pass_number,
        "exception": type(exc).__name__,
        "message": str(exc)[:500],
    }


def _hit_record(hit: SearchHit) -> dict[str, Any]:
    """Serialise one hit with all provenance needed to recompute fusion."""
    return {
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "chunk_strategy": hit.chunk_strategy,
        "chunk_index": hit.chunk_index,
        "score": hit.score,
        "distance": hit.distance,
        "rank": hit.rank,
        "source": hit.source,
        "dense_rank": hit.dense_rank,
        "sparse_rank": hit.sparse_rank,
        "dense_score": hit.dense_score,
        "sparse_score": hit.sparse_score,
        "rrf_score": hit.rrf_score,
        "candidate_sources": list(hit.candidate_sources),
    }


def _measure(
    query: EvalQuery,
    hits: Sequence[SearchHit],
    top_k: int,
) -> dict[str, Any]:
    """Compute one arm's document-level metrics for one query."""
    gold = set(query.relevant_doc_ids)
    all_doc_ids = list(dict.fromkeys(hit.doc_id for hit in hits))
    doc_ids = all_doc_ids[:top_k]
    gold_ranks = {
        doc_id: rank for rank, doc_id in enumerate(all_doc_ids, start=1) if doc_id in gold
    }
    first_gold_rank = next(
        (rank for rank, doc_id in enumerate(all_doc_ids, start=1) if doc_id in gold),
        None,
    )
    recalled = len(set(doc_ids) & gold)
    return {
        "evaluated": True,
        "result_count": len(hits),
        "doc_unique_count": len(all_doc_ids),
        "empty_result": not hits,
        "doc_ids": doc_ids,
        "candidate_doc_ids": all_doc_ids,
        "gold_doc_ids": list(query.relevant_doc_ids),
        "gold_ranks": gold_ranks,
        "first_gold_rank": first_gold_rank,
        "reciprocal_rank": (
            1.0 / first_gold_rank
            if first_gold_rank is not None and first_gold_rank <= top_k
            else 0.0
        ),
        "hit_at_1": bool(doc_ids) and doc_ids[0] in gold,
        "recalled_gold": recalled,
        "gold_count": len(gold),
        "semantic_recall": recall_at_k(doc_ids, query.relevant_doc_ids, top_k),
        "precision_at_k": precision_at_k(doc_ids, query.relevant_doc_ids, top_k),
        "precision_at_k_ceiling": min(len(gold), top_k) / top_k,
        "ndcg_at_k": ndcg_at_k(doc_ids, query.relevant_doc_ids, top_k),
        "hits": [_hit_record(hit) for hit in hits],
    }


def _not_evaluated(error_count: int) -> dict[str, Any]:
    """The shape of an arm record with no successful pass."""
    return {
        "evaluated": False,
        "errors": error_count,
        "result_count": None,
        "doc_ids": [],
        "candidate_doc_ids": [],
        "gold_doc_ids": [],
        "gold_ranks": {},
        "first_gold_rank": None,
        "reciprocal_rank": None,
        "hit_at_1": None,
        "semantic_recall": None,
        "precision_at_k": None,
        "precision_at_k_ceiling": None,
        "ndcg_at_k": None,
        "recalled_gold": None,
        "gold_count": None,
        "hits": [],
    }


def run_hybrid_benchmark(
    config: HybridBenchmarkConfig,
    queries: Sequence[EvalQuery] | None = None,
    *,
    dense_retriever: Retriever | None = None,
    sparse_or_retriever: Retriever | None = None,
    sparse_and_retriever: Retriever | None = None,
    environment: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Run all five arms and return raw per-query results plus aggregates.

    Args:
        config: Top-K, candidate pool, RRF k and repetition settings.
        queries: Labelled queries. Defaults to ``data/eval_queries.json``.
        dense_retriever: Dense arm. Defaults to ``DenseRetriever``.
        sparse_or_retriever: Sparse OR arm. Defaults to ``SparseRetriever``.
        sparse_and_retriever: Sparse AND arm. Defaults to a second
            ``SparseRetriever`` configured with ``query_mode="and"``.
        environment: Pre-collected environment facts. Defaults to
            :func:`collect_environment` (which touches PostgreSQL).
        dataset: Pre-collected dataset facts. Defaults to
            :func:`collect_dataset`.
        progress: Optional line-oriented progress callback.

    Returns:
        A JSON-serializable mapping with parameters, definitions, per-query
        arm records and aggregate metrics. Raw hits retain chunk/doc ids,
        ranks, scores and fusion provenance so every number can be recomputed.
    """
    if not isinstance(config, HybridBenchmarkConfig):
        raise BenchmarkError("config must be a HybridBenchmarkConfig")
    selected = list(queries) if queries is not None else load_eval_queries()
    if not selected:
        raise BenchmarkError("at least one eval query is required")

    dense = dense_retriever or _default_dense(config.candidate_k)
    sparse_or = sparse_or_retriever or _default_sparse(config.candidate_k, "or")
    sparse_and = sparse_and_retriever or _default_sparse(config.candidate_k, "and")
    fusion = RRFFusion(k=config.rrf_k, candidate_k=config.candidate_k)
    notify = progress or (lambda _message: None)

    environment_facts = dict(environment) if environment is not None else collect_environment()
    dataset_facts = dict(dataset) if dataset is not None else collect_dataset(selected)

    per_query: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    total_errors = 0
    warmup_errors = 0
    quality_mismatch_samples = 0

    for index, query in enumerate(selected, start=1):
        notify(f"[{index}/{len(selected)}] {query.query_id}")
        references: dict[str, dict[str, Any]] = {}
        arm_error_counts = {arm: 0 for arm in ARM_NAMES}
        query_latency: dict[str, list[float]] = {arm: [] for arm in ARM_NAMES}
        query_mismatches = 0
        query_warmup_errors = 0

        for warmup_number in range(1, config.warmup_passes + 1):
            attempts = _run_core_arms(
                query,
                config,
                dense=dense,
                sparse_or=sparse_or,
                sparse_and=sparse_and,
                fusion=fusion,
            )
            for arm, attempt in attempts.items():
                if attempt.error is not None:
                    warmup_errors += 1
                    query_warmup_errors += 1
                    errors.append(
                        _error_record(
                            attempt.error,
                            stage="warmup",
                            arm=arm,
                            query_id=query.query_id,
                            pass_number=warmup_number,
                        )
                    )

        for pass_number in range(1, config.repetitions + 1):
            attempts = _run_core_arms(
                query,
                config,
                dense=dense,
                sparse_or=sparse_or,
                sparse_and=sparse_and,
                fusion=fusion,
            )
            for arm in ("dense", "sparse_or", "sparse_and"):
                attempt = attempts[arm]
                if attempt.error is not None:
                    total_errors += 1
                    arm_error_counts[arm] += 1
                    errors.append(
                        _error_record(
                            attempt.error,
                            stage="timed",
                            arm=arm,
                            query_id=query.query_id,
                            pass_number=pass_number,
                        )
                    )
                    continue
                assert attempt.hits is not None
                query_latency[arm].append(attempt.elapsed_ms)
                measured = _measure(query, attempt.hits, config.top_k)
                if arm not in references:
                    references[arm] = measured
                elif _signature(measured) != _signature(references[arm]):
                    query_mismatches += 1
                    quality_mismatch_samples += 1

            for arm in ("candidate_union", "rrf"):
                attempt = attempts[arm]
                if attempt.error is not None:
                    total_errors += 1
                    arm_error_counts[arm] += 1
                    errors.append(
                        _error_record(
                            attempt.error,
                            stage="timed",
                            arm=arm,
                            query_id=query.query_id,
                            pass_number=pass_number,
                        )
                    )
                    continue
                assert attempt.hits is not None
                # Derived-arm latency is the end-to-end span of both core calls
                # plus fusion, so it can be compared with a single retriever.
                query_latency[arm].append(
                    attempts["dense"].elapsed_ms
                    + attempts["sparse_or"].elapsed_ms
                    + attempt.elapsed_ms
                )
                measured = _measure(query, attempt.hits, config.top_k)
                if arm not in references:
                    references[arm] = measured
                elif _signature(measured) != _signature(references[arm]):
                    query_mismatches += 1
                    quality_mismatch_samples += 1

        record: dict[str, Any] = {
            "query_id": query.query_id,
            "query": query.query,
            "arms": {
                arm: references.get(arm) or _not_evaluated(arm_error_counts[arm])
                for arm in ARM_NAMES
            },
            "deltas": {},
            "errors": sum(arm_error_counts.values()),
            "warmup_errors": query_warmup_errors,
            "quality_mismatch_samples": query_mismatches,
            "_latency": query_latency,
        }
        record["deltas"] = _deltas(record["arms"])
        per_query.append(record)
        notify(
            f"  {query.query_id}: dense_rr={record['arms']['dense'].get('reciprocal_rank')} "
            f"sparse_rr={record['arms']['sparse_or'].get('reciprocal_rank')} "
            f"rrf_rr={record['arms']['rrf'].get('reciprocal_rank')}"
        )

    aggregates = _aggregate(per_query, config)
    latency_samples = {arm: _latency_for_arm(per_query, arm) for arm in ARM_NAMES}
    for record in per_query:
        record.pop("_latency", None)
    all_evaluated = all(
        record["arms"][arm]["evaluated"] for record in per_query for arm in ARM_NAMES
    )
    if not all_evaluated or total_errors or warmup_errors:
        status = "FAIL"
    elif quality_mismatch_samples:
        status = "PASS WITH MINOR ISSUES"
    else:
        status = "PASS"

    return {
        "schema_version": 1,
        "metric_version": METRIC_VERSION,
        "labels_exhaustive": LABELS_EXHAUSTIVE,
        "relevance_scheme": RELEVANCE_SCHEME,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "parameters": {
            "queries": len(selected),
            "top_k": config.top_k,
            "candidate_k": config.candidate_k,
            "rrf_k": config.rrf_k,
            "repetitions": config.repetitions,
            "warmup_passes": config.warmup_passes,
            "arm_order": list(ARM_NAMES),
            "dense_config": dense_config_facts(dense),
            "sparse_or_query_mode": "or",
            "sparse_and_query_mode": "and",
        },
        "definitions": _definitions(config),
        "environment": environment_facts,
        "dataset": dataset_facts,
        "arms": aggregates,
        "per_query": per_query,
        "focus_misses": _focus_misses(per_query, config.top_k),
        "totals": {
            "evaluated_queries": sum(
                1
                for record in per_query
                if all(record["arms"][arm]["evaluated"] for arm in ARM_NAMES)
            ),
            "missing_arm_queries": sum(
                1
                for record in per_query
                for arm in ARM_NAMES
                if not record["arms"][arm]["evaluated"]
            ),
            "errors": total_errors,
            "warmup_errors": warmup_errors,
            "quality_mismatch_samples": quality_mismatch_samples,
        },
        "latency_samples": latency_samples,
        "errors": errors,
        "limitations": _limitations(),
    }


def _run_core_arms(
    query: EvalQuery,
    config: HybridBenchmarkConfig,
    *,
    dense: Retriever,
    sparse_or: Retriever,
    sparse_and: Retriever,
    fusion: RRFFusion,
) -> dict[str, _Attempt]:
    """Run the three retrievers once, then derive union and RRF from those hits.

    A failure in one core arm does not prevent the other arms from running.
    ``candidate_union`` and ``rrf`` require both dense and sparse OR hits from
    the same pass; otherwise their attempt records that dependency failure.
    """
    attempts: dict[str, _Attempt] = {}
    attempts["dense"] = _attempt(dense, query.query, config.candidate_k)
    attempts["sparse_or"] = _attempt(sparse_or, query.query, config.candidate_k)
    attempts["sparse_and"] = _attempt(sparse_and, query.query, config.candidate_k)

    dense_hits = attempts["dense"].hits
    sparse_hits = attempts["sparse_or"].hits
    if dense_hits is None or sparse_hits is None:
        cause = attempts["dense"].error or attempts["sparse_or"].error
        attempts["candidate_union"] = _Attempt(
            hits=None,
            elapsed_ms=0.0,
            error=BenchmarkError(f"candidate union needs dense and sparse OR hits: {cause}"),
        )
        attempts["rrf"] = _Attempt(
            hits=None,
            elapsed_ms=0.0,
            error=BenchmarkError(f"RRF needs dense and sparse OR hits: {cause}"),
        )
        return attempts

    started = time.perf_counter()
    try:
        union_hits = interleaved_union(
            dense_hits,
            sparse_hits,
            candidate_k=config.candidate_k,
            top_k=config.top_k,
        )
    except Exception as exc:  # noqa: BLE001 -- isolate the derived arm
        attempts["candidate_union"] = _Attempt(
            hits=None,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            error=exc,
        )
    else:
        attempts["candidate_union"] = _Attempt(
            hits=union_hits,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            error=None,
        )

    started = time.perf_counter()
    try:
        fused_hits = fusion.fuse(
            dense_hits,
            sparse_hits,
            top_k=config.top_k,
        )
    except Exception as exc:  # noqa: BLE001 -- isolate the derived arm
        attempts["rrf"] = _Attempt(
            hits=None,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            error=exc,
        )
    else:
        attempts["rrf"] = _Attempt(
            hits=fused_hits,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            error=None,
        )
    return attempts


def _signature(measured: Mapping[str, Any]) -> tuple[Any, ...]:
    """A deterministic signature of one arm's candidate order."""
    return tuple(
        (hit["chunk_strategy"], hit["chunk_id"], hit["doc_id"]) for hit in measured["hits"]
    )


# ----------------------------------------------------------------- aggregates


def _mean(values: Sequence[float]) -> float | None:
    """Mean of a sequence, or ``None`` when empty."""
    return statistics.fmean(values) if values else None


def _aggregate(
    per_query: Sequence[Mapping[str, Any]],
    config: HybridBenchmarkConfig,
) -> dict[str, Any]:
    """Aggregate every arm's quality and latency over all queries."""
    arms: dict[str, Any] = {}
    for arm in ARM_NAMES:
        records = [record["arms"][arm] for record in per_query if record["arms"][arm]["evaluated"]]
        recalls = [float(record["semantic_recall"]) for record in records]
        precisions = [float(record["precision_at_k"]) for record in records]
        ceilings = [float(record["precision_at_k_ceiling"]) for record in records]
        ndcgs = [float(record["ndcg_at_k"]) for record in records]
        reciprocal = [float(record["reciprocal_rank"]) for record in records]
        gold_total = sum(int(record["gold_count"]) for record in records)
        recalled_total = sum(int(record["recalled_gold"]) for record in records)
        arms[arm] = {
            "evaluated_queries": len(records),
            "semantic_recall_macro": _mean(recalls),
            "semantic_recall_micro": (recalled_total / gold_total) if gold_total else None,
            "precision_at_k_macro": _mean(precisions),
            "precision_at_k_micro": (
                recalled_total / (config.top_k * len(records)) if records else None
            ),
            "precision_at_k_ceiling_macro": _mean(ceilings),
            "ndcg_at_k_macro": _mean(ndcgs),
            "recalled_gold": recalled_total,
            "gold_total": gold_total,
            "mrr": _mean(reciprocal),
            "queries_with_gold_hit": sum(1 for value in reciprocal if value > 0),
            "queries_with_no_gold_hit": sum(1 for value in reciprocal if value == 0),
            "hit_at_1": sum(1 for record in records if record["hit_at_1"]),
            "hit_at_1_rate": (
                sum(1 for record in records if record["hit_at_1"]) / len(records)
                if records
                else None
            ),
            "perfect_recall_queries": sum(1 for value in recalls if value >= 1.0),
            "empty_result_queries": sum(1 for record in records if record["empty_result"]),
            "latency_ms": _latency_summary(_latency_for_arm(per_query, arm)),
            "deltas_vs_dense": _aggregate_delta(per_query, arm, "dense"),
            "deltas_vs_sparse_or": _aggregate_delta(per_query, arm, "sparse_or"),
        }

    dense_gold = arms["dense"]["recalled_gold"]
    sparse_gold = arms["sparse_or"]["recalled_gold"]
    return {
        "arms": arms,
        "config": {
            "top_k": config.top_k,
            "candidate_k": config.candidate_k,
            "rrf_k": config.rrf_k,
        },
        "candidate_coverage": _candidate_coverage(per_query),
        "rrf_sources": _rrf_sources(per_query),
        "dense_only_gold": dense_gold,
        "sparse_or_gold": sparse_gold,
        "gold_total": arms["dense"]["gold_total"],
    }


def _latency_for_arm(
    per_query: Sequence[Mapping[str, Any]],
    arm: str,
) -> list[float]:
    """Read one arm's per-query latency samples from the query records."""
    samples: list[float] = []
    for record in per_query:
        samples.extend(float(value) for value in record.get("_latency", {}).get(arm, []))
    return samples


def _latency_summary(samples: Sequence[float]) -> dict[str, float | int | None]:
    """Count, min, nearest-rank P50/P95, max and mean."""
    if not samples:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
    return {
        "n": len(samples),
        "min": min(samples),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "max": max(samples),
        "mean": statistics.fmean(samples),
    }


def _aggregate_delta(
    per_query: Sequence[Mapping[str, Any]],
    arm: str,
    baseline: str,
) -> dict[str, Any]:
    """Count per-query MRR improvements, ties and regressions against a baseline."""
    deltas = [
        record["deltas"][f"{arm}_vs_{baseline}_reciprocal_rank"]
        for record in per_query
        if record["deltas"].get(f"{arm}_vs_{baseline}_reciprocal_rank") is not None
    ]
    return {
        "mrr_delta_mean": _mean([float(value) for value in deltas]),
        "improved": sum(1 for value in deltas if value > 0),
        "equal": sum(1 for value in deltas if value == 0),
        "worse": sum(1 for value in deltas if value < 0),
    }


def _deltas(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, float | None]:
    """Per-query reciprocal-rank deltas for RRF and the union baseline."""
    deltas: dict[str, float | None] = {}
    for arm in ("candidate_union", "rrf"):
        for baseline in ("dense", "sparse_or"):
            arm_value = arms[arm].get("reciprocal_rank")
            base_value = arms[baseline].get("reciprocal_rank")
            key = f"{arm}_vs_{baseline}_reciprocal_rank"
            deltas[key] = (
                float(arm_value) - float(base_value)
                if arm_value is not None and base_value is not None
                else None
            )
    return deltas


def _candidate_coverage(per_query: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Gold coverage and source overlap of the dense, sparse and fused lists."""
    coverage = {
        "queries_with_all_gold_in_dense_candidates": 0,
        "queries_with_all_gold_in_sparse_candidates": 0,
        "queries_with_all_gold_in_union_candidates": 0,
        "queries_with_all_gold_in_rrf_top_k": 0,
        "both_arms": 0,
        "dense_only": 0,
        "sparse_only": 0,
        "neither": 0,
    }
    for record in per_query:
        arms = record["arms"]
        dense = set(arms["dense"]["candidate_doc_ids"])
        sparse = set(arms["sparse_or"]["candidate_doc_ids"])
        union = set(arms["candidate_union"]["candidate_doc_ids"])
        rrf = set(arms["rrf"]["doc_ids"])
        gold = set(arms["dense"]["gold_doc_ids"])
        coverage["queries_with_all_gold_in_dense_candidates"] += int(gold <= dense)
        coverage["queries_with_all_gold_in_sparse_candidates"] += int(gold <= sparse)
        coverage["queries_with_all_gold_in_union_candidates"] += int(gold <= union)
        coverage["queries_with_all_gold_in_rrf_top_k"] += int(gold <= rrf)
        coverage["both_arms"] += len(gold & dense & sparse)
        coverage["dense_only"] += len((gold & dense) - sparse)
        coverage["sparse_only"] += len((gold & sparse) - dense)
        coverage["neither"] += len(gold - dense - sparse)
    coverage["queries"] = len(per_query)
    return coverage


def _rrf_sources(per_query: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count RRF Top-K hits contributed by dense only, sparse only, or both."""
    counts = {"dense_only": 0, "sparse_only": 0, "both": 0}
    for record in per_query:
        for hit in record["arms"]["rrf"]["hits"]:
            sources = hit["candidate_sources"]
            if len(sources) == 2:
                counts["both"] += 1
            elif sources == ["dense"]:
                counts["dense_only"] += 1
            elif sources == ["sparse"]:
                counts["sparse_only"] += 1
    return counts


def _focus_misses(
    per_query: Sequence[Mapping[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Locate the five Dense-missed gold docs in every arm."""
    by_id = {record["query_id"]: record for record in per_query}
    focus: list[dict[str, Any]] = []
    for query_id, doc_id in FOCUS_MISSES:
        record = by_id.get(query_id)
        if record is None:
            continue
        arms = record["arms"]
        row: dict[str, Any] = {
            "query_id": query_id,
            "doc_id": doc_id,
            "top_k": top_k,
            "gold_count": arms["dense"].get("gold_count"),
            "dense_rank": _doc_rank(arms["dense"], doc_id),
            "sparse_rank": _doc_rank(arms["sparse_or"], doc_id),
            "sparse_and_rank": _doc_rank(arms["sparse_and"], doc_id),
            "union_rank": _doc_rank(arms["candidate_union"], doc_id),
            "rrf_rank": _doc_rank(arms["rrf"], doc_id),
            "rrf_score": _doc_rrf_score(arms["rrf"], doc_id),
            "dense_score": _doc_arm_score(arms["dense"], doc_id),
            "sparse_score": _doc_arm_score(arms["sparse_or"], doc_id),
        }
        rank_fields = {
            "dense": "dense_rank",
            "sparse_or": "sparse_rank",
            "sparse_and": "sparse_and_rank",
            "candidate_union": "union_rank",
            "rrf": "rrf_rank",
        }
        for arm, field in rank_fields.items():
            rank = row[field]
            row[f"in_{arm}_top_k"] = rank is not None and rank <= top_k
        focus.append(row)
    return focus


def _doc_rank(arm: Mapping[str, Any], doc_id: str) -> int | None:
    """First document-level rank of ``doc_id`` in an arm record."""
    rank = arm.get("gold_ranks", {}).get(doc_id)
    return int(rank) if rank is not None else None


def _doc_rrf_score(arm: Mapping[str, Any], doc_id: str) -> float | None:
    """The fused RRF score for the first hit carrying ``doc_id``."""
    for hit in arm.get("hits", []):
        if hit["doc_id"] == doc_id:
            return hit["rrf_score"]
    return None


def _doc_arm_score(arm: Mapping[str, Any], doc_id: str) -> float | None:
    """The raw arm score for the first hit carrying ``doc_id``."""
    for hit in arm.get("hits", []):
        if hit["doc_id"] == doc_id:
            return hit["score"]
    return None


# --------------------------------------------------------------- definitions


def _definitions(config: HybridBenchmarkConfig) -> dict[str, str]:
    """Metric and implementation definitions written into the raw artifact."""
    return {
        "candidate_granularity": "chunk identity (chunk_strategy, chunk_id)",
        "metric_granularity": ("document; distinct doc_id in ranked order, truncated to top_k"),
        "semantic_recall": (
            "recall_at_k against human relevant_doc_ids; denominator is the number of "
            "distinct gold docs, not K; not ANN recall"
        ),
        "mrr": (
            "mean reciprocal rank of the first gold doc at doc rank r <= top_k; "
            "no gold in Top-K contributes 0; denominator is evaluated queries"
        ),
        "rrf": (
            f"sum over arms of 1 / (k + rank), rank starts at 1, "
            f"k={config.rrf_k}; raw dense/sparse scores are never added"
        ),
        "candidate_union": (
            "interleaved dense[0], sparse[0], dense[1], sparse[1], ... after removing "
            "repeated chunk identities; score is a 0.0 placeholder because this baseline "
            "has no fused score"
        ),
        "candidate_pool": f"first {config.candidate_k} hits from each retrieval arm",
        "final_top_k": f"first {config.top_k} ranked documents/hits",
        "no_gold_handling": "reciprocal rank 0.0 and no fabricated gold document",
        "precision_at_k": (
            "|distinct doc_id in top_k ∩ known gold| / k; the denominator is always k, "
            "even when fewer than k documents are returned; duplicate doc_ids are "
            "collapsed in rank order before the cutoff; supplementary because labels "
            "are not exhaustive"
        ),
        "precision_at_k_ceiling": (
            "min(distinct gold doc count, k) / k per query; the maximum Precision@K "
            "reachable on that query, not an observed ranking score"
        ),
        "ndcg_at_k": (
            "binary nDCG: gain 1 for a known gold doc_id and 0 otherwise; DCG uses "
            "1 / log2(rank + 1); IDCG uses the same k cutoff over the full known gold "
            "set; non-gold ids are unlabelled, not asserted irrelevant"
        ),
        "labels_exhaustive": (
            f"{LABELS_EXHAUSTIVE}; non-gold ids have not been judged irrelevant"
        ),
        "relevance_scheme": (
            f"{RELEVANCE_SCHEME}; only existing gold ids receive relevance 1"
        ),
        "metric_version": f"{METRIC_VERSION}",
        "latency": (
            "wall-clock milliseconds per retrieval call, including query embedding for "
            "dense; one warm-up pass precedes timed passes"
        ),
    }


def _limitations() -> list[str]:
    """Known limits that the numbers in this artifact do not establish."""
    return [
        "The corpus has one chunk per document, so chunk-level and document-level "
        "candidate sets coincide here.",
        "Sparse OR already retrieves all 43 gold references (38 distinct documents), "
        "so hybrid recall cannot exceed 1.0 on this dataset.",
        "The 30 queries are English; Chinese and mixed-language hybrid retrieval are "
        "not evaluated.",
        "Precision@K and binary nDCG@K are supplementary and label-limited: non-gold "
        "ids are unlabelled, not confirmed irrelevant.",
        "Host execution may differ from the pinned container environment; see "
        "environment.dependency_versions.",
        "This run measures quality and latency at 150 chunks, not scale or concurrency behaviour.",
    ]


# ------------------------------------------------------------------ environment


def _package_version(name: str) -> str:
    """Return an installed distribution version or ``missing``."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def _sha256(path: Path) -> str:
    """Return the SHA256 of a file, or ``missing``."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "missing"


def collect_environment() -> dict[str, Any]:
    """Collect read-only runtime and database facts for the artifact."""
    settings = get_settings()
    from src.db import connection, scalar

    with connection() as conn:
        address = conn.execute(
            "SELECT coalesce(inet_server_addr()::text, 'unix-socket'), inet_server_port()"
        ).fetchone()
        documents = int(scalar(conn, "SELECT count(*) FROM documents") or 0)
        chunks = int(scalar(conn, "SELECT count(*) FROM chunks") or 0)
        distinct_docs = int(scalar(conn, "SELECT count(DISTINCT doc_id) FROM chunks") or 0)
        vec_version = scalar(conn, "SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        server_version = scalar(conn, "SHOW server_version")
        hnsw = scalar(
            conn,
            """
            SELECT pg_get_indexdef(indexrelid)
            FROM pg_index
            JOIN pg_class ON pg_class.oid = pg_index.indexrelid
            WHERE pg_index.indrelid = 'chunks'::regclass
              AND pg_class.relname = 'chunks_embedding_hnsw_idx'
            """,
        )
        gin = scalar(
            conn,
            """
            SELECT pg_get_indexdef(indexrelid)
            FROM pg_index
            JOIN pg_class ON pg_class.oid = pg_index.indexrelid
            WHERE pg_index.indrelid = 'chunks'::regclass
              AND pg_class.relname = 'chunks_tsv_gin'
            """,
        )
        ef_search = scalar(conn, "SHOW hnsw.ef_search")
        dimensions = sorted(
            int(row[0])
            for row in conn.execute(
                "SELECT DISTINCT vector_dims(embedding) FROM chunks WHERE embedding IS NOT NULL"
            ).fetchall()
        )
    return {
        "execution_environment": (
            "docker container" if Path("/.dockerenv").exists() else "host Python process"
        ),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "postgres_version": str(server_version),
        "vector_version": str(vec_version) if vec_version else "missing",
        "server_address": f"{address[0]}:{address[1]}" if address else "unknown",
        "documents": documents,
        "chunks": chunks,
        "distinct_chunk_docs": distinct_docs,
        "embedding_dimensions": dimensions,
        "embedding_model": settings.embedding_model,
        "query_instruction": settings.query_instruction,
        "hnsw_index": str(hnsw) if hnsw else "missing",
        "gin_index": str(gin) if gin else "missing",
        "ef_search_setting": str(ef_search) if ef_search else "unset (pgvector default 40)",
        "dependency_versions": {
            name: _package_version(name)
            for name in (
                "psycopg",
                "psycopg-pool",
                "pgvector",
                "sentence-transformers",
                "torch",
                "transformers",
                "numpy",
                "pydantic",
                "pytest",
            )
        },
    }


def collect_dataset(
    queries: Sequence[EvalQuery],
    *,
    data_path: Path | None = None,
    query_path: Path | None = None,
) -> dict[str, Any]:
    """Collect file hashes, query counts and gold-document presence."""
    settings = get_settings()
    data_dir = settings.resolve(settings.data_dir)
    kb_path = data_path or data_dir / "kb_documents.json"
    eval_path = query_path or data_dir / "eval_queries.json"
    gold = sorted({doc_id for query in queries for doc_id in query.relevant_doc_ids})

    missing: list[str] = []
    if gold:
        from src.db import connection

        with connection() as conn:
            rows = conn.execute(
                "SELECT doc_id FROM documents WHERE doc_id = ANY(%s)",
                (gold,),
            ).fetchall()
        found = {str(row[0]) for row in rows}
        missing = [doc_id for doc_id in gold if doc_id not in found]

    return {
        "queries": len(queries),
        "gold_sizes": sorted({len(query.relevant_doc_ids) for query in queries}),
        "distinct_gold_docs": len(gold),
        "missing_gold_docs": missing,
        "hashes": {
            "kb_documents.json": _sha256(kb_path),
            "eval_queries.json": _sha256(eval_path),
        },
    }


def dense_config_facts(retriever: Retriever) -> dict[str, Any]:
    """Record the dense retriever's effective parameters without assuming a type."""
    config = getattr(retriever, "config", None)
    if not isinstance(config, DenseConfig):
        return {"top_n": None, "ef_search": None, "metric": None}
    return {
        "top_n": config.top_n,
        "ef_search": config.ef_search,
        "m": config.m,
        "ef_construction": config.ef_construction,
        "metric": config.metric,
    }


# --------------------------------------------------------------------- output


def write_outputs(
    result: Mapping[str, Any],
    json_path: Path | None = None,
    markdown_path: Path | None = None,
) -> tuple[Path, Path]:
    """Write the raw JSON artifact and the rendered Markdown report."""
    json_path = json_path or DEFAULT_OUTPUT
    markdown_path = markdown_path or DEFAULT_REPORT
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render the hybrid benchmark artifact as a reviewable Markdown report."""
    parameters = result.get("parameters", {})
    environment = result.get("environment", {})
    dataset = result.get("dataset", {})
    arms = result.get("arms", {}).get("arms", {})
    lines = [
        "# Hybrid Search / RRF 离线评测（阶段 ⑥-B）",
        "",
        f"- Status: **{result.get('status', 'unknown')}**",
        f"- Generated: {result.get('generated_at', '-')}",
        f"- Top-K: {parameters.get('top_k')}",
        f"- Candidate pool per arm: {parameters.get('candidate_k')}",
        f"- RRF k: {parameters.get('rrf_k')}",
        f"- Queries: {parameters.get('queries')}",
        f"- Repetitions / warm-ups: {parameters.get('repetitions')} / "
        f"{parameters.get('warmup_passes')}",
        "",
        "## 评测口径",
        "",
    ]
    for key, value in result.get("definitions", {}).items():
        lines.append(f"- **{key}**: {value}")

    lines.extend(
        [
            "",
            "## 五臂对比",
            "",
            _md_table(
                [
                    "arm",
                    "Recall macro",
                    "Recall micro",
                    "MRR",
                    "Hit@1",
                    "有 gold 查询",
                    "无 gold 查询",
                    "空结果",
                    "P50 ms",
                    "P95 ms",
                ],
                [
                    [
                        arm,
                        _fmt(record.get("semantic_recall_macro")),
                        _fmt(record.get("semantic_recall_micro")),
                        _fmt(record.get("mrr")),
                        f"{record.get('hit_at_1')}/{record.get('evaluated_queries')}",
                        record.get("queries_with_gold_hit"),
                        record.get("queries_with_no_gold_hit"),
                        record.get("empty_result_queries"),
                        _fmt(record.get("latency_ms", {}).get("p50")),
                        _fmt(record.get("latency_ms", {}).get("p95")),
                    ]
                    for arm in ARM_NAMES
                    if (record := arms.get(arm))
                ],
            ),
            "",
            "### 补充指标：Precision@K / 二元 nDCG@K",
            "",
            f"- labels_exhaustive={result.get('labels_exhaustive')}; "
            f"relevance_scheme={result.get('relevance_scheme')}; "
            f"metric_version={result.get('metric_version')}",
            "- 非 gold 文档只是未标注，不是已确认无关；P@K ceiling 是该 Query 在当前 "
            "gold 数量与 K 下的理论上限。",
            "",
            _md_table(
                [
                    "arm",
                    "P@K macro",
                    "P@K micro",
                    "P@K ceiling macro",
                    "nDCG@K macro",
                ],
                [
                    [
                        arm,
                        _fmt(record.get("precision_at_k_macro")),
                        _fmt(record.get("precision_at_k_micro")),
                        _fmt(record.get("precision_at_k_ceiling_macro")),
                        _fmt(record.get("ndcg_at_k_macro")),
                    ]
                    for arm in ARM_NAMES
                    if (record := arms.get(arm))
                ],
            ),
            "",
            "## 逐查询 MRR",
            "",
            _md_table(
                [
                    "query",
                    "gold",
                    "dense RR",
                    "sparse OR RR",
                    "sparse AND RR",
                    "union RR",
                    "RRF RR",
                    "RRF-dense",
                    "RRF-sparse",
                ],
                [
                    [
                        record["query_id"],
                        ", ".join(record["arms"]["dense"].get("gold_doc_ids", [])),
                        _fmt(record["arms"]["dense"].get("reciprocal_rank")),
                        _fmt(record["arms"]["sparse_or"].get("reciprocal_rank")),
                        _fmt(record["arms"]["sparse_and"].get("reciprocal_rank")),
                        _fmt(record["arms"]["candidate_union"].get("reciprocal_rank")),
                        _fmt(record["arms"]["rrf"].get("reciprocal_rank")),
                        _fmt(record["deltas"].get("rrf_vs_dense_reciprocal_rank")),
                        _fmt(record["deltas"].get("rrf_vs_sparse_or_reciprocal_rank")),
                    ]
                    for record in result.get("per_query", [])
                ],
            ),
            "",
            "## 五个 Dense 漏检 gold 的融合排名",
            "",
            _md_table(
                [
                    "query",
                    "doc",
                    "dense rank",
                    "sparse rank",
                    "union rank",
                    "RRF rank",
                    "RRF score",
                    "in RRF Top-K",
                ],
                [
                    [
                        row["query_id"],
                        row["doc_id"],
                        _fmt(row.get("dense_rank")),
                        _fmt(row.get("sparse_rank")),
                        _fmt(row.get("union_rank")),
                        _fmt(row.get("rrf_rank")),
                        _fmt(row.get("rrf_score")),
                        row.get("in_rrf_top_k"),
                    ]
                    for row in result.get("focus_misses", [])
                ],
            ),
            "",
            "## 候选覆盖",
            "",
        ]
    )
    coverage = result.get("arms", {}).get("candidate_coverage", {})
    lines.append(_md_table(["metric", "value"], [[key, value] for key, value in coverage.items()]))
    sources = result.get("arms", {}).get("rrf_sources", {})
    lines.extend(
        [
            "",
            "RRF Top-K 来源：" + ", ".join(f"{key}={value}" for key, value in sources.items()),
            "",
            "## 环境与数据",
            "",
        ]
    )
    for key in (
        "execution_environment",
        "python_version",
        "postgres_version",
        "vector_version",
        "server_address",
        "documents",
        "chunks",
        "distinct_chunk_docs",
        "embedding_model",
        "embedding_dimensions",
        "ef_search_setting",
        "hnsw_index",
        "gin_index",
    ):
        lines.append(f"- {key}: {environment.get(key)}")
    lines.append(f"- dependency_versions: {environment.get('dependency_versions')}")
    lines.append(f"- dataset: {dataset}")
    lines.extend(["", "## 限制", ""])
    lines.extend(f"- {item}" for item in result.get("limitations", []))
    totals = result.get("totals", {})
    lines.extend(
        [
            "",
            "## 执行事实",
            "",
            f"- evaluated_queries: {totals.get('evaluated_queries')}",
            f"- missing_arm_queries: {totals.get('missing_arm_queries')}",
            f"- errors: {totals.get('errors')}",
            f"- warmup_errors: {totals.get('warmup_errors')}",
            f"- quality_mismatch_samples: {totals.get('quality_mismatch_samples')}",
            "",
            f"原始逐查询与候选结果：[`{DEFAULT_OUTPUT.name}`]({DEFAULT_OUTPUT.name})",
            "",
        ]
    )
    return "\n".join(lines)


def _md_table(headers: Sequence[object], rows: Sequence[Sequence[object]]) -> str:
    """Render a GitHub-flavoured Markdown table, escaping separators."""

    def cell(value: object) -> str:
        """Escape one cell."""
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _fmt(value: object, digits: int = 3) -> str:
    """Format a number without inventing a value for ``None``."""
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)
