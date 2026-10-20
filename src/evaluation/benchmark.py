"""Unified benchmark orchestration.

Runs a named set of retrieval arms over the same queries and records one
schema for every arm, so the five historical benchmark modules keep their
specialised evidence while the cross-arm comparison shares a single contract.

Boundaries:

* this module performs **no retrieval and no model loading** -- arms are
  injected as :class:`Arm` objects built from :mod:`src.retrieval.registry`;
* it computes metrics only with the shared functions in
  :mod:`src.evaluation.metrics`, so its numbers cannot drift from the
  specialised reports;
* it does not decide *which* arms are interesting. The caller (CLI, a script or
  a test) does that, which is what makes the orchestrator unit-testable with
  stub retrievers.

Document-level scoring: hits are chunk-level, so a chunk list is collapsed to
the order of first document appearance before Recall/P@K/nDCG/MRR are computed.
With the ``document`` strategy one chunk is one document; with ``fixed_window``
or ``structured`` several chunks per document are collapsed to one effective
rank.

Status vocabulary:

* ``PASS`` -- every expected query ran on every arm and produced a valid record;
* ``PARTIAL`` -- at least one arm ran, but some query or arm is missing;
* ``FAIL`` -- no arm produced a single valid record.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.evaluation.artifacts import ensure_writable
from src.evaluation.dense_benchmark import EvalQuery
from src.evaluation.metrics import (
    LABELS_EXHAUSTIVE,
    METRIC_VERSION,
    RELEVANCE_SCHEME,
    dedupe_in_order,
    hit_at_k,
    latency_summary,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from src.retrieval.base import Retriever

BENCHMARK_VERSION = 1
"""Schema version of the unified artifact."""

EXPECTED_QUERIES = 30
"""The labelled evaluation set size; a smaller run is explicitly PARTIAL."""


@dataclass(frozen=True)
class BenchmarkConfig:
    """Parameters shared by every arm in one run."""

    top_k: int = 10
    candidate_k: int = 10
    rrf_k: int = 60
    chunk_strategy: str = "document"
    expected_queries: int = EXPECTED_QUERIES

    def __post_init__(self) -> None:
        """Reject impossible sizes before any arm runs."""
        for name in ("top_k", "candidate_k", "rrf_k", "expected_queries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.top_k > self.candidate_k:
            raise ValueError("top_k must not exceed candidate_k")
        if self.chunk_strategy not in ("document", "fixed_window", "structured"):
            raise ValueError(f"unsupported chunk_strategy {self.chunk_strategy!r}")


@dataclass(frozen=True)
class Arm:
    """One retriever plus the metadata the report must record about it."""

    name: str
    retriever: Retriever
    description: str = ""
    parameters: Mapping[str, Any] | None = None


def evaluate_arm(
    arm: Arm,
    queries: Sequence[EvalQuery],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Run ``arm`` over every query and score it at document level.

    A query that raises is recorded with its error and excluded from the
    metric aggregates; it is never replaced with an empty result, because an
    empty list and a failed call are different outcomes.
    """
    records: list[dict[str, Any]] = []
    for query in queries:
        started = time.perf_counter()
        try:
            hits = list(arm.retriever.retrieve(query.query, top_k=top_k))
        except Exception as exc:  # noqa: BLE001 -- failures are data here
            records.append(
                {
                    "query_id": query.query_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 4),
                }
            )
            continue
        latency_ms = round((time.perf_counter() - started) * 1000.0, 4)
        doc_ids = dedupe_in_order([hit.doc_id for hit in hits])
        gold = list(query.relevant_doc_ids)
        records.append(
            {
                "query_id": query.query_id,
                "chunk_keys": [
                    f"{hit.chunk_strategy or 'unknown'}:{hit.chunk_id}" for hit in hits
                ],
                "doc_ids": doc_ids,
                "result_count": len(hits),
                "unique_docs": len(doc_ids),
                "error": None,
                "latency_ms": latency_ms,
                **_query_scores(doc_ids, gold, top_k),
            }
        )

    metrics = _aggregate_metrics(records, top_k)
    latencies = [record["latency_ms"] for record in records if record.get("error") is None]
    failed = [record["query_id"] for record in records if record.get("error") is not None]
    return {
        "name": arm.name,
        "description": arm.description,
        "parameters": dict(arm.parameters or {}),
        "status": (
            "PASS"
            if queries and not failed and len(records) == len(queries)
            else "PARTIAL"
        ),
        "metrics": metrics,
        "latency_ms": {"query": latency_summary(latencies)},
        "failed_queries": failed,
        "per_query": records,
    }


def run_benchmark(
    arms: Sequence[Arm],
    queries: Sequence[EvalQuery],
    config: BenchmarkConfig | None = None,
    *,
    environment: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Run every arm over the same query set and return one unified artifact."""
    resolved = config if config is not None else BenchmarkConfig()
    generated = generated_at if generated_at is not None else datetime.now(UTC)
    arm_results = [evaluate_arm(arm, queries, top_k=resolved.top_k) for arm in arms]

    evaluated = sum(record["metrics"]["evaluated_queries"] for record in arm_results)
    failed = sum(record["metrics"]["failed_queries"] for record in arm_results)
    complete = len(queries) == resolved.expected_queries
    if arm_results and failed == 0 and complete:
        status, reason = "PASS", "all arms evaluated every expected query"
    elif evaluated > 0:
        status, reason = "PARTIAL", _partial_reason(queries, resolved, arm_results)
    else:
        status, reason = "FAIL", "no arm produced a valid record"

    return {
        "schema_version": BENCHMARK_VERSION,
        "experiment_name": "unified_retrieval_benchmark",
        "generated_at": generated.isoformat(),
        "status": status,
        "status_reason": reason,
        "parameters": {
            "top_k": resolved.top_k,
            "candidate_k": resolved.candidate_k,
            "rrf_k": resolved.rrf_k,
            "chunk_strategy": resolved.chunk_strategy,
            "expected_queries": resolved.expected_queries,
            "queries": len(queries),
            "arms": [arm.name for arm in arms],
        },
        "definitions": {
            "labels_exhaustive": LABELS_EXHAUSTIVE,
            "relevance_scheme": RELEVANCE_SCHEME,
            "metric_version": METRIC_VERSION,
            "recall": "set-based; denominator = distinct gold docs; document-level dedupe",
            "precision": "known-gold hits / k; non-gold ids are unlabelled, not negatives",
            "ndcg": "binary relevance; ideal DCG capped by k",
            "mrr": "1 / rank of the first relevant doc within k; 0.0 on a miss",
            "latency": "wall-clock per query, single process, sequential",
        },
        "environment": dict(environment or {}),
        "dataset": dict(dataset or {}),
        "arms": arm_results,
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render the unified artifact as a reviewable Markdown report."""
    parameters = result["parameters"]
    lines = [
        "# 统一检索评测",
        "",
        f"- Status: **{result['status']}**",
        f"- Reason: {result['status_reason']}",
        f"- Generated: {result['generated_at']}",
        f"- Queries: {parameters['queries']} / expected {parameters['expected_queries']}",
        f"- Top-K: {parameters['top_k']}；candidate_k: {parameters['candidate_k']}；"
        f"RRF k: {parameters['rrf_k']}",
        f"- Chunk strategy: `{parameters['chunk_strategy']}`",
        "",
        "> Recall/P@K/nDCG/MRR 都按文档级计算（chunk 列表按 doc_id 去重）。",
        "> Gold 不是穷尽标注：非 gold 位置只是未标注，不是 true negative。",
        "",
        "## 1. Arm 指标",
        "",
        "| Arm | Status | Recall macro | Recall micro | P@K macro | P@K ceiling | "
        "nDCG@K | MRR | Hit@1 | Evaluated |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in result["arms"]:
        metrics = arm["metrics"]
        lines.append(
            "| {name} | {status} | {recall} | {recall_micro} | {p} | {ceiling} | {ndcg} | "
            "{mrr} | {hit1}/{evaluated} | {evaluated} |".format(
                name=arm["name"],
                status=arm["status"],
                recall=_fmt(metrics["recall_at_k"]["macro"]),
                recall_micro=_fmt(metrics["recall_at_k"]["micro"]),
                p=_fmt(metrics["precision_at_k"]["macro"]),
                ceiling=_fmt(metrics["precision_at_k"]["ceiling_macro"]),
                ndcg=_fmt(metrics["ndcg_at_k"]["macro"]),
                mrr=_fmt(metrics["mrr"]),
                hit1=metrics["hit_at_1"],
                evaluated=metrics["evaluated_queries"],
            )
        )

    lines.extend(
        [
            "",
            "## 2. 延迟",
            "",
            "| Arm | n | P50 ms | P95 ms | P99 ms | Max ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in result["arms"]:
        latency = arm["latency_ms"]["query"]
        lines.append(
            f"| {arm['name']} | {latency['n']} | {_fmt(latency['p50'])} | "
            f"{_fmt(latency['p95'])} | {_fmt(latency['p99'])} | {_fmt(latency['max'])} |"
        )

    failures = [
        (arm["name"], record["query_id"], record["error"])
        for arm in result["arms"]
        for record in arm["per_query"]
        if record.get("error")
    ]
    lines.extend(["", "## 3. 失败与限制", ""])
    if failures:
        lines.extend(f"- `{name}` {query_id}: {error}" for name, query_id, error in failures)
    else:
        lines.append("- 没有失败的 Query。")
    lines.extend(
        [
            "",
            "- 指标只衡量检索，不代表答案正确性或生成质量。",
            "- `Precision@K` 的分母固定为 K，且 gold 只有 1–3 篇；绝对值会偏低，"
            "应同时看 ceiling 并只做横向比较。",
            "- 延迟是单进程、顺序、已预热的运行；不含并发、GPU 或跨机器。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    result: Mapping[str, Any],
    json_path: Path,
    report_path: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write the unified JSON and Markdown artifacts, refusing to clobber."""
    ensure_writable((json_path, report_path), overwrite=overwrite)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, report_path


def _query_scores(
    doc_ids: Sequence[str], gold: Sequence[str], top_k: int
) -> dict[str, Any]:
    """Return the per-query metric block for one document ranking."""
    return {
        "gold_size": len(dict.fromkeys(gold)),
        "gold_retrieved": len(set(doc_ids[:top_k]).intersection(gold)),
        "recall": recall_at_k(doc_ids, gold, top_k),
        "precision": precision_at_k(doc_ids, gold, top_k),
        "ndcg": ndcg_at_k(doc_ids, gold, top_k),
        "reciprocal_rank": reciprocal_rank_at_k(doc_ids, gold, top_k),
        "hit_at_1": 1 if hit_at_k(doc_ids, gold, 1) else 0,
        "hit_at_k": 1 if hit_at_k(doc_ids, gold, top_k) else 0,
    }


def _aggregate_metrics(records: Sequence[Mapping[str, Any]], top_k: int) -> dict[str, Any]:
    """Aggregate the per-query blocks; failed queries are not zeros."""
    valid = [record for record in records if record.get("error") is None]
    recalls = [float(record["recall"]) for record in valid]
    precisions = [float(record["precision"]) for record in valid]
    ndcgs = [float(record["ndcg"]) for record in valid]
    rrs = [float(record["reciprocal_rank"]) for record in valid]
    ceilings = [min(int(record["gold_size"]), top_k) / top_k for record in valid]
    gold_total = sum(int(record["gold_size"]) for record in valid)
    gold_retrieved = sum(int(record["gold_retrieved"]) for record in valid)
    return {
        "recall_at_k": {
            "macro": _mean(recalls),
            "micro": (gold_retrieved / gold_total) if gold_total else None,
        },
        "precision_at_k": {
            "macro": _mean(precisions),
            "micro": (gold_retrieved / (len(valid) * top_k)) if valid else None,
            "ceiling_macro": _mean(ceilings),
        },
        "ndcg_at_k": {"macro": _mean(ndcgs)},
        "mrr": _mean(rrs),
        "hit_at_1": sum(int(record["hit_at_1"]) for record in valid),
        "hit_at_k": sum(int(record["hit_at_k"]) for record in valid),
        "evaluated_queries": len(valid),
        "failed_queries": len(records) - len(valid),
    }


def _mean(values: Sequence[float]) -> float | None:
    """Return the arithmetic mean, or ``None`` when there is no sample."""
    return statistics.fmean(values) if values else None


def _partial_reason(
    queries: Sequence[EvalQuery],
    config: BenchmarkConfig,
    arms: Sequence[Mapping[str, Any]],
) -> str:
    """Explain exactly why a run is not PASS."""
    reasons: list[str] = []
    if len(queries) != config.expected_queries:
        reasons.append(
            f"{len(queries)} of {config.expected_queries} expected queries were provided"
        )
    for arm in arms:
        failed = arm["metrics"]["failed_queries"]
        if failed:
            reasons.append(f"{arm['name']} failed {failed} query(s)")
    return "; ".join(reasons) or "one or more arms produced no records"


def _fmt(value: float | int | None) -> str:
    """Format a metric for Markdown, keeping ``None`` visible as ``-``."""
    return "-" if value is None else f"{float(value):.4f}"
