"""Chunking-strategy comparison.

Runs the same retrieval arms over the same labelled queries for each chunking
strategy and reports document-level metrics side by side. The heavy lifting
stays in :func:`src.evaluation.benchmark.run_benchmark`; this module only
orchestrates the per-strategy runs, merges their statuses and adds the chunk
statistics a reader needs to interpret the numbers.

Why document-level: gold labels are doc ids, and a strategy may emit several
chunks per document. The unified runner already collapses chunks to first-doc
appearance before scoring, so ``document`` (1 chunk/doc) and the finer
strategies are compared on the same unit.

Boundaries:

* arms are injected -- this module loads no model and opens no connection;
* chunk statistics are injected by the caller (the script reads them from
  PostgreSQL) and recorded verbatim;
* a strategy that fails is not silently dropped: it makes the whole artifact
  ``PARTIAL`` and its exact failure is listed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.evaluation.artifacts import ensure_writable
from src.evaluation.benchmark import (
    EXPECTED_QUERIES,
    Arm,
    BenchmarkConfig,
    run_benchmark,
)
from src.evaluation.dense_benchmark import EvalQuery

CHUNKING_BENCHMARK_VERSION = 1
"""Schema version of the chunking artifact."""

STRATEGIES: tuple[str, ...] = ("document", "fixed_window", "structured")
"""The implemented chunking strategies, in report order."""


@dataclass(frozen=True)
class ChunkingConfig:
    """Sizes shared by every strategy run."""

    top_k: int = 10
    candidate_k: int = 10
    rrf_k: int = 60
    expected_queries: int = EXPECTED_QUERIES
    strategies: tuple[str, ...] = STRATEGIES

    def __post_init__(self) -> None:
        """Validate sizes and the strategy list before any arm runs."""
        for name in ("top_k", "candidate_k", "rrf_k", "expected_queries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.top_k > self.candidate_k:
            raise ValueError("top_k must not exceed candidate_k")
        if not self.strategies:
            raise ValueError("at least one chunking strategy is required")
        unknown = [name for name in self.strategies if name not in STRATEGIES]
        if unknown:
            raise ValueError(
                f"unknown chunking strategies: {', '.join(unknown)}; "
                f"expected a subset of {', '.join(STRATEGIES)}"
            )
        if len(set(self.strategies)) != len(self.strategies):
            raise ValueError("chunking strategies must not repeat")


def run_chunking_benchmark(
    runs: Mapping[str, Sequence[Arm]],
    queries: Sequence[EvalQuery],
    config: ChunkingConfig | None = None,
    *,
    chunk_stats: Mapping[str, Mapping[str, Any]] | None = None,
    environment: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Run every strategy's arms and return one comparison artifact.

    Args:
        runs: Arms keyed by chunking strategy. Each value is typically built by
            :mod:`src.retrieval.registry` with the matching ``chunk_strategy``.
        queries: The same labelled queries for every strategy.
        config: Shared sizes and the strategy set.
        chunk_stats: Per-strategy statistics read from PostgreSQL.
        environment: Provenance recorded verbatim.
        dataset: Provenance recorded verbatim.
        generated_at: Override for tests.

    Returns:
        The comparison artifact with each strategy's full unified run nested.

    Raises:
        ValueError: the supplied strategies do not match ``config.strategies``,
            or a strategy has no arms.
    """
    resolved = config if config is not None else ChunkingConfig()
    _validate_runs(runs, resolved)
    generated = generated_at if generated_at is not None else datetime.now(UTC)

    strategy_runs: dict[str, dict[str, Any]] = {}
    for strategy in resolved.strategies:
        strategy_config = BenchmarkConfig(
            top_k=resolved.top_k,
            candidate_k=resolved.candidate_k,
            rrf_k=resolved.rrf_k,
            chunk_strategy=strategy,
            expected_queries=resolved.expected_queries,
        )
        strategy_runs[strategy] = run_benchmark(
            runs[strategy],
            queries,
            strategy_config,
            environment=environment,
            dataset=dataset,
            generated_at=generated,
        )

    statuses = {name: run["status"] for name, run in strategy_runs.items()}
    evaluated = sum(
        arm["metrics"]["evaluated_queries"]
        for run in strategy_runs.values()
        for arm in run["arms"]
    )
    if statuses and all(status == "PASS" for status in statuses.values()):
        status, reason = "PASS", "every strategy evaluated every expected query"
    elif evaluated > 0:
        status = "PARTIAL"
        reason = "; ".join(
            f"{name}: {run['status_reason']}" for name, run in strategy_runs.items()
        )
    else:
        status, reason = "FAIL", "no strategy produced a valid record"

    return {
        "schema_version": CHUNKING_BENCHMARK_VERSION,
        "experiment_name": "chunking_comparison",
        "generated_at": generated.isoformat(),
        "status": status,
        "status_reason": reason,
        "parameters": {
            "top_k": resolved.top_k,
            "candidate_k": resolved.candidate_k,
            "rrf_k": resolved.rrf_k,
            "expected_queries": resolved.expected_queries,
            "queries": len(queries),
            "strategies": list(resolved.strategies),
            "arms": {name: [arm.name for arm in arms] for name, arms in runs.items()},
        },
        "environment": dict(environment or {}),
        "dataset": dict(dataset or {}),
        "chunk_stats": {
            name: dict(stats) for name, stats in (chunk_stats or {}).items()
        },
        "summary": _summary(strategy_runs),
        "strategies": strategy_runs,
        "limitations": _limitations(resolved),
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render the comparison as a reviewable Markdown report."""
    parameters = result["parameters"]
    lines = [
        "# 分块策略对比",
        "",
        f"- Status: **{result['status']}**",
        f"- Reason: {result['status_reason']}",
        f"- Generated: {result['generated_at']}",
        f"- Queries: {parameters['queries']} / expected {parameters['expected_queries']}"
        f"；Top-K: {parameters['top_k']}；candidate_k: {parameters['candidate_k']}",
        f"- Strategies: {', '.join(f'`{name}`' for name in parameters['strategies'])}",
        "",
        "> 指标按文档级计算：同一文档的多个 chunk 先归并为一个排名，再计算 Recall/MRR/nDCG。",
        "> Gold 不是穷尽标注；非 gold 位置只是未标注，不是 true negative。",
        "",
        "## 1. 分块统计",
        "",
        "| Strategy | Chunks | Documents | Avg chars | Min chars | Max chars |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy in parameters["strategies"]:
        stats = result.get("chunk_stats", {}).get(strategy, {})
        lines.append(
            f"| {strategy} | {_fmt_int(stats.get('chunk_count'))} | "
            f"{_fmt_int(stats.get('documents'))} | {_fmt(stats.get('avg_chars'), 1)} | "
            f"{_fmt_int(stats.get('min_chars'))} | {_fmt_int(stats.get('max_chars'))} |"
        )

    lines.extend(
        [
            "",
            "## 2. 检索指标",
            "",
            "| Strategy | Arm | Status | Recall macro | Recall micro | P@K | "
            "P@K ceiling | nDCG@K | MRR | Hit@1 | Evaluated |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["summary"]:
        lines.append(
            f"| {row['strategy']} | {row['arm']} | {row['status']} | "
            f"{_fmt(row['recall_macro'])} | {_fmt(row['recall_micro'])} | "
            f"{_fmt(row['precision_macro'])} | {_fmt(row['precision_ceiling'])} | "
            f"{_fmt(row['ndcg_macro'])} | "
            f"{_fmt(row['mrr'])} | {row['hit_at_1']}/{row['evaluated']} | "
            f"{row['evaluated']} |"
        )

    lines.extend(
        [
            "",
            "## 3. 延迟",
            "",
            "| Strategy | Arm | n | P50 ms | P95 ms | P99 ms | Max ms |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["summary"]:
        latency = row["latency_ms"]
        lines.append(
            f"| {row['strategy']} | {row['arm']} | {latency['n']} | "
            f"{_fmt(latency['p50'])} | {_fmt(latency['p95'])} | "
            f"{_fmt(latency['p99'])} | {_fmt(latency['max'])} |"
        )

    lines.extend(["", "## 4. 限制", ""])
    lines.extend(f"- {item}" for item in result["limitations"])
    failures = [
        (row["strategy"], row["arm"], row["failed"])
        for row in result["summary"]
        if row["failed"]
    ]
    if failures:
        lines.extend(["", "## 5. 失败的 Query", ""])
        lines.extend(
            f"- `{strategy}` / `{arm}`: {failed} failed query(s)"
            for strategy, arm, failed in failures
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    result: Mapping[str, Any],
    json_path: Path,
    report_path: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write the comparison JSON and Markdown artifacts, refusing to clobber."""
    ensure_writable((json_path, report_path), overwrite=overwrite)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, report_path


def _validate_runs(
    runs: Mapping[str, Sequence[Arm]], config: ChunkingConfig
) -> None:
    """Require exactly the configured strategies, each with at least one arm."""
    missing = [name for name in config.strategies if name not in runs]
    extra = [name for name in runs if name not in config.strategies]
    if missing:
        raise ValueError(f"no arms supplied for strateg(ies): {', '.join(missing)}")
    if extra:
        raise ValueError(f"arms supplied for unconfigured strateg(ies): {', '.join(extra)}")
    for name in config.strategies:
        if not runs[name]:
            raise ValueError(f"strategy {name!r} has no arms")


def _summary(strategy_runs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the nested runs into one row per (strategy, arm)."""
    rows: list[dict[str, Any]] = []
    for strategy, run in strategy_runs.items():
        for arm in run["arms"]:
            metrics = arm["metrics"]
            rows.append(
                {
                    "strategy": strategy,
                    "arm": arm["name"],
                    "status": arm["status"],
                    "recall_macro": metrics["recall_at_k"]["macro"],
                    "recall_micro": metrics["recall_at_k"]["micro"],
                    "precision_macro": metrics["precision_at_k"]["macro"],
                    "precision_ceiling": metrics["precision_at_k"]["ceiling_macro"],
                    "ndcg_macro": metrics["ndcg_at_k"]["macro"],
                    "mrr": metrics["mrr"],
                    "hit_at_1": metrics["hit_at_1"],
                    "evaluated": metrics["evaluated_queries"],
                    "failed": metrics["failed_queries"],
                    "latency_ms": arm["latency_ms"]["query"],
                }
            )
    return rows


def _limitations(config: ChunkingConfig) -> list[str]:
    """State the boundaries of a chunking comparison."""
    return [
        "指标只衡量检索，不代表答案正确性或生成质量。",
        "分块策略改变候选粒度与数量；延迟差异部分来自 chunk 数量，而不只是策略本身。",
        "所有策略共用同一个 HNSW bootstrap 索引与查询参数；本实验不等同于索引调优。",
        "ANN 扫描先于 chunk_strategy 过滤：ef_search 较小时，单策略可能返回少于 K 条，"
        "不同策略的召回差异部分来自其在全局索引中的占比，而不只是分块质量。",
        "P@K 的分母固定为 K，且 gold 只有 1–3 篇；应与 ceiling 一起横向比较。",
        f"预期 Query 数为 {config.expected_queries}；少于该数的运行会被标为 PARTIAL。",
    ]


def _fmt(value: float | int | None, digits: int = 4) -> str:
    """Format a metric for Markdown, keeping ``None`` visible as ``-``."""
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_int(value: int | None) -> str:
    """Format an integer for Markdown, keeping ``None`` visible as ``-``."""
    if value is None:
        return "-"
    return str(int(value))
