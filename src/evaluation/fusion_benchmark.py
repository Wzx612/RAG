"""RRF parameter sensitivity.

Runs the fused arms over a small grid of ``rrf_k`` and ``candidate_k`` values
and records every cell with the same unified schema. The module is DB- and
model-free: the caller injects one arm list per grid cell (built with the real
registry), so the sweep can be tested with scripted retrievers.

Why a separate artifact: the production RRF constant (60) and candidate pool
are defaults, not measured choices. This sweep makes the choice inspectable --
and if every cell reaches the same Recall (the usual outcome on a 150-chunk
corpus), the honest conclusion is that the corpus is too small to separate the
settings, which the limitations state explicitly.
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

FUSION_BENCHMARK_VERSION = 1
"""Schema version of the fusion-sensitivity artifact."""

DEFAULT_RRF_K_VALUES: tuple[int, ...] = (10, 60, 100)
DEFAULT_CANDIDATE_K_VALUES: tuple[int, ...] = (10, 50)


@dataclass(frozen=True)
class FusionConfig:
    """The grid and the sizes shared by every cell."""

    top_k: int = 10
    expected_queries: int = EXPECTED_QUERIES
    chunk_strategy: str = "document"
    rrf_k_values: tuple[int, ...] = DEFAULT_RRF_K_VALUES
    candidate_k_values: tuple[int, ...] = DEFAULT_CANDIDATE_K_VALUES

    def __post_init__(self) -> None:
        """Validate the grid before any arm runs."""
        for name in ("top_k", "expected_queries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.chunk_strategy not in ("document", "fixed_window", "structured"):
            raise ValueError(f"unsupported chunk_strategy {self.chunk_strategy!r}")
        if not self.rrf_k_values:
            raise ValueError("at least one rrf_k value is required")
        if not self.candidate_k_values:
            raise ValueError("at least one candidate_k value is required")
        for name, values in (
            ("rrf_k_values", self.rrf_k_values),
            ("candidate_k_values", self.candidate_k_values),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not repeat")
            for value in values:
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must contain positive ints, got {value!r}")
        smallest = min(self.candidate_k_values)
        if self.top_k > smallest:
            raise ValueError(
                f"top_k ({self.top_k}) exceeds the smallest candidate_k ({smallest})"
            )

    @property
    def cells(self) -> tuple[tuple[int, int], ...]:
        """The (rrf_k, candidate_k) grid in report order."""
        return tuple(
            (rrf_k, candidate_k)
            for rrf_k in self.rrf_k_values
            for candidate_k in self.candidate_k_values
        )


def cell_key(rrf_k: int, candidate_k: int) -> str:
    """Stable key used in the artifact and the Markdown table."""
    return f"rrf_k={rrf_k},candidate_k={candidate_k}"


def run_fusion_sensitivity(
    runs: Mapping[tuple[int, int], Sequence[Arm]],
    queries: Sequence[EvalQuery],
    config: FusionConfig | None = None,
    *,
    environment: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Run every grid cell and return one sensitivity artifact.

    Raises:
        ValueError: ``runs`` does not match the configured grid, or a cell has
            no arms.
    """
    resolved = config if config is not None else FusionConfig()
    missing = [cell for cell in resolved.cells if cell not in runs]
    extra = [cell for cell in runs if cell not in resolved.cells]
    if missing:
        raise ValueError(f"no arms supplied for grid cell(s): {missing}")
    if extra:
        raise ValueError(f"arms supplied for unconfigured grid cell(s): {extra}")
    for cell in resolved.cells:
        if not runs[cell]:
            raise ValueError(f"grid cell {cell} has no arms")

    generated = generated_at if generated_at is not None else datetime.now(UTC)
    cell_runs: dict[str, dict[str, Any]] = {}
    for rrf_k, candidate_k in resolved.cells:
        cell_runs[cell_key(rrf_k, candidate_k)] = run_benchmark(
            runs[(rrf_k, candidate_k)],
            queries,
            BenchmarkConfig(
                top_k=resolved.top_k,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
                chunk_strategy=resolved.chunk_strategy,
                expected_queries=resolved.expected_queries,
            ),
            environment=environment,
            dataset=dataset,
            generated_at=generated,
        )

    statuses = {key: run["status"] for key, run in cell_runs.items()}
    evaluated = sum(
        arm["metrics"]["evaluated_queries"]
        for run in cell_runs.values()
        for arm in run["arms"]
    )
    if statuses and all(status == "PASS" for status in statuses.values()):
        status, reason = "PASS", "every grid cell evaluated every expected query"
    elif evaluated > 0:
        status = "PARTIAL"
        reason = "; ".join(
            f"{key}: {run['status_reason']}" for key, run in cell_runs.items()
        )
    else:
        status, reason = "FAIL", "no grid cell produced a valid record"

    return {
        "schema_version": FUSION_BENCHMARK_VERSION,
        "experiment_name": "rrf_sensitivity",
        "generated_at": generated.isoformat(),
        "status": status,
        "status_reason": reason,
        "parameters": {
            "top_k": resolved.top_k,
            "chunk_strategy": resolved.chunk_strategy,
            "expected_queries": resolved.expected_queries,
            "queries": len(queries),
            "rrf_k_values": list(resolved.rrf_k_values),
            "candidate_k_values": list(resolved.candidate_k_values),
            "cells": [cell_key(*cell) for cell in resolved.cells],
        },
        "environment": dict(environment or {}),
        "dataset": dict(dataset or {}),
        "summary": _summary(cell_runs),
        "cells": cell_runs,
        "limitations": _limitations(resolved),
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render the sweep as a reviewable Markdown report."""
    parameters = result["parameters"]
    lines = [
        "# RRF 参数敏感性",
        "",
        f"- Status: **{result['status']}**",
        f"- Reason: {result['status_reason']}",
        f"- Generated: {result['generated_at']}",
        f"- Queries: {parameters['queries']} / expected {parameters['expected_queries']}"
        f"；Top-K: {parameters['top_k']}",
        f"- Grid: rrf_k={parameters['rrf_k_values']} × "
        f"candidate_k={parameters['candidate_k_values']}",
        "",
        "> 每一格都独立运行统一评测；指标按文档级计算，与主评测同一口径。",
        "",
        "## 1. 指标",
        "",
        "| rrf_k | candidate_k | Arm | Status | Recall macro | Recall micro | "
        "nDCG@K | MRR | P99 ms | Evaluated |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"]:
        latency = row["latency_ms"]
        lines.append(
            f"| {row['rrf_k']} | {row['candidate_k']} | {row['arm']} | "
            f"{row['status']} | {_fmt(row['recall_macro'])} | "
            f"{_fmt(row['recall_micro'])} | {_fmt(row['ndcg_macro'])} | "
            f"{_fmt(row['mrr'])} | {_fmt(latency['p99'], 2)} | "
            f"{row['evaluated']} |"
        )
    lines.extend(["", "## 2. 限制", ""])
    lines.extend(f"- {item}" for item in result["limitations"])
    failures = [
        (row["rrf_k"], row["candidate_k"], row["arm"], row["failed"])
        for row in result["summary"]
        if row["failed"]
    ]
    if failures:
        lines.extend(["", "## 3. 失败的 Query", ""])
        lines.extend(
            f"- rrf_k={rrf_k}, candidate_k={candidate_k}, `{arm}`: "
            f"{failed} failed query(s)"
            for rrf_k, candidate_k, arm, failed in failures
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    result: Mapping[str, Any],
    json_path: Path,
    report_path: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write the JSON and Markdown artifacts, refusing to clobber."""
    ensure_writable((json_path, report_path), overwrite=overwrite)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, report_path


def _summary(cell_runs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the nested runs into one row per (cell, arm)."""
    rows: list[dict[str, Any]] = []
    for key, run in cell_runs.items():
        rrf_k, candidate_k = _parse_key(key)
        for arm in run["arms"]:
            metrics = arm["metrics"]
            rows.append(
                {
                    "rrf_k": rrf_k,
                    "candidate_k": candidate_k,
                    "cell": key,
                    "arm": arm["name"],
                    "status": arm["status"],
                    "recall_macro": metrics["recall_at_k"]["macro"],
                    "recall_micro": metrics["recall_at_k"]["micro"],
                    "ndcg_macro": metrics["ndcg_at_k"]["macro"],
                    "mrr": metrics["mrr"],
                    "hit_at_1": metrics["hit_at_1"],
                    "evaluated": metrics["evaluated_queries"],
                    "failed": metrics["failed_queries"],
                    "latency_ms": arm["latency_ms"]["query"],
                }
            )
    return rows


def _parse_key(key: str) -> tuple[int, int]:
    """Recover the two integers from a stable cell key."""
    rrf_part, candidate_part = key.split(",")
    return int(rrf_part.split("=")[1]), int(candidate_part.split("=")[1])


def _limitations(config: FusionConfig) -> list[str]:
    """State what the sweep can and cannot show."""
    return [
        "指标只衡量检索，不代表答案正确性或生成质量。",
        "150 篇文档下候选池很小；若所有格子指标相同，说明语料规模不足以区分参数，"
        "不能据此宣称某个参数更优。",
        "rrf_k 只影响 RRF 名次权重，不改变候选池；candidate_k 同时改变召回范围与延迟。",
        f"预期 Query 数为 {config.expected_queries}；少于该数的运行会被标为 PARTIAL。",
    ]


def _fmt(value: float | int | None, digits: int = 4) -> str:
    """Format a number for Markdown, keeping ``None`` visible as ``-``."""
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"
