"""Final report assembly and figures.

Reads the unified artifacts produced by the benchmark scripts and renders one
Markdown report plus optional PNG figures. Every table points back to the
artifact file, the array element it came from and the file's SHA256, so a
reader can recompute any number instead of trusting the prose.

The module is deliberately file-based: it opens no database connection, loads
no model, and computes no metric. If an artifact is missing the section says so
and the report is ``PARTIAL`` -- it is never filled with historical numbers.

Figures require the ``report`` extra (matplotlib) and are imported lazily:
without it the report still renders and the figure section records that they
were skipped. When figures are produced they use the validated categorical
palette, a single sequential blue ramp for the heatmap, solid hairline grids
and one y-axis per plot.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT_VERSION = 1
"""Schema version of the assembled report payload."""

REQUIRED_ARTIFACTS: tuple[str, ...] = ("benchmark", "index", "chunking")
"""The three artifacts the final report expects."""

OPTIONAL_ARTIFACTS: tuple[str, ...] = ("rerank", "judge")
"""Artifacts that enrich the report when present but are never required.

``rerank`` is the live Top-20 pool -> Top-5 cross-encoder run. It is optional
because a corpus small enough to cap the pool at 10 cannot produce it.
``judge`` is the LLM-as-a-Judge summary; it is optional because contexts and
answers are needed first, and a missing judge section must not be invented.
"""

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
"""Fixed categorical order; a series keeps its slot across figures."""

BLUE_STEPS: tuple[str, ...] = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
"""Single-hue sequential ramp for the HNSW heatmap."""


class ReportError(RuntimeError):
    """An input artifact is unreadable or does not have the expected shape."""


class FiguresUnavailableError(ReportError):
    """matplotlib is not installed; the report can still render without figures."""


def load_artifacts(
    paths: Mapping[str, Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load JSON artifacts and return ``(payloads, metadata)``.

    Missing files are omitted from both mappings; malformed files raise
    :class:`ReportError` rather than being rendered as if they existed.
    """
    payloads: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for key, raw_path in paths.items():
        path = Path(raw_path)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReportError(f"cannot read artifact {key!r} at {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ReportError(f"artifact {key!r} at {path} is not a JSON object")
        if "status" not in payload or "generated_at" not in payload:
            raise ReportError(
                f"artifact {key!r} at {path} lacks status/generated_at; "
                "was it written by a benchmark script?"
            )
        payloads[key] = payload
        metadata[key] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "status": payload["status"],
            "generated_at": payload["generated_at"],
        }
    return payloads, metadata


def sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def report_status(
    payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    """Derive the report status from presence and each artifact's status."""
    missing = [key for key in REQUIRED_ARTIFACTS if key not in payloads]
    statuses = {key: str(payload.get("status")) for key, payload in payloads.items()}
    failing = {
        key: status for key, status in statuses.items() if status not in ("PASS",)
    }
    if missing and not payloads:
        return "FAIL", "no artifact was provided"
    if missing or failing:
        reasons: list[str] = []
        if missing:
            reasons.append(f"missing artifact(s): {', '.join(missing)}")
        reasons.extend(
            f"{key} is {status}" for key, status in sorted(failing.items())
        )
        return "PARTIAL", "; ".join(reasons)
    return "PASS", "all three artifacts are present and PASS"


def render_report(
    payloads: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    context: Mapping[str, Any] | None = None,
    figures: Sequence[str] = (),
    unverified: Sequence[str] = (),
    reproduction: Sequence[str] = (),
    generated_at: datetime | None = None,
) -> str:
    """Render the final Markdown report.

    Args:
        payloads: Loaded artifacts keyed by ``benchmark`` / ``index`` / ``chunking``.
        metadata: Path, SHA256 and status for each loaded artifact.
        context: Git and environment facts recorded at the top.
        figures: Figure filenames to embed relative to the report.
        unverified: Claims that must stay explicitly unverified.
        reproduction: Commands a reader can re-run.
        generated_at: Override for tests.

    Returns:
        The Markdown document.
    """
    generated = generated_at if generated_at is not None else datetime.now(UTC)
    status, reason = report_status(payloads)
    lines = [
        "# 信用卡知识库 RAG 检索评测：最终报告",
        "",
        f"- 生成时间: {generated.isoformat()}",
        f"- Status: **{status}**",
        f"- Reason: {reason}",
        "",
        "> 本报告只汇总已落盘的 artifact，不重算指标、不补历史数字。"
        "每个表格都标注了来源文件、SHA256 与字段路径。",
        "",
        "## 1. 输入与可追溯性",
        "",
        "| Artifact | Status | Generated | SHA256 | Path |",
        "|---|---|---|---|---|",
    ]
    for key in REQUIRED_ARTIFACTS:
        meta = metadata.get(key)
        if meta is None:
            lines.append(f"| {key} | missing | - | - | - |")
        else:
            lines.append(
                f"| {key} | {meta['status']} | {meta['generated_at']} | "
                f"`{str(meta['sha256'])[:16]}...` | `{meta['path']}` |"
            )
    if context:
        lines.extend(["", "### 运行上下文", ""])
        for key, value in context.items():
            lines.append(f"- {key}: {value}")

    lines.extend(["", "## 2. 数据与标注口径", ""])
    dataset = _first_dataset(payloads)
    if dataset:
        lines.extend(_dataset_lines(dataset))
    else:
        lines.append("- 没有 artifact 提供 dataset 元数据。")
    lines.extend(
        [
            "",
            "- Gold 不是穷尽标注；非 gold 位置是未标注，不是 true negative。",
            "- Precision@K 的分母固定为 K，并附 ceiling；只用于横向比较。",
            "- 所有指标为文档级：同一文档的多个 chunk 先归并为一个排名。",
            "",
            "## 3. 统一检索评测",
            "",
        ]
    )
    benchmark = payloads.get("benchmark")
    if benchmark is None:
        lines.append("未提供 benchmark artifact。")
    else:
        lines.extend(_benchmark_section(benchmark, metadata.get("benchmark", {})))

    lines.extend(["", "## 4. 索引参数实验", ""])
    index = payloads.get("index")
    if index is None:
        lines.append("未提供 index artifact。")
    else:
        lines.extend(_index_section(index, metadata.get("index", {})))

    lines.extend(["", "## 5. 分块策略对比", ""])
    chunking = payloads.get("chunking")
    if chunking is None:
        lines.append("未提供 chunking artifact。")
    else:
        lines.extend(_chunking_section(chunking, metadata.get("chunking", {})))

    section = 6
    rerank = payloads.get("rerank")
    if rerank is not None:
        lines.extend(["", f"## {section}. Cross-Encoder 精排", ""])
        lines.extend(_rerank_section(rerank, metadata.get("rerank", {})))
        section += 1

    judge = payloads.get("judge")
    if judge is not None:
        lines.extend(["", f"## {section}. LLM-as-a-Judge", ""])
        lines.extend(_judge_section(judge, metadata.get("judge", {})))
        section += 1

    if figures:
        lines.extend(["", f"## {section}. 图表", ""])
        for name in figures:
            lines.extend([f"![{Path(name).stem}](figures/{name})", ""])
        section += 1

    lines.extend(["", f"## {section}. 复现命令", "", "```bash"])
    if reproduction:
        lines.extend(reproduction)
    else:
        lines.append("# (not recorded)")
    section += 1
    lines.extend(["```", "", f"## {section}. 明确未验证的范围", ""])
    judge = payloads.get("judge")
    if judge is None:
        judge_boundary = (
            "未提供 Judge artifact：Context Relevance / Faithfulness 与 "
            "Answer Relevance 均未评估。"
        )
    else:
        judge_parameters = judge.get("parameters", {})
        judged = judge_parameters.get("queries", "-")
        expected = judge_parameters.get("expected_queries", "-")
        scope = (
            f"仅执行 {judged}/{expected} 条"
            if judged != expected
            else f"已执行 {judged}/{expected} 条"
        )
        judge_boundary = (
            "真实 LLM 的答案正确性与 Answer Relevance 未评价；"
            f"Context Relevance / Faithfulness {scope}"
            f"（{judge.get('status')}），其分数是模型判断而非人工真值。"
        )
    boundaries = [
        "生产级可用性、高可用、并发与容量规划均未验证。",
        "10 万级以上规模的索引调优未验证；合成规模只反映索引行为。",
        judge_boundary,
        "容器内外结果不等价；容器内验证以容器 artifact 为准。",
        "中文语料与跨语言检索未单独验证。",
    ]
    lines.extend(f"- {item}" for item in [*boundaries, *unverified])
    return "\n".join(lines) + "\n"


def write_figures(
    payloads: Mapping[str, Mapping[str, Any]],
    figures_dir: Path,
) -> list[Path]:
    """Render the report figures and return their paths.

    Raises:
        FiguresUnavailableError: matplotlib is not installed.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise FiguresUnavailableError(
            "matplotlib is not installed; install the 'report' extra to render figures"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    figures_dir.mkdir(parents=True, exist_ok=True)
    _apply_style(plt)
    written: list[Path] = []

    benchmark = payloads.get("benchmark")
    if benchmark and benchmark.get("arms"):
        path = figures_dir / "arm_metrics.png"
        _arm_metrics_figure(plt, benchmark, path)
        written.append(path)
        path = figures_dir / "arm_latency.png"
        _arm_latency_figure(plt, benchmark, path)
        written.append(path)

    index = payloads.get("index")
    if index and _hnsw_matrices(index):
        path = figures_dir / "hnsw_heatmap.png"
        cmap = LinearSegmentedColormap.from_list("blue", BLUE_STEPS)
        _hnsw_heatmap_figure(plt, cmap, index, path)
        written.append(path)

    chunking = payloads.get("chunking")
    if chunking and chunking.get("summary"):
        path = figures_dir / "chunking_metrics.png"
        _chunking_figure(plt, chunking, path)
        written.append(path)

    judge = payloads.get("judge")
    if judge and judge.get("summary"):
        path = figures_dir / "judge_distribution.png"
        _judge_distribution_figure(plt, judge, path)
        written.append(path)

    plt.close("all")
    return written


# ------------------------------------------------------------------ sections


def _benchmark_section(
    payload: Mapping[str, Any], meta: Mapping[str, Any]
) -> list[str]:
    """Render the unified arm tables."""
    source = _source(meta, "arms[].metrics")
    arms = payload.get("arms", [])
    lines = [
        f"> 来源: {source}",
        "",
        "| Arm | Status | Recall macro | Recall micro | P@K | P@K ceiling | nDCG@K | "
        "MRR | Hit@1 | Evaluated | Source |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for position, arm in enumerate(arms):
        metrics = arm.get("metrics", {})
        recall = metrics.get("recall_at_k", {})
        precision = metrics.get("precision_at_k", {})
        ndcg = metrics.get("ndcg_at_k", {})
        lines.append(
            f"| {arm.get('name')} | {arm.get('status')} | {_fmt(recall.get('macro'))} | "
            f"{_fmt(recall.get('micro'))} | {_fmt(precision.get('macro'))} | "
            f"{_fmt(precision.get('ceiling_macro'))} | {_fmt(ndcg.get('macro'))} | "
            f"{_fmt(metrics.get('mrr'))} | {metrics.get('hit_at_1', '-')} | "
            f"{metrics.get('evaluated_queries', '-')} | `benchmark#arms[{position}]` |"
        )
    lines.extend(
        [
            "",
            f"> 延迟来源: {_source(meta, 'arms[].latency_ms.query')}",
            "",
            "| Arm | n | P50 ms | P95 ms | P99 ms | Max ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in arms:
        latency = arm.get("latency_ms", {}).get("query", {})
        lines.append(
            f"| {arm.get('name')} | {latency.get('n', '-')} | {_fmt(latency.get('p50'))} | "
            f"{_fmt(latency.get('p95'))} | {_fmt(latency.get('p99'))} | "
            f"{_fmt(latency.get('max'))} |"
        )
    lines.extend(["", f"- 本次状态: **{payload.get('status')}** ({payload.get('status_reason')})"])
    return lines


def _index_section(
    payload: Mapping[str, Any], meta: Mapping[str, Any]
) -> list[str]:
    """Render a compact HNSW / IVFFlat summary."""
    parameters = payload.get("parameters", {})
    lines = [
        f"> 来源: {_source(meta, 'configurations[].query_params')}",
        f"- Top-K: {parameters.get('top_k')}；查询数: {parameters.get('queries')}；"
        f"synthetic scale: {parameters.get('synthetic_scale')}",
        "- 每个配置在临时表副本上重建索引；ANN Recall 相对精确 Seq Scan。",
        "- synthetic scale > 1 时向量是确定性复制，只反映索引行为，"
        "不进入语义质量结论。",
        "",
    ]
    hnsw_settings = parameters.get("hnsw_ef_search_values", [])
    hnsw = [
        record
        for record in payload.get("configurations", [])
        if record.get("method") == "hnsw"
    ]
    if hnsw and hnsw_settings:
        header = " | ".join(f"ef_search={value}" for value in hnsw_settings)
        lines.extend(
            [
                f"| m | ef_construction | {header} | Source |",
                "|---:|---:|" + "---:|" * len(hnsw_settings) + "---|",
            ]
        )
        for record in hnsw:
            cells = [
                _recall_cell(record, f"ef_search={value}") for value in hnsw_settings
            ]
            lines.append(
                f"| {record['params'].get('m')} | {record['params'].get('ef_construction')} | "
                + " | ".join(cells)
                + f" | `index#{record['index_name']}` |"
            )
    else:
        lines.append("HNSW 配置缺失或未运行。")

    ivfflat_probes = parameters.get("ivfflat_probes_values", [])
    ivfflat = [
        record
        for record in payload.get("configurations", [])
        if record.get("method") == "ivfflat"
    ]
    if ivfflat and ivfflat_probes:
        header = " | ".join(f"probes={value}" for value in ivfflat_probes)
        lines.extend(
            [
                "",
                f"| lists | {header} | Source |",
                "|---:|" + "---:|" * len(ivfflat_probes) + "---|",
            ]
        )
        for record in ivfflat:
            cells = [
                _recall_cell(record, f"probes={value}") for value in ivfflat_probes
            ]
            lines.append(
                f"| {record['params'].get('lists')} | "
                + " | ".join(cells)
                + f" | `index#{record['index_name']}` |"
            )
    else:
        lines.append("")
        lines.append("IVFFlat 配置缺失或未运行。")
    lines.extend(["", f"- 本次状态: **{payload.get('status')}** ({payload.get('status_reason')})"])
    return lines


def _chunking_section(
    payload: Mapping[str, Any], meta: Mapping[str, Any]
) -> list[str]:
    """Render the per-strategy, per-arm comparison."""
    lines = [
        f"> 来源: {_source(meta, 'summary[]')}",
        "",
        "| Strategy | Arm | Status | Recall macro | MRR | nDCG@K | P99 ms | Evaluated |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("summary", []):
        latency = row.get("latency_ms", {})
        lines.append(
            f"| {row.get('strategy')} | {row.get('arm')} | {row.get('status')} | "
            f"{_fmt(row.get('recall_macro'))} | {_fmt(row.get('mrr'))} | "
            f"{_fmt(row.get('ndcg_macro'))} | {_fmt(latency.get('p99'))} | "
            f"{row.get('evaluated', '-')} |"
        )
    stats = payload.get("chunk_stats", {})
    if stats:
        lines.extend(
            [
                "",
                "| Strategy | Chunks | Documents | Avg chars |",
                "|---|---:|---:|---:|",
            ]
        )
        for strategy, values in stats.items():
            lines.append(
                f"| {strategy} | {values.get('chunk_count', '-')} | "
                f"{values.get('documents', '-')} | {_fmt(values.get('avg_chars'), 1)} |"
            )
    lines.extend(["", f"- 本次状态: **{payload.get('status')}** ({payload.get('status_reason')})"])
    return lines


def _rerank_section(
    payload: Mapping[str, Any], meta: Mapping[str, Any]
) -> list[str]:
    """Render the live candidate-pool -> reranked Top-K comparison."""
    source = _source(meta, "arms[].metrics")
    parameters = payload.get("parameters", {})
    arms = payload.get("arms", [])
    lines = [
        f"> 来源: {source}",
        f"- 候选池 → 最终 Top-K: **{parameters.get('candidate_k')} → "
        f"{parameters.get('top_k')}**；查询数: {parameters.get('queries')}；"
        f"RRF k: {parameters.get('rrf_k')}；chunk strategy: "
        f"`{parameters.get('chunk_strategy')}`",
        "- 两个臂在同一批候选上实时执行：`rrf` 是融合排序基线，`rrf_rerank` "
        "用 Cross-Encoder 重排候选池后截断到最终 Top-K。",
        "",
        "| Arm | Status | Recall macro | Recall micro | nDCG@K | MRR | Hit@1 | "
        "Evaluated | Source |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for position, arm in enumerate(arms):
        metrics = arm.get("metrics", {})
        recall = metrics.get("recall_at_k", {})
        ndcg = metrics.get("ndcg_at_k", {})
        lines.append(
            f"| {arm.get('name')} | {arm.get('status')} | {_fmt(recall.get('macro'))} | "
            f"{_fmt(recall.get('micro'))} | {_fmt(ndcg.get('macro'))} | "
            f"{_fmt(metrics.get('mrr'))} | {metrics.get('hit_at_1', '-')} | "
            f"{metrics.get('evaluated_queries', '-')} | `rerank#arms[{position}]` |"
        )
    lines.extend(
        [
            "",
            f"> 延迟来源: {_source(meta, 'arms[].latency_ms.query')}",
            "",
            "| Arm | n | P50 ms | P95 ms | P99 ms | Max ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in arms:
        latency = arm.get("latency_ms", {}).get("query", {})
        lines.append(
            f"| {arm.get('name')} | {latency.get('n', '-')} | {_fmt(latency.get('p50'))} | "
            f"{_fmt(latency.get('p95'))} | {_fmt(latency.get('p99'))} | "
            f"{_fmt(latency.get('max'))} |"
        )
    lines.extend(
        [
            "",
            "- 两个臂的候选池相同，因此 Recall 的差异只来自候选池构建，"
            "重排只应改变排序；跨策略比较不适用。",
            "- 延迟包含 CPU Cross-Encoder 前向，单进程、顺序执行。",
            f"- 本次状态: **{payload.get('status')}** ({payload.get('status_reason')})",
        ]
    )
    return lines


def _judge_section(
    payload: Mapping[str, Any], meta: Mapping[str, Any]
) -> list[str]:
    """Render the judge summary without turning a smoke run into a quality claim."""
    parameters = payload.get("parameters", {})
    summary = payload.get("summary", {})
    provider = payload.get("provider", {})
    rag = payload.get("rag", {})
    queries = parameters.get("queries", summary.get("queries", "-"))
    expected = parameters.get("expected_queries", summary.get("expected_queries", "-"))
    lines = [
        f"> 来源: {_source(meta, 'summary.metrics')}",
        f"- Status: **{payload.get('status')}** ({payload.get('status_reason')})",
        f"- 评判查询: **{queries} / {expected}**"
        + (
            "（smoke；不代表全量数据集）"
            if isinstance(queries, int) and isinstance(expected, int) and queries < expected
            else ""
        ),
        f"- NOT RUN: **{summary.get('not_run_queries', 0)}**（未评分，未按 0 分计入）",
        f"- RAG provider: `{rag.get('provider', {}).get('kind', '-')}` / "
        f"`{rag.get('provider', {}).get('name', '-')}` / model "
        f"`{rag.get('model', '-')}`",
        f"- Judge provider: `{provider.get('kind', '-')}` / "
        f"`{provider.get('name', '-')}` / model `{provider.get('model', '-')}`",
        f"- 同模型自评: **{summary.get('same_model', False)}**；"
        f"真实 Judge 调用: **{provider.get('real_judge_calls', False)}**",
        f"- Prompt: `{parameters.get('prompt_versions', {})}`",
        "- 均值/中位数/分布只使用 `status=success` 的评分；失败与未实际调用不计 0 分。",
        "",
        "| Metric | Attempted | Success | Failed | Success rate | Valid n | Mean | "
        "Median | Low (< threshold) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("context_relevance", "Context Relevance"),
        ("faithfulness", "Faithfulness"),
    ):
        item = summary.get("metrics", {}).get(key, {})
        lines.append(
            f"| {label} | {item.get('attempted', '-')} | {item.get('success', '-')} | "
            f"{item.get('failed', '-')} | {_fmt(item.get('success_rate'))} | "
            f"{item.get('valid_scores', '-')} | "
            f"{_fmt(item.get('mean'))} | {_fmt(item.get('median'))} | "
            f"{item.get('low_score_count', '-')} |"
        )

    lines.extend(
        [
            "",
            "| Metric | 1 | 2 | 3 | 4 | 5 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("context_relevance", "Context Relevance"),
        ("faithfulness", "Faithfulness"),
    ):
        distribution = summary.get("metrics", {}).get(key, {}).get("distribution", {})
        cells = " | ".join(str(distribution.get(str(score), 0)) for score in range(1, 6))
        lines.append(f"| {label} | {cells} |")

    low_rows = [
        row
        for key in ("context_relevance", "faithfulness")
        for row in summary.get("metrics", {}).get(key, {}).get("low_score_samples", [])
    ]
    if low_rows:
        lines.extend(
            [
                "",
                "低分样例：",
                "",
                "| Query | Metric | Score | Reason |",
                "|---|---|---:|---|",
            ]
        )
        for row in low_rows:
            lines.append(
                f"| {row.get('query_id')} | {row.get('metric')} | {row.get('score')} | "
                f"{_cell(str(row.get('reason', '')))} |"
            )

    failure_rows: list[str] = []
    for key, label in (
        ("context_relevance", "Context Relevance"),
        ("faithfulness", "Faithfulness"),
    ):
        counts = (
            summary.get("metrics", {})
            .get(key, {})
            .get("failure_counts", {})
        )
        for status, count in counts.get("status", {}).items():
            failure_rows.append(f"| {label} | {status} | {count} |")
        for category, count in counts.get("error_category", {}).items():
            failure_rows.append(f"| {label} | provider:{category} | {count} |")
    if failure_rows:
        lines.extend(
            [
                "",
                "失败分类：",
                "",
                "| Metric | Status / error | Count |",
                "|---|---|---:|",
            ]
        )
        lines.extend(failure_rows)

    lines.extend(["", "限制："])
    limitations = payload.get("limitations", [])
    if limitations:
        lines.extend(f"- {item}" for item in limitations)
    else:
        lines.append("- 该 artifact 未记录限制。")
    gold_usage = payload.get("gold_usage", {})
    lines.append(
        "- Faithfulness 只使用本次实际进入生成上下文的 `evidence`；"
        f"`gold_response` 参与评分: **{gold_usage.get('gold_response_used_for_scoring', False)}**。"
    )
    return lines


def _dataset_lines(dataset: Mapping[str, Any]) -> list[str]:
    """Render the dataset block shared by the artifacts."""
    hashes = dataset.get("hashes", {})
    lines = [
        f"- queries: {dataset.get('queries', '-')} / "
        f"on disk {dataset.get('eval_queries_on_disk', '-')}",
        f"- gold doc count: {dataset.get('gold_docs', '-')}",
    ]
    for name, digest in hashes.items():
        lines.append(f"- `{name}`: `{digest}`")
    return lines


def _first_dataset(payloads: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    """Return the first artifact that carries dataset metadata."""
    for key in REQUIRED_ARTIFACTS:
        dataset = payloads.get(key, {}).get("dataset")
        if dataset:
            return dataset
    return {}


def _source(meta: Mapping[str, Any], field: str) -> str:
    """Build the traceability marker used above every table."""
    if not meta:
        return "[artifact: missing]"
    digest = str(meta.get("sha256", ""))[:12]
    return f"[artifact: {meta.get('path')}#{field} sha256={digest}]"


def _recall_cell(record: Mapping[str, Any], setting: str) -> str:
    """Format one recovery/latency cell for the index summary."""
    measurement = record.get("query_params", {}).get(setting)
    if not measurement:
        return "-"
    recall = _fmt(measurement.get("ann_recall_mean"), 3)
    p95 = _fmt(measurement.get("latency_ms", {}).get("p95"), 2)
    return f"{recall} / {p95} ms"


def _fmt(value: float | int | None, digits: int = 4) -> str:
    """Format a number, keeping ``None`` visible as ``-``."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    return f"{float(value):.{digits}f}"


def _cell(value: str) -> str:
    """Escape a Markdown table cell."""
    return value.replace("|", "\\|").replace("\n", " ")


# ------------------------------------------------------------------- figures


def _apply_style(plt: Any) -> None:
    """Apply the report's static chart style once."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def _style_axes(ax: Any) -> None:
    """Recessive chrome: no box, hairline horizontal grid only."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)


def _grouped_bars(
    ax: Any,
    categories: Sequence[str],
    series: Mapping[str, Sequence[float | None]],
    *,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Draw thin grouped bars with surface gaps and a fixed colour order."""
    count = max(len(series), 1)
    width = 0.8 / count
    positions = list(range(len(categories)))
    for index, (label, values) in enumerate(series.items()):
        offset = (index - (count - 1) / 2) * width
        clean = [0.0 if value is None else float(value) for value in values]
        ax.bar(
            [position + offset for position in positions],
            clean,
            width=width * 0.92,
            color=SERIES[index % len(SERIES)],
            edgecolor=SURFACE,
            linewidth=1.5,
            label=label,
            zorder=3,
        )
    ax.set_xticks(positions)
    rotated = len(categories) > 3
    ax.set_xticklabels(
        categories,
        rotation=18 if rotated else 0,
        ha="right" if rotated else "center",
    )
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if len(series) > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=len(series))


def _arm_metrics_figure(plt: Any, payload: Mapping[str, Any], path: Path) -> None:
    """Recall@K and MRR per arm."""
    arms = payload.get("arms", [])
    labels = [str(arm.get("name")) for arm in arms]
    recall = [arm.get("metrics", {}).get("recall_at_k", {}).get("macro") for arm in arms]
    mrr = [arm.get("metrics", {}).get("mrr") for arm in arms]
    figure, ax = plt.subplots(figsize=(9.2, 4.3))
    _grouped_bars(
        ax,
        labels,
        {"Recall@K (macro)": recall, "MRR": mrr},
        ylabel="score",
        ylim=(0.0, 1.05),
    )
    ax.set_title("Retrieval quality by arm", loc="left", pad=22)
    ax.text(
        0.0,
        1.02,
        "Document-level scoring; Recall@K and MRR share the same 0-1 scale",
        transform=ax.transAxes,
        color=INK_SECONDARY,
        fontsize=8.5,
    )
    _style_axes(ax)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _arm_latency_figure(plt: Any, payload: Mapping[str, Any], path: Path) -> None:
    """P50/P95/P99 latency per arm on one axis (log scale when skewed)."""
    arms = payload.get("arms", [])
    labels = [str(arm.get("name")) for arm in arms]
    series = {
        percentile: [
            arm.get("latency_ms", {}).get("query", {}).get(percentile) for arm in arms
        ]
        for percentile in ("p50", "p95", "p99")
    }
    values = [float(value) for group in series.values() for value in group if value is not None]
    figure, ax = plt.subplots(figsize=(9.2, 4.3))
    _grouped_bars(ax, labels, series, ylabel="latency (ms, log scale)")
    if values and max(values) / max(min(values), 1e-6) > 20:
        ax.set_yscale("log")
    ax.set_title("Query latency by arm", loc="left", pad=22)
    ax.text(
        0.0,
        1.02,
        "Single process, sequential, warmed; log scale when the spread exceeds 20x",
        transform=ax.transAxes,
        color=INK_SECONDARY,
        fontsize=8.5,
    )
    _style_axes(ax)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _hnsw_matrices(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return {ef_search: (m_values, efc_values, matrix)} for HNSW configs."""
    hnsw = [
        record for record in payload.get("configurations", []) if record.get("method") == "hnsw"
    ]
    if not hnsw:
        return {}
    m_values = sorted({int(record["params"]["m"]) for record in hnsw})
    efc_values = sorted({int(record["params"]["ef_construction"]) for record in hnsw})
    settings: dict[str, Any] = {}
    for record in hnsw:
        build = (
            int(record["params"]["m"]),
            int(record["params"]["ef_construction"]),
        )
        for key, measurement in record.get("query_params", {}).items():
            settings.setdefault(key, {})[build] = measurement.get("ann_recall_mean")
    matrices: dict[str, Any] = {}
    for key, cells in settings.items():
        matrix = [
            [cells.get((m_value, efc_value)) for efc_value in efc_values]
            for m_value in m_values
        ]
        matrices[key] = (m_values, efc_values, matrix)
    return matrices


def _hnsw_heatmap_figure(
    plt: Any, cmap: Any, payload: Mapping[str, Any], path: Path
) -> None:
    """Small multiples of ANN Recall by m / ef_construction per ef_search."""
    matrices = _hnsw_matrices(payload)
    settings = sorted(matrices, key=lambda key: int(key.split("=")[-1]))
    cmap.set_bad(GRIDLINE)
    figure, axes = plt.subplots(
        1, len(settings), figsize=(4.1 * len(settings), 3.9), squeeze=False
    )
    for ax, setting in zip(axes[0], settings, strict=True):
        m_values, efc_values, matrix = matrices[setting]
        cells = [
            [float(value) if value is not None else float("nan") for value in row]
            for row in matrix
        ]
        image = ax.imshow(cells, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                if value is None:
                    ax.text(
                        column_index,
                        row_index,
                        "n/a",
                        ha="center",
                        va="center",
                        color=INK_MUTED,
                        fontsize=7,
                    )
                    continue
                ax.text(
                    column_index,
                    row_index,
                    f"{float(value):.2f}",
                    ha="center",
                    va="center",
                    color="#ffffff" if float(value) > 0.55 else INK,
                    fontsize=8,
                )
        ax.set_xticks(range(len(efc_values)))
        ax.set_xticklabels([f"efc={value}" for value in efc_values])
        ax.set_yticks(range(len(m_values)))
        ax.set_yticklabels([f"m={value}" for value in m_values])
        ax.set_title(setting, loc="left", fontsize=10)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    figure.colorbar(image, ax=list(axes[0]), shrink=0.85, label="ANN Recall@K")
    figure.suptitle("HNSW: ANN Recall@K by build parameters", x=0.02, ha="left", fontsize=12)
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _chunking_figure(plt: Any, payload: Mapping[str, Any], path: Path) -> None:
    """Recall and MRR per strategy, one panel per metric."""
    summary = payload.get("summary", [])
    strategies = list(dict.fromkeys(str(row.get("strategy")) for row in summary))
    arms = list(dict.fromkeys(str(row.get("arm")) for row in summary))
    lookup = {(row.get("strategy"), row.get("arm")): row for row in summary}
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, metric, title in (
        (axes[0], "recall_macro", "Recall@K (macro)"),
        (axes[1], "mrr", "MRR"),
    ):
        series = {
            arm: [lookup.get((strategy, arm), {}).get(metric) for strategy in strategies]
            for arm in arms
        }
        _grouped_bars(ax, strategies, series, ylabel="score", ylim=(0.0, 1.05))
        ax.set_title(title, loc="left", fontsize=10)
        _style_axes(ax)
    figure.suptitle("Chunking strategy comparison", x=0.02, ha="left", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _judge_distribution_figure(
    plt: Any, payload: Mapping[str, Any], path: Path
) -> None:
    """Score counts on the 1-5 rubric, one panel per metric."""
    summary = payload.get("summary", {})
    metrics = summary.get("metrics", {})
    counts_by_key = {
        key: [
            int(metrics.get(key, {}).get("distribution", {}).get(str(score), 0))
            for score in range(1, 6)
        ]
        for key in ("context_relevance", "faithfulness")
    }
    top = max([count for values in counts_by_key.values() for count in values] + [1]) + 1
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, key, title, color in (
        (axes[0], "context_relevance", "Context Relevance", SERIES[0]),
        (axes[1], "faithfulness", "Faithfulness", SERIES[1]),
    ):
        counts = counts_by_key[key]
        bars = ax.bar(
            range(1, 6),
            counts,
            width=0.62,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=3,
        )
        for bar, count in zip(bars, counts, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(count),
                ha="center",
                va="bottom",
                color=INK_SECONDARY,
                fontsize=8.5,
            )
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("score (1-5)")
        ax.set_ylim(0.0, float(top))
        ax.set_title(title, loc="left", fontsize=10)
        item = metrics.get(key, {})
        ax.text(
            0.0,
            1.02,
            f"successful scores only; n={item.get('valid_scores', '-')}",
            transform=ax.transAxes,
            color=INK_SECONDARY,
            fontsize=8.5,
        )
        _style_axes(ax)
    axes[0].set_ylabel("queries")
    figure.suptitle(
        "LLM-as-a-Judge score distribution", x=0.02, ha="left", fontsize=12
    )
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
