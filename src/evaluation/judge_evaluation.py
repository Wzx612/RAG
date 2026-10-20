"""Batch execution, aggregation and output for LLM-as-a-Judge.

The input is a RAG benchmark artifact: each ``per_query`` record already holds
the question, the generated answer and the exact evidence sent to the answer
model. The judge never re-retrieves and never substitutes ``gold_response`` for
the supplied evidence.

Aggregation is deliberately conservative:

* only ``status="success"`` scores enter mean, median or the 1-5 distribution;
* every attempted-but-failed score is reported separately, never as 0;
* a three-query smoke can never be PASS when the dataset has 30 queries.

Implemented in: LLM-as-a-Judge evaluation.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.evaluation.artifacts import ensure_writable
from src.evaluation.judge import LLMJudge
from src.evaluation.judge_prompts import (
    CONTEXT_RELEVANCE_VERSION,
    FAITHFULNESS_VERSION,
)
from src.schema.judge import (
    ContextRelevanceResult,
    FaithfulnessResult,
    JudgeContext,
    JudgeQueryRecord,
)

ANSWER_STATUSES = frozenset({"ok", "insufficient_evidence"})
"""RAG statuses that count as an answer, including a supported refusal."""

METRIC_KEYS = ("context_relevance", "faithfulness")
"""Metric keys shared by records, summary and report."""


class JudgeEvaluationError(RuntimeError):
    """The RAG artifact or judge configuration cannot support an evaluation."""


def load_rag_artifact(path: Path) -> dict[str, Any]:
    """Read a RAG benchmark artifact without modifying it.

    Raises:
        JudgeEvaluationError: the file is unreadable, not an object, or lacks
            ``per_query`` / ``provider`` / ``status`` / ``generated_at``.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise JudgeEvaluationError(f"cannot read RAG artifact at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise JudgeEvaluationError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise JudgeEvaluationError(f"{path} must hold a JSON object")
    for key in ("per_query", "provider", "status", "generated_at"):
        if key not in payload:
            raise JudgeEvaluationError(f"{path} lacks required key {key!r}")
    records = payload["per_query"]
    if not isinstance(records, list) or not records:
        raise JudgeEvaluationError(f"{path} has no per_query records")
    if not all(isinstance(record, dict) for record in records):
        raise JudgeEvaluationError(f"{path} has a non-object per_query record")
    if not isinstance(payload["provider"], dict):
        raise JudgeEvaluationError(f"{path} provider must be an object")
    return payload


def run_judge_evaluation(
    rag: Mapping[str, Any],
    judge: LLMJudge,
    *,
    expected_queries: int = 30,
    limit: int | None = None,
    max_concurrency: int = 1,
    low_score_threshold: int = 3,
    environment: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    transport: Mapping[str, Any] | None = None,
    rag_artifact_path: Path | None = None,
    rag_artifact_sha256: str | None = None,
    api_key_present: bool = False,
    commands: Sequence[str] = (),
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Judge every selected RAG record and return a summary artifact.

    Args:
        rag: Loaded RAG benchmark artifact.
        judge: Configured judge.
        expected_queries: Size of the full labelled set (30 in this project).
        limit: Judge only the first N RAG records, for a smoke run.
        max_concurrency: Judge calls in flight. ``1`` is sequential.
        low_score_threshold: Scores strictly below this are low-score samples.
        environment: Read-only runtime facts recorded in the artifact.
        dataset: Query/file metadata recorded in the artifact.
        transport: Timeout/retry settings of the judge provider.
        rag_artifact_path: Source path recorded for traceability.
        rag_artifact_sha256: Source file hash recorded for traceability.
        api_key_present: Boolean provenance only; the key is never stored.
        commands: Reproduction commands recorded in the artifact.
        progress: Optional per-query progress callback.

    Returns:
        A JSON-serialisable artifact with full ``per_query`` records.

    Raises:
        JudgeEvaluationError: the limit or concurrency is invalid, or the RAG
            artifact has no usable records.
    """
    entries = list(rag["per_query"])
    if not entries:
        raise JudgeEvaluationError("the RAG artifact has no per_query records")
    if not all(isinstance(entry, Mapping) for entry in entries):
        raise JudgeEvaluationError("every per_query record must be a JSON object")
    if limit is not None:
        if limit < 1 or limit > len(entries):
            raise JudgeEvaluationError(
                f"limit must be between 1 and {len(entries)}, got {limit}"
            )
        entries = entries[:limit]
    if max_concurrency < 1:
        raise JudgeEvaluationError("max_concurrency must be at least 1")

    emit = progress or (lambda _message: None)
    started_at = datetime.now(UTC)
    rag_provider = dict(rag.get("provider", {}))
    rag_run_id = str(rag.get("run_id") or rag.get("generated_at") or "unknown")
    rag_is_real = bool(rag_provider.get("is_real", False))
    judge_model = str(judge.provider.model)
    rag_model = rag_provider.get("model")
    same_model = isinstance(rag_model, str) and bool(rag_model) and rag_model == judge_model

    def work(entry: Mapping[str, Any]) -> dict[str, Any]:
        return _judge_entry(
            entry,
            judge,
            rag_run_id=rag_run_id,
            rag_provider_name=_optional_str(rag_provider.get("name")),
            rag_model=_optional_str(rag_model),
            rag_is_real=rag_is_real,
        )

    records: list[dict[str, Any]] = []
    if max_concurrency == 1:
        for index, entry in enumerate(entries, start=1):
            emit(f"[{index}/{len(entries)}] {entry.get('query_id')}")
            records.append(work(entry))
    else:
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = []
            for index, entry in enumerate(entries, start=1):
                emit(f"[{index}/{len(entries)}] {entry.get('query_id')}")
                futures.append(pool.submit(work, entry))
            records = [future.result() for future in futures]

    summary = summarise(
        records,
        judge_is_real=bool(judge.provider.is_real),
        rag_is_real=rag_is_real,
        expected_queries=expected_queries,
        low_score_threshold=low_score_threshold,
    )
    summary["same_model"] = same_model
    status, reason = verdict(
        summary,
        judge_is_real=bool(judge.provider.is_real),
        rag_is_real=rag_is_real,
        query_count=len(records),
        expected_queries=expected_queries,
    )
    finished_at = datetime.now(UTC)
    run_id = _run_id(started_at, rag_run_id)
    judge_latencies = [
        float(record[key]["latency_ms"])
        for record in records
        for key in METRIC_KEYS
        if record[key]["attempted"]
    ]
    return {
        "schema_version": 1,
        "experiment_name": "llm_as_a_judge",
        "status": status,
        "status_reason": reason,
        "generated_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "run_id": run_id,
        "parameters": {
            "queries": len(records),
            "expected_queries": expected_queries,
            "limit": limit,
            "max_concurrency": max_concurrency,
            "judge_temperature": judge.config.temperature,
            "judge_max_tokens": judge.config.max_tokens,
            "low_score_threshold": low_score_threshold,
            "prompt_versions": {
                "context_relevance": CONTEXT_RELEVANCE_VERSION,
                "faithfulness": FAITHFULNESS_VERSION,
            },
            "score_scale": {"min": 1, "max": 5},
            "judge_transport": dict(transport or {}),
            "rag_artifact": str(rag_artifact_path) if rag_artifact_path else None,
            "rag_status": rag.get("status"),
            "rag_generated_at": rag.get("generated_at"),
            "rag_provider_kind": _rag_parameter(rag, "provider_kind"),
            "rag_retrieval_mode": _rag_parameter(rag, "retrieval_mode"),
            "rag_model": rag_model,
        },
        "provider": {
            "kind": "real" if judge.provider.is_real else "mock",
            "name": judge.provider.name,
            "model": judge_model,
            "is_real": bool(judge.provider.is_real),
            "api_key_present": api_key_present,
            "real_judge_calls": bool(judge.provider.is_real) and bool(judge_latencies),
        },
        "rag": {
            "artifact_path": str(rag_artifact_path) if rag_artifact_path else None,
            "artifact_sha256": rag_artifact_sha256,
            "status": rag.get("status"),
            "status_reason": rag.get("status_reason"),
            "generated_at": rag.get("generated_at"),
            "provider": rag_provider,
            "model": rag_model,
            "is_real": rag_is_real,
            "retrieval_mode": _rag_parameter(rag, "retrieval_mode"),
            "queries": len(rag.get("per_query", [])),
        },
        "environment": dict(environment or {}),
        "dataset": dict(dataset or {}),
        "summary": summary,
        "metrics": _metrics(
            summary,
            records,
            judge_model=judge_model,
            judge_is_real=bool(judge.provider.is_real),
        ),
        "per_query": records,
        "limitations": _limitations(
            summary,
            judge_is_real=bool(judge.provider.is_real),
            rag_is_real=rag_is_real,
            same_model=same_model,
            expected_queries=expected_queries,
        ),
        "gold_usage": {
            "used_for_retrieval": False,
            "gold_response_used_for_scoring": False,
            "reason": (
                "Context Relevance and Faithfulness use only the question, the "
                "answer and the evidence actually supplied to the answer model."
            ),
        },
        "commands": list(commands),
        "integrity": {
            "rag_artifact_sha256": rag_artifact_sha256,
            "rag_status": rag.get("status"),
            "rag_generated_at": rag.get("generated_at"),
        },
    }


def summarise(
    records: Sequence[Mapping[str, Any]],
    *,
    judge_is_real: bool,
    rag_is_real: bool,
    expected_queries: int,
    low_score_threshold: int = 3,
) -> dict[str, Any]:
    """Aggregate both metrics from validated per-query records."""
    metric_summaries = {
        key: _metric_summary(records, key, low_score_threshold) for key in METRIC_KEYS
    }
    statuses = Counter(str(record.get("rag_status")) for record in records)
    both_success = sum(
        1
        for record in records
        if record["context_relevance"]["status"] == "success"
        and record["faithfulness"]["status"] == "success"
    )
    queries_with_failure = sum(
        1
        for record in records
        if record["context_relevance"]["status"] != "success"
        or record["faithfulness"]["status"] != "success"
    )
    rag_failures = sum(1 for record in records if record.get("rag_status") not in ANSWER_STATUSES)
    return {
        "queries": len(records),
        "expected_queries": expected_queries,
        "not_run_queries": max(expected_queries - len(records), 0),
        "rag_status_counts": dict(sorted(statuses.items())),
        "rag_failures": rag_failures,
        "both_metrics_success": both_success,
        "queries_with_judge_failure": queries_with_failure,
        "judge_is_real": judge_is_real,
        "rag_is_real": rag_is_real,
        "metrics": metric_summaries,
    }


def verdict(
    summary: Mapping[str, Any],
    *,
    judge_is_real: bool,
    rag_is_real: bool,
    query_count: int,
    expected_queries: int,
) -> tuple[str, str]:
    """Return PASS/PARTIAL/FAIL and a reason that cannot overclaim."""
    if query_count == 0:
        return "FAIL", "no query was judged"
    if int(summary["rag_failures"]) == query_count:
        return "FAIL", "every RAG query failed before it could be judged"
    valid = sum(
        int(summary["metrics"][key]["valid_scores"]) for key in METRIC_KEYS
    )
    if valid == 0:
        return "FAIL", "neither metric produced a valid score"
    if not judge_is_real:
        return "PARTIAL", "mock judge completed; scores are deterministic fixtures"
    if not rag_is_real:
        return "PARTIAL", "RAG answers are mock, so judged quality is not real model output"
    if query_count < expected_queries:
        return (
            "PARTIAL",
            f"real judge executed on {query_count}/{expected_queries} queries (smoke); "
            "this is not a full-set quality evaluation",
        )
    failures = int(summary["queries_with_judge_failure"])
    if failures:
        return (
            "PARTIAL",
            f"{failures}/{query_count} queries had at least one judge metric failure",
        )
    return "PASS", "all queries were judged with a real judge on real RAG answers"


def write_outputs(
    result: Mapping[str, Any],
    *,
    jsonl_path: Path,
    summary_path: Path,
    report_path: Path,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    """Write JSONL, summary JSON and Markdown, refusing to clobber by default.

    Raises:
        ArtifactExistsError: any target exists and ``overwrite`` is false.
    """
    ensure_writable((jsonl_path, summary_path, report_path), overwrite=overwrite)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=False)
        for record in result.get("per_query", [])
    ]
    jsonl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_judge_markdown(result), encoding="utf-8")
    return jsonl_path, summary_path, report_path


def render_judge_markdown(result: Mapping[str, Any]) -> str:
    """Render the standalone judge report."""
    parameters = result.get("parameters", {})
    summary = result.get("summary", {})
    provider = result.get("provider", {})
    rag = result.get("rag", {})
    lines = [
        "# LLM-as-a-Judge 评测",
        "",
        f"- Status: **{result.get('status')}**",
        f"- Reason: {result.get('status_reason')}",
        f"- Run ID: `{result.get('run_id')}`",
        f"- Generated: {result.get('generated_at')}",
        f"- Queries judged: **{parameters.get('queries')} / "
        f"{parameters.get('expected_queries')}**",
        f"- Not run: **{summary.get('not_run_queries', 0)}** "
        f"(NOT RUN, not scored and not counted as zero)",
        f"- RAG provider: `{rag.get('provider', {}).get('kind', '-')}` / "
        f"`{rag.get('provider', {}).get('name', '-')}` / model "
        f"`{rag.get('provider', {}).get('model', '-')}`",
        f"- Judge provider: `{provider.get('kind', '-')}` / "
        f"`{provider.get('name', '-')}` / model `{provider.get('model', '-')}`",
        f"- Same model as RAG: **{summary.get('same_model', False)}**",
        f"- Prompt versions: `{parameters.get('prompt_versions', {})}`",
        "- Score aggregation: only successful scores enter mean/median/distribution; "
        "failures are reported separately and never scored as 0.",
        "",
    ]
    if not provider.get("is_real"):
        lines.extend(
            [
                "> **MOCK JUDGE RUN.** Scores are deterministic fixtures, not a model's "
                "judgement. This report validates wiring, parsing, failure isolation and "
                "aggregation only.",
                "",
            ]
        )

    lines.extend(
        [
            "## 1. 指标汇总",
            "",
            "| Metric | Attempted | Success | Failed | Success rate | Valid n | Mean | "
            "Median | Low score |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
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
    lines.extend(["", "## 2. 分数分布（仅成功评分）", ""])
    lines.extend(
        [
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

    lines.extend(["", "## 3. 低分样例", ""])
    low_rows = [
        row
        for key in METRIC_KEYS
        for row in summary.get("metrics", {}).get(key, {}).get("low_score_samples", [])
    ]
    if low_rows:
        lines.extend(
            [
                "| Query | Metric | Score | Reason |",
                "|---|---|---:|---|",
            ]
        )
        for row in low_rows:
            lines.append(
                f"| {row.get('query_id')} | {row.get('metric')} | "
                f"{row.get('score')} | {_escape(str(row.get('reason', '')))[:160]} |"
            )
    else:
        lines.append("- 没有低于阈值的成功评分。")

    lines.extend(["", "## 4. 失败分类", ""])
    failure_rows: list[str] = []
    for key, label in (
        ("context_relevance", "Context Relevance"),
        ("faithfulness", "Faithfulness"),
    ):
        item = summary.get("metrics", {}).get(key, {})
        counts = item.get("failure_counts", {})
        for status, count in counts.get("status", {}).items():
            failure_rows.append(f"| {label} | {status} | {count} |")
        for category, count in counts.get("error_category", {}).items():
            failure_rows.append(f"| {label} | provider:{category} | {count} |")
    if failure_rows:
        lines.extend(["| Metric | Status / error | Count |", "|---|---|---:|"])
        lines.extend(failure_rows)
    else:
        lines.append("- 没有 Judge 失败。")

    lines.extend(["", "## 5. 逐查询结果", ""])
    lines.extend(
        [
            "| Query | RAG status | Context Relevance | Faithfulness | Unsupported claims |",
            "|---|---|---|---|---:|",
        ]
    )
    for record in result.get("per_query", []):
        cr = record["context_relevance"]
        fa = record["faithfulness"]
        lines.append(
            f"| {record['query_id']} | {record['rag_status']} | "
            f"{_metric_cell(cr)} | {_metric_cell(fa)} | "
            f"{len(fa.get('unsupported_claims', [])) if fa.get('status') == 'success' else '-'} |"
        )

    lines.extend(
        [
            "",
            "## 6. 限制",
            "",
            *[f"- {item}" for item in result.get("limitations", [])],
            "",
            "## 7. 复现",
            "",
            "```bash",
            *[str(item) for item in result.get("commands", [])],
            "```",
            "",
            "## 8. 数据安全",
            "",
            "- Judge 只读取 RAG artifact 中的问题、答案与证据；"
            "不修改数据库、`data/` 或 gold 标签。",
            "- API Key 只从环境/.env 读取；artifact、日志和本报告只记录 `api_key_present` 布尔值。",
            "- 失败评分不写入 0；均值/中位数/分布只使用成功评分。",
            "",
        ]
    )
    return "\n".join(lines)


def _judge_entry(
    entry: Mapping[str, Any],
    judge: LLMJudge,
    *,
    rag_run_id: str,
    rag_provider_name: str | None,
    rag_model: str | None,
    rag_is_real: bool,
) -> dict[str, Any]:
    """Build one validated per-query record, isolating unexpected failures."""
    query = str(entry.get("query") or "")
    rag_status = str(entry.get("status") or "unknown")
    answer = entry.get("answer")
    answer_text = answer if isinstance(answer, str) else ""
    try:
        contexts = _contexts(entry)
    except (TypeError, ValueError) as exc:
        failure = f"invalid RAG evidence: {exc}"
        return _record(
            entry,
            [],
            ContextRelevanceResult(
                status="rag_failure",
                reason=failure,
                prompt_version=CONTEXT_RELEVANCE_VERSION,
            ),
            FaithfulnessResult(
                status="rag_failure",
                reason=failure,
                prompt_version=FAITHFULNESS_VERSION,
            ),
            rag_run_id=rag_run_id,
            rag_provider_name=rag_provider_name,
            rag_model=rag_model,
            rag_is_real=rag_is_real,
        )

    if rag_status not in ANSWER_STATUSES or not query.strip():
        reason = (
            f"RAG status {rag_status!r} produced no judgeable answer"
            if rag_status not in ANSWER_STATUSES
            else "RAG record has no query text to judge"
        )
        context_result = ContextRelevanceResult(
            status="rag_failure",
            reason=reason,
            prompt_version=CONTEXT_RELEVANCE_VERSION,
        )
        faithfulness_result = FaithfulnessResult(
            status="rag_failure",
            reason=reason,
            prompt_version=FAITHFULNESS_VERSION,
        )
    else:
        try:
            context_result = judge.context_relevance(query, contexts)
        except Exception as exc:  # noqa: BLE001 - keep the batch running
            context_result = ContextRelevanceResult(
                status="provider_error",
                attempted=True,
                reason=f"unexpected judge error: {type(exc).__name__}",
                prompt_version=CONTEXT_RELEVANCE_VERSION,
            )
        try:
            faithfulness_result = judge.faithfulness(query, contexts, answer_text)
        except Exception as exc:  # noqa: BLE001 - keep the batch running
            faithfulness_result = FaithfulnessResult(
                status="provider_error",
                attempted=True,
                reason=f"unexpected judge error: {type(exc).__name__}",
                prompt_version=FAITHFULNESS_VERSION,
            )
    return _record(
        entry,
        contexts,
        context_result,
        faithfulness_result,
        rag_run_id=rag_run_id,
        rag_provider_name=rag_provider_name,
        rag_model=rag_model,
        rag_is_real=rag_is_real,
    )


def _record(
    entry: Mapping[str, Any],
    contexts: Sequence[JudgeContext],
    context_result: ContextRelevanceResult,
    faithfulness_result: FaithfulnessResult,
    *,
    rag_run_id: str,
    rag_provider_name: str | None,
    rag_model: str | None,
    rag_is_real: bool,
) -> dict[str, Any]:
    """Serialise a validated record with its evidence provenance."""
    query_id = str(entry.get("query_id") or "unknown")
    query = str(entry.get("query") or "")
    answer = entry.get("answer")
    answer_text = answer if isinstance(answer, str) else ""
    record = JudgeQueryRecord(
        query_id=query_id,
        query=query,
        rag_run_id=rag_run_id,
        rag_status=str(entry.get("status") or "unknown"),
        rag_provider=_optional_str(entry.get("provider")) or rag_provider_name,
        rag_model=_optional_str(entry.get("model")) or rag_model,
        rag_is_real=rag_is_real,
        answer_sha256=_sha256_text(answer_text),
        context_sha256=_optional_str(entry.get("context_sha256"))
        or _sha256_text("\n\n".join(context.content for context in contexts)),
        context_labels=[context.label for context in contexts],
        context_relevance=context_result,
        faithfulness=faithfulness_result,
    )
    return record.model_dump(mode="json")


def _contexts(entry: Mapping[str, Any]) -> list[JudgeContext]:
    """Extract validated ``E<n>`` contexts from one RAG record."""
    evidence = entry.get("evidence", [])
    if evidence is None:
        return []
    if not isinstance(evidence, list):
        raise TypeError("evidence must be a list")
    contexts: list[JudgeContext] = []
    for index, item in enumerate(evidence, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"evidence {index} is not an object")
        label = item.get("label")
        content = item.get("content")
        if not isinstance(label, str) or not isinstance(content, str) or not content.strip():
            raise ValueError(f"evidence {index} lacks a label or content")
        title = item.get("title")
        contexts.append(
            JudgeContext(
                label=label,
                title=title if isinstance(title, str) else None,
                content=content,
            )
        )
    return contexts


def _metric_summary(
    records: Sequence[Mapping[str, Any]],
    key: str,
    low_score_threshold: int,
) -> dict[str, Any]:
    """Aggregate one metric without treating a failure as a score."""
    results = [record[key] for record in records]
    attempted = sum(1 for result in results if result.get("attempted"))
    successes = [result for result in results if result.get("status") == "success"]
    scores = [int(result["score"]) for result in successes]
    failed = attempted - len(successes)
    distribution = {str(score): 0 for score in range(1, 6)}
    for score in scores:
        distribution[str(score)] += 1
    low_samples = [
        {
            "query_id": record["query_id"],
            "metric": key,
            "score": result["score"],
            "reason": result.get("reason", "")[:300],
        }
        for record, result in zip(records, results, strict=True)
        if result.get("status") == "success" and int(result["score"]) < low_score_threshold
    ]
    status_counts = Counter(
        str(result.get("status"))
        for result in results
        if result.get("status") != "success"
    )
    error_categories = Counter(
        str(result.get("error", {}).get("category"))
        for result in results
        if result.get("status") == "provider_error" and result.get("error")
    )
    return {
        "attempted": attempted,
        "not_attempted": len(results) - attempted,
        "success": len(successes),
        "failed": failed,
        "success_rate": len(successes) / attempted if attempted else None,
        "failure_rate": failed / attempted if attempted else None,
        "valid_scores": len(scores),
        "mean": statistics.fmean(scores) if scores else None,
        "median": statistics.median(scores) if scores else None,
        "distribution": distribution,
        "low_score_threshold": low_score_threshold,
        "low_score_count": len(low_samples),
        "low_score_samples": low_samples,
        "failure_counts": {
            "status": dict(sorted(status_counts.items())),
            "error_category": dict(sorted(error_categories.items())),
        },
    }


def _metrics(
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    judge_model: str,
    judge_is_real: bool,
) -> dict[str, Any]:
    """Return the metric block with explicit definitions and exclusions."""
    latencies = [
        float(record[key]["latency_ms"])
        for record in records
        for key in METRIC_KEYS
        if record[key].get("attempted")
    ]
    return {
        "context_relevance": summary["metrics"]["context_relevance"],
        "faithfulness": summary["metrics"]["faithfulness"],
        "llm_as_a_judge": {
            "executed": True,
            "real_judge_calls": judge_is_real and bool(latencies),
            "judge_model": judge_model,
            "score_scale": {"min": 1, "max": 5},
            "aggregation": (
                "mean, median and distribution use only status=success; "
                "failed or unattempted metrics are excluded and counted separately"
            ),
            "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
            "latency_samples": len(latencies),
        },
    }


def _limitations(
    summary: Mapping[str, Any],
    *,
    judge_is_real: bool,
    rag_is_real: bool,
    same_model: bool,
    expected_queries: int,
) -> list[str]:
    """List what the judge run does not establish."""
    items = [
        "Judge scores are model-generated judgements, not ground truth; no human "
        "review or inter-judge agreement was performed.",
        "Context Relevance scores the supplied evidence only; it is not retrieval "
        "Recall and does not show that the right document was retrieved.",
        "Faithfulness uses only the evidence actually sent to the answer model. A claim "
        "that is true but absent from that evidence is still unsupported here.",
        "The judge may be sensitive to prompt order, context length and language; "
        "repeat-run variance was not measured.",
        "The 30 queries are English; Chinese and mixed-language judging are not evaluated.",
    ]
    if summary.get("queries", 0) < expected_queries:
        items.insert(
            0,
            f"Only {summary.get('queries')} of {expected_queries} queries were judged; "
            "this smoke run does not describe the full evaluation set.",
        )
    if same_model:
        items.append(
            "The judge reuses the same model id as the answer generator; self-evaluation "
            "bias is unquantified and the scores must not be treated as independent."
        )
    if not judge_is_real:
        items.insert(
            0,
            "The judge is a deterministic mock. Its scores validate the pipeline, not "
            "the quality of a real model's judgement.",
        )
    if not rag_is_real:
        items.insert(
            0,
            "The RAG answers are mock output, so judged answer quality is not real model output.",
        )
    return items


def _rag_parameter(rag: Mapping[str, Any], key: str) -> Any:
    """Read one RAG parameter without assuming the artifact shape."""
    parameters = rag.get("parameters", {})
    return parameters.get(key) if isinstance(parameters, dict) else None


def _metric_cell(result: Mapping[str, Any]) -> str:
    """Render one metric cell as ``status (score)`` or just the status."""
    status = result.get("status", "-")
    score = result.get("score")
    return f"{status} ({score})" if score is not None else str(status)


def _optional_str(value: object) -> str | None:
    """Return a non-empty string or ``None``."""
    return value if isinstance(value, str) and value else None


def _sha256_text(text: str) -> str:
    """Return the SHA256 of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_id(moment: datetime, rag_run_id: str) -> str:
    """Build a short, traceable run identifier."""
    digest = hashlib.sha256(f"{moment.isoformat()}:{rag_run_id}".encode()).hexdigest()[:8]
    return f"judge-{moment.strftime('%Y%m%dT%H%M%SZ')}-{digest}"


def _fmt(value: object, digits: int = 4) -> str:
    """Format an optional number, keeping ``None`` visible as ``-``."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _escape(value: str) -> str:
    """Escape pipes and newlines for a Markdown table cell."""
    return value.replace("|", "\\|").replace("\n", " ")
