"""Phase 8 RAG end-to-end evaluation.

This module runs the accepted live retrieval -> RRF -> reranker chain, assembles
the real evidence context, calls an injected provider, validates citations and
records one raw result per query.

What it measures and what it does not
-------------------------------------

Measured automatically:

* pipeline status, including retrieval/rerank/context/provider failures;
* evidence Recall@K, first-gold rank and MRR of the final evidence set;
* citation counts and citation validity against the evidence actually supplied
  to the model;
* end-to-end, retrieval and LLM latency.

Not measured automatically:

* answer correctness, Faithfulness or Answer Relevance. The dataset has
  ``gold_response`` text but no answer-quality rubric and no human score, and no
  LLM judge was run. Those fields are reported as ``None`` with a reason, never
  estimated from Recall or MRR.

Mock results are labelled ``provider.kind="mock"`` and never presented as real
model output.

Implemented in: Phase 8 (RAG end-to-end).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, Settings, get_settings
from src.evaluation.dense_benchmark import BenchmarkError, EvalQuery
from src.evaluation.metrics import percentile
from src.rag.pipeline import RAGPipeline
from src.schema.rag import RAGAnswer

DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "eval_queries.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "rag_benchmark.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "rag_benchmark.md"

ANSWER_STATUSES = frozenset({"ok", "insufficient_evidence"})
"""Statuses that count as a structured answer, including a legitimate refusal."""


@dataclass(frozen=True)
class RAGBenchmarkConfig:
    """Fixed parameters for one RAG evaluation run."""

    top_k: int = 10
    candidates: int = 10
    max_context_chars: int = 8000
    max_chunk_chars: int = 2000
    provider_kind: str = "mock"
    retrieval_mode: str = "live"


def load_raw_queries(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Read the raw query records, including ``gold_response``, without editing them."""
    path = path or DEFAULT_EVAL_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkError(f"cannot read eval queries at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise BenchmarkError(f"{path} must hold a non-empty JSON array")
    records: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise BenchmarkError(f"eval query #{index} is not an object")
        query_id = entry.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise BenchmarkError(f"eval query #{index} has no query_id")
        if query_id in records:
            raise BenchmarkError(f"duplicate eval query_id {query_id!r}")
        records[query_id] = entry
    return records


def run_rag_benchmark(
    pipeline: RAGPipeline,
    queries: Sequence[EvalQuery],
    *,
    raw_queries: Mapping[str, Mapping[str, Any]],
    config: RAGBenchmarkConfig,
    environment: Mapping[str, Any],
    dataset: Mapping[str, Any],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the pipeline over the selected queries and return the raw artifact."""
    if not queries:
        raise BenchmarkError("at least one eval query is required")
    if config.top_k > config.candidates:
        raise BenchmarkError("top_k must not exceed candidates")

    emit = progress or (lambda _message: None)
    started_at = datetime.now(UTC)
    records: list[dict[str, Any]] = []
    process_started = time.perf_counter()

    for index, query in enumerate(queries, start=1):
        emit(f"[{index}/{len(queries)}] {query.query_id}")
        record_started = time.perf_counter()
        answer = pipeline.answer(query.query, query_id=query.query_id)
        wall_ms = (time.perf_counter() - record_started) * 1000.0
        raw = raw_queries.get(query.query_id, {})
        records.append(
            _query_record(
                query=query,
                answer=answer,
                raw=raw,
                wall_ms=wall_ms,
                provider_is_real=bool(getattr(pipeline.provider, "is_real", False)),
            )
        )

    total_ms = (time.perf_counter() - process_started) * 1000.0
    summary = _summarise(records)
    real = bool(getattr(pipeline.provider, "is_real", False))
    provider_attempted = any(record["provider_attempted"] for record in records)
    status, reason = _verdict(records, summary, real_llm=real)
    settings = get_settings()

    return {
        "schema_version": 1,
        "stage": "8",
        "status": status,
        "status_reason": reason,
        "generated_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "parameters": {
            "top_k": config.top_k,
            "candidates": config.candidates,
            "max_context_chars": config.max_context_chars,
            "max_chunk_chars": config.max_chunk_chars,
            "provider_kind": config.provider_kind,
            "retrieval_mode": config.retrieval_mode,
            "queries": len(queries),
            "all_dataset_queries": len(raw_queries),
            "prompt_version": pipeline.config.prompt_version,
            "prompt_sha256_by_query": "recorded per query",
            "temperature": pipeline.config.temperature,
            "max_tokens": pipeline.config.max_tokens,
        },
        "provider": {
            "kind": config.provider_kind,
            "name": pipeline.provider.name,
            "is_real": real,
            "model": pipeline.provider.model,
            "endpoint": (
                getattr(pipeline.provider, "endpoint", None) if real else None
            ),
            "api_key_present": _provider_key_present(pipeline),
            "real_llm_calls": real and provider_attempted,
            "real_llm_evaluation": (
                "executed" if real and provider_attempted else "not_run"
            ),
            "llm_judge": {
                "executed": False,
                "reason": "No judge model, rubric or budget was configured or run.",
            },
        },
        "environment": dict(environment),
        "dataset": dict(dataset),
        "summary": summary,
        "metrics": _metrics(summary, records),
        "per_query": records,
        "timing": {
            "evaluation_loop_ms": total_ms,
            "timing_scope": (
                "wall-clock per query around RAGPipeline.answer; the first query "
                "includes lazy embedding and reranker model loading"
            ),
            "percentile": "nearest rank: ceil(p/100 * n), clamped to [1, n]",
            "sample_count": len(records),
        },
        "gold_usage": {
            "used_for_retrieval": False,
            "used_for_evidence_metrics_only": True,
            "gold_response_used_for_scoring": False,
            "reason": (
                "gold_response is stored for human review only; there is no answer "
                "rubric and no automatic correctness judgement."
            ),
        },
        "integrity": {
            "data_hashes_before": dict(dataset.get("hashes", {})),
            "database_counts_before": {
                key: environment.get(key)
                for key in ("documents", "chunks", "distinct_chunk_docs")
            },
        },
        "limitations": _limitations(real, settings, config),
        "retrieval_provenance": (
            "live Dense + Sparse OR -> RRF -> reranker"
            if config.retrieval_mode == "live"
            else "accepted Phase 6B RRF candidate identities replayed; "
            "live PostgreSQL text and live cross-encoder reranking"
        ),
    }


def _query_record(
    *,
    query: EvalQuery,
    answer: RAGAnswer,
    raw: Mapping[str, Any],
    wall_ms: float,
    provider_is_real: bool,
) -> dict[str, Any]:
    """Build one raw per-query record with observable evidence and citations."""
    evidence_docs = list(dict.fromkeys(item.doc_id for item in answer.evidence))
    gold = list(query.relevant_doc_ids)
    hits = [doc_id for doc_id in evidence_docs if doc_id in set(gold)]
    first_gold = next(
        (rank for rank, doc_id in enumerate(evidence_docs, start=1) if doc_id in set(gold)),
        None,
    )
    recall = len(hits) / len(gold) if gold else 0.0
    citation_docs = list(dict.fromkeys(item.doc_id for item in answer.citations))
    return {
        "query_id": query.query_id,
        "query": query.query,
        "gold_doc_ids": gold,
        "gold_response": raw.get("gold_response"),
        "gold_response_used_for_scoring": False,
        "status": answer.status,
        "answer": answer.answer,
        "answer_chars": len(answer.answer),
        "citations": [item.model_dump(mode="json") for item in answer.citations],
        "citation_labels": [item.label for item in answer.citations],
        "citation_doc_ids": citation_docs,
        "invalid_citations": list(answer.invalid_citations),
        "citation_status": answer.citation_status,
        "insufficient_evidence": answer.insufficient_evidence,
        "provider": answer.provider,
        "model": answer.model,
        "usage": answer.usage.model_dump(mode="json") if answer.usage else None,
        "prompt_version": answer.prompt_version,
        "prompt_sha256": answer.prompt_sha256,
        "context_sha256": answer.context_sha256,
        "context_text": answer.context_text,
        "context_truncated": answer.context_truncated,
        "dropped_evidence": answer.dropped_evidence,
        "evidence": [item.model_dump(mode="json") for item in answer.evidence],
        "evidence_labels": [item.label for item in answer.evidence],
        "evidence_chunk_ids": [item.chunk_id for item in answer.evidence],
        "evidence_doc_ids": evidence_docs,
        "retrieval": answer.retrieval,
        "timings_ms": dict(answer.timings_ms),
        "wall_ms": wall_ms,
        "provider_attempted": answer.provider_attempted,
        "raw_response": answer.raw_response,
        "error": answer.error.model_dump(mode="json") if answer.error else None,
        "real_llm": provider_is_real,
        "evidence_recall": recall,
        "gold_hit_count": len(hits),
        "first_gold_rank": first_gold,
        "reciprocal_rank": (1.0 / first_gold) if first_gold else 0.0,
        "hit_at_k": bool(hits),
        "review_note": (
            "No automatic correctness or faithfulness judgement. Compare answer and "
            "citations with the stored evidence and gold_response manually."
        ),
    }


def _summarise(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate statuses and evidence metrics from the raw per-query records."""
    total = len(records)
    statuses = Counter(str(record["status"]) for record in records)
    answered = sum(statuses[status] for status in ANSWER_STATUSES)
    provider_attempted = sum(1 for record in records if record["provider_attempted"])
    provider_success = sum(
        1
        for record in records
        if record["provider_attempted"] and record["status"] in ANSWER_STATUSES
    )
    generation_failures = sum(
        1
        for record in records
        if record["provider_attempted"] and record["status"] not in ANSWER_STATUSES
    )
    short_circuit_insufficient = sum(
        1
        for record in records
        if record["status"] == "insufficient_evidence" and not record["provider_attempted"]
    )
    insufficient = statuses["insufficient_evidence"]
    valid_citations = sum(len(record["citations"]) for record in records)
    invalid_citations = sum(len(record["invalid_citations"]) for record in records)
    queries_with_citations = sum(1 for record in records if record["citations"])
    queries_with_invalid = sum(1 for record in records if record["invalid_citations"])
    answers_without_citations = sum(
        1 for record in records if record["status"] == "ok" and not record["citations"]
    )
    recall_values = [float(record["evidence_recall"]) for record in records]
    rr_values = [float(record["reciprocal_rank"]) for record in records]
    total_gold = sum(len(record["gold_doc_ids"]) for record in records)
    total_gold_hits = sum(int(record["gold_hit_count"]) for record in records)
    return {
        "queries": total,
        "status_counts": dict(sorted(statuses.items())),
        "answered": answered,
        "end_to_end_success_rate": answered / total if total else None,
        "provider_attempted": provider_attempted,
        "provider_success": provider_success,
        "generation_failures": generation_failures,
        "generation_failure_rate": (
            generation_failures / provider_attempted if provider_attempted else None
        ),
        "pipeline_failures": total - answered,
        "pipeline_failure_rate": (total - answered) / total if total else None,
        "insufficient_evidence": insufficient,
        "insufficient_evidence_rate": insufficient / total if total else None,
        "short_circuit_insufficient": short_circuit_insufficient,
        "no_retrieval_or_context_insufficient": short_circuit_insufficient,
        "retrieval_errors": statuses["retrieval_error"],
        "rerank_errors": statuses["rerank_error"],
        "evidence_errors": statuses["evidence_error"],
        "llm_timeouts": statuses["llm_timeout"],
        "llm_rate_limited": statuses["llm_rate_limited"],
        "llm_network_errors": statuses["llm_network_error"],
        "llm_server_errors": statuses["llm_server_error"],
        "llm_auth_errors": statuses["llm_auth_error"],
        "llm_invalid_responses": statuses["llm_invalid_response"],
        "citations_valid": valid_citations,
        "citations_invalid": invalid_citations,
        "citation_validity_rate": (
            valid_citations / (valid_citations + invalid_citations)
            if valid_citations + invalid_citations
            else None
        ),
        "queries_with_citations": queries_with_citations,
        "queries_with_invalid_citations": queries_with_invalid,
        "answers_without_citations": answers_without_citations,
        "evidence_recall_macro": (
            sum(recall_values) / total if total else None
        ),
        "evidence_recall_micro": (
            total_gold_hits / total_gold if total_gold else None
        ),
        "evidence_mrr": sum(rr_values) / total if total else None,
        "evidence_hit_at_k": sum(1 for record in records if record["hit_at_k"]),
        "queries_with_all_gold_in_evidence": sum(
            1
            for record in records
            if record["gold_doc_ids"]
            and record["gold_hit_count"] == len(record["gold_doc_ids"])
        ),
        "evidence_samples": total,
    }


def _metrics(summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the reported metrics with explicit definitions and exclusions."""
    total_samples = [float(record["timings_ms"].get("total_ms", 0.0)) for record in records]
    retrieval_samples = [
        float(record["timings_ms"]["retrieval_ms"])
        for record in records
        if "retrieval_ms" in record["timings_ms"]
    ]
    llm_samples = [
        float(record["timings_ms"]["llm_ms"])
        for record in records
        if "llm_ms" in record["timings_ms"]
    ]
    return {
        "end_to_end_success_rate": {
            "value": summary["end_to_end_success_rate"],
            "numerator": summary["answered"],
            "denominator": summary["queries"],
            "definition": "status is ok or insufficient_evidence",
        },
        "generation_failure_rate": {
            "value": summary["generation_failure_rate"],
            "numerator": summary["generation_failures"],
            "denominator": summary["provider_attempted"],
            "definition": "provider was called and no structured answer was produced",
        },
        "insufficient_evidence": {
            "value": summary["insufficient_evidence_rate"],
            "numerator": summary["insufficient_evidence"],
            "denominator": summary["queries"],
            "definition": (
                "status insufficient_evidence; includes no-retrieval/empty-context "
                "short circuits and model-declared refusals"
            ),
        },
        "citation_validity": {
            "value": summary["citation_validity_rate"],
            "valid": summary["citations_valid"],
            "invalid": summary["citations_invalid"],
            "definition": (
                "valid labels map to E<n> evidence actually supplied; invalid labels "
                "are retained in invalid_citations but never presented as citations"
            ),
        },
        "evidence_retrieval": {
            "recall_macro": summary["evidence_recall_macro"],
            "recall_micro": summary["evidence_recall_micro"],
            "mrr": summary["evidence_mrr"],
            "hit_at_k": summary["evidence_hit_at_k"],
            "samples": summary["evidence_samples"],
            "definition": (
                "document-level metrics over the final evidence set; this is a "
                "retrieval metric, not answer accuracy"
            ),
        },
        "latency_ms": {
            "total": _latency(total_samples),
            "retrieval": _latency(retrieval_samples),
            "llm": _latency(llm_samples),
            "timing_scope": "wall-clock; nearest-rank percentiles; no warm-up exclusion",
        },
        "answer_correctness": {
            "value": None,
            "status": "not_computed",
            "reason": "No answer rubric or human correctness labels were run.",
        },
        "faithfulness": {
            "value": None,
            "status": "not_computed",
            "reason": "No judge or human faithfulness annotation was run.",
        },
        "answer_relevance": {
            "value": None,
            "status": "not_computed",
            "reason": "No judge or human relevance annotation was run.",
        },
        "llm_as_a_judge": {
            "executed": False,
            "reason": "No judge model, rubric, cost budget or reproducibility run.",
        },
    }


def _latency(samples: Sequence[float]) -> dict[str, Any]:
    """Return P50/P95 and the sample count, or nulls when no sample exists."""
    if not samples:
        return {"p50": None, "p95": None, "samples": 0}
    return {
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "samples": len(samples),
    }


def _verdict(
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    real_llm: bool,
) -> tuple[str, str]:
    """Return PASS/PARTIAL/FAIL and a reason that does not overclaim."""
    if not records:
        return "FAIL", "no queries were evaluated"
    if summary["answered"] == 0:
        return "FAIL", "no query produced a structured answer"
    if summary["provider_attempted"] > 0 and summary["provider_success"] == 0:
        return "FAIL", "every provider call failed; no answer was generated"
    if summary["answered"] < len(records):
        return (
            "PARTIAL",
            f"{summary['answered']}/{len(records)} queries produced a structured answer",
        )
    if not real_llm:
        return (
            "PARTIAL",
            "offline/mock pipeline completed; real LLM evaluation was not run",
        )
    return "PASS", "all queries completed with a real provider"


def _provider_key_present(pipeline: RAGPipeline) -> bool:
    """Return key presence only; never the key or a masked fragment."""
    config = getattr(pipeline.provider, "config", None)
    if config is None:
        return False
    return bool(getattr(config, "api_key_present", False))


def _limitations(real: bool, settings: Settings, config: RAGBenchmarkConfig) -> list[str]:
    """List what this run does not establish."""
    items = [
        "The 30 queries are English; Chinese and mixed-language generation are not evaluated.",
    ]
    if config.retrieval_mode != "live":
        items.append(
            "Retrieval is historical: Dense, Sparse and RRF were not rerun. The accepted "
            "Phase 6B RRF Top-10 identities and ranks were replayed, then joined to live "
            "PostgreSQL text and reranked live. Evidence metrics therefore measure the "
            "replayed pool, not a new retrieval run."
        )
    items += [
        "Answer correctness, Faithfulness and Answer Relevance are not computed: no rubric, "
        "human score or executed judge exists.",
        "The stored gold_response is reference text for human review, not a scored label.",
        "Latency is a single-process host run with no concurrency, GPU or cross-machine claim.",
        "The first query includes lazy model loading (and, in live mode, embedding model "
        "loading); it is not a steady-state latency sample.",
    ]
    if not real:
        items.insert(
            0,
            "The provider is a deterministic mock. Its answers begin with MOCK and are not "
            "evidence that a real LLM produces correct or faithful output.",
        )
    if settings.reranker_local_files_only:
        items.append(
            "The reranker runs with local_files_only=true; a fresh machine without the "
            "cached model must pass --allow-network or populate the cache first."
        )
    if config.max_context_chars < 8000:
        items.append(
            f"The context budget was reduced to {config.max_context_chars} characters for "
            "this run; truncation flags are recorded per query."
        )
    return items


def write_outputs(
    result: Mapping[str, Any],
    output: Path = DEFAULT_OUTPUT,
    report: Path = DEFAULT_REPORT,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write the raw JSON and Markdown report, refusing to clobber by default."""
    for path in (output, report):
        if path.exists() and not overwrite:
            raise BenchmarkError(
                f"{path} already exists; choose another path or pass --overwrite"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    report.write_text(render_markdown(result), encoding="utf-8")
    return output, report


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render the RAG evaluation artifact as a Markdown report."""
    parameters = result["parameters"]
    summary = result["summary"]
    metrics = result["metrics"]
    provider = result["provider"]
    lines = [
        "# RAG 端到端问答评测（阶段 ⑧）",
        "",
        f"- Status: **{result['status']}**",
        f"- Reason: {result['status_reason']}",
        f"- Generated: {result['generated_at']}",
        f"- Queries: {parameters['queries']}",
        f"- Provider: `{provider['kind']}` / `{provider['name']}` / model `{provider['model']}`",
        f"- Real LLM calls: **{provider['real_llm_calls']}**",
        f"- Retrieval mode: **{parameters['retrieval_mode']}**",
        f"- Retrieval provenance: {result['retrieval_provenance']}",
        f"- Final evidence / candidates: {parameters['top_k']} / {parameters['candidates']}",
        f"- Prompt: `{parameters['prompt_version']}` (hash per query)",
        "",
    ]
    if not provider["is_real"]:
        lines.extend(
            [
                "> **MOCK RUN.** The provider is deterministic and offline. Its answer text "
                "begins with `MOCK`. This report validates pipeline wiring, evidence transfer, "
                "citation mapping and failure handling; it is **not** a real model-quality "
                "result.",
                "",
            ]
        )

    lines.extend(
        [
            "## 1. 结论摘要",
            "",
            "| 指标 | 值 | 定义 / 样本 |",
            "|---|---|---|",
            _metric_row(
                "端到端成功率",
                metrics["end_to_end_success_rate"],
                "status ∈ {ok, insufficient_evidence}",
            ),
            _metric_row(
                "生成失败率",
                metrics["generation_failure_rate"],
                "Provider 已调用但未产出结构化答案",
            ),
            _metric_row(
                "证据不足率",
                metrics["insufficient_evidence"],
                "无检索结果/空上下文短路 + 模型声明证据不足",
            ),
            _metric_row(
                "引用有效率",
                metrics["citation_validity"],
                "有效 E 标签 / 全部返回标签",
            ),
            _plain_row(
                "证据 Recall macro",
                _fmt(metrics["evidence_retrieval"]["recall_macro"]),
                f"final evidence, {summary['evidence_samples']} queries",
            ),
            _plain_row(
                "证据 MRR",
                _fmt(metrics["evidence_retrieval"]["mrr"]),
                "final evidence, retrieval metric only",
            ),
            _plain_row(
                "上下文截断查询数",
                str(sum(1 for record in result["per_query"] if record["context_truncated"])),
                f"max_context_chars={parameters['max_context_chars']}",
            ),
            "",
            "| 延迟（ms） | P50 | P95 | 样本数 | 口径 |",
            "|---|---|---|---|---|",
            *_latency_rows(metrics["latency_ms"]),
            "",
            "**未计算：** answer correctness、Faithfulness、Answer Relevance。数据集只有 "
            "`gold_response` 参考文本，没有答案评分 rubric 或人工评分；本阶段也没有运行 "
            "LLM-as-a-Judge。Recall/MRR 是检索指标，不能替代生成质量。",
            "",
            "## 2. 固定条件与评测口径",
            "",
            f"- top_k / candidates: {parameters['top_k']} / {parameters['candidates']}",
            f"- max_context_chars / max_chunk_chars: {parameters['max_context_chars']} / "
            f"{parameters['max_chunk_chars']}",
            f"- retrieval_mode: `{parameters['retrieval_mode']}` — "
            f"{result['retrieval_provenance']}",
            f"- temperature / max_tokens: {parameters['temperature']} / {parameters['max_tokens']}",
            f"- prompt version: `{parameters['prompt_version']}`",
            "- 检索链路：Dense + Sparse OR → RRF(k=60) → Cross-Encoder → 最终 Top-K。",
            "- 证据文本来自 PostgreSQL `chunks.content`；未使用占位文本或伪造文档。",
            "- 引用标签只允许模型引用上下文中实际出现的 `E<n>`；其他标签进入 "
            "`invalid_citations`，不会出现在 `citations` 中。",
            "- 证据 Recall/MRR 以 `gold_doc_ids` 为分母，按最终证据顺序去重；这是检索指标。",
            "- 逐查询原始结果、上下文、证据、引用、耗时与错误全部保存在 JSON。",
            "",
            "## 3. Provider、Prompt 与安全",
            "",
            "| 项 | 值 |",
            "|---|---|",
            f"| provider kind | `{provider['kind']}` |",
            f"| provider name | `{provider['name']}` |",
            f"| is_real | {provider['is_real']} |",
            f"| model | `{provider['model']}` |",
            f"| real_llm_calls | {provider['real_llm_calls']} |",
            f"| real_llm_evaluation | {provider['real_llm_evaluation']} |",
            f"| API key present | {provider['api_key_present']} (value never recorded) |",
            f"| LLM judge | {provider['llm_judge']['executed']} |",
            "",
            "- Prompt 要求只依据证据回答、允许“无法从现有材料确认”、只返回一个 JSON 对象。",
            "- 上下文有总字符上限与单块上限；截断状态逐查询记录。",
            "- 缺失 API Key 时在检索前失败，错误为 `config_error`；不会回退到 mock。",
            "- 日志不包含 API Key、请求头、Prompt 或模型响应正文。",
            "",
            "## 4. 逐查询结果",
            "",
            "| query | status | evidence | citations | invalid | gold recall | "
            "first gold | total ms |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for record in result["per_query"]:
        lines.append(
            f"| {record['query_id']} | {record['status']} | {len(record['evidence'])} | "
            f"{len(record['citations'])} | {len(record['invalid_citations'])} | "
            f"{_fmt(record['evidence_recall'])} | "
            f"{record['first_gold_rank'] if record['first_gold_rank'] is not None else '-'} | "
            f"{_fmt(record['timings_ms'].get('total_ms'))} |"
        )

    lines.extend(
        [
            "",
            "## 5. 状态与失败分类",
            "",
            "| status | count |",
            "|---|---|",
        ]
    )
    for status, count in summary["status_counts"].items():
        lines.append(f"| {status} | {count} |")

    lines.extend(
        [
            "",
            "## 6. 引用校验",
            "",
            f"- 有效引用标签：{summary['citations_valid']}",
            f"- 无效引用标签：{summary['citations_invalid']}",
            f"- 含引用的查询：{summary['queries_with_citations']} / {summary['queries']}",
            f"- 含无效引用的查询：{summary['queries_with_invalid_citations']}",
            f"- `status=ok` 但无引用的查询：{summary['answers_without_citations']}",
            "",
            "无效标签没有被静默丢弃：它们保存在逐查询的 `invalid_citations` 中，且不计入 "
            "`citations`。本阶段只验证“引用是否指向实际提供的证据”，不验证引用是否真正支持"
            "答案中的每一句话；后者需要人工或已执行的 Judge。",
            "",
            "## 7. 证据与检索",
            "",
            f"- 证据 Recall macro / micro：{_fmt(summary['evidence_recall_macro'])} / "
            f"{_fmt(summary['evidence_recall_micro'])}",
            f"- 证据 MRR：{_fmt(summary['evidence_mrr'])}",
            f"- 至少命中一个 gold 的查询：{summary['evidence_hit_at_k']} / "
            f"{summary['evidence_samples']}",
            f"- 全部 gold 都进入最终证据的查询："
            f"{summary['queries_with_all_gold_in_evidence']} / {summary['evidence_samples']}",
            "",
            "这些是最终证据集合上的检索指标，不是生成答案的准确率。",
            "",
            "## 8. 未验证事项",
            "",
        ]
    )
    for item in result["limitations"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "- Docker/容器路径未验证；本阶段在宿主 Python 运行。",
            "- 完整 pytest 未重跑（存在已知原生崩溃风险）；只运行模块级测试。",
            "- 依赖固定未重新安装核验；宿主版本可能仍与 `pyproject.toml` 漂移。",
            "- Embedding 与 Reranker 同进程共存的内存行为按本次实测记录，不推广到其他机器。",
            "",
            "## 9. 复现",
            "",
            "本次正式评测使用的命令（`retrieval_mode="
            + str(result["parameters"].get("retrieval_mode"))
            + "`）：",
            "",
            "```bash",
            "APEX_PG_HOST=127.0.0.1 python scripts/benchmark_rag.py --provider mock "
            "--retrieval-mode historical",
            "```",
            "",
            "`APEX_PG_HOST=127.0.0.1` 是本机绕过：`localhost` 同时解析到 IPv6/IPv4 时，连接池"
            "会在每个地址上耗满 `connect_timeout`。它不是代码默认值。若不加 `--retrieval-mode "
            "historical`，脚本默认走 live 链路（Dense + Sparse + RRF + Cross-Encoder 同进程）。"
            "该链路在本机 reranker 加载阶段触发原生段错误（exit 139），因此本次改用 historical "
            "重放；这不是断言失败，也不是检索结果失败。",
            "",
            "Mock live 链路（默认，不调用真实 API）：",
            "",
            "```bash",
            "python scripts/benchmark_rag.py --provider mock",
            "```",
            "",
            "真实 Provider（需要 `APEX_LLM_API_KEY` 或 `DEEPSEEK_API_KEY`）：",
            "",
            "```bash",
            "python scripts/benchmark_rag.py --provider real --limit 2 --no-write",
            "python scripts/benchmark_rag.py --provider real",
            "```",
            "",
            "冒烟测试（不写报告）：",
            "",
            "```bash",
            "python scripts/benchmark_rag.py --provider mock --limit 2 --no-write",
            "```",
            "",
            "## 10. 数据安全",
            "",
            "本评测只读知识库并调用生成 Provider：不写入数据库、不修改 "
            "`data/kb_documents.json`、`data/eval_queries.json` 或 gold 标签，不把生成答案"
            "写回知识库。原始结果见同目录 JSON。",
            "",
        ]
    )
    return "\n".join(lines)


def _metric_row(label: str, payload: Mapping[str, Any], definition: str) -> str:
    """Render one metric row with numerator/denominator when available."""
    value = payload.get("value")
    numerator = payload.get("numerator")
    denominator = payload.get("denominator")
    rendered = _fmt(value)
    if numerator is not None and denominator is not None:
        rendered = f"{rendered} ({numerator}/{denominator})"
    elif "valid" in payload and "invalid" in payload:
        rendered = f"{rendered} ({payload['valid']}/{payload['valid'] + payload['invalid']})"
    return f"| {label} | {rendered} | {definition} |"


def _plain_row(label: str, value: str, definition: str) -> str:
    """Render a simple metric row."""
    return f"| {label} | {value} | {definition} |"


def _latency_rows(payload: Mapping[str, Any]) -> list[str]:
    """Render latency rows for total, retrieval and LLM scopes."""
    rows: list[str] = []
    for key in ("total", "retrieval", "llm"):
        item = payload[key]
        rows.append(
            f"| {key} | {_fmt(item['p50'])} | {_fmt(item['p95'])} | "
            f"{item['samples']} | {payload['timing_scope']} |"
        )
    return rows


def _fmt(value: object, digits: int = 4) -> str:
    """Format an optional numeric value."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


