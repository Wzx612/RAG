"""Read-only acceptance dashboard over the delivered evaluation artifacts.

The dashboard is a projection, not an evaluator: every number comes from an
artifact on disk, and the overall status comes from
:func:`src.evaluation.report.report_status`. Missing optional artifacts are
shown as missing; missing required artifacts downgrade the status; malformed
artifacts fail the whole response instead of yielding partial numbers.

The router deliberately reads no database, loads no model and registers no
write method. Artifact paths are fixed by :class:`AcceptanceConfig`, so a
request cannot ask the service to open an arbitrary file.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, JSONResponse, Response

from src.config import PROJECT_ROOT
from src.evaluation.report import (
    OPTIONAL_ARTIFACTS,
    REQUIRED_ARTIFACTS,
    ReportError,
    load_artifacts,
    report_status,
)

logger = logging.getLogger(__name__)

ACCEPTANCE_VERSION = 1
"""Schema version of the summary payload."""

ARTIFACT_KEYS: tuple[str, ...] = (*REQUIRED_ARTIFACTS, *OPTIONAL_ARTIFACTS)
"""Configured artifacts in display order."""

DEFAULT_ARTIFACTS: Mapping[str, Path] = {
    "benchmark": PROJECT_ROOT / "reports" / "unified" / "benchmark.json",
    "index": PROJECT_ROOT / "reports" / "unified" / "index_benchmark_scale10.json",
    "chunking": PROJECT_ROOT / "reports" / "unified" / "chunking_benchmark_rerank.json",
    "rerank": PROJECT_ROOT / "reports" / "unified" / "rerank_20to5.json",
    "judge": PROJECT_ROOT / "reports" / "judge" / "judge_summary.json",
}
"""Artifacts used by default. These mirror ``scripts/report.py``."""

DEFAULT_STATIC_DIR = PROJECT_ROOT / "web" / "acceptance"
"""Directory holding the three static dashboard files."""

ARTIFACT_LABELS: Mapping[str, str] = {
    "benchmark": "统一检索",
    "index": "索引参数",
    "chunking": "分块策略",
    "rerank": "精排 20→5",
    "judge": "LLM-as-a-Judge",
}

CAVEATS: tuple[str, ...] = (
    "页面只读：只读取已落盘 artifact，不重算指标、不修改数据、不写数据库。",
    "未提供的值显示为“—”，不按 0 分处理。",
    "artifact 状态与总状态直接来自报告层字段，页面不自行判定。",
    "自动检查是机械规则，不是人工验收结论，也不替代原始报告。",
    "历史报告与容器 artifact 可能来自不同环境，不能互相替代。",
)
"""Page-level statements that must stay visible to a reviewer."""

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; base-uri 'none'; "
    "form-action 'none'; frame-ancestors 'none'"
)

SECURITY_HEADERS: Mapping[str, str] = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": CSP,
}


class AcceptanceArtifactError(RuntimeError):
    """One or more configured artifacts exist but cannot be read."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        """Store the artifact keys without exposing their filesystem paths."""
        self.keys = keys
        super().__init__("artifact unreadable")


@dataclass(frozen=True)
class AcceptanceConfig:
    """Filesystem inputs for the read-only dashboard."""

    artifacts: Mapping[str, Path] = field(
        default_factory=lambda: dict(DEFAULT_ARTIFACTS)
    )
    static_dir: Path = DEFAULT_STATIC_DIR


def build_summary(config: AcceptanceConfig) -> dict[str, Any]:
    """Project the configured artifacts into the dashboard summary payload.

    Raises:
        AcceptanceArtifactError: an existing artifact is malformed. The caller
            receives no partial payload, so a reviewer never sees half a result.
    """
    payloads, metadata, unreadable = _load_configured(config)
    if unreadable:
        raise AcceptanceArtifactError(tuple(unreadable))

    status, reason = report_status(payloads)
    benchmark = payloads.get("benchmark")
    index = payloads.get("index")
    chunking = payloads.get("chunking")
    rerank = payloads.get("rerank")
    judge = payloads.get("judge")

    artifacts = [
        _artifact_entry(
            key,
            Path(config.artifacts[key]),
            payloads.get(key),
            metadata.get(key),
        )
        for key in ARTIFACT_KEYS
        if key in config.artifacts
    ]
    limitations = _collect_limitations(payloads)

    return {
        "schema_version": ACCEPTANCE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "read_only": True,
        "status": status,
        "status_reason": reason,
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "optional_artifacts": list(OPTIONAL_ARTIFACTS),
        "artifacts": artifacts,
        "dataset": _dataset(payloads),
        "environment": _environment(payloads),
        "checks": _build_checks(payloads),
        "sections": {
            "retrieval": (
                _retrieval_section(benchmark)
                if benchmark is not None
                else _missing_section("benchmark（统一检索）")
            ),
            "index": (
                _index_section(index)
                if index is not None
                else _missing_section("index（索引参数）")
            ),
            "chunking": (
                _chunking_section(chunking)
                if chunking is not None
                else _missing_section("chunking（分块策略）")
            ),
            "rerank": (
                _rerank_section(rerank)
                if rerank is not None
                else _missing_section("rerank（精排 20→5）")
            ),
            "judge": (
                _judge_section(judge)
                if judge is not None
                else _missing_section("judge（LLM-as-a-Judge）")
            ),
        },
        "limitations": limitations,
        "caveats": list(CAVEATS),
    }


def build_judge_queries(
    config: AcceptanceConfig, *, limit: int, offset: int
) -> dict[str, Any]:
    """Return a page of projected per-query Judge records.

    Raw provider output, usage, hashes, prompts and provider endpoints are
    intentionally absent. Missing judge artifacts are a normal empty state, not
    an error; a malformed one raises :class:`AcceptanceArtifactError`.
    """
    raw_path = config.artifacts.get("judge")
    if raw_path is None or not Path(raw_path).exists():
        return {
            "available": False,
            "total": 0,
            "expected": None,
            "not_run": None,
            "limit": limit,
            "offset": offset,
            "records": [],
        }

    try:
        payloads, _ = load_artifacts({"judge": Path(raw_path)})
    except (OSError, ReportError) as exc:
        raise AcceptanceArtifactError(("judge",)) from exc
    payload = payloads.get("judge")
    if payload is None:
        raise AcceptanceArtifactError(("judge",))

    raw_records = payload.get("per_query")
    if not isinstance(raw_records, list):
        raise AcceptanceArtifactError(("judge",))
    records = [record for record in raw_records if isinstance(record, dict)]
    summary = _mapping(payload.get("summary"))
    parameters = _mapping(payload.get("parameters"))
    page = records[offset : offset + limit]
    return {
        "available": True,
        "total": len(records),
        "expected": summary.get(
            "expected_queries", parameters.get("expected_queries")
        ),
        "not_run": summary.get("not_run_queries"),
        "limit": limit,
        "offset": offset,
        "records": [_project_judge_record(record) for record in page],
    }


def create_acceptance_router(config: AcceptanceConfig | None = None) -> APIRouter:
    """Build the GET-only acceptance router."""
    settings = config or AcceptanceConfig()
    router = APIRouter(prefix="/acceptance", tags=["acceptance"])

    @router.get("/", include_in_schema=False)
    def index_page() -> Response:
        """Serve the dashboard shell."""
        return _static_response(
            settings.static_dir, "index.html", "text/html; charset=utf-8"
        )

    @router.get("/app.js", include_in_schema=False)
    def dashboard_script() -> Response:
        """Serve the dashboard JavaScript."""
        return _static_response(
            settings.static_dir, "app.js", "application/javascript; charset=utf-8"
        )

    @router.get("/styles.css", include_in_schema=False)
    def dashboard_styles() -> Response:
        """Serve the dashboard stylesheet."""
        return _static_response(
            settings.static_dir, "styles.css", "text/css; charset=utf-8"
        )

    @router.get("/api/summary")
    def acceptance_summary() -> Response:
        """Return the projected summary for all configured artifacts."""
        try:
            payload = build_summary(settings)
        except AcceptanceArtifactError as exc:
            return _json_response(
                {"detail": "artifact unreadable", "artifacts": list(exc.keys)},
                status_code=503,
            )
        except Exception:
            logger.exception("acceptance summary failed")
            return _json_response(
                {"detail": "acceptance summary unavailable"}, status_code=503
            )
        return _json_response(payload)

    @router.get("/api/judge/queries")
    def judge_queries(
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> Response:
        """Return a page of per-query Judge results."""
        try:
            payload = build_judge_queries(settings, limit=limit, offset=offset)
        except AcceptanceArtifactError as exc:
            return _json_response(
                {"detail": "artifact unreadable", "artifacts": list(exc.keys)},
                status_code=503,
            )
        except Exception:
            logger.exception("acceptance judge queries failed")
            return _json_response(
                {"detail": "judge queries unavailable"}, status_code=503
            )
        return _json_response(payload)

    return router


def _load_configured(
    config: AcceptanceConfig,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    list[str],
]:
    """Load each configured artifact, separating missing from unreadable."""
    payloads: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    unreadable: list[str] = []
    for key in ARTIFACT_KEYS:
        raw_path = config.artifacts.get(key)
        if raw_path is None:
            continue
        try:
            loaded, meta = load_artifacts({key: Path(raw_path)})
        except (OSError, ReportError):
            unreadable.append(key)
            continue
        payloads.update(loaded)
        metadata.update(meta)
    return payloads, metadata, unreadable


def _artifact_entry(
    key: str,
    path: Path,
    payload: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Project one artifact registry row."""
    parameters = _mapping(payload.get("parameters")) if payload else {}
    return {
        "key": key,
        "label": ARTIFACT_LABELS.get(key, key),
        "required": key in REQUIRED_ARTIFACTS,
        "present": payload is not None,
        "status": payload.get("status") if payload else None,
        "status_reason": payload.get("status_reason") if payload else None,
        "generated_at": payload.get("generated_at") if payload else None,
        "sha256": metadata.get("sha256") if metadata else None,
        "path": _relative_path(path),
        "experiment_name": payload.get("experiment_name") if payload else None,
        "synthetic": bool(parameters.get("synthetic", False)),
    }


def _relative_path(path: Path) -> str:
    """Return a repository-relative path, falling back to the bare filename."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def _dataset(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return the first artifact dataset block, projected to safe fields."""
    for key in ARTIFACT_KEYS:
        data = _mapping(payloads.get(key, {}).get("dataset"))
        if not data:
            continue
        return {
            "queries": data.get("queries"),
            "eval_queries_on_disk": data.get("eval_queries_on_disk"),
            "gold_docs": data.get("gold_docs", data.get("distinct_gold_docs")),
            "hashes": dict(_mapping(data.get("hashes"))),
        }
    return {
        "queries": None,
        "eval_queries_on_disk": None,
        "gold_docs": None,
        "hashes": {},
    }


def _environment(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return selected environment facts; no hostnames or server addresses."""
    environment: Mapping[str, Any] = {}
    for key in ARTIFACT_KEYS:
        candidate = _mapping(payloads.get(key, {}).get("environment"))
        if candidate:
            environment = candidate
            break
    reranker = _mapping(environment.get("reranker"))
    return {
        "python_version": environment.get("python_version"),
        "postgres_version": environment.get("postgres_version"),
        "vector_version": environment.get("vector_version"),
        "embedding_model": environment.get("embedding_model"),
        "embedding_device": environment.get("embedding_device"),
        "embedding_dimensions": environment.get("embedding_dimensions"),
        "documents": environment.get("documents"),
        "chunks": environment.get("chunks"),
        "execution_environment": environment.get("execution_environment"),
        "reranker_model": reranker.get("model_id"),
    }


def _build_checks(
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Derive mechanical coverage checks; these are not human conclusions."""
    missing_required = [key for key in REQUIRED_ARTIFACTS if key not in payloads]
    missing_optional = [key for key in OPTIONAL_ARTIFACTS if key not in payloads]
    failing = [
        key
        for key, payload in payloads.items()
        if str(payload.get("status")) != "PASS"
    ]
    checks: list[dict[str, Any]] = [
        {
            "id": "required_present",
            "label": "三个必需 artifact 齐全",
            "state": "fail" if missing_required else "pass",
            "detail": (
                f"缺失: {', '.join(missing_required)}"
                if missing_required
                else f"{len(REQUIRED_ARTIFACTS)}/3"
            ),
        },
        {
            "id": "required_pass",
            "label": "三个必需 artifact 均为 PASS",
            "state": (
                "fail"
                if missing_required
                or any(
                    str(payloads[key].get("status")) != "PASS"
                    for key in REQUIRED_ARTIFACTS
                    if key in payloads
                )
                else "pass"
            ),
            "detail": (
                "缺失或非 PASS"
                if missing_required
                else " / ".join(
                    f"{key}: {payloads[key].get('status')}"
                    for key in REQUIRED_ARTIFACTS
                )
            ),
        },
        {
            "id": "optional_present",
            "label": "可选 artifact 齐全",
            "state": "fail" if missing_optional else "pass",
            "detail": (
                f"缺失: {', '.join(missing_optional)}"
                if missing_optional
                else f"{len(OPTIONAL_ARTIFACTS)}/2"
            ),
        },
        {
            "id": "all_loaded_pass",
            "label": "所有已加载 artifact 为 PASS",
            "state": "fail" if failing else "pass",
            "detail": (
                "; ".join(f"{key}: {payloads[key].get('status')}" for key in failing)
                if failing
                else f"{len(payloads)}/{len(ARTIFACT_KEYS)}"
            ),
        },
    ]

    judge = payloads.get("judge")
    if judge is None:
        checks.extend(
            [
                {
                    "id": "judge_coverage",
                    "label": "Judge 覆盖全部预期查询",
                    "state": "not_applicable",
                    "detail": "judge artifact 缺失",
                },
                {
                    "id": "judge_real",
                    "label": "Judge 为真实模型调用",
                    "state": "not_applicable",
                    "detail": "judge artifact 缺失",
                },
            ]
        )
        return checks

    summary = _mapping(judge.get("summary"))
    parameters = _mapping(judge.get("parameters"))
    queries = summary.get("queries", parameters.get("queries"))
    expected = summary.get("expected_queries", parameters.get("expected_queries"))
    if (
        isinstance(queries, int)
        and not isinstance(queries, bool)
        and isinstance(expected, int)
        and not isinstance(expected, bool)
    ):
        coverage_state = "pass" if queries >= expected else "fail"
        coverage_detail = f"{queries}/{expected}"
    else:
        coverage_state = "not_applicable"
        coverage_detail = "缺少 coverage 字段"
    checks.append(
        {
            "id": "judge_coverage",
            "label": "Judge 覆盖全部预期查询",
            "state": coverage_state,
            "detail": coverage_detail,
        }
    )

    provider = _mapping(judge.get("provider"))
    is_real = provider.get("is_real")
    checks.append(
        {
            "id": "judge_real",
            "label": "Judge 为真实模型调用",
            "state": "pass" if is_real is True else "fail",
            "detail": (
                "real"
                if is_real is True
                else f"is_real={is_real!r}"
            ),
        }
    )
    return checks


def _missing_section(label: str) -> dict[str, Any]:
    """Return the explicit empty state for a missing artifact section."""
    return {
        "available": False,
        "reason": f"{label} artifact 缺失，页面不显示数值。",
    }


def _retrieval_section(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project the unified retrieval artifact."""
    parameters = _mapping(payload.get("parameters"))
    return {
        "available": True,
        "parameters": {
            "top_k": parameters.get("top_k"),
            "candidate_k": parameters.get("candidate_k"),
            "rrf_k": parameters.get("rrf_k"),
            "chunk_strategy": parameters.get("chunk_strategy"),
            "queries": parameters.get("queries"),
            "expected_queries": parameters.get("expected_queries"),
        },
        "arms": [_project_arm(arm) for arm in _mappings(payload.get("arms"))],
    }


def _index_section(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project index configurations without samples, ids or plans."""
    parameters = _mapping(payload.get("parameters"))
    configurations: list[dict[str, Any]] = []
    for record in _mappings(payload.get("configurations")):
        query: dict[str, Any] = {}
        for setting, measurement in _mapping(record.get("query_params")).items():
            item = _mapping(measurement)
            latency = _mapping(item.get("latency_ms"))
            query[str(setting)] = {
                "recall": item.get("ann_recall_mean"),
                "p50_ms": latency.get("p50"),
                "p95_ms": latency.get("p95"),
            }
        configurations.append(
            {
                "name": record.get("index_name"),
                "method": record.get("method"),
                "params": dict(_mapping(record.get("params"))),
                "status": record.get("status"),
                "build_ms": record.get("build_ms"),
                "size_pretty": record.get("size_pretty"),
                "row_count": record.get("row_count"),
                "query": query,
            }
        )
    exact = _mapping(payload.get("exact"))
    baseline = (
        {
            "method": exact.get("method"),
            "top_k": exact.get("top_k"),
            "queries": exact.get("query_count"),
        }
        if exact
        else None
    )
    return {
        "available": True,
        "synthetic": bool(parameters.get("synthetic", False)),
        "synthetic_scale": parameters.get("synthetic_scale"),
        "parameters": {
            "top_k": parameters.get("top_k"),
            "queries": parameters.get("queries"),
            "chunk_strategy": parameters.get("chunk_strategy"),
            "repetitions": parameters.get("repetitions"),
        },
        "baseline": baseline,
        "configurations": configurations,
    }


def _chunking_section(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project the chunking comparison and chunk statistics."""
    parameters = _mapping(payload.get("parameters"))
    rows = []
    for row in _mappings(payload.get("summary")):
        latency = _mapping(row.get("latency_ms"))
        rows.append(
            {
                "strategy": row.get("strategy"),
                "arm": row.get("arm"),
                "status": row.get("status"),
                "recall_macro": row.get("recall_macro"),
                "recall_micro": row.get("recall_micro"),
                "precision_macro": row.get("precision_macro"),
                "precision_ceiling": row.get("precision_ceiling"),
                "ndcg_macro": row.get("ndcg_macro"),
                "mrr": row.get("mrr"),
                "hit_at_1": row.get("hit_at_1"),
                "evaluated": row.get("evaluated"),
                "failed": row.get("failed"),
                "p50_ms": latency.get("p50"),
                "p95_ms": latency.get("p95"),
            }
        )
    chunk_stats = {}
    for strategy, values in _mapping(payload.get("chunk_stats")).items():
        item = _mapping(values)
        chunk_stats[str(strategy)] = {
            "chunks": item.get("chunk_count"),
            "documents": item.get("documents"),
            "avg_chars": item.get("avg_chars"),
            "min_chars": item.get("min_chars"),
            "max_chars": item.get("max_chars"),
        }
    return {
        "available": True,
        "parameters": {
            "top_k": parameters.get("top_k"),
            "candidate_k": parameters.get("candidate_k"),
            "rrf_k": parameters.get("rrf_k"),
            "queries": parameters.get("queries"),
        },
        "rows": rows,
        "chunk_stats": chunk_stats,
    }


def _rerank_section(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project the live Top-20 -> Top-5 rerank artifact."""
    parameters = _mapping(payload.get("parameters"))
    return {
        "available": True,
        "parameters": {
            "top_k": parameters.get("top_k"),
            "candidate_k": parameters.get("candidate_k"),
            "rrf_k": parameters.get("rrf_k"),
            "chunk_strategy": parameters.get("chunk_strategy"),
            "queries": parameters.get("queries"),
        },
        "arms": [_project_arm(arm) for arm in _mappings(payload.get("arms"))],
    }


def _judge_section(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project the Judge summary; per-query rows live behind their own route."""
    parameters = _mapping(payload.get("parameters"))
    summary = _mapping(payload.get("summary"))
    provider = _mapping(payload.get("provider"))
    rag = _mapping(payload.get("rag"))
    rag_provider = _mapping(rag.get("provider"))
    metrics = _mapping(summary.get("metrics")) or _mapping(payload.get("metrics"))
    summary_fields = (
        "queries",
        "expected_queries",
        "not_run_queries",
        "rag_failures",
        "both_metrics_success",
        "queries_with_judge_failure",
        "judge_is_real",
        "rag_is_real",
        "same_model",
    )
    projected_summary = {
        key: summary.get(key) for key in summary_fields
    }
    projected_summary["rag_status_counts"] = dict(
        _mapping(summary.get("rag_status_counts"))
    )
    return {
        "available": True,
        "status": payload.get("status"),
        "status_reason": payload.get("status_reason"),
        "provider": {
            "kind": provider.get("kind"),
            "name": provider.get("name"),
            "model": provider.get("model"),
            "is_real": provider.get("is_real"),
            "real_judge_calls": provider.get("real_judge_calls"),
        },
        "rag": {
            "kind": rag_provider.get("kind"),
            "name": rag_provider.get("name"),
            "model": rag.get("model"),
            "is_real": rag.get("is_real", rag_provider.get("is_real")),
            "retrieval_mode": rag.get("retrieval_mode"),
            "queries": rag.get("queries"),
        },
        "summary": projected_summary,
        "metrics": {
            key: _project_judge_metric(metrics.get(key))
            for key in ("context_relevance", "faithfulness")
        },
        "prompt_versions": dict(_mapping(parameters.get("prompt_versions"))),
        "score_scale": dict(_mapping(parameters.get("score_scale"))),
        "gold_usage": dict(_mapping(payload.get("gold_usage"))),
        "limitations": _string_list(payload.get("limitations")),
    }


def _project_judge_metric(value: Any) -> dict[str, Any]:
    """Project one aggregated Judge metric."""
    item = _mapping(value)
    return {
        "attempted": item.get("attempted"),
        "not_attempted": item.get("not_attempted"),
        "success": item.get("success"),
        "failed": item.get("failed"),
        "success_rate": item.get("success_rate"),
        "valid_scores": item.get("valid_scores"),
        "mean": item.get("mean"),
        "median": item.get("median"),
        "distribution": dict(_mapping(item.get("distribution"))),
        "low_score_threshold": item.get("low_score_threshold"),
        "low_score_count": item.get("low_score_count"),
        "low_score_samples": [
            {
                "query_id": sample.get("query_id"),
                "metric": sample.get("metric"),
                "score": sample.get("score"),
                "reason": sample.get("reason"),
            }
            for sample in _mappings(item.get("low_score_samples"))
        ],
        "failure_counts": {
            "status": dict(_mapping(_mapping(item.get("failure_counts")).get("status"))),
            "error_category": dict(
                _mapping(_mapping(item.get("failure_counts")).get("error_category"))
            ),
        },
    }


def _project_judge_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one per-query Judge record without raw provider output."""
    return {
        "query_id": record.get("query_id"),
        "query": record.get("query"),
        "rag_status": record.get("rag_status"),
        "rag_model": record.get("rag_model"),
        "rag_is_real": record.get("rag_is_real"),
        "context_relevance": _project_context_relevance(
            record.get("context_relevance")
        ),
        "faithfulness": _project_faithfulness(record.get("faithfulness")),
    }


def _project_context_relevance(value: Any) -> dict[str, Any]:
    """Project one Context Relevance result."""
    item = _mapping(value)
    return {
        "status": item.get("status"),
        "attempted": item.get("attempted"),
        "score": item.get("score"),
        "reason": item.get("reason"),
        "relevant_context_ids": _string_list(item.get("relevant_context_ids")),
        "irrelevant_context_ids": _string_list(item.get("irrelevant_context_ids")),
        "prompt_version": item.get("prompt_version"),
        "model": item.get("model"),
        "latency_ms": item.get("latency_ms"),
        "attempts": item.get("attempts"),
        "error": _project_error(item.get("error")),
    }


def _project_faithfulness(value: Any) -> dict[str, Any]:
    """Project one Faithfulness result and its claim-level support."""
    item = _mapping(value)
    claims = [
        {
            "claim": claim.get("claim"),
            "support_status": claim.get("support_status"),
            "evidence_context_ids": _string_list(
                claim.get("evidence_context_ids")
            ),
        }
        for claim in _mappings(item.get("claims"))
    ]
    return {
        "status": item.get("status"),
        "attempted": item.get("attempted"),
        "score": item.get("score"),
        "reason": item.get("reason"),
        "claims": claims,
        "unsupported_claims": _string_list(item.get("unsupported_claims")),
        "prompt_version": item.get("prompt_version"),
        "model": item.get("model"),
        "latency_ms": item.get("latency_ms"),
        "attempts": item.get("attempts"),
        "error": _project_error(item.get("error")),
    }


def _project_error(value: Any) -> dict[str, Any] | None:
    """Project an error classification without its free-form message."""
    item = _mapping(value)
    if not item:
        return None
    return {
        "category": item.get("category"),
        "retryable": item.get("retryable"),
        "status_code": item.get("status_code"),
        "attempts": item.get("attempts"),
    }


def _project_arm(arm: Mapping[str, Any]) -> dict[str, Any]:
    """Project one retrieval arm, preserving every source metric as-is."""
    metrics = _mapping(arm.get("metrics"))
    recall = _mapping(metrics.get("recall_at_k"))
    precision = _mapping(metrics.get("precision_at_k"))
    ndcg = _mapping(metrics.get("ndcg_at_k"))
    latency = _mapping(_mapping(arm.get("latency_ms")).get("query"))
    return {
        "name": arm.get("name"),
        "status": arm.get("status"),
        "recall_macro": recall.get("macro"),
        "recall_micro": recall.get("micro"),
        "precision_macro": precision.get("macro"),
        "precision_ceiling": precision.get("ceiling_macro"),
        "ndcg_macro": ndcg.get("macro"),
        "mrr": metrics.get("mrr"),
        "hit_at_1": metrics.get("hit_at_1"),
        "evaluated": metrics.get("evaluated_queries"),
        "failed": metrics.get("failed_queries"),
        "p50_ms": latency.get("p50"),
        "p95_ms": latency.get("p95"),
    }


def _collect_limitations(
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Collect artifact-declared limitations with their source key."""
    result: list[dict[str, str]] = []
    for key in ARTIFACT_KEYS:
        payload = payloads.get(key)
        if payload is None:
            continue
        for item in _string_list(payload.get("limitations")):
            result.append({"source": key, "text": item})
    return result


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping for a JSON object, or an empty mapping otherwise."""
    return value if isinstance(value, dict) else {}


def _mappings(value: Any) -> list[Mapping[str, Any]]:
    """Return only the JSON objects from a list."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    """Return only string members of a list."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _json_response(
    content: dict[str, Any], *, status_code: int = 200
) -> JSONResponse:
    """Build a non-cacheable JSON response with the dashboard security headers."""
    return JSONResponse(
        content=content,
        status_code=status_code,
        headers=dict(SECURITY_HEADERS),
    )


def _static_response(static_dir: Path, filename: str, media_type: str) -> Response:
    """Serve one fixed dashboard asset, never an arbitrary path."""
    path = Path(static_dir) / filename
    if not path.is_file():
        return _json_response(
            {"detail": "acceptance assets unavailable"}, status_code=503
        )
    return FileResponse(
        path,
        media_type=media_type,
        headers=dict(SECURITY_HEADERS),
    )
