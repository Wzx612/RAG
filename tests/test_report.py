"""Final report assembly: status, traceability, rendering and figures.

These tests use small synthetic artifacts shaped exactly like the ones the
benchmark scripts write. They never touch the database, the model cache or the
real reports, so the assertions are about the report contract: every table
cites a file and a field, missing inputs make the report ``PARTIAL``, and a
``None`` metric stays visible as ``-`` rather than becoming a zero.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from src.evaluation.report import (
    FiguresUnavailableError,
    ReportError,
    load_artifacts,
    render_report,
    report_status,
    write_figures,
)

GENERATED = "2026-10-20T00:00:00+00:00"


def _arm(name: str, *, status: str = "PASS", with_latency: bool = True) -> dict[str, Any]:
    """One unified-benchmark arm record."""
    arm: dict[str, Any] = {
        "name": name,
        "description": "",
        "parameters": {},
        "status": status,
        "metrics": {
            "recall_at_k": {"macro": 0.9, "micro": 0.85},
            "precision_at_k": {"macro": 0.2, "ceiling_macro": 0.3},
            "ndcg_at_k": {"macro": 0.88},
            "mrr": 0.8,
            "hit_at_1": 20,
            "evaluated_queries": 30,
            "failed_queries": 0,
        },
        "failed_queries": [],
        "per_query": [],
    }
    if with_latency:
        arm["latency_ms"] = {
            "query": {"n": 30, "p50": 5.0, "p95": 9.0, "p99": 12.0, "max": 15.0}
        }
    return arm


def _benchmark(status: str = "PASS", *, arms: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """A unified retrieval artifact."""
    return {
        "schema_version": 1,
        "experiment_name": "unified_retrieval_benchmark",
        "generated_at": GENERATED,
        "status": status,
        "status_reason": "synthetic fixture",
        "parameters": {
            "top_k": 10,
            "candidate_k": 10,
            "rrf_k": 60,
            "chunk_strategy": "document",
            "queries": 30,
            "expected_queries": 30,
            "arms": ["dense", "rrf"],
        },
        "dataset": {
            "queries": 30,
            "eval_queries_on_disk": 30,
            "gold_docs": 38,
            "hashes": {"eval_queries.json": "a" * 64, "kb_documents.json": "b" * 64},
        },
        "arms": arms if arms is not None else [_arm("dense"), _arm("rrf")],
    }


def _index(status: str = "PASS") -> dict[str, Any]:
    """An index sweep artifact with HNSW, IVFFlat and one failed configuration."""

    def measurement(recall: float, p95: float) -> dict[str, Any]:
        return {"ann_recall_mean": recall, "latency_ms": {"p50": 0.8, "p95": p95, "p99": 2.0}}

    return {
        "schema_version": 1,
        "experiment_name": "index_param_sweep",
        "generated_at": GENERATED,
        "status": status,
        "status_reason": "synthetic fixture",
        "parameters": {
            "top_k": 10,
            "queries": 30,
            "synthetic_scale": 1,
            "hnsw_ef_search_values": [10, 40],
            "ivfflat_probes_values": [1, 2],
        },
        "configurations": [
            {
                "index_name": "idx_hnsw_m8_efc32",
                "method": "hnsw",
                "params": {"m": 8, "ef_construction": 32},
                "status": "PASS",
                "query_params": {
                    "ef_search=10": measurement(0.91, 1.1),
                    "ef_search=40": measurement(0.99, 2.3),
                },
            },
            {
                "index_name": "idx_hnsw_m16_efc64",
                "method": "hnsw",
                "params": {"m": 16, "ef_construction": 64},
                "status": "PASS",
                "query_params": {
                    "ef_search=10": measurement(0.95, 1.4),
                    "ef_search=40": measurement(1.0, 2.9),
                },
            },
            {
                "index_name": "idx_ivfflat_lists4",
                "method": "ivfflat",
                "params": {"lists": 4},
                "status": "PASS",
                "query_params": {
                    "probes=1": measurement(0.7, 0.9),
                    "probes=2": measurement(0.85, 1.2),
                },
            },
            {
                "index_name": "idx_ivfflat_lists8",
                "method": "ivfflat",
                "params": {"lists": 8},
                "status": "FAIL",
                "error": "RuntimeError: forced plan mismatch",
                "query_params": {},
            },
        ],
    }


def _chunking(status: str = "PASS") -> dict[str, Any]:
    """A chunking comparison artifact with two strategies."""
    rows = []
    for strategy in ("document", "fixed_window"):
        rows.append(
            {
                "strategy": strategy,
                "arm": "dense",
                "status": "PASS",
                "recall_macro": 0.9,
                "mrr": 0.8,
                "ndcg_macro": 0.87,
                "evaluated": 30,
                "failed": 0,
                "latency_ms": {"n": 30, "p50": 5.0, "p95": 9.0, "p99": 12.0},
            }
        )
    return {
        "schema_version": 1,
        "experiment_name": "chunking_comparison",
        "generated_at": GENERATED,
        "status": status,
        "status_reason": "synthetic fixture",
        "parameters": {
            "top_k": 10,
            "candidate_k": 10,
            "rrf_k": 60,
            "queries": 30,
            "expected_queries": 30,
            "strategies": ["document", "fixed_window"],
            "arms": {"document": ["dense"], "fixed_window": ["dense"]},
        },
        "chunk_stats": {
            "document": {"chunk_count": 150, "documents": 150, "avg_chars": 800.0},
            "fixed_window": {"chunk_count": 420, "documents": 150, "avg_chars": 300.0},
        },
        "summary": rows,
        "strategies": {},
        "limitations": [],
    }


def _rerank(status: str = "PASS") -> dict[str, Any]:
    """A live candidate-pool -> reranked-Top-K artifact (20 -> 5)."""
    return {
        "schema_version": 1,
        "experiment_name": "unified_retrieval_benchmark",
        "generated_at": GENERATED,
        "status": status,
        "status_reason": "synthetic fixture",
        "parameters": {
            "top_k": 5,
            "candidate_k": 20,
            "rrf_k": 60,
            "chunk_strategy": "document",
            "queries": 30,
            "expected_queries": 30,
            "arms": ["rrf", "rrf_rerank"],
        },
        "arms": [_arm("rrf"), _arm("rrf_rerank")],
    }


def _judge(status: str = "PARTIAL") -> dict[str, Any]:
    """A three-query smoke summary with both metric distributions."""
    return {
        "schema_version": 1,
        "experiment_name": "llm_as_a_judge",
        "generated_at": GENERATED,
        "status": status,
        "status_reason": "real judge executed on 3/30 queries (smoke)",
        "parameters": {
            "queries": 3,
            "expected_queries": 30,
            "prompt_versions": {
                "context_relevance": "judge-context-relevance-v1",
                "faithfulness": "judge-faithfulness-v1",
            },
        },
        "provider": {
            "kind": "real",
            "name": "openai_compatible",
            "model": "deepseek-chat",
            "is_real": True,
            "real_judge_calls": True,
        },
        "rag": {
            "provider": {"kind": "real", "name": "openai_compatible"},
            "model": "deepseek-chat",
        },
        "summary": {
            "queries": 3,
            "expected_queries": 30,
            "not_run_queries": 27,
            "same_model": True,
            "rag_failures": 0,
            "metrics": {
                "context_relevance": {
                    "attempted": 3,
                    "success": 3,
                    "failed": 0,
                    "valid_scores": 3,
                    "mean": 4.3333,
                    "median": 4.0,
                    "distribution": {"1": 0, "2": 0, "3": 1, "4": 1, "5": 1},
                    "low_score_threshold": 3,
                    "low_score_count": 0,
                    "low_score_samples": [],
                    "failure_counts": {"status": {}, "error_category": {}},
                },
                "faithfulness": {
                    "attempted": 3,
                    "success": 3,
                    "failed": 0,
                    "valid_scores": 3,
                    "mean": 4.0,
                    "median": 4.0,
                    "distribution": {"1": 0, "2": 0, "3": 0, "4": 2, "5": 1},
                    "low_score_threshold": 3,
                    "low_score_count": 0,
                    "low_score_samples": [],
                    "failure_counts": {"status": {}, "error_category": {}},
                },
            },
        },
        "limitations": [
            "Only 3 of 30 queries were judged; this smoke run does not describe the full set.",
            "The judge reuses the same model id as the answer generator.",
        ],
        "gold_usage": {"gold_response_used_for_scoring": False},
    }


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> Path:
    """Write a fixture artifact and return its path."""
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _payloads() -> dict[str, dict[str, Any]]:
    """All three artifacts, fully PASS."""
    return {
        "benchmark": _benchmark(),
        "index": _index(),
        "chunking": _chunking(),
    }


def test_report_status_requires_all_three_artifacts() -> None:
    """PASS needs all three; a missing or non-PASS artifact downgrades it."""
    assert report_status(_payloads())[0] == "PASS"
    partial, reason = report_status({"benchmark": _benchmark()})
    assert partial == "PARTIAL"
    assert "index" in reason and "chunking" in reason
    failing, reason = report_status(
        {"benchmark": _benchmark(), "index": _index("FAIL"), "chunking": _chunking()}
    )
    assert failing == "PARTIAL"
    assert "index is FAIL" in reason
    assert report_status({})[0] == "FAIL"


def test_load_artifacts_hashes_the_exact_bytes(tmp_path: Path) -> None:
    """The report cites a SHA256 of the file it read, not a recomputed value."""
    path = _write(tmp_path, "benchmark.json", _benchmark())
    payloads, metadata = load_artifacts({"benchmark": path, "index": tmp_path / "none.json"})
    assert list(payloads) == ["benchmark"]
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["benchmark"]["sha256"] == expected
    assert metadata["benchmark"]["status"] == "PASS"
    assert "index" not in metadata


def test_load_artifacts_rejects_a_non_artifact(tmp_path: Path) -> None:
    """An existing but malformed file is an error, not a pass-through."""
    path = _write(tmp_path, "bad.json", {"foo": "bar"})
    with pytest.raises(ReportError, match="lacks status/generated_at"):
        load_artifacts({"benchmark": path})
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ReportError, match="cannot read artifact"):
        load_artifacts({"benchmark": path})


def test_rendered_report_cites_every_table_and_field() -> None:
    """The report names the artifact, its hash and the field path."""
    payloads = _payloads()
    metadata = {
        "benchmark": {
            "path": "reports/unified/benchmark.json",
            "sha256": "c" * 64,
            "status": "PASS",
            "generated_at": GENERATED,
        },
        "index": {
            "path": "reports/unified/index_benchmark.json",
            "sha256": "d" * 64,
            "status": "PASS",
            "generated_at": GENERATED,
        },
        "chunking": {
            "path": "reports/unified/chunking_benchmark.json",
            "sha256": "e" * 64,
            "status": "PASS",
            "generated_at": GENERATED,
        },
    }
    rendered = render_report(
        payloads,
        metadata,
        context={"git.commit": "abc123", "git.dirty": True},
        figures=["arm_metrics.png"],
        unverified=["额外未验证项。"],
    )
    assert "- Status: **PASS**" in rendered
    assert "[artifact: reports/unified/benchmark.json#arms[].metrics" in rendered
    assert "sha256=cccccccccccc" in rendered
    assert "`benchmark#arms[1]`" in rendered
    assert "ef_search=40" in rendered
    assert "ef_search=10" in rendered
    assert "0.990 / 2.30 ms" in rendered
    assert "idx_ivfflat_lists8" in rendered
    assert "fixed_window" in rendered
    assert "figures/arm_metrics.png" in rendered
    assert "## 6. 图表" in rendered
    assert "## 7. 复现命令" in rendered
    assert "## 8. 明确未验证的范围" in rendered
    assert "额外未验证项。" in rendered
    assert "生产级可用性" in rendered
    assert "git.commit: abc123" in rendered


def test_missing_artifact_is_stated_and_never_filled_in() -> None:
    """A partial input set renders PARTIAL and says which section is absent."""
    payloads = {"benchmark": _benchmark(), "index": _index()}
    rendered = render_report(payloads, {})
    assert "- Status: **PARTIAL**" in rendered
    assert "missing artifact(s): chunking" in rendered
    assert "未提供 chunking artifact。" in rendered
    assert "## 6. 复现命令" in rendered


def test_optional_rerank_artifact_is_rendered_and_shifts_later_sections() -> None:
    """The live 20 -> 5 rerank run gets its own sourced section when present."""
    payloads = dict(_payloads())
    payloads["rerank"] = _rerank()
    metadata = {
        "rerank": {
            "path": "reports/unified/rerank_20to5.json",
            "sha256": "f" * 64,
            "status": "PASS",
            "generated_at": GENERATED,
        }
    }
    rendered = render_report(payloads, metadata, figures=["arm_metrics.png"])
    assert "- Status: **PASS**" in rendered
    assert "## 6. Cross-Encoder 精排" in rendered
    assert "**20 → 5**" in rendered
    assert "[artifact: reports/unified/rerank_20to5.json#arms[].metrics" in rendered
    assert "`rerank#arms[1]`" in rendered
    assert "## 7. 图表" in rendered
    assert "## 8. 复现命令" in rendered
    assert "## 9. 明确未验证的范围" in rendered
    # The optional artifact is never required: a report without it is unaffected.
    assert "Cross-Encoder 精排" not in render_report(_payloads(), {})


def test_optional_judge_artifact_is_rendered_and_downgrades_status() -> None:
    """A 3/30 smoke judge is shown as PARTIAL and shifts later sections."""
    payloads = dict(_payloads())
    payloads["rerank"] = _rerank()
    payloads["judge"] = _judge()
    metadata = {
        "judge": {
            "path": "reports/judge/judge_summary.json",
            "sha256": "1" * 64,
            "status": "PARTIAL",
            "generated_at": GENERATED,
        }
    }
    rendered = render_report(payloads, metadata, figures=["judge_distribution.png"])

    assert "- Status: **PARTIAL**" in rendered
    assert "judge is PARTIAL" in rendered
    assert "## 7. LLM-as-a-Judge" in rendered
    assert "[artifact: reports/judge/judge_summary.json#summary.metrics" in rendered
    assert "**3 / 30**" in rendered
    assert "NOT RUN: **27**" in rendered
    assert "smoke；不代表全量数据集" in rendered
    assert "同模型自评: **True**" in rendered
    assert "Context Relevance" in rendered
    assert "Faithfulness" in rendered
    assert "judge-context-relevance-v1" in rendered
    assert "gold_response` 参与评分: **False**" in rendered
    assert "## 8. 图表" in rendered
    assert "## 9. 复现命令" in rendered
    assert "judge_distribution.png" in rendered
    # The optional section is omitted when there is no judge artifact.
    assert "LLM-as-a-Judge" not in render_report(_payloads(), {})


def test_judge_distribution_figure_is_added_when_present(tmp_path: Path) -> None:
    """The judge summary adds one two-panel PNG without changing other figures."""
    pytest.importorskip("matplotlib")
    payloads = dict(_payloads())
    payloads["judge"] = _judge()
    written = write_figures(payloads, tmp_path)
    names = [path.name for path in written]

    assert names == [
        "arm_metrics.png",
        "arm_latency.png",
        "hnsw_heatmap.png",
        "chunking_metrics.png",
        "judge_distribution.png",
    ]
    judge_figure = written[-1]
    assert judge_figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_none_metrics_stay_visible_as_dashes() -> None:
    """A missing latency is rendered ``-``; it never becomes 0.0000."""
    payloads = {"benchmark": _benchmark(arms=[_arm("dense", with_latency=False)])}
    rendered = render_report(payloads, {})
    assert "| dense | - | - | - |" in rendered


def test_figures_are_pngs_and_skip_absent_sections(tmp_path: Path) -> None:
    """Benchmark-only input yields the two arm figures and nothing invented."""
    pytest.importorskip("matplotlib")
    written = write_figures({"benchmark": _benchmark()}, tmp_path)
    assert [path.name for path in written] == ["arm_metrics.png", "arm_latency.png"]
    for path in written:
        assert path.exists()
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 1000


def test_all_figures_render_from_complete_artifacts(tmp_path: Path) -> None:
    """Four figures come out of the complete input set."""
    pytest.importorskip("matplotlib")
    written = write_figures(_payloads(), tmp_path)
    assert [path.name for path in written] == [
        "arm_metrics.png",
        "arm_latency.png",
        "hnsw_heatmap.png",
        "chunking_metrics.png",
    ]


def test_missing_matplotlib_is_an_explicit_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the report extra the caller gets a named exception, not a crash."""
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(FiguresUnavailableError, match="matplotlib is not installed"):
        write_figures({"benchmark": _benchmark()}, tmp_path)
