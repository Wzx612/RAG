"""Read-only acceptance API contracts, using fixture artifacts only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from src.evaluation.report import load_artifacts, report_status, sha256_file

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from src.api.acceptance import AcceptanceConfig  # noqa: E402
from src.api.app import create_app  # noqa: E402
from src.config import PROJECT_ROOT  # noqa: E402

STATIC_DIR = PROJECT_ROOT / "web" / "acceptance"

GENERATED_AT = "2026-10-20T05:31:33+00:00"


def _benchmark() -> dict[str, Any]:
    """Return a minimal unified-retrieval artifact."""
    return {
        "experiment_name": "unified_retrieval_benchmark",
        "status": "PASS",
        "status_reason": "all arms evaluated every expected query",
        "generated_at": GENERATED_AT,
        "parameters": {
            "top_k": 10,
            "candidate_k": 10,
            "rrf_k": 60,
            "chunk_strategy": "document",
            "queries": 30,
            "expected_queries": 30,
        },
        "dataset": {
            "queries": 30,
            "eval_queries_on_disk": 30,
            "gold_docs": 38,
            "hashes": {"eval_queries.json": "a" * 64},
        },
        "environment": {
            "python_version": "3.12.13",
            "postgres_version": "16.14",
            "vector_version": "0.8.2",
            "embedding_model": "BAAI/bge-large-en-v1.5",
            "documents": 150,
            "chunks": 150,
            "server_address": "172.22.0.2/32:5432",
            "reranker": {"model_id": "BAAI/bge-reranker-base"},
        },
        "arms": [
            {
                "name": "dense",
                "status": "PASS",
                "metrics": {
                    "recall_at_k": {"macro": 0.9444444444444444, "micro": 0.8837209302325582},
                    "precision_at_k": {
                        "macro": 0.12666666666666668,
                        "ceiling_macro": 0.14333333333333334,
                    },
                    "ndcg_at_k": {"macro": 0.900380613054194},
                    "mrr": 0.9166666666666666,
                    "hit_at_1": 25,
                    "evaluated_queries": 30,
                    "failed_queries": 0,
                },
                "latency_ms": {"query": {"p50": 78.9038, "p95": 92.542}},
            }
        ],
        "limitations": ["Fixture benchmark limitation."],
    }


def _index() -> dict[str, Any]:
    """Return a minimal index-parameter artifact with private fields to strip."""
    return {
        "status": "PASS",
        "status_reason": "all configurations ran",
        "generated_at": GENERATED_AT,
        "parameters": {
            "top_k": 10,
            "queries": 30,
            "chunk_strategy": "document",
            "repetitions": 3,
            "synthetic": True,
            "synthetic_scale": 10,
        },
        "exact": {"method": "seq_scan", "top_k": 10, "query_count": 90},
        "configurations": [
            {
                "index_name": "idx_bench_hnsw_m8_efc64",
                "method": "hnsw",
                "params": {"m": 8, "ef_construction": 64},
                "status": "PASS",
                "build_ms": 90.801,
                "size_pretty": "1208 kB",
                "row_count": 1500,
                "query_params": {
                    "ef_search=10": {
                        "latency_ms": {"p50": 0.2647, "p95": 0.3683},
                        "ann_recall_mean": 1.0,
                        "samples_ms": [0.1, 0.2],
                        "ids": {"Q-001": ["DOC-0001"]},
                    },
                    "ef_search=40": {
                        "latency_ms": {"p50": 0.4100, "p95": 0.5100},
                        "ann_recall_mean": 0.99,
                    },
                },
            }
        ],
    }


def _chunking() -> dict[str, Any]:
    """Return a minimal chunking artifact."""
    return {
        "status": "PASS",
        "status_reason": "all strategies evaluated",
        "generated_at": GENERATED_AT,
        "parameters": {"top_k": 10, "candidate_k": 10, "rrf_k": 60, "queries": 30},
        "chunk_stats": {
            "document": {
                "chunk_count": 150,
                "documents": 150,
                "avg_chars": 291.61,
                "min_chars": 220,
                "max_chars": 540,
            }
        },
        "summary": [
            {
                "strategy": "document",
                "arm": "dense",
                "status": "PASS",
                "recall_macro": 0.9444444444444444,
                "recall_micro": 0.8837209302325582,
                "precision_macro": 0.12666666666666668,
                "precision_ceiling": 0.14333333333333334,
                "ndcg_macro": 0.900380613054194,
                "mrr": 0.9166666666666666,
                "hit_at_1": 25,
                "evaluated": 30,
                "failed": 0,
                "latency_ms": {"p50": 78.6434, "p95": 131.601},
            }
        ],
    }


def _rerank() -> dict[str, Any]:
    """Return a minimal Top-20 -> Top-5 rerank artifact."""
    return {
        "status": "PASS",
        "status_reason": "both arms evaluated",
        "generated_at": GENERATED_AT,
        "parameters": {
            "top_k": 5,
            "candidate_k": 20,
            "rrf_k": 60,
            "chunk_strategy": "document",
            "queries": 30,
        },
        "arms": [
            {
                "name": "rrf",
                "status": "PASS",
                "metrics": {
                    "recall_at_k": {"macro": 0.7778, "micro": 0.7000},
                    "precision_at_k": {"macro": 0.2000, "ceiling_macro": 0.2500},
                    "ndcg_at_k": {"macro": 0.8000},
                    "mrr": 0.8100,
                    "hit_at_1": 20,
                    "evaluated_queries": 30,
                    "failed_queries": 0,
                },
                "latency_ms": {"query": {"p50": 400.0, "p95": 500.0}},
            }
        ],
    }


def _judge() -> dict[str, Any]:
    """Return a 3/30 Judge artifact with private provider fields."""
    metrics = {
        "context_relevance": {
            "attempted": 3,
            "not_attempted": 0,
            "success": 3,
            "failed": 0,
            "success_rate": 1.0,
            "valid_scores": 3,
            "mean": 4.333333333333333,
            "median": 4,
            "distribution": {"1": 0, "2": 0, "3": 0, "4": 2, "5": 1},
            "low_score_threshold": 3,
            "low_score_count": 0,
            "low_score_samples": [],
            "failure_counts": {"status": {}, "error_category": {}},
        },
        "faithfulness": {
            "attempted": 3,
            "not_attempted": 0,
            "success": 3,
            "failed": 0,
            "success_rate": 1.0,
            "valid_scores": 3,
            "mean": 5.0,
            "median": 5,
            "distribution": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 3},
            "low_score_threshold": 3,
            "low_score_count": 0,
            "low_score_samples": [],
            "failure_counts": {"status": {}, "error_category": {}},
        },
    }
    records = []
    for position in range(1, 4):
        records.append(
            {
                "query_id": f"Q-00{position}",
                "query": f"Question {position}?",
                "rag_status": "ok",
                "rag_model": "deepseek-chat",
                "rag_is_real": True,
                "answer_sha256": "b" * 64,
                "context_sha256": "c" * 64,
                "context_relevance": {
                    "status": "success",
                    "attempted": True,
                    "score": 4,
                    "reason": "Useful context.",
                    "relevant_context_ids": ["E1"],
                    "irrelevant_context_ids": ["E2"],
                    "prompt_version": "judge-context-relevance-v1",
                    "model": "deepseek-chat",
                    "latency_ms": 1418.71,
                    "attempts": 1,
                    "raw_response": "SECRET RAW RESPONSE",
                },
                "faithfulness": {
                    "status": "success",
                    "attempted": True,
                    "score": 5,
                    "reason": "All claims supported.",
                    "claims": [
                        {
                            "claim": "The fee is waived.",
                            "support_status": "supported",
                            "evidence_context_ids": ["E1"],
                        }
                    ],
                    "unsupported_claims": [],
                    "prompt_version": "judge-faithfulness-v1",
                    "model": "deepseek-chat",
                    "latency_ms": 1562.25,
                    "attempts": 1,
                    "raw_response": "SECRET RAW RESPONSE",
                },
            }
        )
    return {
        "experiment_name": "llm_as_a_judge",
        "status": "PARTIAL",
        "status_reason": "real judge executed on 3/30 queries (smoke)",
        "generated_at": GENERATED_AT,
        "parameters": {
            "queries": 3,
            "expected_queries": 30,
            "prompt_versions": {
                "context_relevance": "judge-context-relevance-v1",
                "faithfulness": "judge-faithfulness-v1",
            },
            "score_scale": {"min": 1, "max": 5},
        },
        "provider": {
            "kind": "real",
            "name": "openai_compatible",
            "model": "deepseek-chat",
            "is_real": True,
            "real_judge_calls": True,
            "api_key": "sk-secret-value",
            "endpoint": "https://api.deepseek.com/chat/completions",
        },
        "rag": {
            "model": "deepseek-chat",
            "is_real": True,
            "retrieval_mode": "live",
            "queries": 3,
            "provider": {
                "kind": "real",
                "name": "openai_compatible",
                "is_real": True,
                "endpoint": "https://api.deepseek.com/chat/completions",
                "api_key_present": True,
            },
        },
        "summary": {
            "queries": 3,
            "expected_queries": 30,
            "not_run_queries": 27,
            "rag_failures": 0,
            "both_metrics_success": 3,
            "queries_with_judge_failure": 0,
            "judge_is_real": True,
            "rag_is_real": True,
            "same_model": True,
            "rag_status_counts": {"ok": 3},
            "metrics": metrics,
        },
        "metrics": metrics,
        "limitations": ["Only 3 of 30 queries were judged."],
        "per_query": records,
    }


def _payloads() -> dict[str, dict[str, Any]]:
    """Return all fixture artifact payloads."""
    return {
        "benchmark": _benchmark(),
        "index": _index(),
        "chunking": _chunking(),
        "rerank": _rerank(),
        "judge": _judge(),
    }


def _write_artifacts(root: Path) -> dict[str, Path]:
    """Write the five fixture artifacts under ``root`` and return their paths."""
    paths = {}
    for key, payload in _payloads().items():
        path = root / f"{key}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        paths[key] = path
    return paths


def _client(
    paths: dict[str, Path], static_dir: Path | None = None
) -> TestClient:
    """Build an app with injected artifact inputs."""
    config = (
        AcceptanceConfig(artifacts=paths)
        if static_dir is None
        else AcceptanceConfig(artifacts=paths, static_dir=static_dir)
    )
    return TestClient(create_app(acceptance_config=config))


def _assert_no_secret_text(text: str) -> None:
    """Assert that private artifact fields never reach a response body."""
    for marker in (
        "raw_response",
        "SECRET RAW RESPONSE",
        "api_key",
        "sk-secret-value",
        "Bearer",
        "/app/",
        "answer_sha256",
        "context_sha256",
        "samples_ms",
        "https://api.deepseek.com",
        "172.22.0.2",
    ):
        assert marker not in text


def test_summary_status_matches_report_status(tmp_path: Path) -> None:
    """The page and the final report share one status computation."""
    paths = _write_artifacts(tmp_path)
    response = _client(paths).get("/acceptance/api/summary")
    assert response.status_code == 200
    body = response.json()
    payloads, _ = load_artifacts(paths)
    status, reason = report_status(payloads)
    assert body["status"] == status == "PARTIAL"
    assert body["status_reason"] == reason
    assert body["read_only"] is True


def test_artifact_registry_matches_paths_and_hashes(tmp_path: Path) -> None:
    """Every registry row points at a real file whose hash is reported."""
    paths = _write_artifacts(tmp_path)
    body = _client(paths).get("/acceptance/api/summary").json()
    entries = {entry["key"]: entry for entry in body["artifacts"]}
    assert list(entries) == ["benchmark", "index", "chunking", "rerank", "judge"]
    for key, path in paths.items():
        entry = entries[key]
        assert entry["present"] is True
        assert entry["sha256"] == sha256_file(path)
        try:
            expected = path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            expected = path.name
        assert entry["path"] == expected
    assert entries["judge"]["status"] == "PARTIAL"
    assert entries["benchmark"]["required"] is True
    assert entries["judge"]["required"] is False


def test_summary_projects_values_without_recomputing(tmp_path: Path) -> None:
    """Each section exposes the source numbers, not a new aggregation."""
    body = _client(_write_artifacts(tmp_path)).get("/acceptance/api/summary").json()
    retrieval = body["sections"]["retrieval"]
    assert retrieval["available"] is True
    assert retrieval["arms"][0]["name"] == "dense"
    assert retrieval["arms"][0]["recall_macro"] == pytest.approx(0.9444444444444444)
    assert retrieval["arms"][0]["p50_ms"] == pytest.approx(78.9038)
    assert retrieval["arms"][0]["precision_ceiling"] == pytest.approx(0.14333333333333334)

    index = body["sections"]["index"]
    configuration = index["configurations"][0]
    assert configuration["name"] == "idx_bench_hnsw_m8_efc64"
    assert configuration["query"]["ef_search=10"]["recall"] == 1.0
    assert configuration["query"]["ef_search=10"]["p50_ms"] == pytest.approx(0.2647)
    assert "query_count" not in index["baseline"]
    assert index["baseline"]["queries"] == 90
    assert "samples_ms" not in json.dumps(index)
    assert "ids" not in json.dumps(index)

    chunking = body["sections"]["chunking"]
    assert chunking["rows"][0]["p50_ms"] == pytest.approx(78.6434)
    assert chunking["chunk_stats"]["document"]["chunks"] == 150

    rerank = body["sections"]["rerank"]
    assert rerank["parameters"]["candidate_k"] == 20
    assert rerank["arms"][0]["recall_macro"] == pytest.approx(0.7778)

    judge = body["sections"]["judge"]
    assert judge["summary"]["queries"] == 3
    assert judge["summary"]["expected_queries"] == 30
    assert judge["summary"]["not_run_queries"] == 27
    assert judge["summary"]["same_model"] is True
    assert judge["metrics"]["context_relevance"]["mean"] == pytest.approx(
        4.333333333333333
    )
    assert judge["provider"]["is_real"] is True
    assert body["dataset"]["queries"] == 30
    assert body["environment"]["reranker_model"] == "BAAI/bge-reranker-base"
    _assert_no_secret_text(json.dumps(body))


def test_missing_optional_artifacts_are_explicit(tmp_path: Path) -> None:
    """A missing optional artifact is shown as missing, never as zero."""
    paths = _write_artifacts(tmp_path)
    required = {key: paths[key] for key in ("benchmark", "index", "chunking")}
    response = _client(required).get("/acceptance/api/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "PASS"
    assert body["sections"]["rerank"]["available"] is False
    assert "arms" not in body["sections"]["rerank"]
    assert body["sections"]["judge"]["available"] is False
    assert "metrics" not in body["sections"]["judge"]
    checks = {check["id"]: check for check in body["checks"]}
    assert checks["optional_present"]["state"] == "fail"
    assert checks["judge_coverage"]["state"] == "not_applicable"
    assert all(entry["present"] for entry in body["artifacts"])


def test_missing_required_artifact_marks_partial(tmp_path: Path) -> None:
    """A missing required artifact downgrades the overall status."""
    paths = _write_artifacts(tmp_path)
    partial_paths = {
        "benchmark": paths["benchmark"],
        "chunking": paths["chunking"],
    }
    body = _client(partial_paths).get("/acceptance/api/summary").json()
    assert body["status"] == "PARTIAL"
    assert "index" in body["status_reason"]
    assert body["sections"]["index"]["available"] is False
    checks = {check["id"]: check for check in body["checks"]}
    assert checks["required_present"]["state"] == "fail"
    assert "index" in checks["required_present"]["detail"]


def test_malformed_artifact_fails_closed(tmp_path: Path) -> None:
    """A broken artifact returns 503 and no partial numbers."""
    paths = _write_artifacts(tmp_path)
    paths["benchmark"].write_text("{not json", encoding="utf-8")
    response = _client(paths).get("/acceptance/api/summary")
    assert response.status_code == 503
    body = response.json()
    assert body == {"detail": "artifact unreadable", "artifacts": ["benchmark"]}
    text = response.text
    assert str(tmp_path) not in text
    assert "dense" not in text


def test_judge_queries_paginate_and_strip_raw_output(tmp_path: Path) -> None:
    """Per-query rows expose scores and reasons, never raw provider output."""
    client = _client(_write_artifacts(tmp_path))
    first = client.get("/acceptance/api/judge/queries")
    assert first.status_code == 200
    body = first.json()
    assert body["available"] is True
    assert body["total"] == 3
    assert body["expected"] == 30
    assert body["not_run"] == 27
    assert [record["query_id"] for record in body["records"]] == ["Q-001", "Q-002", "Q-003"]
    assert body["records"][0]["context_relevance"]["score"] == 4
    assert body["records"][0]["faithfulness"]["claims"][0]["claim"] == "The fee is waived."
    _assert_no_secret_text(first.text)

    page = client.get("/acceptance/api/judge/queries?limit=1&offset=1")
    assert page.status_code == 200
    page_body = page.json()
    assert page_body["limit"] == 1
    assert page_body["offset"] == 1
    assert [record["query_id"] for record in page_body["records"]] == ["Q-002"]


def test_judge_queries_missing_artifact_is_empty(tmp_path: Path) -> None:
    """An absent Judge artifact is an empty state, not an error."""
    paths = _write_artifacts(tmp_path)
    required = {key: paths[key] for key in ("benchmark", "index", "chunking")}
    body = _client(required).get("/acceptance/api/judge/queries").json()
    assert body == {
        "available": False,
        "total": 0,
        "expected": None,
        "not_run": None,
        "limit": 50,
        "offset": 0,
        "records": [],
    }


def test_judge_queries_reject_out_of_range_paging(tmp_path: Path) -> None:
    """Paging is bounded before any artifact is read."""
    client = _client(_write_artifacts(tmp_path))
    assert client.get("/acceptance/api/judge/queries?limit=0").status_code == 422
    assert client.get("/acceptance/api/judge/queries?limit=201").status_code == 422
    assert client.get("/acceptance/api/judge/queries?offset=-1").status_code == 422


def test_only_get_methods_are_registered(tmp_path: Path) -> None:
    """The dashboard exposes no write method."""
    client = _client(_write_artifacts(tmp_path))
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)("/acceptance/api/summary")
        assert response.status_code == 405
    assert client.post("/acceptance/").status_code == 405


def test_static_assets_are_served_with_security_headers(tmp_path: Path) -> None:
    """The dashboard shell and its two assets are local files."""
    client = _client(_write_artifacts(tmp_path), static_dir=STATIC_DIR)
    index = client.get("/acceptance/")
    assert index.status_code == 200
    assert index.headers["content-type"].startswith("text/html")
    assert "评测验收看板" in index.text
    assert index.headers["cache-control"] == "no-store"
    assert index.headers["x-content-type-options"] == "nosniff"

    script = client.get("/acceptance/app.js")
    assert script.status_code == 200
    assert script.headers["content-type"].startswith("application/javascript")
    assert "textContent" in script.text

    styles = client.get("/acceptance/styles.css")
    assert styles.status_code == 200
    assert styles.headers["content-type"].startswith("text/css")


def test_missing_static_directory_returns_503(tmp_path: Path) -> None:
    """A missing asset is an explicit 503, not an empty page."""
    client = _client(
        _write_artifacts(tmp_path), static_dir=tmp_path / "does-not-exist"
    )
    response = client.get("/acceptance/")
    assert response.status_code == 503
    assert response.json() == {"detail": "acceptance assets unavailable"}


def test_summary_headers_prevent_caching(tmp_path: Path) -> None:
    """JSON responses are non-cacheable and marked nosniff."""
    response = _client(_write_artifacts(tmp_path)).get("/acceptance/api/summary")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    policy = response.headers["content-security-policy"]
    assert "default-src 'self'" in policy
    assert "frame-ancestors 'none'" in policy


def test_healthz_is_unaffected(tmp_path: Path) -> None:
    """Mounting the dashboard does not change the liveness endpoint."""
    response = _client(_write_artifacts(tmp_path)).get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_artifact_errors_do_not_include_report_error_text(tmp_path: Path) -> None:
    """A malformed artifact uses the shared loader but hides its path-bearing error."""
    paths = _write_artifacts(tmp_path)
    paths["judge"].write_text("[]", encoding="utf-8")
    client = _client(paths)
    summary = client.get("/acceptance/api/summary")
    queries = client.get("/acceptance/api/judge/queries")
    for response in (summary, queries):
        assert response.status_code == 503
        assert "judge" in response.json()["artifacts"]
    assert "ReportError" not in summary.text
    assert str(tmp_path) not in summary.text
