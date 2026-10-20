"""Index-parameter benchmark: validation, rendering and artifact safety.

No PostgreSQL connection is opened. The sweep itself is exercised by
``scripts/benchmark_index.py`` against a real database; these tests pin the
matrix validation, the fail-before-connecting rule and the report contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from src.evaluation.dense_benchmark import EvalQuery
from src.evaluation.index_benchmark import (
    IndexBenchmarkConfig,
    _sanitize_plan,
    _sql_literal,
    blocked_result,
    render_markdown,
    run_index_benchmark,
    write_outputs,
)


class _NoConnection:
    """A factory that fails the test if anything tries to connect."""

    def __call__(self) -> Any:
        """Fail immediately."""
        raise AssertionError("the database must not be touched")


def _query(query_id: str = "q1") -> EvalQuery:
    """Build one labelled query."""
    return EvalQuery(
        query_id=query_id,
        query="what is the foreign transaction fee?",
        relevant_doc_ids=("d1",),
    )


def _passing_result() -> dict[str, Any]:
    """A minimal PASS artifact shaped exactly like a real run."""
    return {
        "status": "PASS",
        "status_reason": "every index configuration completed",
        "generated_at": "2026-10-20T00:00:00+00:00",
        "parameters": {
            "top_k": 10,
            "queries": 1,
            "chunk_strategy": "document",
            "synthetic_scale": 1,
        },
        "isolation": {"synthetic_label": None},
        "exact": {"method": "seqscan", "query_count": 1},
        "configurations": [
            {
                "index_name": "idx_bench_hnsw_m16_efc64",
                "method": "hnsw",
                "params": {"m": 16, "ef_construction": 64},
                "status": "PASS",
                "error": None,
                "build_ms": 12.5,
                "size_bytes": 8192,
                "size_pretty": "8192 bytes",
                "row_count": 150,
                "query_params": {
                    "ef_search=40": {
                        "latency_ms": {
                            "n": 1,
                            "min": 1.0,
                            "p50": 1.0,
                            "p95": 1.0,
                            "p99": 1.0,
                            "max": 1.0,
                            "mean": 1.0,
                        },
                        "samples_ms": [1.0],
                        "ann_recall_mean": 1.0,
                        "ids": {"q1": ["d1"]},
                    }
                },
                "plan": {"kind": "index_scan", "text": "Index Scan using idx_bench"},
            }
        ],
        "limitations": ["规模限制 A"],
    }


def test_default_matrix_matches_the_plan() -> None:
    """The default sweep is the documented 9 HNSW / 15 IVFFlat grid."""
    config = IndexBenchmarkConfig()
    assert config.hnsw_m_values == (8, 16, 32)
    assert config.hnsw_ef_construction_values == (32, 64, 128)
    assert config.hnsw_ef_search_values == (10, 40, 100)
    assert config.ivfflat_lists_values == (1, 2, 4, 8, 16)
    assert config.ivfflat_probes_values == (1, 2, 4)
    assert config.top_k == 10


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_counts_are_rejected(value: object) -> None:
    """Counts must be positive ints before anything connects."""
    with pytest.raises(ValueError, match="must be a positive int"):
        IndexBenchmarkConfig(repetitions=value)  # type: ignore[arg-type]


def test_sweep_values_must_be_inside_pgvector_ranges() -> None:
    """Reject impossible parameters before pgvector sees them."""
    with pytest.raises(ValueError, match="hnsw_m_values"):
        IndexBenchmarkConfig(hnsw_m_values=(1,))
    with pytest.raises(ValueError, match="hnsw_ef_construction_values"):
        IndexBenchmarkConfig(hnsw_ef_construction_values=(2000,))
    with pytest.raises(ValueError, match="ivfflat_lists_values"):
        IndexBenchmarkConfig(ivfflat_lists_values=())
    with pytest.raises(ValueError, match="ivfflat_probes_values"):
        IndexBenchmarkConfig(ivfflat_probes_values=(0,))


def test_at_least_one_method_must_run() -> None:
    """Disabling both methods cannot produce a measurement."""
    with pytest.raises(ValueError, match="at least one index method"):
        IndexBenchmarkConfig(include_hnsw=False, include_ivfflat=False)


def test_missing_query_vectors_fail_before_connecting() -> None:
    """The engine never opens a connection without one vector per query."""
    factory = _NoConnection()
    with pytest.raises(ValueError, match="missing query vectors for: q1"):
        run_index_benchmark(
            [_query()],
            {},
            IndexBenchmarkConfig(),
            connection_factory=factory,
        )


def test_blocked_result_is_complete_and_renderable() -> None:
    """A BLOCKED artifact has every field the renderer reads."""
    result = blocked_result("PostgreSQL is not reachable", config=IndexBenchmarkConfig())
    assert result["status"] == "BLOCKED"
    assert result["configurations"] == []
    assert result["exact"] is None
    assert "PostgreSQL is not reachable" in result["limitations"]
    rendered = render_markdown(result)
    assert "**BLOCKED**" in rendered
    assert "PostgreSQL is not reachable" in rendered


def test_markdown_reports_every_query_setting_with_p99() -> None:
    """The overview shows one row per index setting, with P99 and row count."""
    rendered = render_markdown(_passing_result())
    assert "# 索引参数实验（HNSW / IVFFlat）" in rendered
    assert "P99 ms" in rendered
    assert "ef_search=40" in rendered
    assert "8192 bytes" in rendered
    assert "| 150 |" in rendered
    assert "ANN Recall 是相对精确 Seq Scan Top-K 的保真度" in rendered
    assert "规模限制 A" in rendered


def test_synthetic_runs_are_labelled_in_the_report() -> None:
    """A synthetic scale cannot be mistaken for real-corpus evidence."""
    result = _passing_result()
    result["parameters"]["synthetic_scale"] = 100
    result["isolation"]["synthetic_label"] = "SYNTHETIC"
    rendered = render_markdown(result)
    assert "SYNTHETIC" in rendered
    assert "仅用于索引行为" in rendered


def test_write_outputs_refuses_to_clobber_an_artifact(tmp_path: Path) -> None:
    """The shared no-clobber rule applies to the index artifacts too."""
    result = _passing_result()
    json_path = tmp_path / "index.json"
    report_path = tmp_path / "index.md"
    write_outputs(result, json_path, report_path)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        write_outputs(result, json_path, report_path)
    write_outputs(result, json_path, report_path, overwrite=True)
    assert json_path.exists() and report_path.exists()


def test_sql_literal_escapes_quotes() -> None:
    """Utility-statement literals escape quotes instead of concatenating blindly."""
    assert _sql_literal("document") == "'document'"
    assert _sql_literal("o'brien") == "'o''brien'"


def test_plan_sanitizer_drops_the_vector_and_caps_long_lines() -> None:
    """EXPLAIN evidence must not carry 1024-dim literals into the artifact."""
    plan = (
        "Index Scan using idx on copy (cost=0.01..10.00 rows=10)\n"
        "  Order By: (embedding <=> '[0.1,0.2,0.3]'::vector)\n"
        "  Sort Key: (embedding <=> '[0.1,0.2,0.3]'::vector)\n"
        + "x" * 400
    )
    sanitized = _sanitize_plan(plan)
    assert "[0.1,0.2,0.3]" not in sanitized
    assert "'<vector>'::vector" in sanitized
    assert "Sort Key: (omitted" in sanitized
    assert max(len(line) for line in sanitized.splitlines()) <= 300


def test_artifact_mapping_is_json_serialisable() -> None:
    """The artifact contains only plain JSON types (no tuples or vectors)."""
    payload = _passing_result()
    assert isinstance(payload, Mapping)
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
