"""Phase 5A: sparse benchmark aggregation, overlap and rendering.

The overlap tests are pure and deterministic. The one integration test runs
two real queries against the live corpus and skips when the database or corpus
is absent; it deliberately does not load the embedding model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import psycopg
import pytest
from src.evaluation.dense_benchmark import BenchmarkError, EvalQuery
from src.evaluation.sparse_benchmark import (
    FOCUS_QUERY_IDS,
    SparseBenchmarkConfig,
    _rows_to_hits,
    attach_query_reference,
    build_config,
    dense_overlap,
    render_markdown,
    run_sparse_benchmark,
    write_outputs,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "eval_queries.json"


@pytest.fixture
def populated_corpus(conn: psycopg.Connection) -> None:
    """Skip integration tests when Phase 3 has not imported the corpus."""
    row = conn.execute("SELECT count(*) FROM chunks").fetchone()
    if row is None or row[0] == 0:
        pytest.skip("corpus not imported -- run `make ingest`")


# --------------------------------------------------------------------- config


def test_build_config_validates_all_knobs() -> None:
    """Every benchmark setting is rejected before a query is issued."""
    config = build_config(10, 5, 1, 5000, "or")
    assert isinstance(config, SparseBenchmarkConfig)
    assert config.top_k == 10
    assert config.repetitions == 5
    assert config.warmup_passes == 1
    assert config.query_mode == "or"

    for args in (
        (0, 1, 0, 5000, "or"),
        (10, 0, 0, 5000, "or"),
        (10, 1, -1, 5000, "or"),
        (10, 1, 0, 0, "or"),
        (10, 1, 0, 5000, "xor"),
        (True, 1, 0, 5000, "or"),
    ):
        with pytest.raises(BenchmarkError):
            build_config(*args)


# ---------------------------------------------------------------------- rows


def test_rows_to_hits_rejects_invalid_scores_and_overflow() -> None:
    """A corrupt score or an over-long result set cannot become a measurement."""
    row = ("chunk", "DOC-0001", "document", 0, "content", 0.5, "t", "c", "tier", 2024, "P")
    bad = ("chunk", "DOC-0001", "document", 0, "content", -1.0, "t", "c", "tier", 2024, "P")
    with pytest.raises(BenchmarkError, match="invalid sparse score"):
        _rows_to_hits([bad], 10)
    with pytest.raises(BenchmarkError, match="returned"):
        _rows_to_hits([row, row], 1)
    hits = _rows_to_hits([row], 10)
    assert hits[0].key == "document:chunk"
    assert hits[0].score == 0.5
    assert hits[0].rank == 1


# ------------------------------------------------------------------- overlap


def _write_artifact(path: Path, per_query: list[dict[str, object]]) -> None:
    """Write a minimal Phase 4C-shaped artifact."""
    path.write_text(
        json.dumps(
            {
                "configs": [
                    {
                        "mode": "natural",
                        "top_k": 10,
                        "per_query": per_query,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_dense_overlap_computes_jaccard_and_gold_coverage(tmp_path: Path) -> None:
    """Chunk/doc overlap and the four gold buckets are computed from real fields."""
    artifact = tmp_path / "dense.json"
    _write_artifact(
        artifact,
        [
            {
                "query_id": "Q-001",
                "keys": ["document:a", "document:b"],
                "doc_ids": ["DOC-0001"],
                "result_count": 2,
            }
        ],
    )
    sparse = [
        {
            "query_id": "Q-001",
            "evaluated": True,
            "keys": ["document:a", "document:c"],
            "doc_ids": ["DOC-0002"],
            "result_count": 2,
        },
        {
            "query_id": "Q-002",
            "evaluated": False,
            "keys": [],
            "doc_ids": [],
            "result_count": None,
        },
    ]
    queries = [
        EvalQuery("Q-001", "q1", ("DOC-0001", "DOC-0002")),
        EvalQuery("Q-002", "q2", ("DOC-0001",)),
    ]

    overlap = dense_overlap(sparse, queries, artifact, 10)

    assert overlap["available"] is True
    assert overlap["compared_queries"] == 1
    assert overlap["chunk_jaccard_mean"] == pytest.approx(1 / 3)
    assert overlap["doc_jaccard_mean"] == pytest.approx(0.0)
    assert overlap["gold_doc_coverage"] == {
        "both": 0,
        "sparse_only": 1,
        "dense_only": 1,
        "neither": 0,
    }
    record = overlap["per_query"][0]
    assert record["chunk_overlap"] == 1
    assert record["sparse_only_chunks"] == 1
    assert record["dense_only_chunks"] == 1
    assert record["gold"]["sparse_only"] == ["DOC-0002"]
    assert record["gold"]["dense_only"] == ["DOC-0001"]


def test_dense_overlap_is_optional_and_never_fakes_a_result(tmp_path: Path) -> None:
    """A missing or malformed artifact is reported as unavailable."""
    missing = dense_overlap([], [], tmp_path / "absent.json", 10)
    assert missing["available"] is False
    assert "not found" in missing["reason"]

    malformed = tmp_path / "bad.json"
    malformed.write_text("{not json", encoding="utf-8")
    unavailable = dense_overlap([], [], malformed, 10)
    assert unavailable["available"] is False
    assert "not valid JSON" in unavailable["reason"]

    no_plan = tmp_path / "no-plan.json"
    no_plan.write_text("{}", encoding="utf-8")
    unavailable = dense_overlap([], [], no_plan, 10)
    assert unavailable["available"] is False
    assert "natural_plan" in unavailable["reason"]


# --------------------------------------------------------------- integration


def test_run_sparse_benchmark_on_two_real_queries(populated_corpus: None) -> None:
    """The benchmark records real hits, semantic recall and latency."""
    queries = [
        EvalQuery(
            entry["query_id"],
            entry["query"],
            tuple(entry["relevant_doc_ids"]),
        )
        for entry in json.loads(DATA.read_text(encoding="utf-8"))[:2]
    ]
    config = build_config(10, 1, 0, 5000, "or")
    result = run_sparse_benchmark(queries, config)

    assert result["status"] == "COMPLETED"
    assert result["parameters"]["query_mode"] == "or"
    assert result["totals"]["evaluated_queries"] == 2
    assert result["totals"]["errors"] == 0
    assert result["totals"]["timeouts"] == 0
    assert result["retrieval"]["path_counts"] == {"fts": 2}
    assert result["retrieval"]["quality"]["semantic_recall_mean"] == 1.0
    assert len(result["retrieval"]["per_query"]) == 2
    for record in result["retrieval"]["per_query"]:
        assert record["evaluated"] is True
        assert record["result_count"] == 10
        assert record["empty_result"] is False
        assert record["top_score"] > 0
        assert record["latency"]["query_ms"]
        assert record["latency"]["db_ms"]
    assert set(result["retrieval"]["focus_queries"]) <= set(FOCUS_QUERY_IDS)


def test_focus_queries_are_extracted_by_id(populated_corpus: None) -> None:
    """The four Phase 4C focus queries are present when they were evaluated."""
    queries = [
        EvalQuery(
            entry["query_id"],
            entry["query"],
            tuple(entry["relevant_doc_ids"]),
        )
        for entry in json.loads(DATA.read_text(encoding="utf-8"))[:14]
    ]
    config = build_config(10, 1, 0, 5000, "or")
    result = run_sparse_benchmark(queries, config)
    assert set(result["retrieval"]["focus_queries"]) == set(FOCUS_QUERY_IDS)


# -------------------------------------------------------------------- report


def test_render_markdown_handles_a_blocked_result() -> None:
    """A blocked artifact is clearly labelled and never looks like a measurement."""
    text = render_markdown({"status": "BLOCKED", "issues": ["database unreachable"]})
    assert "BLOCKED" in text
    assert "database unreachable" in text
    assert "Sparse Retrieval Benchmark" in text


def test_write_outputs_strips_the_private_query_reference(tmp_path: Path) -> None:
    """The in-memory query list is for the renderer only; JSON stays clean."""
    result = attach_query_reference(
        {"status": "COMPLETED", "retrieval": {}, "parameters": {}},
        [EvalQuery("Q-001", "q", ("DOC-0001",))],
    )
    json_path = tmp_path / "result.json"
    report_path = tmp_path / "result.md"
    write_outputs(result, json_path, report_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "_queries" not in payload
    assert report_path.read_text(encoding="utf-8").startswith("# Sparse Retrieval Benchmark")


def test_measure_rejects_a_non_finite_score_path() -> None:
    """The benchmark and the retriever reject the same corrupt score."""
    row = ("chunk", "DOC-0001", "document", 0, "content", math.nan, "t", "c", "tier", 2024, "P")
    with pytest.raises(BenchmarkError, match="invalid sparse score"):
        _rows_to_hits([row], 10)
