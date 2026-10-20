"""Phase 4C: dense benchmark definitions, plans and one real-DB smoke run.

The integration test uses the real 150-chunk corpus and the live PostgreSQL
pool, but it injects a stored chunk vector instead of loading the 1.3 GB model.
That isolates the benchmark's SQL, planner verification, GUC read-back, row
mapping and metrics; the full benchmark script separately uses the real BGE
model. No test writes to the database or changes a global setting.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import psycopg
import pytest
from pgvector import Vector
from src.evaluation.artifacts import ArtifactExistsError
from src.evaluation.dense_benchmark import (
    BenchmarkConfig,
    BenchmarkError,
    EvalQuery,
    build_configs,
    classify_plan,
    load_eval_queries,
    render_markdown,
    run_dense_benchmark,
    sanitize_plan,
    write_outputs,
)
from src.retrieval.dense import DenseRetriever
from src.schema.retrieval import DenseConfig

DATA = Path(__file__).resolve().parents[1] / "data" / "eval_queries.json"


class _StoredVectorEmbedder:
    """Return one already-stored chunk vector for the production sanity check."""

    def __init__(self, vector: Sequence[float]) -> None:
        """Store the vector and its dimension."""
        self._vector = list(vector)
        self.dimension = len(self._vector)

    def embed_query(self, text: str) -> list[float]:
        """Return the stored vector."""
        return list(self._vector)

    def embed_documents(self, texts: Sequence[str], on_batch: Any = None) -> list[list[float]]:
        """Return one stored vector per text; unused by DenseRetriever."""
        del on_batch
        return [list(self._vector) for _ in texts]


# ------------------------------------------------------------------ pure tests


def test_load_eval_queries_reads_the_real_file() -> None:
    """The benchmark sees all 30 ids and Q-001's labelled docs."""
    queries = load_eval_queries(DATA)
    assert len(queries) == 30
    assert len({query.query_id for query in queries}) == 30
    assert queries[0].query_id == "Q-001"
    assert queries[0].relevant_doc_ids


def test_build_configs_has_exact_natural_and_hnsw_arms() -> None:
    """Every arm shares K, and ef values are deduplicated in order."""
    configs = build_configs(10, [10, 40, 10])
    assert [config.name for config in configs] == [
        "exact_seqscan",
        "natural_plan",
        "hnsw_ef_10",
        "hnsw_ef_40",
    ]
    assert [config.mode for config in configs] == ["exact", "natural", "hnsw", "hnsw"]
    assert {config.top_k for config in configs} == {10}
    assert [config.ef_search for config in configs if config.mode == "hnsw"] == [10, 40]


@pytest.mark.parametrize(
    ("top_k", "ef_values"),
    [(0, [10]), (10, []), (10, [0]), (10, [1001])],
)
def test_build_configs_rejects_invalid_parameters(top_k: int, ef_values: list[int]) -> None:
    """The benchmark uses DenseConfig's accepted ranges."""
    with pytest.raises(BenchmarkError):
        build_configs(top_k, ef_values)


def test_classify_plan_distinguishes_the_three_paths() -> None:
    """A plan is classified by the ANN stage, not by a JOIN branch."""
    assert classify_plan("Index Scan using chunks_embedding_hnsw_idx on chunks") == "hnsw"
    assert classify_plan("Seq Scan on chunks  (cost=0.00..1.00 rows=1 width=8)") == "seq_scan"
    assert classify_plan("Index Scan using documents_pkey on documents") == "other_index"
    assert classify_plan("Result") == "other"


def test_sanitize_plan_removes_the_bound_vector() -> None:
    """A 1024-float Sort Key line is replaced, not copied into the report."""
    plan = (
        "Limit\n  Sort Key: '[0.1, 0.2, ...]'\n"
        "  ->  Index Scan using chunks_embedding_hnsw_idx on chunks\n"
        "        Order By: (embedding <=> '[0.1, 0.2, ...]')"
    )
    sanitized = sanitize_plan(plan)
    assert "1024-dim query vector" in sanitized
    assert "embedding <=> (query vector omitted)" in sanitized
    assert "[0.1" not in sanitized


def test_run_rejects_mismatched_top_k_before_touching_the_database() -> None:
    """A non-comparable matrix is refused during validation."""
    query = EvalQuery("Q-X", "query", ("DOC-0001",))
    vector = [0.01] * 1024
    configs = build_configs(5, [5])
    configs.append(
        BenchmarkConfig(
            name="bad_top_k",
            mode="hnsw",
            ef_search=5,
            top_k=6,
            description="deliberately different K",
        )
    )
    with pytest.raises(BenchmarkError, match="same top_k"):
        run_dense_benchmark([query], {"Q-X": vector}, configs, repetitions=1)


# --------------------------------------------------------------- integration


@pytest.fixture(scope="module")
def stored_vector(conn: psycopg.Connection) -> list[float]:
    """One real 1024-dim chunk vector, used as a deterministic query."""
    row = conn.execute(
        "SELECT embedding FROM chunks WHERE chunk_strategy = 'document' "
        "ORDER BY chunk_id LIMIT 1"
    ).fetchone()
    assert row is not None
    value = row[0]
    return value.to_list() if hasattr(value, "to_list") else Vector(value).to_list()


def test_real_db_smoke_run_verifies_plans_guc_and_metrics(
    conn: psycopg.Connection, stored_vector: list[float]
) -> None:
    """One query through exact, natural and two HNSW arms, with real SQL."""
    count_row = conn.execute(
        "SELECT count(*) FROM chunks WHERE chunk_strategy = 'document'"
    ).fetchone()
    assert count_row is not None
    assert count_row[0] == 150
    queries = [query for query in load_eval_queries(DATA) if query.query_id == "Q-001"]
    assert len(queries) == 1
    vector_map = {queries[0].query_id: stored_vector}
    configs = build_configs(5, [5, 40])

    result = run_dense_benchmark(
        queries,
        vector_map,
        configs,
        repetitions=1,
        warmup_passes=0,
        statement_timeout_ms=5000,
    )

    assert result["totals"] == {
        "errors": 0,
        "timeouts": 0,
        "skipped": 0,
        "invalid_configs": 0,
    }
    assert len(result["configs"]) == 4

    by_name = {config["name"]: config for config in result["configs"]}
    exact = by_name["exact_seqscan"]
    assert exact["plan"]["kind"] == "seq_scan"
    assert exact["plan"]["uses_seq_scan_on_chunks"] is True
    assert exact["plan"]["uses_hnsw"] is False
    assert exact["quality"]["recall_vs_exact_mean"] == 1.0
    assert exact["quality"]["order_equal_queries"] == 1

    natural = by_name["natural_plan"]
    assert natural["plan"]["kind"] in {"seq_scan", "hnsw", "other_index"}
    assert natural["quality"]["evaluated_queries"] == 1

    for name, ef in (("hnsw_ef_5", 5), ("hnsw_ef_40", 40)):
        arm = by_name[name]
        assert arm["valid"] is True
        assert arm["plan"]["kind"] == "hnsw"
        assert arm["plan"]["uses_hnsw"] is True
        assert arm["actual_ef_search_values"] == [ef]
        assert arm["ef_mismatch_samples"] == 0
        assert arm["quality"]["evaluated_queries"] == 1
        record = arm["per_query"][0]
        # The strategy filter runs after the ANN scan: with ef_search=5 and three
        # strategies sharing one HNSW index, fewer than 5 of the global top-5 can
        # belong to 'document'. Top-K is an upper bound here, not a guarantee.
        assert 1 <= record["result_count"] <= 5
        assert record["unique_count"] == record["result_count"]
        assert 0.0 <= record["recall_vs_exact"] <= 1.0
        assert record["query_ms"] > 0.0
        assert record["db_ms"] >= record["query_ms"]

    # Production path consistency: same SQL, same stored vector, same config.
    natural_config = next(config for config in configs if config.mode == "natural")
    retriever = DenseRetriever(
        DenseConfig(top_n=natural_config.top_k, ef_search=natural_config.ef_search),
        embedder=_StoredVectorEmbedder(stored_vector),
    )
    production = [
        f"{hit.chunk_strategy}:{hit.chunk_id}" for hit in retriever.retrieve(queries[0].query)
    ]
    assert production == natural["per_query"][0]["keys"]

    report = render_markdown(result)
    assert "# Dense Retrieval Benchmark" in report
    assert "## 3. Exact baseline definition" in report
    assert "ANN Recall@K" in report


def test_write_outputs_creates_json_and_markdown(tmp_path: Path) -> None:
    """The raw artifact and report are written to caller-selected paths."""
    result = {
        "generated_at": "2026-10-17T00:00:00+00:00",
        "status": "PASS",
        "environment": {},
        "dataset": {},
        "parameters": {"queries": 0, "top_k": 10, "repetitions": 0},
        "configs": [],
        "totals": {"errors": 0, "timeouts": 0, "skipped": 0, "invalid_configs": 0},
    }
    json_path = tmp_path / "raw.json"
    report_path = tmp_path / "report.md"
    write_outputs(result, json_path, report_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "PASS"
    assert "Dense Retrieval Benchmark" in report_path.read_text(encoding="utf-8")


def test_write_outputs_refuses_to_clobber_a_historical_report(tmp_path: Path) -> None:
    """A default re-run cannot replace an existing artifact."""
    result = {
        "generated_at": "2026-10-17T00:00:00+00:00",
        "status": "PASS",
        "environment": {},
        "dataset": {},
        "parameters": {"queries": 0, "top_k": 10, "repetitions": 0},
        "configs": [],
        "totals": {"errors": 0, "timeouts": 0, "skipped": 0, "invalid_configs": 0},
    }
    json_path = tmp_path / "raw.json"
    report_path = tmp_path / "report.md"
    write_outputs(result, json_path, report_path)
    before = json_path.read_text(encoding="utf-8")

    with pytest.raises(ArtifactExistsError, match="refusing to overwrite"):
        write_outputs(result, json_path, report_path)

    assert json_path.read_text(encoding="utf-8") == before
    write_outputs(result, json_path, report_path, overwrite=True)
