"""Phase 6B: five-arm hybrid benchmark, with fake retrievers end to end.

These tests do not touch PostgreSQL or the embedding model. They pin the arm
metrics, raw per-query records, candidate provenance, latency sample counts,
error isolation and output serialisation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pytest
from src.evaluation.dense_benchmark import BenchmarkError, EvalQuery
from src.evaluation.hybrid_benchmark import (
    ARM_NAMES,
    _measure,
    build_config,
    render_markdown,
    run_hybrid_benchmark,
    write_outputs,
)
from src.retrieval import RetrievalError
from src.schema.retrieval import SearchHit

ENVIRONMENT = {
    "execution_environment": "unit-test",
    "dependency_versions": {},
}
DATASET = {
    "queries": 2,
    "missing_gold_docs": [],
    "hashes": {},
}


def _hit(
    chunk_id: str,
    *,
    source: Literal["dense", "sparse"],
    rank: int,
    score: float | None = None,
    doc_id: str | None = None,
) -> SearchHit:
    """Build a valid arm hit, optionally with a repeated document id."""
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=doc_id or chunk_id,
        score=score if score is not None else (1.0 / rank if source == "dense" else 2.0 / rank),
        distance=1.0 - (1.0 / rank) if source == "dense" else None,
        rank=rank,
        source=source,
        content=f"content-{chunk_id}",
        chunk_strategy="document",
        chunk_index=0,
        title=f"title-{chunk_id}",
        category="fees",
        card_tier="standard",
        effective_year=2025,
        policy_code="P-1",
    )


class _MapRetriever:
    """Return a fixed ranked list per query and record calls."""

    name = "fake"

    def __init__(
        self,
        mapping: dict[str, list[SearchHit]],
        *,
        error_queries: Sequence[str] = (),
    ) -> None:
        """Store the per-query results and queries that should fail."""
        self.mapping = mapping
        self.error_queries = set(error_queries)
        self.calls: list[tuple[str, int | None]] = []
        self.query_calls: dict[str, int] = {}

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return the fixed hits or raise for a configured query."""
        self.calls.append((query, top_k))
        if query in self.error_queries:
            raise RetrievalError(f"fake failure for {query}")
        hits = self.mapping.get(query, [])
        return list(hits if top_k is None else hits[:top_k])


class _FlipRetriever(_MapRetriever):
    """Reverse the order on every second call, to exercise the consistency gate."""

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return the fixed hits, reversed on every second call for that query."""
        hits = super().retrieve(query, top_k=top_k)
        count = self.query_calls.get(query, 0) + 1
        self.query_calls[query] = count
        if count % 2 == 0:
            return list(reversed(hits))
        return hits


def _queries() -> list[EvalQuery]:
    """Two labelled queries with distinct gold docs."""
    return [
        EvalQuery(query_id="Q-1", query="q1", relevant_doc_ids=("DOC-A",)),
        EvalQuery(query_id="Q-2", query="q2", relevant_doc_ids=("DOC-C",)),
    ]


def _retrievers() -> tuple[_MapRetriever, _MapRetriever, _MapRetriever]:
    """Build the toy dense/sparse-OR/sparse-AND retrievers."""
    dense = _MapRetriever(
        {
            "q1": [_hit("DOC-A", source="dense", rank=1), _hit("DOC-B", source="dense", rank=2)],
            "q2": [_hit("DOC-B", source="dense", rank=1), _hit("DOC-D", source="dense", rank=2)],
        }
    )
    sparse_or = _MapRetriever(
        {
            "q1": [_hit("DOC-B", source="sparse", rank=1)],
            "q2": [_hit("DOC-C", source="sparse", rank=1)],
        }
    )
    sparse_and = _MapRetriever({"q1": [], "q2": []})
    return dense, sparse_or, sparse_and


def _run(
    *,
    config: object | None = None,
    queries: Sequence[EvalQuery] | None = None,
    dense: _MapRetriever | None = None,
    sparse_or: _MapRetriever | None = None,
    sparse_and: _MapRetriever | None = None,
) -> dict[str, object]:
    """Run the benchmark with fakes unless overridden."""
    dense_default, sparse_default, and_default = _retrievers()
    return run_hybrid_benchmark(
        config or build_config(top_k=2, candidate_k=2, rrf_k=60, repetitions=2, warmup_passes=1),
        queries or _queries(),
        dense_retriever=dense or dense_default,
        sparse_or_retriever=sparse_or or sparse_default,
        sparse_and_retriever=sparse_and or and_default,
        environment=ENVIRONMENT,
        dataset=DATASET,
    )


# ---------------------------------------------------------------- arm quality


def test_five_arms_are_evaluated_with_expected_metrics() -> None:
    """Toy data pins each arm's Recall, MRR and Hit@1 independently."""
    result = _run()
    assert result["status"] == "PASS"
    arms = result["arms"]["arms"]

    assert set(arms) == set(ARM_NAMES)
    assert arms["dense"]["semantic_recall_macro"] == pytest.approx(0.5)
    assert arms["dense"]["mrr"] == pytest.approx(0.5)
    assert arms["dense"]["hit_at_1"] == 1

    assert arms["sparse_or"]["semantic_recall_macro"] == pytest.approx(0.5)
    assert arms["sparse_or"]["mrr"] == pytest.approx(0.5)
    assert arms["sparse_or"]["hit_at_1"] == 1

    assert arms["sparse_and"]["semantic_recall_macro"] == 0.0
    assert arms["sparse_and"]["mrr"] == 0.0
    assert arms["sparse_and"]["hit_at_1"] == 0
    assert arms["sparse_and"]["empty_result_queries"] == 2

    assert arms["candidate_union"]["semantic_recall_macro"] == pytest.approx(1.0)
    assert arms["candidate_union"]["mrr"] == pytest.approx(0.75)
    assert arms["candidate_union"]["hit_at_1"] == 1

    assert arms["rrf"]["semantic_recall_macro"] == pytest.approx(1.0)
    assert arms["rrf"]["mrr"] == pytest.approx(0.5)
    assert arms["rrf"]["hit_at_1"] == 0

    assert arms["dense"]["semantic_recall_micro"] == pytest.approx(0.5)
    assert arms["rrf"]["recalled_gold"] == 2
    assert arms["rrf"]["gold_total"] == 2
    assert result["arms"]["candidate_coverage"]["queries_with_all_gold_in_rrf_top_k"] == 2
    assert result["arms"]["candidate_coverage"]["dense_only"] == 1
    assert result["arms"]["candidate_coverage"]["sparse_only"] == 1
    assert result["arms"]["candidate_coverage"]["neither"] == 0
    assert result["arms"]["rrf_sources"] == {"dense_only": 2, "sparse_only": 1, "both": 1}


def test_supplementary_precision_and_ndcg_are_defined_and_aggregated() -> None:
    """P@K and binary nDCG@K are computed per query and aggregated independently."""
    result = _run()
    arms = result["arms"]["arms"]

    # Dense: Q-1 hits gold at rank 1; Q-2 misses. Fixed K=2 denominator.
    assert arms["dense"]["precision_at_k_macro"] == pytest.approx(0.25)
    assert arms["dense"]["precision_at_k_micro"] == pytest.approx(0.25)
    assert arms["dense"]["precision_at_k_ceiling_macro"] == pytest.approx(0.5)
    assert arms["dense"]["ndcg_at_k_macro"] == pytest.approx(0.5)

    # Sparse OR: Q-1 misses; Q-2 has a single gold at rank 1.
    assert arms["sparse_or"]["precision_at_k_macro"] == pytest.approx(0.25)
    assert arms["sparse_or"]["ndcg_at_k_macro"] == pytest.approx(0.5)

    # Sparse AND returns no documents: zero, not a missing value.
    assert arms["sparse_and"]["precision_at_k_macro"] == 0.0
    assert arms["sparse_and"]["precision_at_k_micro"] == 0.0
    assert arms["sparse_and"]["ndcg_at_k_macro"] == 0.0

    # Candidate union: gold at rank 1 and rank 2 in the two queries.
    assert arms["candidate_union"]["precision_at_k_macro"] == pytest.approx(0.5)
    assert arms["candidate_union"]["precision_at_k_micro"] == pytest.approx(0.5)
    assert arms["candidate_union"]["ndcg_at_k_macro"] == pytest.approx(
        0.8154648767857287
    )

    # RRF: both gold docs at effective rank 2, so nDCG = 1/log2(3).
    assert arms["rrf"]["precision_at_k_macro"] == pytest.approx(0.5)
    assert arms["rrf"]["precision_at_k_micro"] == pytest.approx(0.5)
    assert arms["rrf"]["ndcg_at_k_macro"] == pytest.approx(0.6309297535714574)


def test_supplementary_metrics_are_not_coupled_to_hit_at_1_or_mrr() -> None:
    """A rank-2 gold gives Hit@1=False and MRR=0.5 but nonzero P@K and nDCG@K."""
    result = _run()
    record = result["per_query"][0]["arms"]["rrf"]

    assert record["hit_at_1"] is False
    assert record["reciprocal_rank"] == pytest.approx(0.5)
    assert record["precision_at_k"] == pytest.approx(0.5)
    assert record["ndcg_at_k"] == pytest.approx(0.6309297535714574)


def test_output_declares_the_label_and_metric_contract() -> None:
    """The artifact itself states the version, label limit and relevance scheme."""
    result = _run()
    assert result["metric_version"] == 2
    assert result["labels_exhaustive"] is False
    assert result["relevance_scheme"] == "binary_known_gold"
    assert "metric_version" in result["definitions"]
    assert "labels_exhaustive" in result["definitions"]
    assert "relevance_scheme" in result["definitions"]


def test_measure_deduplicates_repeated_doc_ids_before_the_cutoff() -> None:
    """Two chunks of one document occupy one effective rank, not two."""
    query = EvalQuery(query_id="Q-DUP", query="dup", relevant_doc_ids=("DOC-B",))
    hits = [
        _hit("C1", source="dense", rank=1, doc_id="DOC-A"),
        _hit("C2", source="dense", rank=2, doc_id="DOC-A"),
        _hit("C3", source="dense", rank=3, doc_id="DOC-B"),
    ]
    record = _measure(query, hits, top_k=2)

    # Without dedupe the gold DOC-B would fall outside the Top-2.
    assert record["doc_unique_count"] == 2
    assert record["doc_ids"] == ["DOC-A", "DOC-B"]
    assert record["candidate_doc_ids"] == ["DOC-A", "DOC-B"]
    assert record["precision_at_k"] == pytest.approx(0.5)
    assert record["precision_at_k_ceiling"] == pytest.approx(0.5)
    assert record["ndcg_at_k"] == pytest.approx(0.6309297535714574)


def test_per_query_records_keep_raw_candidates_and_provenance() -> None:
    """Every metric can be recomputed from the stored ranked candidates."""
    result = _run()
    first = result["per_query"][0]

    assert first["query_id"] == "Q-1"
    assert first["arms"]["dense"]["doc_ids"] == ["DOC-A", "DOC-B"]
    assert first["arms"]["sparse_or"]["doc_ids"] == ["DOC-B"]
    assert first["arms"]["rrf"]["doc_ids"] == ["DOC-B", "DOC-A"]
    assert first["arms"]["rrf"]["gold_ranks"] == {"DOC-A": 2}
    assert first["arms"]["rrf"]["first_gold_rank"] == 2
    assert first["arms"]["rrf"]["reciprocal_rank"] == pytest.approx(0.5)
    assert first["arms"]["dense"]["precision_at_k"] == pytest.approx(0.5)
    assert first["arms"]["dense"]["precision_at_k_ceiling"] == pytest.approx(0.5)
    assert first["arms"]["dense"]["ndcg_at_k"] == 1.0
    assert first["arms"]["sparse_or"]["precision_at_k"] == 0.0
    assert first["arms"]["sparse_or"]["ndcg_at_k"] == 0.0
    assert first["arms"]["rrf"]["precision_at_k"] == pytest.approx(0.5)
    assert first["arms"]["rrf"]["ndcg_at_k"] == pytest.approx(0.6309297535714574)

    fused_top = first["arms"]["rrf"]["hits"][0]
    assert fused_top["chunk_id"] == "DOC-B"
    assert fused_top["candidate_sources"] == ["dense", "sparse"]
    assert fused_top["dense_rank"] == 2
    assert fused_top["sparse_rank"] == 1
    assert fused_top["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused_top["score"] == pytest.approx(fused_top["rrf_score"])

    assert first["deltas"]["rrf_vs_dense_reciprocal_rank"] == pytest.approx(-0.5)
    assert first["deltas"]["rrf_vs_sparse_or_reciprocal_rank"] == pytest.approx(0.5)
    assert "_latency" not in first
    json.dumps(result)


def test_latency_samples_match_successful_timed_passes() -> None:
    """Two queries x two repetitions yields four samples per arm."""
    result = _run()
    samples = result["latency_samples"]
    for arm in ARM_NAMES:
        assert len(samples[arm]) == 4
        assert all(value >= 0.0 for value in samples[arm])
        assert result["arms"]["arms"][arm]["latency_ms"]["n"] == 4


def test_repeated_passes_are_consistent() -> None:
    """Stable fake results produce no mismatch samples."""
    result = _run()
    assert result["totals"]["quality_mismatch_samples"] == 0
    assert result["totals"]["errors"] == 0
    assert result["totals"]["warmup_errors"] == 0


def test_order_mismatch_across_passes_is_reported_not_hidden() -> None:
    """A non-deterministic arm downgrades the status instead of being ignored."""
    dense, sparse_or, sparse_and = _retrievers()
    result = _run(
        config=build_config(top_k=2, candidate_k=2, repetitions=2, warmup_passes=0),
        dense=_FlipRetriever(dense.mapping),
        sparse_or=sparse_or,
        sparse_and=sparse_and,
    )
    assert result["totals"]["quality_mismatch_samples"] > 0
    assert result["status"] == "PASS WITH MINOR ISSUES"


# ------------------------------------------------------------- error handling


def test_one_query_failure_is_isolated_and_fails_the_status() -> None:
    """A dense failure for one query does not hide the other query's results."""
    dense, sparse_or, sparse_and = _retrievers()
    result = _run(
        config=build_config(top_k=2, candidate_k=2, repetitions=1, warmup_passes=0),
        dense=_MapRetriever(dense.mapping, error_queries=("q1",)),
        sparse_or=sparse_or,
        sparse_and=sparse_and,
    )
    assert result["status"] == "FAIL"
    assert result["totals"]["errors"] >= 1
    first = result["per_query"][0]
    assert first["arms"]["dense"]["evaluated"] is False
    assert first["arms"]["dense"]["precision_at_k"] is None
    assert first["arms"]["dense"]["precision_at_k_ceiling"] is None
    assert first["arms"]["dense"]["ndcg_at_k"] is None
    assert first["arms"]["sparse_or"]["evaluated"] is True
    assert first["arms"]["sparse_and"]["evaluated"] is True
    assert first["arms"]["rrf"]["evaluated"] is False
    assert first["arms"]["candidate_union"]["evaluated"] is False
    assert result["per_query"][1]["arms"]["dense"]["evaluated"] is True


def test_warmup_failure_is_recorded_and_fails_the_status() -> None:
    """Warm-up errors are visible, not silently discarded."""
    dense, sparse_or, sparse_and = _retrievers()
    result = _run(
        config=build_config(top_k=2, candidate_k=2, repetitions=1, warmup_passes=1),
        dense=_MapRetriever(dense.mapping, error_queries=("q1",)),
        sparse_or=sparse_or,
        sparse_and=sparse_and,
    )
    assert result["totals"]["warmup_errors"] >= 1
    assert result["status"] == "FAIL"
    assert any(error["stage"] == "warmup" for error in result["errors"])


# ---------------------------------------------------------------- focus check


def test_focus_misses_are_located_in_every_arm() -> None:
    """Q-004/DOC-0017 is absent from dense and present in sparse/RRF."""
    queries = [EvalQuery(query_id="Q-004", query="notice period", relevant_doc_ids=("DOC-0017",))]
    dense = _MapRetriever({"notice period": [_hit("DOC-9999", source="dense", rank=1)]})
    sparse = _MapRetriever({"notice period": [_hit("DOC-0017", source="sparse", rank=1)]})
    result = run_hybrid_benchmark(
        build_config(top_k=2, candidate_k=2, repetitions=1, warmup_passes=0),
        queries,
        dense_retriever=dense,
        sparse_or_retriever=sparse,
        sparse_and_retriever=_MapRetriever({"notice period": []}),
        environment=ENVIRONMENT,
        dataset={"queries": 1},
    )
    row = result["focus_misses"][0]
    assert row["query_id"] == "Q-004"
    assert row["doc_id"] == "DOC-0017"
    assert row["dense_rank"] is None
    assert row["sparse_rank"] == 1
    assert row["rrf_rank"] == 1
    assert row["in_dense_top_k"] is False
    assert row["in_rrf_top_k"] is True
    assert row["rrf_score"] == pytest.approx(1 / 61)


# ----------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 3, "candidate_k": 2}, "top_k"),
        ({"top_k": 0, "candidate_k": 2}, "top_k"),
        ({"top_k": True, "candidate_k": 2}, "top_k"),
        ({"candidate_k": 0, "top_k": 1}, "candidate_k"),
        ({"rrf_k": 0, "candidate_k": 2, "top_k": 1}, "k"),
        ({"repetitions": 0}, "repetitions"),
        ({"warmup_passes": -1}, "warmup_passes"),
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, object], message: str) -> None:
    """Configuration errors are raised before retrieval starts."""
    with pytest.raises(BenchmarkError, match=message):
        build_config(**kwargs)  # type: ignore[arg-type]


def test_run_rejects_a_non_config_object() -> None:
    """The runner does not silently accept a dict of parameters."""
    with pytest.raises(BenchmarkError, match="HybridBenchmarkConfig"):
        run_hybrid_benchmark("not-a-config")  # type: ignore[arg-type]


def test_run_rejects_an_empty_query_list() -> None:
    """An empty evaluation is a configuration error."""
    with pytest.raises(BenchmarkError, match="at least one"):
        run_hybrid_benchmark(build_config(), [], environment=ENVIRONMENT, dataset=DATASET)


# --------------------------------------------------------------------- output


def test_write_outputs_writes_json_and_markdown(tmp_path: Path) -> None:
    """The JSON can be re-read independently; the Markdown names the status."""
    result = _run()
    json_path = tmp_path / "hybrid.json"
    markdown_path = tmp_path / "hybrid.md"

    written_json, written_md = write_outputs(result, json_path, markdown_path)

    assert written_json == json_path
    assert written_md == markdown_path
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert reloaded["status"] == "PASS"
    assert len(reloaded["per_query"]) == 2
    assert reloaded["parameters"]["rrf_k"] == 60

    report = markdown_path.read_text(encoding="utf-8")
    assert "# Hybrid Search / RRF 离线评测（阶段 ⑥-B）" in report
    assert "## 五臂对比" in report
    assert "### 补充指标：Precision@K / 二元 nDCG@K" in report
    assert "P@K ceiling macro" in report
    assert "labels_exhaustive=False" in report
    assert "## 五个 Dense 漏检 gold 的融合排名" in report
    assert "| rrf |" in report


def test_render_markdown_handles_missing_arm_values() -> None:
    """A failed arm renders as '-' rather than crashing or printing None."""
    result = _run()
    result["arms"]["arms"]["dense"]["mrr"] = None
    report = render_markdown(result)
    assert "| dense |" in report
