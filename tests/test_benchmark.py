"""Unified benchmark orchestration with scripted retrievers.

No database, no model and no network: each test injects a retriever that returns
a fixed document order, so the assertions are about metric and status rules, not
about pgvector.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from src.evaluation.benchmark import (
    Arm,
    BenchmarkConfig,
    evaluate_arm,
    render_markdown,
    run_benchmark,
    write_outputs,
)
from src.evaluation.dense_benchmark import EvalQuery
from src.schema.retrieval import SearchHit


class _ScriptedRetriever:
    """Return a fixed doc order per query text; optionally raise."""

    name = "scripted"

    def __init__(
        self,
        results: dict[str, list[str]],
        *,
        fail_on: Sequence[str] = (),
        duplicate_first: bool = False,
    ) -> None:
        """Store the scripted doc ids and failure set."""
        self._results = results
        self._fail_on = set(fail_on)
        self._duplicate_first = duplicate_first

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return one hit per doc id, or raise for a scripted failure."""
        if query in self._fail_on:
            raise RuntimeError(f"scripted failure for {query!r}")
        doc_ids = self._results.get(query, [])
        if self._duplicate_first and doc_ids:
            doc_ids = [doc_ids[0], doc_ids[0], *doc_ids[1:]]
        hits = [
            SearchHit(
                chunk_id=f"{doc_id}-c0",
                doc_id=doc_id,
                score=1.0 / (index + 1),
                rank=index + 1,
                source="dense",
                chunk_strategy="document",
            )
            for index, doc_id in enumerate(doc_ids)
        ]
        return hits[:top_k] if top_k is not None else hits


def _query(query_id: str, gold: Sequence[str], text: str | None = None) -> EvalQuery:
    """Build one labelled query."""
    return EvalQuery(
        query_id=query_id,
        query=text or f"question {query_id}",
        relevant_doc_ids=tuple(gold),
    )


def _single_arm_run(
    gold: Sequence[str],
    ranking: Sequence[str],
    *,
    duplicate_first: bool = False,
    top_k: int = 10,
) -> dict[str, Any]:
    """Run one query through one arm and return the arm record."""
    query = _query("q1", gold)
    arm = Arm(
        name="stub",
        retriever=_ScriptedRetriever(
            {query.query: list(ranking)}, duplicate_first=duplicate_first
        ),
    )
    return evaluate_arm(arm, [query], top_k=top_k)


def test_perfect_ranking_scores_one() -> None:
    """A one-document gold set found at rank 1 scores 1.0 on rank metrics."""
    record = _single_arm_run(["d1", "d2"], ["d1", "d2", "d3", "d4", "d5"])
    metrics = record["metrics"]
    assert metrics["recall_at_k"] == {"macro": 1.0, "micro": 1.0}
    assert metrics["mrr"] == 1.0
    assert metrics["hit_at_1"] == 1
    assert metrics["ndcg_at_k"]["macro"] == 1.0
    assert metrics["precision_at_k"]["macro"] == pytest.approx(0.2)
    assert metrics["precision_at_k"]["ceiling_macro"] == pytest.approx(0.2)


def test_gold_at_rank_two_halves_mrr() -> None:
    """MRR is the reciprocal of the first relevant rank, not a hit rate."""
    record = _single_arm_run(["d2"], ["d1", "d2"])
    assert record["metrics"]["mrr"] == pytest.approx(0.5)
    assert record["metrics"]["hit_at_1"] == 0


def test_chunks_of_one_document_collapse_to_one_rank() -> None:
    """Two chunks of the same doc cannot occupy two effective ranks."""
    record = _single_arm_run(["d2"], ["d1", "d2", "d3"], duplicate_first=True)
    per_query = record["per_query"][0]
    assert per_query["doc_ids"] == ["d1", "d2", "d3"]
    assert per_query["unique_docs"] == 3
    assert record["metrics"]["mrr"] == pytest.approx(0.5)
    assert record["metrics"]["precision_at_k"]["macro"] == pytest.approx(0.1)


def test_recall_denominator_is_the_gold_set() -> None:
    """Returning one of two gold docs scores 0.5 even at K=10."""
    record = _single_arm_run(["d1", "d9"], ["d1"])
    assert record["metrics"]["recall_at_k"]["macro"] == pytest.approx(0.5)
    assert record["metrics"]["recall_at_k"]["micro"] == pytest.approx(0.5)


def test_failed_query_is_excluded_not_scored_zero() -> None:
    """A failure is recorded and the aggregate uses only valid queries."""
    queries = [_query("q1", ["d1"]), _query("q2", ["d2"])]
    retriever = _ScriptedRetriever({"question q1": ["d1"]}, fail_on=["question q2"])
    record = evaluate_arm(Arm(name="stub", retriever=retriever), queries, top_k=10)
    assert record["status"] == "PARTIAL"
    assert record["failed_queries"] == ["q2"]
    assert record["metrics"]["evaluated_queries"] == 1
    assert record["metrics"]["failed_queries"] == 1
    assert record["metrics"]["recall_at_k"]["macro"] == 1.0
    assert record["latency_ms"]["query"]["n"] == 1


def test_empty_results_are_scored_as_misses() -> None:
    """An empty retrieval is a real zero, unlike a raised error."""
    record = _single_arm_run(["d1"], [])
    assert record["metrics"]["recall_at_k"]["macro"] == 0.0
    assert record["metrics"]["mrr"] == 0.0
    assert record["metrics"]["failed_queries"] == 0


def test_full_query_set_is_pass_and_partial_set_is_explicitly_partial() -> None:
    """30/30 is PASS; a smoke subset must not claim it."""
    queries = [_query(f"q{index}", [f"d{index}"]) for index in range(30)]
    retriever = _ScriptedRetriever(
        {query.query: list(query.relevant_doc_ids) for query in queries}
    )
    full = run_benchmark([Arm(name="stub", retriever=retriever)], queries)
    assert full["status"] == "PASS"
    assert full["parameters"]["queries"] == 30

    partial = run_benchmark([Arm(name="stub", retriever=retriever)], queries[:3])
    assert partial["status"] == "PARTIAL"
    assert "3 of 30" in partial["status_reason"]


def test_no_valid_records_is_fail() -> None:
    """An arm that raises everywhere cannot produce a PASS or a zero score."""
    queries = [_query("q1", ["d1"])]
    retriever = _ScriptedRetriever({}, fail_on=["question q1"])
    result = run_benchmark([Arm(name="stub", retriever=retriever)], queries)
    assert result["status"] == "FAIL"
    assert result["arms"][0]["metrics"]["evaluated_queries"] == 0
    assert result["arms"][0]["metrics"]["recall_at_k"]["macro"] is None


def test_markdown_reports_p99_and_the_label_limit() -> None:
    """The rendered report carries the unified latency and label caveats."""
    query = _query("q1", ["d1"])
    retriever = _ScriptedRetriever({query.query: ["d1"]})
    result = run_benchmark([Arm(name="stub", retriever=retriever)], [query])
    rendered = render_markdown(result)
    assert "# 统一检索评测" in rendered
    assert "P99 ms" in rendered
    assert "ceiling" in rendered
    assert "不代表答案正确性" in rendered


def test_write_outputs_refuses_to_replace_an_artifact(tmp_path: Path) -> None:
    """The unified writer shares the no-clobber rule."""
    query = _query("q1", ["d1"])
    retriever = _ScriptedRetriever({query.query: ["d1"]})
    result = run_benchmark([Arm(name="stub", retriever=retriever)], [query])
    json_path = tmp_path / "unified.json"
    report_path = tmp_path / "unified.md"
    write_outputs(result, json_path, report_path)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        write_outputs(result, json_path, report_path)
    write_outputs(result, json_path, report_path, overwrite=True)


def test_config_rejects_an_impossible_top_k() -> None:
    """A final list longer than the candidate pool is rejected up front."""
    with pytest.raises(ValueError, match="must not exceed candidate_k"):
        BenchmarkConfig(top_k=50, candidate_k=10)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_config_rejects_invalid_counts(value: object) -> None:
    """Counts must be positive ints."""
    with pytest.raises(ValueError, match="must be a positive int"):
        BenchmarkConfig(top_k=value)  # type: ignore[arg-type]
