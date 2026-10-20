"""Chunking comparison orchestration with scripted retrievers.

No database and no model: each strategy receives a stub retriever, so the
assertions cover status merging, document-level capture, rendering and the
no-clobber rule. The real ingest/retrieval path is exercised by
``scripts/benchmark_chunking.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from src.evaluation.benchmark import Arm
from src.evaluation.chunking_benchmark import (
    STRATEGIES,
    ChunkingConfig,
    render_markdown,
    run_chunking_benchmark,
    write_outputs,
)
from src.evaluation.dense_benchmark import EvalQuery
from src.schema.retrieval import SearchHit


class _ScriptedRetriever:
    """Return a fixed doc order per query; optionally raise."""

    name = "scripted"

    def __init__(
        self,
        results: dict[str, list[str]],
        *,
        fail_on: Sequence[str] = (),
        chunks_per_doc: int = 1,
    ) -> None:
        """Store the scripted ranking, failures and chunk multiplicity."""
        self._results = results
        self._fail_on = set(fail_on)
        self._chunks_per_doc = chunks_per_doc

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return chunk hits, several per doc when configured."""
        if query in self._fail_on:
            raise RuntimeError(f"scripted failure for {query!r}")
        hits: list[SearchHit] = []
        for doc_id in self._results.get(query, []):
            for chunk_index in range(self._chunks_per_doc):
                hits.append(
                    SearchHit(
                        chunk_id=f"{doc_id}-c{chunk_index}",
                        doc_id=doc_id,
                        score=1.0 / (len(hits) + 1),
                        rank=len(hits) + 1,
                        source="dense",
                        chunk_strategy="document",
                    )
                )
        return hits[:top_k] if top_k is not None else hits


def _queries(count: int = 30) -> list[EvalQuery]:
    """Build ``count`` labelled queries, each with one gold doc."""
    return [
        EvalQuery(
            query_id=f"q{index}",
            query=f"question {index}",
            relevant_doc_ids=(f"d{index}",),
        )
        for index in range(count)
    ]


def _arms(queries: Sequence[EvalQuery], **kwargs: Any) -> list[Arm]:
    """One scripted arm that finds every gold doc first."""
    retriever = _ScriptedRetriever(
        {query.query: list(query.relevant_doc_ids) for query in queries}, **kwargs
    )
    return [Arm(name="stub", retriever=retriever)]


def _run(
    queries: Sequence[EvalQuery] | None = None,
    *,
    strategies: tuple[str, ...] = ("document", "fixed_window"),
    stats: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the comparison with one perfect arm per strategy."""
    selected = list(queries if queries is not None else _queries(30))
    runs = {name: _arms(selected) for name in strategies}
    config = ChunkingConfig(strategies=strategies)
    return run_chunking_benchmark(
        runs,
        selected,
        config,
        chunk_stats=stats
        if stats is not None
        else {name: {"chunk_count": 150, "documents": 150} for name in strategies},
    )


def test_default_strategy_list_is_complete_and_ordered() -> None:
    """The report order is document, fixed_window, structured."""
    assert STRATEGIES == ("document", "fixed_window", "structured")
    assert ChunkingConfig().strategies == STRATEGIES


def test_unknown_or_duplicate_strategies_are_rejected() -> None:
    """A typo or a repeated strategy cannot select a hidden run."""
    with pytest.raises(ValueError, match="unknown chunking strategies"):
        ChunkingConfig(strategies=("document", "contextual"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not repeat"):
        ChunkingConfig(strategies=("document", "document"))
    with pytest.raises(ValueError, match="at least one chunking strategy"):
        ChunkingConfig(strategies=())


def test_runs_must_match_the_configured_strategies() -> None:
    """Missing strategies fail before any arm runs; extras are refused too."""
    queries = _queries(1)
    config = ChunkingConfig(strategies=("document", "fixed_window"))
    with pytest.raises(ValueError, match="no arms supplied"):
        run_chunking_benchmark(
            {"document": _arms(queries)}, queries, config=config
        )
    with pytest.raises(ValueError, match="unconfigured"):
        run_chunking_benchmark(
            {
                "document": _arms(queries),
                "fixed_window": _arms(queries),
                "structured": _arms(queries),
            },
            queries,
            config=config,
        )


def test_a_strategy_without_arms_is_rejected() -> None:
    """An empty arm list cannot produce a meaningful comparison."""
    queries = _queries(1)
    config = ChunkingConfig(strategies=("document",))
    with pytest.raises(ValueError, match="has no arms"):
        run_chunking_benchmark({"document": []}, queries, config=config)


def test_full_runs_are_pass_and_keep_every_strategy_separate() -> None:
    """Two complete strategies with no failures produce PASS."""
    result = _run()
    assert result["status"] == "PASS"
    assert result["parameters"]["strategies"] == ["document", "fixed_window"]
    assert list(result["strategies"]) == ["document", "fixed_window"]
    assert len(result["summary"]) == 2
    assert result["chunk_stats"]["document"]["chunk_count"] == 150
    for row in result["summary"]:
        assert row["recall_macro"] == 1.0
        assert row["latency_ms"]["p99"] is not None


def test_document_level_capture_collapses_multiple_chunks() -> None:
    """Several chunks of one doc cannot inflate Recall or MRR."""
    selected = _queries(30)
    runs = {
        "document": _arms(selected),
        "fixed_window": _arms(selected, chunks_per_doc=3),
    }
    result = run_chunking_benchmark(
        runs, selected, ChunkingConfig(strategies=("document", "fixed_window"))
    )
    for row in result["summary"]:
        assert row["recall_macro"] == 1.0
        assert row["mrr"] == 1.0


def test_one_failing_strategy_makes_the_comparison_partial() -> None:
    """A failed strategy is named, not silently dropped."""
    selected = _queries(30)
    failing = _arms(selected, fail_on=[query.query for query in selected])
    result = run_chunking_benchmark(
        {"document": _arms(selected), "fixed_window": failing},
        selected,
        ChunkingConfig(strategies=("document", "fixed_window")),
    )
    assert result["status"] == "PARTIAL"
    assert "fixed_window" in result["status_reason"]
    assert result["summary"][-1]["failed"] == 30


def test_a_short_query_set_is_not_pass() -> None:
    """Three queries cannot stand in for the 30-query suite."""
    result = _run(_queries(3))
    assert result["status"] == "PARTIAL"
    assert "3 of 30" in result["status_reason"]


def test_markdown_reports_stats_metrics_latency_and_limits() -> None:
    """The rendered report carries every table a reviewer needs."""
    rendered = render_markdown(_run())
    assert "# 分块策略对比" in rendered
    assert "## 1. 分块统计" in rendered
    assert "## 3. 延迟" in rendered
    assert "P99 ms" in rendered
    assert "按文档级计算" in rendered
    assert "不代表答案正确性" in rendered
    assert "`fixed_window`" in rendered


def test_write_outputs_refuses_to_clobber(tmp_path: Path) -> None:
    """The comparison shares the no-clobber rule."""
    result = _run()
    json_path = tmp_path / "chunking.json"
    report_path = tmp_path / "chunking.md"
    write_outputs(result, json_path, report_path)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        write_outputs(result, json_path, report_path)
    write_outputs(result, json_path, report_path, overwrite=True)
