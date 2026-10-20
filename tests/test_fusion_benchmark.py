"""RRF sensitivity sweep over a scripted retriever (no DB, no model)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from src.evaluation.benchmark import Arm
from src.evaluation.dense_benchmark import EvalQuery
from src.evaluation.fusion_benchmark import (
    FusionConfig,
    cell_key,
    render_markdown,
    run_fusion_sensitivity,
    write_outputs,
)
from src.schema.retrieval import SearchHit


class _ScriptedRetriever:
    """Return a fixed doc ranking; optionally raise on selected queries."""

    name = "scripted"

    def __init__(self, results: dict[str, list[str]], *, fail_on: Sequence[str] = ()) -> None:
        """Store the ranking and the failure set."""
        self._results = results
        self._fail_on = set(fail_on)

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return one hit per scripted doc."""
        if query in self._fail_on:
            raise RuntimeError(f"scripted failure for {query!r}")
        hits = [
            SearchHit(
                chunk_id=f"{doc_id}-c0",
                doc_id=doc_id,
                score=1.0 / (index + 1),
                rank=index + 1,
                source="dense",
                chunk_strategy="document",
            )
            for index, doc_id in enumerate(self._results.get(query, []))
        ]
        return hits[:top_k] if top_k is not None else hits


def _queries(count: int = 30) -> list[EvalQuery]:
    """Build labelled queries with one gold doc each."""
    return [
        EvalQuery(
            query_id=f"q{index}",
            query=f"question {index}",
            relevant_doc_ids=(f"d{index}",),
        )
        for index in range(count)
    ]


def _runs(
    config: FusionConfig,
    queries: Sequence[EvalQuery],
    *,
    fail_cells: Sequence[tuple[int, int]] = (),
) -> dict[tuple[int, int], list[Arm]]:
    """One perfect arm per cell, except for the explicitly failing cells."""
    runs: dict[tuple[int, int], list[Arm]] = {}
    for cell in config.cells:
        fail_on = [query.query for query in queries] if cell in fail_cells else []
        retriever = _ScriptedRetriever(
            {query.query: list(query.relevant_doc_ids) for query in queries},
            fail_on=fail_on,
        )
        runs[cell] = [Arm(name="rrf", retriever=retriever)]
    return runs


def _run(
    queries: Sequence[EvalQuery] | None = None,
    *,
    config: FusionConfig | None = None,
    fail_cells: Sequence[tuple[int, int]] = (),
) -> dict[str, Any]:
    """Run the sweep with scripted arms."""
    selected = list(queries if queries is not None else _queries(30))
    resolved = config if config is not None else FusionConfig()
    return run_fusion_sensitivity(
        _runs(resolved, selected, fail_cells=fail_cells), selected, resolved
    )


def test_default_grid_and_cell_order() -> None:
    """The default grid is 3 rrf_k values by 2 candidate_k values."""
    config = FusionConfig()
    assert config.rrf_k_values == (10, 60, 100)
    assert config.candidate_k_values == (10, 50)
    assert config.cells[0] == (10, 10)
    assert config.cells[-1] == (100, 50)
    assert cell_key(60, 10) == "rrf_k=60,candidate_k=10"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rrf_k_values": ()}, "at least one rrf_k"),
        ({"candidate_k_values": ()}, "at least one candidate_k"),
        ({"rrf_k_values": (60, 60)}, "must not repeat"),
        ({"rrf_k_values": (0,)}, "positive ints"),
        ({"candidate_k_values": (5,)}, "exceeds the smallest candidate_k"),
        ({"chunk_strategy": "contextual"}, "unsupported chunk_strategy"),
    ],
)
def test_invalid_configurations_are_rejected(kwargs: dict[str, Any], message: str) -> None:
    """Bad grids fail before any retrieval."""
    with pytest.raises(ValueError, match=message):
        FusionConfig(**kwargs)


def test_missing_extra_or_empty_cells_are_rejected() -> None:
    """The supplied runs must match the configured grid exactly."""
    queries = _queries(1)
    config = FusionConfig()
    runs = _runs(config, queries)
    del runs[(60, 50)]
    with pytest.raises(ValueError, match="no arms supplied"):
        run_fusion_sensitivity(runs, queries, config)
    runs = _runs(config, queries)
    runs[(999, 10)] = runs[(10, 10)]
    with pytest.raises(ValueError, match="unconfigured grid cell"):
        run_fusion_sensitivity(runs, queries, config)
    runs = _runs(config, queries)
    runs[(10, 10)] = []
    with pytest.raises(ValueError, match="has no arms"):
        run_fusion_sensitivity(runs, queries, config)


def test_complete_grid_is_pass_and_records_every_cell() -> None:
    """Six cells × one arm produce six summary rows."""
    result = _run()
    assert result["status"] == "PASS"
    assert len(result["summary"]) == 6
    assert result["parameters"]["cells"][1] == "rrf_k=10,candidate_k=50"
    for row in result["summary"]:
        assert row["recall_macro"] == 1.0
        assert row["evaluated"] == 30
        assert row["latency_ms"]["p99"] is not None


def test_one_failing_cell_makes_the_sweep_partial() -> None:
    """A failed cell is named instead of being dropped."""
    failing = (60, 10)
    result = _run(fail_cells=[failing])
    assert result["status"] == "PARTIAL"
    assert "rrf_k=60,candidate_k=10" in result["status_reason"]
    row = next(
        row
        for row in result["summary"]
        if row["rrf_k"] == 60 and row["candidate_k"] == 10
    )
    assert row["failed"] == 30


def test_a_short_query_set_is_not_pass() -> None:
    """Three queries cannot stand in for the 30-query suite."""
    result = _run(_queries(3))
    assert result["status"] == "PARTIAL"
    assert "3 of 30" in result["status_reason"]


def test_markdown_has_the_grid_and_limitations() -> None:
    """The report shows rrf_k, candidate_k, percentile latency and the scale caveat."""
    rendered = render_markdown(_run())
    assert "# RRF 参数敏感性" in rendered
    assert "| 60 | 10 | rrf |" in rendered
    assert "P99 ms" in rendered
    assert "语料规模不足以区分参数" in rendered
    assert "不代表答案正确性" in rendered


def test_write_outputs_refuses_to_clobber(tmp_path: Path) -> None:
    """The sweep shares the no-clobber rule."""
    result = _run()
    json_path = tmp_path / "fusion.json"
    report_path = tmp_path / "fusion.md"
    write_outputs(result, json_path, report_path)
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        write_outputs(result, json_path, report_path)
    write_outputs(result, json_path, report_path, overwrite=True)
