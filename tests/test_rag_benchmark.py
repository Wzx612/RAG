"""Phase 8: RAG evaluation artifact, metrics and report safety."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from src.evaluation.dense_benchmark import BenchmarkError, EvalQuery
from src.evaluation.rag_benchmark import (
    RAGBenchmarkConfig,
    load_raw_queries,
    render_markdown,
    run_rag_benchmark,
    write_outputs,
)
from src.rag.pipeline import RAGPipeline
from src.rag.provider import LLMError, MockLLMProvider
from src.schema.rag import LLMResponse, RAGConfig
from src.schema.retrieval import SearchHit

QUERIES = [
    EvalQuery("Q-001", "What is the annual fee?", ("DOC-1", "DOC-2")),
    EvalQuery("Q-002", "What is the late fee?", ("DOC-3",)),
    EvalQuery("Q-003", "Is there a travel benefit?", ("DOC-4",)),
]
RAW = {
    "Q-001": {"query": "What is the annual fee?", "gold_response": "Gold answer 1"},
    "Q-002": {"query": "What is the late fee?", "gold_response": "Gold answer 2"},
    "Q-003": {"query": "Is there a travel benefit?", "gold_response": "Gold answer 3"},
}


def _hit(chunk_id: str, doc_id: str, *, rank: int = 1) -> SearchHit:
    """Build one live-shaped hit."""
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        score=0.9,
        distance=0.1,
        rank=rank,
        source="reranked",
        content=f"Evidence text for {chunk_id} in {doc_id}.",
        chunk_strategy="document",
        chunk_index=0,
        title=f"Title {chunk_id}",
        category="fees",
        card_tier="standard",
        effective_year=2025,
        policy_code="P-1",
        rerank_score=0.8,
    )


class _Retriever:
    """Return fixed hits per query, recording the call."""

    name = "fake-reranked"

    def __init__(self, hits_by_query: Mapping[str, list[SearchHit]]) -> None:
        """Store the per-query pools."""
        self.hits_by_query = hits_by_query
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return the configured pool or an empty list."""
        self.calls.append((query, top_k))
        return list(self.hits_by_query.get(query, []))


class _FailingProvider:
    """A real-looking provider that always fails with a timeout."""

    name = "failing"
    model = "failing-v1"
    is_real = True

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Raise a timeout without touching a network."""
        del messages, temperature, max_tokens
        raise LLMError("timed out", category="timeout", retryable=True, attempts=3)


def _pipeline(provider: Any | None = None) -> RAGPipeline:
    """Build the fake pipeline used by the benchmark tests."""
    retriever = _Retriever(
        {
            "What is the annual fee?": [_hit("c1", "DOC-1"), _hit("c2", "DOC-2", rank=2)],
            "What is the late fee?": [],
            "Is there a travel benefit?": [_hit("c3", "DOC-5")],
        }
    )
    return RAGPipeline(
        retriever,
        provider if provider is not None else MockLLMProvider(),
        RAGConfig(top_k=2, max_context_chars=2000, max_chunk_chars=1000),
    )


def _run(pipeline: RAGPipeline | None = None) -> dict[str, Any]:
    """Run the benchmark over three queries with fixed provenance."""
    return run_rag_benchmark(
        pipeline or _pipeline(),
        QUERIES,
        raw_queries=RAW,
        config=RAGBenchmarkConfig(
            top_k=2,
            candidates=2,
            max_context_chars=2000,
            max_chunk_chars=1000,
            provider_kind="mock",
        ),
        environment={"documents": 150, "chunks": 150},
        dataset={"queries": 3, "hashes": {"kb": "abc"}},
    )


def test_mock_evaluation_records_raw_results_and_is_labelled_partial() -> None:
    """A mock run is useful for wiring but must never claim real model quality."""
    result = _run()

    assert result["status"] == "PARTIAL"
    assert "mock" in result["status_reason"].lower()
    assert result["provider"]["is_real"] is False
    assert result["provider"]["real_llm_calls"] is False
    assert result["provider"]["real_llm_evaluation"] == "not_run"
    assert result["provider"]["api_key_present"] is False
    assert result["provider"]["llm_judge"]["executed"] is False
    assert result["metrics"]["answer_correctness"]["value"] is None
    assert result["metrics"]["faithfulness"]["value"] is None
    assert result["metrics"]["answer_relevance"]["value"] is None
    assert result["gold_usage"]["gold_response_used_for_scoring"] is False
    assert len(result["per_query"]) == 3

    by_id = {record["query_id"]: record for record in result["per_query"]}
    assert by_id["Q-001"]["status"] == "ok"
    assert by_id["Q-001"]["evidence_doc_ids"] == ["DOC-1", "DOC-2"]
    assert by_id["Q-001"]["evidence_labels"] == ["E1", "E2"]
    assert by_id["Q-001"]["citations"][0]["label"] == "E1"
    assert by_id["Q-001"]["citation_status"] == "ok"
    assert by_id["Q-001"]["prompt_sha256"]
    assert by_id["Q-001"]["context_sha256"]
    assert by_id["Q-001"]["gold_response"] == "Gold answer 1"
    assert by_id["Q-001"]["gold_response_used_for_scoring"] is False
    assert by_id["Q-001"]["raw_response"].startswith('{"answer": "MOCK OUTPUT:')

    assert by_id["Q-002"]["status"] == "insufficient_evidence"
    assert by_id["Q-002"]["provider_attempted"] is False
    assert by_id["Q-002"]["evidence"] == []
    assert by_id["Q-003"]["evidence_doc_ids"] == ["DOC-5"]
    assert by_id["Q-003"]["gold_hit_count"] == 0
    assert by_id["Q-003"]["first_gold_rank"] is None


def test_summary_distinguishes_answers_failures_and_evidence_metrics() -> None:
    """Rates and retrieval diagnostics come from the raw records."""
    summary = _run()["summary"]

    assert summary["queries"] == 3
    assert summary["answered"] == 3
    assert summary["end_to_end_success_rate"] == 1.0
    assert summary["provider_attempted"] == 2
    assert summary["generation_failures"] == 0
    assert summary["generation_failure_rate"] == 0.0
    assert summary["insufficient_evidence"] == 1
    assert summary["short_circuit_insufficient"] == 1
    assert summary["citations_valid"] == 2
    assert summary["citations_invalid"] == 0
    assert summary["citation_validity_rate"] == 1.0
    assert summary["queries_with_citations"] == 2
    assert summary["answers_without_citations"] == 0
    assert summary["evidence_recall_macro"] == pytest.approx(1 / 3)
    assert summary["evidence_recall_micro"] == pytest.approx(2 / 4)
    assert summary["evidence_mrr"] == pytest.approx(1 / 3)
    assert summary["evidence_hit_at_k"] == 1
    assert summary["queries_with_all_gold_in_evidence"] == 1
    assert summary["status_counts"] == {"insufficient_evidence": 1, "ok": 2}


def test_latency_reports_sample_counts_and_timing_scope() -> None:
    """P50/P95 are computed only from runs that produced a timing."""
    metrics = _run()["metrics"]
    latency = metrics["latency_ms"]

    assert latency["total"]["samples"] == 3
    assert latency["retrieval"]["samples"] == 3
    assert latency["llm"]["samples"] == 2  # Q-002 short-circuits before the provider
    assert latency["total"]["p50"] is not None
    assert "nearest-rank" in latency["timing_scope"]


def test_provider_failures_make_the_run_fail_and_are_counted() -> None:
    """If every provider call fails, the artifact cannot be PASS or PARTIAL."""
    result = _run(_pipeline(_FailingProvider()))

    assert result["status"] == "FAIL"
    assert result["summary"]["answered"] == 1  # the no-evidence refusal remains valid
    assert result["summary"]["generation_failures"] == 2
    assert result["summary"]["llm_timeouts"] == 2
    assert result["metrics"]["generation_failure_rate"]["value"] == 1.0
    assert result["per_query"][0]["status"] == "llm_timeout"
    assert result["per_query"][0]["answer"] == ""
    assert result["per_query"][0]["error"]["category"] == "timeout"


def test_config_rejects_more_final_results_than_candidates() -> None:
    """A larger top_k than the pool is a configuration error."""
    with pytest.raises(BenchmarkError, match="top_k"):
        run_rag_benchmark(
            _pipeline(),
            QUERIES,
            raw_queries=RAW,
            config=RAGBenchmarkConfig(top_k=3, candidates=2),
            environment={},
            dataset={},
        )


def test_markdown_report_warns_that_mock_is_not_model_output() -> None:
    """The rendered report must carry the mock warning and metric exclusions."""
    markdown = render_markdown(_run())

    assert "MOCK RUN" in markdown
    assert "real model-quality" in markdown
    assert "answer correctness" in markdown
    assert "LLM-as-a-Judge" in markdown
    assert "Q-001" in markdown
    assert "无效引用标签" in markdown
    assert "数据安全" in markdown


def test_write_outputs_refuses_to_overwrite_and_can_be_forced(tmp_path: Path) -> None:
    """Existing artifacts are protected unless overwrite is explicit."""
    result = _run()
    output = tmp_path / "rag_benchmark.json"
    report = tmp_path / "rag_benchmark.md"

    json_path, markdown_path = write_outputs(result, output, report)
    assert json_path == output
    assert markdown_path == report
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["stage"] == "8"
    assert payload["per_query"][0]["query_id"] == "Q-001"
    assert "MOCK RUN" in report.read_text(encoding="utf-8")

    with pytest.raises(BenchmarkError, match="already exists"):
        write_outputs(result, output, report)

    write_outputs(result, output, report, overwrite=True)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PARTIAL"


def test_load_raw_queries_reads_gold_response_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    """Reference answers are read for review but remain separate from scoring."""
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query_id": "Q-1",
                    "query": "q",
                    "relevant_doc_ids": ["DOC-1"],
                    "gold_response": "reference",
                }
            ]
        ),
        encoding="utf-8",
    )
    records = load_raw_queries(path)
    assert records["Q-1"]["gold_response"] == "reference"

    path.write_text(
        json.dumps(
            [
                {"query_id": "Q-1", "query": "q", "relevant_doc_ids": []},
                {"query_id": "Q-1", "query": "q2", "relevant_doc_ids": []},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkError, match="duplicate"):
        load_raw_queries(path)


def test_artifact_contains_no_api_key_field() -> None:
    """The artifact records key presence as a boolean only."""
    serialized = json.dumps(_run())
    assert "api_key" in serialized
    assert "get_secret_value" not in serialized
    assert "sk-" not in serialized
