"""Batch judging, aggregation and artifact-writing tests.

The fixtures mimic the RAG artifact's real shape: one successful answer, one
legitimate insufficient-evidence refusal and one pipeline failure. The tests
assert the two properties that matter most for an honest report: a failed score
is never averaged as 0, and a three-query smoke cannot be PASS when the dataset
has 30 queries.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from src.evaluation.artifacts import ArtifactExistsError
from src.evaluation.judge import JudgeConfig, LLMJudge, MockJudgeProvider
from src.evaluation.judge_evaluation import (
    JudgeEvaluationError,
    load_rag_artifact,
    run_judge_evaluation,
    write_outputs,
)
from src.schema.rag import LLMResponse

GENERATED = "2026-10-20T00:00:00+00:00"


class _FixedProvider:
    """A real-looking judge with a fixed score and a captured prompt list."""

    name = "fake_judge"
    model = "fake-judge-v1"
    is_real = True

    def __init__(self, score: int = 4) -> None:
        """Set the fixed score and start an empty capture buffer."""
        self.score = score
        self.calls: list[list[dict[str, str]]] = []

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return valid JSON for either rubric and capture the messages."""
        del temperature, max_tokens
        normalized = [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in messages
        ]
        self.calls.append(normalized)
        system = next(
            (item["content"] for item in normalized if item["role"] == "system"), ""
        )
        user = next(
            (item["content"] for item in normalized if item["role"] == "user"), ""
        )
        labels = [
            token
            for token in ("E1", "E2", "E3")
            if f"[{token}]" in user
        ]
        if "FAITHFULNESS" in system:
            payload: dict[str, Any] = {
                "score": self.score,
                "reason": "fixed test judgement",
                "claims": [
                    {
                        "claim": "the answer restates the evidence",
                        "support_status": "supported",
                        "evidence_context_ids": labels[:1],
                    }
                ],
                "unsupported_claims": [],
            }
        else:
            payload = {
                "score": self.score,
                "reason": "fixed test judgement",
                "relevant_context_ids": labels,
                "irrelevant_context_ids": [],
            }
        return LLMResponse(
            text=json.dumps(payload),
            model=self.model,
            usage=None,
            finish_reason="stop",
            request_id=None,
            latency_ms=1.0,
            attempts=1,
        )


def _evidence(label: str = "E1", content: str = "The annual fee is $95.") -> dict[str, Any]:
    """One RAG evidence block in the artifact shape."""
    return {
        "label": label,
        "title": "Fees",
        "content": content,
        "chunk_id": f"chunk-{label}",
        "doc_id": "DOC-1",
    }


def _record(
    query_id: str,
    *,
    status: str = "ok",
    answer: str = "The annual fee is $95 [E1].",
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One RAG per-query record."""
    return {
        "query_id": query_id,
        "query": f"Question {query_id}?",
        "status": status,
        "answer": answer,
        "evidence": [_evidence()] if evidence is None and status == "ok" else evidence or [],
        "context_sha256": "a" * 64,
        "provider": "mock",
        "model": "mock-evidence-v1",
        "gold_response": "GOLD RESPONSE SECRET",
    }


def _rag(
    records: list[dict[str, Any]] | None = None,
    *,
    is_real: bool = False,
    model: str = "mock-evidence-v1",
) -> dict[str, Any]:
    """A minimal RAG artifact with the keys the judge consumes."""
    return {
        "run_id": "rag-run-1",
        "status": "PASS" if is_real else "PARTIAL",
        "status_reason": "test fixture",
        "generated_at": GENERATED,
        "parameters": {"provider_kind": "real" if is_real else "mock"},
        "provider": {
            "kind": "real" if is_real else "mock",
            "name": "openai_compatible" if is_real else "mock",
            "model": model,
            "is_real": is_real,
        },
        "per_query": records or [],
    }


def _run(
    rag: Mapping[str, Any],
    provider: object | None = None,
    *,
    expected_queries: int = 3,
    max_concurrency: int = 1,
    low_score_threshold: int = 3,
) -> dict[str, Any]:
    """Run the batch with a mock or fixed provider and no file writes."""
    judge = LLMJudge(
        (provider or MockJudgeProvider()),  # type: ignore[arg-type]
        JudgeConfig(
            temperature=0.0,
            max_tokens=512,
            low_score_threshold=low_score_threshold,
        ),
    )
    return run_judge_evaluation(
        rag,
        judge,
        expected_queries=expected_queries,
        max_concurrency=max_concurrency,
        low_score_threshold=low_score_threshold,
        environment={"execution_environment": "test"},
        dataset={"queries": len(rag["per_query"])},
        api_key_present=False,
    )


def test_mixed_batch_isolates_failures_and_never_scores_them_as_zero() -> None:
    """One success, one empty-context refusal and one RAG failure stay distinct."""
    rag = _rag(
        [
            _record("Q-001"),
            _record("Q-002", status="insufficient_evidence", answer="", evidence=[]),
            _record("Q-003", status="retrieval_error", answer="", evidence=[]),
        ]
    )
    result = _run(rag)

    assert result["status"] == "PARTIAL"
    assert "mock" in result["status_reason"].lower()
    assert result["summary"]["queries"] == 3
    assert result["summary"]["both_metrics_success"] == 1
    assert result["summary"]["queries_with_judge_failure"] == 2
    assert result["summary"]["rag_failures"] == 1
    assert result["summary"]["rag_status_counts"] == {
        "insufficient_evidence": 1,
        "ok": 1,
        "retrieval_error": 1,
    }

    by_id = {record["query_id"]: record for record in result["per_query"]}
    assert by_id["Q-001"]["context_relevance"]["status"] == "success"
    assert by_id["Q-001"]["faithfulness"]["status"] == "success"
    assert by_id["Q-002"]["context_relevance"]["status"] == "empty_context"
    assert by_id["Q-002"]["faithfulness"]["status"] == "empty_context"
    assert by_id["Q-003"]["context_relevance"]["status"] == "rag_failure"
    assert by_id["Q-003"]["faithfulness"]["status"] == "rag_failure"
    assert by_id["Q-003"]["context_relevance"]["attempted"] is False

    for key in ("context_relevance", "faithfulness"):
        metric = result["summary"]["metrics"][key]
        assert metric["attempted"] == 1
        assert metric["success"] == 1
        assert metric["failed"] == 0
        assert metric["valid_scores"] == 1
        assert metric["mean"] == 4.0
        assert metric["median"] == 4.0
        assert metric["distribution"] == {"1": 0, "2": 0, "3": 0, "4": 1, "5": 0}
        assert metric["failure_counts"]["status"] == {
            "empty_context": 1,
            "rag_failure": 1,
        }


def test_real_judge_on_the_full_expected_set_can_pass() -> None:
    """PASS requires real RAG, a real judge, all queries and no metric failures."""
    rag = _rag([_record("Q-001"), _record("Q-002"), _record("Q-003")], is_real=True)
    result = _run(rag, _FixedProvider(), expected_queries=3)

    assert result["status"] == "PASS"
    assert result["provider"]["is_real"] is True
    assert result["provider"]["real_judge_calls"] is True
    assert result["summary"]["metrics"]["faithfulness"]["mean"] == 4.0


def test_real_three_query_smoke_is_partial_against_a_thirty_query_dataset() -> None:
    """A smoke run may not be reported as a full quality evaluation."""
    rag = _rag([_record("Q-001"), _record("Q-002"), _record("Q-003")], is_real=True)
    result = _run(rag, _FixedProvider(), expected_queries=30)

    assert result["status"] == "PARTIAL"
    assert "3/30" in result["status_reason"]
    assert result["summary"]["expected_queries"] == 30
    assert result["summary"]["not_run_queries"] == 27
    assert [record["query_id"] for record in result["per_query"]] == [
        "Q-001",
        "Q-002",
        "Q-003",
    ]


def test_all_rag_failures_make_the_run_fail() -> None:
    """No judgeable answer means no judge result, rather than a mean of zeros."""
    rag = _rag(
        [
            _record("Q-001", status="llm_timeout", answer="", evidence=[]),
            _record("Q-002", status="retrieval_error", answer="", evidence=[]),
            _record("Q-003", status="config_error", answer="", evidence=[]),
        ],
        is_real=True,
    )
    result = _run(rag, _FixedProvider())

    assert result["status"] == "FAIL"
    assert result["summary"]["rag_failures"] == 3
    assert result["summary"]["metrics"]["context_relevance"]["valid_scores"] == 0
    assert result["summary"]["metrics"]["context_relevance"]["mean"] is None
    assert result["metrics"]["llm_as_a_judge"]["real_judge_calls"] is False


def test_low_scores_are_listed_but_stay_in_the_valid_mean() -> None:
    """A low score is a real score; it is flagged, not discarded."""
    rag = _rag([_record("Q-001")], is_real=True)
    result = _run(rag, _FixedProvider(score=2), expected_queries=1)

    metric = result["summary"]["metrics"]["context_relevance"]
    assert metric["valid_scores"] == 1
    assert metric["mean"] == 2.0
    assert metric["low_score_count"] == 1
    assert metric["low_score_samples"][0]["query_id"] == "Q-001"
    assert metric["low_score_samples"][0]["score"] == 2


def test_faithfulness_prompts_never_contain_gold_response() -> None:
    """The judge sees the answer and the supplied evidence, never the label."""
    provider = _FixedProvider()
    rag = _rag([_record("Q-001")])
    _run(rag, provider)

    assert provider.calls
    rendered = "\n".join(
        item["content"] for call in provider.calls for item in call
    )
    assert "GOLD RESPONSE SECRET" not in rendered
    assert "The annual fee is $95." in rendered
    assert "The annual fee is $95 [E1]." in rendered


def test_concurrency_preserves_query_order() -> None:
    """A bounded pool must not reorder the artifact."""
    rag = _rag([_record("Q-001"), _record("Q-002"), _record("Q-003")])
    result = _run(rag, max_concurrency=2)

    assert [record["query_id"] for record in result["per_query"]] == [
        "Q-001",
        "Q-002",
        "Q-003",
    ]


def test_limit_judges_a_prefix_and_validates_its_bounds() -> None:
    """A smoke flag cuts the prefix and is rejected when impossible."""
    rag = _rag([_record("Q-001"), _record("Q-002"), _record("Q-003")])
    judge = LLMJudge(MockJudgeProvider())
    result = run_judge_evaluation(
        rag,
        judge,
        expected_queries=30,
        limit=2,
        low_score_threshold=3,
    )
    assert [record["query_id"] for record in result["per_query"]] == ["Q-001", "Q-002"]
    assert result["status"] == "PARTIAL"

    with pytest.raises(JudgeEvaluationError, match="limit"):
        run_judge_evaluation(rag, judge, expected_queries=30, limit=4, low_score_threshold=3)


def test_write_outputs_creates_jsonl_summary_and_report(tmp_path: Path) -> None:
    """JSONL is one parseable object per line; the summary reloads as JSON."""
    rag = _rag([_record("Q-001"), _record("Q-002"), _record("Q-003")])
    result = _run(rag)
    jsonl = tmp_path / "judge_results.jsonl"
    summary = tmp_path / "judge_summary.json"
    report = tmp_path / "judge_report.md"

    jsonl_path, summary_path, report_path = write_outputs(
        result,
        jsonl_path=jsonl,
        summary_path=summary,
        report_path=report,
    )

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["query_id"] for line in lines] == [
        "Q-001",
        "Q-002",
        "Q-003",
    ]
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["status"] == "PARTIAL"
    assert loaded["provider"]["is_real"] is False
    assert "LLM-as-a-Judge" in report_path.read_text(encoding="utf-8")
    assert "MOCK JUDGE RUN" in report_path.read_text(encoding="utf-8")


def test_write_outputs_refuses_to_clobber_unless_overwrite(tmp_path: Path) -> None:
    """A second run cannot silently replace an existing judgement."""
    rag = _rag([_record("Q-001")])
    result = _run(rag)
    paths = {
        "jsonl_path": tmp_path / "results.jsonl",
        "summary_path": tmp_path / "summary.json",
        "report_path": tmp_path / "report.md",
    }
    write_outputs(result, **paths)

    with pytest.raises(ArtifactExistsError):
        write_outputs(result, **paths)
    write_outputs(result, overwrite=True, **paths)


def test_load_rag_artifact_requires_the_judge_inputs(tmp_path: Path) -> None:
    """A file without per_query/provider/status/generated_at is rejected."""
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_rag([_record("Q-001")])), encoding="utf-8")
    assert load_rag_artifact(valid)["per_query"][0]["query_id"] == "Q-001"

    for missing in ("per_query", "provider", "status", "generated_at"):
        payload = _rag([_record("Q-001")])
        del payload[missing]
        path = tmp_path / f"missing-{missing}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(JudgeEvaluationError, match=missing):
            load_rag_artifact(path)

    with pytest.raises(JudgeEvaluationError, match="cannot read"):
        load_rag_artifact(tmp_path / "absent.json")


def test_metrics_block_marks_failures_as_excluded() -> None:
    """The artifact states the aggregation rule, not just the numbers."""
    result = _run(_rag([_record("Q-001")]))
    block = result["metrics"]["llm_as_a_judge"]

    assert block["executed"] is True
    assert "only status=success" in block["aggregation"]
    assert block["judge_model"] == "mock-judge-v1"
    assert result["gold_usage"]["gold_response_used_for_scoring"] is False
