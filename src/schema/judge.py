"""Data contracts for LLM-as-a-Judge evaluation.

These models separate three things that are easy to conflate:

* what the judge returned (raw text);
* what the evaluator could validate from it (score, claims, context ids);
* why a metric has no score (empty input, invalid JSON, provider failure).

A failed metric always carries ``score=None``. It is never recorded as 0, so a
mean can never silently absorb an unjudged query.

Implemented in: LLM-as-a-Judge evaluation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.schema.rag import LLMUsage

JudgeMetricStatus = Literal[
    "success",
    "empty_context",
    "empty_answer",
    "invalid_json",
    "invalid_schema",
    "out_of_range",
    "unknown_context_id",
    "provider_error",
    "rag_failure",
    "not_run",
]
"""Why one metric does or does not have a score."""

ClaimSupport = Literal["supported", "partially_supported", "unsupported", "contradicted"]
"""Claim-level support labels returned by the faithfulness judge."""


class JudgeContext(BaseModel):
    """One evidence block supplied to the judge.

    ``label`` is the same ``E<n>`` token the answer model saw, so a judge's
    citation of a context can be compared with the RAG evidence list.
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(pattern=r"^E[1-9][0-9]*$")
    title: str | None = None
    content: str = Field(min_length=1)


class JudgeErrorInfo(BaseModel):
    """A failure that contains no prompt, response body or credential."""

    model_config = ConfigDict(extra="forbid")

    category: str
    message: str
    retryable: bool = False
    status_code: int | None = None
    attempts: int = Field(default=0, ge=0)


class ClaimJudgement(BaseModel):
    """One factual claim extracted from the answer and its support status."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1)
    support_status: ClaimSupport
    evidence_context_ids: list[str] = Field(default_factory=list)


class ContextRelevanceResult(BaseModel):
    """Outcome of judging whether the retrieved contexts fit the question."""

    model_config = ConfigDict(extra="forbid")

    metric: Literal["context_relevance"] = "context_relevance"
    status: JudgeMetricStatus
    attempted: bool = False
    score: int | None = Field(default=None, ge=1, le=5)
    reason: str = ""
    relevant_context_ids: list[str] = Field(default_factory=list)
    irrelevant_context_ids: list[str] = Field(default_factory=list)
    prompt_version: str = ""
    prompt_sha256: str = ""
    model: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    attempts: int = Field(default=0, ge=0)
    usage: LLMUsage | None = None
    raw_response: str | None = None
    error: JudgeErrorInfo | None = None

    @model_validator(mode="after")
    def _score_matches_status(self) -> ContextRelevanceResult:
        """Require a score exactly when the metric succeeded."""
        if self.status == "success" and self.score is None:
            raise ValueError("a successful context-relevance result needs a score")
        if self.status != "success" and self.score is not None:
            raise ValueError("a failed context-relevance result cannot carry a score")
        return self


class FaithfulnessResult(BaseModel):
    """Outcome of judging whether answer claims are supported by the contexts."""

    model_config = ConfigDict(extra="forbid")

    metric: Literal["faithfulness"] = "faithfulness"
    status: JudgeMetricStatus
    attempted: bool = False
    score: int | None = Field(default=None, ge=1, le=5)
    reason: str = ""
    claims: list[ClaimJudgement] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    prompt_version: str = ""
    prompt_sha256: str = ""
    model: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    attempts: int = Field(default=0, ge=0)
    usage: LLMUsage | None = None
    raw_response: str | None = None
    error: JudgeErrorInfo | None = None

    @model_validator(mode="after")
    def _score_matches_status(self) -> FaithfulnessResult:
        """Require a score exactly when the metric succeeded."""
        if self.status == "success" and self.score is None:
            raise ValueError("a successful faithfulness result needs a score")
        if self.status != "success" and self.score is not None:
            raise ValueError("a failed faithfulness result cannot carry a score")
        return self


class JudgeQueryRecord(BaseModel):
    """One judged query: RAG provenance plus both metric outcomes."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    query: str
    rag_run_id: str
    rag_status: str
    rag_provider: str | None = None
    rag_model: str | None = None
    rag_is_real: bool = False
    answer_sha256: str = ""
    context_sha256: str = ""
    context_labels: list[str] = Field(default_factory=list)
    context_relevance: ContextRelevanceResult
    faithfulness: FaithfulnessResult


class JudgeMetadata(BaseModel):
    """Provenance for one judge run."""

    model_config = ConfigDict(extra="forbid")

    judge_provider: str
    judge_model: str
    judge_is_real: bool
    same_model: bool
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    judge_temperature: float
    judge_max_tokens: int
    generated_at: str


class JudgeArtifact(BaseModel):
    """The top-level summary artifact. Per-query rows live in JSONL."""

    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    experiment_name: str = "llm_as_a_judge"
    status: str
    status_reason: str
    generated_at: str
    finished_at: str
    run_id: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    provider: dict[str, Any] = Field(default_factory=dict)
    rag: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    per_query: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    integrity: dict[str, Any] = Field(default_factory=dict)
