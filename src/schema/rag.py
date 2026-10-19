"""RAG answer, evidence, citation and provider data shapes.

Pure data. These models are the contract between context assembly, the prompt
builder, the LLM provider, the pipeline and the evaluation report. They make the
difference between a model answer, the evidence it was given, the citations it
returned and a pipeline failure explicit -- rather than collapsing everything
into one string.

Implemented in: Phase 8 (RAG end-to-end).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

RAGStatus = Literal[
    "ok",
    "insufficient_evidence",
    "invalid_request",
    "retrieval_error",
    "rerank_error",
    "evidence_error",
    "config_error",
    "llm_timeout",
    "llm_rate_limited",
    "llm_network_error",
    "llm_server_error",
    "llm_auth_error",
    "llm_invalid_response",
    "llm_request_error",
    "llm_error",
]
"""Stable outcome labels. Only ``ok`` and ``insufficient_evidence`` are answers."""

CitationStatus = Literal["ok", "no_citations", "invalid"]
"""Whether returned citation labels mapped to evidence actually sent to the model."""


class LLMConfig(BaseModel):
    """Generation-provider settings.

    The API key is a :class:`~pydantic.SecretStr` so accidental ``repr`` output
    masks it. Callers must use ``api_key.get_secret_value()`` at the one point
    where the Authorization header is built.
    """

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr | None = None
    base_url: str = "https://api.deepseek.com"
    model: str = Field(default="deepseek-chat", min_length=1)
    connect_timeout: float = Field(default=5.0, gt=0.0, le=120.0)
    read_timeout: float = Field(default=60.0, gt=0.0, le=600.0)
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Retries after the first attempt. Total attempts = max_retries + 1.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=32768)

    @model_validator(mode="after")
    def _validate_base_url(self) -> LLMConfig:
        """Reject a base URL that is not plain HTTP(S)."""
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return self

    @property
    def api_key_present(self) -> bool:
        """Whether a non-blank key is configured. Never returns the key itself."""
        if self.api_key is None:
            return False
        return bool(self.api_key.get_secret_value().strip())


class LLMUsage(BaseModel):
    """Token counts returned by the provider; never estimated by this project."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class LLMResponse(BaseModel):
    """One successful generation response."""

    model_config = ConfigDict(extra="forbid")

    text: str
    model: str
    usage: LLMUsage | None = None
    finish_reason: str | None = None
    request_id: str | None = None
    latency_ms: float = Field(ge=0.0)
    attempts: int = Field(ge=1)


class Evidence(BaseModel):
    """One real database chunk that was placed in the model context.

    ``content`` is the database value. ``context_block`` is the exact formatted
    block that appeared in the prompt, after per-chunk truncation. ``label`` is
    the only citation token the model is allowed to return.
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(pattern=r"^E[1-9][0-9]*$")
    chunk_id: str
    doc_id: str
    chunk_strategy: str | None = None
    chunk_index: int | None = None
    retrieval_rank: int = Field(ge=1)
    score: float
    rerank_score: float | None = None
    title: str | None = None
    category: str | None = None
    card_tier: str | None = None
    effective_year: int | None = None
    policy_code: str | None = None
    content: str
    context_block: str
    original_chars: int = Field(ge=0)
    included_chars: int = Field(ge=0)
    truncated: bool


class Citation(BaseModel):
    """A citation the model returned that mapped to provided evidence."""

    model_config = ConfigDict(extra="forbid")

    label: str
    chunk_id: str
    doc_id: str
    title: str | None = None


class RAGConfig(BaseModel):
    """Context-budget and generation options for one RAG answer."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    top_k: int = Field(default=10, ge=1, le=1000)
    max_context_chars: int = Field(
        default=8000,
        ge=200,
        description="Hard cap on the assembled evidence text, headers included.",
    )
    max_chunk_chars: int = Field(
        default=2000,
        ge=1,
        description="Hard cap on one chunk's body before it enters the context.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    prompt_version: str = Field(default="rag-answer-v1", min_length=1)


class RAGErrorInfo(BaseModel):
    """A structured failure, with no prompt text and no credentials."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    category: str
    message: str
    retryable: bool = False
    status_code: int | None = None
    attempts: int = Field(default=0, ge=0)


class RAGAnswer(BaseModel):
    """The complete result of one question.

    The answer text, its validated citations, the exact evidence context, the
    retrieval trace and any error are separate fields. A failed request never
    carries a plausible-looking fabricated answer: ``answer`` is empty and
    ``error`` explains the failure.
    """

    model_config = ConfigDict(extra="forbid")

    query_id: str | None = None
    query: str
    status: RAGStatus
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    citation_status: CitationStatus = "no_citations"
    insufficient_evidence: bool = False
    evidence: list[Evidence] = Field(default_factory=list)
    context_text: str = ""
    context_sha256: str = ""
    context_truncated: bool = False
    dropped_evidence: int = Field(default=0, ge=0)
    prompt_version: str
    prompt_sha256: str | None = None
    provider: str | None = None
    model: str | None = None
    usage: LLMUsage | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    provider_attempted: bool = False
    raw_response: str | None = None
    error: RAGErrorInfo | None = None
