"""Phase 8: end-to-end RAG pipeline behaviour with fake retrieval/providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from src.rag.pipeline import RAGPipeline
from src.rag.provider import LLMError
from src.retrieval import RetrievalError
from src.retrieval.reranker import RerankError
from src.schema.rag import LLMResponse, LLMUsage, RAGConfig
from src.schema.retrieval import SearchHit


def _hit(
    chunk_id: str,
    *,
    doc_id: str | None = None,
    content: str | None = None,
    rank: int = 1,
) -> SearchHit:
    """Build a retrieval hit that looks like a live database row."""
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=doc_id or f"DOC-{chunk_id}",
        score=0.9,
        distance=0.1,
        rank=rank,
        source="reranked",
        content=f"Policy text for {chunk_id}." if content is None else content,
        chunk_strategy="document",
        chunk_index=0,
        title=f"Title {chunk_id}",
        category="fees",
        card_tier="standard",
        effective_year=2025,
        policy_code="P-1",
        rerank_score=0.8,
    )


@dataclass
class _Details:
    """A minimal RerankedResult-shaped object for the retriever protocol."""

    hits: list[SearchHit]
    base_ms: float = 1.0
    dense_ms: float = 0.5
    sparse_ms: float = 0.4
    fusion_ms: float = 0.1
    rerank_ms: float = 5.0
    total_ms: float = 7.0
    candidate_k: int = 10
    top_k: int = 2
    rrf_k: int = 60
    base_name: str = "hybrid"
    model_id: str = "fake-reranker"
    model_revision: str = "rev-1"
    device: str = "cpu"
    batch_size: int = 16
    max_length: int = 512
    activation: str = "identity"
    empty_candidates: bool = False
    truncated: bool = False
    fallback_used: bool = False


class _Retriever:
    """A retriever returning fixed hits and recording calls."""

    name = "fake-reranked"

    def __init__(
        self,
        hits: list[SearchHit],
        *,
        error: BaseException | None = None,
        details: bool = True,
    ) -> None:
        """Store hits, an optional failure and whether details are exposed."""
        self.hits = hits
        self.error = error
        self.details = details
        self.calls: list[tuple[str, int | None]] = []
        if not details:
            self.retrieve_with_details = None  # type: ignore[method-assign]

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Record the call and return or raise."""
        self.calls.append((query, top_k))
        if self.error is not None:
            raise self.error
        return self.hits

    def retrieve_with_details(
        self, query: str, top_k: int | None = None
    ) -> _Details:
        """Return details when enabled; otherwise behave like ``retrieve``."""
        self.calls.append((query, top_k))
        if self.error is not None:
            raise self.error
        return _Details(hits=self.hits, top_k=top_k or 2)


class _Provider:
    """A scripted provider with optional configuration and completion failures."""

    name = "fake-provider"
    model = "fake-model-v1"
    is_real = True

    def __init__(
        self,
        *,
        text: str = '{"answer": "The fee is $95 [E1].", "citations": ["E1"], '
        '"insufficient_evidence": false}',
        completion_error: BaseException | None = None,
        config_error: BaseException | None = None,
        usage: LLMUsage | None = None,
    ) -> None:
        """Bind the scripted response and failures."""
        self.text = text
        self.completion_error = completion_error
        self.config_error = config_error
        self.usage = usage
        self.calls: list[dict[str, Any]] = []

    def ensure_configured(self) -> None:
        """Raise the scripted configuration failure, if any."""
        if self.config_error is not None:
            raise self.config_error

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Record the prompt and return or raise."""
        self.calls.append(
            {
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self.completion_error is not None:
            raise self.completion_error
        return LLMResponse(
            text=self.text,
            model="fake-model-2026",
            usage=self.usage,
            finish_reason="stop",
            request_id="req-1",
            latency_ms=1.0,
            attempts=1,
        )


def _pipeline(
    *,
    hits: list[SearchHit] | None = None,
    provider: _Provider | None = None,
    retriever: _Retriever | None = None,
    config: RAGConfig | None = None,
) -> tuple[RAGPipeline, _Retriever, _Provider]:
    """Wire a fake pipeline and return its parts for assertions."""
    resolved_retriever = retriever or _Retriever(hits if hits is not None else [_hit("c1")])
    resolved_provider = provider or _Provider()
    return (
        RAGPipeline(
            resolved_retriever,
            resolved_provider,
            config if config is not None else RAGConfig(top_k=2),
        ),
        resolved_retriever,
        resolved_provider,
    )


def test_normal_flow_returns_answer_evidence_and_validated_citations() -> None:
    """A valid response becomes an answer with only real evidence citations."""
    provider = _Provider(
        text='{"answer": "The fee is $95 [E1]; see also E2. Not E99.", '
        '"citations": ["E1", "[E2]", "E99", "E1"], "insufficient_evidence": false}'
    )
    pipeline, retriever, _ = _pipeline(
        hits=[_hit("c1"), _hit("c2", rank=2)], provider=provider
    )
    result = pipeline.answer("What is the annual fee?", query_id="Q-001")

    assert result.status == "ok"
    assert result.query_id == "Q-001"
    assert result.answer.startswith("The fee is $95")
    assert [citation.label for citation in result.citations] == ["E1", "E2"]
    assert [citation.chunk_id for citation in result.citations] == ["c1", "c2"]
    assert result.invalid_citations == ["E99"]
    assert result.citation_status == "invalid"
    assert [item.label for item in result.evidence] == ["E1", "E2"]
    assert [item.chunk_id for item in result.evidence] == ["c1", "c2"]
    assert result.provider == "fake-provider"
    assert result.model == "fake-model-2026"
    assert result.provider_attempted is True
    assert result.error is None
    assert retriever.calls == [("What is the annual fee?", 2)]
    assert provider.calls[0]["temperature"] == 0.0
    assert provider.calls[0]["max_tokens"] == 512
    assert "What is the annual fee?" in provider.calls[0]["messages"][1]["content"]
    assert "[E1] chunk_id=c1" in provider.calls[0]["messages"][1]["content"]
    assert result.prompt_version == "rag-answer-v1"
    assert result.prompt_sha256 and len(result.prompt_sha256) == 64
    assert result.context_sha256 and len(result.context_sha256) == 64
    assert result.retrieval["requested_top_k"] == 2
    assert result.retrieval["doc_ids"] == ["DOC-c1", "DOC-c2"]
    assert result.retrieval["model_id"] == "fake-reranker"
    assert result.timings_ms["retrieval_ms"] >= 0.0


def test_empty_retrieval_short_circuits_without_calling_the_provider() -> None:
    """No hits is a deterministic refusal, not a provider request."""
    pipeline, retriever, provider = _pipeline(hits=[])
    result = pipeline.answer("Unknown question?")

    assert result.status == "insufficient_evidence"
    assert result.insufficient_evidence is True
    assert result.answer.startswith("The available material")
    assert result.evidence == []
    assert result.citations == []
    assert result.citation_status == "no_citations"
    assert result.provider_attempted is False
    assert result.raw_response is None
    assert provider.calls == []
    assert retriever.calls == [("Unknown question?", 2)]


def test_model_declared_insufficient_evidence_is_preserved() -> None:
    """A model refusal is distinct from a retrieval short circuit."""
    provider = _Provider(
        text='{"answer": "Cannot be confirmed from the evidence.", '
        '"citations": [], "insufficient_evidence": true}'
    )
    pipeline, _, _ = _pipeline(provider=provider)
    result = pipeline.answer("Is travel insurance included?")

    assert result.status == "insufficient_evidence"
    assert result.insufficient_evidence is True
    assert result.answer == "Cannot be confirmed from the evidence."
    assert result.citations == []
    assert result.citation_status == "no_citations"
    assert result.provider_attempted is True
    assert len(result.evidence) == 1


def test_citation_labels_not_in_evidence_are_never_trusted() -> None:
    """Fabricated labels are recorded but cannot become citations."""
    provider = _Provider(
        text='{"answer": "Fabricated [E42].", "citations": ["E42", "E1"], '
        '"insufficient_evidence": false}'
    )
    pipeline, _, _ = _pipeline(provider=provider)
    result = pipeline.answer("q")

    assert [citation.label for citation in result.citations] == ["E1"]
    assert result.invalid_citations == ["E42"]
    assert result.citation_status == "invalid"


def test_context_truncation_and_dropped_evidence_are_reported() -> None:
    """The pipeline exposes the exact bounded context it sent."""
    pipeline, _, provider = _pipeline(
        hits=[
            _hit("c1", content="a" * 500),
            _hit("c2", content="b" * 500, rank=2),
        ],
        config=RAGConfig(max_context_chars=300, max_chunk_chars=500, top_k=2),
    )
    result = pipeline.answer("q")

    assert result.status == "ok"
    assert len(result.evidence) == 1
    assert result.evidence[0].truncated is True
    assert result.dropped_evidence == 1
    assert result.context_truncated is True
    assert len(result.context_text) <= 300
    sent = provider.calls[0]["messages"][1]["content"]
    assert result.context_text in sent
    assert "[E2]" not in sent


def test_missing_api_key_fails_before_retrieval() -> None:
    """A config error must not load models or touch the retriever."""
    provider = _Provider(
        config_error=LLMError(
            "LLM API key is not configured", category="config", retryable=False
        )
    )
    pipeline, retriever, _ = _pipeline(provider=provider)
    result = pipeline.answer("q")

    assert result.status == "config_error"
    assert result.error is not None
    assert result.error.stage == "llm_config"
    assert result.error.category == "config"
    assert result.provider_attempted is False
    assert retriever.calls == []
    assert provider.calls == []


@pytest.mark.parametrize(
    ("category", "status"),
    [
        ("timeout", "llm_timeout"),
        ("rate_limit", "llm_rate_limited"),
        ("network", "llm_network_error"),
        ("server", "llm_server_error"),
        ("auth", "llm_auth_error"),
        ("request", "llm_request_error"),
        ("invalid_response", "llm_invalid_response"),
    ],
)
def test_provider_failure_categories_map_to_distinct_statuses(
    category: str,
    status: str,
) -> None:
    """Transport and auth failures must not be reported as answers."""
    provider = _Provider(
        completion_error=LLMError(
            "failure",
            category=category,  # type: ignore[arg-type]
            retryable=True,
            status_code=500,
            attempts=3,
        )
    )
    pipeline, _, _ = _pipeline(provider=provider)
    result = pipeline.answer("q")

    assert result.status == status
    assert result.answer == ""
    assert result.citations == []
    assert result.provider_attempted is True
    assert result.error is not None
    assert result.error.category == category
    assert result.error.attempts == 3
    assert len(result.evidence) == 1


def test_unexpected_provider_exception_is_a_structured_failure() -> None:
    """An unclassified exception cannot leak out as an answer."""
    provider = _Provider(completion_error=ZeroDivisionError("boom"))
    pipeline, _, _ = _pipeline(provider=provider)
    result = pipeline.answer("q")

    assert result.status == "llm_error"
    assert result.answer == ""
    assert result.error is not None
    assert result.error.category == "unexpected"
    assert "ZeroDivisionError" in result.error.message


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("not json", "JSON object"),
        ("", "empty"),
        ('{"answer": "", "citations": []}', "missing or empty"),
        ('{"answer": "x", "citations": "E1"}', "not a list"),
        ('{"answer": "x", "citations": [], "insufficient_evidence": "yes"}', "boolean"),
    ],
)
def test_invalid_or_empty_model_output_is_not_an_answer(text: str, reason: str) -> None:
    """Unparseable output keeps the raw text only for diagnosis."""
    provider = _Provider(text=text)
    pipeline, _, _ = _pipeline(provider=provider)
    result = pipeline.answer("q")

    assert result.status == "llm_invalid_response"
    assert result.answer == ""
    assert result.raw_response == text
    assert result.error is not None
    assert reason in result.error.message


def test_retrieval_failure_is_structured_and_stops_the_chain() -> None:
    """A dense/sparse failure is a retrieval error, not an empty answer."""
    retriever = _Retriever([], error=RetrievalError("database down"))
    pipeline, _, provider = _pipeline(retriever=retriever)
    result = pipeline.answer("q")

    assert result.status == "retrieval_error"
    assert result.answer == ""
    assert result.error is not None
    assert result.error.stage == "retrieval"
    assert provider.calls == []


@pytest.mark.parametrize(
    ("stage", "status"),
    [("inference", "rerank_error"), ("base_retrieval", "retrieval_error")],
)
def test_reranker_failures_are_not_silently_fallen_back(stage: str, status: str) -> None:
    """A reranker failure stays visible with its stage."""
    retriever = _Retriever(
        [], error=RerankError("model failed", stage=stage)  # type: ignore[arg-type]
    )
    pipeline, _, provider = _pipeline(retriever=retriever)
    result = pipeline.answer("q")

    assert result.status == status
    assert result.error is not None
    assert result.error.stage == stage
    assert provider.calls == []


def test_blank_database_content_is_an_evidence_error() -> None:
    """Missing evidence text must not be replaced with a placeholder."""
    pipeline, _, provider = _pipeline(hits=[_hit("c1", content="  ")])
    result = pipeline.answer("q")

    assert result.status == "evidence_error"
    assert result.answer == ""
    assert result.error is not None
    assert result.error.stage == "context"
    assert provider.calls == []


@pytest.mark.parametrize("query", ["", "   ", None, 42])
def test_invalid_queries_are_rejected_before_retrieval(query: object) -> None:
    """Bad input is a request error, never a retrieval or provider call."""
    pipeline, retriever, provider = _pipeline()
    result = pipeline.answer(query)

    assert result.status == "invalid_request"
    assert result.error is not None
    assert result.error.stage == "input"
    assert result.answer == ""
    assert retriever.calls == []
    assert provider.calls == []


def test_prompt_hash_is_stable_for_the_same_question_and_evidence() -> None:
    """A repeated run over identical inputs yields the same prompt hash."""
    pipeline, _, provider = _pipeline()
    first = pipeline.answer("same question")
    second = pipeline.answer("same question")

    assert first.prompt_sha256 == second.prompt_sha256
    assert first.context_sha256 == second.context_sha256
    assert "untrusted data" in provider.calls[0]["messages"][0]["content"]
    assert "<<<EVIDENCE_START>>>" in provider.calls[0]["messages"][1]["content"]


def test_usage_is_copied_only_when_the_provider_returns_it() -> None:
    """Token counts are provenance, not something the pipeline estimates."""
    pipeline, _, _ = _pipeline(
        provider=_Provider(usage=LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    )
    result = pipeline.answer("q")

    assert result.usage is not None
    assert result.usage.total_tokens == 15


def test_retriever_without_details_still_works() -> None:
    """The minimum Retriever protocol is enough for the RAG pipeline."""
    retriever = _Retriever([_hit("c1")], details=False)
    pipeline, _, provider = _pipeline(retriever=retriever)
    result = pipeline.answer("q")

    assert result.status == "ok"
    assert result.retrieval["retriever"] == "fake-reranked"
    assert result.retrieval["timings_ms"] == {}
    assert provider.calls
