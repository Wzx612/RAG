"""End-to-end RAG answer pipeline.

Flow::

    question
      -> input validation
      -> accepted retrieval/rerank chain (injected; no retrieval logic here)
      -> real database evidence context (bounded, labelled E1..En)
      -> evidence-grounded prompt
      -> existing/injected LLM provider
      -> JSON answer parsing
      -> citation validation against the evidence actually supplied
      -> RAGAnswer (answer, citations, evidence, retrieval trace, error)

Every failure is a structured status. Retrieval, reranking, context assembly,
provider transport and response parsing are kept separate so a failure cannot
be mistaken for an answer. No result is written back to PostgreSQL, the corpus
or the gold labels.

Implemented in: Phase 8 (RAG end-to-end).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from typing import Any

from src.rag.context import ContextBundle, ContextError, build_context
from src.rag.prompt import PROMPT_VERSION, build_messages, prompt_sha256
from src.rag.provider import LLMError, LLMProvider
from src.retrieval import RetrievalError
from src.retrieval.reranker import RerankError
from src.schema.rag import (
    Citation,
    CitationStatus,
    Evidence,
    LLMUsage,
    RAGAnswer,
    RAGConfig,
    RAGErrorInfo,
    RAGStatus,
)
from src.schema.retrieval import SearchHit
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CITATION_RE = re.compile(r"^E[1-9][0-9]*$")
_INSUFFICIENT_MESSAGE = (
    "The available material does not contain enough evidence to answer this question."
)

_LLM_STATUS: dict[str, RAGStatus] = {
    "config": "config_error",
    "auth": "llm_auth_error",
    "timeout": "llm_timeout",
    "rate_limit": "llm_rate_limited",
    "network": "llm_network_error",
    "server": "llm_server_error",
    "request": "llm_request_error",
    "invalid_response": "llm_invalid_response",
}


class _AnswerParseError(RuntimeError):
    """The provider returned text that is not the required answer object."""


class RAGPipeline:
    """Compose an existing retriever with context assembly and a provider."""

    def __init__(
        self,
        retriever: Any,
        provider: LLMProvider,
        config: RAGConfig | None = None,
    ) -> None:
        """Bind the retrieval chain, provider and answer options."""
        self._retriever = retriever
        self._provider = provider
        self._config = config if config is not None else RAGConfig()
        if not hasattr(retriever, "retrieve"):
            raise ValueError("retriever must expose retrieve(query, top_k=...)")
        if self._config.prompt_version != PROMPT_VERSION:
            raise ValueError(
                f"prompt_version {self._config.prompt_version!r} does not match the "
                f"implemented prompt {PROMPT_VERSION!r}"
            )

    @property
    def config(self) -> RAGConfig:
        """The effective RAG options."""
        return self._config

    @property
    def retriever(self) -> Any:
        """The injected retrieval chain; used by diagnostics."""
        return self._retriever

    @property
    def provider(self) -> LLMProvider:
        """The injected generation provider."""
        return self._provider

    def answer(self, query: object, *, query_id: str | None = None) -> RAGAnswer:
        """Answer one question and return a structured, traceable result.

        Args:
            query: Question text. Non-string or blank input returns an
                ``invalid_request`` result; it is not sent to retrieval.
            query_id: Optional evaluation id, copied into the result.

        Returns:
            A :class:`~src.schema.rag.RAGAnswer`. ``status`` distinguishes a
            usable answer from insufficient evidence and every failure stage.
        """
        started = time.perf_counter()
        resolved_query, validation_error = _validate_query(query)
        if validation_error is not None:
            return self._failure(
                query=resolved_query,
                query_id=query_id,
                status="invalid_request",
                stage="input",
                category="invalid_request",
                message=validation_error,
                timings={"total_ms": _ms_since(started)},
            )

        ensure = getattr(self._provider, "ensure_configured", None)
        if callable(ensure):
            try:
                ensure()
            except LLMError as exc:
                return self._failure(
                    query=resolved_query,
                    query_id=query_id,
                    status=_LLM_STATUS.get(exc.category, "llm_error"),
                    stage="llm_config",
                    category=exc.category,
                    message=str(exc),
                    retryable=exc.retryable,
                    status_code=exc.status_code,
                    attempts=exc.attempts,
                    timings={"total_ms": _ms_since(started)},
                    provider=self._provider.name,
                    model=getattr(self._provider, "model", None),
                )

        retrieval_started = time.perf_counter()
        details: Any | None = None
        try:
            details_method = getattr(self._retriever, "retrieve_with_details", None)
            if callable(details_method):
                details = details_method(resolved_query, top_k=self._config.top_k)
                hits = list(getattr(details, "hits", None) or getattr(details, "fused_hits", []))
            else:
                hits = list(self._retriever.retrieve(resolved_query, top_k=self._config.top_k))
        except RerankError as exc:
            timings = {"retrieval_ms": _ms_since(retrieval_started)}
            timings["total_ms"] = _ms_since(started)
            status: RAGStatus = (
                "retrieval_error" if exc.stage == "base_retrieval" else "rerank_error"
            )
            logger.error(
                "RAG retrieval failed: stage=%s type=%s query_chars=%d",
                exc.stage,
                type(exc).__name__,
                len(resolved_query),
            )
            return self._failure(
                query=resolved_query,
                query_id=query_id,
                status=status,
                stage=exc.stage,
                category="rerank" if exc.stage != "base_retrieval" else "retrieval",
                message=str(exc),
                timings=timings,
            )
        except (RetrievalError, ValueError) as exc:
            timings = {"retrieval_ms": _ms_since(retrieval_started)}
            timings["total_ms"] = _ms_since(started)
            logger.error(
                "RAG retrieval failed: type=%s query_chars=%d",
                type(exc).__name__,
                len(resolved_query),
            )
            return self._failure(
                query=resolved_query,
                query_id=query_id,
                status="retrieval_error",
                stage="retrieval",
                category="retrieval",
                message=str(exc),
                timings=timings,
            )
        except Exception as exc:  # noqa: BLE001 -- returned as a structured failure
            timings = {"retrieval_ms": _ms_since(retrieval_started)}
            timings["total_ms"] = _ms_since(started)
            logger.error(
                "RAG retrieval raised an unexpected error: type=%s query_chars=%d",
                type(exc).__name__,
                len(resolved_query),
            )
            return self._failure(
                query=resolved_query,
                query_id=query_id,
                status="retrieval_error",
                stage="retrieval",
                category="unexpected",
                message=f"{type(exc).__name__}: {exc}",
                timings=timings,
            )
        retrieval_ms = _ms_since(retrieval_started)
        trace = _retrieval_trace(self._retriever, details, hits, self._config.top_k)

        context_started = time.perf_counter()
        try:
            bundle = build_context(
                hits,
                max_context_chars=self._config.max_context_chars,
                max_chunk_chars=self._config.max_chunk_chars,
            )
        except ContextError as exc:
            timings = {
                "retrieval_ms": retrieval_ms,
                "context_ms": _ms_since(context_started),
                "total_ms": _ms_since(started),
            }
            return self._failure(
                query=resolved_query,
                query_id=query_id,
                status="evidence_error",
                stage="context",
                category="evidence",
                message=str(exc),
                retrieval=trace,
                timings=timings,
            )
        context_ms = _ms_since(context_started)

        if not bundle.evidence:
            timings = {
                "retrieval_ms": retrieval_ms,
                "context_ms": context_ms,
                "total_ms": _ms_since(started),
            }
            return self._insufficient(
                query=resolved_query,
                query_id=query_id,
                bundle=bundle,
                retrieval=trace,
                timings=timings,
                provider_attempted=False,
            )

        prompt_started = time.perf_counter()
        try:
            messages = build_messages(resolved_query, bundle.text)
            digest = prompt_sha256(messages, version=self._config.prompt_version)
        except ValueError as exc:
            timings = {
                "retrieval_ms": retrieval_ms,
                "context_ms": context_ms,
                "prompt_ms": _ms_since(prompt_started),
                "total_ms": _ms_since(started),
            }
            return self._failure(
                query=resolved_query,
                query_id=query_id,
                status="evidence_error",
                stage="prompt",
                category="prompt",
                message=str(exc),
                evidence=bundle.evidence,
                context_text=bundle.text,
                context_sha256=bundle.sha256,
                context_truncated=bundle.truncated,
                dropped_evidence=bundle.dropped,
                retrieval=trace,
                timings=timings,
            )
        prompt_ms = _ms_since(prompt_started)

        llm_started = time.perf_counter()
        try:
            response = self._provider.complete(
                messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
        except LLMError as exc:
            timings = {
                "retrieval_ms": retrieval_ms,
                "context_ms": context_ms,
                "prompt_ms": prompt_ms,
                "llm_ms": _ms_since(llm_started),
                "total_ms": _ms_since(started),
            }
            logger.error(
                "RAG generation failed: category=%s status=%s attempts=%d",
                exc.category,
                exc.status_code,
                exc.attempts,
            )
            return self._failure(
                query=resolved_query,
                query_id=query_id,
                status=_LLM_STATUS.get(exc.category, "llm_error"),
                stage="llm",
                category=exc.category,
                message=str(exc),
                retryable=exc.retryable,
                status_code=exc.status_code,
                attempts=exc.attempts,
                evidence=bundle.evidence,
                context_text=bundle.text,
                context_sha256=bundle.sha256,
                context_truncated=bundle.truncated,
                dropped_evidence=bundle.dropped,
                prompt_version=self._config.prompt_version,
                prompt_sha256=digest,
                retrieval=trace,
                timings=timings,
                provider_attempted=True,
                provider=self._provider.name,
                model=getattr(self._provider, "model", None),
            )
        except Exception as exc:  # noqa: BLE001 -- returned as a structured failure
            timings = {
                "retrieval_ms": retrieval_ms,
                "context_ms": context_ms,
                "prompt_ms": prompt_ms,
                "llm_ms": _ms_since(llm_started),
                "total_ms": _ms_since(started),
            }
            logger.error("RAG provider raised an unexpected error: type=%s", type(exc).__name__)
            return self._failure(
                query=resolved_query,
                query_id=query_id,
                status="llm_error",
                stage="llm",
                category="unexpected",
                message=f"{type(exc).__name__}: {exc}",
                evidence=bundle.evidence,
                context_text=bundle.text,
                context_sha256=bundle.sha256,
                context_truncated=bundle.truncated,
                dropped_evidence=bundle.dropped,
                prompt_version=self._config.prompt_version,
                prompt_sha256=digest,
                retrieval=trace,
                timings=timings,
                provider_attempted=True,
                provider=self._provider.name,
                model=getattr(self._provider, "model", None),
            )
        llm_ms = _ms_since(llm_started)

        try:
            answer_text, raw_citations, insufficient = _parse_answer(response.text)
        except _AnswerParseError as exc:
            timings = {
                "retrieval_ms": retrieval_ms,
                "context_ms": context_ms,
                "prompt_ms": prompt_ms,
                "llm_ms": llm_ms,
                "total_ms": _ms_since(started),
            }
            return self._failure(
                query=resolved_query,
                query_id=query_id,
                status="llm_invalid_response",
                stage="parse",
                category="invalid_response",
                message=str(exc),
                evidence=bundle.evidence,
                context_text=bundle.text,
                context_sha256=bundle.sha256,
                context_truncated=bundle.truncated,
                dropped_evidence=bundle.dropped,
                prompt_version=self._config.prompt_version,
                prompt_sha256=digest,
                retrieval=trace,
                timings=timings,
                provider_attempted=True,
                provider=self._provider.name,
                model=response.model,
                usage=response.usage,
                raw_response=response.text,
            )

        citations, invalid = _resolve_citations(raw_citations, bundle.evidence)
        citation_status: CitationStatus = (
            "invalid" if invalid else "no_citations" if not citations else "ok"
        )
        timings = {
            "retrieval_ms": retrieval_ms,
            "context_ms": context_ms,
            "prompt_ms": prompt_ms,
            "llm_ms": llm_ms,
            "total_ms": _ms_since(started),
        }
        logger.info(
            "RAG answer: status=%s evidence=%d citations=%d invalid_citations=%d "
            "provider=%s total_ms=%.2f",
            "insufficient_evidence" if insufficient else "ok",
            len(bundle.evidence),
            len(citations),
            len(invalid),
            self._provider.name,
            timings["total_ms"],
        )
        return RAGAnswer(
            query_id=query_id,
            query=resolved_query,
            status="insufficient_evidence" if insufficient else "ok",
            answer=answer_text,
            citations=citations,
            invalid_citations=invalid,
            citation_status=citation_status,
            insufficient_evidence=insufficient,
            evidence=bundle.evidence,
            context_text=bundle.text,
            context_sha256=bundle.sha256,
            context_truncated=bundle.truncated,
            dropped_evidence=bundle.dropped,
            prompt_version=self._config.prompt_version,
            prompt_sha256=digest,
            provider=self._provider.name,
            model=response.model,
            usage=response.usage,
            timings_ms=timings,
            retrieval=trace,
            provider_attempted=True,
            raw_response=response.text,
            error=None,
        )

    # ------------------------------------------------------------------ helpers

    def _insufficient(
        self,
        *,
        query: str,
        query_id: str | None,
        bundle: ContextBundle,
        retrieval: dict[str, Any],
        timings: dict[str, float],
        provider_attempted: bool,
    ) -> RAGAnswer:
        """Return the deterministic no-evidence result; no provider is called."""
        return RAGAnswer(
            query_id=query_id,
            query=query,
            status="insufficient_evidence",
            answer=_INSUFFICIENT_MESSAGE,
            citations=[],
            invalid_citations=[],
            citation_status="no_citations",
            insufficient_evidence=True,
            evidence=bundle.evidence,
            context_text=bundle.text,
            context_sha256=bundle.sha256,
            context_truncated=bundle.truncated,
            dropped_evidence=bundle.dropped,
            prompt_version=self._config.prompt_version,
            prompt_sha256=None,
            provider=self._provider.name,
            model=getattr(self._provider, "model", None),
            usage=None,
            timings_ms=timings,
            retrieval=retrieval,
            provider_attempted=provider_attempted,
            raw_response=None,
            error=None,
        )

    def _failure(
        self,
        *,
        query: str,
        query_id: str | None,
        status: RAGStatus,
        stage: str,
        category: str,
        message: str,
        retryable: bool = False,
        status_code: int | None = None,
        attempts: int = 0,
        evidence: Sequence[Evidence] | None = None,
        context_text: str = "",
        context_sha256: str = "",
        context_truncated: bool = False,
        dropped_evidence: int = 0,
        prompt_version: str | None = None,
        prompt_sha256: str | None = None,
        retrieval: dict[str, Any] | None = None,
        timings: dict[str, float] | None = None,
        provider_attempted: bool = False,
        provider: str | None = None,
        model: str | None = None,
        usage: LLMUsage | None = None,
        raw_response: str | None = None,
    ) -> RAGAnswer:
        """Build a failed result with an empty answer and explicit error."""
        return RAGAnswer(
            query_id=query_id,
            query=query,
            status=status,
            answer="",
            citations=[],
            invalid_citations=[],
            citation_status="no_citations",
            insufficient_evidence=False,
            evidence=list(evidence or []),
            context_text=context_text,
            context_sha256=context_sha256,
            context_truncated=context_truncated,
            dropped_evidence=dropped_evidence,
            prompt_version=prompt_version or self._config.prompt_version,
            prompt_sha256=prompt_sha256,
            provider=provider,
            model=model,
            usage=usage,
            timings_ms=timings or {},
            retrieval=retrieval or {},
            provider_attempted=provider_attempted,
            raw_response=raw_response,
            error=RAGErrorInfo(
                stage=stage,
                category=category,
                message=message,
                retryable=retryable,
                status_code=status_code,
                attempts=attempts,
            ),
        )


def _validate_query(query: object) -> tuple[str, str | None]:
    """Return safe display text and an optional validation error."""
    if not isinstance(query, str):
        return repr(query), f"query must be a str, got {type(query).__name__}"
    stripped = query.strip()
    if not stripped:
        return query, "query must be a non-empty string"
    return stripped, None


def _parse_answer(text: object) -> tuple[str, list[object], bool]:
    """Parse the required JSON answer object."""
    if not isinstance(text, str) or not text.strip():
        raise _AnswerParseError("LLM response content is empty")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise _AnswerParseError("LLM response does not contain a JSON object")
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise _AnswerParseError(f"LLM response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise _AnswerParseError("LLM response JSON is not an object")

    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise _AnswerParseError("LLM answer field is missing or empty")
    citations = payload.get("citations", [])
    if not isinstance(citations, list):
        raise _AnswerParseError("LLM citations field is not a list")
    insufficient = payload.get("insufficient_evidence", False)
    if not isinstance(insufficient, bool):
        raise _AnswerParseError("LLM insufficient_evidence field is not a boolean")
    return answer.strip(), citations, insufficient


def _resolve_citations(
    raw_citations: Sequence[object],
    evidence: Sequence[Evidence],
) -> tuple[list[Citation], list[str]]:
    """Map model-provided labels to evidence actually supplied to the model."""
    by_label = {item.label: item for item in evidence}
    valid: list[Citation] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for raw in raw_citations:
        label = _normalize_label(raw)
        if label is None or label not in by_label:
            rendered = raw if isinstance(raw, str) else repr(raw)
            rendered = rendered.strip()[:100]
            if rendered and rendered not in invalid:
                invalid.append(rendered)
            continue
        if label in seen:
            continue
        seen.add(label)
        item = by_label[label]
        valid.append(
            Citation(
                label=item.label,
                chunk_id=item.chunk_id,
                doc_id=item.doc_id,
                title=item.title,
            )
        )
    return valid, invalid


def _normalize_label(raw: object) -> str | None:
    """Accept ``E1`` or ``[E1]`` and nothing else."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if not _CITATION_RE.fullmatch(text):
        return None
    return text


def _retrieval_trace(
    retriever: Any,
    details: Any | None,
    hits: Sequence[SearchHit],
    requested_top_k: int,
) -> dict[str, Any]:
    """Record IDs, stage timings and model provenance without duplicating text."""
    trace: dict[str, Any] = {
        "retriever": str(getattr(retriever, "name", "unknown")),
        "requested_top_k": requested_top_k,
        "hit_count": len(hits),
        "doc_ids": list(dict.fromkeys(hit.doc_id for hit in hits)),
        "candidates": [
            {
                "chunk_id": hit.chunk_id,
                "doc_id": hit.doc_id,
                "rank": hit.rank,
                "source": hit.source,
                "score": hit.score,
                "rrf_score": hit.rrf_score,
                "rerank_score": hit.rerank_score,
            }
            for hit in hits
        ],
        "timings_ms": {},
    }
    if details is None:
        return trace

    for field in (
        "base_ms",
        "dense_ms",
        "sparse_ms",
        "fusion_ms",
        "rerank_ms",
        "total_ms",
    ):
        value = getattr(details, field, None)
        if isinstance(value, (int, float)):
            trace["timings_ms"][field] = float(value)
    for field in (
        "candidate_k",
        "top_k",
        "rrf_k",
        "base_name",
        "model_id",
        "model_revision",
        "device",
        "batch_size",
        "max_length",
        "activation",
        "empty_candidates",
        "truncated",
        "fallback_used",
    ):
        value = getattr(details, field, None)
        if value is not None:
            trace[field] = value
    return trace


def _ms_since(started: float) -> float:
    """Return elapsed wall-clock milliseconds since a ``perf_counter`` value."""
    return max(0.0, (time.perf_counter() - started) * 1000.0)
