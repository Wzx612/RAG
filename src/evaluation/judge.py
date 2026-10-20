"""LLM-as-a-Judge: Context Relevance and Faithfulness.

The judge is a thin, testable layer over the existing
:class:`~src.rag.provider.LLMProvider` interface. It does not retrieve, rerank or
generate answers; it scores an existing RAG result.

Failure handling is deliberately local. A malformed JSON response, an unknown
context id, an out-of-range score or a provider error produces a result whose
``status`` explains the failure and whose ``score`` is ``None``. The batch
runner can therefore continue with the remaining queries, and aggregation can
exclude the failure instead of treating it as a zero.

A deterministic :class:`MockJudgeProvider` exists for tests and wiring checks.
It is always labelled ``is_real=False`` and its reasons start with ``MOCK
JUDGE``; it must never be presented as a real model evaluation.

Implemented in: LLM-as-a-Judge evaluation.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from src.evaluation.judge_prompts import (
    CONTEXT_RELEVANCE_VERSION,
    FAITHFULNESS_VERSION,
    build_context_relevance_messages,
    build_faithfulness_messages,
    context_relevance_sha256,
    faithfulness_sha256,
)
from src.rag.provider import LLMError, LLMProvider
from src.schema.judge import (
    ClaimJudgement,
    ClaimSupport,
    ContextRelevanceResult,
    FaithfulnessResult,
    JudgeContext,
    JudgeErrorInfo,
    JudgeMetricStatus,
)
from src.schema.rag import LLMResponse

_SUPPORT_VALUES: frozenset[str] = frozenset(
    {"supported", "partially_supported", "unsupported", "contradicted"}
)


@dataclass(frozen=True)
class JudgeConfig:
    """Generation options and the low-score threshold for both rubrics."""

    temperature: float = 0.0
    max_tokens: int = 1024
    low_score_threshold: int = 3

    def __post_init__(self) -> None:
        """Validate the small option surface."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("judge temperature must be between 0 and 2")
        if not 1 <= self.max_tokens <= 32768:
            raise ValueError("judge max_tokens must be between 1 and 32768")
        if not 1 <= self.low_score_threshold <= 5:
            raise ValueError("low_score_threshold must be between 1 and 5")


class JudgeParseError(RuntimeError):
    """A judge response could not be turned into a validated score."""

    def __init__(self, status: JudgeMetricStatus, message: str) -> None:
        """Store the stable failure status and a short explanation."""
        super().__init__(message)
        self.status = status


class LLMJudge:
    """Validate and score one RAG result with an injected provider."""

    def __init__(self, provider: LLMProvider, config: JudgeConfig | None = None) -> None:
        """Bind the provider and generation options."""
        self._provider = provider
        self._config = config or JudgeConfig()

    @property
    def provider(self) -> LLMProvider:
        """The provider used for judge calls."""
        return self._provider

    @property
    def config(self) -> JudgeConfig:
        """The judge generation options."""
        return self._config

    def context_relevance(
        self,
        query: str,
        contexts: Sequence[JudgeContext],
    ) -> ContextRelevanceResult:
        """Score whether the supplied contexts fit the question.

        Empty contexts short-circuit without a provider call and produce
        ``status="empty_context"``. All other failures are returned, not raised.
        """
        if not contexts:
            return ContextRelevanceResult(
                status="empty_context",
                reason="no retrieved context was supplied",
                prompt_version=CONTEXT_RELEVANCE_VERSION,
            )

        messages = build_context_relevance_messages(query, contexts)
        prompt_hash = context_relevance_sha256(messages)
        labels = [context.label for context in contexts]
        response, error, call_ms = self._complete(messages)
        if error is not None:
            return ContextRelevanceResult(
                status="provider_error",
                attempted=True,
                reason=error.message,
                prompt_version=CONTEXT_RELEVANCE_VERSION,
                prompt_sha256=prompt_hash,
                model=self._provider.model,
                latency_ms=call_ms,
                attempts=error.attempts,
                error=error,
            )
        assert response is not None
        try:
            score, reason, relevant, irrelevant = parse_context_relevance(
                response.text, labels
            )
        except JudgeParseError as exc:
            return ContextRelevanceResult(
                status=exc.status,
                attempted=True,
                reason=str(exc),
                prompt_version=CONTEXT_RELEVANCE_VERSION,
                prompt_sha256=prompt_hash,
                model=response.model,
                latency_ms=response.latency_ms,
                attempts=response.attempts,
                usage=response.usage,
                raw_response=response.text,
            )
        return ContextRelevanceResult(
            status="success",
            attempted=True,
            score=score,
            reason=reason,
            relevant_context_ids=relevant,
            irrelevant_context_ids=irrelevant,
            prompt_version=CONTEXT_RELEVANCE_VERSION,
            prompt_sha256=prompt_hash,
            model=response.model,
            latency_ms=response.latency_ms,
            attempts=response.attempts,
            usage=response.usage,
            raw_response=response.text,
        )

    def faithfulness(
        self,
        query: str,
        contexts: Sequence[JudgeContext],
        answer: str,
    ) -> FaithfulnessResult:
        """Score whether the answer's claims are supported by the contexts.

        Empty contexts or a blank answer short-circuit without a provider call.
        All other failures are returned, not raised.
        """
        if not contexts:
            return FaithfulnessResult(
                status="empty_context",
                reason="no retrieved context was supplied",
                prompt_version=FAITHFULNESS_VERSION,
            )
        if not answer.strip():
            return FaithfulnessResult(
                status="empty_answer",
                reason="the RAG answer is empty",
                prompt_version=FAITHFULNESS_VERSION,
            )

        messages = build_faithfulness_messages(query, contexts, answer)
        prompt_hash = faithfulness_sha256(messages)
        labels = [context.label for context in contexts]
        response, error, call_ms = self._complete(messages)
        if error is not None:
            return FaithfulnessResult(
                status="provider_error",
                attempted=True,
                reason=error.message,
                prompt_version=FAITHFULNESS_VERSION,
                prompt_sha256=prompt_hash,
                model=self._provider.model,
                latency_ms=call_ms,
                attempts=error.attempts,
                error=error,
            )
        assert response is not None
        try:
            score, reason, claims, unsupported = parse_faithfulness(response.text, labels)
        except JudgeParseError as exc:
            return FaithfulnessResult(
                status=exc.status,
                attempted=True,
                reason=str(exc),
                prompt_version=FAITHFULNESS_VERSION,
                prompt_sha256=prompt_hash,
                model=response.model,
                latency_ms=response.latency_ms,
                attempts=response.attempts,
                usage=response.usage,
                raw_response=response.text,
            )
        return FaithfulnessResult(
            status="success",
            attempted=True,
            score=score,
            reason=reason,
            claims=claims,
            unsupported_claims=unsupported,
            prompt_version=FAITHFULNESS_VERSION,
            prompt_sha256=prompt_hash,
            model=response.model,
            latency_ms=response.latency_ms,
            attempts=response.attempts,
            usage=response.usage,
            raw_response=response.text,
        )

    def _complete(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> tuple[LLMResponse | None, JudgeErrorInfo | None, float]:
        """Call the provider once and convert every failure into metadata."""
        started = time.perf_counter()
        try:
            response = self._provider.complete(
                messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
        except LLMError as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            return (
                None,
                JudgeErrorInfo(
                    category=exc.category,
                    message=str(exc),
                    retryable=exc.retryable,
                    status_code=exc.status_code,
                    attempts=exc.attempts,
                ),
                elapsed,
            )
        except Exception as exc:  # noqa: BLE001 - one bad query must not abort a batch
            elapsed = (time.perf_counter() - started) * 1000.0
            return (
                None,
                JudgeErrorInfo(
                    category="unexpected",
                    message=f"{type(exc).__name__}: {exc}",
                    retryable=False,
                    attempts=0,
                ),
                elapsed,
            )
        return response, None, response.latency_ms


class MockJudgeProvider:
    """A deterministic, clearly-labelled judge fixture.

    It never calls a network and never needs a key. Reasons start with
    ``MOCK JUDGE``, and the evaluator records ``is_real=False``. The fixture is
    for tests and wiring smoke runs only.
    """

    name = "mock_judge"
    model = "mock-judge-v1"
    is_real = False

    def __init__(
        self,
        *,
        mode: Literal["normal", "invalid_json", "out_of_range", "empty"] = "normal",
    ) -> None:
        """Select a deterministic response shape."""
        self.mode = mode

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return a fixed JSON response inferred from the system prompt."""
        del temperature, max_tokens
        system = next(
            (str(item["content"]) for item in messages if item["role"] == "system"),
            "",
        )
        user = next(
            (str(item["content"]) for item in messages if item["role"] == "user"),
            "",
        )
        labels = _LABEL_RE.findall(user)
        faithfulness = "FAITHFULNESS" in system

        if self.mode == "empty":
            text = ""
        elif self.mode == "invalid_json":
            text = "This is not a judge JSON object."
        elif self.mode == "out_of_range":
            text = (
                '{"score": 9, "reason": "MOCK JUDGE: out-of-range fixture.", '
                '"claims": []}'
                if faithfulness
                else '{"score": 9, "reason": "MOCK JUDGE: out-of-range fixture.", '
                '"relevant_context_ids": [], "irrelevant_context_ids": []}'
            )
        elif faithfulness:
            first = labels[0] if labels else None
            evidence = f'["{first}"]' if first else "[]"
            text = (
                '{"score": 4, "reason": "MOCK JUDGE: deterministic fixture; the '
                'answer is treated as supported by the supplied context.", '
                '"claims": [{"claim": "MOCK JUDGE fixture claim.", '
                '"support_status": "supported", '
                f'"evidence_context_ids": {evidence}}}], '
                '"unsupported_claims": []}'
            )
        else:
            rendered = ", ".join(f'"{label}"' for label in labels)
            score = 4 if labels else 1
            text = (
                f'{{"score": {score}, "reason": "MOCK JUDGE: deterministic fixture; '
                'all supplied contexts are treated as relevant.", '
                f'"relevant_context_ids": [{rendered}], "irrelevant_context_ids": []}}'
            )
        return LLMResponse(
            text=text,
            model=self.model,
            usage=None,
            finish_reason="mock",
            request_id=None,
            latency_ms=0.0,
            attempts=1,
        )


def parse_context_relevance(
    text: str,
    labels: Sequence[str],
) -> tuple[int, str, list[str], list[str]]:
    """Parse and validate one Context Relevance response.

    Raises:
        JudgeParseError: invalid JSON, missing fields, an out-of-range score or
            an unknown/duplicated context id.
    """
    payload = _extract_json_object(text)
    score = _parse_score(payload.get("score"))
    reason = _parse_reason(payload.get("reason"))
    allowed = set(labels)
    relevant = _parse_label_list(
        payload.get("relevant_context_ids", []), allowed, "relevant_context_ids"
    )
    irrelevant = _parse_label_list(
        payload.get("irrelevant_context_ids", []), allowed, "irrelevant_context_ids"
    )
    overlap = set(relevant).intersection(irrelevant)
    if overlap:
        raise JudgeParseError(
            "invalid_schema",
            "contexts cannot be both relevant and irrelevant: " + ", ".join(sorted(overlap)),
        )
    return score, reason, relevant, irrelevant


def parse_faithfulness(
    text: str,
    labels: Sequence[str],
) -> tuple[int, str, list[ClaimJudgement], list[str]]:
    """Parse and validate one Faithfulness response.

    Raises:
        JudgeParseError: invalid JSON, missing fields, an out-of-range score, a
            malformed claim or an unknown evidence id.
    """
    payload = _extract_json_object(text)
    score = _parse_score(payload.get("score"))
    reason = _parse_reason(payload.get("reason"))
    allowed = set(labels)
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise JudgeParseError("invalid_schema", "claims must be a non-empty list")

    claims: list[ClaimJudgement] = []
    for index, raw_claim in enumerate(raw_claims, start=1):
        if not isinstance(raw_claim, dict):
            raise JudgeParseError(
                "invalid_schema", f"claim {index} is not a JSON object"
            )
        claim_text = raw_claim.get("claim")
        if not isinstance(claim_text, str) or not claim_text.strip():
            raise JudgeParseError(
                "invalid_schema", f"claim {index} has no non-empty text"
            )
        support = raw_claim.get("support_status")
        if support not in _SUPPORT_VALUES:
            raise JudgeParseError(
                "invalid_schema",
                f"claim {index} has an invalid support_status {support!r}",
            )
        evidence = _parse_label_list(
            raw_claim.get("evidence_context_ids", []),
            allowed,
            f"claim {index} evidence_context_ids",
        )
        claims.append(
            ClaimJudgement(
                claim=claim_text.strip(),
                support_status=cast(ClaimSupport, support),
                evidence_context_ids=evidence,
            )
        )

    derived = [
        claim.claim
        for claim in claims
        if claim.support_status in ("unsupported", "contradicted")
    ]
    raw_unsupported = payload.get("unsupported_claims")
    if raw_unsupported is None:
        unsupported = derived
    else:
        if not isinstance(raw_unsupported, list):
            raise JudgeParseError(
                "invalid_schema", "unsupported_claims must be a list of strings"
            )
        values: list[str] = []
        for item in raw_unsupported:
            if not isinstance(item, str) or not item.strip():
                raise JudgeParseError(
                    "invalid_schema", "unsupported_claims must contain non-empty strings"
                )
            values.append(item.strip())
        unsupported = list(dict.fromkeys(values))
    return score, reason, claims, unsupported


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object, tolerating a Markdown fence around it."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise JudgeParseError("invalid_json", "response contains no JSON object")
    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeParseError("invalid_json", f"response is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise JudgeParseError("invalid_json", "response JSON is not an object")
    return value


def _parse_score(value: object) -> int:
    """Require an integer in ``[1, 5]``; booleans are not integers here."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise JudgeParseError("invalid_schema", "score must be an integer from 1 to 5")
    if not 1 <= value <= 5:
        raise JudgeParseError("out_of_range", f"score {value} is outside 1..5")
    return value


def _parse_reason(value: object) -> str:
    """Require a non-empty reason string."""
    if not isinstance(value, str) or not value.strip():
        raise JudgeParseError("invalid_schema", "reason must be a non-empty string")
    return value.strip()


def _parse_label_list(
    value: object,
    allowed: set[str],
    field: str,
) -> list[str]:
    """Validate a list of E<n> labels against the supplied context labels."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise JudgeParseError("invalid_schema", f"{field} must be a list")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise JudgeParseError("invalid_schema", f"{field} must contain strings")
        if item not in allowed:
            raise JudgeParseError(
                "unknown_context_id",
                f"{field} contains {item!r}, which was not supplied",
            )
        if item in seen:
            raise JudgeParseError("invalid_schema", f"{field} contains duplicate {item!r}")
        seen.add(item)
        result.append(item)
    return result


_LABEL_RE = re.compile(r"\[(E[1-9][0-9]*)\]")
"""Find the rendered ``[E<n>]`` headers in the mock's user prompt."""
