"""OpenAI-compatible chat provider and a deterministic mock provider.

The project had no generation provider, so this module adds the smallest one
that fits the existing design: a direct ``httpx`` POST to an OpenAI-compatible
``/chat/completions`` endpoint. It deliberately does not add the ``openai`` SDK
or a provider-specific abstraction.

Safety and failure behaviour
----------------------------

* The API key is read from :class:`~src.schema.rag.LLMConfig` (ultimately
  ``APEX_LLM_API_KEY`` or ``DEEPSEEK_API_KEY``). It is used only in the
  Authorization header and is never logged, reported or embedded in an error.
* Connect and read timeouts are separate, and retries are bounded. A missing
  key, authentication failure and invalid response are not retried; timeouts,
  network failures, rate limits and server errors are retried with a cap.
* Error categories are stable and distinct: ``config``, ``auth``, ``timeout``,
  ``network``, ``rate_limit``, ``server``, ``request`` and ``invalid_response``.
* Requests and answers are never written to logs. Successful calls log only the
  model, attempt count, latency and token counts when the provider returns them.

Implemented in: Phase 8 (RAG end-to-end).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Protocol

import httpx

from src.schema.rag import LLMConfig, LLMResponse, LLMUsage
from src.utils.logger import get_logger

logger = get_logger(__name__)

LLMCategory = Literal[
    "config",
    "auth",
    "timeout",
    "network",
    "rate_limit",
    "server",
    "request",
    "invalid_response",
]
"""Stable provider failure categories used by the pipeline and its tests."""

_EVIDENCE_LABEL_RE = re.compile(r"\[(E[1-9][0-9]*)\]")


class LLMError(RuntimeError):
    """A generation request could not be completed.

    ``category`` is stable and machine-readable. ``retryable`` states whether
    another attempt could succeed, and ``attempts`` is how many requests were
    actually made. The message never includes the API key, headers, prompt or
    response body.
    """

    def __init__(
        self,
        message: str,
        *,
        category: LLMCategory,
        retryable: bool = False,
        status_code: int | None = None,
        attempts: int = 0,
        retry_after: float | None = None,
    ) -> None:
        """Store the message and its failure metadata."""
        super().__init__(message)
        self.category: LLMCategory = category
        self.retryable = retryable
        self.status_code = status_code
        self.attempts = attempts
        self.retry_after = retry_after


class LLMProvider(Protocol):
    """The generation interface the RAG pipeline depends on.

    Provenance fields are read-only properties so an implementation may expose
    them as properties (as the real provider does) or as class attributes (as
    the mock does) without either being able to rewrite report metadata.
    """

    @property
    def name(self) -> str:
        """Provider name recorded in results."""
        ...

    @property
    def model(self) -> str:
        """Configured model id recorded in results."""
        ...

    @property
    def is_real(self) -> bool:
        """Whether this provider makes real network model calls."""
        ...

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return one complete chat response or raise :class:`LLMError`."""
        ...


class OpenAICompatibleProvider:
    """A bounded-retry client for an OpenAI-compatible chat endpoint.

    Tests inject an ``httpx.Client`` backed by ``httpx.MockTransport``; no test
    touches the network or needs a real key. An injected client owns its own
    lifetime and is not closed by this class.
    """

    name = "openai_compatible"
    is_real = True

    def __init__(
        self,
        config: LLMConfig,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Bind configuration and optional test seams."""
        self._config = config
        self._client = client
        self._sleep = sleep

    @property
    def config(self) -> LLMConfig:
        """The provider configuration. ``repr`` masks the secret key."""
        return self._config

    @property
    def model(self) -> str:
        """Configured model id."""
        return self._config.model

    @property
    def endpoint(self) -> str:
        """The chat-completions URL derived from ``base_url``."""
        return f"{self._config.base_url.rstrip('/')}/chat/completions"

    def ensure_configured(self) -> None:
        """Raise a config-category error when the API key is absent.

        Calling this before retrieval lets a benchmark fail fast without loading
        models. :meth:`complete` calls it too, so a caller cannot skip it.
        """
        if not self._config.api_key_present:
            raise LLMError(
                "LLM API key is not configured; set APEX_LLM_API_KEY "
                "(or DEEPSEEK_API_KEY) in the environment or .env",
                category="config",
                retryable=False,
            )

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send one chat request with bounded retries.

        Args:
            messages: Chat messages with ``role`` and ``content``.
            temperature: Override the configured temperature for this call.
            max_tokens: Override the configured completion cap.

        Returns:
            The parsed response, including the provider model and usage if sent.

        Raises:
            LLMError: invalid input, missing key, HTTP failure, transport
                failure or an unreadable/empty response. The category and
                ``attempts`` are recorded on the exception.
        """
        normalized = _validate_messages(messages)
        self.ensure_configured()

        key = self._config.api_key
        assert key is not None  # ensure_configured() rejected the absent case
        headers = {
            "Authorization": f"Bearer {key.get_secret_value().strip()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload: dict[str, object] = {
            "model": self._config.model,
            "messages": normalized,
            "temperature": (
                self._config.temperature if temperature is None else float(temperature)
            ),
            "max_tokens": self._config.max_tokens if max_tokens is None else int(max_tokens),
            "stream": False,
        }

        total_attempts = self._config.max_retries + 1
        last_error: LLMError | None = None
        for attempt in range(1, total_attempts + 1):
            started = time.perf_counter()
            try:
                response = self._post(payload, headers)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                result = self._parse_response(response, elapsed_ms, attempt)
            except LLMError as exc:
                exc.attempts = attempt
                last_error = exc
                if not exc.retryable or attempt >= total_attempts:
                    raise
                delay = _retry_delay(exc, attempt)
                logger.warning(
                    "LLM request failed: category=%s status=%s attempt=%d/%d retrying_in=%.2fs",
                    exc.category,
                    exc.status_code,
                    attempt,
                    total_attempts,
                    delay,
                )
                self._sleep(delay)
                continue

            logger.info(
                "LLM response: provider=%s model=%s attempts=%d latency_ms=%.2f "
                "prompt_tokens=%s completion_tokens=%s",
                self.name,
                result.model,
                result.attempts,
                result.latency_ms,
                result.usage.prompt_tokens if result.usage else None,
                result.usage.completion_tokens if result.usage else None,
            )
            return result

        assert last_error is not None  # the loop always returns or raises
        raise last_error

    def _post(
        self,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> httpx.Response:
        """POST once, mapping transport failures to stable categories."""
        timeout = httpx.Timeout(
            connect=self._config.connect_timeout,
            read=self._config.read_timeout,
            write=self._config.read_timeout,
            pool=self._config.connect_timeout,
        )
        try:
            if self._client is not None:
                response = self._client.post(
                    self.endpoint,
                    json=dict(payload),
                    headers=dict(headers),
                    timeout=timeout,
                )
            else:
                with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                    response = client.post(
                        self.endpoint,
                        json=dict(payload),
                        headers=dict(headers),
                    )
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"LLM request timed out ({type(exc).__name__})",
                category="timeout",
                retryable=True,
            ) from exc
        except httpx.TransportError as exc:
            raise LLMError(
                f"LLM network failure ({type(exc).__name__})",
                category="network",
                retryable=True,
            ) from exc
        except OSError as exc:
            raise LLMError(
                f"LLM transport failure ({type(exc).__name__})",
                category="network",
                retryable=True,
            ) from exc

        status = response.status_code
        if status in (401, 403):
            raise LLMError(
                f"LLM authentication failed (HTTP {status}); check APEX_LLM_API_KEY",
                category="auth",
                retryable=False,
                status_code=status,
            )
        if status == 429:
            raise LLMError(
                "LLM rate limit exceeded (HTTP 429)",
                category="rate_limit",
                retryable=True,
                status_code=status,
                retry_after=_retry_after_seconds(response),
            )
        if status >= 500:
            raise LLMError(
                f"LLM server error (HTTP {status})",
                category="server",
                retryable=True,
                status_code=status,
            )
        if status >= 400:
            raise LLMError(
                f"LLM request rejected (HTTP {status})",
                category="request",
                retryable=False,
                status_code=status,
            )
        if status >= 300:
            raise LLMError(
                f"LLM endpoint returned an unexpected redirect (HTTP {status})",
                category="request",
                retryable=False,
                status_code=status,
            )
        return response

    def _parse_response(
        self,
        response: httpx.Response,
        elapsed_ms: float,
        attempt: int,
    ) -> LLMResponse:
        """Parse a 2xx response without logging or returning its body on error."""
        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMError(
                "LLM response was not valid JSON",
                category="invalid_response",
                retryable=False,
                status_code=response.status_code,
            ) from exc
        if not isinstance(payload, dict):
            raise LLMError(
                "LLM response JSON is not an object",
                category="invalid_response",
                retryable=False,
                status_code=response.status_code,
            )

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError(
                "LLM response has no choices",
                category="invalid_response",
                retryable=False,
                status_code=response.status_code,
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMError(
                "LLM response choice is not an object",
                category="invalid_response",
                retryable=False,
                status_code=response.status_code,
            )
        message = choice.get("message")
        if not isinstance(message, dict):
            raise LLMError(
                "LLM response has no message object",
                category="invalid_response",
                retryable=False,
                status_code=response.status_code,
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMError(
                "LLM response message content is empty",
                category="invalid_response",
                retryable=False,
                status_code=response.status_code,
            )
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            raise LLMError(
                "LLM response hit the completion token limit; answer is incomplete",
                category="invalid_response",
                retryable=False,
                status_code=response.status_code,
            )

        usage = _parse_usage(payload.get("usage"))
        response_model = payload.get("model")
        if not isinstance(response_model, str) or not response_model:
            response_model = self._config.model
        request_id = response.headers.get("x-request-id")
        if not request_id and isinstance(payload.get("id"), str):
            request_id = payload["id"]

        return LLMResponse(
            text=content,
            model=response_model,
            usage=usage,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            request_id=request_id,
            latency_ms=max(0.0, elapsed_ms),
            attempts=attempt,
        )


class MockLLMProvider:
    """A deterministic, clearly-labelled fake provider for tests and smoke runs.

    It never calls a network, never needs a key and never pretends to be a
    model: its answer text starts with ``MOCK``. The benchmark records
    ``is_real=False`` and must not present its output as real LLM output.
    """

    name = "mock"
    model = "mock-evidence-v1"
    is_real = False

    def __init__(
        self,
        *,
        mode: Literal["normal", "insufficient", "invalid_json", "empty"] = "normal",
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
        """Return a fixed response derived from the first evidence label."""
        del temperature, max_tokens
        normalized = _validate_messages(messages)
        user_text = next(
            (item["content"] for item in normalized if item["role"] == "user"),
            "",
        )
        labels = _EVIDENCE_LABEL_RE.findall(user_text)
        label = labels[0] if labels else None

        if self.mode == "empty":
            text = ""
        elif self.mode == "invalid_json":
            text = "This is not JSON."
        elif self.mode == "insufficient" or label is None:
            text = (
                '{"answer": "The provided evidence is insufficient to answer this question.", '
                '"citations": [], "insufficient_evidence": true}'
            )
        else:
            text = (
                '{"answer": "MOCK OUTPUT: evidence label ' + label + " was supplied to the "
                'provider. This deterministic fixture is not a model answer.", '
                '"citations": ["' + label + '"], "insufficient_evidence": false}'
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


def _validate_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Validate and materialise the chat messages."""
    items = list(messages)
    if not items:
        raise LLMError("at least one chat message is required", category="config")
    normalized: list[dict[str, str]] = []
    for position, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            raise LLMError(f"message {position} is not a mapping", category="config")
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            raise LLMError(f"message {position} has an invalid role", category="config")
        if not isinstance(content, str) or not content:
            raise LLMError(f"message {position} has empty content", category="config")
        normalized.append({"role": role, "content": content})
    return normalized


def _parse_usage(value: object) -> LLMUsage | None:
    """Read token counts only when the provider actually returned integers."""
    if not isinstance(value, dict):
        return None
    fields: dict[str, int | None] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            fields[key] = item
        else:
            fields[key] = None
    if all(item is None for item in fields.values()):
        return None
    return LLMUsage(**fields)


def _retry_delay(error: LLMError, attempt: int) -> float:
    """Return a bounded exponential delay; rate limits honour ``Retry-After``."""
    if error.retry_after is not None:
        return min(max(error.retry_after, 0.0), 30.0)
    return min(2.0 ** (attempt - 1), 8.0)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse a numeric ``Retry-After`` header; HTTP dates are ignored."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None
