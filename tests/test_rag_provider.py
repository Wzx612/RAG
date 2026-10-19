"""Phase 8: provider retries, failure categories and key hygiene."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

import httpx
import pytest
from src.rag.provider import (
    LLMError,
    MockLLMProvider,
    OpenAICompatibleProvider,
)
from src.schema.rag import LLMConfig

MESSAGES = [
    {"role": "system", "content": "system"},
    {"role": "user", "content": "QUESTION: q\nEVIDENCE\n[E1] chunk"},
]


def _config(**overrides: object) -> LLMConfig:
    """Build a configured provider with a test-only key."""
    values: dict[str, object] = {
        "api_key": "sk-test-secret",  # noqa: S106 -- test fixture, never a real key
        "base_url": "https://api.example.test/v1",
        "model": "test-model",
        "max_retries": 0,
        "temperature": 0.0,
        "max_tokens": 128,
    }
    values.update(overrides)
    return LLMConfig(**values)  # type: ignore[arg-type]


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    """Build a client whose only transport is the supplied handler."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def _success(content: str = '{"answer": "yes", "citations": ["E1"]}') -> httpx.Response:
    """Return a minimal successful OpenAI-compatible response."""
    return httpx.Response(
        200,
        json={
            "id": "req-1",
            "model": "test-model-2026",
            "choices": [
                {"message": {"content": content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        },
        headers={"x-request-id": "header-1"},
    )


def test_success_parses_text_usage_and_sends_the_key_only_in_the_header() -> None:
    """The request is correct and the parsed response keeps provider provenance."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _success()

    provider = OpenAICompatibleProvider(_config(), client=_client(handler))
    response = provider.complete(MESSAGES)

    assert response.text == '{"answer": "yes", "citations": ["E1"]}'
    assert response.model == "test-model-2026"
    assert response.attempts == 1
    assert response.request_id == "header-1"
    assert response.usage is not None
    assert response.usage.total_tokens == 18
    assert len(seen) == 1
    request = seen[0]
    assert request.url == "https://api.example.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test-secret"
    payload = json.loads(request.content)
    assert payload["model"] == "test-model"
    assert payload["stream"] is False
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 128
    assert payload["messages"] == MESSAGES


def test_missing_key_fails_before_any_network_request() -> None:
    """A missing key is a config failure, not a provider call."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _success()

    provider = OpenAICompatibleProvider(_config(api_key=None), client=_client(handler))
    with pytest.raises(LLMError) as caught:
        provider.complete(MESSAGES)

    assert caught.value.category == "config"
    assert caught.value.retryable is False
    assert calls == 0
    assert "APEX_LLM_API_KEY" in str(caught.value)


def test_auth_error_is_not_retried() -> None:
    """401 is terminal even when retries are configured."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="unauthorized")

    provider = OpenAICompatibleProvider(_config(max_retries=2), client=_client(handler))
    with pytest.raises(LLMError) as caught:
        provider.complete(MESSAGES)

    assert caught.value.category == "auth"
    assert caught.value.status_code == 401
    assert caught.value.attempts == 1
    assert calls == 1


def test_server_error_is_retried_up_to_the_bound() -> None:
    """5xx retries exactly ``max_retries`` times, then raises."""
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="unavailable")

    provider = OpenAICompatibleProvider(
        _config(max_retries=2),
        client=_client(handler),
        sleep=delays.append,
    )
    with pytest.raises(LLMError) as caught:
        provider.complete(MESSAGES)

    assert caught.value.category == "server"
    assert caught.value.retryable is True
    assert caught.value.attempts == 3
    assert calls == 3
    assert delays == [1.0, 2.0]


def test_rate_limit_honours_retry_after_and_then_succeeds() -> None:
    """429 waits for Retry-After (bounded) and succeeds on the next attempt."""
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "7"}, text="slow down")
        return _success()

    provider = OpenAICompatibleProvider(
        _config(max_retries=1),
        client=_client(handler),
        sleep=delays.append,
    )
    response = provider.complete(MESSAGES)

    assert response.attempts == 2
    assert delays == [7.0]


def test_timeout_is_retried_and_reported_as_timeout() -> None:
    """Transport timeouts are retryable and classified separately from network errors."""
    calls = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectTimeout("timed out", request=request)

    provider = OpenAICompatibleProvider(
        _config(max_retries=1),
        client=_client(handler),
        sleep=delays.append,
    )
    with pytest.raises(LLMError) as caught:
        provider.complete(MESSAGES)

    assert caught.value.category == "timeout"
    assert caught.value.attempts == 2
    assert calls == 2


def test_network_error_is_classified_as_network() -> None:
    """A connection failure is retryable and distinct from a timeout."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused", request=request)

    provider = OpenAICompatibleProvider(_config(max_retries=0), client=_client(handler))
    with pytest.raises(LLMError) as caught:
        provider.complete(MESSAGES)

    assert caught.value.category == "network"
    assert calls == 1


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("not json", "valid JSON"),
        (json.dumps({"choices": []}), "no choices"),
        (json.dumps({"choices": [{"message": {"content": ""}}]}), "empty"),
        (
            json.dumps(
                {"choices": [{"message": {"content": "answer"}, "finish_reason": "length"}]}
            ),
            "token limit",
        ),
    ],
)
def test_unusable_success_payloads_are_invalid_responses(body: str, reason: str) -> None:
    """Malformed or incomplete 2xx payloads are never treated as answers."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, text=body)

    provider = OpenAICompatibleProvider(_config(max_retries=2), client=_client(handler))
    with pytest.raises(LLMError, match=reason) as caught:
        provider.complete(MESSAGES)

    assert caught.value.category == "invalid_response"
    assert caught.value.attempts == 1
    assert calls == 1


def test_bad_message_shape_is_rejected_before_http() -> None:
    """Invalid messages are a config failure and never leave the process."""
    provider = OpenAICompatibleProvider(
        _config(), client=_client(lambda request: _success())
    )
    with pytest.raises(LLMError) as caught:
        provider.complete([{"role": "tool", "content": "x"}])
    assert caught.value.category == "config"


def test_key_is_not_logged_on_success_or_failure() -> None:
    """Neither success logs nor failure logs may reveal the API key."""
    def success_handler(request: httpx.Request) -> httpx.Response:
        return _success()

    def failure_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(self.format(record))

    project_logger = logging.getLogger("apex")
    handler = _Capture()
    previous_level = project_logger.level
    project_logger.addHandler(handler)
    project_logger.setLevel(logging.DEBUG)
    try:
        OpenAICompatibleProvider(_config(), client=_client(success_handler)).complete(MESSAGES)
        with pytest.raises(LLMError):
            OpenAICompatibleProvider(
                _config(), client=_client(failure_handler)
            ).complete(MESSAGES)
    finally:
        project_logger.removeHandler(handler)
        project_logger.setLevel(previous_level)

    joined = "\n".join(captured)
    assert joined  # the provider does log metadata
    assert "sk-test-secret" not in joined
    assert "Authorization" not in joined
    assert MESSAGES[1]["content"] not in joined


def test_mock_provider_is_explicitly_labelled_and_cites_supplied_evidence() -> None:
    """Mock output must be distinguishable from a real model answer."""
    provider = MockLLMProvider()
    response = provider.complete(MESSAGES)

    assert provider.is_real is False
    assert provider.name == "mock"
    assert response.text.startswith('{"answer": "MOCK OUTPUT:')
    assert "E1" in response.text
    assert response.usage is None


@pytest.mark.parametrize(
    ("mode", "marker"),
    [
        ("invalid_json", "not JSON"),
        ("empty", ""),
        ("insufficient", "insufficient"),
    ],
)
def test_mock_modes_cover_failure_and_refusal_paths(mode: str, marker: str) -> None:
    """The deterministic modes let pipeline tests exercise each failure branch."""
    response = MockLLMProvider(mode=mode).complete(MESSAGES)  # type: ignore[arg-type]
    if marker:
        assert marker in response.text
    else:
        assert response.text == ""


def test_success_response_without_usage_does_not_invent_tokens() -> None:
    """Usage is only recorded when the provider actually sent integers."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "answer"}}]},
        )

    provider = OpenAICompatibleProvider(_config(), client=_client(handler))
    response = provider.complete(MESSAGES)
    assert response.usage is None
