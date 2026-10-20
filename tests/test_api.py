"""API transport contract with an injected pipeline (no DB, no model)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import psycopg
import pytest
from src.retrieval import RetrievalError
from src.schema.rag import RAGAnswer
from src.schema.retrieval import SearchHit

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from src.api.app import create_app  # noqa: E402


class _FakePipeline:
    """Return one canned RAGAnswer and record the call."""

    def __init__(self, result: RAGAnswer) -> None:
        """Store the canned answer."""
        self._result = result
        self.calls: list[tuple[str, str | None]] = []

    def answer(self, query: object, *, query_id: str | None = None) -> RAGAnswer:
        """Record the query and return the canned answer."""
        self.calls.append((str(query), query_id))
        return self._result.model_copy(
            update={"query": str(query), "query_id": query_id}
        )


def _answer(status: str = "ok", **kwargs: Any) -> RAGAnswer:
    """Build a valid RAGAnswer for the requested status."""
    return RAGAnswer(
        query="question",
        status=status,
        answer="Answer text.",
        prompt_version="test",
        **kwargs,
    )


def _client(result: RAGAnswer) -> tuple[TestClient, _FakePipeline]:
    """Build an app with one injected pipeline."""
    pipeline = _FakePipeline(result)
    app = create_app(pipeline_factory=lambda: pipeline)
    return TestClient(app), pipeline


def test_healthz_does_not_touch_dependencies() -> None:
    """Liveness only needs the process."""
    client, _ = _client(_answer())
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "apex-rag-api"}


def test_answer_returns_the_projection_not_the_raw_payload() -> None:
    """The response carries answer/evidence but never context or raw output."""
    client, pipeline = _client(_answer(raw_response="raw", context_text="secret context"))
    response = client.post("/answer", json={"query": "fee?", "query_id": "q1"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["answer"] == "Answer text."
    assert body["query_id"] == "q1"
    assert "raw_response" not in body
    assert "context_text" not in body
    assert pipeline.calls == [("fee?", "q1")]


def test_request_validation_rejects_empty_and_unknown_fields() -> None:
    """Blank questions and client-side Top-K are refused by the schema."""
    client, pipeline = _client(_answer())
    assert client.post("/answer", json={"query": ""}).status_code == 422
    assert client.post("/answer", json={"query": "x", "top_k": 5}).status_code == 422
    assert pipeline.calls == []


def test_insufficient_evidence_is_a_valid_answer() -> None:
    """A refusal to answer is a 200 outcome, distinct from a failure."""
    client, _ = _client(
        _answer("insufficient_evidence", insufficient_evidence=True)
    )
    response = client.post("/answer", json={"query": "unknown"})
    assert response.status_code == 200
    assert response.json()["insufficient_evidence"] is True


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("invalid_request", 422),
        ("llm_timeout", 504),
        ("llm_rate_limited", 429),
        ("llm_auth_error", 503),
        ("llm_invalid_response", 502),
        ("retrieval_error", 503),
        ("rerank_error", 503),
    ],
)
def test_pipeline_failures_map_to_http_statuses(status: str, expected: int) -> None:
    """Each stable pipeline label has one HTTP meaning."""
    client, _ = _client(_answer(status))
    response = client.post("/answer", json={"query": "question"})
    assert response.status_code == expected
    assert response.json()["status"] == status


def test_unexpected_pipeline_failure_is_a_generic_500() -> None:
    """A construction error does not leak its message to the caller."""

    def failing() -> _FakePipeline:
        raise RuntimeError("api key sk-secret should not leak")

    app = create_app(pipeline_factory=failing)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/answer", json={"query": "question"})
    assert response.status_code == 500
    assert response.json() == {"detail": "internal error"}
    assert "sk-secret" not in response.text


def test_missing_model_cache_is_a_503_not_a_500() -> None:
    """A local model/transport failure is retryable and never leaks its path."""

    def unavailable() -> _FakePipeline:
        raise OSError("/models/bge-large missing")

    app = create_app(pipeline_factory=unavailable)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/answer", json={"query": "question"})
    assert response.status_code == 503
    assert response.json() == {"detail": "backend unavailable"}
    assert "bge" not in response.text


def test_readyz_reports_each_database_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness mirrors check_database; one failure degrades it."""

    def healthy() -> list[Any]:
        return [SimpleNamespace(name="connection", ok=True, detail="ok")]

    monkeypatch.setattr("src.api.app.check_database", healthy)
    client, _ = _client(_answer())
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["checks"] == [
        {"name": "connection", "ok": True, "detail": "ok"}
    ]

    def degraded() -> list[Any]:
        return [SimpleNamespace(name="connection", ok=False, detail="refused")]

    monkeypatch.setattr("src.api.app.check_database", degraded)
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["checks"][0]["ok"] is False


def test_readyz_survives_a_check_that_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken probe reports degraded instead of a traceback."""

    def broken() -> list[Any]:
        raise RuntimeError("driver exploded")

    monkeypatch.setattr("src.api.app.check_database", broken)
    client, _ = _client(_answer())
    response = client.get("/readyz")
    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "degraded"
    assert "RuntimeError" in payload["checks"][0]["detail"]


class _FakeSearchRetriever:
    """Return canned hits from the retrieval surface used by /api/v1/search."""

    name = "fake-reranked"

    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        """Store the canned hits and expose the candidate-pool limit."""
        self._hits = hits if hits is not None else _hits()
        self.config = SimpleNamespace(candidates=10, top_k=10)

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return the canned hits."""
        return self._hits

    def retrieve_with_details(self, query: str, top_k: int | None = None) -> Any:
        """Return a result object shaped like RerankedResult."""
        resolved = top_k or 10
        return SimpleNamespace(
            hits=self._hits[:resolved],
            top_k=resolved,
            base_ms=0.5,
            rerank_ms=1.25,
            total_ms=1.75,
            base_name="fake-hybrid",
        )


def _hits() -> list[SearchHit]:
    """Build two complete retrieval hits."""
    return [
        SearchHit(
            chunk_id="chunk-1",
            doc_id="doc-1",
            score=0.91,
            rank=1,
            source="reranked",
            rerank_score=3.5,
            title="Foreign transaction fee",
            category="fees",
            card_tier="platinum",
            effective_year=2026,
            policy_code="FEE-01",
            content="A 3% foreign transaction fee applies.",
        ),
        SearchHit(
            chunk_id="chunk-2",
            doc_id="doc-2",
            score=0.72,
            rank=2,
            source="reranked",
            rerank_score=1.2,
            title="Billing disputes",
            category="disputes",
            content="Disputes may be filed within 60 days.",
        ),
    ]


def _search_client(retriever: Any) -> TestClient:
    """Build an app whose search endpoint uses the injected retriever."""
    return TestClient(create_app(retriever_factory=lambda: retriever))


def test_search_returns_ranked_chunks_not_answers() -> None:
    """The search surface returns chunks and drops generation-only fields."""
    retriever = _FakeSearchRetriever()
    client = _search_client(retriever)
    response = client.post("/api/v1/search", json={"query": "fee?", "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "fee?"
    assert body["top_k"] == 2
    assert body["hit_count"] == 2
    assert body["retriever"] == "fake-hybrid"
    assert body["timings_ms"] == {
        "retrieval_ms": body["timings_ms"]["retrieval_ms"],
        "base_ms": 0.5,
        "rerank_ms": 1.25,
        "total_ms": 1.75,
    }
    first = body["results"][0]
    assert first["chunk_id"] == "chunk-1"
    assert first["doc_id"] == "doc-1"
    assert first["rank"] == 1
    assert first["policy_code"] == "FEE-01"
    assert first["content"].startswith("A 3%")
    assert "answer" not in body
    assert "raw_response" not in response.text


def test_search_rejects_blank_queries_and_out_of_range_top_k() -> None:
    """Blank text, zero Top-K and unknown fields are refused before retrieval."""
    retriever = _FakeSearchRetriever()
    client = _search_client(retriever)
    assert client.post("/api/v1/search", json={"query": "   "}).status_code == 422
    assert client.post("/api/v1/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/api/v1/search", json={"query": "x", "top_k": 51}).status_code == 422
    assert client.post(
        "/api/v1/search", json={"query": "x", "strategy": "dense"}
    ).status_code == 422


def test_search_rejects_top_k_above_the_candidate_pool() -> None:
    """The final Top-K cannot exceed the pool the service was configured with."""
    client = _search_client(_FakeSearchRetriever())
    response = client.post("/api/v1/search", json={"query": "fee?", "top_k": 11})
    assert response.status_code == 422
    assert "10" in response.json()["detail"]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (psycopg.Error("connection refused at 10.0.0.5"), 503),
        (RetrievalError("dense retrieval database error: connection refused"), 503),
        (OSError("/models/bge-large missing"), 503),
        (RuntimeError("api key sk-secret should not leak"), 500),
    ],
)
def test_search_failures_do_not_leak_details(error: Exception, expected: int) -> None:
    """Backend failures map to stable statuses without echoing exception text."""

    class _Failing:
        name = "failing"
        config = SimpleNamespace(candidates=10, top_k=10)

        def retrieve_with_details(self, query: str, top_k: int | None = None) -> Any:
            """Raise the parametrized backend failure."""
            raise error

    client = TestClient(
        create_app(retriever_factory=lambda: _Failing()),
        raise_server_exceptions=False,
    )
    response = client.post("/api/v1/search", json={"query": "fee?"})
    assert response.status_code == expected
    assert "sk-secret" not in response.text
    assert "bge" not in response.text
    assert "10.0.0.5" not in response.text


def test_openapi_documents_the_search_endpoint() -> None:
    """Swagger exposes /api/v1/search with its request and response schema."""
    client = _search_client(_FakeSearchRetriever())
    spec = client.get("/openapi.json").json()
    assert "/api/v1/search" in spec["paths"]
    operation = spec["paths"]["/api/v1/search"]["post"]
    assert operation["tags"] == ["search"]
    assert "requestBody" in operation
    assert "200" in operation["responses"]
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "swagger" in docs.text.lower()
