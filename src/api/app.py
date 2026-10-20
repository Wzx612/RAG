"""Read-only FastAPI service for the accepted RAG chain.

Endpoints:

* ``GET /healthz`` -- process liveness; no database, no model.
* ``GET /readyz``  -- database readiness via :func:`src.db.check_database`.
  Models stay lazy: a missing model cache surfaces as a 503 on ``/answer`` or
  ``/api/v1/search``, not as a readiness lie.
* ``POST /answer``  -- one question through :func:`src.rag.factory.build_pipeline`.
* ``POST /api/v1/search`` -- hybrid retrieval over the knowledge base. It
  returns ranked chunks only, never calls the LLM provider, and reuses the
  answer pipeline's retrieval chain so only one embedding/reranker pair is
  resident per process.
* ``GET /acceptance/`` -- read-only artifact acceptance dashboard; it needs no
  database and no model.
* ``GET /docs`` / ``GET /openapi.json`` -- generated OpenAPI documentation.

The service never exposes the provider key, the prompt or the raw provider
payload. Unexpected exceptions are logged by type and returned as a generic 500.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import psycopg
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from src.api.acceptance import AcceptanceConfig, create_acceptance_router
from src.api.schema import (
    AnswerRequest,
    AnswerResponse,
    HealthResponse,
    ReadinessCheck,
    ReadinessResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from src.db import check_database
from src.rag.factory import build_pipeline
from src.rag.pipeline import RAGPipeline
from src.retrieval import RetrievalError
from src.retrieval.base import Retriever
from src.schema.rag import RAGStatus
from src.schema.retrieval import SearchHit

logger = logging.getLogger(__name__)

PipelineFactory = Callable[[], RAGPipeline]
RetrieverFactory = Callable[[], Retriever]

STATUS_CODES: dict[RAGStatus, int] = {
    "ok": 200,
    "insufficient_evidence": 200,
    "invalid_request": 422,
    "retrieval_error": 503,
    "rerank_error": 503,
    "evidence_error": 503,
    "config_error": 503,
    "llm_timeout": 504,
    "llm_rate_limited": 429,
    "llm_network_error": 503,
    "llm_server_error": 503,
    "llm_auth_error": 503,
    "llm_invalid_response": 502,
    "llm_request_error": 502,
    "llm_error": 503,
}
"""HTTP status per pipeline outcome. Unlisted outcomes are 500."""


def _candidate_limit(retriever: Retriever) -> int | None:
    """Return the retriever's candidate-pool size when it exposes one."""
    config = getattr(retriever, "config", None)
    candidates = getattr(config, "candidates", None)
    if isinstance(candidates, bool) or not isinstance(candidates, int):
        return None
    return candidates


def _run_search(
    retriever: Retriever,
    query: str,
    requested_top_k: int | None,
) -> tuple[list[SearchHit], int, dict[str, float], str]:
    """Run one retrieval pass and return hits, effective Top-K and timings."""
    started = time.perf_counter()
    details_method = getattr(retriever, "retrieve_with_details", None)
    if callable(details_method):
        details = details_method(query, top_k=requested_top_k)
        hits = list(getattr(details, "hits", None) or [])
        resolved = getattr(details, "top_k", requested_top_k)
        if isinstance(resolved, int) and not isinstance(resolved, bool) and resolved >= 1:
            top_k = resolved
        else:
            top_k = requested_top_k if requested_top_k is not None else max(len(hits), 1)
        timings = {"retrieval_ms": _ms_since(started)}
        for field in ("base_ms", "rerank_ms", "total_ms"):
            value = getattr(details, field, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                timings[field] = round(float(value), 3)
        name = str(
            getattr(details, "base_name", None)
            or getattr(retriever, "name", "unknown")
        )
        return hits, top_k, timings, name

    hits = list(retriever.retrieve(query, top_k=requested_top_k))
    resolved_top_k: int
    if requested_top_k is not None:
        resolved_top_k = requested_top_k
    else:
        config = getattr(retriever, "config", None)
        configured = getattr(config, "top_k", None)
        if isinstance(configured, int) and not isinstance(configured, bool) and configured >= 1:
            resolved_top_k = configured
        else:
            resolved_top_k = max(len(hits), 1)
    timings = {"retrieval_ms": _ms_since(started)}
    return hits, resolved_top_k, timings, str(getattr(retriever, "name", "unknown"))


def _project_hit(hit: SearchHit) -> SearchResult:
    """Project a retrieval hit onto the documented response shape."""
    return SearchResult(
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        rank=hit.rank,
        score=hit.score,
        source=hit.source,
        rerank_score=hit.rerank_score,
        title=hit.title,
        category=hit.category,
        card_tier=hit.card_tier,
        effective_year=hit.effective_year,
        policy_code=hit.policy_code,
        content=hit.content,
    )


def _ms_since(started: float) -> float:
    """Return elapsed wall-clock milliseconds as a rounded float."""
    return round(max(0.0, (time.perf_counter() - started) * 1000.0), 3)


def create_app(
    pipeline_factory: PipelineFactory | None = None,
    *,
    acceptance_config: AcceptanceConfig | None = None,
    retriever_factory: RetrieverFactory | None = None,
) -> FastAPI:
    """Build the app, optionally with injected pipeline, retriever and artifacts.

    The default pipeline factory calls :func:`build_pipeline`; its construction
    is cheap and lazy, and the instance is created on the first ``/answer`` or
    ``/api/v1/search`` request. The search endpoint reuses the pipeline's
    retrieval chain by default so the embedding and reranker models are loaded
    once per process. The acceptance router is always mounted and reads only
    static files and artifacts, so ``/acceptance/`` works while the database is
    down.
    """
    application = FastAPI(
        title="Apex RAG API",
        version="1.0.0",
        description=(
            "Hybrid retrieval and grounded question answering over the "
            "card-services knowledge base. Search returns ranked chunks only; "
            "answer adds LLM generation with validated citations. Interactive "
            "OpenAPI documentation is available at /docs."
        ),
        openapi_tags=[
            {
                "name": "search",
                "description": "Hybrid dense + sparse retrieval with reranking; no LLM call.",
            },
            {
                "name": "answer",
                "description": "Grounded answer generation with citation validation.",
            },
            {
                "name": "health",
                "description": "Liveness and database readiness probes.",
            },
            {
                "name": "acceptance",
                "description": "Read-only artifact acceptance dashboard.",
            },
        ],
    )
    application.include_router(create_acceptance_router(acceptance_config))
    factory = pipeline_factory or build_pipeline
    state: dict[str, object] = {}
    lock = threading.Lock()

    def get_pipeline() -> RAGPipeline:
        """Return the process-wide pipeline, building it once on demand."""
        if "pipeline" not in state:
            with lock:
                if "pipeline" not in state:
                    state["pipeline"] = factory()
        return state["pipeline"]  # type: ignore[return-value]

    def get_retriever() -> Retriever:
        """Return the process-wide retrieval chain, sharing the pipeline's models."""
        if "retriever" in state:
            return state["retriever"]  # type: ignore[return-value]
        candidate = (
            retriever_factory()
            if retriever_factory is not None
            else get_pipeline().retriever
        )
        with lock:
            state.setdefault("retriever", candidate)
        return state["retriever"]  # type: ignore[return-value]

    @application.get("/healthz", response_model=HealthResponse, tags=["health"])
    def healthz() -> HealthResponse:
        """Report that the process is alive."""
        return HealthResponse()

    @application.get("/readyz", response_model=ReadinessResponse, tags=["health"])
    def readyz(response: Response) -> ReadinessResponse:
        """Report database readiness; model availability is checked on first use."""
        try:
            results = check_database()
        except Exception as exc:  # noqa: BLE001 -- readiness must not crash the probe
            checks = [
                ReadinessCheck(
                    name="database",
                    ok=False,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            ]
        else:
            checks = [
                ReadinessCheck(name=item.name, ok=item.ok, detail=item.detail)
                for item in results
            ]
        ready = bool(checks) and all(check.ok for check in checks)
        if not ready:
            response.status_code = 503
        return ReadinessResponse(
            status="ready" if ready else "degraded",
            checks=checks,
        )

    @application.post("/answer", response_model=AnswerResponse, tags=["answer"])
    def answer(request: AnswerRequest) -> JSONResponse:
        """Answer one question and return the pipeline's structured outcome."""
        try:
            pipeline = get_pipeline()
            result = pipeline.answer(request.query, query_id=request.query_id)
        except psycopg.Error:
            logger.exception("answer request failed: database unavailable")
            return JSONResponse(
                status_code=503,
                content={"detail": "retrieval backend unavailable"},
            )
        except OSError:
            # ``ConnectionError`` is an ``OSError``: transport failures and a
            # missing model cache both land here. Both are retryable, and the
            # path or endpoint must not reach the caller.
            logger.exception("answer request failed: local backend unavailable")
            return JSONResponse(
                status_code=503,
                content={"detail": "backend unavailable"},
            )
        except Exception:
            logger.exception("answer request failed unexpectedly")
            return JSONResponse(
                status_code=500,
                content={"detail": "internal error"},
            )
        payload = AnswerResponse.model_validate(result)
        return JSONResponse(
            status_code=STATUS_CODES.get(result.status, 500),
            content=payload.model_dump(mode="json"),
        )

    @application.post(
        "/api/v1/search",
        response_model=SearchResponse,
        tags=["search"],
        summary="Hybrid retrieval over the card-services knowledge base",
        responses={
            422: {"description": "Invalid query or Top-K outside the configured pool."},
            503: {"description": "Retrieval backend or local model unavailable."},
            500: {"description": "Unexpected server error."},
        },
    )
    def search(request: SearchRequest) -> JSONResponse:
        """Return ranked chunks; this endpoint never calls the LLM provider."""
        started = time.perf_counter()
        try:
            retriever = get_retriever()
            limit = _candidate_limit(retriever)
            if request.top_k is not None and limit is not None and request.top_k > limit:
                return JSONResponse(
                    status_code=422,
                    content={
                        "detail": (
                            "top_k must not exceed the configured candidate pool "
                            f"({limit})"
                        )
                    },
                )
            hits, top_k, timings, retriever_name = _run_search(
                retriever, request.query, request.top_k
            )
        except psycopg.Error:
            logger.exception("search request failed: database unavailable")
            return JSONResponse(
                status_code=503,
                content={"detail": "retrieval backend unavailable"},
            )
        except RetrievalError:
            # The retrieval stages wrap driver and model failures in their own
            # stable exception, with the same 503 meaning as /answer.
            logger.exception("search request failed: retrieval stage unavailable")
            return JSONResponse(
                status_code=503,
                content={"detail": "retrieval backend unavailable"},
            )
        except OSError:
            logger.exception("search request failed: local backend unavailable")
            return JSONResponse(
                status_code=503,
                content={"detail": "backend unavailable"},
            )
        except Exception:
            logger.exception("search request failed unexpectedly")
            return JSONResponse(
                status_code=500,
                content={"detail": "internal error"},
            )

        payload = SearchResponse(
            query=request.query,
            top_k=top_k,
            hit_count=len(hits),
            retriever=retriever_name,
            results=[_project_hit(hit) for hit in hits],
            timings_ms=timings,
        )
        logger.info(
            "search: hits=%d top_k=%d retriever=%s total_ms=%.2f",
            len(hits),
            top_k,
            retriever_name,
            (time.perf_counter() - started) * 1000.0,
        )
        return JSONResponse(
            status_code=200,
            content=payload.model_dump(mode="json"),
        )

    return application


app = create_app()
"""ASGI entry point for ``uvicorn src.api.app:app``."""
