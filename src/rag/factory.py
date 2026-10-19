"""Factories for the Phase 8 RAG chain.

The wiring is deliberately thin: this module constructs the already accepted
DenseRetriever, SparseRetriever, RRFFusion, HybridRetriever,
CrossEncoderReranker and RerankedRetriever. It adds no retrieval behaviour.

Model loading stays lazy. Building the chain for a dry run or a mock benchmark
does not load the embedding model or the cross-encoder.

Implemented in: Phase 8 (RAG end-to-end).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import Settings, get_settings
from src.ingestion.embedder import EmbeddingProvider
from src.rag.historical import DEFAULT_HISTORICAL, HistoricalRRFRetriever
from src.rag.pipeline import RAGPipeline
from src.rag.provider import LLMProvider, OpenAICompatibleProvider
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.pipeline import HybridRetriever, RerankedRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse import SparseRetriever
from src.schema.rag import LLMConfig, RAGConfig
from src.schema.retrieval import DenseConfig, RerankConfig, SparseConfig


def build_retriever(
    settings: Settings | None = None,
    *,
    local_files_only: bool | None = None,
    embedder: EmbeddingProvider | None = None,
) -> RerankedRetriever:
    """Build Dense + Sparse OR -> RRF -> cross-encoder reranker.

    Args:
        settings: Configuration override; defaults to the project singleton.
        local_files_only: Override the reranker's offline/online loading mode.
            ``None`` uses ``settings.reranker_local_files_only``.
        embedder: Optional query embedder for tests or diagnostics.

    Raises:
        ValueError: the configured final Top-K exceeds the candidate pool.
    """
    settings = settings or get_settings()
    if settings.rag_top_k > settings.rag_candidates:
        raise ValueError(
            f"rag_top_k ({settings.rag_top_k}) must not exceed "
            f"rag_candidates ({settings.rag_candidates})"
        )

    dense = DenseRetriever(
        DenseConfig(top_n=settings.rag_candidates, ef_search=40),
        embedder=embedder,
    )
    sparse = SparseRetriever(
        SparseConfig(top_n=settings.rag_candidates, query_mode="or")
    )
    fusion = RRFFusion(k=60, candidate_k=settings.rag_candidates)
    hybrid = HybridRetriever(
        dense,
        sparse,
        fusion,
        default_top_k=settings.rag_candidates,
    )

    rerank_config = _rerank_config(settings, local_files_only)
    reranker = CrossEncoderReranker(rerank_config)
    return RerankedRetriever(
        hybrid,
        reranker,
        rerank_config,
        default_top_k=settings.rag_top_k,
    )


def _rerank_config(
    settings: Settings,
    local_files_only: bool | None,
) -> RerankConfig:
    """Build the accepted reranker settings shared by live and replay modes."""
    return RerankConfig(
        enabled=True,
        model=settings.reranker_model,
        revision=settings.reranker_revision or None,
        candidates=settings.rag_candidates,
        top_k=settings.rag_top_k,
        batch_size=settings.reranker_batch_size,
        max_length=settings.reranker_max_length,
        device=settings.reranker_device,
        activation="identity",
        local_files_only=(
            settings.reranker_local_files_only
            if local_files_only is None
            else local_files_only
        ),
        cache_folder=str(settings.resolve(settings.reranker_cache_folder)),
    )


def build_historical_retriever(
    settings: Settings | None = None,
    *,
    historical_path: Path = DEFAULT_HISTORICAL,
    local_files_only: bool | None = None,
) -> RerankedRetriever:
    """Replay the accepted RRF pool and rerun the accepted reranker live.

    This mode exists because the host crashed when the embedding model and the
    cross-encoder were resident in one process. Retrieval is therefore labelled
    historical: the candidate identities and RRF ranks come from the accepted
    Phase 6B artifact, while the chunk text is read from PostgreSQL and the
    cross-encoder scores the pool live.
    """
    settings = settings or get_settings()
    if settings.rag_top_k > settings.rag_candidates:
        raise ValueError(
            f"rag_top_k ({settings.rag_top_k}) must not exceed "
            f"rag_candidates ({settings.rag_candidates})"
        )
    base = HistoricalRRFRetriever(
        historical_path,
        candidates=settings.rag_candidates,
    )
    rerank_config = _rerank_config(settings, local_files_only)
    reranker = CrossEncoderReranker(rerank_config)
    return RerankedRetriever(
        base,
        reranker,
        rerank_config,
        default_top_k=settings.rag_top_k,
    )


def build_provider(settings: Settings | None = None) -> OpenAICompatibleProvider:
    """Build the OpenAI-compatible provider from project settings.

    The provider is constructed even when the key is absent; the first call
    raises a config-category error. Scripts can call ``ensure_configured()``
    before loading models to fail fast.
    """
    settings = settings or get_settings()
    return OpenAICompatibleProvider(
        LLMConfig(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            connect_timeout=settings.llm_connect_timeout,
            read_timeout=settings.llm_read_timeout,
            max_retries=settings.llm_max_retries,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    )


def build_pipeline(
    settings: Settings | None = None,
    *,
    provider: LLMProvider | None = None,
    retriever: Any | None = None,
    local_files_only: bool | None = None,
    embedder: EmbeddingProvider | None = None,
) -> RAGPipeline:
    """Build the complete accepted retrieval chain plus the generation layer.

    Args:
        settings: Configuration override.
        provider: Injected generator; defaults to the configured provider.
        retriever: Injected accepted-chain retriever. When omitted, the live
            Dense + Sparse OR -> RRF -> reranker chain is constructed.
        local_files_only: Reranker offline/online loading override.
        embedder: Optional query embedder for tests or diagnostics.
    """
    settings = settings or get_settings()
    resolved_retriever = (
        retriever
        if retriever is not None
        else build_retriever(
            settings,
            local_files_only=local_files_only,
            embedder=embedder,
        )
    )
    resolved_provider = provider if provider is not None else build_provider(settings)
    config = RAGConfig(
        top_k=settings.rag_top_k,
        max_context_chars=settings.rag_max_context_chars,
        max_chunk_chars=settings.rag_max_chunk_chars,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    return RAGPipeline(resolved_retriever, resolved_provider, config)
