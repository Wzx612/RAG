"""RAG answer pipeline.

The package reuses the accepted retrieval and reranking stages and adds only
the generation-facing layers: context assembly, prompt construction, an
OpenAI-compatible provider, citation validation and a structured answer.

Implemented in: Phase 8 (RAG end-to-end).
"""

from __future__ import annotations

from src.rag.context import ContextBundle, ContextError, build_context
from src.rag.historical import HistoricalRRFRetriever
from src.rag.pipeline import RAGPipeline

__all__ = [
    "ContextBundle",
    "ContextError",
    "HistoricalRRFRetriever",
    "RAGPipeline",
    "build_context",
]
