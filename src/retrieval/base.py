"""Retriever protocol.

The interface every retrieval arm implements. It is structural: a class only
has to provide ``name`` and a compatible ``retrieve`` method, so the dense,
sparse and fused stages stay independent of one another.

``top_k`` is optional here as a convenience: callers that pass it get exactly
that many candidates requested, while callers that leave it out get the
retriever's configured default (``DenseConfig.top_n`` for dense retrieval).

Implemented by: :class:`src.retrieval.dense.DenseRetriever` (Phase 4). Phase 5
adds the sparse, fused and reranked implementations.
"""

from __future__ import annotations

from typing import Protocol

from src.schema.retrieval import SearchHit


class Retriever(Protocol):
    """A named stage that turns one query into ranked hits."""

    name: str

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return at most ``top_k`` hits, ranked best first.

        Args:
            query: Non-empty natural-language query text.
            top_k: Requested number of hits; ``None`` uses the stage default.

        Raises:
            RetrievalError: when the stage cannot produce results.
        """
        ...
