"""Embedding backend.

Text -> vectors. Wraps SentenceTransformer behind one interface, and handles
the BGE query-instruction prefix. No storage, no search.

Implemented in: Phase 3 (ingestion)
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from numbers import Real
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.config import Settings, get_settings
from src.ingestion import IngestionError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# (batches_done, batches_total) -- lets the CLI print "Batch 2/5" without the
# embedder knowing anything about terminals.
ProgressFn = Callable[[int, int], None]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Text -> vectors. The only thing the write path needs from a model."""

    dimension: int

    def embed_documents(
        self, texts: Sequence[str], on_batch: ProgressFn | None = None
    ) -> list[list[float]]:
        """Embed passages. No instruction prefix -- BGE applies it to queries only."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query. Unused until Phase 4; defined so the prefix lives here."""
        ...


class SentenceTransformerEmbedder:
    """Local SentenceTransformer backend.

    The model is loaded on first use, not at construction: importing this module
    must stay free so that unit tests can inject a stub without pulling 1.3 GB
    of weights off disk.
    """

    def __init__(self, settings: Settings | None = None, model: Any | None = None) -> None:
        """Bind configuration and an optional pre-built backend.

        Args:
            settings: Override the settings singleton (tests).
            model: Backend to use instead of loading real weights. Tests pass a
                stub here; production leaves it None so the model loads lazily.
        """
        self._settings = settings or get_settings()
        self._model = model

    @property
    def dimension(self) -> int:
        """Vector width this provider emits, from configuration."""
        return self._settings.embedding_dim

    @property
    def model(self) -> Any:
        """The underlying SentenceTransformer, loaded once."""
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self) -> Any:
        from sentence_transformers import SentenceTransformer

        name = self._settings.embedding_model
        logger.info("loading embedding model %s on %s", name, self._settings.embedding_device)
        started = time.perf_counter()
        model = SentenceTransformer(name, device=self._settings.embedding_device)
        logger.info("model ready in %.1fs", time.perf_counter() - started)
        return model

    def embed_documents(
        self, texts: Sequence[str], on_batch: ProgressFn | None = None
    ) -> list[list[float]]:
        """Embed passages in fixed-size batches, validating every batch.

        A batch that fails is retried a bounded number of times with exponential
        backoff; if it still fails the import aborts rather than storing a
        partial corpus.
        """
        texts = list(texts)
        if not texts:
            return []

        batch_size = max(1, self._settings.embedding_batch_size)
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        vectors: list[list[float]] = []
        for number, batch in enumerate(batches, start=1):
            encoded = self._encode(batch, batch_number=number, batch_count=len(batches))
            vectors.extend(encoded)
            if on_batch is not None:
                on_batch(number, len(batches))

        validate_vectors(vectors, self.dimension, source=self._settings.embedding_model)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one query, with the instruction prefix BGE was trained with.

        The asymmetry is deliberate and required by the model: bge-en-v1.5 asks
        for the prefix on the query side only, so documents are embedded without
        it (see :meth:`embed_documents`).
        """
        vector = self._encode([self._settings.query_instruction + text])[0]
        validate_vectors([vector], self.dimension, source="query")
        return vector

    def _encode(self, batch: Sequence[str], *, batch_number: int = 1, batch_count: int = 1) -> Any:
        """Encode one batch as ``list[list[float]]``, retrying transient failures."""
        attempts = max(1, self._settings.embedding_max_retries)
        last: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                encoded = self.model.encode(
                    list(batch),
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                return np.asarray(encoded).tolist()
            except Exception as exc:  # noqa: BLE001 -- any backend failure is retryable here
                last = exc
                if attempt == attempts:
                    break
                delay = 2 ** (attempt - 1)
                logger.warning(
                    "batch %d/%d failed (attempt %d/%d): %s: %s -- retrying in %ds",
                    batch_number,
                    batch_count,
                    attempt,
                    attempts,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise IngestionError(
            f"embedding batch {batch_number}/{batch_count} failed after {attempts} attempts: "
            f"{type(last).__name__}: {last}"
        )


def validate_vectors(
    vectors: Sequence[Sequence[float]], expected_dimension: int, *, source: str = "documents"
) -> None:
    """Reject anything that must not reach a ``VECTOR(n)`` column.

    Width mismatches are never padded, truncated or reshaped -- a silently
    resized vector produces a plausible-looking index and a wrong Recall@K.
    Non-finite components are rejected for the same reason: PostgreSQL stores
    ``NaN`` happily and pgvector then orders it in a way no one expects.

    Raises:
        IngestionError: naming the offending index and both dimensions.
    """
    for index, vector in enumerate(vectors):
        if not isinstance(vector, (list, tuple)):
            raise IngestionError(
                f"{source}[{index}] is {type(vector).__name__}, expected a vector of floats"
            )
        if len(vector) != expected_dimension:
            raise IngestionError(
                f"{source}[{index}] has dimension {len(vector)}, expected {expected_dimension}. "
                "Refusing to pad, truncate or reshape."
            )
        for position, value in enumerate(vector):
            # `numbers.Real` rather than a `float()` coercion: float("0.5")
            # succeeds, so coercing would let a string through the one guard
            # standing between a bad batch and a VECTOR(n) column. `bool` is
            # excluded explicitly -- it is an `int`, but not a component.
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise IngestionError(
                    f"{source}[{index}][{position}] is {value!r}, which is not a finite number"
                )
