r"""Cross-encoder reranking.

A reranker is a second stage: it receives a query and the candidates produced
by an existing retriever, scores each ``(query, passage)`` pair, and returns the
candidates in relevance order. It does no retrieval, no fusion and no database
access of its own.

This module wraps ``sentence_transformers.CrossEncoder`` and keeps the model
lazy: importing :mod:`src.retrieval.reranker` must not load ~1.1 GB of weights,
so tests and callers that never rerank stay cheap. One instance owns one model
and must be reused across queries; loading a second copy is not necessary.

Score direction and activation
-----------------------------

``BAAI/bge-reranker-base`` is a one-label sequence-classification model. Its
raw output is an unbounded relevance logit: higher means more relevant, and
only the relative order is meaningful. The default ``activation="identity"``
therefore returns the raw logits. ``activation="sigmoid"`` maps them to
``(0, 1)`` with a monotonic transform, which preserves the order but changes
the score scale. Reranker scores are never comparable with cosine similarity
or ``ts_rank_cd``.

Text and identity
-----------------

A pair's passage text is ``title + "\\n" + content`` when a title is present,
otherwise just ``content``. Missing or blank content is an input error, not a
candidate to drop silently. Candidate identity is
``(chunk_strategy, chunk_id)``; duplicate identities are rejected because they
would make score-to-ID mapping ambiguous.

Failure behaviour
-----------------

No silent fallback exists. A model-initialisation failure, inference failure,
resource shortage or invalid input raises :class:`RerankError`; the original
exception is kept as ``__cause__``. Callers must not present the pre-rerank
list as a reranked result.

Implemented in: Phase 7B.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

import numpy as np

from src.retrieval import RetrievalError
from src.schema.retrieval import RerankConfig, SearchHit

RerankStage = Literal["input", "model_init", "inference", "resource", "base_retrieval"]
"""Stable failure categories recorded by callers and tests."""

TOP_K_MAX = 1000
"""Largest accepted final Top-K, matching the other retrievers' cap."""


class RerankError(RetrievalError):
    """A reranking request could not be completed.

    ``stage`` distinguishes invalid input from model initialisation, model
    inference, a resource shortage, and a failure in the wrapped base retriever.
    The original exception is always available as ``__cause__``.
    """

    def __init__(self, message: str, *, stage: RerankStage = "inference") -> None:
        """Store the message and its stable failure category."""
        super().__init__(message)
        self.stage: RerankStage = stage


class Reranker(Protocol):
    """The interface a reranking stage exposes to a pipeline.

    Provenance fields are read-only properties because the wrapper computes
    them from its configuration; a caller must not be able to overwrite the
    model identity a report records.
    """

    name: str

    @property
    def model_id(self) -> str:
        """Configured model id or local snapshot path."""
        ...

    @property
    def revision(self) -> str | None:
        """Configured model revision, if any."""
        ...

    @property
    def device(self) -> str:
        """Execution device."""
        ...

    @property
    def batch_size(self) -> int:
        """Pairs per forward pass."""
        ...

    @property
    def max_length(self) -> int:
        """Tokenizer truncation length."""
        ...

    @property
    def activation(self) -> str:
        """Score transform."""
        ...

    def score(self, query: str, candidates: Sequence[SearchHit]) -> list[float]:
        """Return one relevance score per candidate, in input order."""
        ...

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchHit],
        top_k: int | None = None,
    ) -> list[SearchHit]:
        """Return candidates ordered by relevance, truncated to ``top_k``."""
        ...


class CrossEncoderReranker:
    """A lazy, reusable cross-encoder scoring stage.

    The model is loaded on first use and kept for the process lifetime. Tests
    inject a fake backend through ``model`` or ``model_factory``; neither path
    touches the real weights.
    """

    name = "cross_encoder"

    def __init__(
        self,
        config: RerankConfig | None = None,
        *,
        model: Any | None = None,
        model_factory: Callable[[], Any] | None = None,
    ) -> None:
        """Bind configuration and an optional backend.

        Args:
            config: Model id, revision, device, batch size and scoring options.
            model: Pre-built backend exposing ``predict``. If supplied, no
                loading occurs and ``model_factory`` must be omitted.
            model_factory: Callable returning a backend, used by tests to
                simulate initialisation success or failure without weights.

        Raises:
            ValueError: if both ``model`` and ``model_factory`` are supplied, or
                an injected model has no callable ``predict``.
        """
        if model is not None and model_factory is not None:
            raise ValueError("pass either model or model_factory, not both")
        if model is not None and not callable(getattr(model, "predict", None)):
            raise ValueError("injected model must expose a callable predict()")
        self._config = config if config is not None else RerankConfig()
        self._model = model
        self._model_factory = model_factory
        self._load_ms: float | None = None

    # ------------------------------------------------------------- properties

    @property
    def config(self) -> RerankConfig:
        """The effective reranker configuration."""
        return self._config

    @property
    def model_id(self) -> str:
        """Configured model id or local snapshot path; recorded in reports."""
        return self._config.model

    @property
    def revision(self) -> str | None:
        """Configured model revision, if any."""
        return self._config.revision

    @property
    def device(self) -> str:
        """Execution device passed to the backend."""
        return self._config.device

    @property
    def batch_size(self) -> int:
        """Pairs per forward pass used by :meth:`score`."""
        return self._config.batch_size

    @property
    def max_length(self) -> int:
        """Tokenizer truncation length for each pair."""
        return self._config.max_length

    @property
    def activation(self) -> str:
        """Score transform: ``identity`` (raw logits) or ``sigmoid``."""
        return self._config.activation

    @property
    def is_loaded(self) -> bool:
        """Whether a backend is currently held by this instance."""
        return self._model is not None

    @property
    def backend(self) -> Any | None:
        """The loaded backend, or ``None``. Read-only; used by diagnostics.

        Exposing the object lets the latency probe reuse one loaded model across
        batch-size variants instead of loading a second copy.
        """
        return self._model

    @property
    def load_ms(self) -> float | None:
        """Wall-clock milliseconds spent in :meth:`load`, or ``None``."""
        return self._load_ms

    # ------------------------------------------------------------------ model

    def load(self) -> CrossEncoderReranker:
        """Load the backend once and return ``self``.

        Raises:
            RerankError: initialisation failed. ``stage`` is ``resource`` when
                the failure is a recognised memory-allocation error, otherwise
                ``model_init``.
        """
        if self._model is not None:
            return self
        started = time.perf_counter()
        try:
            backend = self._model_factory() if self._model_factory else self._build_model()
        except Exception as exc:  # noqa: BLE001 -- re-raised with a stable category
            stage: RerankStage = "resource" if _is_resource_failure(exc) else "model_init"
            raise RerankError(
                f"reranker model initialization failed for {self.model_id!r}: {exc}",
                stage=stage,
            ) from exc
        if not callable(getattr(backend, "predict", None)):
            raise RerankError(
                f"reranker model initialization failed for {self.model_id!r}: "
                "backend has no callable predict()",
                stage="model_init",
            )
        self._model = backend
        self._load_ms = (time.perf_counter() - started) * 1000.0
        return self

    def _build_model(self) -> Any:
        """Construct the configured SentenceTransformers CrossEncoder."""
        from sentence_transformers import CrossEncoder

        kwargs: dict[str, Any] = {
            "model_name_or_path": self._config.model,
            "device": self._config.device,
            "max_length": self._config.max_length,
            "activation_fn": _activation_fn(self._config.activation),
            "local_files_only": self._config.local_files_only,
        }
        if self._config.revision is not None:
            kwargs["revision"] = self._config.revision
        if self._config.cache_folder is not None:
            kwargs["cache_folder"] = self._config.cache_folder
        return CrossEncoder(**kwargs)

    # ----------------------------------------------------------------- scoring

    def score(self, query: str, candidates: Sequence[SearchHit]) -> list[float]:
        """Score every candidate against ``query`` in one batched call.

        Args:
            query: Non-empty natural-language query.
            candidates: Hits to score, in the order their scores should be
                returned. Empty input returns ``[]`` without loading the model.

        Returns:
            One finite relevance score per candidate, in input order.

        Raises:
            RerankError: invalid input, duplicate candidate identity, missing
                passage text, model-initialisation failure, inference failure or
                a non-finite/count-mismatched model output.
        """
        resolved_query = _validate_query(query)
        items = _validate_candidates(candidates)
        if not items:
            return []
        pairs = [[resolved_query, rerank_text(hit)] for hit in items]
        backend = self.load()._model
        assert backend is not None  # load() either raised or set the model

        try:
            raw = backend.predict(
                pairs,
                batch_size=self.batch_size,
                activation_fn=_activation_fn(self.activation),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001 -- re-raised with a stable category
            stage: RerankStage = "resource" if _is_resource_failure(exc) else "inference"
            raise RerankError(
                f"reranker inference failed for {len(pairs)} pair(s): {exc}",
                stage=stage,
            ) from exc

        try:
            values = np.asarray(raw, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise RerankError(
                f"reranker returned an unreadable score array: {exc}",
                stage="inference",
            ) from exc
        if values.size != len(pairs):
            raise RerankError(
                f"reranker returned {values.size} scores for {len(pairs)} pairs; "
                "expected exactly one relevance score per pair",
                stage="inference",
            )
        if not np.all(np.isfinite(values)):
            raise RerankError(
                "reranker returned a non-finite score",
                stage="inference",
            )
        return [float(value) for value in values]

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchHit],
        top_k: int | None = None,
    ) -> list[SearchHit]:
        """Score and order candidates, then truncate to ``top_k``.

        Sorting is deterministic: score descending, then the candidate's
        pre-rerank rank, then ``(chunk_strategy, chunk_id)``. Returned hits have
        ``source="reranked"``, ``rank`` 1..N, ``rerank_score``,
        ``pre_rerank_rank`` and ``pre_rerank_score``; all other metadata and
        fusion provenance are preserved. ``distance`` is cleared because the new
        score is no longer a cosine-derived value.

        Args:
            query: Non-empty natural-language query.
            candidates: Candidate pool, normally the fused Top-K.
            top_k: Final list length. ``None`` uses ``RerankConfig.top_k``. A
                value larger than the pool returns the whole pool.

        Returns:
            Reranked hits, best first.

        Raises:
            RerankError: same input and backend failures as :meth:`score`.
            ValueError: invalid ``top_k``.
        """
        resolved_query = _validate_query(query)
        items = _validate_candidates(candidates)
        resolved_top_k = _resolve_top_k(top_k, len(items), self._config.top_k)
        if not items:
            return []

        scores = self.score(resolved_query, items)
        order = sorted(
            range(len(items)),
            key=lambda index: (
                -scores[index],
                items[index].rank,
                items[index].chunk_strategy or "",
                items[index].chunk_id,
            ),
        )
        return [
            _reranked_hit(items[index], score=scores[index], rank=new_rank)
            for new_rank, index in enumerate(order[:resolved_top_k], start=1)
        ]


def rerank_text(hit: SearchHit) -> str:
    """Return the passage text a reranker scores for ``hit``.

    ``title`` is prepended when present, separated by a newline. Content is
    required: a missing or blank body raises instead of silently scoring the
    title alone, because that would change what the candidate represents.

    Raises:
        RerankError: ``hit`` has no usable content.
    """
    content = hit.content
    if not isinstance(content, str) or not content.strip():
        raise RerankError(
            f"candidate {hit.chunk_id!r} has no content to rerank",
            stage="input",
        )
    title = hit.title.strip() if isinstance(hit.title, str) else ""
    return f"{title}\n{content}" if title else content


def _activation_fn(name: str) -> Any:
    """Return the configured output activation as a torch module."""
    import torch

    if name == "identity":
        return torch.nn.Identity()
    if name == "sigmoid":
        return torch.nn.Sigmoid()
    raise ValueError(f"unsupported reranker activation: {name!r}")


def _validate_query(query: object) -> str:
    """Return a stripped non-empty query, or raise an input-category error."""
    if not isinstance(query, str):
        raise RerankError(
            f"query must be a str, got {type(query).__name__}",
            stage="input",
        )
    stripped = query.strip()
    if not stripped:
        raise RerankError("query must be a non-empty string", stage="input")
    return stripped


def _validate_candidates(candidates: Sequence[SearchHit]) -> list[SearchHit]:
    """Materialise and validate the candidate list, including identities."""
    items = list(candidates)
    seen: set[tuple[str, str]] = set()
    for position, hit in enumerate(items, start=1):
        if not isinstance(hit, SearchHit):
            raise RerankError(
                f"candidate item {position} is not a SearchHit",
                stage="input",
            )
        if not isinstance(hit.chunk_id, str) or not hit.chunk_id:
            raise RerankError(f"candidate item {position} has an empty chunk_id", stage="input")
        key = (hit.chunk_strategy or "", hit.chunk_id)
        if key in seen:
            raise RerankError(
                f"duplicate candidate identity {key!r}; refusing ambiguous score mapping",
                stage="input",
            )
        seen.add(key)
    return items


def _resolve_top_k(top_k: int | None, pool_size: int, default: int) -> int:
    """Validate a final Top-K; ``None`` uses the configured default."""
    if top_k is None:
        return min(default, pool_size)
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError(f"top_k must be an int, got {type(top_k).__name__}")
    if not 1 <= top_k <= TOP_K_MAX:
        raise ValueError(f"top_k must be between 1 and {TOP_K_MAX}, got {top_k}")
    return min(top_k, pool_size)


def _reranked_hit(hit: SearchHit, *, score: float, rank: int) -> SearchHit:
    """Copy one candidate with reranked ordering and score provenance."""
    return SearchHit(
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        score=score,
        distance=None,
        rank=rank,
        source="reranked",
        dense_rank=hit.dense_rank,
        sparse_rank=hit.sparse_rank,
        dense_score=hit.dense_score,
        sparse_score=hit.sparse_score,
        rrf_score=hit.rrf_score,
        candidate_sources=hit.candidate_sources,
        rerank_score=score,
        pre_rerank_rank=hit.rank,
        pre_rerank_score=hit.score,
        content=hit.content,
        chunk_strategy=hit.chunk_strategy,
        chunk_index=hit.chunk_index,
        title=hit.title,
        category=hit.category,
        card_tier=hit.card_tier,
        effective_year=hit.effective_year,
        policy_code=hit.policy_code,
    )


def _is_resource_failure(exc: BaseException) -> bool:
    """Recognise common memory-allocation failures without masking others."""
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__.lower()
    if "outofmemory" in name or "out of memory" in name:
        return True
    message = str(exc).lower()
    markers = (
        "out of memory",
        "not enough memory",
        "insufficient memory",
        "cannot allocate memory",
        "can't allocate memory",
        "bad_alloc",
    )
    return any(marker in message for marker in markers)
