"""Phase 7B: cross-encoder reranking, without real weights or a database.

The fake backend records every ``predict`` call and returns controlled scores,
so these tests pin input validation, score-to-ID mapping, ordering, Top-K
truncation, provenance, failure categories and the pipeline's no-fallback rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from src.retrieval import RetrievalError
from src.retrieval.pipeline import RerankedRetriever
from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankError,
    rerank_text,
)
from src.schema.retrieval import RerankConfig, SearchHit


class _FakeBackend:
    """A CrossEncoder-shaped backend with scripted scores or an exception."""

    def __init__(
        self,
        scores: Sequence[float] | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        """Store scripted output and count inference calls."""
        self.scores = list(scores) if scores is not None else None
        self.error = error
        self.predict_calls = 0
        self.predictions: list[tuple[list[str], dict[str, Any]]] = []

    def predict(self, pairs: Sequence[Sequence[str]], **kwargs: Any) -> np.ndarray:
        """Return the next scripted scores, or raise the scripted exception."""
        self.predict_calls += 1
        self.predictions.append(([f"{left} || {right}" for left, right in pairs], dict(kwargs)))
        if self.error is not None:
            raise self.error
        if self.scores is None:
            return np.zeros(len(pairs), dtype=float)
        return np.asarray(self.scores[: len(pairs)], dtype=float)


class _FixedBase:
    """A retriever that returns a fixed list and records calls."""

    name = "fixed_base"

    def __init__(
        self,
        hits: Sequence[SearchHit],
        *,
        error: BaseException | None = None,
    ) -> None:
        """Store the fixed list and an optional failure."""
        self.hits = list(hits)
        self.error = error
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return the fixed list, truncated to ``top_k``."""
        self.calls.append((query, top_k))
        if self.error is not None:
            raise self.error
        if top_k is None:
            return list(self.hits)
        return list(self.hits[:top_k])


def _config(**overrides: Any) -> RerankConfig:
    """Build a small test configuration with optional overrides."""
    values: dict[str, Any] = {
        "model": "fake-reranker",
        "candidates": 50,
        "top_k": 10,
        "batch_size": 4,
        "max_length": 128,
    }
    values.update(overrides)
    return RerankConfig(**values)


def _hit(
    chunk_id: str,
    *,
    rank: int = 1,
    score: float = 0.5,
    content: str | None = None,
    title: str | None = None,
    source: Literal["dense", "sparse", "fused"] = "fused",
    doc_id: str | None = None,
) -> SearchHit:
    """Build a candidate with rich provenance for preservation checks."""
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=doc_id or f"DOC-{chunk_id}",
        score=score,
        distance=1.0 - score if source == "dense" else None,
        rank=rank,
        source=source,
        dense_rank=rank if source in {"dense", "fused"} else None,
        sparse_rank=rank if source in {"sparse", "fused"} else None,
        dense_score=0.9 if source in {"dense", "fused"} else None,
        sparse_score=4.2 if source in {"sparse", "fused"} else None,
        rrf_score=score if source == "fused" else None,
        candidate_sources=(
            ("dense", "sparse")
            if source == "fused"
            else (source,)
        ),
        content=content if content is not None else f"content for {chunk_id}",
        chunk_strategy="document",
        chunk_index=0,
        title=title if title is not None else f"title {chunk_id}",
        category="fees",
        card_tier="platinum",
        effective_year=2025,
        policy_code="BT-001",
    )


def _reranker(
    backend: _FakeBackend,
    **config: Any,
) -> CrossEncoderReranker:
    """Build a reranker around an already constructed fake backend."""
    return CrossEncoderReranker(_config(**config), model=backend)


# ------------------------------------------------------------------ model lifecycle


def test_load_is_lazy_and_reuses_one_backend() -> None:
    """Construction does not load; repeated load() calls reuse one backend."""
    backend = _FakeBackend(scores=[1.0])
    calls: list[int] = []

    def factory() -> _FakeBackend:
        calls.append(1)
        return backend

    reranker = CrossEncoderReranker(_config(), model_factory=factory)

    assert not reranker.is_loaded
    assert reranker.load_ms is None
    assert reranker.load() is reranker
    assert reranker.load() is reranker
    assert calls == [1]
    assert reranker.is_loaded
    assert reranker.load_ms is not None
    assert reranker.model_id == "fake-reranker"


def test_model_initialization_failure_is_categorised_and_chained() -> None:
    """A factory failure becomes a model_init RerankError with its cause."""
    def factory() -> Any:
        raise RuntimeError("weights missing")

    reranker = CrossEncoderReranker(_config(), model_factory=factory)

    with pytest.raises(RerankError) as raised:
        reranker.score("q", [_hit("A")])

    assert raised.value.stage == "model_init"
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "fake-reranker" in str(raised.value)
    assert not reranker.is_loaded


def test_model_initialization_memory_failure_is_resource_stage() -> None:
    """An initialisation MemoryError is classified as a resource failure."""
    def factory() -> Any:
        raise MemoryError("not enough memory")

    reranker = CrossEncoderReranker(_config(), model_factory=factory)
    with pytest.raises(RerankError) as raised:
        reranker.load()

    assert raised.value.stage == "resource"
    assert isinstance(raised.value.__cause__, MemoryError)


def test_injected_backend_is_not_loaded_or_replaced() -> None:
    """An injected backend is used directly and is never loaded twice."""
    backend = _FakeBackend(scores=[0.4])
    reranker = _reranker(backend)

    assert reranker.is_loaded
    assert reranker.load() is reranker
    assert backend.predict_calls == 0
    reranker.score("q", [_hit("A")])
    assert backend.predict_calls == 1


# --------------------------------------------------------------------- text input


def test_rerank_text_includes_title_then_content() -> None:
    """Title and body are both presented to the cross-encoder."""
    assert rerank_text(_hit("A", title="My title", content="My body")) == "My title\nMy body"


def test_rerank_text_allows_a_missing_title() -> None:
    """A missing title does not hide a valid body."""
    assert rerank_text(_hit("A", title="", content="Body only")) == "Body only"


def test_rerank_text_requires_content() -> None:
    """Missing or blank content is an input error, not a dropped candidate."""
    for content in (None, "", "   "):
        hit = _hit("A").model_copy(update={"content": content})
        with pytest.raises(RerankError) as raised:
            rerank_text(hit)
        assert raised.value.stage == "input"
        assert "A" in str(raised.value)


@pytest.mark.parametrize("query", [None, 123, "", "   "])
def test_invalid_query_is_an_input_error(query: object) -> None:
    """A non-string or blank query is rejected before the model runs."""
    backend = _FakeBackend(scores=[1.0])
    reranker = _reranker(backend)

    with pytest.raises(RerankError) as raised:
        reranker.score(query, [_hit("A")])  # type: ignore[arg-type]

    assert raised.value.stage == "input"
    assert backend.predict_calls == 0


def test_invalid_candidate_item_is_an_input_error() -> None:
    """A non-SearchHit item is rejected instead of crashing later."""
    backend = _FakeBackend(scores=[1.0])
    reranker = _reranker(backend)

    with pytest.raises(RerankError) as raised:
        reranker.score("q", [object()])  # type: ignore[list-item]

    assert raised.value.stage == "input"
    assert backend.predict_calls == 0


# ------------------------------------------------------------- scoring and mapping


def test_score_returns_one_finite_score_per_candidate_in_input_order() -> None:
    """The output count equals the input count and keeps input order."""
    backend = _FakeBackend(scores=[0.1, 2.5, -3.0])
    reranker = _reranker(backend)
    candidates = [_hit("A"), _hit("B"), _hit("C")]

    scores = reranker.score("q", candidates)

    assert scores == pytest.approx([0.1, 2.5, -3.0])
    assert len(scores) == len(candidates)
    assert all(np.isfinite(scores))


def test_rerank_sorts_descending_and_maps_scores_back_to_ids() -> None:
    """Higher scores rank first and each score stays with its candidate ID."""
    backend = _FakeBackend(scores=[0.2, 2.0, 1.0])
    reranker = _reranker(backend)
    candidates = [_hit("A", rank=1), _hit("B", rank=2), _hit("C", rank=3)]

    result = reranker.rerank("q", candidates, top_k=3)

    assert [hit.chunk_id for hit in result] == ["B", "C", "A"]
    assert [hit.rank for hit in result] == [1, 2, 3]
    assert [hit.rerank_score for hit in result] == pytest.approx([2.0, 1.0, 0.2])
    assert [hit.pre_rerank_rank for hit in result] == [2, 3, 1]
    assert [hit.pre_rerank_score for hit in result] == pytest.approx(
        [candidates[1].score, candidates[2].score, candidates[0].score]
    )


def test_ties_are_broken_by_pre_rerank_rank() -> None:
    """Equal scores fall back to the pre-rerank order, deterministically."""
    backend = _FakeBackend(scores=[1.0, 1.0, 1.0])
    reranker = _reranker(backend)
    candidates = [_hit("A", rank=2), _hit("B", rank=1), _hit("C", rank=3)]

    result = reranker.rerank("q", candidates, top_k=3)

    assert [hit.chunk_id for hit in result] == ["B", "A", "C"]


def test_all_pairs_are_scored_in_one_batched_call() -> None:
    """The reranker batches the pool instead of calling predict per candidate."""
    backend = _FakeBackend(scores=[0.1, 0.2, 0.3, 0.4, 0.5])
    reranker = _reranker(backend, batch_size=3)

    reranker.score("q", [_hit(name) for name in "ABCDE"])

    assert backend.predict_calls == 1
    assert len(backend.predictions[0][0]) == 5
    assert backend.predictions[0][1]["batch_size"] == 3
    assert backend.predictions[0][1]["show_progress_bar"] is False


def test_identity_activation_is_the_default() -> None:
    """The default output transform is identity, i.e. raw logits."""
    backend = _FakeBackend(scores=[0.5])
    reranker = _reranker(backend)
    reranker.score("q", [_hit("A")])

    assert isinstance(backend.predictions[0][1]["activation_fn"], torch.nn.Identity)


def test_sigmoid_activation_is_forwarded_to_the_backend() -> None:
    """The configured activation is passed on every predict call."""
    backend = _FakeBackend(scores=[0.5])
    reranker = _reranker(backend, activation="sigmoid")
    reranker.score("q", [_hit("A")])

    assert isinstance(backend.predictions[0][1]["activation_fn"], torch.nn.Sigmoid)


# ------------------------------------------------------------------ empty and Top-K


def test_empty_candidates_return_empty_without_loading() -> None:
    """An empty pool is valid and must not load the model."""
    calls: list[int] = []

    def factory() -> Any:
        calls.append(1)
        raise AssertionError("model must not load for an empty pool")

    reranker = CrossEncoderReranker(_config(), model_factory=factory)

    assert reranker.score("q", []) == []
    assert reranker.rerank("q", []) == []
    assert calls == []
    assert not reranker.is_loaded


def test_duplicate_candidate_identity_is_rejected() -> None:
    """Ambiguous score-to-ID mapping is rejected, not silently deduplicated."""
    backend = _FakeBackend(scores=[1.0, 2.0])
    reranker = _reranker(backend)

    with pytest.raises(RerankError) as raised:
        reranker.score("q", [_hit("A"), _hit("A")])

    assert raised.value.stage == "input"
    assert "duplicate" in str(raised.value)
    assert backend.predict_calls == 0


def test_top_k_truncates_after_sorting_and_renumbers() -> None:
    """Only the best ``top_k`` survive, with ranks 1..N and provenance intact."""
    backend = _FakeBackend(scores=[4.0, 3.0, 2.0, 1.0])
    reranker = _reranker(backend)
    candidates = [_hit("A", rank=1), _hit("B", rank=2), _hit("C", rank=3), _hit("D", rank=4)]

    result = reranker.rerank("q", candidates, top_k=2)

    assert [hit.chunk_id for hit in result] == ["A", "B"]
    assert [hit.rank for hit in result] == [1, 2]
    assert [hit.pre_rerank_rank for hit in result] == [1, 2]


def test_top_k_none_uses_config_default() -> None:
    """Omitting top_k applies RerankConfig.top_k."""
    backend = _FakeBackend(scores=[4.0, 3.0, 2.0, 1.0])
    reranker = _reranker(backend, top_k=2)

    result = reranker.rerank("q", [_hit("A"), _hit("B"), _hit("C"), _hit("D")])

    assert len(result) == 2


def test_top_k_larger_than_pool_returns_the_whole_pool() -> None:
    """A larger requested Top-K does not invent candidates or raise."""
    backend = _FakeBackend(scores=[1.0, 0.5, 0.1])
    reranker = _reranker(backend)

    result = reranker.rerank("q", [_hit("A"), _hit("B"), _hit("C")], top_k=20)

    assert len(result) == 3


@pytest.mark.parametrize("top_k", [0, -1, True, "10"])
def test_invalid_top_k_raises_value_error(top_k: object) -> None:
    """Top-K follows the same integer rules as the other retrievers."""
    backend = _FakeBackend(scores=[1.0])
    reranker = _reranker(backend)

    with pytest.raises(ValueError):
        reranker.rerank("q", [_hit("A")], top_k=top_k)  # type: ignore[arg-type]


# -------------------------------------------------------------------- failure modes


def test_inference_failure_is_wrapped_and_not_fallback() -> None:
    """Inference errors become RerankError; no candidate list is returned."""
    backend = _FakeBackend(error=RuntimeError("boom"))
    reranker = _reranker(backend)

    with pytest.raises(RerankError) as raised:
        reranker.rerank("q", [_hit("A"), _hit("B")])

    assert raised.value.stage == "inference"
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "2 pair" in str(raised.value)


def test_inference_memory_failure_is_resource_stage() -> None:
    """A memory-allocation inference error is classified as resource."""
    backend = _FakeBackend(error=RuntimeError("DefaultCPUAllocator: can't allocate memory"))
    reranker = _reranker(backend)

    with pytest.raises(RerankError) as raised:
        reranker.score("q", [_hit("A")])

    assert raised.value.stage == "resource"


def test_non_finite_score_is_rejected() -> None:
    """A NaN or infinity would corrupt ordering, so it is an error."""
    backend = _FakeBackend(scores=[float("nan")])
    reranker = _reranker(backend)

    with pytest.raises(RerankError) as raised:
        reranker.score("q", [_hit("A")])

    assert raised.value.stage == "inference"
    assert "non-finite" in str(raised.value)


def test_score_count_mismatch_is_rejected() -> None:
    """Two scores for three pairs cannot be mapped back to IDs."""
    backend = _FakeBackend(scores=[1.0, 2.0])
    reranker = _reranker(backend)

    with pytest.raises(RerankError) as raised:
        reranker.score("q", [_hit("A"), _hit("B"), _hit("C")])

    assert raised.value.stage == "inference"
    assert "2 scores for 3 pairs" in str(raised.value)


# ----------------------------------------------------------- preservation and input


def test_input_candidates_are_not_mutated() -> None:
    """Reranking returns new hits and leaves the base pool untouched."""
    backend = _FakeBackend(scores=[0.2, 2.0])
    reranker = _reranker(backend)
    candidates = [_hit("A", rank=1), _hit("B", rank=2)]
    before = [hit.model_dump() for hit in candidates]

    result = reranker.rerank("q", candidates, top_k=2)

    assert [hit.model_dump() for hit in candidates] == before
    assert result[0] is not candidates[1]
    assert all(hit.source == "reranked" for hit in result)


def test_metadata_and_fusion_provenance_are_preserved() -> None:
    """A reranked hit keeps every field the later report needs."""
    backend = _FakeBackend(scores=[0.5])
    reranker = _reranker(backend)
    original = _hit("A", rank=7, score=0.031, doc_id="DOC-0042")

    result = reranker.rerank("q", [original], top_k=1)[0]

    assert result.chunk_id == original.chunk_id
    assert result.doc_id == "DOC-0042"
    assert result.chunk_strategy == "document"
    assert result.chunk_index == 0
    assert result.title == original.title
    assert result.content == original.content
    assert result.category == "fees"
    assert result.card_tier == "platinum"
    assert result.effective_year == 2025
    assert result.policy_code == "BT-001"
    assert result.dense_rank == 7
    assert result.sparse_rank == 7
    assert result.dense_score == pytest.approx(0.9)
    assert result.sparse_score == pytest.approx(4.2)
    assert result.rrf_score == pytest.approx(0.031)
    assert result.candidate_sources == ("dense", "sparse")
    assert result.pre_rerank_rank == 7
    assert result.pre_rerank_score == pytest.approx(0.031)
    assert result.distance is None
    assert result.source == "reranked"


def test_dense_sparse_and_fused_hits_are_all_accepted() -> None:
    """Reranking is compatible with every existing arm's SearchHit shape."""
    backend = _FakeBackend(scores=[3.0, 2.0, 1.0])
    reranker = _reranker(backend)
    candidates = [
        _hit("D", source="dense"),
        _hit("S", source="sparse"),
        _hit("F", source="fused"),
    ]

    result = reranker.rerank("q", candidates, top_k=3)

    assert [hit.chunk_id for hit in result] == ["D", "S", "F"]
    assert all(hit.source == "reranked" for hit in result)


# --------------------------------------------------------------- pipeline behaviour


def test_reranked_retriever_requests_candidate_pool_and_truncates() -> None:
    """The base receives candidate_k and the result is the reranked Top-K."""
    hits = [_hit(f"C{index:02d}", rank=index + 1, score=1.0 / (index + 1)) for index in range(50)]
    base = _FixedBase(hits)
    backend = _FakeBackend(scores=[float(100 - index) for index in range(50)])
    reranker = _reranker(backend, candidates=50, top_k=10)
    pipeline = RerankedRetriever(base, reranker)

    result = pipeline.retrieve_with_details("q")

    assert base.calls == [("q", 50)]
    assert len(result.base_hits) == 50
    assert len(result.hits) == 10
    assert result.candidate_k == 50
    assert result.top_k == 10
    assert result.truncated is True
    assert result.empty_candidates is False
    assert result.fallback_used is False
    assert result.base_name == "fixed_base"
    assert result.model_id == "fake-reranker"
    assert result.model_loaded is True
    assert [hit.rank for hit in result.hits] == list(range(1, 11))
    assert [hit.chunk_id for hit in result.hits] == [f"C{index:02d}" for index in range(10)]


def test_reranked_retriever_reuses_one_model_across_queries() -> None:
    """The model is loaded once, not once per query."""
    base = _FixedBase([_hit("A")])
    backend = _FakeBackend(scores=[1.0])
    calls: list[int] = []

    def factory() -> _FakeBackend:
        calls.append(1)
        return backend

    reranker = CrossEncoderReranker(_config(top_k=10, candidates=10), model_factory=factory)
    pipeline = RerankedRetriever(base, reranker)

    pipeline.retrieve("q1")
    pipeline.retrieve("q2")

    assert calls == [1]
    assert backend.predict_calls == 2


def test_reranked_retriever_empty_pool_does_not_load_model() -> None:
    """An empty base result short-circuits before inference."""
    base = _FixedBase([])
    calls: list[int] = []

    def factory() -> Any:
        calls.append(1)
        raise AssertionError("model must not load for an empty pool")

    reranker = CrossEncoderReranker(_config(), model_factory=factory)
    pipeline = RerankedRetriever(base, reranker)

    result = pipeline.retrieve_with_details("q")

    assert result.hits == []
    assert result.base_hits == []
    assert result.empty_candidates is True
    assert result.model_loaded is False
    assert result.fallback_used is False
    assert calls == []
    assert base.calls == [("q", 50)]


def test_reranked_retriever_base_failure_is_base_retrieval() -> None:
    """A base retrieval failure is categorised, not silently swallowed."""
    base = _FixedBase([], error=RetrievalError("database connection failed"))
    backend = _FakeBackend(scores=[1.0])
    reranker = _reranker(backend)
    pipeline = RerankedRetriever(base, reranker)

    with pytest.raises(RerankError) as raised:
        pipeline.retrieve("q")

    assert raised.value.stage == "base_retrieval"
    assert isinstance(raised.value.__cause__, RetrievalError)
    assert "database connection failed" in str(raised.value)
    assert backend.predict_calls == 0


def test_reranked_retriever_has_no_silent_fallback() -> None:
    """A reranker failure raises; the base candidates are never returned."""
    base = _FixedBase([_hit("A"), _hit("B")])
    backend = _FakeBackend(error=RuntimeError("inference exploded"))
    reranker = _reranker(backend)
    pipeline = RerankedRetriever(base, reranker)

    with pytest.raises(RerankError) as raised:
        pipeline.retrieve("q")

    assert raised.value.stage == "inference"
    assert "inference exploded" in str(raised.value)


@pytest.mark.parametrize("top_k", [0, 51])
def test_reranked_retriever_validates_top_k(top_k: int) -> None:
    """A final Top-K outside 1..candidate_k is rejected before retrieval."""
    base = _FixedBase([_hit("A")])
    reranker = _reranker(_FakeBackend(scores=[1.0]), candidates=50, top_k=10)
    pipeline = RerankedRetriever(base, reranker)

    with pytest.raises(ValueError):
        pipeline.retrieve("q", top_k=top_k)

    assert base.calls == []


def test_reranked_retriever_output_is_retriever_compatible() -> None:
    """The pipeline exposes the protocol fields and a ranked hit list."""
    base = _FixedBase([_hit("A", rank=1), _hit("B", rank=2)])
    reranker = _reranker(_FakeBackend(scores=[0.9, 0.1]))
    pipeline = RerankedRetriever(base, reranker)

    hits = pipeline.retrieve("q", top_k=2)

    assert pipeline.name == "reranked"
    assert [hit.source for hit in hits] == ["reranked", "reranked"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert [hit.doc_id for hit in hits] == ["DOC-A", "DOC-B"]


# ------------------------------------------------------------------------ config


def test_config_rejects_top_k_above_candidates() -> None:
    """The declared final Top-K cannot exceed the pool that is scored."""
    with pytest.raises(ValidationError):
        RerankConfig(candidates=10, top_k=11)


@pytest.mark.parametrize("activation", ["softmax", "", "Identity"])
def test_config_rejects_unknown_activation(activation: str) -> None:
    """Only the two documented score transforms are accepted."""
    with pytest.raises(ValidationError):
        RerankConfig(activation=activation)
