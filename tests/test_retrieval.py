"""Phase 6B: HybridRetriever composition and failure behaviour.

Every test here uses fake retrievers. The real dense and sparse arms already
have their own database/embedding tests; this file isolates the composition
contract: what is requested, what is merged, what is recorded, and what happens
when an arm fails.
"""

from __future__ import annotations

from typing import Literal

import pytest
from src.retrieval import RetrievalError
from src.retrieval.fusion import RRFFusion
from src.retrieval.pipeline import HybridResult, HybridRetriever
from src.schema.retrieval import SearchHit


def _hit(
    chunk_id: str,
    *,
    source: Literal["dense", "sparse"] = "dense",
    rank: int = 1,
    score: float = 1.0,
) -> SearchHit:
    """Build a valid arm hit."""
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=f"DOC-{chunk_id}",
        score=score,
        distance=1.0 - score if source == "dense" else None,
        rank=rank,
        source=source,
        content=f"content-{chunk_id}",
        chunk_strategy="document",
        chunk_index=0,
        title=f"title-{chunk_id}",
        category="fees",
        card_tier="standard",
        effective_year=2025,
        policy_code="P-1",
    )


class _FakeRetriever:
    """A retriever returning fixed hits and recording every call."""

    name = "fake"

    def __init__(
        self,
        hits: list[SearchHit],
        error: BaseException | None = None,
    ) -> None:
        """Store the fixed hits and optional failure."""
        self.hits = hits
        self.error = error
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Record the call, then return hits or raise the configured error."""
        self.calls.append((query, top_k))
        if self.error is not None:
            raise self.error
        return list(self.hits if top_k is None else self.hits[:top_k])


def _hybrid(
    dense_hits: list[SearchHit],
    sparse_hits: list[SearchHit],
    *,
    candidate_k: int = 10,
    default_top_k: int = 10,
    dense_error: BaseException | None = None,
    sparse_error: BaseException | None = None,
) -> tuple[HybridRetriever, _FakeRetriever, _FakeRetriever]:
    """Build a hybrid pipeline over two fakes and return all three objects."""
    dense = _FakeRetriever(dense_hits, dense_error)
    sparse = _FakeRetriever(sparse_hits, sparse_error)
    fusion = RRFFusion(k=60, candidate_k=candidate_k)
    hybrid = HybridRetriever(
        dense=dense,
        sparse=sparse,
        fusion=fusion,
        default_top_k=default_top_k,
    )
    return hybrid, dense, sparse


# --------------------------------------------------------------- composition


def test_hybrid_retriever_satisfies_the_retriever_interface() -> None:
    """The pipeline is a normal stage: a name and retrieve(query, top_k)."""
    hybrid, _, _ = _hybrid(
        [_hit("A")],
        [_hit("B", source="sparse")],
    )
    assert hybrid.name == "hybrid"
    assert callable(hybrid.retrieve)
    hits = hybrid.retrieve("transfer fee", top_k=2)
    assert isinstance(hits, list)
    assert all(isinstance(hit, SearchHit) for hit in hits)


def test_both_arms_are_asked_for_candidate_k_not_final_top_k() -> None:
    """Fusion needs the whole candidate pool, not only the final Top-K."""
    hybrid, dense, sparse = _hybrid(
        [_hit("A", rank=1), _hit("B", rank=2), _hit("C", rank=3)],
        [
            _hit("B", source="sparse", rank=1),
            _hit("C", source="sparse", rank=2),
            _hit("D", source="sparse", rank=3),
        ],
        candidate_k=10,
        default_top_k=2,
    )
    result = hybrid.retrieve_with_details("  transfer fee  ", top_k=2)

    assert dense.calls == [("transfer fee", 10)]
    assert sparse.calls == [("transfer fee", 10)]
    assert result.query == "transfer fee"
    assert result.top_k == 2
    assert result.candidate_k == 10
    assert result.rrf_k == 60
    assert [hit.chunk_id for hit in result.fused_hits] == ["B", "C"]
    assert len(result.dense_hits) == 3
    assert len(result.sparse_hits) == 3


def test_hybrid_result_records_both_candidate_lists_and_provenance() -> None:
    """The report can prove which arm produced every fused candidate."""
    hybrid, _, _ = _hybrid(
        [_hit("A", rank=1, score=0.9), _hit("B", rank=2, score=0.8)],
        [
            _hit("B", source="sparse", rank=1, score=5.0),
            _hit("C", source="sparse", rank=2, score=4.0),
        ],
    )
    result = hybrid.retrieve_with_details("q", top_k=3)

    assert isinstance(result, HybridResult)
    assert result.query == "q"
    assert [hit.chunk_id for hit in result.dense_hits] == ["A", "B"]
    assert [hit.chunk_id for hit in result.sparse_hits] == ["B", "C"]
    assert [hit.chunk_id for hit in result.fused_hits] == ["B", "A", "C"]
    assert result.fused_hits[0].candidate_sources == ("dense", "sparse")
    assert result.fused_hits[0].dense_rank == 2
    assert result.fused_hits[0].sparse_rank == 1
    assert result.dense_ms >= 0.0
    assert result.sparse_ms >= 0.0
    assert result.fusion_ms >= 0.0
    assert result.total_ms >= 0.0


def test_default_top_k_is_used_when_omitted() -> None:
    """A caller can use the pipeline like any Retriever protocol object."""
    hybrid, dense, sparse = _hybrid(
        [_hit("A", rank=1), _hit("B", rank=2), _hit("C", rank=3)],
        [],
        candidate_k=5,
        default_top_k=2,
    )
    assert len(hybrid.retrieve("q")) == 2
    assert dense.calls == [("q", 5)]
    assert sparse.calls == [("q", 5)]
    assert hybrid.default_top_k == 2


def test_one_empty_arm_still_returns_the_other_arms_candidates() -> None:
    """An empty candidate list is a normal outcome, not an arm failure."""
    hybrid, _, _ = _hybrid([], [_hit("S", source="sparse", score=3.0)])
    result = hybrid.retrieve_with_details("q", top_k=5)
    assert [hit.chunk_id for hit in result.fused_hits] == ["S"]
    assert result.fused_hits[0].score == pytest.approx(1 / 61)
    assert result.fused_hits[0].candidate_sources == ("sparse",)


def test_arms_returning_more_than_candidate_k_are_truncated() -> None:
    """The recorded candidate lists are exactly what fusion consumed."""
    dense_hits = [_hit(f"D{index}", rank=index) for index in range(1, 21)]
    sparse_hits = [_hit(f"S{index}", source="sparse", rank=index) for index in range(1, 21)]
    hybrid, _, _ = _hybrid(dense_hits, sparse_hits, candidate_k=3, default_top_k=3)
    result = hybrid.retrieve_with_details("q")

    assert [hit.chunk_id for hit in result.dense_hits] == ["D1", "D2", "D3"]
    assert [hit.chunk_id for hit in result.sparse_hits] == ["S1", "S2", "S3"]
    assert len(result.fused_hits) == 3


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("bad_query", ["", "   ", "\n\t", 123])
def test_invalid_query_is_rejected_before_either_arm_runs(bad_query: object) -> None:
    """Validation happens in the pipeline, not after expensive work."""
    hybrid, dense, sparse = _hybrid([], [])
    with pytest.raises(ValueError, match="query must be"):
        hybrid.retrieve_with_details(bad_query)  # type: ignore[arg-type]
    assert dense.calls == []
    assert sparse.calls == []


@pytest.mark.parametrize("bad_top_k", [0, -1, True, 1.5, 11, 1001])
def test_invalid_top_k_is_rejected_before_either_arm_runs(bad_top_k: object) -> None:
    """top_k must be a positive int within the candidate pool and global cap."""
    hybrid, dense, sparse = _hybrid([], [], candidate_k=10)
    with pytest.raises(ValueError, match="top_k"):
        hybrid.retrieve_with_details("q", top_k=bad_top_k)  # type: ignore[arg-type]
    assert dense.calls == []
    assert sparse.calls == []


@pytest.mark.parametrize("bad_default", [0, -1, True, 11])
def test_invalid_default_top_k_is_rejected_at_construction(bad_default: object) -> None:
    """The configured default is validated once, against candidate_k."""
    dense = _FakeRetriever([])
    sparse = _FakeRetriever([])
    fusion = RRFFusion(candidate_k=10)
    with pytest.raises(ValueError, match="top_k"):
        HybridRetriever(  # type: ignore[arg-type]
            dense=dense,
            sparse=sparse,
            fusion=fusion,
            default_top_k=bad_default,
        )


# ------------------------------------------------------------ failure paths


def test_dense_failure_is_wrapped_with_the_stage_name_and_stops_the_run() -> None:
    """A hybrid must not silently fall back to sparse-only."""
    hybrid, dense, sparse = _hybrid(
        [_hit("A")],
        [_hit("B", source="sparse")],
        dense_error=RetrievalError("embedding backend down"),
    )
    with pytest.raises(RetrievalError, match="hybrid dense stage failed: embedding"):
        hybrid.retrieve_with_details("q")
    assert len(dense.calls) == 1
    assert sparse.calls == []


def test_sparse_failure_is_wrapped_with_the_stage_name() -> None:
    """Dense already ran; the sparse error still reaches the caller."""
    hybrid, dense, sparse = _hybrid(
        [_hit("A")],
        [_hit("B", source="sparse")],
        sparse_error=RetrievalError("database timeout"),
    )
    with pytest.raises(RetrievalError, match="hybrid sparse stage failed: database timeout"):
        hybrid.retrieve_with_details("q")
    assert len(dense.calls) == 1
    assert len(sparse.calls) == 1


def test_non_retrieval_errors_are_not_hidden() -> None:
    """A programming error is not relabelled as a retrieval failure."""
    hybrid, _, _ = _hybrid(
        [_hit("A")],
        [],
        dense_error=RuntimeError("bug"),
    )
    with pytest.raises(RuntimeError, match="bug"):
        hybrid.retrieve_with_details("q")


def test_duplicate_candidates_from_one_arm_are_rejected() -> None:
    """Fusion's arm contract is enforced through the pipeline."""
    hybrid, _, _ = _hybrid(
        [_hit("A", rank=1), _hit("A", rank=2)],
        [],
    )
    with pytest.raises(ValueError, match="duplicate chunk identity"):
        hybrid.retrieve_with_details("q")
