"""Arm registry: names, parameters and construction.

These tests never connect to PostgreSQL and never load a model: the dense arm
receives a fixed-vector embedder stub and the rerank arm a reranker stub. Only
construction and declarative metadata are asserted; real retrieval is covered by
the integration benchmarks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from src.retrieval.registry import (
    ARM_NAMES,
    ARMS,
    ArmContext,
    arm_parameters,
    arm_spec,
    build_arm,
)
from src.schema.retrieval import SearchHit


class _StubEmbedder:
    """Minimal embedder returning a fixed-width vector."""

    dimension = 4

    def embed_query(self, text: str) -> list[float]:
        """Return a deterministic vector for any text."""
        del text
        return [0.0, 0.0, 0.0, 1.0]

    def embed_documents(
        self, texts: Sequence[str], on_batch: Any = None
    ) -> list[list[float]]:
        """Return one vector per text."""
        del on_batch
        return [self.embed_query(text) for text in texts]


class _StubReranker:
    """Reranker stub exposing the provenance attributes the pipeline reads."""

    model_id = "stub-reranker"
    revision = "deadbeef"
    device = "cpu"
    batch_size = 8
    max_length = 512
    activation = "identity"

    def rerank(
        self, query: str, hits: Sequence[SearchHit], top_k: int | None = None
    ) -> list[SearchHit]:
        """Return the candidates unchanged; never called by these tests."""
        del query
        return list(hits[:top_k] if top_k is not None else hits)


def _context(**overrides: Any) -> ArmContext:
    """Build a valid context, with optional overrides."""
    values: dict[str, Any] = {
        "embedder": _StubEmbedder(),
        "top_k": 5,
        "candidate_k": 10,
        "rrf_k": 60,
        "chunk_strategy": "document",
    }
    values.update(overrides)
    return ArmContext(**values)


def test_arm_names_match_the_registry() -> None:
    """The public tuple and the mapping cannot drift apart."""
    assert tuple(ARMS) == ARM_NAMES
    assert ARM_NAMES == (
        "dense",
        "sparse_or",
        "sparse_and",
        "union",
        "rrf",
        "rrf_rerank",
    )


def test_unknown_arm_is_rejected_with_the_registered_names() -> None:
    """A typo must not silently select another arm."""
    with pytest.raises(ValueError, match="unknown arm 'dense_v2'"):
        arm_spec("dense_v2")


def test_only_the_rerank_arm_requires_a_reranker() -> None:
    """``needs_reranker`` is declared per arm and enforced at build time."""
    assert [name for name, spec in ARMS.items() if spec.needs_reranker] == [
        "rrf_rerank"
    ]
    with pytest.raises(ValueError, match="requires a reranker"):
        build_arm("rrf_rerank", _context())


def test_every_arm_builds_with_the_expected_stage_name() -> None:
    """Construction needs no database or model download."""
    expected = {
        "dense": "dense",
        "sparse_or": "sparse",
        "sparse_and": "sparse",
        "union": "union",
        "rrf": "hybrid",
        "rrf_rerank": "reranked",
    }
    context = _context(reranker=_StubReranker())
    for name, stage in expected.items():
        retriever = build_arm(name, context)
        assert retriever.name == stage


def test_arm_parameters_record_the_shared_knobs() -> None:
    """Every arm records Top-K and its chunk strategy for the report."""
    context = _context(reranker=_StubReranker(), chunk_strategy="structured")
    for name in ARM_NAMES:
        parameters = arm_parameters(name, context)
        assert parameters["top_k"] == 5
        assert parameters["chunk_strategy"] == "structured"
    assert arm_parameters("rrf", context)["rrf_k"] == 60
    assert arm_parameters("rrf", context)["candidate_k"] == 10
    assert arm_parameters("sparse_or", context)["query_mode"] == "or"
    assert arm_parameters("sparse_and", context)["query_mode"] == "and"


def test_top_k_above_the_candidate_pool_is_rejected() -> None:
    """A final list longer than the fused pool cannot be produced."""
    with pytest.raises(ValueError, match="must not exceed candidate_k"):
        _context(top_k=50, candidate_k=10)


@pytest.mark.parametrize("field", ["top_k", "candidate_k", "rrf_k", "ef_search"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_context_rejects_invalid_counts(field: str, value: object) -> None:
    """Counts must be positive ints; bools are not counts."""
    with pytest.raises(ValueError, match=f"{field} must"):
        _context(**{field: value})


def test_context_rejects_a_retired_chunk_strategy() -> None:
    """``contextual`` is a retired name, not a selectable strategy."""
    with pytest.raises(ValueError, match="chunk_strategy"):
        _context(chunk_strategy="contextual")
