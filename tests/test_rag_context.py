"""Phase 8: bounded evidence context assembly."""

from __future__ import annotations

import hashlib

import pytest
from src.rag.context import ContextError, build_context
from src.schema.retrieval import SearchHit


def _hit(
    chunk_id: str,
    *,
    doc_id: str | None = None,
    content: str | None = None,
    strategy: str = "document",
    rank: int = 1,
) -> SearchHit:
    """Build a database-shaped hit for context tests."""
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=doc_id or f"DOC-{chunk_id}",
        score=0.9,
        distance=0.1,
        rank=rank,
        source="dense",
        content=f"content for {chunk_id}" if content is None else content,
        chunk_strategy=strategy,
        chunk_index=0,
        title=f"Title {chunk_id}",
        category="fees",
        card_tier="standard",
        effective_year=2025,
        policy_code="P-1",
    )


def test_labels_are_sequential_and_match_rank_order() -> None:
    """Evidence labels must be stable E1..En in the order given to the model."""
    bundle = build_context([_hit("c1", rank=1), _hit("c2", rank=2), _hit("c3", rank=3)])

    assert bundle.labels == ["E1", "E2", "E3"]
    assert [item.chunk_id for item in bundle.evidence] == ["c1", "c2", "c3"]
    assert [item.retrieval_rank for item in bundle.evidence] == [1, 2, 3]
    assert bundle.text.count("[E") == 3
    assert "content for c2" in bundle.text
    assert bundle.dropped == 0
    assert not bundle.truncated


def test_context_uses_database_text_and_records_hash() -> None:
    """The prompt gets the real content and a hash of exactly that text."""
    bundle = build_context([_hit("c1", content="The annual fee is $95.")])

    assert "The annual fee is $95." in bundle.text
    assert bundle.sha256 == hashlib.sha256(bundle.text.encode("utf-8")).hexdigest()


def test_duplicate_identity_is_included_once() -> None:
    """The same chunk strategy/id pair must not consume two evidence labels."""
    bundle = build_context([_hit("c1"), _hit("c1"), _hit("c2", rank=2)])

    assert bundle.labels == ["E1", "E2"]
    assert [item.chunk_id for item in bundle.evidence] == ["c1", "c2"]


def test_per_chunk_truncation_is_explicit() -> None:
    """An over-long chunk is cut to the per-chunk cap and flagged."""
    bundle = build_context([_hit("c1", content="x" * 500)], max_chunk_chars=100)

    evidence = bundle.evidence[0]
    assert evidence.original_chars == 500
    assert evidence.included_chars == 100
    assert evidence.truncated is True
    assert bundle.truncated is True
    assert len(evidence.context_block) < 300


def test_total_budget_drops_later_candidates() -> None:
    """A hard total budget must stop adding blocks, not silently overflow."""
    first = _hit("c1", content="a" * 100)
    second = _hit("c2", content="b" * 400, rank=2)
    bundle = build_context([first, second], max_context_chars=250, max_chunk_chars=400)

    assert [item.chunk_id for item in bundle.evidence] == ["c1"]
    assert bundle.dropped == 1
    assert bundle.truncated is True
    assert len(bundle.text) <= 250


def test_first_block_is_truncated_to_fit_the_budget() -> None:
    """With room for one partial block, keep the head of the chunk and flag it."""
    bundle = build_context(
        [_hit("c1", content="z" * 1000)],
        max_context_chars=300,
        max_chunk_chars=1000,
    )

    evidence = bundle.evidence[0]
    assert evidence.truncated is True
    assert bundle.truncated is True
    assert len(bundle.text) <= 300
    assert bundle.text.endswith("z")


def test_missing_content_is_an_error_not_an_empty_passage() -> None:
    """Blank database content must fail evidence assembly."""
    with pytest.raises(ContextError, match="no content"):
        build_context([_hit("c1", content="   ")])


def test_budget_too_small_for_one_header_is_an_error() -> None:
    """A configured budget that cannot hold a header must not emit a label-only block."""
    with pytest.raises(ContextError, match="too small"):
        build_context([_hit("c" * 400)], max_context_chars=200, max_chunk_chars=1)


def test_invalid_budgets_are_rejected() -> None:
    """Bad limits fail before any evidence is constructed."""
    with pytest.raises(ContextError):
        build_context([], max_context_chars=199)
    with pytest.raises(ContextError):
        build_context([], max_chunk_chars=0)
    with pytest.raises(ContextError):
        build_context([], max_context_chars=True)  # type: ignore[arg-type]


def test_empty_hits_return_empty_bundle() -> None:
    """No hits is a valid empty context; the pipeline decides what it means."""
    bundle = build_context([])

    assert bundle.evidence == []
    assert bundle.text == ""
    assert bundle.dropped == 0
    assert bundle.sha256 == hashlib.sha256(b"").hexdigest()
