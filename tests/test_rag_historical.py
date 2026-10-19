"""Phase 8: historical RRF replay joined to live database text."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from src.rag.historical import HistoricalRRFRetriever
from src.retrieval import RetrievalError

QUERY = "What is the annual fee?"


def _artifact(tmp_path: Path, *, rows: list[dict[str, Any]] | None = None) -> Path:
    """Write a tiny accepted-artifact-shaped file."""
    hits = rows or [
        {
            "chunk_id": "c1",
            "doc_id": "DOC-1",
            "chunk_strategy": "document",
            "chunk_index": 0,
            "score": 0.032,
            "distance": None,
            "rank": 1,
            "source": "rrf",
            "dense_rank": 1,
            "sparse_rank": 2,
            "dense_score": 0.8,
            "sparse_score": 0.3,
            "rrf_score": 0.032,
            "candidate_sources": ["dense", "sparse"],
        },
        {
            "chunk_id": "c2",
            "doc_id": "DOC-2",
            "chunk_strategy": "document",
            "chunk_index": 0,
            "score": 0.016,
            "distance": None,
            "rank": 2,
            "source": "rrf",
            "dense_rank": 2,
            "sparse_rank": None,
            "dense_score": 0.7,
            "sparse_score": None,
            "rrf_score": 0.016,
            "candidate_sources": ["dense"],
        },
    ]
    path = tmp_path / "hybrid_benchmark.json"
    path.write_text(
        json.dumps(
            {
                "per_query": [
                    {
                        "query_id": "Q-001",
                        "query": QUERY,
                        "arms": {"rrf": {"hits": hits}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


class _FakeConnection:
    """A connection returning fixed chunk rows and counting calls."""

    def __init__(self, rows: list[tuple[Any, ...]], calls: list[list[Any]]) -> None:
        """Store rows and a shared call counter."""
        self._rows = rows
        self._calls = calls

    def execute(self, sql: str, parameters: list[Any]) -> _FakeConnection:
        """Record the query parameters."""
        del sql
        self._calls.append(parameters)
        return self

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Return the fixed rows."""
        return self._rows


def _factory(rows: list[tuple[Any, ...]], calls: list[list[Any]]) -> Any:
    """Build a connection factory usable as a context manager."""
    @contextmanager
    def _connect() -> Iterator[_FakeConnection]:
        yield _FakeConnection(rows, calls)

    return _connect


def _row(
    chunk_id: str,
    doc_id: str,
    *,
    content: str | None = None,
    strategy: str = "document",
) -> tuple[Any, ...]:
    """Build one chunks-table row."""
    return (
        strategy,
        chunk_id,
        doc_id,
        0,
        content or f"Live text for {chunk_id}.",
        f"Title {chunk_id}",
        "fees",
        "standard",
        2025,
        "P-1",
    )


def test_historical_pool_is_joined_to_live_text_in_rrf_order(tmp_path: Path) -> None:
    """Candidate order and scores come from the artifact; text comes from the DB."""
    calls: list[list[Any]] = []
    retriever = HistoricalRRFRetriever(
        _artifact(tmp_path),
        candidates=2,
        connection_factory=_factory([_row("c2", "DOC-2"), _row("c1", "DOC-1")], calls),
    )
    hits = retriever.retrieve(QUERY, top_k=2)

    assert [hit.chunk_id for hit in hits] == ["c1", "c2"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert [hit.rrf_score for hit in hits] == [0.032, 0.016]
    assert hits[0].content == "Live text for c1."
    assert hits[0].doc_id == "DOC-1"
    assert hits[0].title == "Title c1"
    assert hits[0].source == "fused"
    assert len(calls) == 1


def test_second_call_uses_the_cached_pool(tmp_path: Path) -> None:
    """The database is read once per query, not once per answer."""
    calls: list[list[Any]] = []
    retriever = HistoricalRRFRetriever(
        _artifact(tmp_path),
        candidates=2,
        connection_factory=_factory(
            [_row("c1", "DOC-1"), _row("c2", "DOC-2")], calls
        ),
    )
    retriever.retrieve(QUERY)
    retriever.retrieve(QUERY)
    assert len(calls) == 1


def test_candidate_limit_truncates_the_replayed_pool(tmp_path: Path) -> None:
    """The configured candidate count bounds what reaches the reranker."""
    retriever = HistoricalRRFRetriever(
        _artifact(tmp_path),
        candidates=1,
        connection_factory=_factory(
            [_row("c1", "DOC-1"), _row("c2", "DOC-2")], []
        ),
    )
    hits = retriever.retrieve(QUERY, top_k=10)
    assert [hit.chunk_id for hit in hits] == ["c1"]


def test_unknown_query_is_a_retrieval_error(tmp_path: Path) -> None:
    """Historical mode only covers its recorded queries."""
    retriever = HistoricalRRFRetriever(
        _artifact(tmp_path), connection_factory=_factory([], [])
    )
    with pytest.raises(RetrievalError, match="not present"):
        retriever.retrieve("some other question")


def test_missing_database_text_is_a_retrieval_error(tmp_path: Path) -> None:
    """A historical chunk absent from the live database must not be invented."""
    retriever = HistoricalRRFRetriever(
        _artifact(tmp_path),
        connection_factory=_factory([_row("c1", "DOC-1")], []),
    )
    with pytest.raises(RetrievalError, match="no text"):
        retriever.retrieve(QUERY)


def test_doc_id_mismatch_is_a_retrieval_error(tmp_path: Path) -> None:
    """A changed identity mapping is surfaced, not silently accepted."""
    retriever = HistoricalRRFRetriever(
        _artifact(tmp_path),
        connection_factory=_factory(
            [_row("c1", "DOC-9"), _row("c2", "DOC-2")], []
        ),
    )
    with pytest.raises(RetrievalError, match="doc_id"):
        retriever.retrieve(QUERY)


def test_artifact_without_query_text_is_rejected(tmp_path: Path) -> None:
    """Malformed artifacts fail at construction."""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"per_query": [{"query_id": "Q-1"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="query text"):
        HistoricalRRFRetriever(path, connection_factory=_factory([], []))


def test_duplicate_query_text_is_rejected(tmp_path: Path) -> None:
    """Ambiguous query lookup would make the replay non-deterministic."""
    path = tmp_path / "duplicate.json"
    record = {
        "query": QUERY,
        "arms": {"rrf": {"hits": [{"chunk_id": "c1"}]}},
    }
    path.write_text(
        json.dumps({"per_query": [record, dict(record)]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        HistoricalRRFRetriever(path, connection_factory=_factory([], []))


def test_default_artifact_is_readable_without_database_access() -> None:
    """The accepted artifact still contains the 30 replayable queries."""
    from src.rag.historical import DEFAULT_HISTORICAL

    retriever = HistoricalRRFRetriever(
        DEFAULT_HISTORICAL, connection_factory=_factory([], [])
    )
    assert len(retriever._records) == 30  # noqa: SLF001 -- structural check of the artifact
    assert all("query" in record for record in retriever._records)
