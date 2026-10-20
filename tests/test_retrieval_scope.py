"""Chunk-strategy isolation: every retrieval names the strategy it may see.

The ``chunks`` table's primary key is ``(chunk_strategy, chunk_id)``, so one
document may have several chunkings side by side. Without a filter, a Top-K can
mix strategies and every metric computed from it is meaningless. These tests
are pure unit tests: a recording connection proves the strategy is bound as a
SQL parameter, and ``_rows_to_hits`` proves a mismatched row fails loudly.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import pytest
from src.retrieval import RetrievalError
from src.retrieval.dense import DENSE_SEARCH_SQL, DenseRetriever
from src.retrieval.dense import _rows_to_hits as _dense_hits
from src.retrieval.sparse import (
    SPARSE_CJK_SQL,
    SPARSE_SEARCH_SQL,
    SparseRetriever,
)
from src.retrieval.sparse import (
    _rows_to_hits as _sparse_hits,
)
from src.schema.retrieval import DenseConfig, SparseConfig

RETRIEVABLE = ("document", "fixed_window", "structured")


# ------------------------------------------------------------------ test doubles


class _StubEmbedder:
    """A shape-valid 1024-dim embedder; never loads a model."""

    dimension = 1024

    def embed_documents(
        self,
        texts: Sequence[str],
        on_batch: Any = None,
    ) -> list[list[float]]:
        del on_batch
        return [[1.0] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        del text
        return [1.0] * self.dimension


class _FakeResult:
    """The narrow result surface psycopg queries use here."""

    def __init__(self, rows: Sequence[Sequence[Any]]) -> None:
        self._rows = list(rows)

    def fetchall(self) -> list[Sequence[Any]]:
        return list(self._rows)

    def fetchone(self) -> Sequence[Any] | None:
        return self._rows[0] if self._rows else None


class _RecordingConnection:
    """Records ``(sql, params)`` and returns canned rows.

    ``current_setting('hnsw.ef_search', true)`` is answered separately because
    the dense retriever reads back the live GUC inside the same transaction.
    """

    def __init__(self, rows: Sequence[Sequence[Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, Any]] = []

    @contextmanager
    def transaction(self) -> Iterator[_RecordingConnection]:
        yield self

    def execute(
        self,
        sql: str,
        params: Any = None,
        **kwargs: Any,
    ) -> _FakeResult:
        del kwargs
        self.calls.append((sql, params))
        if "current_setting" in sql:
            return _FakeResult([("40",)])
        return _FakeResult(self.rows)

    def retrieval_call(self) -> tuple[str, Any]:
        """The one call that ran the retrieval statement.

        ``set_config`` also binds parameters, so the match is narrowed to the
        statement that actually reads ``chunks``.
        """
        matches = [
            call for call in self.calls if call[1] is not None and "chunks" in call[0]
        ]
        assert len(matches) == 1, self.calls
        return matches[0]


def _dense_row(*, strategy: str = "document", distance: float = 0.1) -> tuple[Any, ...]:
    return (
        "DOC-001#0",
        "DOC-001",
        strategy,
        0,
        "a chunk",
        distance,
        "A title",
        "fees",
        "Standard",
        2025,
        "PC-1",
    )


def _sparse_row(*, strategy: str = "document", score: float = 0.4) -> tuple[Any, ...]:
    return (
        "DOC-001#0",
        "DOC-001",
        strategy,
        0,
        "a chunk",
        score,
        "A title",
        "fees",
        "Standard",
        2025,
        "PC-1",
    )


def _dense_retriever(strategy: str, conn: _RecordingConnection) -> DenseRetriever:
    return DenseRetriever(
        DenseConfig(top_n=5, ef_search=40, chunk_strategy=strategy),  # type: ignore[arg-type]
        embedder=_StubEmbedder(),
        connection_factory=lambda: _context(conn),
    )


def _sparse_retriever(strategy: str, conn: _RecordingConnection) -> SparseRetriever:
    return SparseRetriever(
        SparseConfig(top_n=5, chunk_strategy=strategy),  # type: ignore[arg-type]
        connection_factory=lambda: _context(conn),
    )


@contextmanager
def _context(conn: _RecordingConnection) -> Iterator[_RecordingConnection]:
    yield conn


# ------------------------------------------------------------------- config scope


def test_defaults_are_document_and_all_retrievable_strategies_validate() -> None:
    """The default keeps every existing report valid; each strategy is accepted."""
    assert DenseConfig().chunk_strategy == "document"
    assert SparseConfig().chunk_strategy == "document"
    for strategy in RETRIEVABLE:
        dense = DenseConfig(chunk_strategy=strategy)  # type: ignore[arg-type]
        sparse = SparseConfig(chunk_strategy=strategy)  # type: ignore[arg-type]
        assert dense.chunk_strategy == strategy
        assert sparse.chunk_strategy == strategy


@pytest.mark.parametrize("strategy", ["contextual", "", "semantic", "DOCUMENT"])
def test_non_retrievable_strategies_are_rejected(strategy: str) -> None:
    """``contextual`` is a retired config name, not an algorithm to search."""
    with pytest.raises(ValueError):
        DenseConfig(chunk_strategy=strategy)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SparseConfig(chunk_strategy=strategy)  # type: ignore[arg-type]


# ------------------------------------------------------------------ SQL binding


def test_dense_sql_filters_on_the_configured_strategy() -> None:
    """The strategy is a bound parameter, never interpolated into SQL."""
    assert "WHERE chunk_strategy = %s" in DENSE_SEARCH_SQL
    conn = _RecordingConnection([_dense_row(strategy="fixed_window")])
    hits = _dense_retriever("fixed_window", conn).retrieve("a query", top_k=5)
    sql, params = conn.retrieval_call()
    assert sql == DENSE_SEARCH_SQL
    assert params[1] == "fixed_window"
    assert params[2] == 5
    assert hits[0].chunk_strategy == "fixed_window"


def test_sparse_fts_sql_filters_on_the_configured_strategy() -> None:
    """The English path binds the strategy in position 3 (0-based)."""
    assert "AND c.chunk_strategy = %s" in SPARSE_SEARCH_SQL
    conn = _RecordingConnection([_sparse_row(strategy="structured")])
    hits = _sparse_retriever("structured", conn).retrieve("balance transfer fee", top_k=5)
    sql, params = conn.retrieval_call()
    assert sql == SPARSE_SEARCH_SQL
    assert params[3] == "structured"
    assert params[4] == 5
    assert hits[0].chunk_strategy == "structured"


def test_sparse_cjk_sql_filters_on_the_configured_strategy() -> None:
    """The CJK fallback has the same scope as the English path."""
    assert "AND c.chunk_strategy = %s" in SPARSE_CJK_SQL
    conn = _RecordingConnection([_sparse_row(strategy="fixed_window")])
    hits = _sparse_retriever("fixed_window", conn).retrieve("余额转账", top_k=5)
    sql, params = conn.retrieval_call()
    assert sql == SPARSE_CJK_SQL
    assert params[2] == "fixed_window"
    assert params[3] == 5
    assert hits[0].chunk_strategy == "fixed_window"


# ------------------------------------------------------- defence-in-depth checks


def test_dense_rows_from_another_strategy_are_rejected() -> None:
    """A wrong query or test double must fail, not silently mix strategies."""
    with pytest.raises(RetrievalError, match="expected 'document'"):
        _dense_hits([_dense_row(strategy="fixed_window")], expected_strategy="document")


def test_sparse_rows_from_another_strategy_are_rejected() -> None:
    """The sparse mapper applies the same check."""
    with pytest.raises(RetrievalError, match="expected 'document'"):
        _sparse_hits([_sparse_row(strategy="fixed_window")], expected_strategy="document")


def test_mappers_stay_backwards_compatible_when_scope_is_omitted() -> None:
    """Existing unit tests that pass rows without a scope keep working."""
    assert _dense_hits([_dense_row(strategy="fixed_window")])[0].chunk_strategy == "fixed_window"
    assert _sparse_hits([_sparse_row(strategy="structured")])[0].chunk_strategy == "structured"
