"""Index-management guards: names, bounds and shadow naming.

Pure unit tests only. The build functions themselves need PostgreSQL and a real
``CREATE INDEX``; here a fake connection exercises the naming/validation logic
that runs before any DDL.
"""

from __future__ import annotations

from typing import Any

import pytest
from src.ingestion.indexer import (
    BOOTSTRAP_INDEX,
    HNSW_INDEX_PREFIX,
    IVFFLAT_INDEX_PREFIX,
    IngestionError,
    _bounded_int,
    _parse_options,
    _resolve_index_name,
    _validate_identifier,
    collect_index_stats,
    drop_index,
)


class _FakeResult:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)


class _NameConnection:
    """Answers the shadow-name query with a fixed set of existing indexes."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    def execute(self, sql: str, params: Any = None) -> _FakeResult:
        del sql, params
        return _FakeResult([(name,) for name in self._names])


# ------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "name",
    ["chunks_embedding_hnsw_v001", "_private", "a", "a" * 63],
)
def test_safe_identifiers_are_accepted(name: str) -> None:
    """Accept names made of lowercase letters, digits and underscores."""
    assert _validate_identifier(name) == name


@pytest.mark.parametrize(
    "name",
    ["Bad-Name", "index; DROP TABLE chunks", "", "1starts_with_digit", "a" * 64, "Quoted"],
)
def test_unsafe_identifiers_are_rejected(name: str) -> None:
    """Reject anything that is not a safe SQL identifier."""
    with pytest.raises(IngestionError):
        _validate_identifier(name)


def test_bootstrap_index_is_protected_from_drop() -> None:
    """The guard runs before any connection is borrowed."""
    with pytest.raises(IngestionError, match="refusing to drop"):
        drop_index(BOOTSTRAP_INDEX)


def test_bad_names_fail_before_the_database_is_touched() -> None:
    """Validate the name before borrowing a connection."""
    with pytest.raises(IngestionError, match="invalid index name"):
        drop_index("bad name")
    with pytest.raises(IngestionError, match="invalid index name"):
        collect_index_stats("bad name")


@pytest.mark.parametrize("value", [True, False, 1.5, "16", None])
def test_bounded_int_rejects_non_ints(value: object) -> None:
    """Refuse bools, floats, strings and None."""
    with pytest.raises(IngestionError):
        _bounded_int(value, 2, 100, "m")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [1, 101, 0, -5])
def test_bounded_int_rejects_out_of_range(value: int) -> None:
    """Refuse values outside the inclusive range."""
    with pytest.raises(IngestionError):
        _bounded_int(value, 2, 100, "m")


def test_bounded_int_accepts_the_range_edges() -> None:
    """Keep both endpoints of the range."""
    assert _bounded_int(2, 2, 100, "m") == 2
    assert _bounded_int(100, 2, 100, "m") == 100


# ----------------------------------------------------------------- shadow names


def test_first_shadow_name_starts_at_001() -> None:
    """Allocate v001 when no shadow index exists."""
    conn = _NameConnection([])
    assert _resolve_index_name(conn, "hnsw", name=None, allow_production=False) == (
        f"{HNSW_INDEX_PREFIX}001"
    )
    assert _resolve_index_name(conn, "ivfflat", name=None, allow_production=False) == (
        f"{IVFFLAT_INDEX_PREFIX}001"
    )


def test_next_shadow_name_continues_after_the_highest_version() -> None:
    """Ignore unparsable names and continue after the maximum."""
    conn = _NameConnection(
        [
            f"{HNSW_INDEX_PREFIX}001",
            f"{HNSW_INDEX_PREFIX}007",
            "chunks_embedding_hnsw_idx",
            "chunks_embedding_hnsw_vnotanumber",
        ]
    )
    assert _resolve_index_name(conn, "hnsw", name=None, allow_production=False) == (
        f"{HNSW_INDEX_PREFIX}008"
    )


def test_explicit_name_is_returned_and_the_bootstrap_needs_opt_in() -> None:
    """Require allow_production for the served index name."""
    conn = _NameConnection([])
    assert _resolve_index_name(
        conn, "hnsw", name="my_hnsw", allow_production=False
    ) == "my_hnsw"
    with pytest.raises(IngestionError, match="served index"):
        _resolve_index_name(
            conn, "hnsw", name=BOOTSTRAP_INDEX, allow_production=False
        )
    assert (
        _resolve_index_name(
            conn, "hnsw", name=BOOTSTRAP_INDEX, allow_production=True
        )
        == BOOTSTRAP_INDEX
    )


# ---------------------------------------------------------------- reloptions


def test_reloptions_are_parsed_into_ints() -> None:
    """Parse pgvector reloptions and skip malformed pairs."""
    assert _parse_options("m=16,ef_construction=64") == {"m": 16, "ef_construction": 64}
    assert _parse_options("lists=100") == {"lists": 100}
    assert _parse_options("") == {}
    assert _parse_options("m=notanint,lists=10") == {"lists": 10}
