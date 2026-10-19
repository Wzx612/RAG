"""Phase 2: database infrastructure.

These tests talk to a live PostgreSQL. When none is reachable they *skip* rather
than fail, so `pytest` stays useful on a machine that has not run `make up` yet
-- but the skip message is loud enough to notice. The ``conn`` fixture is shared
and lives in ``conftest.py``.

Scope is the infrastructure only: the schema, the pool, and the per-query
setting scoping that Phase 4's parameter sweep will lean on. Retrieval, fusion,
reranking and evaluation have their own test modules, still to be written.
"""

from __future__ import annotations

import psycopg
import pytest
from psycopg_pool import ConnectionPool
from src.config import Settings
from src.db import (
    EF_SEARCH_DEFAULT,
    EF_SEARCH_MAX,
    EF_SEARCH_MIN,
    EF_SEARCH_SETTING,
    connection,
    current_ef_search,
    ef_search,
    get_pool,
    validate_ef_search,
)

EMBEDDING_DIM = 1024
"""The dimension Phase 1 committed to: BAAI/bge-large-en-v1.5.

Pinned here on purpose, so changing the default in config.py fails a test
rather than silently reinterpreting an existing database.
"""

# --------------------------------------------------------------------- checks


def test_connection_is_alive(conn: psycopg.Connection) -> None:
    """SELECT 1 is the smallest proof that a usable connection exists."""
    assert conn.execute("SELECT 1").fetchone() == (1,)


def test_pgvector_extension_is_installed(conn: psycopg.Connection) -> None:
    """Without the extension every vector column below would be a lie."""
    rows = conn.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'").fetchall()
    assert rows == [("vector",)]


def test_documents_table_exists(conn: psycopg.Connection) -> None:
    """The canonical document store, one row per entry in kb_documents.json."""
    assert conn.execute("SELECT to_regclass('public.documents')").fetchone() == ("documents",)


def test_chunks_table_exists(conn: psycopg.Connection) -> None:
    """Chunks are where embeddings and the search vector live."""
    assert conn.execute("SELECT to_regclass('public.chunks')").fetchone() == ("chunks",)


def test_embedding_column_is_a_vector_of_the_configured_dimension(
    conn: psycopg.Connection,
) -> None:
    """A dimension mismatch fails silently at insert time; catch it here."""
    row = conn.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = 'chunks'
          AND a.attname = 'embedding'
          AND a.attnum > 0
          AND NOT a.attisdropped
        """
    ).fetchone()
    assert row == (f"vector({EMBEDDING_DIM})",)
    assert Settings().embedding_dim == EMBEDDING_DIM


def test_tsv_column_is_generated(conn: psycopg.Connection) -> None:
    """The weighted tsvector must be maintained by the database, not by code."""
    row = conn.execute(
        """
        SELECT a.attgenerated
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = 'chunks' AND a.attname = 'tsv' AND a.attnum > 0
        """
    ).fetchone()
    assert row == ("s",)


def test_hnsw_index_exists_with_cosine_operator_class(conn: psycopg.Connection) -> None:
    """HNSW on the wrong opclass silently returns wrong neighbours."""
    rows = conn.execute(
        """
        SELECT i.relname, pg_get_indexdef(x.indexrelid)
        FROM pg_index x
        JOIN pg_class i ON i.oid = x.indexrelid
        JOIN pg_class t ON t.oid = x.indrelid
        JOIN pg_am am  ON am.oid = i.relam
        WHERE t.relname = 'chunks' AND am.amname = 'hnsw'
        """
    ).fetchall()
    assert len(rows) == 1, f"expected exactly one HNSW index, found {len(rows)}"
    name, ddl = rows[0]
    assert name == "chunks_embedding_hnsw_idx"
    assert "vector_cosine_ops" in ddl


def test_gin_index_exists_on_tsv(conn: psycopg.Connection) -> None:
    """Sparse retrieval in Phase 5 depends on this index being present."""
    rows = conn.execute(
        """
        SELECT i.relname
        FROM pg_index x
        JOIN pg_class i ON i.oid = x.indexrelid
        JOIN pg_class t ON t.oid = x.indrelid
        JOIN pg_am am  ON am.oid = i.relam
        WHERE t.relname = 'chunks' AND am.amname = 'gin'
        """
    ).fetchall()
    assert ("chunks_tsv_gin",) in rows


def test_every_document_has_at_least_one_chunk(conn: psycopg.Connection) -> None:
    """Chunks and documents stay in step, whichever stage the database is at.

    Holds when the corpus is empty and when it is fully imported, and fails on
    the state actually worth catching -- an import that stopped half way.
    """
    documents = conn.execute("SELECT count(*) FROM documents").fetchone()
    chunked = conn.execute("SELECT count(DISTINCT doc_id) FROM chunks").fetchone()
    assert chunked == documents


def test_no_zero_vectors_are_stored(conn: psycopg.Connection) -> None:
    """Guards the rule that no fake embedding is ever inserted.

    A zero vector satisfies every NOT NULL and every dimension check while
    encoding nothing, so it is the shape a placeholder would take. ``<#>`` is
    the negative inner product, so ``x <#> x = 0`` exactly when x is all zeros.
    """
    zeros = conn.execute("SELECT count(*) FROM chunks WHERE embedding <#> embedding = 0").fetchone()
    assert zeros == (0,)


# --------------------------------------------------------- configuration only


def test_database_url_overrides_the_individual_fields() -> None:
    """CI and reviewers can point the app at a database with one variable."""
    settings = Settings(database_url="postgresql://u:p@example.test:6000/db")
    assert settings.dsn == "postgresql://u:p@example.test:6000/db"


def test_dsn_is_built_from_parts_when_database_url_is_absent() -> None:
    """The default path: components assembled into a libpq string."""
    settings = Settings(database_url=None, pg_host="h", pg_port=1234, pg_database="d")
    assert "host=h" in settings.dsn
    assert "port=1234" in settings.dsn
    assert "dbname=d" in settings.dsn


# --------------------------------------------- hnsw.ef_search scoping (Phase 4)
#
# Phase 4 sweeps ef_search against a fixed index, so the value has to be both
# settable and *contained*: settable with a bound parameter rather than string
# interpolation, and contained so one sweep point cannot colour the next.


def _backend_pid(conn: psycopg.Connection) -> int:
    """Identify the physical server backend behind a borrowed connection."""
    return conn.execute("SELECT pg_backend_pid()").fetchone()[0]


def test_a_query_plan_uses_the_hnsw_index(conn: psycopg.Connection) -> None:
    """ef_search only means anything if the index is what answers the query.

    At 150 rows the planner picks a sequential scan on cost, and a sequential
    scan ignores ef_search entirely -- so a sweep would measure noise. This
    pins the shape of the query Phase 4 will run, with seqscan disabled the way
    a benchmark must disable it for the same reason.
    """
    if conn.execute("SELECT count(*) FROM chunks").fetchone() == (0,):
        pytest.skip("corpus not imported -- run `make ingest` first")

    with conn.transaction():
        conn.execute("SET LOCAL enable_seqscan = off")
        plan = "\n".join(
            row[0]
            for row in conn.execute(
                """
                EXPLAIN SELECT chunk_id
                FROM chunks
                ORDER BY embedding <=> (SELECT embedding FROM chunks LIMIT 1)
                LIMIT 5
                """
            ).fetchall()
        )
    assert "chunks_embedding_hnsw_idx" in plan, plan


def test_a_real_vector_query_returns_the_expected_neighbours(
    conn: psycopg.Connection,
) -> None:
    """A genuine pgvector query, run at a swept ef_search, inside one scope.

    DOC-0001 is the balance-transfer rule for Apex Platinum; its nearest
    neighbours are the same rule written for the other three tiers, which makes
    them a stable expectation for any correct cosine search.
    """
    if conn.execute("SELECT count(*) FROM chunks").fetchone() == (0,):
        pytest.skip("corpus not imported -- run `make ingest` first")

    with ef_search(conn, 40):
        # The read-back is the point: it proves the query below and the setting
        # above are in the same scope, not merely set at some earlier moment.
        assert current_ef_search(conn) == 40
        rows = conn.execute(
            """
            SELECT c.doc_id
            FROM chunks c, (SELECT embedding FROM chunks WHERE chunk_id = 'DOC-0001') p
            WHERE c.chunk_id <> 'DOC-0001'
            ORDER BY c.embedding <=> p.embedding
            LIMIT 3
            """
        ).fetchall()

    assert {row[0] for row in rows} == {"DOC-0002", "DOC-0003", "DOC-0004"}

    # ...and the setting is gone again once the block ends.
    assert current_ef_search(conn) == EF_SEARCH_DEFAULT


def test_set_takes_a_literal_so_it_cannot_be_parameterised(
    conn: psycopg.Connection,
) -> None:
    """The trap that makes set_config() the only safe spelling.

    psycopg rewrites ``%s`` to a server-side ``$1``, and PostgreSQL's SET does
    not accept parameters -- so the naive version cannot work at all, and the
    only "fix" that appears to is string interpolation. This test exists so
    that a future refactor back to SET fails here rather than in a benchmark.
    """
    with pytest.raises(psycopg.errors.SyntaxError):
        conn.execute(f"SET {EF_SEARCH_SETTING} = %s", (137,))

    bound = conn.execute("SELECT set_config(%s, %s, true)", (EF_SEARCH_SETTING, "137")).fetchone()
    assert bound == ("137",)

    # The failed SET above must not have poisoned the connection.
    assert conn.execute("SELECT 1").fetchone() == (1,)


def test_ef_search_is_visible_inside_its_block(conn: psycopg.Connection) -> None:
    """The value the caller asked for is the value the server reports."""
    with ef_search(conn, 137):
        assert current_ef_search(conn) == 137


def test_ef_search_reverts_on_commit(conn: psycopg.Connection) -> None:
    """Leaving the block is what ends the setting, not leaving the connection."""
    before = current_ef_search(conn)
    with ef_search(conn, 137):
        assert current_ef_search(conn) == 137
    assert current_ef_search(conn) == before


def test_ef_search_reverts_on_rollback(conn: psycopg.Connection) -> None:
    """An exception mid-experiment must not leave the parameter behind."""
    before = current_ef_search(conn)
    with pytest.raises(RuntimeError, match="boom"), ef_search(conn, 137):
        assert current_ef_search(conn) == 137
        raise RuntimeError("boom")
    assert current_ef_search(conn) == before


def test_ef_search_reverts_after_a_failing_query(conn: psycopg.Connection) -> None:
    """The realistic failure: the query itself errors part way through."""
    before = current_ef_search(conn)
    with pytest.raises(psycopg.errors.UndefinedTable), ef_search(conn, 137):
        conn.execute("SELECT 1 FROM a_table_that_does_not_exist")
    assert current_ef_search(conn) == before
    assert conn.execute("SELECT 1").fetchone() == (1,)


def test_the_transaction_scoped_setting_never_reaches_the_next_borrower() -> None:
    """A connection returned to the pool carries nothing from its last block."""
    pool = get_pool()
    with connection() as conn:
        pid = _backend_pid(conn)
        with ef_search(conn, 137):
            assert current_ef_search(conn) == 137

    leaked = _value_on_reused_connection(pool, pid)
    assert leaked == EF_SEARCH_DEFAULT, (
        f"connection {pid} came back carrying ef_search={leaked}, "
        f"expected the default {EF_SEARCH_DEFAULT}"
    )


def test_a_session_wide_setting_is_cleared_before_the_next_borrower() -> None:
    """The pool's reset callback is what makes this true.

    ``ef_search()`` is transaction-scoped and cannot leak, so this deliberately
    does the wrong thing -- a session-wide ``set_config(..., false)`` -- to
    prove the safety net catches callers that do not go through it.
    """
    pool = get_pool()
    with connection() as conn:
        pid = _backend_pid(conn)
        conn.execute("SELECT set_config(%s, %s, false)", (EF_SEARCH_SETTING, "137"))
        assert current_ef_search(conn) == 137, "the session-wide set did not take"

    leaked = _value_on_reused_connection(pool, pid)
    assert leaked == EF_SEARCH_DEFAULT, (
        f"connection {pid} was handed out still carrying ef_search={leaked}"
    )


def _value_on_reused_connection(pool: ConnectionPool, pid: int) -> int:
    """Borrow until the pool hands back the connection with ``pid``, then read it.

    Reuse cannot be assumed -- the pool is free to open another connection --
    so this checks the backend pid of every borrow rather than trusting that two
    consecutive checkouts are the same physical connection.
    """
    seen: list[int] = []
    for _ in range(pool.max_size + 4):
        with connection() as conn:
            seen.append(_backend_pid(conn))
            if seen[-1] == pid:
                return current_ef_search(conn)
    pytest.fail(f"pool never handed back connection {pid}; saw {seen}")


# ------------------------------------------------------ ef_search validation


@pytest.mark.parametrize("value", [0, -1, EF_SEARCH_MAX + 1, 10_000])
def test_ef_search_outside_the_supported_range_is_rejected(value: int) -> None:
    """Rejected here as well as by pgvector, whose own message names no parameter."""
    with pytest.raises(ValueError, match="between"):
        validate_ef_search(value)


@pytest.mark.parametrize("value", [True, False, 40.0, "40", None])
def test_ef_search_of_the_wrong_type_is_rejected(value: object) -> None:
    """``True`` is an int in Python; without the bool check it would mean 1."""
    with pytest.raises(ValueError, match="must be an int"):
        validate_ef_search(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [EF_SEARCH_MIN, 40, EF_SEARCH_MAX])
def test_ef_search_inside_the_supported_range_is_accepted(value: int) -> None:
    """The bounds are inclusive and come from pgvector, not from a guess."""
    assert validate_ef_search(value) == value
