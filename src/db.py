"""Database access.

The connection pool, plus the schema diagnostic behind ``make db-check``.

Business queries belong to the modules that own them -- ``ingestion.indexer``
writes, the retrievers read -- and DDL belongs to ``docker/init.sql``. This
module only knows how to reach PostgreSQL, how to report what it finds there,
and how to scope per-query settings (:func:`ef_search`).
"""

from __future__ import annotations

import atexit
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool, PoolTimeout

from src.config import Settings, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_pool: ConnectionPool | None = None

EF_SEARCH_SETTING = "hnsw.ef_search"
"""The pgvector GUC that sizes the HNSW candidate list at *query* time.

Distinct from ``m`` / ``ef_construction``, which are baked into the index at
build time and cannot be changed without rebuilding it. This one is per query,
which is why Experiment 4 sweeps it against a fixed index. Scoped per query by
:func:`ef_search` and cleared by :func:`_reset`.
"""

EF_SEARCH_DEFAULT = 40
"""pgvector's built-in default, used when the GUC has never been set.

Not read from the server: PostgreSQL reports an unset ``hnsw.ef_search`` as the
empty string rather than as its boot value, so the documented default is the
only thing there is to report.
"""

EF_SEARCH_MIN, EF_SEARCH_MAX = 1, 1000
"""pgvector's accepted range (``HNSW_MIN_EF_SEARCH`` / ``HNSW_MAX_EF_SEARCH``).

Checked here so a sweep value outside it fails with a sentence naming the
parameter, instead of a bare ``value ... is out of bounds for option``.
"""


def _configure(conn: psycopg.Connection) -> None:
    """Register the pgvector adapters on every new pooled connection.

    Without this, VECTOR columns come back as raw strings and numpy arrays do
    not round-trip. The extension may legitimately be missing -- ``db-check``
    exists to report exactly that -- so a failure here is logged, not raised.
    """
    try:
        register_vector(conn)
    except psycopg.Error as exc:
        logger.warning("pgvector type adapters unavailable: %s", exc)


def _reset(conn: psycopg.Connection) -> None:
    """Clear session state as a connection goes back to the pool.

    Defence in depth, not the primary guarantee. The supported way to set a
    search parameter is :func:`ef_search`, which is transaction-local and so is
    already gone by the time a connection is returned. This catches the other
    case: a caller that sets the GUC session-wide, whose value would otherwise
    be handed to the next experiment as if it had asked for it.

    A failure is logged rather than raised -- the extension may legitimately be
    absent, which ``db-check`` reports properly, and neither is a reason to
    throw away a working connection.
    """
    try:
        conn.execute(f"RESET {EF_SEARCH_SETTING}")
    except psycopg.Error as exc:
        logger.warning("could not reset %s: %s", EF_SEARCH_SETTING, exc)


def get_pool(settings: Settings | None = None) -> ConnectionPool:
    """Return the process-wide pool, opening it on first use.

    Raises:
        ConnectionError: if the database cannot be reached, naming the target
            host, port and database. A bare ``PoolTimeout`` would tell a
            reviewer nothing about what to fix.
    """
    global _pool
    if _pool is not None:
        return _pool

    settings = settings or get_settings()
    pool = ConnectionPool(
        conninfo=settings.dsn,
        min_size=1,
        max_size=4,
        open=False,
        kwargs={"autocommit": True},
        configure=_configure,
        reset=_reset,
    )
    try:
        pool.open(wait=True, timeout=settings.pg_connect_timeout)
    except PoolTimeout as exc:
        pool.close()
        raise ConnectionError(
            f"cannot reach PostgreSQL at {settings.pg_host}:{settings.pg_port} "
            f"database {settings.pg_database!r} within {settings.pg_connect_timeout}s "
            "-- is it running? (`make up`)"
        ) from exc

    _pool = pool
    atexit.register(close_pool)
    return _pool


def close_pool() -> None:
    """Release every pooled connection. Safe to call more than once."""
    global _pool
    if _pool is None:
        return
    _pool.close()
    _pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Borrow a pooled connection and hand it back on exit.

    The only way the rest of the codebase touches the database, so no module
    opens a connection of its own.
    """
    with get_pool().connection() as conn:
        yield conn


# ---------------------------------------------------------- per-query settings


def validate_ef_search(value: int) -> int:
    """Return ``value`` if it is a usable ``hnsw.ef_search``, else raise.

    ``bool`` is rejected explicitly: it *is* an ``int`` in Python, so without
    that check ``ef_search=True`` would quietly mean 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"ef_search must be an int, got {type(value).__name__}")
    if not EF_SEARCH_MIN <= value <= EF_SEARCH_MAX:
        raise ValueError(
            f"ef_search must be between {EF_SEARCH_MIN} and {EF_SEARCH_MAX}, got {value}"
        )
    return value


@contextmanager
def ef_search(conn: psycopg.Connection, value: int) -> Iterator[int]:
    """Run a block with ``hnsw.ef_search`` set to ``value``, and nothing after.

    Two traps make the obvious spellings wrong, and both fail *quietly* or
    confusingly rather than loudly:

    ``SET hnsw.ef_search = %s``  cannot be parameterised. PostgreSQL's ``SET``
    takes a literal, so psycopg's ``%s`` arrives as a server-side ``$1`` and the
    statement dies with ``syntax error at or near "$1"``. The "fix" that looks
    like it works is interpolating the string by hand, which is how a search
    parameter turns into an injection point. ``set_config()`` is an ordinary
    function call, so it takes a bound parameter like any other.

    ``SET LOCAL`` / ``is_local => true``  lasts only for the enclosing
    transaction. The pool runs ``autocommit=True``, so ``SET LOCAL`` issued as a
    standalone statement expires with that statement -- the query that follows
    runs at whatever value was already on the connection. An explicit
    transaction is therefore not optional here; it is the mechanism.

    Being transaction-local is also what makes the value leak-proof: COMMIT and
    ROLLBACK both revert it, so a swept parameter cannot survive into the next
    experiment, even when the query in between raised.

    Yields:
        The validated value, so a caller can record what it actually ran at
        without a round trip to read it back.
    """
    value = validate_ef_search(value)
    with conn.transaction():
        conn.execute("SELECT set_config(%s, %s, true)", (EF_SEARCH_SETTING, str(value)))
        logger.debug("%s = %d for this transaction", EF_SEARCH_SETTING, value)
        yield value


def current_ef_search(conn: psycopg.Connection) -> int:
    """Return the ``hnsw.ef_search`` this connection would use right now.

    Reads the live setting rather than the configured one. The two differ
    whenever a transaction-local value is in scope, and "the config says 100" is
    not evidence that the query ran at 100 -- which is exactly the confusion
    this function exists to settle.
    """
    raw = scalar(conn, f"SELECT current_setting('{EF_SEARCH_SETTING}', true)")
    return EF_SEARCH_DEFAULT if raw in (None, "") else int(raw)


# --------------------------------------------------------------------- checks


@dataclass(frozen=True)
class CheckResult:
    """One line of the ``db-check`` report."""

    name: str
    ok: bool
    detail: str


def scalar(conn: psycopg.Connection, sql: str) -> Any:
    """Return the first column of the first row, or None when there is no row.

    Typed ``Any`` because the caller knows what its own query selects -- the
    alternative is a cast at every call site for no added safety.
    """
    row = conn.execute(sql).fetchone()
    return row[0] if row else None


def count(conn: psycopg.Connection, sql: str) -> int:
    """Run a ``count(*)`` query and return it as an int, never None."""
    return int(scalar(conn, sql) or 0)


def check_database() -> list[CheckResult]:
    """Verify the Phase 2 infrastructure against the live database.

    Returns one :class:`CheckResult` per check instead of raising on the first
    problem, so a single missing piece still produces a full report.
    """
    try:
        return _run_checks()
    except (psycopg.Error, ConnectionError) as exc:
        return [CheckResult("PostgreSQL connection", False, str(exc))]


def _run_checks() -> list[CheckResult]:
    results: list[CheckResult] = []

    with connection() as conn:
        results.append(CheckResult("PostgreSQL connection", True, "SELECT 1"))

        version = scalar(conn, "SHOW server_version")
        results.append(CheckResult("PostgreSQL version", True, str(version)))

        extension = scalar(conn, "SELECT extname FROM pg_extension WHERE extname = 'vector'")
        results.append(
            CheckResult(
                "pgvector extension",
                extension == "vector",
                str(extension) if extension else "not installed",
            )
        )

        tables = scalar(
            conn,
            """
            SELECT string_agg(relname, ', ' ORDER BY relname)
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND c.relkind = 'r'
              AND relname IN ('documents', 'chunks')
            """,
        )
        tables_ok = tables == "chunks, documents"
        results.append(
            CheckResult(
                "documents / chunks tables",
                tables_ok,
                str(tables) if tables else "missing",
            )
        )

        embedding_type = scalar(
            conn,
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND c.relname = 'chunks'
              AND a.attname = 'embedding' AND a.attnum > 0 AND NOT a.attisdropped
            """,
        )
        # Compared against the configured dimension, not a constant: changing
        # the embedding model without recreating the column is exactly the kind
        # of silent mismatch this check exists to catch.
        expected_type = f"vector({get_settings().embedding_dim})"
        results.append(
            CheckResult(
                "embedding vector type",
                embedding_type == expected_type,
                f"{embedding_type} (expected {expected_type})"
                if embedding_type
                else "column missing",
            )
        )

        generated = scalar(
            conn,
            """
            SELECT a.attgenerated
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            WHERE c.relname = 'chunks' AND a.attname = 'tsv' AND a.attnum > 0
            """,
        )
        results.append(
            CheckResult(
                "search_vector generated column",
                generated == "s",
                "tsv is GENERATED ALWAYS ... STORED" if generated == "s" else "not generated",
            )
        )

        hnsw = conn.execute(
            """
            SELECT i.relname, pg_get_indexdef(x.indexrelid)
            FROM pg_index x
            JOIN pg_class i ON i.oid = x.indexrelid
            JOIN pg_class t ON t.oid = x.indrelid
            JOIN pg_am am  ON am.oid = i.relam
            WHERE t.relname = 'chunks' AND am.amname = 'hnsw'
            """
        ).fetchall()
        hnsw_ok = len(hnsw) == 1 and "vector_cosine_ops" in hnsw[0][1]
        results.append(
            CheckResult(
                "HNSW index (cosine)",
                hnsw_ok,
                f"{hnsw[0][0]} on vector_cosine_ops" if hnsw else "no HNSW index found",
            )
        )

        gin = scalar(
            conn,
            """
            SELECT i.relname
            FROM pg_index x
            JOIN pg_class i ON i.oid = x.indexrelid
            JOIN pg_class t ON t.oid = x.indrelid
            JOIN pg_am am  ON am.oid = i.relam
            WHERE t.relname = 'chunks' AND am.amname = 'gin'
            """,
        )
        results.append(
            CheckResult("GIN index on search_vector", gin is not None, str(gin or "missing"))
        )

        # Guarded: counting a table that does not exist would abort the whole
        # report and hide the checks that already passed.
        #
        # This asserts a relationship rather than a size, so it holds both before
        # and after `ingest`: an empty corpus is consistent, and so is one where
        # every document has at least one chunk. It fails on the state that
        # actually matters -- a half-finished import.
        if tables_ok:
            docs = count(conn, "SELECT count(*) FROM documents")
            chunks = count(conn, "SELECT count(*) FROM chunks")
            results.append(
                CheckResult(
                    "corpus consistent",
                    chunks >= docs,
                    f"documents={docs} chunks={chunks}"
                    + ("" if docs else " (empty -- run `make ingest`)"),
                )
            )

    return results
