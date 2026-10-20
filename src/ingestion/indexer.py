"""Index writer and vector-index management.

Chunk + vector -> PostgreSQL, plus HNSW / IVFFlat index management with
tunable build parameters. Retrieval queries live in ``src.retrieval``; this
module owns writes and index DDL only.

Index builds are **shadow by default**: they create a new versioned index
(``chunks_embedding_hnsw_v001`` ...) and leave the existing production index
untouched. Replacing or dropping a production index is a separate, explicit
call. ``CREATE INDEX CONCURRENTLY`` cannot run inside a transaction block, so it
uses the pool's autocommit connection and is refused otherwise.

Build-time parameters (``m``, ``ef_construction``, ``lists``) are integers that
PostgreSQL's ``WITH (...)`` clause cannot accept as bind parameters. They are
validated as bounded integers and identifiers are validated against a strict
pattern before interpolation; no user text ever reaches the SQL string.

Implemented in: Phase 3 (ingestion). The index-management API was added later;
it is covered by pure unit tests only and no index DDL has been executed.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import psycopg

from src.db import connection, count
from src.ingestion import IngestionError
from src.ingestion.embedder import validate_vectors
from src.schema.document import Chunk, Document
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Both statements are upserts against a primary key the corpus already carries,
# which is what makes `ingest` re-runnable: the second run re-embeds and rewrites
# the same 150 rows instead of appending 150 more.
#
# `tsv` is absent from both column lists and from the DO UPDATE set -- it is a
# GENERATED column, and PostgreSQL recomputes it from the new row on its own.
INSERT_DOCUMENT = """
    INSERT INTO documents (
        doc_id, title, category, card_tier, effective_year, policy_code, content
    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (doc_id) DO UPDATE SET
        title          = EXCLUDED.title,
        category       = EXCLUDED.category,
        card_tier      = EXCLUDED.card_tier,
        effective_year = EXCLUDED.effective_year,
        policy_code    = EXCLUDED.policy_code,
        content        = EXCLUDED.content
"""

INSERT_CHUNK = """
    INSERT INTO chunks (
        chunk_id, doc_id, chunk_strategy, chunk_index, title, content, embedded_text,
        embedding, category, card_tier, effective_year, policy_code
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (chunk_strategy, chunk_id) DO UPDATE SET
        doc_id         = EXCLUDED.doc_id,
        chunk_index    = EXCLUDED.chunk_index,
        title          = EXCLUDED.title,
        content        = EXCLUDED.content,
        embedded_text  = EXCLUDED.embedded_text,
        embedding      = EXCLUDED.embedding,
        category       = EXCLUDED.category,
        card_tier      = EXCLUDED.card_tier,
        effective_year = EXCLUDED.effective_year,
        policy_code    = EXCLUDED.policy_code
"""


@dataclass(frozen=True)
class UpsertStats:
    """What one call to :func:`upsert_corpus` did to the database.

    Insert and update are told apart by comparing row counts across the same
    transaction, so the numbers describe real state changes rather than what the
    statement claimed it would do.
    """

    documents_written: int
    chunks_written: int
    documents_inserted: int
    chunks_inserted: int

    @property
    def documents_updated(self) -> int:
        """Rows that already existed and were rewritten."""
        return self.documents_written - self.documents_inserted

    @property
    def chunks_updated(self) -> int:
        """Rows that already existed and were rewritten."""
        return self.chunks_written - self.chunks_inserted


@dataclass(frozen=True)
class ImportStats:
    """What the database actually holds after an import."""

    documents: int
    chunks: int
    embedded_chunks: int
    distinct_doc_ids: int
    min_dimension: int | None
    max_dimension: int | None
    strategies: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexStats:
    """A vector index as PostgreSQL currently holds it.

    ``build_ms`` is client-side wall time for the call that created the index;
    it is ``None`` when the statistics were read back by
    :func:`collect_index_stats`, because PostgreSQL does not record build
    duration. Never report ``None`` as zero.
    """

    name: str
    method: Literal["hnsw", "ivfflat"]
    definition: str
    size_bytes: int
    size_pretty: str
    parameters: dict[str, int] = field(default_factory=dict)
    build_ms: float | None = None
    valid: bool = True


@dataclass(frozen=True)
class IndexInfo:
    """One row of :func:`list_vector_indexes`."""

    name: str
    method: Literal["hnsw", "ivfflat"]
    definition: str
    size_bytes: int
    size_pretty: str
    parameters: dict[str, int] = field(default_factory=dict)
    valid: bool = True


HNSW_INDEX_PREFIX = "chunks_embedding_hnsw_v"
IVFFLAT_INDEX_PREFIX = "chunks_embedding_ivfflat_v"
BOOTSTRAP_INDEX = "chunks_embedding_hnsw_idx"
"""The index created by ``docker/init.sql``. Never dropped by default."""

_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
"""Safe unquoted SQL identifier. Index names are interpolated, so this is a gate."""

_METHOD_AMNAME = {"hnsw": "hnsw", "ivfflat": "ivfflat"}

_HNSW_M_MIN, _HNSW_M_MAX = 2, 100
_HNSW_EF_CONSTRUCTION_MIN, _HNSW_EF_CONSTRUCTION_MAX = 4, 1000
_IVFFLAT_LISTS_MIN, _IVFFLAT_LISTS_MAX = 1, 32768


def upsert_corpus(
    documents: Sequence[Document],
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
) -> UpsertStats:
    """Write documents and their embeddings in one transaction.

    All or nothing: a failure part way through leaves the database exactly as it
    was, so a re-run starts from a known state rather than a half-imported
    corpus. At 150 rows the cost of one transaction is nil.

    Raises:
        IngestionError: ``chunks`` and ``vectors`` disagree in length, a vector
            has the wrong width, or a value is not finite.
        psycopg.Error: propagated unchanged; the transaction is rolled back.
    """
    if len(chunks) != len(vectors):
        raise IngestionError(
            f"got {len(chunks)} chunks but {len(vectors)} vectors; they must correspond one to one"
        )

    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) > 1:
        raise IngestionError(f"vectors disagree on dimension: found widths {sorted(dimensions)}")
    for width in dimensions:
        validate_vectors(vectors, width, source="chunks")

    titles = {document.doc_id: document.title for document in documents}
    missing = {chunk.doc_id for chunk in chunks} - titles.keys()
    if missing:
        raise IngestionError(
            f"chunk(s) reference document(s) that are not in this batch: {sorted(missing)}"
        )

    with connection() as conn, conn.transaction():
        before = _row_counts(conn)

        with conn.cursor() as cur:
            cur.executemany(
                INSERT_DOCUMENT,
                [
                    (
                        document.doc_id,
                        document.title,
                        document.category,
                        document.card_tier,
                        document.effective_year,
                        document.policy_code,
                        document.content,
                    )
                    for document in documents
                ],
            )
            # Documents first: chunks.doc_id is a foreign key, and the constraint
            # is immediate -- a chunk cannot precede its document in one transaction.
            cur.executemany(
                INSERT_CHUNK,
                [
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.chunk_strategy,
                        chunk.chunk_index,
                        # Denormalised from the document: the weighted tsvector is
                        # a GENERATED column and cannot reach across tables.
                        titles[chunk.doc_id],
                        chunk.content,
                        chunk.embedded_text,
                        # A plain list: pgvector's dumper has a dedicated path for
                        # lists, and `pgvector.Vector` is a value wrapper, not an
                        # ndarray subclass -- wrapping gains nothing here.
                        list(vector),
                        chunk.category,
                        chunk.card_tier,
                        chunk.effective_year,
                        chunk.policy_code,
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )

        after = _row_counts(conn)

    stats = UpsertStats(
        documents_written=len(documents),
        chunks_written=len(chunks),
        documents_inserted=after[0] - before[0],
        chunks_inserted=after[1] - before[1],
    )
    logger.info(
        "wrote %d documents (%d new) and %d chunks (%d new)",
        stats.documents_written,
        stats.documents_inserted,
        stats.chunks_written,
        stats.chunks_inserted,
    )
    return stats


def collect_import_stats() -> ImportStats:
    """Read back what the database holds. One round trip per figure.

    Every count is a query the acceptance rubric names explicitly, so the CLI
    reports measured numbers rather than the numbers it intended to write.
    """
    with connection() as conn:
        documents = count(conn, "SELECT count(*) FROM documents")
        chunks = count(conn, "SELECT count(*) FROM chunks")
        embedded = count(conn, "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
        distinct_docs = count(conn, "SELECT count(DISTINCT doc_id) FROM chunks")

        # NULL when the table is empty, which is why these stay optional.
        dimensions = conn.execute(
            "SELECT min(vector_dims(embedding)), max(vector_dims(embedding)) FROM chunks"
        ).fetchone()

        strategies = dict(
            conn.execute(
                "SELECT chunk_strategy, count(*) FROM chunks GROUP BY chunk_strategy ORDER BY 1"
            ).fetchall()
        )

    return ImportStats(
        documents=documents,
        chunks=chunks,
        embedded_chunks=embedded,
        distinct_doc_ids=distinct_docs,
        min_dimension=dimensions[0] if dimensions else None,
        max_dimension=dimensions[1] if dimensions else None,
        strategies={str(key): int(value) for key, value in strategies.items()},
    )


def _row_counts(conn: psycopg.Connection) -> tuple[int, int]:
    """Return ``(documents, chunks)`` row counts."""
    return (
        count(conn, "SELECT count(*) FROM documents"),
        count(conn, "SELECT count(*) FROM chunks"),
    )


# ------------------------------------------------------------ index management


def build_hnsw_index(
    *,
    m: int = 16,
    ef_construction: int = 64,
    name: str | None = None,
    concurrently: bool = False,
    allow_production: bool = False,
) -> IndexStats:
    """Build an HNSW cosine index as a shadow index.

    Args:
        m: Connections per layer (pgvector range 2..100).
        ef_construction: Build candidate list size (pgvector range 4..1000).
        name: Explicit index name. ``None`` allocates the next
            ``chunks_embedding_hnsw_vNNN``.
        concurrently: Use ``CREATE INDEX CONCURRENTLY``. Requires an
            autocommit connection; it cannot run inside a transaction block.
        allow_production: Permit writing to the bootstrap index name. Off by
            default because replacing the served index is a separate decision.

    Returns:
        The index as PostgreSQL reports it after the build, plus client-side
        ``build_ms``.

    Raises:
        IngestionError: invalid parameters/name, the name already exists, or a
            concurrent build was requested on a transactional connection.
        psycopg.Error: propagated unchanged.
    """
    resolved_m = _bounded_int(m, _HNSW_M_MIN, _HNSW_M_MAX, "m")
    resolved_ef = _bounded_int(
        ef_construction,
        _HNSW_EF_CONSTRUCTION_MIN,
        _HNSW_EF_CONSTRUCTION_MAX,
        "ef_construction",
    )
    return _build_index(
        "hnsw",
        name=name,
        concurrently=concurrently,
        allow_production=allow_production,
        options_sql=f"m = {resolved_m}, ef_construction = {resolved_ef}",
    )


def build_ivfflat_index(
    *,
    lists: int = 100,
    name: str | None = None,
    concurrently: bool = False,
    allow_production: bool = False,
) -> IndexStats:
    """Build an IVFFlat cosine index as a shadow index.

    Args:
        lists: Number of inverted lists (pgvector range 1..32768). A common
            starting point is ``rows / 1000`` for small corpora.
        name: Explicit index name. ``None`` allocates the next
            ``chunks_embedding_ivfflat_vNNN``.
        concurrently: See :func:`build_hnsw_index`.
        allow_production: See :func:`build_hnsw_index`.

    Raises:
        IngestionError: invalid parameters/name or a name collision.
        psycopg.Error: propagated unchanged.
    """
    resolved_lists = _bounded_int(lists, _IVFFLAT_LISTS_MIN, _IVFFLAT_LISTS_MAX, "lists")
    return _build_index(
        "ivfflat",
        name=name,
        concurrently=concurrently,
        allow_production=allow_production,
        options_sql=f"lists = {resolved_lists}",
    )


def _build_index(
    method: Literal["hnsw", "ivfflat"],
    *,
    name: str | None,
    concurrently: bool,
    allow_production: bool,
    options_sql: str,
) -> IndexStats:
    """Create one versioned shadow index and read it back."""
    with connection() as conn:
        resolved_name = _resolve_index_name(
            conn,
            method,
            name=name,
            allow_production=allow_production,
        )
        if _index_row(conn, resolved_name) is not None:
            raise IngestionError(
                f"index {resolved_name!r} already exists; drop it explicitly before rebuilding"
            )
        if concurrently and not conn.autocommit:
            raise IngestionError(
                "CREATE INDEX CONCURRENTLY cannot run inside a transaction; "
                "use an autocommit connection or concurrently=False"
            )
        sql = (
            f"CREATE INDEX{' CONCURRENTLY' if concurrently else ''} {resolved_name} "
            f"ON chunks USING {method} (embedding vector_cosine_ops) "
            f"WITH ({options_sql})"
        )
        started = time.perf_counter()
        conn.execute(sql)
        build_ms = (time.perf_counter() - started) * 1000.0
        row = _index_row(conn, resolved_name)
    if row is None:
        raise IngestionError(f"index {resolved_name!r} was not found after creation")
    stats = _stats_from_row(row, expected_method=method, build_ms=build_ms)
    logger.info(
        "built %s index %s in %.2f ms (%s)",
        method,
        stats.name,
        build_ms,
        stats.size_pretty,
    )
    return stats


def _validate_identifier(name: str) -> str:
    """Return ``name`` when it is safe to interpolate as an SQL identifier."""
    if not _IDENTIFIER_RE.fullmatch(name):
        raise IngestionError(
            f"invalid index name {name!r}: use lowercase letters, digits and "
            "underscores, starting with a letter or underscore"
        )
    return name


def _resolve_index_name(
    conn: psycopg.Connection,
    method: Literal["hnsw", "ivfflat"],
    *,
    name: str | None,
    allow_production: bool,
) -> str:
    """Validate an explicit name or allocate the next shadow version."""
    if name is not None:
        _validate_identifier(name)
        if name == BOOTSTRAP_INDEX and not allow_production:
            raise IngestionError(
                f"{BOOTSTRAP_INDEX!r} is the served index; pass allow_production=True "
                "and drop/switch explicitly"
            )
        return name
    prefix = HNSW_INDEX_PREFIX if method == "hnsw" else IVFFLAT_INDEX_PREFIX
    rows = conn.execute(
        """
        SELECT c.relname
        FROM pg_class AS c
        JOIN pg_index AS i ON i.indexrelid = c.oid
        WHERE i.indrelid = 'chunks'::regclass
          AND c.relname LIKE %s
        """,
        (prefix + "%",),
    ).fetchall()
    highest = 0
    for (existing,) in rows:
        match = re.fullmatch(re.escape(prefix) + r"(\d{3,})", str(existing))
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{prefix}{highest + 1:03d}"


def drop_index(
    name: str,
    *,
    concurrently: bool = False,
    allow_production: bool = False,
) -> None:
    """Drop one index, refusing the served bootstrap index by default.

    Args:
        name: Index name. Validated against a strict identifier pattern.
        concurrently: Use ``DROP INDEX CONCURRENTLY``; requires autocommit.
        allow_production: Permit dropping the bootstrap index.

    Raises:
        IngestionError: invalid name, the bootstrap guard is active, the index
            does not exist, or a concurrent drop meets a transactional
            connection.
        psycopg.Error: propagated unchanged.
    """
    _validate_identifier(name)
    if name == BOOTSTRAP_INDEX and not allow_production:
        raise IngestionError(
            f"refusing to drop the served index {BOOTSTRAP_INDEX!r}; "
            "pass allow_production=True only after planning a replacement"
        )
    with connection() as conn:
        if _index_row(conn, name) is None:
            raise IngestionError(f"index {name!r} does not exist")
        if concurrently and not conn.autocommit:
            raise IngestionError(
                "DROP INDEX CONCURRENTLY cannot run inside a transaction; "
                "use an autocommit connection or concurrently=False"
            )
        conn.execute(f"DROP INDEX{' CONCURRENTLY' if concurrently else ''} {name}")
    logger.info("dropped index %s", name)


def list_vector_indexes() -> list[IndexInfo]:
    """Return every pgvector HNSW/IVFFlat index on ``chunks``."""
    with connection() as conn:
        rows = conn.execute(_INDEX_ROWS_SQL).fetchall()
    return [
        IndexInfo(
            name=str(row[0]),
            method=_method(str(row[6])),
            definition=str(row[1]),
            size_bytes=int(row[2]),
            size_pretty=str(row[3]),
            parameters=_parse_options(str(row[4])),
            valid=bool(row[5]),
        )
        for row in rows
    ]


def collect_index_stats(name: str) -> IndexStats:
    """Read one index's definition, size, reloptions and validity.

    ``build_ms`` is ``None``: PostgreSQL does not record how long an existing
    index took to build.
    """
    _validate_identifier(name)
    with connection() as conn:
        row = _index_row(conn, name)
    if row is None:
        raise IngestionError(f"index {name!r} does not exist")
    return _stats_from_row(row, expected_method=None, build_ms=None)


_INDEX_ROWS_SQL = """
    SELECT
        c.relname,
        pg_get_indexdef(i.indexrelid),
        pg_relation_size(c.oid),
        pg_size_pretty(pg_relation_size(c.oid)),
        coalesce(array_to_string(c.reloptions, ','), ''),
        i.indisvalid,
        am.amname
    FROM pg_class AS c
    JOIN pg_index AS i ON i.indexrelid = c.oid
    JOIN pg_am AS am ON am.oid = c.relam
    WHERE i.indrelid = 'chunks'::regclass
      AND am.amname IN ('hnsw', 'ivfflat')
    ORDER BY c.relname
"""


def _index_row(conn: psycopg.Connection, name: str) -> tuple[Any, ...] | None:
    """Return one index's catalog row, or ``None`` when it does not exist."""
    row = conn.execute(
        """
        SELECT
            c.relname,
            pg_get_indexdef(i.indexrelid),
            pg_relation_size(c.oid),
            pg_size_pretty(pg_relation_size(c.oid)),
            coalesce(array_to_string(c.reloptions, ','), ''),
            i.indisvalid,
            am.amname
        FROM pg_class AS c
        JOIN pg_index AS i ON i.indexrelid = c.oid
        JOIN pg_am AS am ON am.oid = c.relam
        WHERE i.indrelid = 'chunks'::regclass
          AND c.relname = %s
        """,
        (name,),
    ).fetchone()
    return tuple(row) if row is not None else None


def _stats_from_row(
    row: tuple[Any, ...],
    *,
    expected_method: Literal["hnsw", "ivfflat"] | None,
    build_ms: float | None,
) -> IndexStats:
    """Map a catalog row to :class:`IndexStats` and check the method."""
    method = _method(str(row[6]))
    if expected_method is not None and method != expected_method:
        raise IngestionError(
            f"index {row[0]!r} is {method}, expected {expected_method}"
        )
    return IndexStats(
        name=str(row[0]),
        method=method,
        definition=str(row[1]),
        size_bytes=int(row[2]),
        size_pretty=str(row[3]),
        parameters=_parse_options(str(row[4])),
        build_ms=build_ms,
        valid=bool(row[5]),
    )


def _method(amname: str) -> Literal["hnsw", "ivfflat"]:
    """Narrow a catalog access-method name."""
    if amname not in _METHOD_AMNAME:
        raise IngestionError(f"unexpected vector index access method {amname!r}")
    return amname  # type: ignore[return-value]


def _parse_options(reloptions: str) -> dict[str, int]:
    """Parse ``m=16,ef_construction=64``-style reloptions into ints."""
    options: dict[str, int] = {}
    for item in reloptions.split(","):
        key, _, value = item.partition("=")
        key = key.strip()
        if not key:
            continue
        try:
            options[key] = int(value.strip())
        except ValueError:
            continue
    return options


def _bounded_int(value: int, low: int, high: int, field_name: str) -> int:
    """Return ``value`` when it is an int in range; reject bools explicitly."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise IngestionError(f"{field_name} must be an int, got {type(value).__name__}")
    if not low <= value <= high:
        raise IngestionError(f"{field_name} must be between {low} and {high}, got {value}")
    return value
