"""Index writer.

Chunk + vector -> PostgreSQL. Bulk insert, tsvector maintenance, and
HNSW / IVFFlat index (re)build with tunable parameters. No querying.

Implemented in: Phases 3-4 (ingestion, index tuning)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

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
