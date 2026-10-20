"""Dense retrieval over the pgvector HNSW index.

Call flow: query text -> the Phase 3 embedder (BGE query instruction included)
-> one transaction with ``hnsw.ef_search`` set -> the inner ANN SELECT (Top-K
chunks) -> a real ``documents`` JOIN for metadata -> :class:`SearchHit` list.

Two properties are deliberate:

* The inner SELECT fixes the candidate set with ``LIMIT`` *before* the join, so
  the join can enrich rows but cannot inflate Top-K or duplicate chunks.
* ``distance`` is pgvector's raw cosine distance (lower is better) and
  ``score`` is cosine similarity ``1 - distance`` (higher is better). Both are
  returned so a later benchmark never has to guess which one it is.

At the current 150-chunk scale PostgreSQL may answer the inner SELECT with a
sequential scan, in which case ``ef_search`` has no effect. That is a planner
choice, not a configuration failure: callers that need to measure ANN behaviour
must verify the plan separately (see ``scripts/diagnose_dense.py``) instead of
inferring it from latency or result changes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Any

import psycopg
from pgvector import Vector
from psycopg_pool import PoolTimeout

from src.db import connection, current_ef_search, ef_search
from src.ingestion import IngestionError
from src.ingestion.embedder import (
    EmbeddingProvider,
    SentenceTransformerEmbedder,
    validate_vectors,
)
from src.retrieval import RetrievalError
from src.schema.retrieval import DenseConfig, SearchHit
from src.utils.logger import get_logger
from src.utils.timing import timer

logger = get_logger(__name__)

TOP_K_MIN, TOP_K_MAX = 1, 1000
"""Supported ``top_k`` range, matching ``DenseConfig.top_n`` and ef_search's cap."""

ConnectionFactory = Callable[[], AbstractContextManager[psycopg.Connection]]
"""A context manager that yields a pooled connection (injectable for tests)."""

# Keep this column order in sync with _rows_to_hits().
#
# The inner SELECT is the ANN stage: pgvector can answer its ORDER BY ... LIMIT
# with chunks_embedding_hnsw_idx, and its LIMIT fixes Top-K before the join.
# The outer join only attaches canonical document metadata. `%s` is bound as a
# pgvector.Vector; no vector, query text or numeric value is interpolated into
# the SQL string.
#
# `chunk_strategy` is a hard filter, not an afterthought: the primary key is
# (chunk_strategy, chunk_id), so one document may have several chunkings in the
# same table. Without the filter a Top-K could mix strategies and every metric
# computed from it would be meaningless.
DENSE_SEARCH_SQL = """
    SELECT
        c.chunk_id,
        c.doc_id,
        c.chunk_strategy,
        c.chunk_index,
        c.content,
        c.distance,
        d.title,
        d.category,
        d.card_tier,
        d.effective_year,
        d.policy_code
    FROM (
        SELECT
            chunk_id,
            doc_id,
            chunk_strategy,
            chunk_index,
            content,
            embedding <=> %s::vector AS distance
        FROM chunks
        WHERE chunk_strategy = %s
        ORDER BY distance
        LIMIT %s
    ) AS c
    JOIN documents AS d ON d.doc_id = c.doc_id
    ORDER BY c.distance
"""
"""Top-K chunks by cosine distance, enriched with their owning document."""


class DenseRetriever:
    """ANN retrieval against ``chunks.embedding`` and its HNSW cosine index.

    ``retrieve(query)`` uses ``DenseConfig.top_n`` as Top-K; passing ``top_k``
    overrides it. ``DenseConfig.ef_search`` is applied through the accepted
    P2-4 helper, inside the same transaction as the query. ``m`` and
    ``ef_construction`` are index build-time settings and are intentionally not
    applied here: they cannot change an existing index.
    """

    name = "dense"

    def __init__(
        self,
        config: DenseConfig | None = None,
        embedder: EmbeddingProvider | None = None,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        """Bind the search configuration and injectable dependencies.

        Args:
            config: ANN parameters. Defaults to project defaults (top_n=50,
                ef_search=40). They are deliberately unchanged from Phase 1.
            embedder: Query embedding provider. Defaults to the Phase 3 local
                BGE backend, so the model and 1024 dimensions stay identical to
                the indexed corpus.
            connection_factory: Context manager yielding a PostgreSQL
                connection. Defaults to the project pool. Tests inject a
                diagnostic transaction here; production must not force planner
                settings.
        """
        self._config = config if config is not None else DenseConfig()
        self._embedder = embedder if embedder is not None else SentenceTransformerEmbedder()
        self._connection_factory = (
            connection_factory if connection_factory is not None else connection
        )
        self._warned_truncation: set[tuple[int, int]] = set()

    @property
    def config(self) -> DenseConfig:
        """The ANN configuration this retriever actually uses."""
        return self._config

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return the nearest chunks for ``query``, ranked best first.

        Args:
            query: Non-empty natural-language query text. It is stripped before
                embedding; no other transformation is applied here, so the BGE
                query instruction remains the embedder's responsibility.
            top_k: Requested number of hits. ``None`` uses ``DenseConfig.top_n``.
                When ``ef_search < top_k`` pgvector's HNSW scan may return
                fewer rows; the retriever returns what HNSW produced and logs a
                warning rather than inventing rows.

        Returns:
            Chunk-level hits carrying both raw cosine distance and similarity
            score, plus chunk/document identifiers and metadata.

        Raises:
            ValueError: empty query or an invalid ``top_k`` (before embedding).
            RetrievalError: embedding failure, database failure/timeout, or a
                non-finite distance returned by the database.
        """
        query = _validate_query(query)
        limit = _validate_top_k(top_k, self._config.top_n)

        with timer() as total_timer:
            try:
                with timer() as embed_timer:
                    vector = self._embedder.embed_query(query)
                    validate_vectors([vector], self._embedder.dimension, source="query")
            except IngestionError as exc:
                logger.error(
                    "dense retrieval failed during embedding: %s: %s (embed_ms=%.2f)",
                    type(exc).__name__,
                    exc,
                    embed_timer["ms"],
                )
                raise RetrievalError(f"query embedding failed: {exc}") from exc

            try:
                with timer() as db_timer, self._connection_factory() as conn:
                    rows, actual_ef = self._search(conn, vector, limit)
            except psycopg.errors.QueryCanceled as exc:
                logger.error(
                    "dense retrieval cancelled by the server (db_ms=%.2f); "
                    "check statement_timeout or cancel the query",
                    db_timer["ms"],
                )
                raise RetrievalError("dense retrieval was cancelled or timed out") from exc
            except PoolTimeout as exc:
                logger.error(
                    "dense retrieval could not borrow a database connection (db_ms=%.2f)",
                    db_timer["ms"],
                )
                raise RetrievalError("timed out waiting for a database connection") from exc
            except psycopg.Error as exc:
                logger.error(
                    "dense retrieval failed in the database: %s: %s (db_ms=%.2f)",
                    type(exc).__name__,
                    exc,
                    db_timer["ms"],
                )
                raise RetrievalError(f"dense retrieval database error: {exc}") from exc
            except ConnectionError as exc:
                logger.error(
                    "dense retrieval database connection failed (db_ms=%.2f): %s",
                    db_timer["ms"],
                    exc,
                )
                raise RetrievalError(f"database connection failed: {exc}") from exc

            try:
                hits = _rows_to_hits(rows, expected_strategy=self._config.chunk_strategy)
            except RetrievalError as exc:
                logger.error(
                    "dense retrieval rejected database rows: %s (db_ms=%.2f)",
                    exc,
                    db_timer["ms"],
                )
                raise
            self._warn_if_truncated(len(hits), limit, actual_ef)

        logger.info(
            "dense retrieval: hits=%d top_k=%d ef_search=%d query_chars=%d "
            "embed_ms=%.2f db_ms=%.2f total_ms=%.2f",
            len(hits),
            limit,
            actual_ef,
            len(query),
            embed_timer["ms"],
            db_timer["ms"],
            total_timer["ms"],
        )
        return hits

    def _search(
        self,
        conn: psycopg.Connection,
        vector: Sequence[float],
        top_k: int,
    ) -> tuple[list[tuple[Any, ...]], int]:
        """Run the ANN query and read back the live GUC in the same transaction.

        ``prepare=False`` is intentional. psycopg's server-side prepared
        statements can cache a generic plan; Phase 4A showed that such a plan
        survives ``SET LOCAL enable_seqscan=off`` and silently ruins a plan
        measurement. Parsing this one statement per call is cheap at this scale.

        Returns:
            The raw rows and the ``hnsw.ef_search`` actually visible inside the
            transaction (not merely the configured value).
        """
        with ef_search(conn, self._config.ef_search):
            rows = conn.execute(
                DENSE_SEARCH_SQL,
                (Vector(list(vector)), self._config.chunk_strategy, top_k),
                prepare=False,
            ).fetchall()
            actual_ef = current_ef_search(conn)
        return rows, actual_ef

    def _warn_if_truncated(self, hits: int, top_k: int, actual_ef: int) -> None:
        """Warn once per (ef_search, top_k) when HNSW capped the result set."""
        if hits >= top_k or actual_ef >= top_k:
            return
        key = (actual_ef, top_k)
        if key in self._warned_truncation:
            return
        self._warned_truncation.add(key)
        logger.warning(
            "dense retrieval returned %d of %d requested hits: hnsw.ef_search=%d "
            "limits the HNSW scan; raise ef_search to at least top_k for a full Top-K",
            hits,
            top_k,
            actual_ef,
        )


def _validate_query(query: str) -> str:
    """Return the stripped query, or raise before any expensive work."""
    if not isinstance(query, str):
        raise ValueError(f"query must be a str, got {type(query).__name__}")
    stripped = query.strip()
    if not stripped:
        raise ValueError("query must be a non-empty string")
    return stripped


def _validate_top_k(top_k: int | None, default: int) -> int:
    """Return a validated Top-K, falling back to the configured default."""
    if top_k is None:
        return default
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError(f"top_k must be an int, got {type(top_k).__name__}")
    if not TOP_K_MIN <= top_k <= TOP_K_MAX:
        raise ValueError(f"top_k must be between {TOP_K_MIN} and {TOP_K_MAX}, got {top_k}")
    return top_k


def _rows_to_hits(
    rows: list[tuple[Any, ...]],
    *,
    expected_strategy: str | None = None,
) -> list[SearchHit]:
    """Map the SELECT's column order to validated :class:`SearchHit` objects.

    ``expected_strategy`` is a defence-in-depth check: the SQL already filters
    on it, and a mismatch here means the query or the test double is wrong.
    """
    hits: list[SearchHit] = []
    for rank, row in enumerate(rows, start=1):
        if expected_strategy is not None and str(row[2]) != expected_strategy:
            raise RetrievalError(
                f"dense retrieval returned chunk_strategy={row[2]!r} for "
                f"chunk {row[0]!r}, expected {expected_strategy!r}"
            )
        distance = float(row[5])
        if not math.isfinite(distance):
            raise RetrievalError(f"database returned a non-finite distance for chunk {row[0]!r}")
        hits.append(
            SearchHit(
                chunk_id=row[0],
                doc_id=row[1],
                chunk_strategy=row[2],
                chunk_index=row[3],
                content=row[4],
                distance=distance,
                score=1.0 - distance,
                rank=rank,
                source="dense",
                title=row[6],
                category=row[7],
                card_tier=row[8],
                effective_year=row[9],
                policy_code=row[10],
            )
        )
    return hits
