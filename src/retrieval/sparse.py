"""Sparse retrieval over PostgreSQL full-text search.

The indexed text is the existing generated ``chunks.tsv`` column:

* ``title`` and ``card_tier`` are weighted ``A``;
* ``content`` is weighted ``B``;
* the column is built with the ``english`` text-search configuration and has a
  ``chunks_tsv_gin`` index.

English queries use ``websearch_to_tsquery`` and rank with ``ts_rank_cd``. The
score is a weighted cover-density rank: higher is better, it is **not** a
probability, and it is not comparable across queries.

Plain natural-language queries default to ``query_mode="or"``. PostgreSQL's
``websearch_to_tsquery`` joins bare terms with AND, which makes an ordinary
question ("What is the fee for the Everyday Freedom card?") require every
content word and often return zero rows. The retriever therefore joins plain
terms with the websearch ``or`` operator, preserving recall while
``ts_rank_cd`` still ranks documents that cover more of the query higher.
Quoted phrases and ``-term`` are passed through unchanged; a bare ``or`` is
treated as ordinary prose and removed before terms are OR-joined.
``query_mode="and"`` is available for strict all-term matching.

Chinese is handled by a deliberately small fallback. PostgreSQL's ``english``
configuration does not segment Chinese -- a run of Han characters becomes one
lexeme -- and no Chinese segmentation extension (zhparser/pg_jieba) is
installed. The fallback therefore matches overlapping two-character substrings
with ``strpos``. That is substring matching, not word segmentation. The corpus
and the 30-query evaluation set are English, so the fallback is behaviourally
verified but not shown to retrieve Chinese documents; see the Phase 5A report.

No schema change, index rebuild, or write is performed here.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Any, Literal

import psycopg
from psycopg_pool import PoolTimeout

from src.db import connection
from src.retrieval import RetrievalError
from src.schema.retrieval import QueryMode, SearchHit, SparseConfig
from src.utils.logger import get_logger
from src.utils.timing import timer

logger = get_logger(__name__)

TOP_K_MIN, TOP_K_MAX = 1, 1000
"""Supported ``top_k`` range, matching ``SparseConfig.top_n``."""

ConnectionFactory = Callable[[], AbstractContextManager[psycopg.Connection]]
"""A context manager that yields a pooled connection (injectable for tests)."""

SearchPath = Literal["fts", "cjk"]
"""Which real-text path produced a result set."""


CJK_RUN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")
"""Matches a contiguous run of Han characters."""

EXPLICIT_QUERY_SYNTAX_RE = re.compile(r'"|\s-\S')
"""Detects websearch syntax that already expresses the caller's intent.

Quoted phrases and a negated term are left untouched when ``query_mode="or"``
builds an OR query; re-writing them would change meaning. A bare ``or`` is not
treated as explicit syntax: in a natural question such as "Global Entry or TSA
PreCheck" it is ordinary prose, and rewriting the whole question as OR still
preserves the caller's alternatives.
"""

# Keep this column order in sync with _rows_to_hits().
#
# `%s::float4[]` is the {D, C, B, A} weight vector, `%s::regconfig` is the text
# search configuration, `%s` is the query text, and the final `%s` is Top-K.
# Every value is bound; no user text is interpolated into SQL.
SPARSE_SEARCH_SQL = """
    SELECT
        c.chunk_id,
        c.doc_id,
        c.chunk_strategy,
        c.chunk_index,
        c.content,
        ts_rank_cd(%s::float4[], c.tsv, q.query) AS score,
        d.title,
        d.category,
        d.card_tier,
        d.effective_year,
        d.policy_code
    FROM chunks AS c
    JOIN documents AS d ON d.doc_id = c.doc_id
    CROSS JOIN (
        SELECT websearch_to_tsquery(%s::regconfig, %s) AS query
    ) AS q
    WHERE c.tsv @@ q.query
    ORDER BY score DESC, c.chunk_strategy, c.chunk_id
    LIMIT %s
"""
"""Top-K chunks by weighted PostgreSQL full-text rank."""

SPARSE_CJK_SQL = """
    SELECT
        c.chunk_id,
        c.doc_id,
        c.chunk_strategy,
        c.chunk_index,
        c.content,
        (
            SELECT count(*)
            FROM unnest(%s::text[]) AS bg
            WHERE strpos(c.title, bg) > 0
               OR strpos(c.content, bg) > 0
               OR strpos(c.card_tier, bg) > 0
        )::double precision AS score,
        d.title,
        d.category,
        d.card_tier,
        d.effective_year,
        d.policy_code
    FROM chunks AS c
    JOIN documents AS d ON d.doc_id = c.doc_id
    WHERE EXISTS (
        SELECT 1
        FROM unnest(%s::text[]) AS bg
        WHERE strpos(c.title, bg) > 0
           OR strpos(c.content, bg) > 0
           OR strpos(c.card_tier, bg) > 0
    )
    ORDER BY score DESC, c.chunk_strategy, c.chunk_id
    LIMIT %s
"""
"""CJK fallback: rank chunks by how many query bigrams occur as substrings.

The score is a count of matched bigrams, not ``ts_rank_cd`` and not a
probability. It is only meaningful within one query's result list.
"""


class SparseRetriever:
    """Full-text retrieval against ``chunks.tsv``.

    ``retrieve(query)`` uses ``SparseConfig.top_n`` as Top-K; passing ``top_k``
    overrides it. A query containing Han characters uses the CJK substring
    fallback; all other queries use PostgreSQL FTS. The two paths return the
    same :class:`SearchHit` shape but their scores have different meanings (see
    the module docstring).
    """

    name = "sparse"

    def __init__(
        self,
        config: SparseConfig | None = None,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        """Bind the sparse configuration and injectable connection source.

        Args:
            config: FTS Top-K, language and rank weights. Defaults to
                ``SparseConfig()`` (Top-50, english, PostgreSQL defaults).
            connection_factory: Context manager yielding a PostgreSQL
                connection. Defaults to the project pool. Tests inject a
                diagnostic connection here; production does not force planner
                settings.
        """
        self._config = config if config is not None else SparseConfig()
        self._connection_factory = (
            connection_factory if connection_factory is not None else connection
        )

    @property
    def config(self) -> SparseConfig:
        """The sparse configuration this retriever actually uses."""
        return self._config

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return the best free-text matches for ``query``, ranked best first.

        Args:
            query: Non-empty natural-language query text. It is stripped before
                searching; PostgreSQL applies stop-word removal and stemming.
                Plain terms are OR-joined by default (see ``SparseConfig``).
            top_k: Requested number of hits. ``None`` uses ``SparseConfig.top_n``.

        Returns:
            Chunk-level hits carrying a non-negative sparse score and the same
            document metadata as dense retrieval. ``distance`` is ``None``
            because full-text rank is not a vector distance.

        Raises:
            ValueError: empty/non-string query or an invalid ``top_k``.
            RetrievalError: database failure/timeout or a non-finite/negative
                score returned by PostgreSQL.
        """
        query = _validate_query(query)
        limit = _validate_top_k(top_k, self._config.top_n)

        with timer() as total_timer:
            try:
                with timer() as db_timer, self._connection_factory() as conn:
                    rows, path, query_ms = self._search(conn, query, limit)
            except psycopg.errors.QueryCanceled as exc:
                logger.error(
                    "sparse retrieval cancelled by the server (db_ms=%.2f); "
                    "check statement_timeout or cancel the query",
                    db_timer["ms"],
                )
                raise RetrievalError("sparse retrieval was cancelled or timed out") from exc
            except PoolTimeout as exc:
                logger.error(
                    "sparse retrieval could not borrow a database connection (db_ms=%.2f)",
                    db_timer["ms"],
                )
                raise RetrievalError("timed out waiting for a database connection") from exc
            except psycopg.Error as exc:
                logger.error(
                    "sparse retrieval failed in the database: %s: %s (db_ms=%.2f)",
                    type(exc).__name__,
                    exc,
                    db_timer["ms"],
                )
                raise RetrievalError(f"sparse retrieval database error: {exc}") from exc
            except ConnectionError as exc:
                logger.error(
                    "sparse retrieval database connection failed (db_ms=%.2f): %s",
                    db_timer["ms"],
                    exc,
                )
                raise RetrievalError(f"database connection failed: {exc}") from exc

            try:
                hits = _rows_to_hits(rows)
            except RetrievalError as exc:
                logger.error(
                    "sparse retrieval rejected database rows: %s (db_ms=%.2f)",
                    exc,
                    db_timer["ms"],
                )
                raise

        logger.info(
            "sparse retrieval: path=%s hits=%d top_k=%d query_chars=%d "
            "query_ms=%.2f db_ms=%.2f total_ms=%.2f",
            path,
            len(hits),
            limit,
            len(query),
            query_ms,
            db_timer["ms"],
            total_timer["ms"],
        )
        return hits

    def _search(
        self,
        conn: psycopg.Connection,
        query: str,
        top_k: int,
    ) -> tuple[list[tuple[Any, ...]], SearchPath, float]:
        """Run one real-text query and return rows, path and SQL time.

        ``prepare=False`` matches the dense path: a generic prepared plan can
        hide changes in planner behaviour, and parsing one statement per call
        is cheap at this scale.
        """
        cjk_terms = _cjk_bigrams(query)
        if cjk_terms:
            started = time.perf_counter()
            rows = conn.execute(
                SPARSE_CJK_SQL,
                (cjk_terms, cjk_terms, top_k),
                prepare=False,
            ).fetchall()
            query_ms = (time.perf_counter() - started) * 1000.0
            return rows, "cjk", query_ms

        fts_query = _fts_query_text(query, self._config.query_mode)
        started = time.perf_counter()
        rows = conn.execute(
            SPARSE_SEARCH_SQL,
            (list(self._config.rank_weights), self._config.language, fts_query, top_k),
            prepare=False,
        ).fetchall()
        query_ms = (time.perf_counter() - started) * 1000.0
        return rows, "fts", query_ms


def _validate_query(query: str) -> str:
    """Return the stripped query, or raise before touching the database."""
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


def _fts_query_text(query: str, mode: QueryMode) -> str:
    """Build the text passed to ``websearch_to_tsquery``.

    ``mode="and"`` returns the query unchanged, because bare websearch terms
    are AND-joined by PostgreSQL. ``mode="or"`` joins whitespace-separated
    terms with the websearch ``or`` operator, unless the query already uses
    explicit websearch syntax. Every result is still bound as a SQL parameter;
    this function only chooses the boolean operator between plain terms.
    """
    if mode not in {"or", "and"}:
        raise ValueError(f"query_mode must be 'or' or 'and', got {mode!r}")
    if mode == "and" or EXPLICIT_QUERY_SYNTAX_RE.search(query):
        return query
    terms = [term for term in query.split() if term.lower() != "or"]
    return " or ".join(terms) if terms else query


def _cjk_bigrams(text: str) -> list[str]:
    """Return distinct overlapping two-character CJK substrings.

    A one-character run yields that character, since no bigram exists. Runs are
    processed independently, so punctuation between Chinese phrases does not
    create a cross-boundary term. The result is not a linguistic segmentation;
    it is a recall-oriented substring tokenisation used by the fallback SQL.
    """
    terms: list[str] = []
    for run in CJK_RUN_RE.findall(text):
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    return list(dict.fromkeys(terms))


def _rows_to_hits(rows: Sequence[Sequence[Any]]) -> list[SearchHit]:
    """Map the SELECT's column order to validated :class:`SearchHit` objects."""
    hits: list[SearchHit] = []
    for rank, row in enumerate(rows, start=1):
        score = float(row[5])
        if not math.isfinite(score) or score < 0:
            raise RetrievalError(
                f"database returned an invalid sparse score for chunk {row[0]!r}: {score}"
            )
        hits.append(
            SearchHit(
                chunk_id=row[0],
                doc_id=row[1],
                chunk_strategy=row[2],
                chunk_index=row[3],
                content=row[4],
                score=score,
                distance=None,
                rank=rank,
                source="sparse",
                title=row[6],
                category=row[7],
                card_tier=row[8],
                effective_year=row[9],
                policy_code=row[10],
            )
        )
    return hits
