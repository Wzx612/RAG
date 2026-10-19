"""Historical RRF candidate replay for the RAG evaluation.

The live chain needs the embedding model and the cross-encoder in one process.
On this host that combination crashed with a native segmentation fault after
dense and sparse retrieval returned, so the formal Phase 8 evaluation cannot
claim a live embedding+reranker run. This module provides an explicit
alternative that keeps the accepted retrieval artifacts and the accepted
reranker:

* candidate identities and RRF ranks come from the accepted Phase 6B artifact
  (``reports/hybrid_benchmark.json``), not from a new retrieval experiment;
* the chunk text and metadata are read live from PostgreSQL, so the context is
  real database content;
* the accepted cross-encoder reranker is run live over that pool;
* results are labelled ``historical_rrf_replay``; they are not a live
  Dense/Sparse/RRF run.

Implemented in: Phase 8 (RAG end-to-end).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, cast

import psycopg

from src.config import PROJECT_ROOT
from src.retrieval import RetrievalError
from src.schema.retrieval import SearchHit

DEFAULT_HISTORICAL = PROJECT_ROOT / "reports" / "hybrid_benchmark.json"
"""Accepted Phase 6B artifact with the RRF Top-10 pool for every query."""

ConnectionFactory = Callable[[], AbstractContextManager[psycopg.Connection]]


class HistoricalRRFRetriever:
    """Replay a query's accepted RRF pool, joined to live database text.

    The pool is loaded lazily per query and cached. A missing query, missing
    chunk or document-id mismatch raises :class:`~src.retrieval.RetrievalError`;
    the pipeline therefore reports a retrieval failure instead of inventing a
    candidate.
    """

    name = "historical-rrf"

    def __init__(
        self,
        path: Path = DEFAULT_HISTORICAL,
        *,
        candidates: int = 10,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        """Load and validate the artifact; do not query the database yet.

        Args:
            path: Accepted Phase 6B JSON artifact.
            candidates: Maximum pool size taken from each query's RRF list.
            connection_factory: Injectable context manager for tests.
        """
        if candidates < 1:
            raise ValueError("candidates must be positive")
        self._path = path
        self._candidates = candidates
        factory = connection_factory
        if factory is None:
            from src.db import connection as default_connection

            factory = cast(ConnectionFactory, default_connection)

        self._connection_factory: ConnectionFactory = factory
        self._records = _load_records(path)
        self._by_query = {
            str(record["query"]).strip(): record for record in self._records
        }
        if len(self._by_query) != len(self._records):
            raise ValueError("historical artifact contains duplicate query text")
        self._cache: dict[str, list[SearchHit]] = {}

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return the replayed, reranker-ready pool for ``query``.

        Args:
            query: Exact evaluation query text.
            top_k: Optional truncation. The RAG chain normally requests the
                configured candidate count.

        Raises:
            RetrievalError: unknown query, missing database text or an
                identity/doc mismatch.
        """
        normalized = query.strip() if isinstance(query, str) else ""
        record = self._by_query.get(normalized)
        if record is None:
            raise RetrievalError(
                "query is not present in the historical RRF artifact; "
                "historical mode only covers the accepted 30 evaluation queries"
            )
        cached = self._cache.get(normalized)
        hits = cached if cached is not None else self._build(record)
        if cached is None:
            self._cache[normalized] = hits
        limit = self._candidates if top_k is None else min(top_k, self._candidates)
        return list(hits[:limit])

    def _build(self, record: Mapping[str, Any]) -> list[SearchHit]:
        """Join historical identities to live text and build ordered hits."""
        try:
            pool = record["arms"]["rrf"]["hits"]
        except (KeyError, TypeError) as exc:
            raise RetrievalError("historical artifact has no RRF hit list") from exc
        if not isinstance(pool, list) or not pool:
            raise RetrievalError("historical RRF pool is empty")
        selected = pool[: self._candidates]
        identities = [
            (str(item["chunk_strategy"]), str(item["chunk_id"])) for item in selected
        ]
        texts = _fetch_texts(identities, self._connection_factory)
        missing = [identity for identity in identities if identity not in texts]
        if missing:
            raise RetrievalError(f"database has no text for historical chunks: {missing}")
        hits: list[SearchHit] = []
        for item in selected:
            text = texts[(str(item["chunk_strategy"]), str(item["chunk_id"]))]
            if str(item["doc_id"]) != str(text["doc_id"]):
                raise RetrievalError(
                    f"chunk {item['chunk_id']} has doc_id {text['doc_id']} in the "
                    f"database but {item['doc_id']} in the historical artifact"
                )
            hits.append(
                SearchHit(
                    chunk_id=str(item["chunk_id"]),
                    doc_id=str(item["doc_id"]),
                    score=float(item["score"]),
                    distance=item.get("distance"),
                    rank=int(item["rank"]),
                    source="fused",
                    dense_rank=item.get("dense_rank"),
                    sparse_rank=item.get("sparse_rank"),
                    dense_score=item.get("dense_score"),
                    sparse_score=item.get("sparse_score"),
                    rrf_score=item.get("rrf_score"),
                    candidate_sources=tuple(item.get("candidate_sources", ())),
                    content=text["content"],
                    chunk_strategy=str(text["chunk_strategy"]),
                    chunk_index=int(text["chunk_index"]),
                    title=text["title"],
                    category=text["category"],
                    card_tier=text["card_tier"],
                    effective_year=text["effective_year"],
                    policy_code=text["policy_code"],
                )
            )
        return hits


def _load_records(path: Path) -> list[dict[str, Any]]:
    """Read and minimally validate the accepted artifact."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read historical artifact {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must hold a JSON object")
    records = payload.get("per_query")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} has no per_query records")
    for position, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("query"), str):
            raise ValueError(f"{path} per_query[{position}] has no query text")
        arms = record.get("arms")
        if not isinstance(arms, dict) or not isinstance(arms.get("rrf"), dict):
            raise ValueError(f"{path} per_query[{position}] has no RRF arm")
    return records


def _fetch_texts(
    identities: list[tuple[str, str]],
    connection_factory: ConnectionFactory,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Read the real chunk text for the historical identities."""
    if not identities:
        return {}
    placeholders = ", ".join(["(%s, %s)"] * len(identities))
    parameters = [value for identity in identities for value in identity]
    sql = (
        "SELECT chunk_strategy, chunk_id, doc_id, chunk_index, content, title, "
        "category, card_tier, effective_year, policy_code "
        "FROM chunks WHERE (chunk_strategy, chunk_id) IN (" + placeholders + ")"
    )
    with connection_factory() as conn:
        rows = conn.execute(sql, parameters).fetchall()
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        result[(str(row[0]), str(row[1]))] = {
            "chunk_strategy": str(row[0]),
            "chunk_id": str(row[1]),
            "doc_id": str(row[2]),
            "chunk_index": int(row[3]),
            "content": row[4],
            "title": row[5],
            "category": row[6],
            "card_tier": row[7],
            "effective_year": row[8],
            "policy_code": row[9],
        }
    return result
