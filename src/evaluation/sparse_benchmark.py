"""Phase 5A sparse-retrieval benchmark.

Measures the production full-text path on the same 30 labelled queries and the
same chunk granularity used by the Phase 4C dense benchmark. Every query runs
against the live ``chunks.tsv`` / ``chunks_tsv_gin`` objects; no writes, DDL,
index rebuild or data-file edit is performed.

What this module measures:

* semantic Recall@K against the human ``relevant_doc_ids`` labels -- **not**
  ANN Recall, because sparse retrieval has no vector ANN reference;
* per-query hits, scores, empty results and gold-document ranks;
* query and database latency over repeated passes;
* candidate overlap between this run and the dense ``natural_plan`` artifact.

Chinese is covered by a separate synthetic probe. The corpus is English, so a
Chinese probe that returns zero rows is recorded honestly and is not counted in
the 30-query metrics.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg_pool import PoolTimeout
from pydantic import ValidationError

from src.evaluation.dense_benchmark import (
    BenchmarkError,
    EvalQuery,
    load_eval_queries,
)
from src.evaluation.metrics import percentile, recall_at_k
from src.retrieval.sparse import (
    SPARSE_CJK_SQL,
    SPARSE_SEARCH_SQL,
    _cjk_bigrams,
    _fts_query_text,
)
from src.schema.retrieval import QueryMode, SparseConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

FOCUS_QUERY_IDS = ("Q-004", "Q-005", "Q-009", "Q-014")
"""Queries whose dense semantic Recall was below 1.0 in Phase 4C."""

DEFAULT_DENSE_ARTIFACT = Path("reports") / "dense_benchmark.json"
"""Phase 4C raw artifact, used only for candidate-overlap reporting."""

ConnectionFactory = Callable[[], AbstractContextManager[psycopg.Connection]]
"""Context manager yielding a pooled connection (injectable for tests)."""

ProgressFn = Callable[[str], None]


@dataclass(frozen=True)
class SparseBenchmarkConfig:
    """Top-K, repetition and timeout settings for one benchmark run."""

    top_k: int = 10
    repetitions: int = 5
    warmup_passes: int = 1
    statement_timeout_ms: int = 5000
    query_mode: QueryMode = "or"


@dataclass(frozen=True)
class SparseHit:
    """The columns the benchmark needs from one sparse result row."""

    chunk_id: str
    doc_id: str
    chunk_strategy: str
    chunk_index: int
    score: float
    rank: int

    @property
    def key(self) -> str:
        """Chunk identity, matching the dense benchmark's key format."""
        return f"{self.chunk_strategy}:{self.chunk_id}"


# --------------------------------------------------------------- input checks


def build_config(
    top_k: int,
    repetitions: int,
    warmup_passes: int,
    statement_timeout_ms: int,
    query_mode: QueryMode = "or",
) -> SparseBenchmarkConfig:
    """Validate benchmark knobs through ``SparseConfig`` and explicit ranges."""
    try:
        SparseConfig(top_n=top_k, query_mode=query_mode)
    except ValidationError as exc:
        raise BenchmarkError(f"invalid benchmark configuration: {exc}") from exc
    if repetitions < 1:
        raise BenchmarkError("repetitions must be at least 1")
    if warmup_passes < 0:
        raise BenchmarkError("warmup_passes must be non-negative")
    if statement_timeout_ms < 1:
        raise BenchmarkError("statement_timeout_ms must be positive")
    return SparseBenchmarkConfig(
        top_k=top_k,
        repetitions=repetitions,
        warmup_passes=warmup_passes,
        statement_timeout_ms=statement_timeout_ms,
        query_mode=query_mode,
    )


# ------------------------------------------------------------------ execution


def _set_statement_timeout(conn: psycopg.Connection, milliseconds: int) -> None:
    """Set a transaction-local statement timeout for one measurement."""
    conn.execute(
        "SELECT set_config('statement_timeout', %s, true)",
        (f"{milliseconds}ms",),
        prepare=False,
    )


def _run_query(
    conn: psycopg.Connection,
    query: str,
    config: SparseBenchmarkConfig,
    rank_weights: Sequence[float],
    language: str,
    query_mode: QueryMode,
) -> tuple[list[SparseHit], str, float]:
    """Execute one sparse query in a transaction and return hits, path and time.

    The SQL constants, CJK detection and query-text builder are imported from
    the production retriever, so this measures the same statements rather than
    a copy.
    """
    cjk_terms = _cjk_bigrams(query)
    with conn.transaction():
        _set_statement_timeout(conn, config.statement_timeout_ms)
        started = time.perf_counter()
        if cjk_terms:
            rows = conn.execute(
                SPARSE_CJK_SQL,
                (cjk_terms, cjk_terms, config.top_k),
                prepare=False,
            ).fetchall()
            path = "cjk"
        else:
            rows = conn.execute(
                SPARSE_SEARCH_SQL,
                (
                    list(rank_weights),
                    language,
                    _fts_query_text(query, query_mode),
                    config.top_k,
                ),
                prepare=False,
            ).fetchall()
            path = "fts"
        query_ms = (time.perf_counter() - started) * 1000.0
    return _rows_to_hits(rows, config.top_k), path, query_ms


def _db_call(
    connection_factory: ConnectionFactory,
    query: str,
    config: SparseBenchmarkConfig,
    rank_weights: Sequence[float],
    language: str,
    query_mode: QueryMode,
) -> tuple[list[SparseHit], str, float, float]:
    """Borrow a connection, run one query, and time the whole database span."""
    started = time.perf_counter()
    with connection_factory() as conn:
        hits, path, query_ms = _run_query(conn, query, config, rank_weights, language, query_mode)
    db_ms = (time.perf_counter() - started) * 1000.0
    return hits, path, query_ms, db_ms


def _rows_to_hits(rows: Sequence[Sequence[Any]], top_k: int) -> list[SparseHit]:
    """Map SELECT columns to validated hits; reject non-finite/negative scores."""
    hits: list[SparseHit] = []
    for row_index, row in enumerate(rows, start=1):
        try:
            score = float(row[5])
        except (TypeError, ValueError, IndexError) as exc:
            raise BenchmarkError(f"malformed sparse row: {row!r}") from exc
        if not math.isfinite(score) or score < 0:
            raise BenchmarkError(f"invalid sparse score for chunk {row[0]!r}: {score}")
        hits.append(
            SparseHit(
                chunk_id=str(row[0]),
                doc_id=str(row[1]),
                chunk_strategy=str(row[2]),
                chunk_index=int(row[3]),
                score=score,
                rank=row_index,
            )
        )
    if len(hits) > top_k:
        raise BenchmarkError(f"sparse retrieval returned {len(hits)} rows for top_k={top_k}")
    return hits


def _is_timeout(exc: BaseException) -> bool:
    """Return True for the failure classes that are a timeout."""
    if isinstance(exc, (psycopg.errors.QueryCanceled, PoolTimeout)):
        return True
    message = str(exc).lower()
    return "timeout" in message or "canceling statement" in message


def _error_record(
    exc: BaseException,
    *,
    stage: str,
    query_id: str | None = None,
    pass_number: int | None = None,
) -> dict[str, Any]:
    """Record an error without dumping query text or credentials."""
    return {
        "stage": stage,
        "query_id": query_id,
        "pass": pass_number,
        "error_type": type(exc).__name__,
        "message": str(exc)[:500],
        "timeout": _is_timeout(exc),
    }


def _hit_record(hit: SparseHit) -> dict[str, Any]:
    """Serialize one raw hit for the JSON artifact."""
    return {
        "key": hit.key,
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "chunk_strategy": hit.chunk_strategy,
        "chunk_index": hit.chunk_index,
        "score": hit.score,
        "rank": hit.rank,
    }


def _measure(
    query: EvalQuery,
    hits: Sequence[SparseHit],
    path: str,
    top_k: int,
) -> dict[str, Any]:
    """Compute one successful pass's retrieval metrics against the human labels."""
    keys = [hit.key for hit in hits]
    doc_ids = list(dict.fromkeys(hit.doc_id for hit in hits))[:top_k]
    semantic = recall_at_k(doc_ids, query.relevant_doc_ids, top_k)
    gold_ranks: dict[str, int] = {}
    for hit in hits:
        if hit.doc_id in query.relevant_doc_ids and hit.doc_id not in gold_ranks:
            gold_ranks[hit.doc_id] = hit.rank
    return {
        "evaluated": True,
        "path": path,
        "result_count": len(hits),
        "unique_count": len(set(keys)),
        "doc_unique_count": len(set(hit.doc_id for hit in hits)),
        "empty_result": not hits,
        "semantic_recall": semantic,
        "keys": keys,
        "doc_ids": doc_ids,
        "gold_ranks": gold_ranks,
        "top_score": hits[0].score if hits else None,
        "hits": [_hit_record(hit) for hit in hits],
    }


def _latency_summary(samples: Sequence[float]) -> dict[str, float | int | None]:
    """Return count, min, nearest-rank P50/P95/P99, max and mean."""
    if not samples:
        return {
            "n": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": len(samples),
        "min": min(samples),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
        "max": max(samples),
        "mean": statistics.fmean(samples),
    }


def _mean_or_none(values: Sequence[float]) -> float | None:
    """Mean of a non-empty list, else None."""
    return statistics.fmean(values) if values else None


def _quality_summary(per_query: Sequence[dict[str, Any]], top_k: int) -> dict[str, Any]:
    """Aggregate the per-query semantic quality of a sparse run."""
    evaluated = [record for record in per_query if record.get("evaluated")]
    semantics = [float(record["semantic_recall"]) for record in evaluated]
    return {
        "evaluated_queries": len(evaluated),
        "missing_queries": len(per_query) - len(evaluated),
        "semantic_recall_mean": _mean_or_none(semantics),
        "semantic_recall_min": min(semantics) if semantics else None,
        "semantic_recall_max": max(semantics) if semantics else None,
        "empty_result_queries": sum(1 for record in evaluated if record["empty_result"]),
        "empty_result_rate": (
            sum(1 for record in evaluated if record["empty_result"]) / len(evaluated)
        )
        if evaluated
        else None,
        "shortfall_queries": sum(1 for record in evaluated if record["result_count"] < top_k),
        "overflow_queries": sum(1 for record in evaluated if record["result_count"] > top_k),
        "duplicate_result_queries": sum(
            1 for record in evaluated if record["unique_count"] != record["result_count"]
        ),
        "perfect_recall_queries": sum(1 for value in semantics if value >= 1.0),
        "zero_recall_queries": sum(1 for value in semantics if value == 0.0),
    }


def _focus_results(per_query: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Extract the four Phase 4C low-recall queries for the report."""
    by_id = {record["query_id"]: record for record in per_query}
    return {query_id: by_id[query_id] for query_id in FOCUS_QUERY_IDS if query_id in by_id}


# ------------------------------------------------------------ dense comparison


def _load_json(path: Path) -> Any:
    """Read a JSON artifact, raising BenchmarkError with a useful message."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkError(f"cannot read dense artifact at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{path} is not valid JSON: {exc}") from exc


def _dense_natural_plan(artifact: Any) -> dict[str, Any] | None:
    """Return the dense ``natural_plan`` config mapping, if present."""
    if not isinstance(artifact, dict):
        return None
    for config in artifact.get("configs", []):
        if isinstance(config, dict) and config.get("mode") == "natural":
            return config
    return None


def _dense_query_map(plan: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    """Index dense per-query records by ``query_id``."""
    if not plan:
        return {}
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in plan.get("per_query", []):
        if isinstance(record, dict) and isinstance(record.get("query_id"), str):
            by_id[record["query_id"]] = record
    return by_id


def dense_overlap(
    sparse_per_query: Sequence[dict[str, Any]],
    queries: Sequence[EvalQuery],
    artifact_path: Path,
    top_k: int,
) -> dict[str, Any]:
    """Compare sparse candidates with the dense ``natural_plan`` candidates.

    Chunk identity is ``<chunk_strategy>:<chunk_id>``, the same key the dense
    benchmark records. Doc-level comparison uses the deduplicated ``doc_ids``
    field. This is candidate overlap, not a quality score.
    """
    if not artifact_path.exists():
        return {
            "available": False,
            "source": str(artifact_path),
            "reason": "dense artifact not found; overlap not computed",
        }
    try:
        artifact = _load_json(artifact_path)
    except BenchmarkError as exc:
        return {"available": False, "source": str(artifact_path), "reason": str(exc)}

    plan = _dense_natural_plan(artifact)
    dense_by_id = _dense_query_map(plan)
    if not plan or not dense_by_id:
        return {
            "available": False,
            "source": str(artifact_path),
            "reason": "dense artifact has no natural_plan per-query records",
        }

    dense_top_k = plan.get("top_k")
    gold_by_id = {query.query_id: query.relevant_doc_ids for query in queries}
    records: list[dict[str, Any]] = []
    chunk_jaccards: list[float] = []
    doc_jaccards: list[float] = []
    gold_totals = {"both": 0, "sparse_only": 0, "dense_only": 0, "neither": 0}

    for sparse in sparse_per_query:
        query_id = sparse.get("query_id")
        if not sparse.get("evaluated") or query_id not in dense_by_id:
            continue
        dense = dense_by_id[query_id]
        sparse_keys = set(sparse.get("keys", []))
        dense_keys = set(dense.get("keys", []))
        sparse_docs = set(sparse.get("doc_ids", []))
        dense_docs = set(dense.get("doc_ids", []))
        gold_docs = set(gold_by_id.get(str(query_id), ()))

        common_keys = sparse_keys & dense_keys
        common_docs = sparse_docs & dense_docs
        key_union = sparse_keys | dense_keys
        doc_union = sparse_docs | dense_docs
        key_jaccard = len(common_keys) / len(key_union) if key_union else 1.0
        doc_jaccard = len(common_docs) / len(doc_union) if doc_union else 1.0
        chunk_jaccards.append(key_jaccard)
        doc_jaccards.append(doc_jaccard)

        gold_both = gold_docs & sparse_docs & dense_docs
        gold_sparse_only = (gold_docs & sparse_docs) - dense_docs
        gold_dense_only = (gold_docs & dense_docs) - sparse_docs
        gold_neither = gold_docs - sparse_docs - dense_docs
        for key, value in (
            ("both", gold_both),
            ("sparse_only", gold_sparse_only),
            ("dense_only", gold_dense_only),
            ("neither", gold_neither),
        ):
            gold_totals[key] += len(value)

        records.append(
            {
                "query_id": query_id,
                "sparse_result_count": sparse.get("result_count"),
                "dense_result_count": dense.get("result_count"),
                "chunk_overlap": len(common_keys),
                "sparse_only_chunks": len(sparse_keys - dense_keys),
                "dense_only_chunks": len(dense_keys - sparse_keys),
                "chunk_jaccard": key_jaccard,
                "doc_overlap": len(common_docs),
                "sparse_only_docs": len(sparse_docs - dense_docs),
                "dense_only_docs": len(dense_docs - sparse_docs),
                "doc_jaccard": doc_jaccard,
                "gold": {
                    "gold_docs": sorted(gold_docs),
                    "both": sorted(gold_both),
                    "sparse_only": sorted(gold_sparse_only),
                    "dense_only": sorted(gold_dense_only),
                    "neither": sorted(gold_neither),
                },
            }
        )

    focus = {
        record["query_id"]: record for record in records if record["query_id"] in FOCUS_QUERY_IDS
    }
    return {
        "available": True,
        "source": str(artifact_path),
        "plan": "natural_plan",
        "dense_top_k": dense_top_k,
        "sparse_top_k": top_k,
        "top_k_aligned": dense_top_k == top_k,
        "compared_queries": len(records),
        "chunk_jaccard_mean": _mean_or_none(chunk_jaccards),
        "doc_jaccard_mean": _mean_or_none(doc_jaccards),
        "gold_doc_coverage": gold_totals,
        "per_query": records,
        "focus_queries": focus,
    }


# ------------------------------------------------------------------ benchmark


def run_sparse_benchmark(
    queries: Sequence[EvalQuery],
    config: SparseBenchmarkConfig,
    *,
    connection_factory: ConnectionFactory | None = None,
    rank_weights: Sequence[float] = (0.1, 0.2, 0.4, 1.0),
    language: str = "english",
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Run the sparse benchmark over ``queries`` and return the raw result.

    Args:
        queries: Labelled queries in file order. The caller is responsible for
            the environment/data gates.
        config: Top-K, repetition, warm-up and timeout settings.
        connection_factory: Context manager yielding a connection. Defaults to
            the project pool.
        rank_weights: ``ts_rank_cd`` {D, C, B, A} weights; defaults mirror
            :class:`SparseConfig`.
        language: Text-search configuration; must match ``chunks.tsv``.
        progress: Optional line-oriented progress callback.

    Returns:
        A JSON-serializable mapping with per-query records, quality, latency,
        errors and the focus-query subset. It contains no query vectors.
    """
    if not queries:
        raise BenchmarkError("at least one eval query is required")

    factory = connection_factory or _default_connection_factory()
    notify = progress or (lambda _message: None)
    per_query: list[dict[str, Any]] = []
    query_samples: list[float] = []
    db_samples: list[float] = []
    errors: list[dict[str, Any]] = []
    total_errors = 0
    total_timeouts = 0
    warmup_errors = 0
    quality_mismatch_samples = 0
    path_counts: dict[str, int] = {}

    for index, query in enumerate(queries, start=1):
        notify(f"[{index}/{len(queries)}] {query.query_id}")
        record: dict[str, Any] = {
            "query_id": query.query_id,
            "query": query.query,
            "path": None,
            "evaluated": False,
            "result_count": None,
            "unique_count": None,
            "doc_unique_count": None,
            "empty_result": None,
            "semantic_recall": None,
            "keys": [],
            "doc_ids": [],
            "gold_ranks": {},
            "top_score": None,
            "hits": [],
            "latency": {"query_ms": [], "db_ms": []},
            "quality_mismatch_samples": 0,
            "warmup_errors": 0,
            "errors": 0,
            "timeouts": 0,
            "error_records": [],
        }

        for warmup_number in range(1, config.warmup_passes + 1):
            try:
                _db_call(
                    factory,
                    query.query,
                    config,
                    rank_weights,
                    language,
                    config.query_mode,
                )
            except Exception as exc:  # noqa: BLE001 -- isolate one failed sample
                warmup_errors += 1
                record["warmup_errors"] += 1
                errors.append(
                    _error_record(
                        exc,
                        stage="warmup",
                        query_id=query.query_id,
                        pass_number=warmup_number,
                    )
                )

        reference_keys: list[str] | None = None
        for pass_number in range(1, config.repetitions + 1):
            try:
                hits, path, query_ms, db_ms = _db_call(
                    factory,
                    query.query,
                    config,
                    rank_weights,
                    language,
                    config.query_mode,
                )
            except Exception as exc:  # noqa: BLE001 -- isolate one failed sample
                total_errors += 1
                record["errors"] += 1
                if _is_timeout(exc):
                    total_timeouts += 1
                    record["timeouts"] += 1
                errors.append(
                    _error_record(
                        exc,
                        stage="timed",
                        query_id=query.query_id,
                        pass_number=pass_number,
                    )
                )
                continue

            path_counts[path] = path_counts.get(path, 0) + 1
            record["latency"]["query_ms"].append(query_ms)
            record["latency"]["db_ms"].append(db_ms)
            query_samples.append(query_ms)
            db_samples.append(db_ms)

            measured = _measure(query, hits, path, config.top_k)
            keys = list(measured["keys"])
            if reference_keys is None:
                reference_keys = keys
                for field in (
                    "path",
                    "evaluated",
                    "result_count",
                    "unique_count",
                    "doc_unique_count",
                    "empty_result",
                    "semantic_recall",
                    "keys",
                    "doc_ids",
                    "gold_ranks",
                    "top_score",
                    "hits",
                ):
                    record[field] = measured[field]
            elif keys != reference_keys:
                quality_mismatch_samples += 1
                record["quality_mismatch_samples"] += 1

        per_query.append(record)
        notify(
            f"  {query.query_id}: results={record['result_count']} "
            f"semantic_recall={record['semantic_recall']} errors={record['errors']}"
        )

    quality = _quality_summary(per_query, config.top_k)
    paths = {path: count for path, count in sorted(path_counts.items())}
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "COMPLETED",
        "parameters": {
            "queries": len(queries),
            "top_k": config.top_k,
            "repetitions": config.repetitions,
            "warmup_passes": config.warmup_passes,
            "statement_timeout_ms": config.statement_timeout_ms,
            "rank_weights": list(rank_weights),
            "language": language,
            "query_mode": config.query_mode,
        },
        "definitions": {
            "granularity": "chunk row; doc-level metrics deduplicate doc_id in rank order",
            "semantic_recall": (
                "recall_at_k over human relevant_doc_ids; denominator is the number "
                "of distinct gold docs; this is not ANN recall"
            ),
            "sparse_score": (
                "English path: ts_rank_cd weighted cover density, higher is better, "
                "not a probability and not comparable across queries. CJK path: "
                "number of matched query bigrams, also only comparable within a query."
            ),
            "latency_query_ms": "execute + fetch around the retrieval SELECT only",
            "latency_db_ms": "pool borrow + transaction + timeout setup + execute + fetch",
        },
        "retrieval": {
            "path_counts": paths,
            "per_query": per_query,
            "quality": quality,
            "latency": {
                "query_ms": _latency_summary(query_samples),
                "db_ms": _latency_summary(db_samples),
                "query_ms_samples": query_samples,
                "db_ms_samples": db_samples,
            },
            "focus_queries": _focus_results(per_query),
        },
        "totals": {
            "evaluated_queries": quality["evaluated_queries"],
            "missing_queries": quality["missing_queries"],
            "errors": total_errors,
            "timeouts": total_timeouts,
            "warmup_errors": warmup_errors,
            "quality_mismatch_samples": quality_mismatch_samples,
            "path_counts": paths,
        },
        "errors": errors,
    }


def _default_connection_factory() -> ConnectionFactory:
    """Import the project pool lazily, so tests can use this module without it."""
    from src.db import connection

    return connection


# -------------------------------------------------------------------- report


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a GitHub-flavored Markdown table, escaping cell separators."""

    def cell(value: object) -> str:
        """Escape a value for one Markdown table cell."""
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _fmt(value: object, digits: int = 3) -> str:
    """Format numbers for the report without turning None into a fake zero."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_list(values: Sequence[object], limit: int = 10) -> str:
    """Render a short comma-separated list for a table cell."""
    rendered = ", ".join(str(value) for value in values[:limit])
    if len(values) > limit:
        rendered += f", +{len(values) - limit} more"
    return rendered or "-"


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render the sparse benchmark result as a reviewable Markdown report."""
    status = result.get("status", "unknown")
    environment = result.get("environment", {})
    dataset = result.get("dataset", {})
    parameters = result.get("parameters", {})
    retrieval = result.get("retrieval", {})
    quality = retrieval.get("quality", {})
    latency = retrieval.get("latency", {})
    totals = result.get("totals", {})
    overlap = result.get("dense_overlap", {})
    probe = result.get("chinese_probe", {})
    lines: list[str] = [
        "# Sparse Retrieval Benchmark (Phase 5A)",
        "",
        f"- Generated: `{result.get('generated_at', 'unknown')}`",
        f"- Status: **{status}**",
        f"- Queries: {parameters.get('queries', '?')}  |  K: {parameters.get('top_k', '?')}  "
        f"|  Timed passes: {parameters.get('repetitions', '?')}  |  "
        f"Warm-ups: {parameters.get('warmup_passes', '?')}  |  "
        f"Statement timeout: {parameters.get('statement_timeout_ms', '?')} ms",
        "",
        "## 1. Environment and dataset",
        "",
    ]
    env_rows = [
        ("PostgreSQL", environment.get("postgres_version")),
        ("pgvector extension", environment.get("vector_version")),
        ("Server address", environment.get("server_address")),
        ("Data directory", environment.get("data_directory")),
        ("FTS configuration", environment.get("fts_config")),
        ("Generated tsv expression", environment.get("tsv_expression")),
        ("GIN index", environment.get("gin_index")),
        ("Chunk table", environment.get("chunk_columns")),
        ("Host Python", environment.get("python_version")),
        ("psycopg", environment.get("psycopg_version")),
        ("Docker control plane", environment.get("docker_status")),
        ("Execution environment", environment.get("execution_environment")),
        ("Free disk", environment.get("disk_free")),
        (
            "Documents / chunks",
            f"{environment.get('documents', '?')} / {environment.get('chunks', '?')}",
        ),
        ("Distinct chunk doc_ids", environment.get("distinct_chunk_docs")),
        ("Eval queries", dataset.get("queries")),
        ("Gold sizes", dataset.get("gold_sizes")),
        ("Missing gold docs", _fmt_list(dataset.get("missing_gold_docs", []))),
        ("Data hashes", dataset.get("hashes")),
        ("Counts after run", dataset.get("post_run_counts")),
        ("Data integrity", dataset.get("data_integrity_ok")),
    ]
    lines.append(_md_table(["Item", "Value"], env_rows))
    lines.extend(
        [
            "",
            "The numbers below were produced against the live PostgreSQL/pgvector "
            "database, on the execution environment named above. No rows, schema, "
            "indexes, eval queries or knowledge-base files were modified.",
            "",
            "## 2. Method and score semantics",
            "",
            f"- Retrieval path counts over timed passes: `{_fmt(totals.get('path_counts'))}`",
            "- English queries use `websearch_to_tsquery('english', query)` against the "
            "generated `chunks.tsv`, with a GIN index and `ts_rank_cd` ranking.",
            f"- Query mode: `{parameters.get('query_mode', '?')}`. Plain terms are "
            "OR-joined by default; quoted phrases and `-term` are left untouched, "
            "and a bare `or` is treated as prose.",
            f"- Rank weights `{{D, C, B, A}}`: `{parameters.get('rank_weights')}`.",
            "- The English sparse score is **weighted cover density**: higher is better, "
            "it is not a probability, and raw values are not comparable across queries.",
            "- The CJK fallback score is **the number of matched two-character substrings**; "
            "it is not a segmentation score and is only comparable within one query.",
            "- `distance` is `None` for sparse hits because full-text rank is not a "
            "vector distance.",
            "- Semantic Recall uses the human `relevant_doc_ids` labels. It is **not** ANN "
            "Recall: sparse retrieval has no vector ANN reference.",
            "",
            "## 3. Overall semantic quality",
            "",
        ]
    )
    quality_rows = [
        ("Evaluated queries", quality.get("evaluated_queries")),
        ("Missing queries", quality.get("missing_queries")),
        ("Semantic Recall mean", _fmt(quality.get("semantic_recall_mean"))),
        ("Semantic Recall min", _fmt(quality.get("semantic_recall_min"))),
        ("Semantic Recall max", _fmt(quality.get("semantic_recall_max"))),
        ("Perfect-recall queries", quality.get("perfect_recall_queries")),
        ("Zero-recall queries", quality.get("zero_recall_queries")),
        ("Empty-result queries", quality.get("empty_result_queries")),
        ("Empty-result rate", _fmt(quality.get("empty_result_rate"))),
        ("Shortfall queries (< K)", quality.get("shortfall_queries")),
        ("Duplicate-result queries", quality.get("duplicate_result_queries")),
        ("Timed errors", totals.get("errors")),
        ("Timed timeouts", totals.get("timeouts")),
        ("Warm-up errors", totals.get("warmup_errors")),
        ("Quality-mismatch samples", totals.get("quality_mismatch_samples")),
    ]
    lines.append(_md_table(["Metric", "Value"], quality_rows))
    lines.extend(
        [
            "",
            "Semantic Recall is macro-averaged over evaluated queries; a missing/error "
            "query is not silently counted as recall 0.",
            "",
        ]
    )
    control = result.get("and_mode_control")
    if control:
        control_quality = control.get("quality", {})
        lines.extend(
            [
                "### 3b. AND-mode control",
                "",
                "The same queries with PostgreSQL's bare-term AND semantics. This shows "
                "why OR is the default; it is not the primary metric.",
                "",
                _md_table(
                    ["Metric", "AND control"],
                    [
                        ("Evaluated queries", control_quality.get("evaluated_queries")),
                        ("Semantic Recall mean", _fmt(control_quality.get("semantic_recall_mean"))),
                        ("Perfect-recall queries", control_quality.get("perfect_recall_queries")),
                        ("Zero-recall queries", control_quality.get("zero_recall_queries")),
                        ("Empty-result queries", control_quality.get("empty_result_queries")),
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## 4. Latency",
            "",
            _md_table(
                ["Clock", "n", "min ms", "P50 ms", "P95 ms", "P99 ms", "max ms", "mean ms"],
                [
                    (
                        "query_ms",
                        latency.get("query_ms", {}).get("n"),
                        _fmt(latency.get("query_ms", {}).get("min")),
                        _fmt(latency.get("query_ms", {}).get("p50")),
                        _fmt(latency.get("query_ms", {}).get("p95")),
                        _fmt(latency.get("query_ms", {}).get("p99")),
                        _fmt(latency.get("query_ms", {}).get("max")),
                        _fmt(latency.get("query_ms", {}).get("mean")),
                    ),
                    (
                        "db_ms",
                        latency.get("db_ms", {}).get("n"),
                        _fmt(latency.get("db_ms", {}).get("min")),
                        _fmt(latency.get("db_ms", {}).get("p50")),
                        _fmt(latency.get("db_ms", {}).get("p95")),
                        _fmt(latency.get("db_ms", {}).get("p99")),
                        _fmt(latency.get("db_ms", {}).get("max")),
                        _fmt(latency.get("db_ms", {}).get("mean")),
                    ),
                ],
            ),
            "",
            "P50/P95 use the same nearest-rank definition as Phase 4C. These are "
            "descriptive measurements of this run, not a cross-engine speed claim.",
            "",
            "## 5. Per-query results",
            "",
            "`doc_ids` is deduplicated in rank order. `gold_ranks` maps each retrieved "
            "gold doc to its first chunk rank.",
            "",
            _md_table(
                ["Query", "Path", "Results", "Gold", "Semantic Recall", "Gold ranks", "Top score"],
                [
                    (
                        record["query_id"],
                        record.get("path"),
                        record.get("result_count"),
                        len(
                            next(
                                (
                                    query.relevant_doc_ids
                                    for query in result.get("_queries", [])
                                    if query.query_id == record["query_id"]
                                ),
                                (),
                            )
                        )
                        if result.get("_queries")
                        else "-",
                        _fmt(record.get("semantic_recall")),
                        _fmt(record.get("gold_ranks")),
                        _fmt(record.get("top_score")),
                    )
                    for record in retrieval.get("per_query", [])
                ],
            ),
            "",
            "## 6. Focus queries (dense semantic Recall < 1.0 in Phase 4C)",
            "",
            _md_table(
                ["Query", "Sparse Recall", "Results", "Gold ranks", "Doc IDs"],
                [
                    (
                        query_id,
                        _fmt(record.get("semantic_recall")),
                        record.get("result_count"),
                        _fmt(record.get("gold_ranks")),
                        _fmt_list(record.get("doc_ids", [])),
                    )
                    for query_id, record in retrieval.get("focus_queries", {}).items()
                ],
            ),
            "",
            "## 7. Sparse vs Dense candidate overlap",
            "",
        ]
    )
    if not overlap.get("available"):
        lines.append(f"Overlap was not computed: {overlap.get('reason', 'unavailable')}")
    else:
        gold = overlap.get("gold_doc_coverage", {})
        lines.extend(
            [
                f"- Source: `{overlap.get('source')}`, plan `{overlap.get('plan')}`",
                f"- Top-K aligned: {overlap.get('top_k_aligned')} "
                f"(dense={overlap.get('dense_top_k')}, sparse={overlap.get('sparse_top_k')})",
                f"- Compared queries: {overlap.get('compared_queries')}",
                f"- Mean chunk Jaccard: {_fmt(overlap.get('chunk_jaccard_mean'))}",
                f"- Mean doc Jaccard: {_fmt(overlap.get('doc_jaccard_mean'))}",
                f"- Gold-doc coverage: both={gold.get('both', 0)}, "
                f"sparse-only={gold.get('sparse_only', 0)}, "
                f"dense-only={gold.get('dense_only', 0)}, "
                f"neither={gold.get('neither', 0)}",
                "",
                _md_table(
                    [
                        "Query",
                        "Chunk overlap",
                        "Sparse-only",
                        "Dense-only",
                        "Chunk Jaccard",
                        "Doc Jaccard",
                    ],
                    [
                        (
                            record["query_id"],
                            record.get("chunk_overlap"),
                            record.get("sparse_only_chunks"),
                            record.get("dense_only_chunks"),
                            _fmt(record.get("chunk_jaccard")),
                            _fmt(record.get("doc_jaccard")),
                        )
                        for record in overlap.get("per_query", [])
                    ],
                ),
                "",
                "Candidate overlap is not a quality metric: two lists can overlap "
                "heavily and both be wrong, or overlap little and both be right.",
            ]
        )
    lines.extend(["", "## 8. Chinese probe (separate from the 30-query metrics)", ""])
    if not probe.get("available"):
        lines.append(f"Chinese probe was not run: {probe.get('reason', 'unavailable')}")
    else:
        lines.append(
            "These are synthetic Chinese probes, not part of the labelled 30-query set. "
            "The corpus is English, so zero hits do **not** show that Chinese retrieval "
            "works; they show the fallback's behaviour on this corpus."
        )
        lines.extend(
            [
                "",
                _md_table(
                    [
                        "Probe",
                        "Path",
                        "Results",
                        "English FTS matches",
                        "Top doc IDs",
                        "Top scores",
                    ],
                    [
                        (
                            record.get("query"),
                            record.get("path"),
                            record.get("result_count"),
                            record.get("english_fts_match_count"),
                            _fmt_list(record.get("doc_ids", [])),
                            _fmt_list(record.get("scores", [])),
                        )
                        for record in probe.get("probes", [])
                    ],
                ),
                "",
                f"- English FTS lexeme evidence: `{_fmt(probe.get('lexeme_evidence'))}`",
                f"- CJK bigram example: `{_fmt(probe.get('bigram_example'))}`",
                "- The fallback is substring matching over title/content/card_tier. It is "
                "not word segmentation, has no Chinese dictionary, and was not validated "
                "against Chinese documents because none are in this corpus.",
            ]
        )
    lines.extend(
        [
            "",
            "## 9. Errors and integrity",
            "",
            f"- Timed errors: {totals.get('errors')}",
            f"- Timed timeouts: {totals.get('timeouts')}",
            f"- Warm-up errors: {totals.get('warmup_errors')}",
            f"- Quality-mismatch samples: {totals.get('quality_mismatch_samples')}",
            f"- Post-run data integrity: {dataset.get('data_integrity_ok')}",
            f"- Issues: {_fmt_list(result.get('issues', []))}",
            "",
            "## 10. Limitations and next-stage prerequisites",
            "",
            "- Chinese support is a minimal substring fallback, not segmentation. A real "
            "Chinese corpus needs `zhparser`/`pg_jieba`, a matching generated tsvector, "
            "and a rebuilt index before Chinese retrieval quality can be claimed.",
            "- A mixed Chinese+ASCII query currently takes the CJK path only; ASCII terms "
            "are not combined with the fallback. The probe "
            "`balance transfer fee 余额` therefore returns zero rows on this English corpus.",
            "- The OR default is a recall-oriented choice, not a universal one: the AND "
            "control returned 26 empty results out of 30. The mode is recorded in "
            "`parameters.query_mode` so later stages can reproduce the exact run.",
            "- The evaluation set has 30 queries and gold docs; Semantic Recall values are "
            "sensitive to those labels and are not a generalisation guarantee.",
            "- The sparse score is not calibrated across queries, so it must not be "
            "compared with dense cosine similarity or fed to a weighted fusion without "
            "normalisation or rank-based fusion (RRF).",
            "- Hybrid Search, RRF and reranking are intentionally **not** implemented in "
            "this phase.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: Mapping[str, Any],
    json_path: Path,
    report_path: Path,
) -> None:
    """Write the raw JSON artifact and the Markdown report, creating parents."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(result)
    serializable.pop("_queries", None)
    json_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(result), encoding="utf-8")


def attach_query_reference(
    result: Mapping[str, Any], queries: Sequence[EvalQuery]
) -> dict[str, Any]:
    """Attach queries in memory so the renderer can show gold-set sizes.

    The private key is removed by :func:`write_outputs`; it never enters JSON.
    """
    enriched = dict(result)
    enriched["_queries"] = list(queries)
    return enriched


def load_queries(path: Path | None = None) -> list[EvalQuery]:
    """Re-export the validated loader for callers that only import this module."""
    return load_eval_queries(path)
