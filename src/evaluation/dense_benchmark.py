"""Phase 4C dense-retrieval benchmark.

Runs the production ``DENSE_SEARCH_SQL`` in three controlled modes against the
same query vectors, Top-K and chunk granularity:

* ``exact`` -- a transaction-local planner override forces a sequential scan of
  ``chunks``; this is the exact cosine Top-K reference.
* ``natural`` -- no planner override, matching what production actually does.
* ``hnsw`` -- a transaction-local ``SET LOCAL enable_seqscan = off`` forces the
  HNSW index path, then sweeps ``hnsw.ef_search``.

The module only measures and records. It computes ANN Recall@K against the
exact reference, not semantic retrieval quality against the human labels; that
second number is recorded separately from the labelled doc ids. No writes, DDL,
or global GUC changes are performed.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import psycopg
from pgvector import Vector
from psycopg_pool import PoolTimeout
from pydantic import ValidationError

from src.config import PROJECT_ROOT
from src.db import (
    EF_SEARCH_DEFAULT,
    connection,
    current_ef_search,
    ef_search,
)
from src.evaluation.metrics import percentile, recall_at_k
from src.retrieval.dense import DENSE_SEARCH_SQL
from src.schema.retrieval import DenseConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

HNSW_INDEX_NAME = "chunks_embedding_hnsw_idx"
"""The Phase 2 cosine HNSW index. Its presence in EXPLAIN proves the ANN path."""

DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "eval_queries.json"

PLAN_LINE_LIMIT = 300
"""Long EXPLAIN lines can contain a bound vector; keep the report readable."""

Mode = Literal["exact", "natural", "hnsw"]

ConnectionFactory = Callable[[], AbstractContextManager[psycopg.Connection]]
"""Context manager yielding a pooled connection (injectable for tests)."""

ProgressFn = Callable[[str], None]


class BenchmarkError(RuntimeError):
    """The benchmark cannot produce a trustworthy measurement."""


@dataclass(frozen=True)
class EvalQuery:
    """One labelled evaluation query, without its free-text answer."""

    query_id: str
    query: str
    relevant_doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkConfig:
    """One benchmark arm."""

    name: str
    mode: Mode
    ef_search: int
    top_k: int
    description: str


@dataclass(frozen=True)
class Hit:
    """The columns the benchmark needs from one retrieval row."""

    chunk_id: str
    doc_id: str
    chunk_strategy: str
    chunk_index: int
    distance: float

    @property
    def key(self) -> str:
        """Chunk identity. A chunk is identified by strategy plus chunk id."""
        return f"{self.chunk_strategy}:{self.chunk_id}"


# ------------------------------------------------------------------- data shape


def load_eval_queries(path: Path | None = None) -> list[EvalQuery]:
    """Read and validate the labelled query set without modifying it.

    Args:
        path: Override the default ``data/eval_queries.json``.

    Returns:
        Queries in file order, each with a non-empty text and gold doc ids.

    Raises:
        BenchmarkError: the file is unreadable, malformed, empty, has duplicate
            query ids, or a record lacks a query / relevant doc ids.
    """
    path = path or DEFAULT_EVAL_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkError(f"cannot read eval queries at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, list) or not payload:
        raise BenchmarkError(f"{path} must hold a non-empty JSON array")

    queries: list[EvalQuery] = []
    seen: set[str] = set()
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise BenchmarkError(f"eval query #{index} is {type(entry).__name__}, expected object")
        query_id = entry.get("query_id")
        query = entry.get("query")
        gold = entry.get("relevant_doc_ids")
        if not isinstance(query_id, str) or not query_id.strip():
            raise BenchmarkError(f"eval query #{index} has no usable query_id")
        if query_id in seen:
            raise BenchmarkError(f"duplicate eval query_id {query_id!r}")
        seen.add(query_id)
        if not isinstance(query, str) or not query.strip():
            raise BenchmarkError(f"eval query {query_id} has an empty query")
        if not isinstance(gold, list) or not gold:
            raise BenchmarkError(f"eval query {query_id} has no relevant_doc_ids")
        if any(not isinstance(doc_id, str) or not doc_id for doc_id in gold):
            raise BenchmarkError(f"eval query {query_id} has a non-string gold doc id")
        unique_gold = tuple(dict.fromkeys(gold))
        queries.append(
            EvalQuery(query_id=query_id, query=query.strip(), relevant_doc_ids=unique_gold)
        )

    return queries


def build_configs(top_k: int, ef_values: Sequence[int]) -> list[BenchmarkConfig]:
    """Build the exact, natural and HNSW arms for one controlled experiment.

    Args:
        top_k: Top-K shared by every arm. It is validated through
            :class:`DenseConfig`, the same model production uses.
        ef_values: Distinct ``hnsw.ef_search`` values for the HNSW arms. Values
            below ``top_k`` are legal but can truncate the HNSW result; the
            report exposes that shortfall rather than hiding it.

    Returns:
        ``exact_seqscan``, ``natural_plan``, then one ``hnsw_ef_<value>`` arm per
        distinct value, in the order supplied.

    Raises:
        BenchmarkError: ``top_k`` or an ``ef_search`` value is outside the
            accepted ``DenseConfig`` range, or ``ef_values`` is empty.
    """
    values = list(dict.fromkeys(ef_values))
    if not values:
        raise BenchmarkError("at least one ef_search value is required")
    try:
        DenseConfig(top_n=top_k)
        for value in values:
            DenseConfig(top_n=top_k, ef_search=value)
    except ValidationError as exc:
        raise BenchmarkError(f"invalid benchmark configuration: {exc}") from exc

    configs = [
        BenchmarkConfig(
            name="exact_seqscan",
            mode="exact",
            ef_search=EF_SEARCH_DEFAULT,
            top_k=top_k,
            description="Exact cosine Top-K via a forced Seq Scan on chunks.",
        ),
        BenchmarkConfig(
            name="natural_plan",
            mode="natural",
            ef_search=EF_SEARCH_DEFAULT,
            top_k=top_k,
            description="Production path: no planner forcing, default ef_search.",
        ),
    ]
    configs.extend(
        BenchmarkConfig(
            name=f"hnsw_ef_{value}",
            mode="hnsw",
            ef_search=value,
            top_k=top_k,
            description=f"Forced HNSW path at hnsw.ef_search={value}.",
        )
        for value in values
    )
    return configs


# --------------------------------------------------------------------- planning


def classify_plan(plan: str) -> str:
    """Return the ANN stage's plan shape from an EXPLAIN text."""
    if HNSW_INDEX_NAME in plan:
        return "hnsw"
    if "Seq Scan on chunks" in plan:
        return "seq_scan"
    if "Index Only Scan" in plan or "Index Scan" in plan:
        return "other_index"
    return "other"


def sanitize_plan(plan: str) -> str:
    """Drop the bound query vector from EXPLAIN output and bound line length."""
    lines: list[str] = []
    for line in plan.splitlines():
        if "Sort Key:" in line:
            lines.append("  Sort Key: (omitted: contains the 1024-dim query vector)")
        elif "embedding <=> '" in line:
            prefix = line.split("embedding <=> '", 1)[0]
            lines.append(prefix + "embedding <=> (query vector omitted))")
        elif len(line) > PLAN_LINE_LIMIT:
            lines.append(line[: PLAN_LINE_LIMIT - 3] + "...")
        else:
            lines.append(line)
    return "\n".join(lines)


def _execution_ms(plan: str) -> float | None:
    """Extract ``Execution Time`` from EXPLAIN ANALYZE, if present."""
    match = re.search(r"Execution Time:\s+([0-9.]+)\s+ms", plan)
    return float(match.group(1)) if match else None


def _configure_planner(conn: psycopg.Connection, mode: Mode) -> None:
    """Apply transaction-local planner overrides for one measurement.

    The exact arm disables every index path available to ``chunks``; the HNSW
    arm only makes a sequential scan unattractive. Both are ``SET LOCAL`` and
    therefore vanish when the enclosing transaction ends. No global setting is
    touched.
    """
    if mode == "exact":
        conn.execute("SET LOCAL enable_indexscan = off", prepare=False)
        conn.execute("SET LOCAL enable_bitmapscan = off", prepare=False)
        conn.execute("SET LOCAL enable_indexonlyscan = off", prepare=False)
        conn.execute("SET LOCAL enable_seqscan = on", prepare=False)
    elif mode == "hnsw":
        conn.execute("SET LOCAL enable_seqscan = off", prepare=False)
        conn.execute("SET LOCAL enable_indexscan = on", prepare=False)
        conn.execute("SET LOCAL enable_bitmapscan = on", prepare=False)


def _set_statement_timeout(conn: psycopg.Connection, milliseconds: int) -> None:
    """Set a transaction-local statement timeout for one measurement."""
    conn.execute(
        "SELECT set_config('statement_timeout', %s, true)",
        (f"{milliseconds}ms",),
        prepare=False,
    )


def _capture_plan(
    conn: psycopg.Connection,
    vector: Sequence[float],
    config: BenchmarkConfig,
    statement_timeout_ms: int,
) -> dict[str, Any]:
    """Run EXPLAIN ANALYZE and return a validated, sanitized plan record."""
    previous = conn.prepare_threshold
    conn.prepare_threshold = None
    try:
        with conn.transaction():
            _set_statement_timeout(conn, statement_timeout_ms)
            _configure_planner(conn, config.mode)
            with ef_search(conn, config.ef_search):
                rows = conn.execute(
                    "EXPLAIN (ANALYZE, BUFFERS) " + DENSE_SEARCH_SQL,
                    (Vector(list(vector)), config.top_k),
                    prepare=False,
                ).fetchall()
    finally:
        conn.prepare_threshold = previous

    plan = "\n".join(str(row[0]) for row in rows)
    kind = classify_plan(plan)
    uses_hnsw = HNSW_INDEX_NAME in plan
    uses_seq_scan = "Seq Scan on chunks" in plan
    record: dict[str, Any] = {
        "kind": kind,
        "uses_hnsw": uses_hnsw,
        "uses_seq_scan_on_chunks": uses_seq_scan,
        "execution_ms": _execution_ms(plan),
        "text": sanitize_plan(plan),
    }
    if config.mode == "exact" and kind != "seq_scan":
        raise BenchmarkError(
            "exact baseline did not use Seq Scan on chunks; refusing to call it exact "
            f"(plan kind={kind!r})"
        )
    if config.mode == "hnsw" and not uses_hnsw:
        raise BenchmarkError(
            "forced HNSW arm did not use the HNSW index; the ef_search comparison would "
            f"be invalid (plan kind={kind!r})"
        )
    return record


# ------------------------------------------------------------------ execution


def _run_query(
    conn: psycopg.Connection,
    vector: Sequence[float],
    config: BenchmarkConfig,
    statement_timeout_ms: int,
) -> tuple[list[Hit], int, float]:
    """Execute one retrieval and return hits, live GUC value and query time."""
    with conn.transaction():
        _set_statement_timeout(conn, statement_timeout_ms)
        _configure_planner(conn, config.mode)
        with ef_search(conn, config.ef_search):
            started = time.perf_counter()
            rows = conn.execute(
                DENSE_SEARCH_SQL,
                (Vector(list(vector)), config.top_k),
                prepare=False,
            ).fetchall()
            query_ms = (time.perf_counter() - started) * 1000.0
            actual_ef = current_ef_search(conn)
    return _rows_to_hits(rows, config.top_k), actual_ef, query_ms


def _rows_to_hits(rows: Sequence[Sequence[Any]], top_k: int) -> list[Hit]:
    """Map SELECT columns to validated hits; reject a non-finite distance."""
    hits: list[Hit] = []
    for row in rows:
        try:
            distance = float(row[5])
        except (TypeError, ValueError, IndexError) as exc:
            raise BenchmarkError(f"malformed retrieval row: {row!r}") from exc
        if not math.isfinite(distance):
            raise BenchmarkError(f"non-finite distance for chunk {row[0]!r}")
        hits.append(
            Hit(
                chunk_id=str(row[0]),
                doc_id=str(row[1]),
                chunk_strategy=str(row[2]),
                chunk_index=int(row[3]),
                distance=distance,
            )
        )
    if len(hits) > top_k:
        raise BenchmarkError(f"retrieval returned {len(hits)} rows for top_k={top_k}")
    return hits


def _db_call(
    connection_factory: ConnectionFactory,
    vector: Sequence[float],
    config: BenchmarkConfig,
    statement_timeout_ms: int,
) -> tuple[list[Hit], int, float, float]:
    """Borrow a connection, run one query, and time the whole database span."""
    started = time.perf_counter()
    with connection_factory() as conn:
        hits, actual_ef, query_ms = _run_query(conn, vector, config, statement_timeout_ms)
    db_ms = (time.perf_counter() - started) * 1000.0
    return hits, actual_ef, query_ms, db_ms


def _is_timeout(exc: BaseException) -> bool:
    """Return True for the two failure classes that are a timeout."""
    return isinstance(exc, (psycopg.errors.QueryCanceled, PoolTimeout))


def _error_record(
    exc: BaseException,
    *,
    stage: str,
    query_id: str | None = None,
    pass_number: int | None = None,
) -> dict[str, Any]:
    """Record an error without dumping query text or a vector."""
    return {
        "stage": stage,
        "query_id": query_id,
        "pass": pass_number,
        "error_type": type(exc).__name__,
        "message": str(exc)[:500],
        "timeout": _is_timeout(exc),
    }


def _hit_record(hit: Hit) -> dict[str, Any]:
    """Serialize one raw hit for the JSON artifact."""
    return {
        "key": hit.key,
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "chunk_strategy": hit.chunk_strategy,
        "chunk_index": hit.chunk_index,
        "distance": hit.distance,
    }


def _measure(
    query: EvalQuery,
    hits: Sequence[Hit],
    exact_keys: Sequence[str],
    top_k: int,
) -> dict[str, Any]:
    """Compute one sample's retrieval metrics against exact and gold references."""
    keys = [hit.key for hit in hits]
    doc_ids = list(dict.fromkeys(hit.doc_id for hit in hits))[:top_k]
    semantic = recall_at_k(doc_ids, query.relevant_doc_ids, top_k)

    exact_positions = {key: position for position, key in enumerate(exact_keys)}
    common = [
        (exact_positions[hit.key], position)
        for position, hit in enumerate(hits)
        if hit.key in exact_positions
    ]
    displacement = (
        statistics.fmean(abs(exact_rank - ann_rank) for exact_rank, ann_rank in common)
        if common
        else None
    )
    recall = recall_at_k(keys, list(exact_keys), top_k) if exact_keys else None
    return {
        "result_count": len(hits),
        "unique_count": len(set(keys)),
        "doc_unique_count": len(set(hit.doc_id for hit in hits)),
        "recall_vs_exact": recall,
        "semantic_recall": semantic,
        "order_equal": keys == list(exact_keys),
        "top1_match": keys[:1] == list(exact_keys)[:1],
        "rank_displacement_mean": displacement,
        "keys": keys,
        "doc_ids": doc_ids,
        "hits": [_hit_record(hit) for hit in hits],
    }


def _validate_inputs(
    queries: Sequence[EvalQuery],
    query_vectors: Mapping[str, Sequence[float]],
    configs: Sequence[BenchmarkConfig],
    repetitions: int,
    warmup_passes: int,
    statement_timeout_ms: int,
) -> None:
    """Reject inputs that would make the comparison unfair or meaningless."""
    if not queries:
        raise BenchmarkError("at least one eval query is required")
    if len({query.query_id for query in queries}) != len(queries):
        raise BenchmarkError("eval queries contain duplicate query ids")
    if not configs:
        raise BenchmarkError("at least one benchmark config is required")
    if len({config.name for config in configs}) != len(configs):
        raise BenchmarkError("benchmark config names must be unique")
    exact = [config for config in configs if config.mode == "exact"]
    if len(exact) != 1:
        raise BenchmarkError("exactly one exact baseline config is required")
    if len({config.top_k for config in configs}) != 1:
        raise BenchmarkError("every config must use the same top_k")
    try:
        for config in configs:
            DenseConfig(top_n=config.top_k, ef_search=config.ef_search)
    except ValidationError as exc:
        raise BenchmarkError(f"invalid benchmark config: {exc}") from exc
    if repetitions < 1:
        raise BenchmarkError("repetitions must be at least 1")
    if warmup_passes < 0:
        raise BenchmarkError("warmup_passes must not be negative")
    if statement_timeout_ms <= 0:
        raise BenchmarkError("statement_timeout_ms must be positive")

    missing = [query.query_id for query in queries if query.query_id not in query_vectors]
    if missing:
        raise BenchmarkError(f"missing query vectors for: {', '.join(missing)}")
    dimensions: set[int] = set()
    for query in queries:
        vector = query_vectors[query.query_id]
        if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
            raise BenchmarkError(f"query vector for {query.query_id} is not a sequence")
        if not vector:
            raise BenchmarkError(f"query vector for {query.query_id} is empty")
        dimensions.add(len(vector))
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BenchmarkError(f"query vector for {query.query_id} holds a non-number")
            if not math.isfinite(float(value)):
                raise BenchmarkError(f"query vector for {query.query_id} holds a non-finite value")
    if len(dimensions) != 1:
        raise BenchmarkError(f"query vectors have mixed dimensions: {sorted(dimensions)}")


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


def _min_or_none(values: Sequence[float]) -> float | None:
    """Minimum of a non-empty list, else None."""
    return min(values) if values else None


def _max_or_none(values: Sequence[float]) -> float | None:
    """Maximum of a non-empty list, else None."""
    return max(values) if values else None


def _quality_summary(per_query: Sequence[dict[str, Any]], top_k: int) -> dict[str, Any]:
    """Aggregate per-query quality metrics for one config."""
    evaluated = [record for record in per_query if "result_count" in record]
    recalls = [
        float(record["recall_vs_exact"])
        for record in evaluated
        if record["recall_vs_exact"] is not None
    ]
    semantics = [float(record["semantic_recall"]) for record in evaluated]
    return {
        "evaluated_queries": len(evaluated),
        "missing_queries": len(per_query) - len(evaluated),
        "baseline_missing_queries": sum(
            1 for record in evaluated if record["recall_vs_exact"] is None
        ),
        "recall_vs_exact_mean": _mean_or_none(recalls),
        "recall_vs_exact_min": _min_or_none(recalls),
        "recall_vs_exact_max": _max_or_none(recalls),
        "order_equal_queries": sum(1 for record in evaluated if record["order_equal"]),
        "top1_match_queries": sum(1 for record in evaluated if record["top1_match"]),
        "semantic_recall_mean": _mean_or_none(semantics),
        "semantic_recall_min": _min_or_none(semantics),
        "semantic_recall_max": _max_or_none(semantics),
        "shortfall_queries": sum(1 for record in evaluated if record["result_count"] < top_k),
        "overflow_queries": sum(1 for record in evaluated if record["result_count"] > top_k),
        "duplicate_result_queries": sum(
            1 for record in evaluated if record["unique_count"] != record["result_count"]
        ),
    }


def _empty_config_result(
    config: BenchmarkConfig,
    reason: str,
    skipped: int,
    plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a clearly invalid config record instead of pretending it ran."""
    return {
        "name": config.name,
        "mode": config.mode,
        "ef_search": config.ef_search,
        "top_k": config.top_k,
        "description": config.description,
        "plan": plan,
        "valid": False,
        "skipped_reason": reason,
        "skipped": skipped,
        "errors": 0,
        "timeouts": 0,
        "warmup_errors": 0,
        "ef_mismatch_samples": 0,
        "quality_mismatch_samples": 0,
        "actual_ef_search_values": [],
        "per_query": [],
        "latency": {
            "query_ms": _latency_summary([]),
            "db_ms": _latency_summary([]),
            "query_ms_samples": [],
            "db_ms_samples": [],
        },
        "quality": _quality_summary([], config.top_k),
    }


def run_dense_benchmark(
    queries: Sequence[EvalQuery],
    query_vectors: Mapping[str, Sequence[float]],
    configs: Sequence[BenchmarkConfig],
    *,
    repetitions: int = 5,
    warmup_passes: int = 1,
    statement_timeout_ms: int = 5000,
    connection_factory: ConnectionFactory = connection,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Run every config over every query and return raw measurements.

    Args:
        queries: Labelled queries, in evaluation order.
        query_vectors: Query id -> the exact vector sent to PostgreSQL. The
            runner never embeds text; the caller is responsible for using the
            same embedder/model as production.
        configs: Arms from :func:`build_configs` (or equivalent). Exactly one
            must be ``exact``; all must share one ``top_k``.
        repetitions: Timed passes over the query set per config. Each query
            therefore contributes ``repetitions`` latency samples.
        warmup_passes: Untimed passes over the query set before timing. The
            exact reference pass is also untimed and never enters latency.
        statement_timeout_ms: Transaction-local cap for each statement.
        connection_factory: Pooled connection context manager. Tests may inject
            one; production uses the project pool.
        progress: Optional callback for human-readable progress lines.

    Returns:
        A JSON-serializable mapping with environment-neutral parameters, plan
        evidence, per-query raw hits, latency samples, quality summaries and
        error/timeout/skip counts.

    Raises:
        BenchmarkError: inputs are invalid, or the exact baseline did not scan.
    """
    _validate_inputs(
        queries,
        query_vectors,
        configs,
        repetitions,
        warmup_passes,
        statement_timeout_ms,
    )
    emit = progress or (lambda _message: None)
    exact_config = next(config for config in configs if config.mode == "exact")
    first_query = queries[0]
    first_vector = query_vectors[first_query.query_id]
    started_at = datetime.now(UTC).isoformat()

    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": started_at,
        "parameters": {
            "queries": len(queries),
            "top_k": configs[0].top_k,
            "repetitions": repetitions,
            "warmup_passes": warmup_passes,
            "statement_timeout_ms": statement_timeout_ms,
            "ef_search_values": [config.ef_search for config in configs if config.mode == "hnsw"],
        },
        "definitions": _definitions(configs[0].top_k),
        "query_ids": [query.query_id for query in queries],
        "query_gold_sizes": {query.query_id: len(query.relevant_doc_ids) for query in queries},
        "exact_reference": {"queries": {}, "errors": []},
        "configs": [],
        "errors": [],
    }

    # The exact plan is captured before any reference result is trusted. If the
    # planner did not scan, the whole experiment has no baseline.
    emit(f"capturing exact plan with {first_query.query_id}")
    with connection_factory() as conn:
        exact_plan = _capture_plan(conn, first_vector, exact_config, statement_timeout_ms)

    emit("running untimed exact reference")
    exact_results: dict[str, list[Hit]] = {}
    for query in queries:
        try:
            hits, _actual_ef, _query_ms, _db_ms = _db_call(
                connection_factory,
                query_vectors[query.query_id],
                exact_config,
                statement_timeout_ms,
            )
            exact_results[query.query_id] = hits
            result["exact_reference"]["queries"][query.query_id] = {
                "keys": [hit.key for hit in hits],
                "doc_ids": list(dict.fromkeys(hit.doc_id for hit in hits)),
                "hits": [_hit_record(hit) for hit in hits],
            }
        except (psycopg.Error, PoolTimeout, ConnectionError) as exc:
            result["exact_reference"]["errors"].append(
                _error_record(exc, stage="exact_reference", query_id=query.query_id)
            )

    for config in configs:
        emit(f"config {config.name}: plan")
        plan: dict[str, Any] | None
        try:
            if config.mode == "exact":
                plan = exact_plan
            else:
                with connection_factory() as conn:
                    plan = _capture_plan(conn, first_vector, config, statement_timeout_ms)
        except (psycopg.Error, PoolTimeout, ConnectionError) as exc:
            error = _error_record(exc, stage=f"{config.name}:plan")
            result["errors"].append(error)
            if config.mode == "hnsw":
                result["configs"].append(
                    _empty_config_result(
                        config,
                        "plan_capture_failed",
                        len(queries) * repetitions,
                        None,
                    )
                )
                continue
            raise
        except BenchmarkError as exc:
            if config.mode == "hnsw":
                result["errors"].append(_error_record(exc, stage=f"{config.name}:plan"))
                result["configs"].append(
                    _empty_config_result(
                        config,
                        str(exc),
                        len(queries) * repetitions,
                        None,
                    )
                )
                continue
            raise

        assert plan is not None
        if config.mode == "hnsw" and not plan["uses_hnsw"]:
            result["configs"].append(
                _empty_config_result(
                    config,
                    "plan_did_not_use_hnsw",
                    len(queries) * repetitions,
                    plan,
                )
            )
            continue

        emit(f"config {config.name}: {warmup_passes} warm-up pass(es)")
        warmup_errors = 0
        for _pass in range(warmup_passes):
            for query in queries:
                try:
                    _db_call(
                        connection_factory,
                        query_vectors[query.query_id],
                        config,
                        statement_timeout_ms,
                    )
                except (psycopg.Error, PoolTimeout, ConnectionError) as exc:
                    warmup_errors += 1
                    result["errors"].append(
                        _error_record(
                            exc,
                            stage=f"{config.name}:warmup",
                            query_id=query.query_id,
                        )
                    )

        emit(f"config {config.name}: {repetitions} timed pass(es)")
        query_samples: list[float] = []
        db_samples: list[float] = []
        per_query: dict[str, dict[str, Any]] = {}
        first_signature: dict[str, tuple[str, ...]] = {}
        quality_mismatch_samples = 0
        quality_mismatch_queries: set[str] = set()
        ef_mismatch_samples = 0
        actual_ef_values: set[int] = set()
        errors = 0
        timeouts = 0

        for pass_number in range(1, repetitions + 1):
            for query in queries:
                try:
                    hits, actual_ef, query_ms, db_ms = _db_call(
                        connection_factory,
                        query_vectors[query.query_id],
                        config,
                        statement_timeout_ms,
                    )
                except (psycopg.Error, PoolTimeout, ConnectionError) as exc:
                    if _is_timeout(exc):
                        timeouts += 1
                    else:
                        errors += 1
                    result["errors"].append(
                        _error_record(
                            exc,
                            stage=config.name,
                            query_id=query.query_id,
                            pass_number=pass_number,
                        )
                    )
                    per_query.setdefault(
                        query.query_id,
                        {
                            "query_id": query.query_id,
                            "error": type(exc).__name__,
                            "recall_vs_exact": None,
                        },
                    )
                    continue

                actual_ef_values.add(actual_ef)
                if actual_ef != config.ef_search:
                    ef_mismatch_samples += 1
                query_samples.append(query_ms)
                db_samples.append(db_ms)

                exact_hits = exact_results.get(query.query_id, [])
                exact_keys = [hit.key for hit in exact_hits]
                measured = _measure(query, hits, exact_keys, config.top_k)
                measured.update(
                    {
                        "query_id": query.query_id,
                        "pass": pass_number,
                        "query_ms": query_ms,
                        "db_ms": db_ms,
                        "actual_ef_search": actual_ef,
                    }
                )
                signature = tuple(measured["keys"])
                if query.query_id not in first_signature:
                    first_signature[query.query_id] = signature
                elif signature != first_signature[query.query_id]:
                    quality_mismatch_samples += 1
                    quality_mismatch_queries.add(query.query_id)
                if query.query_id not in per_query:
                    per_query[query.query_id] = measured

        config_result = {
            "name": config.name,
            "mode": config.mode,
            "ef_search": config.ef_search,
            "top_k": config.top_k,
            "description": config.description,
            "plan": plan,
            "valid": True,
            "skipped_reason": None,
            "skipped": 0,
            "errors": errors,
            "timeouts": timeouts,
            "warmup_errors": warmup_errors,
            "ef_mismatch_samples": ef_mismatch_samples,
            "quality_mismatch_samples": quality_mismatch_samples,
            "quality_mismatch_queries": sorted(quality_mismatch_queries),
            "actual_ef_search_values": sorted(actual_ef_values),
            "per_query": [
                per_query[query.query_id] for query in queries if query.query_id in per_query
            ],
            "latency": {
                "query_ms": _latency_summary(query_samples),
                "db_ms": _latency_summary(db_samples),
                "query_ms_samples": query_samples,
                "db_ms_samples": db_samples,
            },
            "quality": _quality_summary(
                [per_query[query.query_id] for query in queries if query.query_id in per_query],
                config.top_k,
            ),
        }
        result["configs"].append(config_result)

    result["totals"] = _totals(result)
    logger.info(
        "dense benchmark finished: configs=%d queries=%d repetitions=%d "
        "errors=%d timeouts=%d skipped=%d",
        len(result["configs"]),
        len(queries),
        repetitions,
        result["totals"]["errors"],
        result["totals"]["timeouts"],
        result["totals"]["skipped"],
    )
    return result


def _definitions(top_k: int) -> dict[str, Any]:
    """Return the benchmark's written definitions, stored with the raw data."""
    return {
        "granularity": "chunk-level; one result is one chunks row, joined to documents",
        "metric": "cosine distance (<=>); lower is better; distance is raw pgvector output",
        "top_k": top_k,
        "exact_baseline": (
            "same DENSE_SEARCH_SQL, same query vector, same distance, same Top-K and "
            "granularity, with transaction-local enable_indexscan=off, "
            "enable_bitmapscan=off, enable_indexonlyscan=off and enable_seqscan=on; "
            "verified to produce Seq Scan on chunks"
        ),
        "ann_recall": (
            "Recall@K of the ANN chunk keys against the exact cosine Top-K keys; "
            "denominator is the exact result count"
        ),
        "semantic_recall": (
            "doc-level Recall@K of the returned unique doc_ids against relevant_doc_ids; "
            "this is labelled retrieval quality, not ANN fidelity"
        ),
        "latency_query_ms": "execute + fetch around the retrieval SELECT only",
        "latency_db_ms": "pool borrow + transaction + planner/GUC setup + execute + fetch",
        "warmup": (
            "model load and first encode are measured separately and excluded; exact reference "
            "is untimed; each config gets untimed passes before timed passes"
        ),
        "percentile": "nearest rank: ceil(p/100 * n), clamped to [1, n]",
    }


def _totals(result: Mapping[str, Any]) -> dict[str, int]:
    """Aggregate error, timeout and skip counts across all arms.

    Timed samples are counted in their config's ``errors``/``timeouts`` fields;
    plan and warm-up failures are in the top-level ``errors`` list. A timeout is
    counted once, in ``timeouts``, and never also in ``errors``.
    """
    configs = list(result.get("configs", []))
    exact_errors = list(result.get("exact_reference", {}).get("errors", []))
    top_errors = list(result.get("errors", []))

    def non_timeouts(records: Sequence[Mapping[str, Any]]) -> int:
        return sum(1 for record in records if not record.get("timeout"))

    def timeouts(records: Sequence[Mapping[str, Any]]) -> int:
        return sum(1 for record in records if record.get("timeout"))

    return {
        "errors": non_timeouts(exact_errors)
        + sum(int(config.get("errors", 0)) for config in configs)
        + non_timeouts(top_errors),
        "timeouts": timeouts(exact_errors)
        + sum(int(config.get("timeouts", 0)) for config in configs)
        + timeouts(top_errors),
        "skipped": sum(int(config.get("skipped", 0)) for config in configs),
        "invalid_configs": sum(1 for config in configs if not config.get("valid", False)),
    }


# ---------------------------------------------------------------------- report


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


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render the benchmark result as a reviewable Markdown report."""
    environment = result.get("environment", {})
    dataset = result.get("dataset", {})
    parameters = result.get("parameters", {})
    configs = list(result.get("configs", []))
    totals = result.get("totals", {})
    status = result.get("status", "unknown")
    lines: list[str] = [
        "# Dense Retrieval Benchmark (Phase 4C)",
        "",
        f"- Generated: `{result.get('generated_at', 'unknown')}`",
        f"- Status: **{status}**",
        f"- Queries: {parameters.get('queries', '?')}  |  K: {parameters.get('top_k', '?')}  "
        f"|  Timed passes/config: {parameters.get('repetitions', '?')}  |  "
        f"Statement timeout: {parameters.get('statement_timeout_ms', '?')} ms",
        "",
        "## 1. Environment and dataset",
        "",
    ]
    env_rows = [
        ("PostgreSQL", environment.get("postgres_version")),
        ("pgvector extension", environment.get("vector_version")),
        ("HNSW index", environment.get("hnsw_index")),
        ("HNSW parameters", environment.get("hnsw_parameters")),
        ("Server address", environment.get("server_address")),
        ("Data directory", environment.get("data_directory")),
        ("Host Python", environment.get("python_version")),
        ("Host model stack", environment.get("model_stack")),
        ("Embedding model", environment.get("embedding_model")),
        ("Docker control plane", environment.get("docker_status")),
        ("Free disk", environment.get("disk_free")),
        (
            "Documents / chunks",
            f"{environment.get('documents', '?')} / {environment.get('chunks', '?')}",
        ),
        ("Distinct chunk doc_ids", environment.get("distinct_chunk_docs")),
        ("Eval queries", dataset.get("queries")),
        ("Gold sizes", dataset.get("gold_sizes")),
        ("Missing gold docs", dataset.get("missing_gold_docs")),
        ("Data files", dataset.get("hashes")),
        ("Counts after run", dataset.get("post_run_counts")),
    ]
    lines.append(_md_table(["Item", "Value"], env_rows))
    lines.extend(
        [
            "",
            "The numbers below were produced on the host Python fallback against the live "
            "PostgreSQL container. The Docker CLI/API was unresponsive during the run, so "
            "container-internal execution is **not verified** by this report; the database "
            "data plane was healthy and all benchmark statements are read-only.",
            "",
            "## 2. Experiment matrix and parameters",
            "",
            _md_table(
                ["Arm", "Mode", "ef_search", "Top-K", "Planner control", "Definition"],
                [
                    (
                        config["name"],
                        config["mode"],
                        config["ef_search"],
                        config["top_k"],
                        {
                            "exact": "index/bitmap/indexonly off, seqscan on",
                            "natural": "none",
                            "hnsw": "seqscan off",
                        }[config["mode"]],
                        config["description"],
                    )
                    for config in configs
                ],
            ),
            "",
            "The exact arm is the baseline; the natural arm is the production path; the "
            "HNSW arms are controlled diagnostics. `ef_search` values are legal pgvector "
            "values in the project-validated range 1--1000.",
            "",
            "## 3. Exact baseline definition",
            "",
            "The exact baseline uses the **same** `DENSE_SEARCH_SQL`, query vectors, cosine "
            "metric, chunk granularity and Top-K as every ANN arm. Only the transaction-local "
            "planner settings differ: `enable_indexscan=off`, `enable_bitmapscan=off`, "
            "`enable_indexonlyscan=off`, `enable_seqscan=on`. The captured plan must contain "
            "`Seq Scan on chunks` or the run refuses to call it exact.",
            "",
            "## 4. Plan evidence",
            "",
        ]
    )
    plan_rows = []
    for config in configs:
        plan = config.get("plan") or {}
        plan_rows.append(
            (
                config["name"],
                plan.get("kind", "missing"),
                plan.get("uses_hnsw", "-"),
                plan.get("uses_seq_scan_on_chunks", "-"),
                _fmt(plan.get("execution_ms")),
                config.get("valid", False),
            )
        )
    lines.append(
        _md_table(
            ["Arm", "Plan kind", "Uses HNSW", "Seq Scan on chunks", "EXPLAIN ms", "Valid"],
            plan_rows,
        )
    )
    lines.extend(["", "Captured plan snippets:", ""])
    for config in configs:
        plan = config.get("plan") or {}
        lines.extend(
            [
                f"### `{config['name']}`",
                "",
                "```text",
                str(plan.get("text", "no plan captured")),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## 5. Latency and Recall@K",
            "",
            "Latency is reported separately for `query_ms` (execute + fetch only) and "
            "`db_ms` (pool borrow + transaction + GUC/planner setup + execute + fetch). "
            "Percentiles are nearest-rank over all per-query timed samples.",
            "",
            _md_table(
                [
                    "Arm",
                    "n",
                    "query P50",
                    "query P95",
                    "query P99",
                    "db P50",
                    "db P95",
                    "ANN Recall@K mean",
                    "ANN Recall@K min",
                    "Semantic Recall mean",
                    "Order-equal",
                    "Top-1 match",
                    "Errors",
                    "Timeouts",
                    "Skipped",
                ],
                [
                    (
                        config["name"],
                        config["latency"]["query_ms"]["n"],
                        _fmt(config["latency"]["query_ms"]["p50"]),
                        _fmt(config["latency"]["query_ms"]["p95"]),
                        _fmt(config["latency"]["query_ms"]["p99"]),
                        _fmt(config["latency"]["db_ms"]["p50"]),
                        _fmt(config["latency"]["db_ms"]["p95"]),
                        _fmt(config["quality"]["recall_vs_exact_mean"]),
                        _fmt(config["quality"]["recall_vs_exact_min"]),
                        _fmt(config["quality"]["semantic_recall_mean"]),
                        config["quality"]["order_equal_queries"],
                        config["quality"]["top1_match_queries"],
                        config["errors"],
                        config["timeouts"],
                        config["skipped"],
                    )
                    for config in configs
                ],
            ),
            "",
            "`ANN Recall@K` is against the exact chunk Top-K. `Semantic Recall` is against "
            "`relevant_doc_ids` and is a different question: it measures whether the returned "
            "documents are the labelled ones, not whether ANN reproduced exact vector order.",
            "",
            "## 6. Per-arm detail",
            "",
        ]
    )
    detail_rows = []
    for config in configs:
        quality = config["quality"]
        latency = config["latency"]
        detail_rows.append(
            (
                config["name"],
                quality["evaluated_queries"],
                quality["missing_queries"],
                quality["baseline_missing_queries"],
                quality["shortfall_queries"],
                quality["overflow_queries"],
                quality["duplicate_result_queries"],
                config["ef_mismatch_samples"],
                config["quality_mismatch_samples"],
                ", ".join(str(value) for value in config["actual_ef_search_values"]) or "-",
                config["valid"],
                config["skipped_reason"] or "-",
                _fmt(latency["query_ms"]["min"]),
                _fmt(latency["query_ms"]["max"]),
                _fmt(latency["db_ms"]["min"]),
                _fmt(latency["db_ms"]["max"]),
            )
        )
    lines.append(
        _md_table(
            [
                "Arm",
                "Evaluated",
                "Missing",
                "Baseline missing",
                "Shortfall",
                "Overflow",
                "Duplicates",
                "ef mismatches",
                "Quality mismatches",
                "Observed ef",
                "Valid",
                "Skip reason",
                "query min",
                "query max",
                "db min",
                "db max",
            ],
            detail_rows,
        )
    )
    lines.extend(
        [
            "",
            "`Shortfall` counts queries returning fewer than K rows (a legal HNSW outcome when "
            "`ef_search < K`); `Overflow` and `Duplicates` must be zero for a sound Top-K. "
            "`ef mismatches` compares the read-back live GUC to the requested value; it must "
            "be zero. `Quality mismatches` counts timed samples whose result signature differs "
            "from that query's first sample.",
            "",
            "## 7. Interpretation and limitations",
            "",
        ]
    )
    natural = next((config for config in configs if config["mode"] == "natural"), None)
    exact = next((config for config in configs if config["mode"] == "exact"), None)
    if natural is not None and exact is not None:
        natural_quality = natural["quality"]
        lines.append(
            f"- Natural vs exact on {natural_quality['evaluated_queries']} queries: "
            f"mean ANN Recall@K = {_fmt(natural_quality['recall_vs_exact_mean'])}, "
            f"order-equal = {natural_quality['order_equal_queries']}, "
            f"top-1 match = {natural_quality['top1_match_queries']}."
        )
        natural_plan = natural.get("plan") or {}
        lines.append(
            f"- Natural plan kind: `{natural_plan.get('kind', 'missing')}`. At 150 chunks a "
            "Seq Scan is plausible; `ef_search` is then not observable in the natural path."
        )
    for config in configs:
        if config["mode"] != "hnsw" or not config["valid"]:
            continue
        quality = config["quality"]
        lines.append(
            f"- `{config['name']}`: observed ef = "
            f"{config['actual_ef_search_values']}, ANN Recall@K mean = "
            f"{_fmt(quality['recall_vs_exact_mean'])}, order-equal = "
            f"{quality['order_equal_queries']}/{quality['evaluated_queries']}, "
            f"query P50/P95 = {_fmt(config['latency']['query_ms']['p50'])}/"
            f"{_fmt(config['latency']['query_ms']['p95'])} ms."
        )
    lines.extend(
        [
            "- The corpus is only 150 chunks. If every ANN arm reproduces exact Top-K, that is "
            "a scale observation, not proof that `ef_search` is ineffective at larger N. It also "
            "does not establish an HNSW latency advantage.",
            "- Latency here is single-process, sequential and warm; no concurrency or "
            "throughput claim is made. Embedding time is measured separately from database "
            "retrieval and is not folded into `query_ms`.",
            "- The human-labelled `relevant_doc_ids` measure retrieval quality; ANN Recall@K "
            "measures fidelity to the exact vector search. Neither is a substitute for the other.",
            "- `SET LOCAL enable_seqscan=off` is a diagnostic control, never a production or "
            "global setting. The natural arm is the production path.",
            "",
            "## 8. Errors, timeouts, skips",
            "",
            f"- Totals: errors={totals.get('errors', '?')}, "
            f"timeouts={totals.get('timeouts', '?')}, skipped={totals.get('skipped', '?')}, "
            f"invalid configs={totals.get('invalid_configs', '?')}.",
            "- Exact reference errors: "
            f"{len(result.get('exact_reference', {}).get('errors', []))}.",
            "",
            "Every failed timed sample is stored with its stage, query id, pass, error type and "
            "timeout flag in the JSON artifact. A timeout is a `QueryCanceled` or pool timeout; "
            "it is not silently converted into a zero-latency or zero-recall result.",
            "",
            "## 9. Reproduce",
            "",
            "Host fallback (PowerShell), from the repository root:",
            "",
            "```powershell",
            "$env:APEX_PG_HOST='127.0.0.1'; $env:HF_HUB_CACHE='D:\\hf_cache'; "
            "$env:HF_HUB_OFFLINE='1';",
            "python scripts/benchmark_dense.py --top-k 10 --ef-search 10 40 100 "
            "--repetitions 5 --warmup-passes 1",
            "```",
            "",
            "Container path, once the Docker control plane is healthy:",
            "",
            "```bash",
            "docker compose run --rm -T app python scripts/benchmark_dense.py "
            "--top-k 10 --ef-search 10 40 100 --repetitions 5 --warmup-passes 1",
            "```",
            "",
            "Smoke first:",
            "",
            "```bash",
            "python scripts/benchmark_dense.py --limit 3 --top-k 10 --ef-search 10 40 "
            "--repetitions 1 --warmup-passes 1 --dry-run",
            "```",
            "",
            "## 10. Data safety",
            "",
            "The benchmark issues SELECT, EXPLAIN and transaction-local `set_config` only. It "
            "does not write rows, change schema, rebuild indexes, set a global GUC, or modify "
            "`data/kb_documents.json` / `data/eval_queries.json`. Dataset hashes and the "
            "post-run row counts are recorded above. The raw per-sample JSON is in "
            "`reports/dense_benchmark.json`.",
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
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(result), encoding="utf-8")
