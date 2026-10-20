"""HNSW / IVFFlat parameter experiments.

Measures what the other benchmarks cannot: build-time index parameters. Every
configuration is built from scratch and queried with pgvector's query-time knobs
(``hnsw.ef_search``, ``ivfflat.probes``), and each measurement records build
time, index size, the ANN result, the exact result, and the ``EXPLAIN`` plan as
evidence.

Isolation
---------
pgvector offers no per-index query hint, and the served ``chunks`` table already
carries the bootstrap HNSW index. Two HNSW indexes on the same column would make
the planner choose between them by cost, so a shadow index could not be shown to
be the one that ran. Each configuration therefore runs on a **temporary table**
created with ``CREATE TEMP TABLE ... AS SELECT`` over the selected strategy's
rows (same column types, copied vectors) and dropped when its configuration
finishes. Nothing in this module issues ``CREATE INDEX`` against ``chunks``, and
the bootstrap index is never read, replaced or dropped.

The consequence, stated in the artifact as well: these are real pgvector builds
and real ANN queries, but they are not queries against the served table. The
row count per configuration is recorded so the evidence can be checked.

Scale
-----
150 vectors are too few to separate ``m`` / ``ef_construction`` meaningfully.
``synthetic_scale > 1`` deterministically duplicates every vector (with derived
chunk ids) so the index can be exercised at a larger size. Any run with
``synthetic_scale > 1`` is labelled ``SYNTHETIC`` and must not be used for
semantic quality claims: it measures index behaviour only.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from pgvector import Vector

from src.db import connection
from src.evaluation.artifacts import ensure_writable
from src.evaluation.dense_benchmark import EvalQuery
from src.evaluation.metrics import latency_summary, recall_at_k
from src.retrieval.fusion import COUNT_MAX

INDEX_BENCHMARK_VERSION = 1
"""Schema version of the index artifact."""

_SEARCH_SQL = """
    SELECT chunk_id, embedding <=> %s AS distance
    FROM {table}
    ORDER BY embedding <=> %s
    LIMIT %s
"""

_VECTOR_LITERAL_RE = re.compile(r"'\[[^\]]*\]'::vector")

_PLAN_LINE_LIMIT = 300
"""Long EXPLAIN lines can contain a bound vector; keep the artifact readable."""


@dataclass(frozen=True)
class IndexBenchmarkConfig:
    """The sweep matrix and measurement parameters."""

    top_k: int = 10
    repetitions: int = 3
    warmup_passes: int = 1
    statement_timeout_ms: int = 5000
    chunk_strategy: str = "document"
    synthetic_scale: int = 1
    hnsw_m_values: tuple[int, ...] = (8, 16, 32)
    hnsw_ef_construction_values: tuple[int, ...] = (32, 64, 128)
    hnsw_ef_search_values: tuple[int, ...] = (10, 40, 100)
    ivfflat_lists_values: tuple[int, ...] = (1, 2, 4, 8, 16)
    ivfflat_probes_values: tuple[int, ...] = (1, 2, 4)
    include_hnsw: bool = True
    include_ivfflat: bool = True

    def __post_init__(self) -> None:
        """Validate every count and pgvector range before any DDL runs."""
        for name in (
            "top_k",
            "repetitions",
            "warmup_passes",
            "statement_timeout_ms",
            "synthetic_scale",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.top_k > COUNT_MAX:
            raise ValueError(f"top_k must not exceed {COUNT_MAX}")
        if self.chunk_strategy not in ("document", "fixed_window", "structured"):
            raise ValueError(f"unsupported chunk_strategy {self.chunk_strategy!r}")
        _require_range(self.hnsw_m_values, 2, 100, "hnsw_m_values")
        _require_range(self.hnsw_ef_construction_values, 4, 1000, "hnsw_ef_construction_values")
        _require_range(self.hnsw_ef_search_values, 1, 1000, "hnsw_ef_search_values")
        _require_range(self.ivfflat_lists_values, 1, 32768, "ivfflat_lists_values")
        _require_range(self.ivfflat_probes_values, 1, 32768, "ivfflat_probes_values")
        if not self.include_hnsw and not self.include_ivfflat:
            raise ValueError("at least one index method must be enabled")


def run_index_benchmark(
    queries: Sequence[EvalQuery],
    vectors: Mapping[str, Sequence[float]],
    config: IndexBenchmarkConfig | None = None,
    *,
    connection_factory: Callable[[], Any] = connection,
    progress: Callable[[str], None] | None = None,
    environment: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the configured sweep and return one reviewable artifact.

    Args:
        queries: Labelled queries. Only their text and id are used.
        vectors: Pre-computed query vectors keyed by ``query_id``.
        config: Sweep matrix; defaults to the plan's 9 HNSW / 15 IVFFlat grid.
        connection_factory: Context manager yielding a connection (tests inject
            a diagnostic one; production uses the pool).
        progress: Optional callback for long builds.
        environment: Provenance recorded verbatim.
        dataset: Provenance recorded verbatim.

    Returns:
        The artifact. A configuration that fails is recorded with its error and
        does not abort the other configurations.
    """
    resolved = config if config is not None else IndexBenchmarkConfig()
    missing = [query.query_id for query in queries if query.query_id not in vectors]
    if missing:
        raise ValueError(f"missing query vectors for: {', '.join(missing)}")

    generated = datetime.now(UTC)
    started = time.perf_counter()
    exact = _exact_reference(queries, vectors, resolved, connection_factory, progress)
    hnsw_results = (
        _sweep_hnsw(queries, vectors, resolved, exact, connection_factory, progress)
        if resolved.include_hnsw
        else []
    )
    ivfflat_results = (
        _sweep_ivfflat(queries, vectors, resolved, exact, connection_factory, progress)
        if resolved.include_ivfflat
        else []
    )

    configurations = [*hnsw_results, *ivfflat_results]
    failed = [record for record in configurations if record["status"] != "PASS"]
    if configurations and not failed:
        status, reason = "PASS", "every index configuration completed"
    elif configurations and len(failed) < len(configurations):
        status = "PARTIAL"
        reason = f"{len(failed)} of {len(configurations)} configurations failed"
    else:
        status, reason = "FAIL", "no index configuration completed"

    return {
        "schema_version": INDEX_BENCHMARK_VERSION,
        "experiment_name": "index_param_sweep",
        "generated_at": generated.isoformat(),
        "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "status": status,
        "status_reason": reason,
        "parameters": _parameters(resolved, queries),
        "environment": dict(environment or {}),
        "dataset": dict(dataset or {}),
        "isolation": _isolation(resolved),
        "exact": exact,
        "configurations": configurations,
        "limitations": _limitations(resolved),
    }


def blocked_result(
    message: str,
    *,
    config: IndexBenchmarkConfig | None = None,
    environment: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete ``BLOCKED`` artifact for a gate that failed before DDL."""
    resolved = config if config is not None else IndexBenchmarkConfig()
    return {
        "schema_version": INDEX_BENCHMARK_VERSION,
        "experiment_name": "index_param_sweep",
        "generated_at": datetime.now(UTC).isoformat(),
        "duration_ms": 0.0,
        "status": "BLOCKED",
        "status_reason": message,
        "parameters": _parameters(resolved, []),
        "environment": dict(environment or {}),
        "dataset": dict(dataset or {}),
        "isolation": _isolation(resolved),
        "exact": None,
        "configurations": [],
        "limitations": [message, *_limitations(resolved)],
    }


def _isolation(config: IndexBenchmarkConfig) -> dict[str, Any]:
    """Describe how the experiment is isolated from the served table."""
    return {
        "table": (
            "temporary copy of the selected chunk rows, created and dropped "
            "per configuration"
        ),
        "creates_index_on_chunks": False,
        "bootstrap_index_touched": False,
        "synthetic_label": "SYNTHETIC" if config.synthetic_scale > 1 else None,
        "reason": (
            "pgvector has no per-index hint; with the bootstrap index present a shadow "
            "index could not be shown to be the one the planner used"
        ),
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render the index artifact as a Markdown report."""
    parameters = result["parameters"]
    synthetic = result["isolation"].get("synthetic_label")
    exact = result.get("exact") or {}
    lines = [
        "# 索引参数实验（HNSW / IVFFlat）",
        "",
        f"- Status: **{result['status']}**",
        f"- Reason: {result['status_reason']}",
        f"- Generated: {result['generated_at']}",
        f"- Top-K: {parameters['top_k']}；查询数: {parameters['queries']}；"
        f"chunk strategy: `{parameters['chunk_strategy']}`",
        f"- Exact reference: {exact.get('method', 'not computed')}；"
        f"queries: {exact.get('query_count', 0)}",
        f"- Synthetic scale: {parameters['synthetic_scale']}"
        + ("（**SYNTHETIC**，仅用于索引行为）" if synthetic else "（真实 150 行）"),
        "",
        "> 每个配置在临时表副本上重建索引，不修改 `chunks` 表及其 bootstrap 索引。",
        "> ANN Recall 是相对精确 Seq Scan Top-K 的保真度，不是语义召回。",
        "",
        "## 1. 总览",
        "",
        "| Method | Params | Query setting | Build ms | Size | Rows | ANN Recall@K | "
        "P50 ms | P95 ms | P99 ms | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in result["configurations"]:
        if not record["query_params"]:
            lines.append(
                f"| {record['method']} | {_format_params(record['params'])} | - | - | "
                f"- | - | - | - | - | - | {record['status']} |"
            )
            continue
        for setting, measurement in record["query_params"].items():
            latency = measurement["latency_ms"]
            lines.append(
                "| {method} | {params} | {setting} | {build} | {size} | {rows} | "
                "{recall} | {p50} | {p95} | {p99} | {status} |".format(
                    method=record["method"],
                    params=_format_params(record["params"]),
                    setting=setting,
                    build=_fmt(record.get("build_ms")),
                    size=record.get("size_pretty") or "-",
                    rows=record.get("row_count", "-"),
                    recall=_fmt(measurement.get("ann_recall_mean")),
                    p50=_fmt(latency["p50"]),
                    p95=_fmt(latency["p95"]),
                    p99=_fmt(latency["p99"]),
                    status=record["status"],
                )
            )

    lines.extend(["", "## 2. 限制", ""])
    lines.extend(f"- {item}" for item in result["limitations"])
    failures = [
        (record["index_name"], record.get("error"))
        for record in result["configurations"]
        if record["status"] != "PASS"
    ]
    if failures:
        lines.extend(["", "## 3. 失败的配置", ""])
        lines.extend(f"- `{name}`: {error}" for name, error in failures)
    return "\n".join(lines) + "\n"


def write_outputs(
    result: Mapping[str, Any],
    json_path: Path,
    report_path: Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write the index JSON and Markdown artifacts, refusing to clobber."""
    ensure_writable((json_path, report_path), overwrite=overwrite)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, report_path


# ------------------------------------------------------------------- sweeps


def _sweep_hnsw(
    queries: Sequence[EvalQuery],
    vectors: Mapping[str, Sequence[float]],
    config: IndexBenchmarkConfig,
    exact: Mapping[str, Any],
    connection_factory: Callable[[], Any],
    progress: Callable[[str], None] | None,
) -> list[dict[str, Any]]:
    """Run the HNSW grid."""
    results: list[dict[str, Any]] = []
    for m in config.hnsw_m_values:
        for ef_construction in config.hnsw_ef_construction_values:
            name = f"idx_bench_hnsw_m{m}_efc{ef_construction}"
            params = {"m": m, "ef_construction": ef_construction}
            results.append(
                _run_configuration(
                    name,
                    "hnsw",
                    params,
                    queries,
                    vectors,
                    config,
                    exact,
                    connection_factory,
                    progress,
                    query_parameter="ef_search",
                    query_values=config.hnsw_ef_search_values,
                )
            )
    return results


def _sweep_ivfflat(
    queries: Sequence[EvalQuery],
    vectors: Mapping[str, Sequence[float]],
    config: IndexBenchmarkConfig,
    exact: Mapping[str, Any],
    connection_factory: Callable[[], Any],
    progress: Callable[[str], None] | None,
) -> list[dict[str, Any]]:
    """Run the IVFFlat grid."""
    results: list[dict[str, Any]] = []
    for lists in config.ivfflat_lists_values:
        name = f"idx_bench_ivfflat_l{lists}"
        results.append(
            _run_configuration(
                name,
                "ivfflat",
                {"lists": lists},
                queries,
                vectors,
                config,
                exact,
                connection_factory,
                progress,
                query_parameter="probes",
                query_values=config.ivfflat_probes_values,
            )
        )
    return results


def _run_configuration(
    name: str,
    method: str,
    params: Mapping[str, int],
    queries: Sequence[EvalQuery],
    vectors: Mapping[str, Sequence[float]],
    config: IndexBenchmarkConfig,
    exact: Mapping[str, Any],
    connection_factory: Callable[[], Any],
    progress: Callable[[str], None] | None,
    *,
    query_parameter: str,
    query_values: Sequence[int],
) -> dict[str, Any]:
    """Build one index on a temp copy, query it at each setting, drop it."""
    if progress is not None:
        progress(f"{name}: building")
    try:
        with connection_factory() as conn:
            row_count = _create_copy_table(conn, config)
            try:
                build_ms = _build_temp_index(conn, name, method, params)
                size = _index_size(conn, name)
                measurements = _measure_queries(
                    conn,
                    queries,
                    vectors,
                    config,
                    exact,
                    method=method,
                    query_parameter=query_parameter,
                    query_values=query_values,
                )
                plan = _explain(
                    conn,
                    vectors[queries[0].query_id],
                    config,
                    name,
                    method=method,
                    query_parameter=query_parameter,
                    value=query_values[0],
                )
                used = name in plan["text"]
                if not used:
                    raise RuntimeError(
                        f"forced {method} query did not use {name!r}; "
                        f"plan kind={plan['kind']!r}"
                    )
            finally:
                conn.execute("DROP TABLE IF EXISTS idx_bench_copy", prepare=False)
    except Exception as exc:  # noqa: BLE001 -- one configuration failing is data
        return {
            "index_name": name,
            "method": method,
            "params": dict(params),
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "build_ms": None,
            "size_bytes": None,
            "size_pretty": None,
            "row_count": None,
            "query_params": {},
        }

    return {
        "index_name": name,
        "method": method,
        "params": dict(params),
        "status": "PASS",
        "error": None,
        "build_ms": round(build_ms, 3),
        "size_bytes": size[0],
        "size_pretty": size[1],
        "row_count": row_count,
        "query_params": measurements,
        "plan": plan,
    }


def _exact_reference(
    queries: Sequence[EvalQuery],
    vectors: Mapping[str, Sequence[float]],
    config: IndexBenchmarkConfig,
    connection_factory: Callable[[], Any],
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    """Compute the exact Top-K for every query on a copy table."""
    if progress is not None:
        progress("exact: computing Seq Scan reference")
    reference: dict[str, list[str]] = {}
    with connection_factory() as conn:
        _create_copy_table(conn, config)
        try:
            for query in queries:
                vector = Vector(list(vectors[query.query_id]))
                with conn.transaction():
                    conn.execute("SET LOCAL enable_indexscan = off", prepare=False)
                    conn.execute("SET LOCAL enable_bitmapscan = off", prepare=False)
                    conn.execute("SET LOCAL enable_indexonlyscan = off", prepare=False)
                    conn.execute("SET LOCAL enable_seqscan = on", prepare=False)
                    rows = conn.execute(
                        _SEARCH_SQL.format(table="idx_bench_copy"),
                        (vector, vector, config.top_k),
                        prepare=False,
                    ).fetchall()
                reference[query.query_id] = [str(row[0]) for row in rows]
        finally:
            conn.execute("DROP TABLE IF EXISTS idx_bench_copy", prepare=False)
    return {
        "method": "seqscan",
        "top_k": config.top_k,
        "query_count": len(reference),
        "ids": reference,
    }


def _measure_queries(
    conn: psycopg.Connection,
    queries: Sequence[EvalQuery],
    vectors: Mapping[str, Sequence[float]],
    config: IndexBenchmarkConfig,
    exact: Mapping[str, Any],
    *,
    method: str,
    query_parameter: str,
    query_values: Sequence[int],
) -> dict[str, Any]:
    """Time every query at each query-time setting and compare with exact."""
    measurements: dict[str, Any] = {}
    exact_ids = exact["ids"]
    for value in query_values:
        key = f"{query_parameter}={value}"
        samples: list[float] = []
        recalls: list[float] = []
        returned_ids: dict[str, list[str]] = {}
        for query in queries:
            vector = Vector(list(vectors[query.query_id]))
            for _ in range(config.warmup_passes):
                _ann_once(conn, vector, config, method, value)
            for _ in range(config.repetitions):
                ids, latency = _ann_once(conn, vector, config, method, value)
                samples.append(latency)
                returned_ids[query.query_id] = ids
            recalls.append(recall_at_k(ids, exact_ids[query.query_id], config.top_k))
        summary = latency_summary(samples)
        measurements[key] = {
            "latency_ms": summary,
            "samples_ms": samples,
            "ann_recall_mean": statistics.fmean(recalls) if recalls else None,
            "ids": returned_ids,
        }
    return measurements


def _ann_once(
    conn: psycopg.Connection,
    vector: Vector,
    config: IndexBenchmarkConfig,
    method: str,
    value: int,
) -> tuple[list[str], float]:
    """Run one indexed query inside a transaction with the query-time setting."""
    with conn.transaction():
        conn.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (f"{config.statement_timeout_ms}ms",),
            prepare=False,
        )
        conn.execute("SET LOCAL enable_seqscan = off", prepare=False)
        conn.execute("SET LOCAL enable_indexscan = on", prepare=False)
        setting = "hnsw.ef_search" if method == "hnsw" else "ivfflat.probes"
        conn.execute(
            "SELECT set_config(%s, %s, true)",
            (setting, str(value)),
            prepare=False,
        )
        started = time.perf_counter()
        rows = conn.execute(
            _SEARCH_SQL.format(table="idx_bench_copy"),
            (vector, vector, config.top_k),
            prepare=False,
        ).fetchall()
        latency = (time.perf_counter() - started) * 1000.0
    return [str(row[0]) for row in rows], latency


def _explain(
    conn: psycopg.Connection,
    vector: Sequence[float],
    config: IndexBenchmarkConfig,
    index_name: str,
    *,
    method: str,
    query_parameter: str,
    value: int,
) -> dict[str, Any]:
    """Capture one sanitized EXPLAIN ANALYZE plan for the configuration."""
    with conn.transaction():
        conn.execute("SET LOCAL enable_seqscan = off", prepare=False)
        setting = "hnsw.ef_search" if method == "hnsw" else "ivfflat.probes"
        conn.execute(
            "SELECT set_config(%s, %s, true)", (setting, str(value)), prepare=False
        )
        rows = conn.execute(
            "EXPLAIN (ANALYZE, BUFFERS) " + _SEARCH_SQL.format(table="idx_bench_copy"),
            (Vector(list(vector)), Vector(list(vector)), config.top_k),
            prepare=False,
        ).fetchall()
    text = "\n".join(str(row[0]) for row in rows)
    sanitized = _sanitize_plan(text)
    if "Index Scan" in sanitized:
        kind = "index_scan"
    elif "Seq Scan" in sanitized:
        kind = "seq_scan"
    else:
        kind = "other"
    return {
        "kind": kind,
        "index_name": index_name,
        "query_parameter": query_parameter,
        "value": value,
        "text": sanitized,
    }


def _sanitize_plan(text: str) -> str:
    """Drop the bound query vector from EXPLAIN output and bound line length."""
    lines: list[str] = []
    for line in _VECTOR_LITERAL_RE.sub("'<vector>'::vector", text).splitlines():
        if "Sort Key:" in line:
            lines.append("  Sort Key: (omitted: contains the 1024-dim query vector)")
        elif len(line) > _PLAN_LINE_LIMIT:
            lines.append(line[: _PLAN_LINE_LIMIT - 3] + "...")
        else:
            lines.append(line)
    return "\n".join(lines)


def _create_copy_table(conn: psycopg.Connection, config: IndexBenchmarkConfig) -> int:
    """Create and populate the temporary per-configuration table.

    ``CREATE TABLE AS`` is a utility statement, so psycopg cannot bind its
    parameters; the strategy is therefore rendered as a validated SQL literal.

    Returns:
        The number of copied rows.
    """
    conn.execute("DROP TABLE IF EXISTS idx_bench_copy", prepare=False)
    conn.execute(
        "CREATE TEMP TABLE idx_bench_copy AS "
        "SELECT chunk_id, doc_id, chunk_strategy, embedding FROM chunks "
        f"WHERE chunk_strategy = {_sql_literal(config.chunk_strategy)} "
        "AND embedding IS NOT NULL",
        prepare=False,
    )
    if config.synthetic_scale > 1:
        extra_copies = config.synthetic_scale - 1
        conn.execute(
            "INSERT INTO idx_bench_copy (chunk_id, doc_id, chunk_strategy, embedding) "
            "SELECT chunk_id || '-syn' || g, doc_id, chunk_strategy, embedding "
            f"FROM idx_bench_copy CROSS JOIN generate_series(1, {extra_copies}) AS g",
            prepare=False,
        )
    rows = conn.execute("SELECT count(*) FROM idx_bench_copy").fetchone()
    if not rows or rows[0] < 1:
        raise RuntimeError(f"no chunks for strategy {config.chunk_strategy!r}")
    return int(rows[0])


def _build_temp_index(
    conn: psycopg.Connection,
    name: str,
    method: str,
    params: Mapping[str, int],
) -> float:
    """Build the index and return client-side build milliseconds."""
    if method == "hnsw":
        options = f"m = {int(params['m'])}, ef_construction = {int(params['ef_construction'])}"
    elif method == "ivfflat":
        options = f"lists = {int(params['lists'])}"
    else:
        raise ValueError(f"unsupported index method {method!r}")
    sql = (
        f"CREATE INDEX {name} ON idx_bench_copy USING {method} "
        f"(embedding vector_cosine_ops) WITH ({options})"
    )
    started = time.perf_counter()
    conn.execute(sql, prepare=False)
    return (time.perf_counter() - started) * 1000.0


def _index_size(conn: psycopg.Connection, name: str) -> tuple[int, str]:
    """Return the index size in bytes and pretty form."""
    row = conn.execute(
        "SELECT pg_relation_size(%s), pg_size_pretty(pg_relation_size(%s))",
        (name, name),
        prepare=False,
    ).fetchone()
    if row is None:
        raise RuntimeError(f"could not read size of {name!r}")
    return int(row[0]), str(row[1])


def _parameters(
    config: IndexBenchmarkConfig, queries: Sequence[EvalQuery]
) -> dict[str, Any]:
    """Return the parameters the report must record."""
    return {
        "queries": len(queries),
        "top_k": config.top_k,
        "repetitions": config.repetitions,
        "warmup_passes": config.warmup_passes,
        "statement_timeout_ms": config.statement_timeout_ms,
        "chunk_strategy": config.chunk_strategy,
        "synthetic_scale": config.synthetic_scale,
        "synthetic": config.synthetic_scale > 1,
        "hnsw_m_values": list(config.hnsw_m_values),
        "hnsw_ef_construction_values": list(config.hnsw_ef_construction_values),
        "hnsw_ef_search_values": list(config.hnsw_ef_search_values),
        "ivfflat_lists_values": list(config.ivfflat_lists_values),
        "ivfflat_probes_values": list(config.ivfflat_probes_values),
    }


def _limitations(config: IndexBenchmarkConfig) -> list[str]:
    """List the boundaries a reader must not cross when citing these numbers."""
    items = [
        "每个配置在临时表副本上建索引；不是对服务表 chunks 的直接查询。",
        "ANN Recall 是相对精确 Seq Scan Top-K 的保真度，不是语义 Recall。",
        "延迟为单进程、顺序、已预热；不含并发、GPU 或跨机器。",
        "IVFFlat 需要先有数据再建索引；本实验按此顺序执行。",
    ]
    if config.synthetic_scale > 1:
        items.insert(
            0,
            f"SYNTHETIC：向量按 {config.synthetic_scale} 倍复制，仅用于索引行为，"
            "不能用于语义或质量结论。",
        )
        items.insert(
            1,
            "复制产生大量等距向量（tie），ANN 与精确 Top-K 在 tie 中选行不同，"
            "因此即使索引正确，ANN Recall 也可能低于 1.0；这不是索引质量下降。",
        )
    if config.synthetic_scale == 1:
        items.append("150 行规模下参数差异可能不可观察；不能外推到 10 万级以上。")
    return items


def _require_range(values: Sequence[int], low: int, high: int, name: str) -> None:
    """Require every sweep value to be an int inside pgvector's range."""
    if not values:
        raise ValueError(f"{name} must not be empty")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must contain ints, got {type(value).__name__}")
        if not low <= value <= high:
            raise ValueError(f"{name} value {value} is outside {low}..{high}")


def _sql_literal(text: str) -> str:
    """Render a validated string as a SQL literal for utility statements."""
    return "'" + text.replace("'", "''") + "'"


def _format_params(params: Mapping[str, int]) -> str:
    """Format build parameters for Markdown."""
    return ", ".join(f"{key}={value}" for key, value in params.items())


def _fmt(value: float | int | None) -> str:
    """Format a number for Markdown, keeping ``None`` visible as ``-``."""
    if value is None:
        return "-"
    if isinstance(value, float) and not math.isfinite(value):
        return "-"
    return f"{float(value):.3f}"
