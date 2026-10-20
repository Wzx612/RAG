"""Index-parameter benchmark: HNSW ``m`` / ``ef_construction`` and IVFFlat.

Read-only with respect to the served schema. The sweep builds each index on a
per-configuration temporary copy of the selected chunk rows, so it never runs
``CREATE INDEX`` against ``chunks`` and never touches the bootstrap HNSW index.
See :mod:`src.evaluation.index_benchmark` for why that isolation is required.

Usage::

    python scripts/benchmark_index.py --dry-run
    python scripts/benchmark_index.py --limit 3 --repetitions 1
    python scripts/benchmark_index.py --m 8 16 32 --ef-construction 32 64 128
    python scripts/benchmark_index.py --synthetic-scale 100

``--synthetic-scale > 1`` copies every vector N times to give the indexes a
larger population. Such an artifact is labelled ``SYNTHETIC`` and must not be
cited as a semantic-quality result.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg  # noqa: E402
from src.config import PROJECT_ROOT, get_settings  # noqa: E402
from src.db import connection, scalar  # noqa: E402
from src.evaluation.artifacts import ArtifactExistsError, ensure_writable  # noqa: E402
from src.evaluation.dense_benchmark import (  # noqa: E402
    BenchmarkError,
    EvalQuery,
    load_eval_queries,
)
from src.evaluation.index_benchmark import (  # noqa: E402
    IndexBenchmarkConfig,
    blocked_result,
    run_index_benchmark,
    write_outputs,
)
from src.evaluation.metrics import latency_summary  # noqa: E402
from src.ingestion.embedder import SentenceTransformerEmbedder, validate_vectors  # noqa: E402

DATA_DIR = get_settings().resolve(get_settings().data_dir)
KB_PATH = DATA_DIR / "kb_documents.json"
QUERY_PATH = DATA_DIR / "eval_queries.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "unified" / "index_benchmark.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "unified" / "index_benchmark.md"


class _ArtifactWriter:
    """Write the artifact once, honouring ``--no-write`` and ``--overwrite``."""

    def __init__(
        self, output: Path, report: Path, *, overwrite: bool, enabled: bool
    ) -> None:
        """Store the targets and the two safety flags."""
        self.output = output
        self.report = report
        self.overwrite = overwrite
        self.enabled = enabled

    def __call__(self, payload: dict[str, Any]) -> None:
        """Write both artifacts, or print that the write was skipped."""
        if not self.enabled:
            print("--no-write/--dry-run: artifacts were not written")
            return
        write_outputs(payload, self.output, self.report, overwrite=self.overwrite)
        print(f"Wrote {self.output} and {self.report}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the benchmark's read-only options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--statement-timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--chunk-strategy",
        choices=("document", "fixed_window", "structured"),
        default="document",
    )
    parser.add_argument(
        "--synthetic-scale",
        type=int,
        default=1,
        help="Copy every vector N times (labelled SYNTHETIC; index behaviour only)",
    )
    parser.add_argument("--m", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--ef-construction", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--ef-search", type=int, nargs="+", default=[10, 40, 100])
    parser.add_argument("--lists", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--probes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--skip-hnsw", action="store_true")
    parser.add_argument("--skip-ivfflat", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N eval queries (smoke test; the full run uses 30)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print matrix and exit")
    parser.add_argument("--no-write", action="store_true", help="Do not write artifacts")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing artifacts at the selected paths",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def _package_version(name: str) -> str:
    """Return an installed distribution version or ``missing``."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def _sha256(path: Path) -> str:
    """Return the SHA256 of a data file, or ``missing``."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "missing"


def _index_inventory(conn: psycopg.Connection) -> list[str]:
    """Return the DDL of every index on ``chunks``, to prove none changed."""
    rows = conn.execute(
        "SELECT indexdef FROM pg_indexes WHERE tablename = 'chunks' ORDER BY indexname"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _read_environment(chunk_strategy: str) -> dict[str, Any]:
    """Collect read-only database facts and the run environment."""
    settings = get_settings()
    with connection() as conn:
        version_row = conn.execute(
            "SELECT coalesce(inet_server_addr()::text, 'unix-socket'), inet_server_port()"
        ).fetchone()
        strategies = conn.execute(
            "SELECT chunk_strategy, count(*) FROM chunks GROUP BY chunk_strategy "
            "ORDER BY chunk_strategy"
        ).fetchall()
        dimensions = [
            int(row[0])
            for row in conn.execute(
                "SELECT DISTINCT vector_dims(embedding) FROM chunks "
                "WHERE embedding IS NOT NULL"
            ).fetchall()
        ]
        strategy_chunks = conn.execute(
            "SELECT count(*) FROM chunks WHERE chunk_strategy = %s",
            (chunk_strategy,),
        ).fetchone()
        environment = {
            "postgres_version": str(scalar(conn, "SHOW server_version")),
            "vector_version": str(
                scalar(
                    conn,
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
                )
                or "missing"
            ),
            "server_address": (
                f"{version_row[0]}:{version_row[1]}" if version_row else "unknown"
            ),
            "documents": int(scalar(conn, "SELECT count(*) FROM documents") or 0),
            "chunks": int(scalar(conn, "SELECT count(*) FROM chunks") or 0),
            "chunk_strategies": {str(row[0]): int(row[1]) for row in strategies},
            "chunk_strategy": chunk_strategy,
            "strategy_chunks": int(strategy_chunks[0]) if strategy_chunks else 0,
            "embedding_dimensions": sorted(dimensions),
            "index_inventory": _index_inventory(conn),
        }
    environment.update(
        {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processors": platform.processor() or "unknown",
            "embedding_model": settings.embedding_model,
            "embedding_device": settings.embedding_device,
            "packages": {
                "sentence-transformers": _package_version("sentence-transformers"),
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
                "psycopg": _package_version("psycopg"),
                "pgvector": _package_version("pgvector"),
            },
            "synthetic": False,
        }
    )
    return environment


def _dataset_metadata(
    queries: list[EvalQuery], all_count: int, limit: int | None
) -> dict[str, Any]:
    """Describe the selected query set and the on-disk inputs."""
    return {
        "queries": len(queries),
        "eval_queries_on_disk": all_count,
        "query_limit": limit,
        "gold_sizes": sorted({len(query.relevant_doc_ids) for query in queries}),
        "hashes": {
            "kb_documents.json": _sha256(KB_PATH),
            "eval_queries.json": _sha256(QUERY_PATH),
        },
    }


def _build_config(args: argparse.Namespace) -> IndexBenchmarkConfig:
    """Translate parsed options into a validated configuration."""
    return IndexBenchmarkConfig(
        top_k=args.top_k,
        repetitions=args.repetitions,
        warmup_passes=args.warmup_passes,
        statement_timeout_ms=args.statement_timeout_ms,
        chunk_strategy=args.chunk_strategy,
        synthetic_scale=args.synthetic_scale,
        hnsw_m_values=tuple(args.m),
        hnsw_ef_construction_values=tuple(args.ef_construction),
        hnsw_ef_search_values=tuple(args.ef_search),
        ivfflat_lists_values=tuple(args.lists),
        ivfflat_probes_values=tuple(args.probes),
        include_hnsw=not args.skip_hnsw,
        include_ivfflat=not args.skip_ivfflat,
    )


def _estimate(
    config: IndexBenchmarkConfig, query_count: int, row_count: int
) -> dict[str, int]:
    """Estimate index builds and queries before anything expensive starts."""
    hnsw = (
        len(config.hnsw_m_values) * len(config.hnsw_ef_construction_values)
        if config.include_hnsw
        else 0
    )
    ivfflat = len(config.ivfflat_lists_values) if config.include_ivfflat else 0
    hnsw_queries = (
        hnsw
        * len(config.hnsw_ef_search_values)
        * query_count
        * (config.repetitions + config.warmup_passes)
    )
    ivfflat_queries = (
        ivfflat
        * len(config.ivfflat_probes_values)
        * query_count
        * (config.repetitions + config.warmup_passes)
    )
    return {
        "hnsw_indexes": hnsw,
        "ivfflat_indexes": ivfflat,
        "exact_queries": query_count,
        "indexed_queries": hnsw_queries + ivfflat_queries,
        "plans": hnsw + ivfflat,
        "rows_per_index": row_count * config.synthetic_scale,
    }


def _print_matrix(
    config: IndexBenchmarkConfig, estimated: dict[str, int]
) -> None:
    """Print the sweep matrix and the estimated workload."""
    print("Sweep matrix:")
    if config.include_hnsw:
        print(
            f"  HNSW    m={config.hnsw_m_values} x "
            f"ef_construction={config.hnsw_ef_construction_values} | "
            f"ef_search={config.hnsw_ef_search_values}"
        )
    if config.include_ivfflat:
        print(
            f"  IVFFlat lists={config.ivfflat_lists_values} | "
            f"probes={config.ivfflat_probes_values}"
        )
    print(f"  chunk_strategy={config.chunk_strategy} top_k={config.top_k} "
          f"synthetic_scale={config.synthetic_scale}")
    print(f"  repetitions={config.repetitions} warmup_passes={config.warmup_passes}")
    if config.synthetic_scale > 1:
        print(
            "  WARNING: SYNTHETIC vectors -- index behaviour only, "
            "not a semantic-quality result"
        )
    print("Estimated workload:")
    for key, value in estimated.items():
        print(f"  {key}: {value}")


def _print_environment(environment: dict[str, Any], dataset: dict[str, Any]) -> None:
    """Print the gate facts before a dry run or a measurement."""
    print("Environment:")
    for key in (
        "postgres_version",
        "vector_version",
        "server_address",
        "documents",
        "chunks",
        "chunk_strategies",
        "chunk_strategy",
        "strategy_chunks",
        "embedding_dimensions",
        "index_inventory",
        "embedding_model",
        "embedding_device",
    ):
        print(f"  {key}: {environment.get(key)}")
    print("Dataset:")
    for key in ("eval_queries_on_disk", "queries", "query_limit", "hashes"):
        print(f"  {key}: {dataset.get(key)}")


def _embed_queries(
    queries: list[EvalQuery],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Warm the embedding model and embed every query once."""
    embedder = SentenceTransformerEmbedder()
    started = time.perf_counter()
    _ = embedder.model
    model_load_ms = (time.perf_counter() - started) * 1000.0

    vectors: dict[str, list[float]] = {}
    samples: list[float] = []
    for query in queries:
        started = time.perf_counter()
        vector = embedder.embed_query(query.query)
        samples.append((time.perf_counter() - started) * 1000.0)
        validate_vectors([vector], embedder.dimension, source=query.query_id)
        vectors[query.query_id] = vector

    return vectors, {
        "model_load_ms": round(model_load_ms, 3),
        "per_query_ms": latency_summary(samples),
        "dimension": embedder.dimension,
        "model_id": get_settings().embedding_model,
    }


def _fmt_number(value: float | int | None, digits: int = 3) -> str:
    """Format a measurement for the console, keeping missing values visible."""
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _print_summary(result: dict[str, Any]) -> None:
    """Print a compact per-configuration summary."""
    print("\nResults:")
    print(
        f"  {'config':<30} {'setting':<14} {'recall':>7} {'p50ms':>9} "
        f"{'p95ms':>9} {'build_ms':>9} {'size':>8} status"
    )
    for record in result["configurations"]:
        if not record["query_params"]:
            print(
                f"  {record['index_name']:<30} {'-':<14} {'-':>7} {'-':>9} "
                f"{'-':>9} {'-':>9} {'-':>8} {record['status']}"
            )
            continue
        for setting, measurement in record["query_params"].items():
            latency = measurement["latency_ms"]
            print(
                f"  {record['index_name']:<30} {setting:<14} "
                f"{_fmt_number(measurement['ann_recall_mean']):>7} "
                f"{_fmt_number(latency['p50']):>9} "
                f"{_fmt_number(latency['p95']):>9} "
                f"{_fmt_number(record['build_ms'], 1):>9} "
                f"{record['size_pretty'] or '-':>8} {record['status']}"
            )
    print(f"  status: {result['status']} ({result['status_reason']})")


def main(argv: list[str] | None = None) -> int:
    """Run the gate, dry-run or full sweep, returning an exit code."""
    args = _parse_args(argv)
    try:
        config = _build_config(args)
    except ValueError as exc:
        print(f"BLOCKED: invalid configuration: {exc}")
        return 2
    if args.limit is not None and args.limit < 1:
        print("BLOCKED: --limit must be positive")
        return 2

    writer = _ArtifactWriter(
        args.output,
        args.report,
        overwrite=args.overwrite,
        enabled=not args.no_write and not args.dry_run,
    )
    if not args.no_write and not args.dry_run:
        try:
            ensure_writable((args.output, args.report), overwrite=args.overwrite)
        except ArtifactExistsError as exc:
            print(f"BLOCKED: {exc}")
            return 2

    try:
        environment = _read_environment(config.chunk_strategy)
    except (psycopg.Error, ConnectionError) as exc:
        message = f"PostgreSQL is not reachable; sweep not started: {exc}"
        print(f"BLOCKED: {message}")
        writer(blocked_result(message, config=config))
        return 2

    if config.synthetic_scale > 1:
        environment["synthetic"] = True

    gate_problems: list[str] = []
    if environment["vector_version"] == "missing":
        gate_problems.append("pgvector extension is missing")
    if environment["strategy_chunks"] < 1:
        gate_problems.append(
            f"no chunks for strategy {config.chunk_strategy!r}; run ingest first"
        )
    expected_dims = get_settings().embedding_dim
    if environment["embedding_dimensions"] != [expected_dims]:
        gate_problems.append(
            f"embedding dimensions={environment['embedding_dimensions']}, "
            f"expected [{expected_dims}]"
        )
    if gate_problems:
        message = "environment gate failed: " + "; ".join(gate_problems)
        print(f"BLOCKED: {message}")
        writer(blocked_result(message, config=config, environment=environment))
        return 2

    try:
        all_queries = load_eval_queries(QUERY_PATH)
    except BenchmarkError as exc:
        message = f"eval queries are not usable; sweep not started: {exc}"
        print(f"BLOCKED: {message}")
        writer(blocked_result(message, config=config, environment=environment))
        return 2

    selected = list(all_queries[: args.limit] if args.limit else all_queries)
    dataset = _dataset_metadata(selected, len(all_queries), args.limit)
    estimated = _estimate(config, len(selected), environment["strategy_chunks"])

    _print_environment(environment, dataset)
    _print_matrix(config, estimated)

    if args.dry_run:
        print("Dry run only: no model load and no index was built.")
        return 0

    print("\nWarming embedding model and embedding queries...")
    try:
        vectors, embedding = _embed_queries(selected)
    except Exception as exc:  # noqa: BLE001 -- a model failure blocks the sweep
        message = f"embedding failed; sweep not started: {type(exc).__name__}: {exc}"
        print(f"BLOCKED: {message}")
        writer(
            blocked_result(
                message, config=config, environment=environment, dataset=dataset
            )
        )
        return 2
    print(
        f"  model_load_ms={embedding['model_load_ms']:.2f} "
        f"dimension={embedding['dimension']}"
    )

    try:
        result = run_index_benchmark(
            selected,
            vectors,
            config,
            progress=lambda message: print(f"  {message}"),
            environment=environment,
            dataset=dataset,
        )
    except Exception as exc:  # noqa: BLE001 -- record the failure, never fake a run
        message = f"sweep aborted: {type(exc).__name__}: {exc}"
        print(f"BLOCKED: {message}")
        writer(
            blocked_result(
                message, config=config, environment=environment, dataset=dataset
            )
        )
        return 2

    result["embedding"] = embedding
    result["dataset"] = dataset
    result["environment"] = environment
    _print_summary(result)
    writer(result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
