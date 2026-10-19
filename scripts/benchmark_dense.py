"""Phase 4C: reproducible dense-retrieval benchmark.

Read-only orchestration around :mod:`src.evaluation.dense_benchmark`. It checks
the environment and dataset, warms the model, embeds every query once, runs the
exact reference plus the natural/HNSW arms, and writes the raw JSON and
Markdown report.

The script never writes to PostgreSQL, never changes a global GUC, never
rebuilds an index, and never edits the data files. Use ``--dry-run`` to print
the matrix and workload without loading the model or issuing benchmark queries.

Usage::

    python scripts/benchmark_dense.py --dry-run
    python scripts/benchmark_dense.py --limit 3 --repetitions 1
    python scripts/benchmark_dense.py --top-k 10 --ef-search 10 40 100

On Windows, set ``APEX_PG_HOST=127.0.0.1`` if ``localhost`` resolves to IPv6
first. The host fallback is documented in the generated report; the container
command is included there as well.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg  # noqa: E402
from src.config import PROJECT_ROOT, get_settings  # noqa: E402
from src.db import connection, current_ef_search, scalar  # noqa: E402
from src.evaluation.dense_benchmark import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkError,
    EvalQuery,
    build_configs,
    load_eval_queries,
    run_dense_benchmark,
    write_outputs,
)
from src.evaluation.metrics import percentile  # noqa: E402
from src.ingestion.embedder import SentenceTransformerEmbedder, validate_vectors  # noqa: E402
from src.ingestion.loader import inspect_source  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.schema.retrieval import DenseConfig  # noqa: E402

DATA_DIR = get_settings().resolve(get_settings().data_dir)
KB_PATH = DATA_DIR / "kb_documents.json"
QUERY_PATH = DATA_DIR / "eval_queries.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "dense_benchmark.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "dense_benchmark.md"


class _FixedVectorEmbedder:
    """Reuse one already-computed query vector for the production sanity check."""

    def __init__(self, vector: Sequence[float]) -> None:
        """Store the vector and its width."""
        self._vector = list(vector)
        self.dimension = len(self._vector)

    def embed_query(self, text: str) -> list[float]:
        """Return the stored vector, ignoring the text."""
        return list(self._vector)

    def embed_documents(self, texts: Sequence[str], on_batch: Any = None) -> list[list[float]]:
        """Return one stored vector per text; unused by DenseRetriever."""
        del on_batch
        return [list(self._vector) for _ in texts]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the benchmark's read-only options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=10, help="Top-K shared by all arms")
    parser.add_argument(
        "--ef-search",
        type=int,
        nargs="+",
        default=[10, 40, 100],
        help="hnsw.ef_search values for the forced-HNSW arms",
    )
    parser.add_argument("--repetitions", type=int, default=5, help="Timed passes per arm")
    parser.add_argument("--warmup-passes", type=int, default=1, help="Untimed passes per arm")
    parser.add_argument(
        "--statement-timeout-ms",
        type=int,
        default=5000,
        help="Transaction-local statement timeout",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N eval queries (smoke test; the full report uses 30)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print matrix and exit")
    parser.add_argument(
        "--docker-timeout",
        type=float,
        default=5.0,
        help="Seconds allowed for the read-only Docker CLI probe",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def _fmt_gb(free_bytes: int) -> str:
    """Format a byte count as GB with two decimals."""
    return f"{free_bytes / 1_000_000_000:.2f}GB"


def _disk_free() -> str:
    """Report free space for the project and, on Windows, the C and D drives."""
    paths = [("project", PROJECT_ROOT)]
    if platform.system() == "Windows":
        paths.extend([("C", Path("C:/")), ("D", Path("D:/"))])
    parts: list[str] = []
    for label, path in paths:
        try:
            parts.append(f"{label}={_fmt_gb(shutil.disk_usage(path).free)} free")
        except OSError:
            continue
    return "; ".join(parts) or "unavailable"


def _docker_status(timeout: float) -> str:
    """Probe the Docker CLI without starting, stopping or changing anything."""
    try:
        completed = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return "not installed"
    except subprocess.TimeoutExpired:
        return f"unresponsive (> {timeout:g}s); no container command executed"
    except OSError as exc:
        return f"probe failed: {type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().replace("\n", " ")
        return f"error rc={completed.returncode}: {detail[:200]}"
    return f"ok: server {completed.stdout.strip()}"


def _package_version(name: str) -> str:
    """Return an installed distribution version or ``missing``."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def _model_stack() -> str:
    """Summarize the Python packages that determine embedding behaviour."""
    return (
        f"sentence-transformers={_package_version('sentence-transformers')}, "
        f"torch={_package_version('torch')}, "
        f"transformers={_package_version('transformers')}, "
        f"numpy={_package_version('numpy')}, "
        f"psycopg={_package_version('psycopg')}, "
        f"pgvector={_package_version('pgvector')}"
    )


def _sha256(path: Path) -> str:
    """Return the SHA256 of a data file, or ``missing``."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "missing"


def _read_db_facts() -> dict[str, Any]:
    """Collect read-only database facts used by the report and the gates."""
    settings = get_settings()
    with connection() as conn:
        addr = conn.execute(
            "SELECT coalesce(inet_server_addr()::text, 'unix-socket'), inet_server_port()"
        ).fetchone()
        hnsw = conn.execute(
            """
            SELECT pg_get_indexdef(indexrelid)
            FROM pg_index
            JOIN pg_class ON pg_class.oid = pg_index.indexrelid
            JOIN pg_am ON pg_am.oid = pg_class.relam
            WHERE pg_index.indrelid = 'chunks'::regclass
              AND pg_am.amname = 'hnsw'
            """
        ).fetchone()
        reloptions = conn.execute(
            "SELECT reloptions FROM pg_class WHERE relname = 'chunks_embedding_hnsw_idx'"
        ).fetchone()
        documents = int(scalar(conn, "SELECT count(*) FROM documents") or 0)
        chunks = int(scalar(conn, "SELECT count(*) FROM chunks") or 0)
        distinct_docs = int(scalar(conn, "SELECT count(DISTINCT doc_id) FROM chunks") or 0)
        strategies = conn.execute(
            "SELECT chunk_strategy, count(*) FROM chunks GROUP BY chunk_strategy "
            "ORDER BY chunk_strategy"
        ).fetchall()
        dimensions = [
            int(row[0])
            for row in conn.execute(
                "SELECT DISTINCT vector_dims(embedding) FROM chunks WHERE embedding IS NOT NULL"
            ).fetchall()
        ]
        server_version = scalar(conn, "SHOW server_version")
        vector_version = scalar(
            conn, "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        data_directory = scalar(conn, "SHOW data_directory")
        ef_default = current_ef_search(conn)

    return {
        "postgres_version": str(server_version),
        "vector_version": str(vector_version) if vector_version else "missing",
        "server_address": f"{addr[0]}:{addr[1]}" if addr else "unknown",
        "data_directory": str(data_directory),
        "hnsw_index": str(hnsw[0]) if hnsw else "missing",
        "hnsw_parameters": ", ".join(str(option) for option in (reloptions[0] or []))
        if reloptions and reloptions[0]
        else "not recorded",
        "ef_search_default": ef_default,
        "documents": documents,
        "chunks": chunks,
        "distinct_chunk_docs": distinct_docs,
        "chunk_strategies": dict(strategies),
        "embedding_dimensions": sorted(dimensions),
        "embedding_model": settings.embedding_model,
        "embedding_device": settings.embedding_device,
    }


def _gold_missing(queries: Sequence[EvalQuery]) -> list[str]:
    """Return gold doc ids that are absent from the live ``documents`` table."""
    gold = sorted({doc_id for query in queries for doc_id in query.relevant_doc_ids})
    if not gold:
        return []
    with connection() as conn:
        rows = conn.execute(
            "SELECT doc_id FROM documents WHERE doc_id = ANY(%s)",
            (gold,),
        ).fetchall()
    found = {str(row[0]) for row in rows}
    return [doc_id for doc_id in gold if doc_id not in found]


def _post_run_counts() -> dict[str, Any]:
    """Read the row counts again, to prove the benchmark did not write."""
    with connection() as conn:
        return {
            "documents": int(scalar(conn, "SELECT count(*) FROM documents") or 0),
            "chunks": int(scalar(conn, "SELECT count(*) FROM chunks") or 0),
            "distinct_chunk_docs": int(
                scalar(conn, "SELECT count(DISTINCT doc_id) FROM chunks") or 0
            ),
            "embedding_dimensions": sorted(
                int(row[0])
                for row in conn.execute(
                    "SELECT DISTINCT vector_dims(embedding) FROM chunks WHERE embedding IS NOT NULL"
                ).fetchall()
            ),
        }


def _dataset_metadata(
    queries: Sequence[EvalQuery],
    all_query_count: int,
    limit: int | None,
    missing_gold: Sequence[str],
) -> dict[str, Any]:
    """Describe the selected query set and the on-disk files."""
    return {
        "queries": len(queries),
        "eval_queries_on_disk": all_query_count,
        "query_limit": limit,
        "gold_sizes": sorted({len(query.relevant_doc_ids) for query in queries}),
        "missing_gold_docs": list(missing_gold),
        "hashes": {
            "kb_documents.json": _sha256(KB_PATH),
            "eval_queries.json": _sha256(QUERY_PATH),
        },
        "post_run_counts": None,
        "data_integrity_ok": False,
    }


def _estimate_workload(
    query_count: int, configs: Sequence[BenchmarkConfig], repetitions: int, warmups: int
) -> dict[str, int]:
    """Estimate benchmark statements before anything expensive starts."""
    return {
        "exact_reference_queries": query_count,
        "plans": len(configs),
        "warmup_queries": len(configs) * query_count * warmups,
        "timed_queries": len(configs) * query_count * repetitions,
        "database_statements": (
            query_count
            + len(configs)
            + len(configs) * query_count * warmups
            + len(configs) * query_count * repetitions
        ),
        "embedding_calls": 2 + query_count,
    }


def _print_environment(env: dict[str, Any], dataset: dict[str, Any]) -> None:
    """Print the gate facts before a dry run or a measurement."""
    print("Environment:")
    for key in (
        "postgres_version",
        "vector_version",
        "server_address",
        "data_directory",
        "hnsw_index",
        "hnsw_parameters",
        "ef_search_default",
        "documents",
        "chunks",
        "distinct_chunk_docs",
        "chunk_strategies",
        "embedding_dimensions",
        "embedding_model",
        "embedding_device",
        "python_version",
        "model_stack",
        "docker_status",
        "execution_environment",
        "disk_free",
    ):
        print(f"  {key}: {env.get(key)}")
    print("Dataset:")
    for key in (
        "eval_queries_on_disk",
        "queries",
        "query_limit",
        "gold_sizes",
        "missing_gold_docs",
        "hashes",
    ):
        print(f"  {key}: {dataset.get(key)}")


def _print_matrix(configs: Sequence[BenchmarkConfig], estimated: dict[str, int]) -> None:
    """Print the experiment matrix and the statement budget."""
    print("Experiment matrix:")
    for config in configs:
        control = {
            "exact": "index/bitmap/indexonly off, seqscan on",
            "natural": "none",
            "hnsw": "seqscan off",
        }[config.mode]
        print(
            f"  {config.name:<16} mode={config.mode:<8} ef={config.ef_search:<4} "
            f"K={config.top_k:<4} planner={control}"
        )
    print("Estimated workload:")
    for key, value in estimated.items():
        print(f"  {key}: {value}")


def _write_blocked(
    message: str,
    output: Path,
    report: Path,
    *,
    environment: dict[str, Any] | None = None,
    dataset: dict[str, Any] | None = None,
) -> None:
    """Write a clearly labelled blocked artifact instead of a fake result."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "BLOCKED",
        "error": message,
        "environment": environment or {},
        "dataset": dataset or {},
        "parameters": {},
        "configs": [],
        "totals": {"errors": 0, "timeouts": 0, "skipped": 0, "invalid_configs": 0},
    }
    write_outputs(payload, output, report)
    print(f"BLOCKED: {message}")
    print(f"Wrote blocked artifacts to {output} and {report}")


def _embed_queries(
    queries: Sequence[EvalQuery],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Warm the model, embed every query, and return vectors plus timings."""
    embedder = SentenceTransformerEmbedder()
    started = time.perf_counter()
    _ = embedder.model
    model_load_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    first_vector = embedder.embed_query(queries[0].query)
    first_encode_ms = (time.perf_counter() - started) * 1000.0
    validate_vectors([first_vector], embedder.dimension, source="warmup-1")

    started = time.perf_counter()
    second_vector = embedder.embed_query(queries[0].query)
    second_encode_ms = (time.perf_counter() - started) * 1000.0
    validate_vectors([second_vector], embedder.dimension, source="warmup-2")

    vectors: dict[str, list[float]] = {}
    samples: list[float] = []
    for query in queries:
        started = time.perf_counter()
        vector = embedder.embed_query(query.query)
        samples.append((time.perf_counter() - started) * 1000.0)
        validate_vectors([vector], embedder.dimension, source=query.query_id)
        vectors[query.query_id] = vector

    timing = {
        "model_load_ms": model_load_ms,
        "first_encode_ms": first_encode_ms,
        "second_warm_encode_ms": second_encode_ms,
        "per_query_ms": {
            "n": len(samples),
            "min": min(samples) if samples else None,
            "p50": percentile(samples, 50) if samples else None,
            "p95": percentile(samples, 95) if samples else None,
            "max": max(samples) if samples else None,
            "mean": sum(samples) / len(samples) if samples else None,
        },
        "dimension": embedder.dimension,
        "samples_ms": samples,
    }
    return vectors, timing


def _production_sanity(
    query: EvalQuery,
    vector: Sequence[float],
    config: BenchmarkConfig,
    benchmark_keys: Sequence[str],
) -> dict[str, Any]:
    """Compare the benchmark's natural arm with the production DenseRetriever.

    The same pre-computed vector is injected, so this isolates the SQL/GUC/row
    mapping path from model variance. It is a consistency check, not a second
    measurement.
    """
    retriever = DenseRetriever(
        DenseConfig(top_n=config.top_k, ef_search=config.ef_search),
        embedder=_FixedVectorEmbedder(vector),
    )
    hits = retriever.retrieve(query.query)
    production_keys = [f"{hit.chunk_strategy}:{hit.chunk_id}" for hit in hits]
    return {
        "query_id": query.query_id,
        "production_keys": production_keys,
        "benchmark_keys": list(benchmark_keys),
        "match": production_keys == list(benchmark_keys),
    }


def _print_summary(result: dict[str, Any]) -> None:
    """Print a compact result table and the status."""
    print("\nResults:")
    print(
        f"  {'arm':<16} {'plan':<10} {'P50ms':>9} {'P95ms':>9} "
        f"{'ANNRecall':>10} {'orderEq':>8} {'errors':>7} {'timeouts':>8}"
    )
    for config in result.get("configs", []):
        latency = config["latency"]["query_ms"]
        quality = config["quality"]
        recall = quality["recall_vs_exact_mean"]
        recall_text = "-" if recall is None else recall
        print(
            f"  {config['name']:<16} {str((config.get('plan') or {}).get('kind', '-')):<10} "
            f"{latency['p50'] if latency['p50'] is not None else '-':>9} "
            f"{latency['p95'] if latency['p95'] is not None else '-':>9} "
            f"{recall_text:>10} "
            f"{quality['order_equal_queries']:>8} {config['errors']:>7} {config['timeouts']:>8}"
        )
    print(f"  status: {result.get('status')}")
    for issue in result.get("issues", []):
        print(f"  issue: {issue}")


def main(argv: list[str] | None = None) -> int:
    """Run the gate, dry-run or full benchmark, returning an exit code."""
    args = _parse_args(argv)
    if args.top_k < 1:
        print("BLOCKED: --top-k must be positive")
        return 2
    if args.limit is not None and args.limit < 1:
        print("BLOCKED: --limit must be positive")
        return 2

    try:
        env = _read_db_facts()
        env.update(
            {
                "python_version": platform.python_version(),
                "model_stack": _model_stack(),
                "docker_status": _docker_status(args.docker_timeout),
                "execution_environment": "host Python fallback",
                "container_execution": "not run (host fallback)",
                "disk_free": _disk_free(),
            }
        )
    except (psycopg.Error, ConnectionError) as exc:
        message = f"PostgreSQL is not reachable; benchmark not started: {exc}"
        _write_blocked(message, args.output, args.report)
        return 2

    try:
        report = inspect_source()
    except Exception as exc:  # noqa: BLE001 -- surface the corpus problem as a gate
        message = f"knowledge base is not usable; benchmark not started: {exc}"
        _write_blocked(message, args.output, args.report, environment=env)
        return 2
    if not report.ok:
        message = (
            f"knowledge base has {len(report.problems)} invalid record(s); benchmark not started"
        )
        _write_blocked(message, args.output, args.report, environment=env)
        return 2

    try:
        all_queries = load_eval_queries(QUERY_PATH)
    except BenchmarkError as exc:
        message = f"eval queries are not usable; benchmark not started: {exc}"
        _write_blocked(message, args.output, args.report, environment=env)
        return 2

    missing_gold = _gold_missing(all_queries)
    if missing_gold:
        message = (
            f"{len(missing_gold)} gold doc id(s) are absent from the database: "
            f"{', '.join(missing_gold[:10])}"
        )
        data = _dataset_metadata(all_queries, len(all_queries), args.limit, missing_gold)
        _write_blocked(message, args.output, args.report, environment=env, dataset=data)
        return 2

    expected_documents = get_settings().expected_document_count
    gate_problems: list[str] = []
    if len(report.documents) != expected_documents:
        gate_problems.append(
            f"on-disk documents={len(report.documents)}, expected {expected_documents}"
        )
    if env["documents"] != expected_documents:
        gate_problems.append(
            f"database documents={env['documents']}, expected {expected_documents}"
        )
    if env["chunks"] != expected_documents:
        gate_problems.append(f"database chunks={env['chunks']}, expected {expected_documents}")
    if env["distinct_chunk_docs"] != expected_documents:
        gate_problems.append(
            f"distinct chunk doc_ids={env['distinct_chunk_docs']}, expected {expected_documents}"
        )
    if env["embedding_dimensions"] != [get_settings().embedding_dim]:
        gate_problems.append(f"embedding dimensions={env['embedding_dimensions']}, expected [1024]")
    if env["hnsw_index"] == "missing":
        gate_problems.append("HNSW index is missing")
    if env["vector_version"] == "missing":
        gate_problems.append("pgvector extension is missing")
    if gate_problems:
        message = "environment gate failed: " + "; ".join(gate_problems)
        data = _dataset_metadata(all_queries, len(all_queries), args.limit, missing_gold)
        _write_blocked(message, args.output, args.report, environment=env, dataset=data)
        return 2

    selected = list(all_queries[: args.limit] if args.limit else all_queries)
    configs = build_configs(args.top_k, args.ef_search)
    dataset = _dataset_metadata(selected, len(all_queries), args.limit, missing_gold)
    estimated = _estimate_workload(len(selected), configs, args.repetitions, args.warmup_passes)

    _print_environment(env, dataset)
    _print_matrix(configs, estimated)

    if args.dry_run:
        print("Dry run only: no model load and no benchmark queries were executed.")
        return 0

    print("\nWarming embedding model and embedding queries...")
    try:
        vectors, embedding = _embed_queries(selected)
    except Exception as exc:  # noqa: BLE001 -- a model failure blocks the benchmark
        message = f"embedding failed; benchmark not started: {type(exc).__name__}: {exc}"
        _write_blocked(message, args.output, args.report, environment=env, dataset=dataset)
        return 2
    print(
        f"model_load_ms={embedding['model_load_ms']:.2f} "
        f"first_encode_ms={embedding['first_encode_ms']:.2f} "
        f"second_encode_ms={embedding['second_warm_encode_ms']:.2f} "
        f"query_p50_ms={embedding['per_query_ms']['p50']:.2f} "
        f"query_p95_ms={embedding['per_query_ms']['p95']:.2f}"
    )

    try:
        result = run_dense_benchmark(
            selected,
            vectors,
            configs,
            repetitions=args.repetitions,
            warmup_passes=args.warmup_passes,
            statement_timeout_ms=args.statement_timeout_ms,
            progress=lambda message: print(f"  {message}"),
        )
    except BenchmarkError as exc:
        message = f"benchmark could not establish a valid baseline: {exc}"
        _write_blocked(message, args.output, args.report, environment=env, dataset=dataset)
        return 2

    natural = next((config for config in result["configs"] if config["mode"] == "natural"), None)
    sanity: dict[str, Any] | None = None
    if natural is not None and natural["per_query"]:
        first_query = selected[0]
        benchmark_keys = natural["per_query"][0].get("keys", [])
        try:
            sanity = _production_sanity(
                first_query,
                vectors[first_query.query_id],
                next(config for config in configs if config.mode == "natural"),
                benchmark_keys,
            )
        except (psycopg.Error, ConnectionError) as exc:
            sanity = {"query_id": first_query.query_id, "error": str(exc), "match": None}

    post = _post_run_counts()
    dataset["post_run_counts"] = post
    dataset["data_integrity_ok"] = (
        post["documents"] == env["documents"]
        and post["chunks"] == env["chunks"]
        and post["distinct_chunk_docs"] == env["distinct_chunk_docs"]
        and post["embedding_dimensions"] == env["embedding_dimensions"]
    )

    result["environment"] = env
    result["dataset"] = dataset
    result["embedding"] = embedding
    result["execution"] = {
        "environment": "host Python fallback",
        "container_execution": "not run",
        "docker_status": env["docker_status"],
        "production_sanity": sanity,
    }

    issues: list[str] = []
    totals = result["totals"]
    if totals["errors"]:
        issues.append(f"{totals['errors']} error(s) recorded")
    if totals["timeouts"]:
        issues.append(f"{totals['timeouts']} timeout(s) recorded")
    if totals["invalid_configs"]:
        issues.append(f"{totals['invalid_configs']} invalid config(s)")
    if not dataset["data_integrity_ok"]:
        issues.append("post-run row counts changed")
    if sanity is not None and sanity.get("match") is not True:
        issues.append(
            "benchmark natural arm could not be confirmed against production DenseRetriever"
        )
    result["issues"] = issues
    result["status"] = "PASS" if not issues else "ISSUES"

    write_outputs(result, args.output, args.report)
    _print_summary(result)
    print(f"Wrote {args.output} and {args.report}")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
