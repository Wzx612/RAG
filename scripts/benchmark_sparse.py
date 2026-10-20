"""Phase 5A: reproducible sparse-retrieval benchmark.

Read-only orchestration around :mod:`src.evaluation.sparse_benchmark`. It checks
the environment and dataset, runs the production full-text path over the
labelled queries, optionally compares candidates with the Phase 4C dense
artifact, probes Chinese behaviour, and writes the raw JSON and Markdown report.

The script never writes to PostgreSQL, never changes a global GUC, never
rebuilds an index, and never edits the data files. Use ``--dry-run`` to print
the environment and workload without issuing benchmark queries.

Usage::

    python scripts/benchmark_sparse.py --dry-run
    python scripts/benchmark_sparse.py --limit 3 --repetitions 1
    python scripts/benchmark_sparse.py --top-k 10 --repetitions 5

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
from collections.abc import Sequence
from datetime import UTC, datetime
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
from src.evaluation.sparse_benchmark import (  # noqa: E402
    attach_query_reference,
    build_config,
    dense_overlap,
    run_sparse_benchmark,
    write_outputs,
)
from src.ingestion.loader import inspect_source  # noqa: E402
from src.retrieval import RetrievalError  # noqa: E402
from src.retrieval.sparse import SparseRetriever, _cjk_bigrams  # noqa: E402
from src.schema.retrieval import QueryMode, SparseConfig  # noqa: E402

DATA_DIR = get_settings().resolve(get_settings().data_dir)
KB_PATH = DATA_DIR / "kb_documents.json"
QUERY_PATH = DATA_DIR / "eval_queries.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "sparse_benchmark.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "sparse_benchmark.md"
DEFAULT_DENSE_ARTIFACT = PROJECT_ROOT / "reports" / "dense_benchmark.json"


class _ArtifactWriter:
    """Write this benchmark's artifacts once, honouring the safety flags.

    ``--no-write`` suppresses every write, including blocked artifacts, so a
    smoke run cannot touch the historical reports. ``--overwrite`` is the only
    way to replace an existing artifact.
    """

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
            print("--no-write: artifacts were not written")
            return
        write_outputs(payload, self.output, self.report, overwrite=self.overwrite)
        print(f"Wrote {self.output} and {self.report}")

# Synthetic Chinese probes. They are deliberately not in data/eval_queries.json;
# they measure fallback behaviour on an English corpus, not Chinese quality.
CHINESE_PROBES: tuple[str, ...] = (
    "\u4f59\u989d\u8f6c\u8d26\u624b\u7eed\u8d39",  # 余额转账手续费
    "\u4fe1\u7528\u5361\u5e74\u8d39\u51cf\u514d\u653f\u7b56",  # 信用卡年费减免政策
    "\u5883\u5916\u6d88\u8d39\u8fd4\u73b0",  # 境外消费返现
    "\u4fe1\u7528",  # 信用
    "balance transfer fee \u4f59\u989d",  # mixed Chinese/English
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the benchmark's read-only options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for sparse retrieval")
    parser.add_argument(
        "--query-mode",
        choices=("or", "and"),
        default="or",
        help="Plain-query semantics: recall-oriented OR (default) or strict AND",
    )
    parser.add_argument(
        "--chunk-strategy",
        choices=("document", "fixed_window", "structured"),
        default="document",
        help="Chunking strategy every retrieval is restricted to",
    )
    parser.add_argument("--repetitions", type=int, default=5, help="Timed passes per query")
    parser.add_argument("--warmup-passes", type=int, default=1, help="Untimed passes per query")
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
    parser.add_argument("--dry-run", action="store_true", help="Print environment and exit")
    parser.add_argument("--no-write", action="store_true", help="Do not write report files")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing report files at the selected paths",
    )
    parser.add_argument(
        "--docker-timeout",
        type=float,
        default=5.0,
        help="Seconds allowed for the read-only Docker CLI probe",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--dense-artifact",
        type=Path,
        default=DEFAULT_DENSE_ARTIFACT,
        help="Phase 4C raw JSON used only for candidate-overlap reporting",
    )
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


def _sha256(path: Path) -> str:
    """Return the SHA256 of a data file, or ``missing``."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "missing"


def _read_db_facts(chunk_strategy: str = "document") -> dict[str, Any]:
    """Collect read-only database facts used by the report and the gates.

    The gate counts are scoped to ``chunk_strategy``: several chunkings may
    coexist in ``chunks``, and a global count would reject a perfectly valid
    single-strategy run. Global counts stay in the report as context.
    """
    settings = get_settings()
    with connection() as conn:
        addr = conn.execute(
            "SELECT coalesce(inet_server_addr()::text, 'unix-socket'), inet_server_port()"
        ).fetchone()
        documents = int(scalar(conn, "SELECT count(*) FROM documents") or 0)
        chunks = int(scalar(conn, "SELECT count(*) FROM chunks") or 0)
        distinct_docs = int(scalar(conn, "SELECT count(DISTINCT doc_id) FROM chunks") or 0)
        strategy_counts = conn.execute(
            "SELECT count(*), count(DISTINCT doc_id) FROM chunks WHERE chunk_strategy = %s",
            (chunk_strategy,),
        ).fetchone()
        strategy_chunks, strategy_docs = (
            (int(strategy_counts[0]), int(strategy_counts[1]))
            if strategy_counts is not None
            else (0, 0)
        )
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
        fts_config = conn.execute(
            "SELECT cfgname FROM pg_ts_config WHERE cfgname = 'english'"
        ).fetchone()
        tsv_expression = conn.execute(
            """
            SELECT pg_get_expr(adbin, adrelid)
            FROM pg_attrdef
            WHERE adrelid = 'chunks'::regclass
              AND adnum = (
                  SELECT attnum FROM pg_attribute
                  WHERE attrelid = 'chunks'::regclass AND attname = 'tsv'
              )
            """
        ).fetchone()
        gin_index = conn.execute(
            """
            SELECT pg_get_indexdef(indexrelid)
            FROM pg_index
            JOIN pg_class ON pg_class.oid = pg_index.indexrelid
            JOIN pg_am ON pg_am.oid = pg_class.relam
            WHERE pg_index.indrelid = 'chunks'::regclass
              AND pg_am.amname = 'gin'
              AND pg_get_indexdef(indexrelid) LIKE '%tsv%'
            """
        ).fetchone()
        chunk_columns = [
            str(row[0])
            for row in conn.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'chunks'
                ORDER BY ordinal_position
                """
            ).fetchall()
        ]
        lexemes = conn.execute(
            "SELECT websearch_to_tsquery('english', %s)::text",
            ("balance transfer fee",),
        ).fetchone()

    return {
        "postgres_version": str(server_version),
        "vector_version": str(vector_version) if vector_version else "missing",
        "server_address": f"{addr[0]}:{addr[1]}" if addr else "unknown",
        "data_directory": str(data_directory),
        "fts_config": str(fts_config[0]) if fts_config else "missing",
        "tsv_expression": str(tsv_expression[0]) if tsv_expression else "missing",
        "gin_index": str(gin_index[0]) if gin_index else "missing",
        "chunk_columns": ", ".join(chunk_columns),
        "chunk_strategy": chunk_strategy,
        "documents": documents,
        "chunks": chunks,
        "distinct_chunk_docs": distinct_docs,
        "strategy_chunks": strategy_chunks,
        "strategy_docs": strategy_docs,
        "chunk_strategies": dict(strategies),
        "embedding_dimensions": sorted(dimensions),
        "embedding_model": settings.embedding_model,
        "english_tsquery_example": str(lexemes[0]) if lexemes else "missing",
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


def _post_run_counts(chunk_strategy: str = "document") -> dict[str, Any]:
    """Read the row counts again, to prove the benchmark did not write.

    The scoped counts are the ones the gate compares: unrestricted global
    counts would move whenever an unrelated chunking strategy is written.
    """
    with connection() as conn:
        strategy_counts = conn.execute(
            "SELECT count(*), count(DISTINCT doc_id) FROM chunks WHERE chunk_strategy = %s",
            (chunk_strategy,),
        ).fetchone()
        return {
            "documents": int(scalar(conn, "SELECT count(*) FROM documents") or 0),
            "chunks": int(scalar(conn, "SELECT count(*) FROM chunks") or 0),
            "distinct_chunk_docs": int(
                scalar(conn, "SELECT count(DISTINCT doc_id) FROM chunks") or 0
            ),
            "strategy_chunks": int(strategy_counts[0]) if strategy_counts else 0,
            "strategy_docs": int(strategy_counts[1]) if strategy_counts else 0,
            "embedding_dimensions": sorted(
                int(row[0])
                for row in conn.execute(
                    "SELECT DISTINCT vector_dims(embedding) FROM chunks "
                    "WHERE embedding IS NOT NULL"
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
    hashes = {
        "kb_documents.json": _sha256(KB_PATH),
        "eval_queries.json": _sha256(QUERY_PATH),
    }
    return {
        "queries": len(queries),
        "eval_queries_on_disk": all_query_count,
        "query_limit": limit,
        "gold_sizes": sorted({len(query.relevant_doc_ids) for query in queries}),
        "missing_gold_docs": list(missing_gold),
        "hashes": hashes,
        "hashes_after": None,
        "post_run_counts": None,
        "data_integrity_ok": False,
    }


def _print_environment(env: dict[str, Any], dataset: dict[str, Any]) -> None:
    """Print the gate facts before a dry run or a measurement."""
    print("Environment:")
    for key in (
        "postgres_version",
        "vector_version",
        "server_address",
        "data_directory",
        "fts_config",
        "tsv_expression",
        "gin_index",
        "chunk_strategy",
        "documents",
        "chunks",
        "distinct_chunk_docs",
        "strategy_chunks",
        "strategy_docs",
        "chunk_strategies",
        "embedding_dimensions",
        "english_tsquery_example",
        "python_version",
        "psycopg_version",
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


def _print_workload(estimated: dict[str, int]) -> None:
    """Print the statement budget before anything expensive starts."""
    print("Estimated workload:")
    for key, value in estimated.items():
        print(f"  {key}: {value}")


def _estimate_workload(query_count: int, repetitions: int, warmups: int) -> dict[str, int]:
    """Estimate benchmark statements before anything expensive starts."""
    return {
        "warmup_queries": query_count * warmups,
        "timed_queries": query_count * repetitions,
        "database_statements": query_count * (warmups + repetitions),
        "chinese_probes": len(CHINESE_PROBES) + 1,
    }


def _write_blocked(
    message: str,
    writer: _ArtifactWriter,
    *,
    environment: dict[str, Any] | None = None,
    dataset: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
) -> None:
    """Write a clearly labelled blocked artifact instead of a fake result."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "BLOCKED",
        "error": message,
        "environment": environment or {},
        "dataset": dataset or {},
        "parameters": parameters or {},
        "retrieval": {
            "path_counts": {},
            "per_query": [],
            "quality": {},
            "latency": {"query_ms": {}, "db_ms": {}},
            "focus_queries": {},
        },
        "dense_overlap": {"available": False, "reason": "benchmark blocked"},
        "chinese_probe": {"available": False, "reason": "benchmark blocked"},
        "totals": {
            "evaluated_queries": 0,
            "missing_queries": 0,
            "errors": 0,
            "timeouts": 0,
            "warmup_errors": 0,
            "quality_mismatch_samples": 0,
            "path_counts": {},
        },
        "issues": [message],
    }
    print(f"BLOCKED: {message}")
    writer(payload)


def _production_sanity(
    query: EvalQuery,
    top_k: int,
    benchmark_keys: Sequence[str],
    query_mode: QueryMode,
    chunk_strategy: str = "document",
) -> dict[str, Any]:
    """Compare the benchmark SQL path with the production ``SparseRetriever``."""
    retriever = SparseRetriever(
        SparseConfig(top_n=top_k, query_mode=query_mode, chunk_strategy=chunk_strategy)
    )
    hits = retriever.retrieve(query.query)
    production_keys = [f"{hit.chunk_strategy}:{hit.chunk_id}" for hit in hits]
    return {
        "query_id": query.query_id,
        "production_keys": production_keys,
        "benchmark_keys": list(benchmark_keys),
        "match": production_keys == list(benchmark_keys),
    }


def _english_fts_match_count(query: str, chunk_strategy: str = "document") -> int:
    """Count chunks matched by the English FTS parser for one probe."""
    with connection() as conn:
        row = conn.execute(
            "SELECT count(*) FROM chunks "
            "WHERE tsv @@ websearch_to_tsquery('english', %s) AND chunk_strategy = %s",
            (query, chunk_strategy),
        ).fetchone()
        return int(row[0]) if row else 0


def _chinese_probe(
    top_k: int,
    query_mode: QueryMode = "or",
    chunk_strategy: str = "document",
) -> dict[str, Any]:
    """Run synthetic Chinese probes and record honest, separate results."""
    retriever = SparseRetriever(
        SparseConfig(top_n=top_k, query_mode=query_mode, chunk_strategy=chunk_strategy)
    )
    records: list[dict[str, Any]] = []
    for query in CHINESE_PROBES:
        cjk_terms = _cjk_bigrams(query)
        try:
            hits = retriever.retrieve(query, top_k=top_k)
            records.append(
                {
                    "query": query,
                    "path": "cjk" if cjk_terms else "fts",
                    "result_count": len(hits),
                    "doc_ids": [hit.doc_id for hit in hits],
                    "scores": [hit.score for hit in hits],
                    "error": None,
                }
            )
        except (RetrievalError, ValueError) as exc:
            records.append(
                {
                    "query": query,
                    "path": "cjk" if cjk_terms else "fts",
                    "result_count": None,
                    "doc_ids": [],
                    "scores": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    first_probe = CHINESE_PROBES[0]
    with connection() as conn:
        lexemes = conn.execute(
            "SELECT to_tsvector('english', %s)::text, websearch_to_tsquery('english', %s)::text",
            (first_probe, first_probe),
        ).fetchone()
    control_hits = retriever.retrieve("balance transfer fee", top_k=top_k)
    control = {
        "query": "balance transfer fee",
        "path": "fts",
        "result_count": len(control_hits),
        "doc_ids": [hit.doc_id for hit in control_hits],
    }
    return {
        "available": True,
        "note": (
            "Synthetic probes, not part of the labelled 30-query set. The corpus is "
            "English; zero results do not demonstrate Chinese retrieval quality."
        ),
        "probes": [
            {
                **record,
                "english_fts_match_count": _english_fts_match_count(
                    record["query"], chunk_strategy
                ),
            }
            for record in records
        ],
        "lexeme_evidence": {
            "text": str(lexemes[0]) if lexemes else "missing",
            "query": str(lexemes[1]) if lexemes else "missing",
        },
        "bigram_example": _cjk_bigrams(first_probe),
        "control": control,
    }


def _print_summary(result: dict[str, Any]) -> None:
    """Print a compact result table and the status."""
    retrieval = result.get("retrieval", {})
    quality = retrieval.get("quality", {})
    latency = retrieval.get("latency", {})
    print("\nResults:")
    print(
        f"  queries={quality.get('evaluated_queries')}/"
        f"{result.get('parameters', {}).get('queries')} "
        f"semantic_recall_mean={quality.get('semantic_recall_mean')} "
        f"empty={quality.get('empty_result_queries')} "
        f"P50ms={latency.get('query_ms', {}).get('p50')} "
        f"P95ms={latency.get('query_ms', {}).get('p95')} "
        f"errors={result.get('totals', {}).get('errors')} "
        f"timeouts={result.get('totals', {}).get('timeouts')}"
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

    writer = _ArtifactWriter(
        args.output,
        args.report,
        overwrite=args.overwrite,
        enabled=not args.no_write,
    )
    if not args.no_write and not args.dry_run:
        try:
            ensure_writable((args.output, args.report), overwrite=args.overwrite)
        except ArtifactExistsError as exc:
            print(f"BLOCKED: {exc}")
            return 2

    try:
        config = build_config(
            args.top_k,
            args.repetitions,
            args.warmup_passes,
            args.statement_timeout_ms,
            args.query_mode,
            args.chunk_strategy,
        )
    except BenchmarkError as exc:
        print(f"BLOCKED: {exc}")
        return 2

    try:
        env = _read_db_facts(args.chunk_strategy)
        env.update(
            {
                "python_version": platform.python_version(),
                "psycopg_version": _package_version("psycopg"),
                "docker_status": _docker_status(args.docker_timeout),
                "execution_environment": "host Python fallback",
                "container_execution": "not run (host fallback)",
                "disk_free": _disk_free(),
            }
        )
    except (psycopg.Error, ConnectionError) as exc:
        message = f"PostgreSQL is not reachable; benchmark not started: {exc}"
        _write_blocked(message, writer)
        return 2

    try:
        source_report = inspect_source()
    except Exception as exc:  # noqa: BLE001 -- surface the corpus problem as a gate
        message = f"knowledge base is not usable; benchmark not started: {exc}"
        _write_blocked(message, writer, environment=env)
        return 2
    if not source_report.ok:
        message = (
            f"knowledge base has {len(source_report.problems)} invalid record(s); "
            "benchmark not started"
        )
        _write_blocked(message, writer, environment=env)
        return 2

    try:
        all_queries = load_eval_queries(QUERY_PATH)
    except BenchmarkError as exc:
        message = f"eval queries are not usable; benchmark not started: {exc}"
        _write_blocked(message, writer, environment=env)
        return 2

    missing_gold = _gold_missing(all_queries)
    if missing_gold:
        message = (
            f"{len(missing_gold)} gold doc id(s) are absent from the database: "
            f"{', '.join(missing_gold[:10])}"
        )
        data = _dataset_metadata(all_queries, len(all_queries), args.limit, missing_gold)
        _write_blocked(message, writer, environment=env, dataset=data)
        return 2

    expected_documents = get_settings().expected_document_count
    gate_problems: list[str] = []
    if len(source_report.documents) != expected_documents:
        gate_problems.append(
            f"on-disk documents={len(source_report.documents)}, expected {expected_documents}"
        )
    if env["documents"] != expected_documents:
        gate_problems.append(
            f"database documents={env['documents']}, expected {expected_documents}"
        )
    if env["strategy_chunks"] < expected_documents:
        gate_problems.append(
            f"chunks for strategy {env['chunk_strategy']!r}={env['strategy_chunks']}, "
            f"expected at least {expected_documents}"
        )
    if config.chunk_strategy == "document" and env["strategy_chunks"] != expected_documents:
        gate_problems.append(
            f"document-strategy chunks={env['strategy_chunks']}, expected {expected_documents}"
        )
    if env["strategy_docs"] != expected_documents:
        gate_problems.append(
            f"distinct chunk doc_ids for strategy {env['chunk_strategy']!r}="
            f"{env['strategy_docs']}, expected {expected_documents}"
        )
    if env["embedding_dimensions"] != [get_settings().embedding_dim]:
        gate_problems.append(
            f"embedding dimensions={env['embedding_dimensions']}, expected [1024]"
        )
    if env["vector_version"] == "missing":
        gate_problems.append("pgvector extension is missing")
    if env["fts_config"] == "missing":
        gate_problems.append("english text-search configuration is missing")
    if env["tsv_expression"] == "missing":
        gate_problems.append("chunks.tsv generated column is missing")
    if env["gin_index"] == "missing":
        gate_problems.append("GIN index on chunks.tsv is missing")
    if gate_problems:
        message = "environment gate failed: " + "; ".join(gate_problems)
        data = _dataset_metadata(all_queries, len(all_queries), args.limit, missing_gold)
        _write_blocked(message, writer, environment=env, dataset=data)
        return 2

    selected = list(all_queries[: args.limit] if args.limit else all_queries)
    dataset = _dataset_metadata(selected, len(all_queries), args.limit, missing_gold)
    estimated = _estimate_workload(len(selected), config.repetitions, config.warmup_passes)

    _print_environment(env, dataset)
    _print_workload(estimated)

    if args.dry_run:
        print("Dry run only: no benchmark queries were executed.")
        return 0

    print("\nRunning sparse benchmark...")
    try:
        result = run_sparse_benchmark(
            selected,
            config,
            progress=lambda message: print(f"  {message}"),
        )
    except BenchmarkError as exc:
        message = f"benchmark could not complete: {exc}"
        _write_blocked(message, writer, environment=env, dataset=dataset)
        return 2

    sanity_record = next(
        (
            record
            for record in result["retrieval"]["per_query"]
            if record.get("evaluated") and record.get("keys")
        ),
        result["retrieval"]["per_query"][0],
    )
    first_query = next(query for query in selected if query.query_id == sanity_record["query_id"])
    benchmark_keys = sanity_record.get("keys", [])
    try:
        sanity = _production_sanity(
            first_query,
            config.top_k,
            benchmark_keys,
            config.query_mode,
            config.chunk_strategy,
        )
    except (psycopg.Error, ConnectionError, RetrievalError) as exc:
        sanity = {"query_id": first_query.query_id, "error": str(exc), "match": None}

    try:
        probe = _chinese_probe(config.top_k, config.query_mode, config.chunk_strategy)
    except (psycopg.Error, ConnectionError) as exc:
        probe = {"available": False, "reason": f"probe failed: {exc}"}

    if config.query_mode == "or":
        control_config = build_config(
            config.top_k,
            1,
            0,
            config.statement_timeout_ms,
            "and",
            config.chunk_strategy,
        )
        control = run_sparse_benchmark(selected, control_config)
        result["and_mode_control"] = {
            "note": (
                "Control run: PostgreSQL's default bare-term AND semantics. "
                "It is not the production default and is not part of the primary metrics."
            ),
            "parameters": control["parameters"],
            "quality": control["retrieval"]["quality"],
            "totals": control["totals"],
            "per_query": [
                {
                    "query_id": record["query_id"],
                    "result_count": record["result_count"],
                    "semantic_recall": record["semantic_recall"],
                    "empty_result": record["empty_result"],
                }
                for record in control["retrieval"]["per_query"]
            ],
        }
    else:
        result["and_mode_control"] = None

    result["dense_overlap"] = dense_overlap(
        result["retrieval"]["per_query"], selected, args.dense_artifact, config.top_k
    )

    post = _post_run_counts(args.chunk_strategy)
    hashes_after = {
        "kb_documents.json": _sha256(KB_PATH),
        "eval_queries.json": _sha256(QUERY_PATH),
    }
    dataset["hashes_after"] = hashes_after
    dataset["post_run_counts"] = post
    dataset["data_integrity_ok"] = (
        post["documents"] == env["documents"]
        and post["chunks"] == env["chunks"]
        and post["strategy_chunks"] == env["strategy_chunks"]
        and post["strategy_docs"] == env["strategy_docs"]
        and post["embedding_dimensions"] == env["embedding_dimensions"]
        and hashes_after == dataset["hashes"]
    )

    result["environment"] = env
    result["dataset"] = dataset
    result["chinese_probe"] = probe
    result["execution"] = {
        "environment": "host Python fallback",
        "container_execution": "not run",
        "docker_status": env["docker_status"],
        "production_sanity": sanity,
        "dense_artifact": str(args.dense_artifact),
    }

    issues: list[str] = []
    totals = result["totals"]
    if totals["errors"]:
        issues.append(f"{totals['errors']} error(s) recorded")
    if totals["timeouts"]:
        issues.append(f"{totals['timeouts']} timeout(s) recorded")
    if totals["warmup_errors"]:
        issues.append(f"{totals['warmup_errors']} warm-up error(s) recorded")
    if totals["missing_queries"]:
        issues.append(f"{totals['missing_queries']} query(s) have no successful timed pass")
    if totals["quality_mismatch_samples"]:
        issues.append(f"{totals['quality_mismatch_samples']} quality-mismatch sample(s)")
    if not dataset["data_integrity_ok"]:
        issues.append("post-run row counts or data hashes changed")
    if sanity.get("match") is not True:
        issues.append("benchmark SQL could not be confirmed against production SparseRetriever")
    unexpected_paths = {
        path for path in result["retrieval"]["path_counts"] if path not in {"fts", "cjk"}
    }
    if unexpected_paths:
        issues.append(f"unexpected retrieval path(s): {sorted(unexpected_paths)}")
    if not probe.get("available"):
        issues.append("Chinese probe did not run")
    result["issues"] = issues
    result["status"] = "PASS" if not issues else "ISSUES"

    _print_summary(result)
    writer(attach_query_reference(result, selected))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
