"""Phase 6B: five-arm hybrid / RRF offline evaluation.

Read-only orchestration around :mod:`src.evaluation.hybrid_benchmark`. It checks
PostgreSQL and the labelled dataset, runs dense-only, sparse-OR-only,
sparse-AND-only, candidate-union and RRF over the same queries and candidate
pool, then writes the raw JSON and Markdown report.

Artificial constraints (deliberate):

* one ``--candidate-k`` for every arm, one ``--rrf-k``, one final ``--top-k``;
* no parameter sweep -- changing a value is a separate, recorded experiment;
* no writes to PostgreSQL and no edits to the data files; hashes and row counts
  are checked again after the run;
* Docker is probed once, read-only, with a timeout; an unresponsive CLI is
  recorded, not retried.

Recommended command::

    python scripts/benchmark_hybrid.py

Container equivalent::

    docker compose run --rm app python scripts/benchmark_hybrid.py

Smoke test::

    python scripts/benchmark_hybrid.py --limit 2 --repetitions 1 --warmup-passes 0

Dry run (no model load and no retrieval)::

    python scripts/benchmark_hybrid.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402
from src.evaluation.dense_benchmark import BenchmarkError, load_eval_queries  # noqa: E402
from src.evaluation.hybrid_benchmark import (  # noqa: E402
    DEFAULT_OUTPUT,
    DEFAULT_REPORT,
    HybridBenchmarkConfig,
    build_config,
    collect_dataset,
    collect_environment,
    run_hybrid_benchmark,
    write_outputs,
)

PARAMETER_KEYS = (
    "top_k",
    "candidate_k",
    "rrf_k",
    "repetitions",
    "warmup_passes",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the read-only benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=10, help="Final Top-K for metrics")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="Per-arm candidate pool; must be at least --top-k",
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="Standard RRF k (default 60)")
    parser.add_argument("--repetitions", type=int, default=5, help="Timed passes per query")
    parser.add_argument("--warmup-passes", type=int, default=1, help="Untimed passes per query")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N eval queries (smoke test; the report uses 30)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print environment and exit")
    parser.add_argument("--no-write", action="store_true", help="Do not write report files")
    parser.add_argument(
        "--docker-timeout",
        type=float,
        default=5.0,
        help="Seconds allowed for the single read-only Docker CLI probe",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def _docker_status(timeout: float) -> str:
    """Probe the Docker CLI once, without starting or changing anything."""
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


def _validate_dataset(environment: dict[str, Any], dataset: dict[str, Any]) -> list[str]:
    """Return the reasons the evaluation would not be trustworthy."""
    settings = get_settings()
    problems: list[str] = []
    if environment.get("documents") != settings.expected_document_count:
        problems.append(
            f"documents={environment.get('documents')}, expected {settings.expected_document_count}"
        )
    if environment.get("chunks", 0) < environment.get("documents", 0):
        problems.append(
            f"chunks={environment.get('chunks')} < documents={environment.get('documents')}"
        )
    dimensions = environment.get("embedding_dimensions") or []
    if dimensions != [settings.embedding_dim]:
        problems.append(f"embedding dimensions={dimensions}, expected [{settings.embedding_dim}]")
    if dataset.get("missing_gold_docs"):
        problems.append(f"gold docs absent from database: {dataset['missing_gold_docs']}")
    return problems


def _estimate_workload(
    query_count: int,
    repetitions: int,
    warmups: int,
) -> dict[str, int]:
    """Estimate retriever calls before anything expensive starts."""
    passes = repetitions + warmups
    return {
        "queries": query_count,
        "passes_per_query": passes,
        "dense_calls": query_count * passes,
        "sparse_or_calls": query_count * passes,
        "sparse_and_calls": query_count * passes,
        "derived_union_and_rrf": query_count * passes * 2,
    }


def _print_environment(environment: dict[str, Any], dataset: dict[str, Any]) -> None:
    """Print the gate facts before a dry run or a measurement."""
    print("Environment:")
    for key in (
        "execution_environment",
        "python_version",
        "postgres_version",
        "vector_version",
        "server_address",
        "documents",
        "chunks",
        "distinct_chunk_docs",
        "embedding_model",
        "embedding_dimensions",
        "ef_search_setting",
        "hnsw_index",
        "gin_index",
        "docker_status",
        "command",
    ):
        print(f"  {key}: {environment.get(key)}")
    print(f"  dependency_versions: {environment.get('dependency_versions')}")
    print("Dataset:")
    for key in ("queries", "gold_sizes", "distinct_gold_docs", "missing_gold_docs", "hashes"):
        print(f"  {key}: {dataset.get(key)}")


def _print_summary(result: dict[str, Any]) -> None:
    """Print the comparison table and the status."""
    print("")
    print(f"Status: {result['status']}")
    print(
        f"top_k={result['parameters']['top_k']} "
        f"candidate_k={result['parameters']['candidate_k']} "
        f"rrf_k={result['parameters']['rrf_k']} "
        f"queries={result['parameters']['queries']} "
        f"repetitions={result['parameters']['repetitions']}"
    )
    print("")
    print(f"{'arm':<16} {'recall':>8} {'mrr':>8} {'hit@1':>7} {'errors':>7}")
    for arm, record in result["arms"]["arms"].items():
        print(
            f"{arm:<16} "
            f"{_fmt(record.get('semantic_recall_macro')):>8} "
            f"{_fmt(record.get('mrr')):>8} "
            f"{str(record.get('hit_at_1')) + '/' + str(record.get('evaluated_queries')):>7} "
            f"{result['totals']['errors']:>7}"
        )
    print("")
    print("Focus misses (Dense miss -> sparse / RRF rank):")
    for row in result.get("focus_misses", []):
        print(
            f"  {row['query_id']}/{row['doc_id']}: "
            f"dense={row['dense_rank']} sparse={row['sparse_rank']} "
            f"union={row['union_rank']} rrf={row['rrf_rank']}"
        )


def _fmt(value: object) -> str:
    """Format an optional metric."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _run(args: argparse.Namespace) -> int:
    """Execute the benchmark and return the process exit code."""
    config: HybridBenchmarkConfig = build_config(
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
        repetitions=args.repetitions,
        warmup_passes=args.warmup_passes,
    )
    all_queries = load_eval_queries()
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(all_queries):
            raise BenchmarkError(
                f"--limit must be between 1 and {len(all_queries)}, got {args.limit}"
            )
        queries = all_queries[: args.limit]
    else:
        queries = all_queries

    environment = collect_environment()
    environment["docker_status"] = _docker_status(args.docker_timeout)
    environment["command"] = " ".join([Path(sys.executable).name, *sys.argv])
    dataset = collect_dataset(queries)

    print("Parameters:")
    for key in PARAMETER_KEYS:
        print(f"  {key}: {getattr(args, key)}")
    print("")
    _print_environment(environment, dataset)
    print("Workload:")
    for key, value in _estimate_workload(
        len(queries), args.repetitions, args.warmup_passes
    ).items():
        print(f"  {key}: {value}")
    print("")

    problems = _validate_dataset(environment, dataset)
    if problems:
        print("BLOCKED: environment/data gate failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    if args.dry_run:
        print("Dry run: no model load, no retrieval, no files written.")
        return 0

    print("Running five arms...")
    result = run_hybrid_benchmark(
        config,
        queries,
        environment=environment,
        dataset=dataset,
        progress=print,
    )

    environment_after = collect_environment()
    dataset_after = collect_dataset(queries)
    counts_before = {
        key: environment.get(key) for key in ("documents", "chunks", "distinct_chunk_docs")
    }
    counts_after = {
        key: environment_after.get(key) for key in ("documents", "chunks", "distinct_chunk_docs")
    }
    integrity_ok = (
        counts_before == counts_after
        and dataset.get("hashes") == dataset_after.get("hashes")
        and not dataset_after.get("missing_gold_docs")
    )
    result["integrity"] = {
        "counts_before": counts_before,
        "counts_after": counts_after,
        "hashes_before": dataset.get("hashes"),
        "hashes_after": dataset_after.get("hashes"),
        "missing_gold_docs_after": dataset_after.get("missing_gold_docs"),
        "data_integrity_ok": integrity_ok,
    }
    if not integrity_ok:
        result["status"] = "FAIL"

    _print_summary(result)
    print(f"Integrity: {integrity_ok}")

    if args.no_write:
        print("--no-write: report files were not written.")
    else:
        json_path, markdown_path = write_outputs(result, args.output, args.report)
        print(f"Raw results: {json_path}")
        print(f"Markdown report: {markdown_path}")

    return 0 if result["status"] in {"PASS", "PASS WITH MINOR ISSUES"} else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI with clear errors and no fabricated output."""
    args = _parse_args(list(argv) if argv is not None else None)
    try:
        return _run(args)
    except (BenchmarkError, RuntimeError, OSError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
