"""Run the benchmark suites in order and summarise their exit codes.

The orchestrator is a thin, honest wrapper: each suite runs as its own child
process (so a native crash in one model-backed run does not take the others
down), writes its own artifact under ``reports/unified/``, and keeps its own
status. This script never merges numbers or rewrites an artifact; it only
preflights the shared inputs and reports what happened.

Suites:

* ``retrieval``   -- unified arms via ``scripts/benchmark_retrieval.py``
* ``index``       -- HNSW / IVFFlat sweep via ``scripts/benchmark_index.py``
* ``chunking``    -- strategy comparison via ``scripts/benchmark_chunking.py``
* ``fusion``      -- RRF ``rrf_k`` × ``candidate_k`` sweep (optional)
* ``specialized`` -- dense, sparse, hybrid, rerank, rag scripts (optional)
* ``all``         -- every suite above

Usage::

    python scripts/benchmark_all.py --dry-run
    python scripts/benchmark_all.py
    python scripts/benchmark_all.py --with-specialized
    python scripts/benchmark_all.py --suites retrieval index --limit 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, get_settings  # noqa: E402
from src.db import connection, scalar  # noqa: E402
from src.evaluation.dense_benchmark import (  # noqa: E402
    BenchmarkError,
    load_eval_queries,
)

DATA_DIR = get_settings().resolve(get_settings().data_dir)
KB_PATH = DATA_DIR / "kb_documents.json"
QUERY_PATH = DATA_DIR / "eval_queries.json"
UNIFIED_DIR = PROJECT_ROOT / "reports" / "unified"

SUITE_CHOICES = ("retrieval", "index", "chunking", "fusion", "specialized")
DEFAULT_SUITES = ("retrieval", "index", "chunking")


@dataclass(frozen=True)
class Step:
    """One child process: its label, script and arguments."""

    suite: str
    script: str
    args: tuple[str, ...]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse suite selection and the flags forwarded to every child."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=(*SUITE_CHOICES, "all"),
        default=list(DEFAULT_SUITES),
    )
    parser.add_argument(
        "--with-specialized",
        action="store_true",
        help="Also run dense, sparse, hybrid, rerank and rag benchmarks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Forward --limit N to suites that support it (smoke runs)",
    )
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="Drop the rrf_rerank arm from the retrieval suite",
    )
    parser.add_argument(
        "--synthetic-scale",
        type=int,
        default=1,
        help="Forward the index suite's synthetic vector multiplier",
    )
    parser.add_argument("--overwrite", action="store_true", help="Forward --overwrite")
    parser.add_argument("--no-write", action="store_true", help="Forward --no-write")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    return parser.parse_args(argv)


def _expand_suites(args: argparse.Namespace) -> list[str]:
    """Expand ``all`` / ``--with-specialized`` into a stable order."""
    suites: list[str] = []
    for name in args.suites:
        expanded = list(SUITE_CHOICES) if name == "all" else [name]
        for suite in expanded:
            if suite not in suites:
                suites.append(suite)
    if args.with_specialized and "specialized" not in suites:
        suites.append("specialized")
    return suites


def _common_flags(args: argparse.Namespace) -> list[str]:
    """Flags every benchmark script accepts."""
    flags: list[str] = []
    if args.overwrite:
        flags.append("--overwrite")
    if args.no_write:
        flags.append("--no-write")
    return flags


def _steps(suites: list[str], args: argparse.Namespace) -> list[Step]:
    """Build the child commands for the selected suites."""
    common = _common_flags(args)
    limit = ["--limit", str(args.limit)] if args.limit is not None else []
    steps: list[Step] = []
    for suite in suites:
        if suite == "retrieval":
            extra: list[str] = []
            if args.skip_rerank:
                extra.extend(
                    ["--arms", "dense", "sparse_or", "sparse_and", "union", "rrf"]
                )
            steps.append(
                Step(
                    suite,
                    "scripts/benchmark_retrieval.py",
                    (*limit, *extra, *common),
                )
            )
        elif suite == "index":
            steps.append(
                Step(
                    suite,
                    "scripts/benchmark_index.py",
                    (
                        *limit,
                        "--synthetic-scale",
                        str(args.synthetic_scale),
                        *common,
                    ),
                )
            )
        elif suite == "chunking":
            steps.append(
                Step(suite, "scripts/benchmark_chunking.py", (*limit, *common))
            )
        elif suite == "fusion":
            steps.append(
                Step(suite, "scripts/benchmark_fusion.py", (*limit, *common))
            )
        elif suite == "specialized":
            for name in ("dense", "sparse", "hybrid", "rerank", "rag"):
                child_limit = (
                    ["--limit", str(args.limit)]
                    if name == "rag" and args.limit is not None
                    else []
                )
                steps.append(
                    Step(
                        f"specialized:{name}",
                        f"scripts/benchmark_{name}.py",
                        (
                            *child_limit,
                            "--output",
                            str(UNIFIED_DIR / f"{name}.json"),
                            "--report",
                            str(UNIFIED_DIR / f"{name}.md"),
                            *common,
                        ),
                    )
                )
    return steps


def _preflight(limit: int | None) -> list[str]:
    """Check the shared inputs once, before any model or sweep starts."""
    problems: list[str] = []
    for path in (KB_PATH, QUERY_PATH):
        if not path.exists():
            problems.append(f"missing data file: {path}")
    if problems:
        return problems
    try:
        queries = load_eval_queries(QUERY_PATH)
    except BenchmarkError as exc:
        problems.append(f"eval queries are not usable: {exc}")
        return problems
    expected = 30 if limit is None else limit
    if limit is None and len(queries) != expected:
        problems.append(f"eval query count={len(queries)}, expected {expected}")
    if limit is not None and len(queries) < limit:
        problems.append(f"--limit {limit} exceeds the {len(queries)} queries on disk")
    try:
        with connection() as conn:
            version = scalar(
                conn,
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
            )
            if not version:
                problems.append("pgvector extension is missing")
            documents = int(scalar(conn, "SELECT count(*) FROM documents") or 0)
            expected_documents = get_settings().expected_document_count
            if documents != expected_documents:
                problems.append(
                    f"documents={documents}, expected {expected_documents}"
                )
    except Exception as exc:  # noqa: BLE001 -- an unreachable DB blocks every suite
        problems.append(f"PostgreSQL is not reachable: {type(exc).__name__}: {exc}")
    return problems


def _print_plan(steps: list[Step], problems: list[str]) -> None:
    """Print the resolved commands and the preflight result."""
    print("Benchmark plan:")
    for step in steps:
        printed = " ".join([sys.executable, step.script, *step.args])
        print(f"  {step.suite:<20} {printed}")
    print("\nPreflight:")
    if problems:
        for problem in problems:
            print(f"  BLOCKED: {problem}")
    else:
        print("  OK: data files, eval queries, PostgreSQL and pgvector are ready")


def main(argv: list[str] | None = None) -> int:
    """Preflight, run each suite as a child process, and summarise."""
    args = _parse_args(argv)
    if args.limit is not None and args.limit < 1:
        print("BLOCKED: --limit must be positive")
        return 2
    suites = _expand_suites(args)
    steps = _steps(suites, args)
    problems = _preflight(args.limit)
    _print_plan(steps, problems)
    if args.dry_run:
        print("\nDry run only: no suite was executed.")
        return 0
    if problems:
        print("\nBLOCKED: fix the preflight problems before running the suites.")
        return 2

    results: list[tuple[str, int, float]] = []
    for step in steps:
        command = [sys.executable, step.script, *step.args]
        print(f"\n=== {step.suite} ===", flush=True)
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        elapsed = time.perf_counter() - started
        results.append((step.suite, completed.returncode, elapsed))
        print(f"--- {step.suite}: exit {completed.returncode} in {elapsed:.1f}s")

    print("\nSuite summary:")
    for suite, code, elapsed in results:
        print(f"  {suite:<20} exit={code:<3} {elapsed:8.1f}s")
    failed = [suite for suite, code, _ in results if code != 0]
    if failed:
        print(f"\nNot all suites passed: {', '.join(failed)}")
        print("Inspect each suite's artifact and status_reason before citing results.")
        return 1
    print("\nAll selected suites exited 0.")
    if not args.no_write:
        print("Next: python scripts/report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
