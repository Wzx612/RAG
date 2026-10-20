"""Assemble the final report from unified benchmark artifacts.

This script is read-only with respect to the database and the raw data: it
loads the JSON artifacts written by the benchmark scripts, hashes them, renders
``reports/final_report.md`` and (when matplotlib is available) the figures in
``reports/figures/``. It never recomputes a metric, so a number in the report
either comes from a cited artifact or is absent.

The final report is guarded by the same no-clobber rule as every benchmark
artifact: it refuses to overwrite an existing file unless ``--overwrite`` is
given. Figures are derived output and are regenerated on each run.

Usage::

    python scripts/report.py --dry-run
    python scripts/report.py
    python scripts/report.py --benchmark reports/unified/benchmark.json
    python scripts/report.py --no-write
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT  # noqa: E402
from src.evaluation.artifacts import ArtifactExistsError, ensure_writable  # noqa: E402
from src.evaluation.report import (  # noqa: E402
    OPTIONAL_ARTIFACTS,
    REQUIRED_ARTIFACTS,
    FiguresUnavailableError,
    ReportError,
    load_artifacts,
    render_report,
    report_status,
    write_figures,
)

DEFAULT_ARTIFACTS: dict[str, Path] = {
    "benchmark": PROJECT_ROOT / "reports" / "unified" / "benchmark.json",
    "index": PROJECT_ROOT / "reports" / "unified" / "index_benchmark_scale10.json",
    "chunking": PROJECT_ROOT / "reports" / "unified" / "chunking_benchmark_rerank.json",
    "rerank": PROJECT_ROOT / "reports" / "unified" / "rerank_20to5.json",
    "judge": PROJECT_ROOT / "reports" / "judge" / "judge_summary.json",
}
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "final_report.md"
DEFAULT_FIGURES = PROJECT_ROOT / "reports" / "figures"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse paths and safety flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_ARTIFACTS["benchmark"],
        help="Unified retrieval benchmark JSON",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_ARTIFACTS["index"],
        help="HNSW/IVFFlat benchmark JSON",
    )
    parser.add_argument(
        "--chunking",
        type=Path,
        default=DEFAULT_ARTIFACTS["chunking"],
        help="Chunking comparison JSON",
    )
    parser.add_argument(
        "--rerank",
        type=Path,
        default=DEFAULT_ARTIFACTS["rerank"],
        help="Optional live Top-K -> reranked artifact; omitted when the file is absent",
    )
    parser.add_argument(
        "--judge",
        type=Path,
        default=DEFAULT_ARTIFACTS["judge"],
        help="Optional LLM-as-a-Judge summary; omitted when the file is absent",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Render the Markdown report without generating PNG figures",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing final report",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="Render but do not write any file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the resolved inputs and output, then exit",
    )
    return parser.parse_args(argv)


def _git_context() -> dict[str, Any]:
    """Record the commit the report was generated from, when git is available."""
    context: dict[str, Any] = {}

    def git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    if branch:
        context["git.branch"] = branch
    if commit:
        context["git.commit"] = commit
    if status is not None:
        context["git.dirty"] = bool(status)
    return context


def _reproduction(args: argparse.Namespace) -> list[str]:
    """Commands a reader can re-run to reproduce the cited numbers."""
    return [
        "python scripts/benchmark_all.py --overwrite",
        "python scripts/benchmark_retrieval.py --arms rrf rrf_rerank --top-k 5 "
        "--candidate-k 20 --output reports/unified/rerank_20to5.json "
        "--report reports/unified/rerank_20to5.md",
        "python scripts/evaluate_judge.py --rag-artifact "
        "reports/judge/rag_answers.json --judge-provider real",
        f"python scripts/report.py --benchmark {args.benchmark} "
        f"--index {args.index} --chunking {args.chunking} --rerank {args.rerank} "
        f"--judge {args.judge}",
        "python -m pytest tests/test_report.py -q",
    ]


def _print_plan(paths: dict[str, Path], args: argparse.Namespace) -> None:
    """Print the resolved inputs without reading or writing anything."""
    print("Report plan")
    for key, path in paths.items():
        state = "present" if path.exists() else "missing"
        print(f"  {key:<9} {path} ({state})")
    print(f"  output    {args.output}")
    print(f"  figures   {'skipped (--no-figures)' if args.no_figures else args.figures_dir}")
    print(f"  write     {'no (--no-write)' if args.no_write else 'yes'}")


def main(argv: list[str] | None = None) -> int:
    """Load artifacts, render the report and write the outputs."""
    args = _parse_args(argv)
    paths = {
        "benchmark": args.benchmark,
        "index": args.index,
        "chunking": args.chunking,
        "rerank": args.rerank,
        "judge": args.judge,
    }
    if args.dry_run:
        _print_plan(paths, args)
        return 0

    try:
        payloads, metadata = load_artifacts(paths)
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not payloads:
        print("error: no artifact found; run the benchmarks first", file=sys.stderr)
        return 1
    missing = [key for key in REQUIRED_ARTIFACTS if key not in paths]
    for key in missing:
        print(f"warning: required {key} artifact missing at {paths[key]}", file=sys.stderr)
    for key in OPTIONAL_ARTIFACTS:
        if key not in payloads:
            print(f"note: optional {key} artifact not found at {paths[key]}; "
                  "its section will be omitted", file=sys.stderr)

    figure_names: list[str] = []
    written: list[Path] = []
    if not args.no_figures and not args.no_write:
        try:
            written = write_figures(payloads, args.figures_dir)
            figure_names = [path.name for path in written]
        except FiguresUnavailableError as exc:
            print(f"warning: {exc}; continuing without figures", file=sys.stderr)

    report = render_report(
        payloads,
        metadata,
        context=_git_context(),
        figures=figure_names,
        reproduction=_reproduction(args),
    )
    if args.no_write:
        print(report)
        print("--no-write: final report and figures were not written")
        return 0

    try:
        ensure_writable((args.output,), overwrite=args.overwrite)
    except ArtifactExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")
    for path in written:
        print(f"Wrote {path}")
    print(f"Report status: {report_status(payloads)[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
