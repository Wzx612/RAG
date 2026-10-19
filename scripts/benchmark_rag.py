"""Phase 8: end-to-end RAG evaluation with a mock or real provider.

The script runs the accepted live retrieval chain (Dense + Sparse OR -> RRF ->
cross-encoder), assembles the exact database evidence, calls the configured
provider and records the answer, citations, status and latency for every query.

Safety defaults:

* the provider is ``mock`` unless ``--provider real`` is passed explicitly;
* a real run requires ``APEX_LLM_API_KEY`` (or ``DEEPSEEK_API_KEY``) and fails
  before loading retrieval models when it is absent;
* no API key, header, prompt or response body is printed or written;
* PostgreSQL is read-only; data hashes and counts are checked before and after;
* existing report files are not overwritten without ``--overwrite``;
* Docker is probed once with a short timeout.

Recommended commands::

    python scripts/benchmark_rag.py --provider mock
    python scripts/benchmark_rag.py --provider real --limit 2 --no-write

Smoke test::

    python scripts/benchmark_rag.py --provider mock --limit 2 --no-write

Dry run (environment and key checks; no model load, no provider call)::

    python scripts/benchmark_rag.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings  # noqa: E402
from src.evaluation.dense_benchmark import BenchmarkError, load_eval_queries  # noqa: E402
from src.evaluation.rag_benchmark import (  # noqa: E402
    DEFAULT_OUTPUT,
    DEFAULT_REPORT,
    RAGBenchmarkConfig,
    load_raw_queries,
    run_rag_benchmark,
    write_outputs,
)
from src.rag.factory import (  # noqa: E402
    build_historical_retriever,
    build_pipeline,
    build_provider,
)
from src.rag.historical import DEFAULT_HISTORICAL  # noqa: E402
from src.rag.pipeline import RAGPipeline  # noqa: E402
from src.rag.provider import LLMError, LLMProvider, MockLLMProvider  # noqa: E402

PARAMETER_KEYS = (
    "top_k",
    "candidates",
    "max_context_chars",
    "max_chunk_chars",
    "provider",
    "retrieval_mode",
    "limit",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the evaluation options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=("mock", "real"), default="mock")
    parser.add_argument(
        "--retrieval-mode",
        choices=("live", "historical"),
        default="live",
        help=(
            "live runs Dense+Sparse+RRF+reranker; historical replays the accepted "
            "Phase 6B RRF pool, then joins live DB text and reranks live"
        ),
    )
    parser.add_argument(
        "--historical",
        type=Path,
        default=DEFAULT_HISTORICAL,
        help="Accepted Phase 6B artifact used by --retrieval-mode historical",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Final evidence size")
    parser.add_argument("--candidates", type=int, default=None, help="RRF candidate pool")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
        help="Hard context character budget",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=None,
        help="Per-chunk character cap",
    )
    parser.add_argument("--limit", type=int, default=None, help="First N queries (smoke)")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face network access for the reranker",
    )
    parser.add_argument("--dry-run", action="store_true", help="Check gates and exit")
    parser.add_argument("--no-write", action="store_true", help="Do not write reports")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing report files at the selected paths",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--docker-timeout",
        type=float,
        default=5.0,
        help="Seconds allowed for the single read-only Docker probe",
    )
    return parser.parse_args(argv)


def _docker_status(timeout: float) -> str:
    """Probe the Docker CLI once, without changing anything."""
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


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """Apply CLI overrides without mutating the cached settings singleton."""
    overrides: dict[str, Any] = {}
    if args.top_k is not None:
        overrides["rag_top_k"] = args.top_k
    if args.candidates is not None:
        overrides["rag_candidates"] = args.candidates
    if args.max_context_chars is not None:
        overrides["rag_max_context_chars"] = args.max_context_chars
    if args.max_chunk_chars is not None:
        overrides["rag_max_chunk_chars"] = args.max_chunk_chars
    return Settings(**overrides)


def _validate_dataset(environment: dict[str, Any], dataset: dict[str, Any]) -> list[str]:
    """Return reasons the database or labels are not trustworthy."""
    settings = Settings()
    problems: list[str] = []
    if environment.get("documents") != settings.expected_document_count:
        problems.append(
            f"documents={environment.get('documents')}, expected "
            f"{settings.expected_document_count}"
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


def _memory_snapshot() -> dict[str, Any]:
    """Read process RSS and system available memory when psutil exists."""
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        return {"available": False}
    process = psutil.Process()
    memory = psutil.virtual_memory()
    return {
        "available": True,
        "process_rss_mib": round(process.memory_info().rss / (1024 * 1024), 1),
        "system_available_gib": round(memory.available / (1024**3), 2),
    }


def _print_facts(
    environment: dict[str, Any],
    dataset: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Print the gate facts before a dry run or a measurement."""
    print("Parameters:")
    for key in PARAMETER_KEYS:
        print(f"  {key}: {getattr(args, key)}")
    print(f"  allow_network: {args.allow_network}")
    print("")
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
        "docker_status",
        "command",
    ):
        print(f"  {key}: {environment.get(key)}")
    print(f"  dependency_versions: {environment.get('dependency_versions')}")
    print("Dataset:")
    for key in ("queries", "gold_sizes", "distinct_gold_docs", "missing_gold_docs", "hashes"):
        print(f"  {key}: {dataset.get(key)}")
    print(f"  memory: {_memory_snapshot()}")


def _print_summary(result: dict[str, Any]) -> None:
    """Print the headline metrics and per-status counts."""
    summary = result["summary"]
    metrics = result["metrics"]
    print("")
    print(f"Status: {result['status']} ({result['status_reason']})")
    print(
        f"Answered: {summary['answered']}/{summary['queries']} "
        f"insufficient_evidence={summary['insufficient_evidence']} "
        f"generation_failures={summary['generation_failures']}"
    )
    print(
        "Citations: "
        f"valid={summary['citations_valid']} invalid={summary['citations_invalid']} "
        f"queries_with_citations={summary['queries_with_citations']}"
    )
    print(
        "Evidence: "
        f"recall_macro={_fmt(summary['evidence_recall_macro'])} "
        f"recall_micro={_fmt(summary['evidence_recall_micro'])} "
        f"mrr={_fmt(summary['evidence_mrr'])} "
        f"hit_at_k={summary['evidence_hit_at_k']}/{summary['evidence_samples']}"
    )
    latency = metrics["latency_ms"]
    print(
        "Latency ms: "
        f"total P50={_fmt(latency['total']['p50'])} P95={_fmt(latency['total']['p95'])} "
        f"n={latency['total']['samples']}; "
        f"llm P50={_fmt(latency['llm']['p50'])} n={latency['llm']['samples']}"
    )
    if not result["provider"]["is_real"]:
        print("MOCK provider: the answers are deterministic fixtures, not model output.")


def _fmt(value: object, digits: int = 4) -> str:
    """Format an optional metric."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _provider_for(args: argparse.Namespace, settings: Settings) -> LLMProvider:
    """Construct the explicitly selected provider, failing fast without a key."""
    if args.provider == "mock":
        return MockLLMProvider()
    provider = build_provider(settings)
    try:
        provider.ensure_configured()
    except LLMError as exc:
        raise BenchmarkError(
            f"real provider is not configured: {exc}. Set APEX_LLM_API_KEY or "
            "DEEPSEEK_API_KEY, or use --provider mock."
        ) from exc
    return provider


def _run(args: argparse.Namespace) -> int:
    """Execute the evaluation and return the process exit code."""
    from src.evaluation.hybrid_benchmark import collect_dataset, collect_environment

    settings = _settings_from_args(args)
    if settings.rag_top_k > settings.rag_candidates:
        raise BenchmarkError("--top-k must not exceed --candidates")

    all_queries = load_eval_queries()
    raw_queries = load_raw_queries()
    missing_raw = [query.query_id for query in all_queries if query.query_id not in raw_queries]
    if missing_raw:
        raise BenchmarkError(f"raw query records missing for: {missing_raw}")
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
    environment["memory_before"] = _memory_snapshot()
    dataset = collect_dataset(all_queries)

    print(
        "RAG end-to-end evaluation"
        + (" (MOCK provider)" if args.provider == "mock" else "")
        + f" [retrieval={args.retrieval_mode}]"
    )
    print("")
    _print_facts(environment, dataset, args)
    print(
        "Workload: "
        f"{len(queries)} queries x {settings.rag_candidates} candidates, "
        f"final top_k={settings.rag_top_k}, "
        f"max_context_chars={settings.rag_max_context_chars}"
    )
    print("")

    problems = _validate_dataset(environment, dataset)
    if args.provider == "real" and not settings.llm_api_key:
        problems.append(
            "real provider selected but APEX_LLM_API_KEY/DEEPSEEK_API_KEY is absent"
        )
    if problems:
        print("BLOCKED: environment/data/provider gate failed:")
        for problem in problems:
            print(f"  - {problem}")
        if args.provider == "real" and not settings.llm_api_key:
            print(
                "Real LLM evaluation is incomplete. Mock results cannot substitute "
                "for real model output."
            )
        return 2 if args.provider == "real" and not settings.llm_api_key else 1

    provider = _provider_for(args, settings)
    if args.dry_run:
        print(
            "Dry run: no retrieval model load, no provider call, no files written. "
            f"Provider kind={args.provider}, key_present={bool(settings.llm_api_key)}."
        )
        return 0

    retriever = (
        build_historical_retriever(
            settings,
            historical_path=args.historical,
            local_files_only=not args.allow_network,
        )
        if args.retrieval_mode == "historical"
        else None
    )
    pipeline: RAGPipeline = build_pipeline(
        settings,
        provider=provider,
        retriever=retriever,
        local_files_only=not args.allow_network,
    )
    config = RAGBenchmarkConfig(
        top_k=settings.rag_top_k,
        candidates=settings.rag_candidates,
        max_context_chars=settings.rag_max_context_chars,
        max_chunk_chars=settings.rag_max_chunk_chars,
        provider_kind=args.provider,
        retrieval_mode=args.retrieval_mode,
    )
    result = run_rag_benchmark(
        pipeline,
        queries,
        raw_queries=raw_queries,
        config=config,
        environment=environment,
        dataset=dataset,
        progress=print,
    )

    environment_after = collect_environment()
    dataset_after = collect_dataset(all_queries)
    counts_before = {
        key: environment.get(key) for key in ("documents", "chunks", "distinct_chunk_docs")
    }
    counts_after = {
        key: environment_after.get(key)
        for key in ("documents", "chunks", "distinct_chunk_docs")
    }
    integrity_ok = (
        counts_before == counts_after
        and dataset.get("hashes") == dataset_after.get("hashes")
        and not dataset_after.get("missing_gold_docs")
    )
    result["environment_after"] = {
        "memory_after": _memory_snapshot(),
        "documents": environment_after.get("documents"),
        "chunks": environment_after.get("chunks"),
    }
    result["integrity"].update(
        {
            "counts_before": counts_before,
            "counts_after": counts_after,
            "hashes_after": dataset_after.get("hashes"),
            "missing_gold_docs_after": dataset_after.get("missing_gold_docs"),
            "data_integrity_ok": integrity_ok,
        }
    )
    if not integrity_ok:
        result["status"] = "FAIL"
        result["status_reason"] = "database counts or data hashes changed during the run"

    _print_summary(result)
    print(f"Integrity: {integrity_ok}")

    if args.no_write:
        print("--no-write: report files were not written.")
    else:
        json_path, markdown_path = write_outputs(
            result,
            args.output,
            args.report,
            overwrite=args.overwrite,
        )
        print(f"Raw results: {json_path}")
        print(f"Markdown report: {markdown_path}")

    return 0 if result["status"] in {"PASS", "PARTIAL"} else 1


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
