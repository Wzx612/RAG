"""Phase 7B: RRF + cross-encoder reranker offline evaluation.

Read-only orchestration around :mod:`src.evaluation.rerank_benchmark`. The
accepted Phase 6B artifact supplies the dense, sparse-OR, sparse-AND and RRF
arms; this run replays each query's historical RRF Top-10 through a locally
cached ``BAAI/bge-reranker-base`` and records the reranked ordering.

Artificial constraints (deliberate):

* the historical pool is fixed at the ⑥-B RRF Top-10; no larger pool is used,
  so Recall cannot be inflated by changing the candidate set;
* one model, one batch size, one max length and one activation per run; no
  parameter sweep;
* the embedding model is not loaded in this process, so the reranker's memory
  is measured without a second model co-resident;
* PostgreSQL is read-only; data hashes and row counts are checked before and
  after the run;
* a missing model or failed load is reported as FAIL, never replaced with the
  pre-rerank list;
* Docker is probed once, read-only, with a timeout.

Recommended command::

    python scripts/benchmark_rerank.py --top-k 10 --candidate-k 10 --rrf-k 60

Smoke test::

    python scripts/benchmark_rerank.py --limit 2 --no-latency-probe --no-write

Dry run (checks environment, model files and historical artifact; no model load)::

    python scripts/benchmark_rerank.py --dry-run
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
from src.evaluation.artifacts import ArtifactExistsError, ensure_writable  # noqa: E402
from src.evaluation.dense_benchmark import BenchmarkError, load_eval_queries  # noqa: E402
from src.evaluation.rerank_benchmark import (  # noqa: E402
    DEFAULT_CACHE_FOLDER,
    DEFAULT_HISTORICAL,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    DEFAULT_REPORT,
    DEFAULT_REVISION,
    RerankBenchmarkConfig,
    build_config,
    collect_dataset,
    collect_environment,
    load_historical,
    run_rerank_benchmark,
    verify_historical,
    write_outputs,
)

PARAMETER_KEYS = (
    "top_k",
    "candidate_k",
    "rrf_k",
    "batch_size",
    "max_length",
    "activation",
    "device",
    "latency_repetitions",
    "chunk_strategy",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the read-only benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=10, help="Final Top-K for metrics")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="Replayed historical RRF pool size (1..10)",
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="Historical RRF k; must be 60")
    parser.add_argument("--batch-size", type=int, default=16, help="Pairs per forward pass")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer truncation")
    parser.add_argument(
        "--activation",
        choices=("identity", "sigmoid"),
        default="identity",
        help="identity keeps raw logits; sigmoid only rescales them",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, default cpu")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Cross-encoder model id")
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Pinned model revision; pass an empty string to use the cache default",
    )
    parser.add_argument(
        "--cache-folder",
        type=Path,
        default=DEFAULT_CACHE_FOLDER,
        help="Hugging Face cache root used for local loading",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face network access (default: local files only)",
    )
    parser.add_argument(
        "--latency-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the 10/50-candidate, batch 1/16 latency probe",
    )
    parser.add_argument(
        "--latency-repetitions",
        type=int,
        default=2,
        help="Timed calls per latency combination after one warm-up",
    )
    parser.add_argument(
        "--chunk-strategy",
        choices=("document", "fixed_window", "structured"),
        default="document",
        help="Chunking strategy used by the live sparse latency pool",
    )
    parser.add_argument(
        "--direction-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score Q-001/Q-004/Q-014 for score-direction diagnostics",
    )
    parser.add_argument(
        "--historical",
        type=Path,
        default=DEFAULT_HISTORICAL,
        help="Accepted Phase 6B JSON artifact providing the four baseline arms",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N eval queries (smoke test; the report uses 30)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Check gates and exit")
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
    expected_documents = settings.expected_document_count
    problems: list[str] = []
    if environment.get("documents") != expected_documents:
        problems.append(
            f"documents={environment.get('documents')}, expected {expected_documents}"
        )
    strategy_chunks = environment.get("strategy_chunks", environment.get("chunks", 0))
    strategy_docs = environment.get("strategy_docs", environment.get("documents", 0))
    if strategy_docs != expected_documents:
        problems.append(
            f"chunk doc_ids for strategy {environment.get('chunk_strategy')!r}="
            f"{strategy_docs}, expected {expected_documents}"
        )
    if strategy_chunks < expected_documents:
        problems.append(
            f"chunks for strategy {environment.get('chunk_strategy')!r}="
            f"{strategy_chunks}, expected at least {expected_documents}"
        )
    dimensions = environment.get("embedding_dimensions") or []
    if dimensions != [settings.embedding_dim]:
        problems.append(f"embedding dimensions={dimensions}, expected [{settings.embedding_dim}]")
    if dataset.get("missing_gold_docs"):
        problems.append(f"gold docs absent from database: {dataset['missing_gold_docs']}")
    return problems


def _validate_model_files(config: RerankBenchmarkConfig) -> list[str]:
    """Check that the pinned local snapshot exists; never download automatically."""
    from src.evaluation.rerank_benchmark import _model_files

    files = _model_files(config)
    if not files.get("available"):
        return [str(files.get("reason", "model snapshot is not available"))]
    missing = [
        name
        for name, record in files["files"].items()
        if not record.get("bytes")
    ]
    if missing:
        return [f"model snapshot is missing or empty files: {missing}"]
    return []


def _estimate_workload(query_count: int, config: RerankBenchmarkConfig) -> dict[str, int]:
    """Estimate pair scores before loading the model."""
    probe_pairs = (10 + 50) * 2 * (config.latency_repetitions + 1)
    return {
        "queries": query_count,
        "pairs_per_query": config.candidate_k,
        "rerank_pairs": query_count * config.candidate_k,
        "warmup_rerank_pairs": config.candidate_k,
        "direction_check_pairs": 3 * config.candidate_k * 3,
        "latency_probe_pairs": probe_pairs,
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
        "chunk_strategy",
        "documents",
        "chunks",
        "distinct_chunk_docs",
        "strategy_chunks",
        "strategy_docs",
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
    """Print the five-arm comparison and the reranker ordering summary."""
    print("")
    print(f"Status: {result['status']}")
    parameters = result.get("parameters", {})
    print(
        f"top_k={parameters.get('top_k')} candidate_k={parameters.get('candidate_k')} "
        f"rrf_k={parameters.get('rrf_k')} queries={parameters.get('queries')} "
        f"batch_size={parameters.get('batch_size')} "
        f"activation={parameters.get('activation')}"
    )
    print("")
    print(f"{'arm':<16} {'recall':>8} {'mrr':>8} {'hit@1':>7}")
    for arm, record in result.get("arms", {}).get("arms", {}).items():
        print(
            f"{arm:<16} "
            f"{_fmt(record.get('semantic_recall_macro')):>8} "
            f"{_fmt(record.get('mrr')):>8} "
            f"{str(record.get('hit_at_1')) + '/' + str(record.get('evaluated_queries')):>7}"
        )
    ordering = result.get("arms", {}).get("rrf_reranker_ordering", {})
    print("")
    print(
        "RRF -> RRF+Reranker MRR: "
        f"improved={ordering.get('improved')} equal={ordering.get('equal')} "
        f"worse={ordering.get('worse')} mean_delta={_fmt(ordering.get('mrr_delta_mean'))}"
    )
    print(f"Moved queries: {', '.join(ordering.get('moved_queries', [])) or 'none'}")
    probe = result.get("probe") or {}
    if probe.get("combinations"):
        print("")
        print("Latency probe:")
        for row in probe["combinations"]:
            print(
                f"  candidates={row.get('candidates')} batch={row.get('batch_size')} "
                f"mean_ms={_fmt(row.get('mean_ms'))} per_candidate_ms="
                f"{_fmt(row.get('per_candidate_ms'))} status={row.get('status')}"
            )
    else:
        print("")
        print(f"Latency probe: {probe.get('reason', 'not run')}")


def _fmt(value: object, digits: int = 4) -> str:
    """Format an optional metric."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _run(args: argparse.Namespace) -> int:
    """Execute the benchmark and return the process exit code."""
    if not args.no_write and not args.dry_run:
        ensure_writable((args.output, args.report), overwrite=args.overwrite)
    revision = args.revision or None
    config: RerankBenchmarkConfig = build_config(
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
        batch_size=args.batch_size,
        max_length=args.max_length,
        activation=args.activation,
        device=args.device,
        model=args.model,
        revision=revision,
        cache_folder=args.cache_folder,
        local_files_only=not args.allow_network,
        latency_repetitions=args.latency_repetitions,
        chunk_strategy=args.chunk_strategy,
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

    environment = collect_environment(args.chunk_strategy)
    environment["docker_status"] = _docker_status(args.docker_timeout)
    environment["command"] = " ".join([Path(sys.executable).name, *sys.argv])
    dataset = collect_dataset(all_queries)

    print("Parameters:")
    for key in PARAMETER_KEYS:
        print(f"  {key}: {getattr(args, key)}")
    print(f"  model: {args.model}")
    print(f"  revision: {revision}")
    print(f"  local_files_only: {not args.allow_network}")
    print("")
    _print_environment(environment, dataset)
    print("Workload:")
    for key, value in _estimate_workload(len(queries), config).items():
        print(f"  {key}: {value}")
    print("")

    problems = _validate_dataset(environment, dataset)
    problems += _validate_model_files(config)
    historical = load_historical(args.historical)
    verification = verify_historical(historical, all_queries, config)
    if not verification["ok"]:
        problems.append(
            "historical artifact verification failed: "
            + "; ".join(verification["mismatches"][:5])
        )
    if problems:
        print("BLOCKED: environment/data/model gate failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        "Historical artifact: verified ("
        f"{verification['rrf_formula_checked_hits']} RRF hits, "
        f"{verification['queries_with_all_gold_in_rrf_top_k']}/{len(all_queries)} queries "
        "with all gold in the historical RRF Top-K)"
    )
    if args.dry_run:
        print("Dry run: no model load, no reranking, no files written.")
        return 0

    print("Running reranker comparison...")
    result = run_rerank_benchmark(
        config,
        queries,
        historical=historical,
        environment=environment,
        dataset=dataset,
        progress=print,
        run_probe=args.latency_probe,
        run_direction=args.direction_check,
        check_hashes=True,
    )

    environment_after = collect_environment(args.chunk_strategy)
    dataset_after = collect_dataset(all_queries)
    count_keys = ("documents", "chunks", "distinct_chunk_docs", "strategy_chunks", "strategy_docs")
    counts_before = {key: environment.get(key) for key in count_keys}
    counts_after = {key: environment_after.get(key) for key in count_keys}
    integrity_ok = (
        counts_before == counts_after
        and dataset.get("hashes") == dataset_after.get("hashes")
        and not dataset_after.get("missing_gold_docs")
    )
    result["integrity"].update(
        {
            "historical_artifact": str(args.historical),
            "counts_before": counts_before,
            "counts_after": counts_after,
            "hashes_before": dataset.get("hashes"),
            "hashes_after": dataset_after.get("hashes"),
            "missing_gold_docs_after": dataset_after.get("missing_gold_docs"),
            "data_integrity_ok": integrity_ok,
        }
    )
    if not integrity_ok:
        result["status"] = "FAIL"

    _print_summary(result)
    print(f"Integrity: {integrity_ok}")

    if args.no_write:
        print("--no-write: report files were not written.")
    else:
        json_path, markdown_path = write_outputs(
            result, args.output, args.report, overwrite=args.overwrite
        )
        print(f"Raw results: {json_path}")
        print(f"Markdown report: {markdown_path}")

    return 0 if result["status"] in {"PASS", "PARTIAL"} else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI with clear errors and no fabricated output."""
    args = _parse_args(list(argv) if argv is not None else None)
    try:
        return _run(args)
    except ArtifactExistsError as exc:
        print(f"BLOCKED: {exc}")
        return 2
    except (BenchmarkError, RuntimeError, OSError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
