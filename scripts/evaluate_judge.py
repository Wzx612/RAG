r"""LLM-as-a-Judge evaluation: score an existing (or freshly generated) RAG run.

The script has two independent provider switches:

* ``--provider`` controls answer generation (default ``mock``);
* ``--judge-provider`` controls the judge (default ``mock``).

A real call is therefore always an explicit choice, and a mock run can never be
mistaken for real model evaluation in the artifact: it records ``is_real=False``,
its reasons start with ``MOCK`` and its status is ``PARTIAL``.

The judge consumes the exact evidence that was sent to the answer model. It
does not re-retrieve, does not read ``gold_response`` and does not modify the
database, ``data/`` or any earlier artifact. Real calls require the existing
``APEX_LLM_API_KEY`` / ``DEEPSEEK_API_KEY``; a missing key stops the run before
models are loaded or files are written, and that run is reported as ``BLOCKED``.

Recommended commands::

    python scripts/evaluate_judge.py --provider mock --judge-provider mock --limit 3
    python scripts/evaluate_judge.py --rag-artifact reports/judge/rag_answers.json
    python scripts/evaluate_judge.py --provider real --judge-provider real --limit 3 \\
        --retrieval-mode live --allow-network

Smoke test::

    python scripts/evaluate_judge.py --provider mock --judge-provider mock \\
        --limit 3 --no-write
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT, Settings  # noqa: E402
from src.evaluation.artifacts import ArtifactExistsError, ensure_writable  # noqa: E402
from src.evaluation.dense_benchmark import BenchmarkError, load_eval_queries  # noqa: E402
from src.evaluation.hybrid_benchmark import collect_dataset, collect_environment  # noqa: E402
from src.evaluation.judge import JudgeConfig, LLMJudge, MockJudgeProvider  # noqa: E402
from src.evaluation.judge_evaluation import (  # noqa: E402
    JudgeEvaluationError,
    load_rag_artifact,
    run_judge_evaluation,
)
from src.evaluation.judge_evaluation import (  # noqa: E402
    write_outputs as write_judge_outputs,
)
from src.evaluation.rag_benchmark import (  # noqa: E402
    RAGBenchmarkConfig,
    load_raw_queries,
    run_rag_benchmark,
)
from src.evaluation.rag_benchmark import (  # noqa: E402
    write_outputs as write_rag_outputs,
)
from src.rag.factory import (  # noqa: E402
    build_historical_retriever,
    build_pipeline,
    build_provider,
)
from src.rag.historical import DEFAULT_HISTORICAL  # noqa: E402
from src.rag.pipeline import RAGPipeline  # noqa: E402
from src.rag.provider import (  # noqa: E402
    LLMError,
    LLMProvider,
    MockLLMProvider,
    OpenAICompatibleProvider,
)
from src.schema.rag import LLMConfig  # noqa: E402

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "judge"
"""Fresh directory; existing benchmark reports are never written here."""

JUDGE_FILES = ("judge_results.jsonl", "judge_summary.json", "judge_report.md")
RAG_FILES = ("rag_answers.json", "rag_answers.md")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse generation, judge and output options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--provider",
        choices=("mock", "real"),
        default="mock",
        help="Answer-generation provider",
    )
    parser.add_argument(
        "--judge-provider",
        choices=("mock", "real"),
        default="mock",
        help="Judge provider",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model id; defaults to APEX_JUDGE_MODEL, else APEX_LLM_MODEL",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=None,
        help="Judge temperature (default: settings.judge_temperature)",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=None,
        help="Judge completion cap (default: settings.judge_max_tokens)",
    )
    parser.add_argument(
        "--judge-max-concurrency",
        type=int,
        default=None,
        help="Judge calls in flight (default: settings.judge_max_concurrency)",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("live", "historical"),
        default="live",
        help="live runs the accepted chain; historical replays the accepted RRF pool",
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
        "--max-context-chars", type=int, default=None, help="Hard context character budget"
    )
    parser.add_argument(
        "--max-chunk-chars", type=int, default=None, help="Per-chunk character cap"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Judge the first N queries only; this is a smoke run, not a full evaluation",
    )
    parser.add_argument(
        "--rag-artifact",
        type=Path,
        default=None,
        help="Re-judge an existing RAG JSON instead of generating answers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory (default: reports/judge)",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face network access for the RAG models",
    )
    parser.add_argument("--dry-run", action="store_true", help="Check gates and exit")
    parser.add_argument("--no-write", action="store_true", help="Do not write any file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in the selected output directory",
    )
    parser.add_argument(
        "--docker-timeout",
        type=float,
        default=5.0,
        help="Seconds allowed for the single read-only Docker probe",
    )
    return parser.parse_args(argv)


def _settings_from_args(args: argparse.Namespace) -> Settings:
    """Apply RAG overrides without mutating the cached settings singleton."""
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


def _sha256_file(path: Path) -> str:
    """Return the SHA256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rag_provider(args: argparse.Namespace, settings: Settings) -> LLMProvider:
    """Build the answer provider, failing before model load when keyless."""
    if args.provider == "mock":
        return MockLLMProvider()
    provider = build_provider(settings)
    try:
        provider.ensure_configured()
    except LLMError as exc:
        raise JudgeEvaluationError(
            "real answer provider is not configured; set APEX_LLM_API_KEY or "
            "DEEPSEEK_API_KEY, or use --provider mock"
        ) from exc
    return provider


def _judge_provider(args: argparse.Namespace, settings: Settings) -> LLMProvider:
    """Build the judge provider, reusing the existing key and endpoint."""
    if args.judge_provider == "mock":
        return MockJudgeProvider()
    model = args.judge_model or settings.resolved_judge_model
    temperature = (
        settings.judge_temperature
        if args.judge_temperature is None
        else args.judge_temperature
    )
    max_tokens = (
        settings.judge_max_tokens if args.judge_max_tokens is None else args.judge_max_tokens
    )
    provider = OpenAICompatibleProvider(
        LLMConfig(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=model,
            connect_timeout=settings.llm_connect_timeout,
            read_timeout=settings.llm_read_timeout,
            max_retries=settings.llm_max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )
    try:
        provider.ensure_configured()
    except LLMError as exc:
        raise JudgeEvaluationError(
            "real judge is not configured; set APEX_LLM_API_KEY or DEEPSEEK_API_KEY, "
            "or use --judge-provider mock"
        ) from exc
    return provider


def _judge_config(args: argparse.Namespace, settings: Settings) -> JudgeConfig:
    """Resolve judge generation options and the low-score threshold."""
    temperature = (
        settings.judge_temperature
        if args.judge_temperature is None
        else args.judge_temperature
    )
    max_tokens = (
        settings.judge_max_tokens if args.judge_max_tokens is None else args.judge_max_tokens
    )
    return JudgeConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        low_score_threshold=settings.judge_low_score_threshold,
    )


def _print_facts(
    settings: Settings,
    args: argparse.Namespace,
    *,
    queries: int,
    output_dir: Path,
    key_present: bool,
) -> None:
    """Print resolved inputs without printing a key or a prompt."""
    print("LLM-as-a-Judge evaluation")
    print("")
    print("Parameters:")
    for key in (
        "provider",
        "judge_provider",
        "judge_model",
        "judge_temperature",
        "judge_max_tokens",
        "judge_max_concurrency",
        "retrieval_mode",
        "top_k",
        "candidates",
        "max_context_chars",
        "max_chunk_chars",
        "limit",
        "rag_artifact",
        "allow_network",
    ):
        print(f"  {key}: {getattr(args, key, None)}")
    effective_model = args.judge_model or settings.resolved_judge_model
    print(f"  resolved_judge_model: {effective_model}")
    print(f"  key_present: {key_present}")
    print(f"  dataset_queries: {queries}")
    print(f"  output_dir: {output_dir}")
    print("")


def _print_summary(result: Mapping[str, Any]) -> None:
    """Print the headline counts and explicitly label mock output."""
    summary = result["summary"]
    provider = result["provider"]
    print("")
    print(f"Status: {result['status']} ({result['status_reason']})")
    print(
        f"Queries judged: {summary['queries']}/{summary['expected_queries']} "
        f"(NOT RUN={summary.get('not_run_queries', 0)}, "
        f"RAG failures={summary['rag_failures']}, "
        f"both metrics success={summary['both_metrics_success']})"
    )
    for key in ("context_relevance", "faithfulness"):
        item = summary["metrics"][key]
        print(
            f"{key}: attempted={item['attempted']} success={item['success']} "
            f"failed={item['failed']} valid={item['valid_scores']} "
            f"mean={_fmt(item['mean'])} median={_fmt(item['median'])} "
            f"low={item['low_score_count']}"
        )
    print(
        f"Judge: {provider['kind']} / {provider['name']} / model {provider['model']}; "
        f"same_model={summary.get('same_model')}"
    )
    if not provider["is_real"]:
        print("MOCK judge: scores are deterministic fixtures, not model judgement.")


def _fmt(value: object) -> str:
    """Format an optional number."""
    if value is None:
        return "-"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    return str(value)


def _collect_environment_safely(settings: Settings, docker_timeout: float) -> dict[str, Any]:
    """Collect runtime facts; a re-judge can still run when the DB is absent."""
    try:
        environment = collect_environment(settings.retrieval_chunk_strategy)
    except Exception as exc:  # noqa: BLE001 - facts must not block artifact-only judging
        environment = {
            "execution_environment": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }
    environment["docker_status"] = _docker_status(docker_timeout)
    environment["command"] = " ".join([Path(sys.executable).name, *sys.argv])
    environment["memory_before"] = _memory_snapshot()
    return environment


def _collect_dataset_safely(queries: Sequence[Any]) -> dict[str, Any]:
    """Collect dataset hashes; a DB failure is recorded rather than fabricated."""
    try:
        return collect_dataset(queries)
    except Exception as exc:  # noqa: BLE001 - an artifact-only judge still needs file facts
        return {"queries": len(queries), "error": f"{type(exc).__name__}: {exc}"}


def _run(args: argparse.Namespace) -> int:
    """Run RAG (unless re-judging) and judge, then write the evidence."""
    settings = _settings_from_args(args)
    if settings.rag_top_k > settings.rag_candidates:
        raise JudgeEvaluationError("--top-k must not exceed --candidates")

    all_queries = load_eval_queries()
    raw_queries = load_raw_queries()
    missing_raw = [query.query_id for query in all_queries if query.query_id not in raw_queries]
    if missing_raw:
        raise JudgeEvaluationError(f"raw query records missing for: {missing_raw}")
    if args.limit is not None and (args.limit < 1 or args.limit > len(all_queries)):
        raise JudgeEvaluationError(
            f"--limit must be between 1 and {len(all_queries)}, got {args.limit}"
        )
    if args.judge_max_concurrency is not None and args.judge_max_concurrency < 1:
        raise JudgeEvaluationError("--judge-max-concurrency must be at least 1")

    output_dir = settings.resolve(args.output_dir or settings.judge_output_dir)
    key_present = bool(
        settings.llm_api_key and settings.llm_api_key.get_secret_value().strip()
    )
    _print_facts(
        settings,
        args,
        queries=len(all_queries),
        output_dir=output_dir,
        key_present=key_present,
    )

    uses_real = args.provider == "real" or args.judge_provider == "real"
    if uses_real and not key_present:
        print(
            "BLOCKED: a real provider was requested but APEX_LLM_API_KEY / "
            "DEEPSEEK_API_KEY is absent."
        )
        print(
            "No artifact was written. Mock scores must not stand in for a real "
            "evaluation."
        )
        return 2

    if args.dry_run:
        print(
            "Dry run: no retrieval model load, no provider call, no files written. "
            f"key_present={key_present}."
        )
        return 0

    environment = _collect_environment_safely(settings, args.docker_timeout)
    dataset = _collect_dataset_safely(all_queries)
    if not args.no_write:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Check every target before any paid call or model load, so a clash is
        # reported up front instead of after the run has already happened.
        targets = [output_dir / name for name in JUDGE_FILES]
        if args.rag_artifact is None:
            targets.extend(output_dir / name for name in RAG_FILES)
        ensure_writable(targets, overwrite=args.overwrite)

    rag_result: dict[str, Any]
    rag_path: Path | None
    rag_sha: str | None
    if args.rag_artifact is not None:
        rag_result = load_rag_artifact(args.rag_artifact)
        rag_path = args.rag_artifact
        rag_sha = _sha256_file(args.rag_artifact)
        print(f"Re-judging existing RAG artifact: {rag_path}")
        known_ids = {query.query_id for query in all_queries}
        unknown_ids = [
            str(record.get("query_id"))
            for record in rag_result["per_query"]
            if str(record.get("query_id")) not in known_ids
        ]
        if unknown_ids:
            raise JudgeEvaluationError(
                "RAG artifact contains query ids absent from the labelled set: "
                + ", ".join(unknown_ids[:5])
            )
        if len(rag_result["per_query"]) < (args.limit or 0):
            raise JudgeEvaluationError(
                f"--limit {args.limit} exceeds the {len(rag_result['per_query'])} "
                "records in the RAG artifact"
            )
    else:
        provider = _rag_provider(args, settings)
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
        queries = all_queries[: args.limit] if args.limit is not None else all_queries
        rag_result = run_rag_benchmark(
            pipeline,
            queries,
            raw_queries=raw_queries,
            config=RAGBenchmarkConfig(
                top_k=settings.rag_top_k,
                candidates=settings.rag_candidates,
                max_context_chars=settings.rag_max_context_chars,
                max_chunk_chars=settings.rag_max_chunk_chars,
                provider_kind=args.provider,
                retrieval_mode=args.retrieval_mode,
            ),
            environment=environment,
            dataset=dataset,
            progress=print,
        )
        rag_path = None if args.no_write else output_dir / RAG_FILES[0]
        rag_sha = None
        if rag_path is not None:
            json_path, markdown_path = write_rag_outputs(
                rag_result,
                rag_path,
                output_dir / RAG_FILES[1],
                overwrite=args.overwrite,
            )
            rag_sha = _sha256_file(json_path)
            print(f"RAG answers: {json_path}")
            print(f"RAG Markdown: {markdown_path}")

    judge_provider = _judge_provider(args, settings)
    judge = LLMJudge(judge_provider, _judge_config(args, settings))
    concurrency = (
        settings.judge_max_concurrency
        if args.judge_max_concurrency is None
        else args.judge_max_concurrency
    )
    result = run_judge_evaluation(
        rag_result,
        judge,
        expected_queries=len(all_queries),
        limit=args.limit,
        max_concurrency=concurrency,
        low_score_threshold=settings.judge_low_score_threshold,
        environment=environment,
        dataset=dataset,
        transport={
            "connect_timeout_s": settings.llm_connect_timeout,
            "read_timeout_s": settings.llm_read_timeout,
            "max_retries": settings.llm_max_retries,
        },
        rag_artifact_path=rag_path,
        rag_artifact_sha256=rag_sha,
        api_key_present=key_present,
        commands=[
            "python scripts/evaluate_judge.py " + " ".join(sys.argv[1:])
        ],
        progress=print,
    )

    environment_after = _collect_environment_safely(settings, args.docker_timeout)
    dataset_after = _collect_dataset_safely(all_queries)
    count_keys = (
        "documents",
        "chunks",
        "distinct_chunk_docs",
        "strategy_chunks",
        "strategy_docs",
    )
    counts_before = {key: environment.get(key) for key in count_keys}
    counts_after = {key: environment_after.get(key) for key in count_keys}
    integrity_ok = (
        counts_before == counts_after
        and dataset.get("hashes") == dataset_after.get("hashes")
        and dataset.get("missing_gold_docs") == dataset_after.get("missing_gold_docs")
        if "error" not in environment and "error" not in dataset
        else None
    )
    result["integrity"].update(
        {
            "counts_before": counts_before,
            "counts_after": counts_after,
            "hashes_after": dataset_after.get("hashes"),
            "missing_gold_docs_after": dataset_after.get("missing_gold_docs"),
            "data_integrity_ok": integrity_ok,
        }
    )
    if integrity_ok is False:
        result["status"] = "FAIL"
        result["status_reason"] = "database counts or data hashes changed during the run"
    _print_summary(result)
    print(f"Integrity: {integrity_ok}")

    if args.no_write:
        print("--no-write: judge artifacts were not written.")
    else:
        jsonl_path, summary_path, report_path = (
            output_dir / JUDGE_FILES[0],
            output_dir / JUDGE_FILES[1],
            output_dir / JUDGE_FILES[2],
        )
        ensure_writable(
            (jsonl_path, summary_path, report_path), overwrite=args.overwrite
        )
        write_judge_outputs(
            result,
            jsonl_path=jsonl_path,
            summary_path=summary_path,
            report_path=report_path,
            overwrite=args.overwrite,
        )
        print(f"Per-query JSONL: {jsonl_path}")
        print(f"Summary JSON: {summary_path}")
        print(f"Markdown: {report_path}")

    return 0 if result["status"] in {"PASS", "PARTIAL"} else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI with clear errors and no fabricated output."""
    args = _parse_args(list(argv) if argv is not None else None)
    try:
        return _run(args)
    except (
        ArtifactExistsError,
        BenchmarkError,
        JudgeEvaluationError,
        LLMError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
