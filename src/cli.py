"""CLI entrypoint.

Single entrypoint: ``python -m src.cli <command>``. The CLI wires configuration
to the ingestion and index modules directly and forwards benchmark, report and
pipeline work to the canonical scripts in ``scripts/`` as child processes. That
keeps one implementation per experiment (the scripts), gives each model-backed
run its own process, and leaves the CLI free of duplicated benchmark logic.

Commands:
``db-check``, ``ingest``, ``index build|list|drop|tune``, ``benchmark``,
``evaluate``, ``evaluate-judge``, ``report``, ``pipeline``, ``serve``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import psycopg
import typer

from src.config import PROJECT_ROOT, get_settings
from src.db import check_database
from src.ingestion import IngestionError
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import SentenceTransformerEmbedder
from src.ingestion.indexer import (
    UpsertStats,
    build_hnsw_index,
    build_ivfflat_index,
    collect_import_stats,
    drop_index,
    list_vector_indexes,
    upsert_corpus,
)
from src.ingestion.loader import inspect_source
from src.schema.retrieval import ChunkingConfig, ChunkStrategy

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Apex RAG -- vector retrieval and reranking for the card-services knowledge base.",
)

index_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Inspect or build shadow vector indexes on the chunks table.",
)
app.add_typer(index_app, name="index")

BENCHMARK_SCRIPTS: dict[str, str] = {
    "dense": "benchmark_dense.py",
    "sparse": "benchmark_sparse.py",
    "hybrid": "benchmark_hybrid.py",
    "rerank": "benchmark_rerank.py",
    "rag": "benchmark_rag.py",
    "retrieval": "benchmark_retrieval.py",
    "index": "benchmark_index.py",
    "chunking": "benchmark_chunking.py",
    "fusion": "benchmark_fusion.py",
}
"""Suite name -> canonical script. ``all`` goes through ``benchmark_all.py``."""


@app.callback()
def _main() -> None:
    """Apex RAG command line.

    An explicit callback is what keeps this a command *group*. Without one,
    Typer collapses a single-command app into a plain command and `db-check`
    is parsed as a stray argument to it.
    """


@app.command("db-check")
def db_check() -> None:
    """Verify PostgreSQL, pgvector, the schema and the HNSW index.

    Exits non-zero when anything is missing, so it works as a gate in CI.
    """
    results = check_database()
    for result in results:
        typer.echo(f"[{'PASS' if result.ok else 'FAIL'}] {result.name}: {result.detail}")

    failures = [result for result in results if not result.ok]
    if failures:
        typer.echo(f"\n{len(failures)} of {len(results)} checks failed.")
        raise typer.Exit(code=1)
    typer.echo(f"\nAll {len(results)} checks passed.")


@app.command("ingest")
def ingest(
    strategy: ChunkStrategy = typer.Option(
        "document",
        "--strategy",
        "-s",
        help=(
            "Chunking strategy: document, fixed_window or structured. "
            "'contextual' is retired; use document with context_header."
        ),
    ),
) -> None:
    """Load, validate, embed and store the knowledge base.

    Idempotent: every write is an upsert on a key the corpus already carries, so
    running the same strategy twice rewrites the same rows rather than appending
    duplicates. The ``document`` strategy writes one row per document.

    Exits non-zero if the corpus fails validation, a vector is rejected, or the
    database does not hold what was just written.
    """
    settings = get_settings()

    try:
        stats = _run_ingest(strategy)
    except (IngestionError, psycopg.Error, ConnectionError) as exc:
        typer.echo(f"\n[FAIL] {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("\nDatabase:")
    typer.echo(
        f"  {stats.documents_written} documents written "
        f"({stats.documents_inserted} new, {stats.documents_updated} updated)"
    )
    typer.echo(
        f"  {stats.chunks_written} chunks written "
        f"({stats.chunks_inserted} new, {stats.chunks_updated} updated)"
    )

    typer.echo("\nVerification:")
    failures, total = _verify(
        strategy,
        settings.expected_document_count,
        settings.embedding_dim,
        expected_chunks=stats.chunks_written,
    )

    if failures:
        typer.echo(f"\n[FAIL] {len(failures)} of {total} checks failed.", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"\nAll {total} checks passed.")


@index_app.command("build")
def index_build(
    method: str = typer.Option("hnsw", "--method", help="hnsw or ivfflat"),
    m: int = typer.Option(16, "--m", help="HNSW connections per layer (2..100)"),
    ef_construction: int = typer.Option(
        64, "--ef-construction", help="HNSW build candidate list (4..1000)"
    ),
    lists: int = typer.Option(100, "--lists", help="IVFFlat list count (1..32768)"),
    name: str | None = typer.Option(None, "--name", help="Explicit index name"),
    concurrently: bool = typer.Option(
        False, "--concurrently", help="CREATE INDEX CONCURRENTLY"
    ),
    allow_production: bool = typer.Option(
        False,
        "--allow-production",
        help="Permit the served bootstrap index name (never do this casually)",
    ),
) -> None:
    """Build one versioned shadow index and print what PostgreSQL holds."""
    resolved = method.lower()
    if resolved not in ("hnsw", "ivfflat"):
        raise typer.BadParameter("--method must be hnsw or ivfflat")
    try:
        if resolved == "hnsw":
            stats = build_hnsw_index(
                m=m,
                ef_construction=ef_construction,
                name=name,
                concurrently=concurrently,
                allow_production=allow_production,
            )
        else:
            stats = build_ivfflat_index(
                lists=lists,
                name=name,
                concurrently=concurrently,
                allow_production=allow_production,
            )
    except (IngestionError, psycopg.Error) as exc:
        typer.echo(f"[FAIL] {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    timing = f", {stats.build_ms:.1f} ms" if stats.build_ms is not None else ""
    typer.echo(
        f"Built {stats.name} ({stats.method}, {stats.size_pretty}{timing})"
    )
    if stats.parameters:
        rendered = ", ".join(f"{key}={value}" for key, value in stats.parameters.items())
        typer.echo(f"  parameters: {rendered}")
    typer.echo(f"  definition: {stats.definition}")


@index_app.command("list")
def index_list() -> None:
    """List every HNSW / IVFFlat index on chunks."""
    try:
        indexes = list_vector_indexes()
    except psycopg.Error as exc:
        typer.echo(f"[FAIL] {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not indexes:
        typer.echo("No vector indexes found on chunks.")
        return
    typer.echo(f"{'name':<40} {'method':<8} {'size':>10} {'valid':>6} parameters")
    for item in indexes:
        parameters = ", ".join(
            f"{key}={value}" for key, value in sorted(item.parameters.items())
        )
        typer.echo(
            f"{item.name:<40} {item.method:<8} {item.size_pretty:>10} "
            f"{'yes' if item.valid else 'no':>6} {parameters}"
        )


@index_app.command("drop")
def index_drop(
    name: str = typer.Option(..., "--name", help="Index name to drop"),
    concurrently: bool = typer.Option(False, "--concurrently"),
    allow_production: bool = typer.Option(
        False, "--allow-production", help="Permit dropping the served bootstrap index"
    ),
) -> None:
    """Drop one shadow index; the bootstrap index needs an explicit override."""
    try:
        drop_index(
            name,
            concurrently=concurrently,
            allow_production=allow_production,
        )
    except (IngestionError, psycopg.Error) as exc:
        typer.echo(f"[FAIL] {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Dropped index {name}.")


@index_app.command("tune")
def index_tune(
    limit: int | None = typer.Option(None, "--limit", help="First N eval queries"),
    synthetic_scale: int = typer.Option(
        1, "--synthetic-scale", help="Duplicate vectors N times (artifact labelled SYNTHETIC)"
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
    no_write: bool = typer.Option(False, "--no-write"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run the HNSW / IVFFlat parameter sweep (``scripts/benchmark_index.py``)."""
    forwarded: list[str] = ["--synthetic-scale", str(synthetic_scale)]
    forwarded += _common_flags(
        limit=limit, overwrite=overwrite, no_write=no_write, dry_run=dry_run
    )
    _run_script("benchmark_index.py", forwarded)


@app.command("benchmark")
def benchmark(
    suite: str = typer.Option(
        "all",
        "--suite",
        help="dense|sparse|hybrid|rerank|rag|retrieval|index|chunking|fusion|all",
    ),
    limit: int | None = typer.Option(None, "--limit", help="First N eval queries"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    no_write: bool = typer.Option(False, "--no-write"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run one benchmark suite through its canonical script."""
    selected = suite.lower()
    limit_supported = {"retrieval", "index", "chunking", "fusion", "rag", "all"}
    if selected == "all":
        script = "benchmark_all.py"
        forwarded: list[str] = ["--suites", "all"]
    else:
        script = BENCHMARK_SCRIPTS.get(selected, "")
        if not script:
            options = ", ".join(sorted([*BENCHMARK_SCRIPTS, "all"]))
            raise typer.BadParameter(f"unknown suite {suite!r}; choose from {options}")
        forwarded = []
    if limit is not None:
        if selected not in limit_supported:
            raise typer.BadParameter(f"suite {suite!r} does not accept --limit")
        forwarded.extend(["--limit", str(limit)])
    forwarded += _common_flags(
        limit=None, overwrite=overwrite, no_write=no_write, dry_run=dry_run
    )
    _run_script(script, forwarded)


@app.command("evaluate")
def evaluate(
    artifact: Path = typer.Option(
        PROJECT_ROOT / "reports" / "unified" / "benchmark.json",
        "--artifact",
        help="Unified benchmark artifact to read",
    ),
    arm: str | None = typer.Option(None, "--arm", help="Only print one arm"),
) -> None:
    """Print the metrics recorded in a unified artifact.

    This command reads the JSON the benchmark wrote; it recomputes nothing and
    never fills a missing metric with a zero.
    """
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        typer.echo(f"[FAIL] cannot read artifact {artifact}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not isinstance(payload, dict):
        typer.echo(f"[FAIL] artifact {artifact} is not a JSON object", err=True)
        raise typer.Exit(code=1)
    records = payload.get("arms")
    if not isinstance(records, list):
        typer.echo(
            f"[FAIL] artifact {artifact} has no arms list; was it a unified benchmark?",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(
        f"status={payload.get('status')}  reason={payload.get('status_reason')}  "
        f"generated={payload.get('generated_at')}"
    )
    typer.echo(
        f"{'arm':<14} {'recall':>8} {'mrr':>8} {'ndcg':>8} {'p99ms':>10} "
        f"{'evaluated':>9} status"
    )
    printed = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        if arm is not None and record.get("name") != arm:
            continue
        metrics = record.get("metrics", {})
        latency = record.get("latency_ms", {}).get("query", {})
        typer.echo(
            f"{str(record.get('name')):<14} "
            f"{_metric(metrics.get('recall_at_k', {}).get('macro')):>8} "
            f"{_metric(metrics.get('mrr')):>8} "
            f"{_metric(metrics.get('ndcg_at_k', {}).get('macro')):>8} "
            f"{_metric(latency.get('p99')):>10} "
            f"{str(metrics.get('evaluated_queries', '-')):>9} "
            f"{record.get('status')}"
        )
        printed += 1
    if arm is not None and printed == 0:
        typer.echo(f"No arm named {arm!r} in {artifact}.", err=True)
        raise typer.Exit(code=1)


@app.command("evaluate-judge")
def evaluate_judge(
    provider: str = typer.Option("mock", "--provider", help="Answer provider: mock or real"),
    judge_provider: str = typer.Option(
        "mock", "--judge-provider", help="Judge provider: mock or real"
    ),
    judge_model: str | None = typer.Option(
        None, "--judge-model", help="Defaults to APEX_JUDGE_MODEL, then APEX_LLM_MODEL"
    ),
    judge_temperature: float | None = typer.Option(None, "--judge-temperature"),
    judge_max_tokens: int | None = typer.Option(None, "--judge-max-tokens"),
    judge_max_concurrency: int | None = typer.Option(None, "--judge-max-concurrency"),
    limit: int | None = typer.Option(None, "--limit", help="First N queries (smoke)"),
    top_k: int | None = typer.Option(None, "--top-k"),
    candidates: int | None = typer.Option(None, "--candidates"),
    max_context_chars: int | None = typer.Option(None, "--max-context-chars"),
    max_chunk_chars: int | None = typer.Option(None, "--max-chunk-chars"),
    retrieval_mode: str = typer.Option(
        "live", "--retrieval-mode", help="live or historical"
    ),
    rag_artifact: Path | None = typer.Option(
        None, "--rag-artifact", help="Re-judge an existing RAG JSON"
    ),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    allow_network: bool = typer.Option(False, "--allow-network"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    no_write: bool = typer.Option(False, "--no-write"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    docker_timeout: float = typer.Option(5.0, "--docker-timeout"),
) -> None:
    """Run Context Relevance and Faithfulness judging (``scripts/evaluate_judge.py``)."""
    if provider not in {"mock", "real"}:
        raise typer.BadParameter("--provider must be mock or real")
    if judge_provider not in {"mock", "real"}:
        raise typer.BadParameter("--judge-provider must be mock or real")
    if retrieval_mode not in {"live", "historical"}:
        raise typer.BadParameter("--retrieval-mode must be live or historical")
    forwarded = [
        "--provider",
        provider,
        "--judge-provider",
        judge_provider,
        "--retrieval-mode",
        retrieval_mode,
        "--docker-timeout",
        str(docker_timeout),
    ]
    optional_values: list[tuple[str, object]] = [
        ("--judge-model", judge_model),
        ("--judge-temperature", judge_temperature),
        ("--judge-max-tokens", judge_max_tokens),
        ("--judge-max-concurrency", judge_max_concurrency),
        ("--top-k", top_k),
        ("--candidates", candidates),
        ("--max-context-chars", max_context_chars),
        ("--max-chunk-chars", max_chunk_chars),
        ("--output-dir", output_dir),
        ("--rag-artifact", rag_artifact),
    ]
    for flag, value in optional_values:
        if value is not None:
            forwarded.extend([flag, str(value)])
    if allow_network:
        forwarded.append("--allow-network")
    forwarded += _common_flags(
        limit=limit, overwrite=overwrite, no_write=no_write, dry_run=dry_run
    )
    _run_script("evaluate_judge.py", forwarded)


@app.command("report")
def report(
    benchmark_path: Path = typer.Option(
        PROJECT_ROOT / "reports" / "unified" / "benchmark.json",
        "--benchmark",
    ),
    index_path: Path = typer.Option(
        PROJECT_ROOT / "reports" / "unified" / "index_benchmark_scale10.json",
        "--index",
    ),
    chunking_path: Path = typer.Option(
        PROJECT_ROOT / "reports" / "unified" / "chunking_benchmark_rerank.json",
        "--chunking",
    ),
    rerank_path: Path = typer.Option(
        PROJECT_ROOT / "reports" / "unified" / "rerank_20to5.json",
        "--rerank",
    ),
    judge_path: Path = typer.Option(
        PROJECT_ROOT / "reports" / "judge" / "judge_summary.json",
        "--judge",
    ),
    output: Path = typer.Option(
        PROJECT_ROOT / "reports" / "final_report.md", "--output"
    ),
    no_figures: bool = typer.Option(False, "--no-figures"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    no_write: bool = typer.Option(False, "--no-write"),
) -> None:
    """Assemble ``reports/final_report.md`` and figures from the artifacts."""
    forwarded = [
        "--benchmark",
        str(benchmark_path),
        "--index",
        str(index_path),
        "--chunking",
        str(chunking_path),
        "--rerank",
        str(rerank_path),
        "--judge",
        str(judge_path),
        "--output",
        str(output),
    ]
    if no_figures:
        forwarded.append("--no-figures")
    if overwrite:
        forwarded.append("--overwrite")
    if no_write:
        forwarded.append("--no-write")
    _run_script("report.py", forwarded)


@app.command("pipeline")
def pipeline(
    limit: int | None = typer.Option(None, "--limit", help="First N eval queries"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    no_write: bool = typer.Option(False, "--no-write"),
    skip_rerank: bool = typer.Option(False, "--skip-rerank"),
    synthetic_scale: int = typer.Option(1, "--synthetic-scale"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run retrieval, index and chunking suites, then assemble the report."""
    forwarded = _common_flags(
        limit=limit, overwrite=overwrite, no_write=no_write, dry_run=dry_run
    )
    if skip_rerank:
        forwarded.append("--skip-rerank")
    forwarded += ["--synthetic-scale", str(synthetic_scale)]
    _run_script("benchmark_all.py", forwarded)
    if dry_run or no_write:
        return
    report_forwarded = ["--overwrite"] if overwrite else []
    _run_script("report.py", report_forwarded)


@app.command("serve")
def serve(
    host: str | None = typer.Option(None, "--host", help="Bind address"),
    port: int | None = typer.Option(None, "--port", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Reload on code changes"),
) -> None:
    """Serve the read-only FastAPI app (``src.api.app:app``)."""
    settings = get_settings()
    try:
        import uvicorn
    except ImportError as exc:
        typer.echo(
            "[FAIL] uvicorn/FastAPI are not installed; install the 'api' extra "
            "(uv sync --extra api)",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    uvicorn.run(
        "src.api.app:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


def _common_flags(
    *,
    limit: int | None,
    overwrite: bool,
    no_write: bool,
    dry_run: bool,
) -> list[str]:
    """Forward the flags every benchmark script understands."""
    flags: list[str] = []
    if limit is not None:
        flags.extend(["--limit", str(limit)])
    if overwrite:
        flags.append("--overwrite")
    if no_write:
        flags.append("--no-write")
    if dry_run:
        flags.append("--dry-run")
    return flags


def _run_script(script: str, args: Sequence[str]) -> None:
    """Run a canonical script as a child process and propagate its exit code."""
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / script), *args]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise typer.Exit(code=completed.returncode)


def _metric(value: float | int | None) -> str:
    """Format a metric for the console, keeping ``None`` visible as ``-``."""
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _verify(
    strategy: str,
    expected_documents: int,
    expected_dimension: int,
    expected_chunks: int,
) -> tuple[list[str], int]:
    """Read the counts back out of PostgreSQL and compare them to the intent.

    Every figure comes from a query -- nothing here reports what the writer
    believed it wrote. ``expected_chunks`` is supplied by the caller because a
    multi-chunk strategy has no fixed chunks-per-document ratio.

    Returns:
        ``(failed_labels, checks_run)``.
    """
    stats = collect_import_stats()
    failures: list[str] = []

    added = stats.strategies.get(strategy, 0)
    checks: list[tuple[str, bool, str]] = [
        (
            "documents in database",
            stats.documents == expected_documents,
            f"count = {stats.documents}",
        ),
        (
            f"chunks for strategy '{strategy}'",
            added == expected_chunks,
            f"count = {added}, expected = {expected_chunks}",
        ),
        (
            "chunks with a non-null embedding",
            stats.embedded_chunks == stats.chunks and stats.chunks > 0,
            f"{stats.embedded_chunks} / {stats.chunks}",
        ),
        (
            "distinct doc_ids among chunks",
            stats.distinct_doc_ids == expected_documents,
            f"count = {stats.distinct_doc_ids}",
        ),
        (
            "embedding dimension",
            stats.min_dimension == expected_dimension and stats.max_dimension == expected_dimension,
            f"min = {stats.min_dimension}, max = {stats.max_dimension}, "
            f"expected = {expected_dimension}",
        ),
    ]

    for label, ok, detail in checks:
        typer.echo(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
        if not ok:
            failures.append(label)

    if stats.strategies:
        rendered = ", ".join(f"{name}={count}" for name, count in sorted(stats.strategies.items()))
        typer.echo(f"  [INFO] chunk strategies present: {rendered}")

    return failures, len(checks)


def _run_ingest(strategy: ChunkStrategy) -> UpsertStats:
    """Drive Load -> Validate -> Embed -> Write, printing progress."""
    config = ChunkingConfig(strategy=strategy)

    typer.echo("Loading documents...")
    report = inspect_source()
    typer.echo(f"Found {report.total} documents.\n")

    typer.echo("Validation:")
    typer.echo(f"  {len(report.documents)} valid")
    typer.echo(f"  {len(report.problems)} invalid")
    if report.problems:
        for problem in report.problems:
            typer.echo(f"    [INVALID] {problem}")
        raise IngestionError(f"{len(report.problems)} document(s) failed validation")

    chunks = chunk_documents(report.documents, config)

    settings = get_settings()
    typer.echo(
        f"\nEmbedding ({settings.embedding_model}, dim {settings.embedding_dim}, "
        f"batch {settings.embedding_batch_size}):"
    )
    embedder = SentenceTransformerEmbedder(settings)
    vectors = embedder.embed_documents(
        [chunk.embedded_text for chunk in chunks],
        on_batch=lambda done, total: typer.echo(f"  Batch {done}/{total}"),
    )

    typer.echo("\nWriting to PostgreSQL...")
    return upsert_corpus(report.documents, chunks, vectors)


if __name__ == "__main__":
    app()
