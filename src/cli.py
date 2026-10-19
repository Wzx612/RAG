"""CLI entrypoint.

Single entrypoint: ``python -m src.cli {db-check,ingest}``. Wires config to
modules. No business logic of its own.

Only ``db-check`` and ``ingest`` exist. Retrieval evaluation is exposed by the
read-only scripts ``scripts/benchmark_dense.py``, ``scripts/benchmark_sparse.py``
and ``scripts/benchmark_hybrid.py``. The ``index``, ``benchmark``, ``evaluate``,
``report`` and ``pipeline`` subcommands are planned and are not implemented.
"""

from __future__ import annotations

import psycopg
import typer

from src.config import get_settings
from src.db import check_database
from src.ingestion import IngestionError
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import SentenceTransformerEmbedder
from src.ingestion.indexer import UpsertStats, collect_import_stats, upsert_corpus
from src.ingestion.loader import inspect_source
from src.schema.retrieval import ChunkingConfig, ChunkStrategy

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Apex RAG -- vector retrieval and reranking for the card-services knowledge base.",
)


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
        help="Chunking strategy. Only 'document' is implemented at this stage.",
    ),
) -> None:
    """Load, validate, embed and store the knowledge base.

    Idempotent: every write is an upsert on a key the corpus already carries, so
    running this twice leaves 150 rows, not 300.

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
    failures, total = _verify(strategy, settings.expected_document_count, settings.embedding_dim)

    if failures:
        typer.echo(f"\n[FAIL] {len(failures)} of {total} checks failed.", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"\nAll {total} checks passed.")


def _verify(
    strategy: str, expected_documents: int, expected_dimension: int
) -> tuple[list[str], int]:
    """Read the counts back out of PostgreSQL and compare them to the intent.

    Every figure comes from a query -- nothing here reports what the writer
    believed it wrote.

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
            added == expected_documents,
            f"count = {added}",
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
