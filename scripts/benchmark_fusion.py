"""RRF sensitivity sweep: ``rrf_k`` × ``candidate_k`` for the fused arms.

Read-only. Each grid cell runs the same unified benchmark with a different
fusion configuration; the artifact keeps every cell separate and never claims
a winner when the corpus is too small to separate them.

Usage::

    python scripts/benchmark_fusion.py --dry-run
    python scripts/benchmark_fusion.py --limit 3 --arms rrf
    python scripts/benchmark_fusion.py --rrf-k 10 60 100 --candidate-k 10 50
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg  # noqa: E402
from src.config import PROJECT_ROOT, get_settings  # noqa: E402
from src.db import connection, scalar  # noqa: E402
from src.evaluation.artifacts import ArtifactExistsError, ensure_writable  # noqa: E402
from src.evaluation.benchmark import Arm  # noqa: E402
from src.evaluation.dense_benchmark import (  # noqa: E402
    BenchmarkError,
    EvalQuery,
    load_eval_queries,
)
from src.evaluation.fusion_benchmark import (  # noqa: E402
    FusionConfig,
    run_fusion_sensitivity,
    write_outputs,
)
from src.ingestion.embedder import SentenceTransformerEmbedder, validate_vectors  # noqa: E402
from src.retrieval.registry import (  # noqa: E402
    ArmContext,
    arm_parameters,
    arm_spec,
    build_arm,
)
from src.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from src.schema.retrieval import RerankConfig, RetrievalStrategy  # noqa: E402

DATA_DIR = get_settings().resolve(get_settings().data_dir)
KB_PATH = DATA_DIR / "kb_documents.json"
QUERY_PATH = DATA_DIR / "eval_queries.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "unified" / "fusion_sensitivity.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "unified" / "fusion_sensitivity.md"
FUSED_ARMS = ("rrf", "rrf_rerank")


class _ArtifactWriter:
    """Write the artifact once, honouring ``--no-write`` and ``--overwrite``."""

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
            print("--no-write/--dry-run: artifacts were not written")
            return
        write_outputs(payload, self.output, self.report, overwrite=self.overwrite)
        print(f"Wrote {self.output} and {self.report}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the sweep's read-only options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rrf-k", type=int, nargs="+", default=[10, 60, 100])
    parser.add_argument("--candidate-k", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--arms", nargs="+", choices=FUSED_ARMS, default=["rrf"])
    parser.add_argument(
        "--chunk-strategy",
        choices=("document", "fixed_window", "structured"),
        default="document",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=40)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N eval queries (smoke test; the full run uses 30)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the grid and exit")
    parser.add_argument("--no-write", action="store_true", help="Do not write artifacts")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing artifacts at the selected paths",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def _blocked_payload(
    message: str,
    *,
    config: FusionConfig,
    arms: list[str],
    environment: dict[str, Any] | None = None,
    dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a renderable BLOCKED artifact without pretending a run happened."""
    return {
        "schema_version": 1,
        "experiment_name": "rrf_sensitivity",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "BLOCKED",
        "status_reason": message,
        "parameters": {
            "top_k": config.top_k,
            "chunk_strategy": config.chunk_strategy,
            "expected_queries": config.expected_queries,
            "queries": 0,
            "rrf_k_values": list(config.rrf_k_values),
            "candidate_k_values": list(config.candidate_k_values),
            "cells": [],
            "arms": arms,
        },
        "environment": environment or {},
        "dataset": dataset or {},
        "summary": [],
        "cells": {},
        "limitations": [message],
    }


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


def _read_environment(
    strategy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect read-only database facts and the strategy's chunk count."""
    settings = get_settings()
    with connection() as conn:
        address = conn.execute(
            "SELECT coalesce(inet_server_addr()::text, 'unix-socket'), inet_server_port()"
        ).fetchone()
        dimensions = [
            int(row[0])
            for row in conn.execute(
                "SELECT DISTINCT vector_dims(embedding) FROM chunks "
                "WHERE embedding IS NOT NULL AND chunk_strategy = %s",
                (strategy,),
            ).fetchall()
        ]
        stats = conn.execute(
            "SELECT count(*), count(DISTINCT doc_id) FROM chunks WHERE chunk_strategy = %s",
            (strategy,),
        ).fetchone()
        environment = {
            "postgres_version": str(scalar(conn, "SHOW server_version")),
            "vector_version": str(
                scalar(
                    conn,
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
                )
                or "missing"
            ),
            "server_address": f"{address[0]}:{address[1]}" if address else "unknown",
            "documents": int(scalar(conn, "SELECT count(*) FROM documents") or 0),
            "chunks": int(scalar(conn, "SELECT count(*) FROM chunks") or 0),
            "embedding_dimensions": sorted(dimensions),
            "embedding_model": settings.embedding_model,
            "embedding_device": settings.embedding_device,
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "packages": {
                "sentence-transformers": _package_version("sentence-transformers"),
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
                "psycopg": _package_version("psycopg"),
            },
        }
        chunk_stats = {
            "chunk_count": int(stats[0]) if stats else 0,
            "documents": int(stats[1]) if stats else 0,
        }
    return environment, chunk_stats


def _gold_ids(queries: list[EvalQuery]) -> list[str]:
    """Return every distinct gold doc id in the selected query set."""
    return sorted({doc_id for query in queries for doc_id in query.relevant_doc_ids})


def _missing_gold(
    conn: psycopg.Connection, strategy: str, gold: list[str]
) -> list[str]:
    """Return gold doc ids that have no chunk under ``strategy``."""
    if not gold:
        return []
    rows = conn.execute(
        "SELECT DISTINCT doc_id FROM chunks WHERE chunk_strategy = %s AND doc_id = ANY(%s)",
        (strategy, gold),
    ).fetchall()
    present = {str(row[0]) for row in rows}
    return [doc_id for doc_id in gold if doc_id not in present]


def _warm_embedder(queries: list[EvalQuery]) -> tuple[Any, dict[str, Any]]:
    """Load the embedding model once and warm it with a real query."""
    embedder = SentenceTransformerEmbedder()
    started = time.perf_counter()
    _ = embedder.model
    model_load_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    vector = embedder.embed_query(queries[0].query if queries else "warmup")
    warm_ms = (time.perf_counter() - started) * 1000.0
    validate_vectors([vector], embedder.dimension, source="warmup")
    return embedder, {
        "model_id": get_settings().embedding_model,
        "dimension": embedder.dimension,
        "model_load_ms": round(model_load_ms, 3),
        "warm_encode_ms": round(warm_ms, 3),
    }


def _build_runs(
    config: FusionConfig,
    arm_names: list[str],
    *,
    embedder: Any,
    reranker: Any,
    ef_search: int,
) -> dict[tuple[int, int], list[Arm]]:
    """Build the fused arms for every grid cell."""
    runs: dict[tuple[int, int], list[Arm]] = {}
    for rrf_k, candidate_k in config.cells:
        context = ArmContext(
            embedder=embedder,
            reranker=reranker,
            top_k=config.top_k,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            ef_search=ef_search,
            chunk_strategy=cast(RetrievalStrategy, config.chunk_strategy),
        )
        arms: list[Arm] = []
        for name in arm_names:
            spec = arm_spec(name)
            arms.append(
                Arm(
                    name=name,
                    retriever=build_arm(name, context),
                    description=spec.description,
                    parameters=arm_parameters(name, context),
                )
            )
        runs[(rrf_k, candidate_k)] = arms
    return runs


def _print_plan(
    config: FusionConfig, arm_names: list[str], query_count: int
) -> None:
    """Print the grid before any model load."""
    print(
        f"Grid: {len(config.cells)} cells; rrf_k={config.rrf_k_values}; "
        f"candidate_k={config.candidate_k_values}; arms={', '.join(arm_names)}"
    )
    print(
        f"  strategy={config.chunk_strategy} queries={query_count} "
        f"top_k={config.top_k}"
    )


def _print_summary(result: dict[str, Any]) -> None:
    """Print a compact per-cell summary."""
    print("\nResults:")
    print(
        f"  {'rrf_k':>5} {'cand':>5} {'arm':<12} {'recall':>7} {'mrr':>6} "
        f"{'p99ms':>9} evaluated status"
    )
    for row in result["summary"]:
        latency = row["latency_ms"]
        p99 = "-" if latency["p99"] is None else f"{latency['p99']:.2f}"
        print(
            f"  {row['rrf_k']:>5} {row['candidate_k']:>5} {row['arm']:<12} "
            f"{_num(row['recall_macro']):>7} {_num(row['mrr']):>6} {p99:>9} "
            f"{row['evaluated']:>9} {row['status']}"
        )
    print(f"  status: {result['status']} ({result['status_reason']})")


def _num(value: float | None) -> str:
    """Format a metric for the console."""
    return "-" if value is None else f"{value:.3f}"


def main(argv: list[str] | None = None) -> int:
    """Run the gate, dry-run or sweep, returning an exit code."""
    args = _parse_args(argv)
    if args.limit is not None and args.limit < 1:
        print("BLOCKED: --limit must be positive")
        return 2
    arm_names = list(dict.fromkeys(args.arms))
    try:
        config = FusionConfig(
            top_k=args.top_k,
            chunk_strategy=args.chunk_strategy,
            rrf_k_values=tuple(dict.fromkeys(args.rrf_k)),
            candidate_k_values=tuple(dict.fromkeys(args.candidate_k)),
        )
    except ValueError as exc:
        print(f"BLOCKED: invalid configuration: {exc}")
        return 2

    writer = _ArtifactWriter(
        args.output,
        args.report,
        overwrite=args.overwrite,
        enabled=not args.no_write and not args.dry_run,
    )
    if not args.no_write and not args.dry_run:
        try:
            ensure_writable((args.output, args.report), overwrite=args.overwrite)
        except ArtifactExistsError as exc:
            print(f"BLOCKED: {exc}")
            return 2

    try:
        environment, chunk_stats = _read_environment(args.chunk_strategy)
    except (psycopg.Error, ConnectionError) as exc:
        message = f"PostgreSQL is not reachable; sweep not started: {exc}"
        print(f"BLOCKED: {message}")
        writer(_blocked_payload(message, config=config, arms=arm_names))
        return 2

    expected_documents = get_settings().expected_document_count
    gate_problems: list[str] = []
    if environment["vector_version"] == "missing":
        gate_problems.append("pgvector extension is missing")
    if environment["documents"] != expected_documents:
        gate_problems.append(
            f"documents={environment['documents']}, expected {expected_documents}"
        )
    if chunk_stats["chunk_count"] == 0:
        gate_problems.append(
            f"chunk strategy {args.chunk_strategy!r} is not ingested; "
            f"run ingest --strategy {args.chunk_strategy}"
        )
    elif chunk_stats["documents"] != expected_documents:
        gate_problems.append(
            f"chunk strategy {args.chunk_strategy!r} covers "
            f"{chunk_stats['documents']} documents, expected {expected_documents}"
        )
    if environment["embedding_dimensions"] != [get_settings().embedding_dim]:
        gate_problems.append(
            f"embedding dimensions={environment['embedding_dimensions']}, "
            f"expected [{get_settings().embedding_dim}]"
        )

    try:
        all_queries = load_eval_queries(QUERY_PATH)
    except BenchmarkError as exc:
        message = f"eval queries are not usable; sweep not started: {exc}"
        print(f"BLOCKED: {message}")
        writer(
            _blocked_payload(
                message, config=config, arms=arm_names, environment=environment
            )
        )
        return 2

    selected = list(all_queries[: args.limit] if args.limit else all_queries)
    gold = _gold_ids(selected)
    with connection() as conn:
        missing = _missing_gold(conn, args.chunk_strategy, gold)
    if missing:
        gate_problems.append(
            f"strategy {args.chunk_strategy!r} has no chunk for {len(missing)} "
            f"gold doc(s): {', '.join(missing[:5])}"
        )

    dataset = {
        "queries": len(selected),
        "eval_queries_on_disk": len(all_queries),
        "query_limit": args.limit,
        "gold_docs": len(gold),
        "hashes": {
            "kb_documents.json": _sha256(KB_PATH),
            "eval_queries.json": _sha256(QUERY_PATH),
        },
    }
    if gate_problems:
        message = "environment gate failed: " + "; ".join(gate_problems)
        print(f"BLOCKED: {message}")
        writer(
            _blocked_payload(
                message,
                config=config,
                arms=arm_names,
                environment=environment,
                dataset=dataset,
            )
        )
        return 2

    _print_plan(config, arm_names, len(selected))
    if args.dry_run:
        print("Dry run only: no model load and no retrieval was executed.")
        return 0

    print("\nWarming embedding model...")
    try:
        embedder, embedding = _warm_embedder(selected)
    except Exception as exc:  # noqa: BLE001 -- a model failure blocks the sweep
        message = f"embedding failed; sweep not started: {type(exc).__name__}: {exc}"
        print(f"BLOCKED: {message}")
        writer(
            _blocked_payload(
                message,
                config=config,
                arms=arm_names,
                environment=environment,
                dataset=dataset,
            )
        )
        return 2

    reranker = None
    if any(arm_spec(name).needs_reranker for name in arm_names):
        settings = get_settings()
        reranker = CrossEncoderReranker(
            RerankConfig(
                enabled=True,
                model=settings.reranker_model,
                revision=settings.reranker_revision or None,
                batch_size=settings.reranker_batch_size,
                max_length=settings.reranker_max_length,
                device=settings.reranker_device,
                activation="identity",
                local_files_only=settings.reranker_local_files_only,
                cache_folder=str(settings.resolve(settings.reranker_cache_folder)),
            )
        )
        environment["reranker"] = {
            "model_id": reranker.model_id,
            "revision": reranker.revision,
            "device": reranker.config.device,
        }

    try:
        runs = _build_runs(
            config,
            arm_names,
            embedder=embedder,
            reranker=reranker,
            ef_search=args.ef_search,
        )
    except ValueError as exc:
        message = f"could not build the arms: {exc}"
        print(f"BLOCKED: {message}")
        writer(
            _blocked_payload(
                message,
                config=config,
                arms=arm_names,
                environment=environment,
                dataset=dataset,
            )
        )
        return 2

    try:
        result = run_fusion_sensitivity(
            runs,
            selected,
            config,
            environment=environment,
            dataset=dataset,
        )
    except Exception as exc:  # noqa: BLE001 -- record the failure, never fake a run
        message = f"sweep aborted: {type(exc).__name__}: {exc}"
        print(f"BLOCKED: {message}")
        writer(
            _blocked_payload(
                message,
                config=config,
                arms=arm_names,
                environment=environment,
                dataset=dataset,
            )
        )
        return 2

    result["embedding"] = embedding
    _print_summary(result)
    writer(result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
