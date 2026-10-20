"""Read-only dense retrieval diagnostic.

Answers four separate questions without conflating them:

1. does a real query return real chunks with valid distances? (``retrieve``)
2. does ``DenseConfig.ef_search`` reach the live PostgreSQL GUC? (read-back)
3. which plan does the query get naturally, and can it use HNSW? (EXPLAIN)
4. at the current corpus size, does ``ef_search`` change the returned rows?

``--force-hnsw`` runs a second retrieval inside a transaction with
``SET LOCAL enable_seqscan = off``. That is diagnosis only -- it is never a
production setting and lasts exactly one transaction.

The script issues no DDL and no writes; every statement is a SELECT, EXPLAIN,
SHOW-style ``current_setting`` read, or transaction-local ``set_config``.

Usage::

    python scripts/diagnose_dense.py
    python scripts/diagnose_dense.py --query-id Q-001 --top-k 10 --ef-search 7
    python scripts/diagnose_dense.py --ef-search 7 --top-k 50 --force-hnsw
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from pgvector import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import connection, current_ef_search, ef_search  # noqa: E402
from src.ingestion.embedder import SentenceTransformerEmbedder  # noqa: E402
from src.retrieval.dense import DENSE_SEARCH_SQL, DenseRetriever  # noqa: E402
from src.schema.retrieval import DenseConfig, SearchHit  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "eval_queries.json"


@contextmanager
def _forced_hnsw() -> Iterator[psycopg.Connection]:
    """Borrow a connection with seqscan disabled for the enclosing transaction."""
    with connection() as conn, conn.transaction():
        conn.execute("SET LOCAL enable_seqscan = off", prepare=False)
        yield conn


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the diagnostic's read-only options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--query-id", default="Q-001", help="eval query id from data/eval_queries.json"
    )
    parser.add_argument("--query", help="query text; overrides --query-id")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=40)
    parser.add_argument(
        "--chunk-strategy",
        choices=("document", "fixed_window", "structured"),
        default="document",
        help="Chunking strategy the diagnostic is scoped to",
    )
    parser.add_argument(
        "--force-hnsw",
        action="store_true",
        help="also run retrieval under SET LOCAL enable_seqscan=off (diagnosis only)",
    )
    return parser.parse_args(argv)


def _load_query(args: argparse.Namespace) -> tuple[str, str]:
    """Return ``(query_id, query_text)`` from the arguments or the eval file."""
    if args.query:
        return "<custom>", args.query
    queries = json.loads(DATA.read_text(encoding="utf-8"))
    for entry in queries:
        if entry["query_id"] == args.query_id:
            return entry["query_id"], entry["query"]
    raise SystemExit(f"query id {args.query_id!r} not found in {DATA}")


def _print_hits(label: str, hits: list[SearchHit]) -> None:
    """Print the ranked hits without dumping chunk contents."""
    print(f"\n[{label}] hits={len(hits)}")
    for hit in hits:
        title = (hit.title or "")[:60]
        print(
            f"  {hit.rank:>2}. {hit.doc_id}  {hit.chunk_id}  "
            f"distance={hit.distance:.6f}  score={hit.score:.6f}  {title}"
        )


def _print_scoped_guc(conn: psycopg.Connection, requested: int) -> None:
    """Show ef_search outside, inside and after an explicit transaction."""
    before = current_ef_search(conn)
    with ef_search(conn, requested):
        during = current_ef_search(conn)
    after = current_ef_search(conn)
    print(
        f"\n[GUC] requested={requested} before={before} inside_transaction={during} after={after}"
    )


def _explain(
    conn: psycopg.Connection,
    vector: list[float],
    top_k: int,
    ef_value: int,
    chunk_strategy: str = "document",
    *,
    force: bool,
) -> str:
    """Explain the retrieval SQL, optionally inside a forced-HNSW transaction.

    ``prepare_threshold`` is disabled so psycopg cannot serve a cached generic
    plan: Phase 4A showed such a plan survives ``enable_seqscan=off`` and would
    make this report claim a plan that did not run.
    """
    previous = conn.prepare_threshold
    conn.prepare_threshold = None
    try:
        if force:
            with conn.transaction():
                conn.execute("SET LOCAL enable_seqscan = off", prepare=False)
                with ef_search(conn, ef_value):
                    rows = conn.execute(
                        "EXPLAIN (ANALYZE, BUFFERS) " + DENSE_SEARCH_SQL,
                        (Vector(vector), chunk_strategy, top_k),
                        prepare=False,
                    ).fetchall()
        else:
            with ef_search(conn, ef_value):
                rows = conn.execute(
                    "EXPLAIN (ANALYZE, BUFFERS) " + DENSE_SEARCH_SQL,
                    (Vector(vector), chunk_strategy, top_k),
                    prepare=False,
                ).fetchall()
    finally:
        conn.prepare_threshold = previous
    return "\n".join(row[0] for row in rows)


def _print_plan(label: str, plan: str) -> None:
    """Print the plan lines that matter for the ANN stage.

    ``Sort Key`` lines echo the bound query vector and are dropped: they add
    no information about the plan and would bury it in 1024 floats.
    """
    print(f"\n[{label}]")
    interesting = ("Limit", "Scan", "Join", "Sort", "Gather", "Aggregate")
    for line in plan.splitlines():
        if "Sort Key:" in line:
            continue
        if any(token in line for token in interesting):
            text = line.strip()
            if len(text) > 180:
                text = text[:177] + "..."
            print("  " + text)


def main(argv: list[str] | None = None) -> int:
    """Run the diagnostic and return a process exit code."""
    args = _parse_args(argv)
    query_id, query = _load_query(args)
    config = DenseConfig(
        top_n=args.top_k,
        ef_search=args.ef_search,
        chunk_strategy=args.chunk_strategy,
    )

    print(
        f"query_id={query_id} top_k={args.top_k} ef_search={args.ef_search} "
        f"chunk_strategy={args.chunk_strategy}"
    )
    print(f"query={query!r}")

    embedder = SentenceTransformerEmbedder()
    natural = DenseRetriever(config, embedder=embedder)
    forced = DenseRetriever(config, embedder=embedder, connection_factory=_forced_hnsw)

    with connection() as conn:
        documents = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        print(f"corpus: documents={documents} chunks={chunks}")
        _print_scoped_guc(conn, args.ef_search)

    _print_hits("natural retrieval", natural.retrieve(query, top_k=args.top_k))

    vector = embedder.embed_query(query)
    with connection() as conn:
        _print_plan(
            "natural plan (no planner forcing)",
            _explain(
                conn,
                vector,
                args.top_k,
                args.ef_search,
                args.chunk_strategy,
                force=False,
            ),
        )
        _print_plan(
            "forced plan (SET LOCAL enable_seqscan=off, diagnosis only)",
            _explain(
                conn,
                vector,
                args.top_k,
                args.ef_search,
                args.chunk_strategy,
                force=True,
            ),
        )
        assert current_ef_search(conn) == 40, "ef_search leaked out of its transaction"

    if args.force_hnsw:
        _print_hits(
            "forced-HNSW retrieval (diagnosis only)",
            forced.retrieve(query, top_k=args.top_k),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
