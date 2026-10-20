"""Download the embedding and reranker models into the Hugging Face cache.

The container runs with ``HF_HOME=/app/.cache/huggingface`` mounted as the
``hf_cache`` volume, so warming once makes later runs independent of the
network. This script only populates that cache: it writes no reports, touches
no database and changes no data.

The reranker is loaded with network access here even though the production
configuration defaults to ``reranker_local_files_only=true``; the whole point of
the warm step is to fetch the weights once. Verify the offline path afterwards
with ``HF_HUB_OFFLINE=1`` and the normal benchmark command.

Usage::

    python scripts/warm_cache.py
    python scripts/warm_cache.py --only embedding
    python scripts/warm_cache.py --only reranker --check-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402
from src.ingestion.embedder import SentenceTransformerEmbedder  # noqa: E402
from src.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from src.schema.retrieval import RerankConfig, SearchHit  # noqa: E402

PROBE_QUERY = "What is the foreign transaction fee?"
PROBE_PASSAGE = "A 3% foreign transaction fee applies to purchases made outside the United States."


def _warm_embedding(check_only: bool) -> int:
    """Load the embedding model and encode one query."""
    settings = get_settings()
    started = time.perf_counter()
    embedder = SentenceTransformerEmbedder(settings)
    vector = embedder.embed_query(PROBE_QUERY)
    elapsed = (time.perf_counter() - started) * 1000.0
    if len(vector) != settings.embedding_dim:
        print(
            f"[FAIL] embedding returned {len(vector)} dims, expected {settings.embedding_dim}",
            file=sys.stderr,
        )
        return 1
    mode = "cache check" if check_only else "warm"
    print(
        f"[PASS] embedding {settings.embedding_model} ({mode}) "
        f"{elapsed:.0f} ms, {len(vector)} dims"
    )
    return 0


def _warm_reranker(check_only: bool) -> int:
    """Load the cross-encoder and score one pair."""
    settings = get_settings()
    config = RerankConfig(
        enabled=True,
        model=settings.reranker_model,
        revision=settings.reranker_revision,
        batch_size=settings.reranker_batch_size,
        max_length=settings.reranker_max_length,
        device=settings.reranker_device,
        activation="identity",
        # A warm step must be allowed to download; --check-only honours the
        # configured offline flag so a cache miss fails instead of silently
        # mutating the volume.
        local_files_only=check_only and settings.reranker_local_files_only,
        cache_folder=str(settings.resolve(settings.reranker_cache_folder)),
    )
    reranker = CrossEncoderReranker(config)
    candidate = SearchHit(
        chunk_id="warm-cache",
        doc_id="warm-cache",
        score=0.0,
        rank=1,
        source="dense",
        content=PROBE_PASSAGE,
    )
    started = time.perf_counter()
    scores = reranker.score(PROBE_QUERY, [candidate])
    elapsed = (time.perf_counter() - started) * 1000.0
    mode = "cache check" if check_only else "warm"
    print(
        f"[PASS] reranker {settings.reranker_model}@{settings.reranker_revision[:8]} "
        f"({mode}) {elapsed:.0f} ms, score {scores[0]:.4f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Warm or verify the model cache; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("embedding", "reranker", "both"),
        default="both",
        help="Which model to load (default: both).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify the existing cache instead of downloading missing files.",
    )
    args = parser.parse_args(argv)

    failures = 0
    if args.only in ("embedding", "both"):
        failures += _warm_embedding(args.check_only)
    if args.only in ("reranker", "both"):
        failures += _warm_reranker(args.check_only)

    if failures:
        print(f"[FAIL] {failures} model check(s) failed.", file=sys.stderr)
        return 1
    print("[PASS] model cache is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
