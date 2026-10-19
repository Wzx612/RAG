"""Ask one question through the Phase 8 RAG chain.

The command reuses the accepted retrieval chain and prints the answer, the
evidence labels and the citations. It is a diagnostic entry point, not the
evaluation harness; use ``scripts/benchmark_rag.py`` for the 30-query run.

Examples::

    python scripts/ask_rag.py "What is the foreign transaction fee?" --mock
    python scripts/ask_rag.py "What is the foreign transaction fee?" --json

A real provider requires ``APEX_LLM_API_KEY`` or ``DEEPSEEK_API_KEY``. A mock
run is labelled as such and never presented as real model output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings  # noqa: E402
from src.rag.factory import (  # noqa: E402
    build_historical_retriever,
    build_pipeline,
    build_provider,
)
from src.rag.historical import DEFAULT_HISTORICAL  # noqa: E402
from src.rag.pipeline import RAGPipeline  # noqa: E402
from src.rag.provider import LLMError, LLMProvider, MockLLMProvider  # noqa: E402
from src.schema.rag import RAGAnswer  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the single-question options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", help="Question text")
    parser.add_argument("--mock", action="store_true", help="Use the offline mock provider")
    parser.add_argument(
        "--retrieval-mode",
        choices=("live", "historical"),
        default="live",
        help=(
            "live runs the full retrieval chain; historical replays the accepted "
            "RRF pool and needs no embedding model"
        ),
    )
    parser.add_argument(
        "--historical",
        type=Path,
        default=DEFAULT_HISTORICAL,
        help="Accepted Phase 6B artifact used by historical mode",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Final evidence size")
    parser.add_argument("--candidates", type=int, default=None, help="RRF candidate pool")
    parser.add_argument(
        "--max-context-chars", type=int, default=None, help="Hard context budget"
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face network access for the reranker",
    )
    parser.add_argument("--json", action="store_true", help="Print the full answer as JSON")
    return parser.parse_args(argv)


def _settings(args: argparse.Namespace) -> Settings:
    """Apply CLI overrides without mutating the cached settings singleton."""
    overrides: dict[str, Any] = {}
    if args.top_k is not None:
        overrides["rag_top_k"] = args.top_k
    if args.candidates is not None:
        overrides["rag_candidates"] = args.candidates
    if args.max_context_chars is not None:
        overrides["rag_max_context_chars"] = args.max_context_chars
    return Settings(**overrides)


def _provider(settings: Settings, *, mock: bool) -> LLMProvider:
    """Construct the selected provider and fail fast when a real key is absent."""
    if mock:
        return MockLLMProvider()
    provider = build_provider(settings)
    try:
        provider.ensure_configured()
    except LLMError as exc:
        raise SystemExit(
            f"ERROR: {exc}. No answer was generated. "
            "Pass --mock for an offline pipeline check."
        ) from exc
    return provider


def _print_answer(answer: RAGAnswer) -> None:
    """Print the answer, citations and evidence in separate sections."""
    print(f"Status: {answer.status}")
    print(f"Provider: {answer.provider} / {answer.model}")
    print("")
    print("Answer:")
    print(answer.answer or "(no answer text)")
    print("")
    print("Citations:")
    if answer.citations:
        for citation in answer.citations:
            print(
                f"  {citation.label} -> {citation.chunk_id} "
                f"({citation.doc_id}) {citation.title or ''}".rstrip()
            )
    else:
        print("  none")
    if answer.invalid_citations:
        print("Invalid citation labels (not trusted): " + ", ".join(answer.invalid_citations))
    print("")
    print(f"Evidence supplied: {len(answer.evidence)} blocks")
    for evidence in answer.evidence:
        print(
            f"  {evidence.label} rank={evidence.retrieval_rank} "
            f"chunk={evidence.chunk_id} doc={evidence.doc_id} "
            f"chars={evidence.included_chars}/{evidence.original_chars}"
            + (" truncated" if evidence.truncated else "")
        )
    print("")
    print(
        "Timings ms: "
        + ", ".join(f"{key}={value:.1f}" for key, value in answer.timings_ms.items())
    )
    print(f"Context truncated: {answer.context_truncated}")
    print(f"Context sha256: {answer.context_sha256}")
    print(f"Prompt sha256: {answer.prompt_sha256}")
    if answer.usage:
        print(f"Usage: {answer.usage.model_dump(exclude_none=True)}")
    if answer.error:
        print(
            f"Error: stage={answer.error.stage} category={answer.error.category} "
            f"retryable={answer.error.retryable} message={answer.error.message}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run one question and return a process exit code."""
    args = _parse_args(list(argv) if argv is not None else None)
    try:
        settings = _settings(args)
        provider = _provider(settings, mock=args.mock)
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
        answer = pipeline.answer(args.query)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        _print_answer(answer)
    return 0 if answer.status in {"ok", "insufficient_evidence"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
