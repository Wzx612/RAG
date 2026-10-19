"""Evidence context assembly for the RAG prompt.

The retriever returns ranked :class:`~src.schema.retrieval.SearchHit` objects.
This module turns them into a bounded, stable, citable text block:

* duplicate ``(chunk_strategy, chunk_id)`` identities are kept once;
* every included chunk gets a sequential label ``E1``, ``E2``, ... in rank order;
* the exact database ``content`` is used -- nothing is synthesised;
* a per-chunk cap and a hard total character budget are applied explicitly;
* the block records whether truncation occurred and how many candidates were
  dropped for lack of budget.

Missing content is an input error, never an empty or invented passage.

Implemented in: Phase 8 (RAG end-to-end).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from src.schema.rag import Evidence
from src.schema.retrieval import SearchHit

MIN_CONTEXT_CHARS = 200
"""Smallest meaningful total context budget accepted by :func:`build_context`."""


class ContextError(RuntimeError):
    """Evidence could not be assembled into a trustworthy context."""


@dataclass(frozen=True)
class ContextBundle:
    """The exact evidence blocks placed in the prompt and their provenance."""

    evidence: list[Evidence]
    text: str
    truncated: bool
    dropped: int
    sha256: str

    @property
    def labels(self) -> list[str]:
        """Citation labels that exist in this bundle, in prompt order."""
        return [item.label for item in self.evidence]


def build_context(
    hits: Sequence[SearchHit],
    *,
    max_context_chars: int = 8000,
    max_chunk_chars: int = 2000,
) -> ContextBundle:
    """Build the bounded evidence block from ranked retrieval hits.

    Args:
        hits: Ranked hits, best first. They may be dense, sparse, fused or
            reranked; only their order, identity, metadata and content matter.
        max_context_chars: Hard cap on the total formatted text, headers
            included. At least one chunk is always attempted; if even its header
            cannot fit, :class:`ContextError` is raised.
        max_chunk_chars: Per-chunk body cap applied before the total budget.

    Returns:
        A :class:`ContextBundle` whose labels run ``E1..En`` in the order the
        evidence appears in the prompt.

    Raises:
        ContextError: invalid arguments, a non-``SearchHit`` item, a duplicate
            identity with conflicting metadata, missing/blank content, or a
            budget too small to fit one evidence header.
    """
    if isinstance(max_context_chars, bool) or not isinstance(max_context_chars, int):
        raise ContextError("max_context_chars must be an int")
    if max_context_chars < MIN_CONTEXT_CHARS:
        raise ContextError(
            f"max_context_chars must be at least {MIN_CONTEXT_CHARS}, got {max_context_chars}"
        )
    if isinstance(max_chunk_chars, bool) or not isinstance(max_chunk_chars, int):
        raise ContextError("max_chunk_chars must be an int")
    if max_chunk_chars < 1:
        raise ContextError(f"max_chunk_chars must be positive, got {max_chunk_chars}")

    unique: list[SearchHit] = []
    seen: set[tuple[str, str]] = set()
    for position, hit in enumerate(hits, start=1):
        if not isinstance(hit, SearchHit):
            raise ContextError(f"candidate item {position} is not a SearchHit")
        identity = (hit.chunk_strategy or "", hit.chunk_id)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(hit)

    evidence: list[Evidence] = []
    blocks: list[str] = []
    used = 0
    truncated_any = False

    for hit in unique:
        content = hit.content
        if not isinstance(content, str) or not content.strip():
            raise ContextError(
                f"candidate {hit.chunk_id!r} has no content; refusing to build an "
                "evidence block from missing database text"
            )

        label = f"E{len(evidence) + 1}"
        original_chars = len(content)
        body = content
        chunk_truncated = False
        if len(body) > max_chunk_chars:
            body = body[:max_chunk_chars]
            chunk_truncated = True

        header = _header(label, hit)
        separator = 2 if blocks else 0
        remaining = max_context_chars - used - separator

        if len(header) + 2 > remaining:
            if evidence:
                break
            raise ContextError(
                "max_context_chars is too small to fit even one evidence header; "
                f"header needs {len(header) + 2}, budget is {max_context_chars}"
            )

        if len(header) + 1 + len(body) > remaining:
            if evidence:
                break
            available = remaining - len(header) - 1
            if available < 1:
                raise ContextError(
                    "max_context_chars leaves no room for evidence text after the header"
                )
            body = body[:available].rstrip()
            chunk_truncated = True

        if not body:
            # A non-blank content string can still become empty after a very
            # small per-chunk/budget truncation; do not emit a header-only block.
            if evidence:
                break
            raise ContextError("max_context_chars trimmed the first evidence block to empty text")

        block = f"{header}\n{body}"
        blocks.append(block)
        used += separator + len(block)
        truncated_any = truncated_any or chunk_truncated
        evidence.append(
            Evidence(
                label=label,
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                chunk_strategy=hit.chunk_strategy,
                chunk_index=hit.chunk_index,
                retrieval_rank=hit.rank,
                score=hit.score,
                rerank_score=hit.rerank_score,
                title=hit.title,
                category=hit.category,
                card_tier=hit.card_tier,
                effective_year=hit.effective_year,
                policy_code=hit.policy_code,
                content=content,
                context_block=block,
                original_chars=original_chars,
                included_chars=len(body),
                truncated=chunk_truncated,
            )
        )

    text = "\n\n".join(blocks)
    return ContextBundle(
        evidence=evidence,
        text=text,
        truncated=truncated_any or len(evidence) < len(unique),
        dropped=len(unique) - len(evidence),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _header(label: str, hit: SearchHit) -> str:
    """Return a single-line header with stable retrieval metadata."""
    fields = [
        f"chunk_id={_clean(hit.chunk_id)}",
        f"doc_id={_clean(hit.doc_id)}",
    ]
    optional = (
        ("title", hit.title),
        ("category", hit.category),
        ("card_tier", hit.card_tier),
        ("effective_year", hit.effective_year),
        ("policy_code", hit.policy_code),
    )
    for name, value in optional:
        if value is not None and str(value).strip():
            fields.append(f"{name}={_clean(str(value))}")
    return f"[{label}] " + " | ".join(fields)


def _clean(value: str) -> str:
    """Keep headers on one line without changing the underlying metadata."""
    return " ".join(value.replace("\r", " ").replace("\n", " ").split())
