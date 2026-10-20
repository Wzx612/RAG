"""Exact text spans shared by the chunking strategies.

A span records what a strategy selected from the source document, independent
of the database and of the ``Chunk`` model. Phase 2 maps spans to ``Chunk``
objects and stable chunk ids; keeping the span type separate lets the pure
chunkers be tested without PostgreSQL, embeddings or a model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    """One exact, non-empty slice of a document.

    ``content`` must equal the source text between ``char_start`` and
    ``char_end``. Boundary whitespace is trimmed from both the content and the
    offsets, so a citation can always be resolved back to the original
    document. ``sentence_start`` and ``sentence_end`` are only populated by the
    structured strategy, where they identify the body sentence range.
    """

    chunk_index: int
    content: str
    char_start: int
    char_end: int
    sentence_start: int | None = None
    sentence_end: int | None = None
    oversize_split: bool = False
    section_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the span before it can reach a strategy result."""
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if not self.content:
            raise ValueError("chunk content must not be empty")
        if self.char_start < 0:
            raise ValueError("char_start must be non-negative")
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        if self.char_end - self.char_start != len(self.content):
            raise ValueError("char offsets must match the content length")
        if self.sentence_start is not None and self.sentence_start < 0:
            raise ValueError("sentence_start must be non-negative")
        if self.sentence_end is not None and self.sentence_start is None:
            raise ValueError("sentence_end requires sentence_start")
        if (
            self.sentence_start is not None
            and self.sentence_end is not None
            and self.sentence_end < self.sentence_start
        ):
            raise ValueError("sentence_end must not precede sentence_start")
