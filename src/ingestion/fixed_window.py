"""Character-based fixed-window chunking.

This is the deliberately structure-blind control arm. Windows are measured in
characters, not tokens: the corpus is English policy text, and character units
avoid a tokenizer dependency and a second model-specific length definition.
The strategy preserves exact source offsets and uses a deterministic tail
policy so reruns produce the same spans.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion import IngestionError
from src.ingestion.spans import ChunkSpan

DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_MIN_TAIL = 48


@dataclass(frozen=True, slots=True)
class FixedWindowParams:
    """Parameters for character-based fixed windows.

    ``chunk_size`` is the maximum normal window length in characters.
    ``chunk_overlap`` is the number of characters shared by adjacent windows.
    ``min_tail`` is the shortest final window accepted without being merged
    into its predecessor. All three are measured in characters.
    """

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    min_tail: int = DEFAULT_MIN_TAIL

    def __post_init__(self) -> None:
        """Reject values that would make the window loop ambiguous."""
        for name, value in (
            ("chunk_size", self.chunk_size),
            ("chunk_overlap", self.chunk_overlap),
            ("min_tail", self.min_tail),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an int, got {type(value).__name__}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.min_tail < 0:
            raise ValueError(f"min_tail must be non-negative, got {self.min_tail}")
        if self.min_tail >= self.chunk_size:
            raise ValueError("min_tail must be smaller than chunk_size")


def chunk_fixed_window(
    text: str,
    params: FixedWindowParams | None = None,
) -> list[ChunkSpan]:
    """Split ``text`` into overlapping character windows.

    The first window starts at offset 0. Every following window starts
    ``chunk_size - chunk_overlap`` characters after its predecessor. A final
    window shorter than ``min_tail`` is shifted left so it ends at the document
    end; it therefore keeps ``chunk_size`` characters but overlaps its
    predecessor by more than ``chunk_overlap``. No chunk exceeds
    ``chunk_size``.

    Args:
        text: Source document text. Must contain at least one non-whitespace
            character.
        params: Window parameters. Defaults to 256/64/48 characters.

    Returns:
        Ordered spans with contiguous chunk indices; adjacent character windows
        overlap by at least ``chunk_overlap``. Boundary whitespace is trimmed
        and offsets remain exact.

    Raises:
        TypeError: ``text`` is not a string.
        IngestionError: ``text`` is empty or whitespace-only.
        ValueError: The parameters are invalid.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    if not text.strip():
        raise IngestionError("cannot chunk empty text")
    config = params or FixedWindowParams()

    spans: list[ChunkSpan] = []
    for start, end in _window_boundaries(len(text), config):
        span = _trimmed_span(text, start, end, len(spans))
        if span is not None:
            spans.append(span)
    if not spans:
        raise IngestionError("fixed-window chunking produced no non-empty spans")
    return spans


def _window_boundaries(length: int, params: FixedWindowParams) -> list[tuple[int, int]]:
    """Return raw ``(start, end)`` character boundaries before trimming."""
    if length <= params.chunk_size:
        return [(0, length)]

    step = params.chunk_size - params.chunk_overlap
    boundaries: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + params.chunk_size, length)
        boundaries.append((start, end))
        if end == length:
            break
        start += step

    if len(boundaries) >= 2:
        last_start, last_end = boundaries[-1]
        if last_end - last_start < params.min_tail:
            boundaries[-1] = (last_end - params.chunk_size, last_end)
    return boundaries


def _trimmed_span(text: str, start: int, end: int, index: int) -> ChunkSpan | None:
    """Build a span with boundary whitespace removed from content and offsets."""
    raw = text[start:end]
    content = raw.strip()
    if not content:
        return None
    left = len(raw) - len(raw.lstrip())
    right = len(raw) - len(raw.rstrip())
    return ChunkSpan(
        chunk_index=index,
        content=content,
        char_start=start + left,
        char_end=end - right,
    )
