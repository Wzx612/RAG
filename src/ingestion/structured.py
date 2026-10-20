"""Structure-aware sentence chunking.

This strategy is intentionally named ``structured``, not ``semantic``. It uses
the document's fixed header, sentence boundaries and simple clause heuristics;
it does not call an embedding model, a parser or an LLM. The corpus has one
paragraph per document, so sentence and clause boundaries are the real
structure available.

The output is deterministic: the same text and parameters produce the same
spans, offsets and indices.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from src.ingestion import IngestionError
from src.ingestion.spans import ChunkSpan

DEFAULT_TARGET_CHARS = 220
DEFAULT_MAX_CHARS = 360
DEFAULT_MIN_CHARS = 60

DETAILS_MARKER = "Details:"
SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$\"“(])")
ABBREVIATIONS = ("U.S.", "U.K.", "e.g.", "i.e.", "Inc.", "No.", "vs.", "approx.")
CONDITIONAL_PREFIXES = ("If ", "When ", "Unless ", "Where ")
BINDING_PHRASES = ("subject to", "provided that")


@dataclass(frozen=True, slots=True)
class StructuredParams:
    """Parameters for sentence-aware structured chunking.

    ``target_chars`` is the preferred packed length. ``max_chars`` is the hard
    upper bound for a normal chunk; only an individual sentence that cannot be
    split safely may approach it. ``min_chars`` is the shortest tail accepted
    without being merged into its predecessor. All values are characters.
    """

    target_chars: int = DEFAULT_TARGET_CHARS
    max_chars: int = DEFAULT_MAX_CHARS
    min_chars: int = DEFAULT_MIN_CHARS

    def __post_init__(self) -> None:
        """Reject values that would make packing ambiguous."""
        for name, value in (
            ("target_chars", self.target_chars),
            ("max_chars", self.max_chars),
            ("min_chars", self.min_chars),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an int, got {type(value).__name__}")
        if self.min_chars < 0:
            raise ValueError(f"min_chars must be non-negative, got {self.min_chars}")
        if self.target_chars <= 0:
            raise ValueError(f"target_chars must be positive, got {self.target_chars}")
        if self.max_chars <= 0:
            raise ValueError(f"max_chars must be positive, got {self.max_chars}")
        if self.target_chars > self.max_chars:
            raise ValueError("target_chars must not exceed max_chars")
        if self.min_chars > self.target_chars:
            raise ValueError("min_chars must not exceed target_chars")


def chunk_structured(
    text: str,
    params: StructuredParams | None = None,
) -> list[ChunkSpan]:
    """Split ``text`` into sentence-aware chunks.

    Text before ``Details:`` is header metadata and is not emitted as chunk
    content. If the marker is absent, or the body after it is empty, the whole
    document is treated as body with an empty section path. Complete sentences
    are packed up to ``target_chars``; a group is split only when it exceeds
    ``max_chars``. A sentence that is itself over the limit is split at a
    semicolon, comma or space, and those pieces are marked ``oversize_split``.

    A sentence that introduces a condition is kept with the following sentence
    when the pair fits within ``max_chars``; otherwise the conditional sentence
    is moved to the next group so it is not left alone. This is a deterministic
    heuristic, not semantic understanding.

    Args:
        text: Source document text. Must contain at least one non-whitespace
            character.
        params: Packing parameters. Defaults to 220/360/60 characters.

    Returns:
        Ordered spans with exact source offsets and sentence indices.

    Raises:
        TypeError: ``text`` is not a string.
        IngestionError: ``text`` is empty or whitespace-only.
        ValueError: The parameters are invalid.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    if not text.strip():
        raise IngestionError("cannot chunk empty text")
    config = params or StructuredParams()

    body_start, section_path = _details_body_start(text)
    if not text[body_start:].strip():
        body_start = _first_non_space(text)
        section_path = ()

    sentences = _sentence_spans(text, body_start)
    if not sentences:
        span = _make_span(
            text,
            body_start,
            len(text),
            0,
            0,
            0,
            oversize_split=False,
            section_path=section_path,
        )
        if span is None:
            raise IngestionError("structured chunking produced no non-empty spans")
        return [span]

    groups = _group_sentences(sentences, config.target_chars)
    groups = _bind_conditions(text, sentences, groups, config.max_chars)
    spans = _group_spans(text, sentences, groups, config, section_path)
    spans = _merge_short_tail(text, spans, config)
    return [replace(span, chunk_index=index) for index, span in enumerate(spans)]


def _details_body_start(text: str) -> tuple[int, tuple[str, ...]]:
    """Return the body offset and section path implied by ``Details:``."""
    marker = text.find(DETAILS_MARKER)
    if marker < 0:
        return 0, ()
    start = marker + len(DETAILS_MARKER)
    while start < len(text) and text[start].isspace():
        start += 1
    return start, ("Details",)


def _first_non_space(text: str) -> int:
    """Return the offset of the first non-whitespace character, or zero."""
    for index, character in enumerate(text):
        if not character.isspace():
            return index
    return 0


def _sentence_spans(text: str, start: int) -> list[tuple[int, int]]:
    """Return trimmed absolute ``(start, end)`` ranges for body sentences."""
    sentences: list[tuple[int, int]] = []
    cursor = start
    for match in SENTENCE_BREAK_RE.finditer(text, start):
        boundary = match.start()
        if boundary <= cursor:
            continue
        if _is_abbreviation_boundary(text, start, boundary):
            continue
        trimmed = _trim_range(text, cursor, boundary)
        if trimmed is not None:
            sentences.append(trimmed)
        cursor = match.end()
    trimmed = _trim_range(text, cursor, len(text))
    if trimmed is not None:
        sentences.append(trimmed)
    return sentences


def _is_abbreviation_boundary(text: str, start: int, boundary: int) -> bool:
    """Return whether the whitespace follows a known abbreviation."""
    prefix = text[max(start, boundary - 12) : boundary]
    return any(prefix.endswith(abbreviation) for abbreviation in ABBREVIATIONS)


def _group_sentences(
    sentences: list[tuple[int, int]],
    target_chars: int,
) -> list[list[int]]:
    """Greedily pack sentence indices up to ``target_chars`` characters."""
    groups: list[list[int]] = []
    current: list[int] = []
    group_start = 0
    for index, (start, end) in enumerate(sentences):
        if not current:
            current = [index]
            group_start = start
        elif end - group_start <= target_chars:
            current.append(index)
        else:
            groups.append(current)
            current = [index]
            group_start = start
    if current:
        groups.append(current)
    return groups


def _bind_conditions(
    text: str,
    sentences: list[tuple[int, int]],
    groups: list[list[int]],
    max_chars: int,
) -> list[list[int]]:
    """Keep a conditional sentence with the sentence that qualifies it."""
    index = 0
    while index < len(groups) - 1:
        last = groups[index][-1]
        if not _needs_following(text, sentences[last]):
            index += 1
            continue

        if len(groups[index]) > 1:
            moved = groups[index].pop()
            groups[index + 1].insert(0, moved)
        else:
            following = groups[index + 1][0]
            if sentences[following][1] - sentences[last][0] <= max_chars:
                groups[index].append(following)
                groups[index + 1].pop(0)
                if not groups[index + 1]:
                    groups.pop(index + 1)
        index += 1

    return [group for group in groups if group]


def _needs_following(text: str, sentence: tuple[int, int]) -> bool:
    """Return whether a sentence begins or contains a binding condition."""
    start, end = sentence
    content = text[start:end].strip()
    if content.startswith(CONDITIONAL_PREFIXES):
        return True
    lowered = content.lower()
    return any(phrase in lowered for phrase in BINDING_PHRASES)


def _group_spans(
    text: str,
    sentences: list[tuple[int, int]],
    groups: list[list[int]],
    params: StructuredParams,
    section_path: tuple[str, ...],
) -> list[ChunkSpan]:
    """Turn sentence groups into spans, splitting only over-size groups."""
    spans: list[ChunkSpan] = []
    for group in groups:
        start = sentences[group[0]][0]
        end = sentences[group[-1]][1]
        if end - start <= params.max_chars:
            _append_group(text, sentences, group, spans, params.max_chars, section_path)
            continue

        subgroup = [group[0]]
        for sentence_index in group[1:]:
            if sentences[sentence_index][1] - sentences[subgroup[0]][0] <= params.max_chars:
                subgroup.append(sentence_index)
            else:
                _append_group(
                    text,
                    sentences,
                    subgroup,
                    spans,
                    params.max_chars,
                    section_path,
                )
                subgroup = [sentence_index]
        _append_group(text, sentences, subgroup, spans, params.max_chars, section_path)
    return spans


def _append_group(
    text: str,
    sentences: list[tuple[int, int]],
    group: list[int],
    spans: list[ChunkSpan],
    max_chars: int,
    section_path: tuple[str, ...],
) -> None:
    """Append one group, splitting a sentence that is itself over the limit."""
    start = sentences[group[0]][0]
    end = sentences[group[-1]][1]
    if end - start <= max_chars:
        span = _make_span(
            text,
            start,
            end,
            len(spans),
            group[0],
            group[-1],
            oversize_split=False,
            section_path=section_path,
        )
        if span is not None:
            spans.append(span)
        return

    if len(group) == 1:
        for piece_start, piece_end in _split_oversize(text, start, end, max_chars):
            span = _make_span(
                text,
                piece_start,
                piece_end,
                len(spans),
                group[0],
                group[-1],
                oversize_split=True,
                section_path=section_path,
            )
            if span is not None:
                spans.append(span)
        return

    for sentence_index in group:
        _append_group(
            text,
            sentences,
            [sentence_index],
            spans,
            max_chars,
            section_path,
        )


def _split_oversize(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Split an over-size sentence at a semicolon, comma or space."""
    pieces: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        limit = min(cursor + max_chars, end)
        if limit >= end:
            trimmed = _trim_range(text, cursor, end)
            if trimmed is not None:
                pieces.append(trimmed)
            break

        cut = -1
        semicolon = text.rfind(";", cursor + 1, limit)
        if semicolon >= 0:
            cut = semicolon + 1
        else:
            comma = text.rfind(",", cursor + 1, limit)
            if comma >= 0:
                cut = comma + 1
            else:
                space = text.rfind(" ", cursor + 1, limit)
                if space >= 0:
                    cut = space
        if cut <= cursor:
            cut = limit

        trimmed = _trim_range(text, cursor, cut)
        if trimmed is not None:
            pieces.append(trimmed)
        cursor = cut
    return pieces


def _merge_short_tail(
    text: str,
    spans: list[ChunkSpan],
    params: StructuredParams,
) -> list[ChunkSpan]:
    """Merge a tail shorter than ``min_chars`` when it still fits ``max_chars``."""
    if len(spans) < 2:
        return spans
    last = spans[-1]
    previous = spans[-2]
    if len(last.content) >= params.min_chars:
        return spans
    if last.oversize_split or previous.oversize_split:
        return spans
    if last.section_path != previous.section_path:
        return spans
    if last.char_end - previous.char_start > params.max_chars:
        return spans

    trimmed = _trim_range(text, previous.char_start, last.char_end)
    if trimmed is None:
        return spans
    start, end = trimmed
    merged = ChunkSpan(
        chunk_index=previous.chunk_index,
        content=text[start:end],
        char_start=start,
        char_end=end,
        sentence_start=previous.sentence_start,
        sentence_end=last.sentence_end,
        oversize_split=False,
        section_path=previous.section_path,
    )
    return [*spans[:-2], merged]


def _make_span(
    text: str,
    start: int,
    end: int,
    index: int,
    sentence_start: int,
    sentence_end: int,
    *,
    oversize_split: bool,
    section_path: tuple[str, ...],
) -> ChunkSpan | None:
    """Build a trimmed span, returning ``None`` for a whitespace-only range."""
    trimmed = _trim_range(text, start, end)
    if trimmed is None:
        return None
    trimmed_start, trimmed_end = trimmed
    return ChunkSpan(
        chunk_index=index,
        content=text[trimmed_start:trimmed_end],
        char_start=trimmed_start,
        char_end=trimmed_end,
        sentence_start=sentence_start,
        sentence_end=sentence_end,
        oversize_split=oversize_split,
        section_path=section_path,
    )


def _trim_range(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Trim whitespace from a range and return adjusted offsets."""
    raw = text[start:end]
    content = raw.strip()
    if not content:
        return None
    left = len(raw) - len(raw.lstrip())
    right = len(raw) - len(raw.rstrip())
    return start + left, end - right
