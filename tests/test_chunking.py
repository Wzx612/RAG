"""Phase 1: pure chunking strategies.

These tests exercise ``fixed_window`` and ``structured`` without PostgreSQL,
embeddings or a model. They assert exact source offsets, deterministic output,
sentence integrity and the documented tail and over-size behaviours.
"""

from __future__ import annotations

import pytest
from src.ingestion import IngestionError
from src.ingestion.fixed_window import FixedWindowParams, chunk_fixed_window
from src.ingestion.structured import StructuredParams, chunk_structured

# --------------------------------------------------------------- fixed window


def test_fixed_window_short_text_is_one_span() -> None:
    """A document shorter than the window is returned unchanged as one span."""
    text = "Short policy."
    spans = chunk_fixed_window(text)
    assert len(spans) == 1
    assert spans[0].chunk_index == 0
    assert spans[0].content == text
    assert (spans[0].char_start, spans[0].char_end) == (0, len(text))


def test_fixed_window_exact_size_is_one_span() -> None:
    """A document exactly as long as the window is not split."""
    text = "a" * 256
    spans = chunk_fixed_window(text)
    assert len(spans) == 1
    assert spans[0].content == text


def test_fixed_window_splits_with_exact_overlap() -> None:
    """Adjacent windows share exactly the requested number of characters."""
    text = "0123456789" * 10
    params = FixedWindowParams(chunk_size=30, chunk_overlap=10, min_tail=5)
    spans = chunk_fixed_window(text, params)

    assert [(span.char_start, span.char_end) for span in spans] == [
        (0, 30),
        (20, 50),
        (40, 70),
        (60, 90),
        (80, 100),
    ]
    for span in spans:
        assert text[span.char_start : span.char_end] == span.content
    assert [span.chunk_index for span in spans] == [0, 1, 2, 3, 4]


def test_fixed_window_shifts_a_short_tail_left() -> None:
    """A final window shorter than ``min_tail`` is shifted to full width."""
    text = "x" * 34
    params = FixedWindowParams(chunk_size=30, chunk_overlap=10, min_tail=15)
    spans = chunk_fixed_window(text, params)

    assert [(span.char_start, span.char_end) for span in spans] == [
        (0, 30),
        (4, 34),
    ]
    assert all(span.char_end - span.char_start == 30 for span in spans)


def test_fixed_window_keeps_a_tail_at_the_minimum_length() -> None:
    """A tail exactly as long as ``min_tail`` is not merged."""
    text = "x" * 34
    params = FixedWindowParams(chunk_size=30, chunk_overlap=10, min_tail=14)
    spans = chunk_fixed_window(text, params)

    assert len(spans) == 2
    assert spans[-1].char_end - spans[-1].char_start == 14


def test_fixed_window_trims_boundary_whitespace_but_keeps_offsets_exact() -> None:
    """Trimmed content still resolves to the original document slice."""
    text = "  abcdefghij  "
    params = FixedWindowParams(chunk_size=8, chunk_overlap=2, min_tail=1)
    spans = chunk_fixed_window(text, params)

    assert len(spans) == 2
    for span in spans:
        assert text[span.char_start : span.char_end] == span.content
    assert spans[0].content == "abcdef"
    assert spans[1].content == "efghij"


def test_fixed_window_rejects_empty_text() -> None:
    """An empty or whitespace-only document has no valid chunk."""
    for text in ("", "   ", "\n\t"):
        with pytest.raises(IngestionError, match="empty text"):
            chunk_fixed_window(text)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_size": 0},
        {"chunk_size": True},
        {"chunk_size": 1.5},
        {"chunk_overlap": -1},
        {"chunk_overlap": 256},
        {"min_tail": -1},
        {"min_tail": 256},
    ],
)
def test_fixed_window_rejects_invalid_params(kwargs: dict[str, object]) -> None:
    """Window parameters must describe an unambiguous forward scan."""
    with pytest.raises(ValueError):
        FixedWindowParams(**kwargs)  # type: ignore[arg-type]


def test_fixed_window_is_deterministic() -> None:
    """Repeated calls with the same inputs return the same spans."""
    text = "abcdefghij" * 40
    params = FixedWindowParams(chunk_size=64, chunk_overlap=16, min_tail=20)
    first = chunk_fixed_window(text, params)
    second = chunk_fixed_window(text, params)
    assert first == second


# ----------------------------------------------------------------- structured


def test_structured_uses_details_body_and_keeps_section_path() -> None:
    """Header metadata is not chunk content; body chunks carry its section."""
    header = (
        "Policy Code: BT-001-APEX-2024. Applicable Product: Apex Platinum. "
        "Policy Effective Year: 2024. Category: Balance Transfers. "
    )
    body = "First sentence. Second sentence. Third sentence."
    text = header + "Details: " + body
    params = StructuredParams(target_chars=40, max_chars=80, min_chars=10)

    spans = chunk_structured(text, params)

    assert len(spans) == 2
    assert all(span.section_path == ("Details",) for span in spans)
    assert spans[0].char_start == text.index("First")
    assert all("Policy Code:" not in span.content for span in spans)
    assert all("Applicable Product:" not in span.content for span in spans)
    for span in spans:
        assert text[span.char_start : span.char_end] == span.content


def test_structured_does_not_split_a_sentence() -> None:
    """Small target lengths still yield whole sentences."""
    text = (
        "Policy Code: X. Details: First sentence. Second sentence. Third sentence."
    )
    params = StructuredParams(target_chars=10, max_chars=200, min_chars=0)

    spans = chunk_structured(text, params)

    assert [span.content for span in spans] == [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]
    assert all(not span.oversize_split for span in spans)


def test_structured_does_not_split_a_decimal_point() -> None:
    """A decimal number is not a sentence boundary."""
    text = (
        "Policy Code: X. Details: The APR is 18.24% variable. "
        "A second sentence."
    )
    params = StructuredParams(target_chars=30, max_chars=100, min_chars=0)

    spans = chunk_structured(text, params)

    assert "18.24%" in spans[0].content
    assert all("18.24%" in span.content or "second" in span.content for span in spans)


def test_structured_keeps_condition_with_its_following_sentence() -> None:
    """A conditional opener is not emitted without the qualified sentence."""
    text = (
        "Policy Code: X. Details: "
        "If the card is lost, call support immediately. "
        "A replacement is sent within 48 hours. Unrelated sentence."
    )
    params = StructuredParams(target_chars=40, max_chars=120, min_chars=0)

    spans = chunk_structured(text, params)

    condition_chunks = [
        span
        for span in spans
        if "If the card is lost" in span.content
    ]
    assert len(condition_chunks) == 1
    assert "A replacement is sent" in condition_chunks[0].content


def test_structured_moves_a_trailing_condition_to_the_next_group() -> None:
    """A condition at the end of a packed group moves with its consequence."""
    text = (
        "Policy Code: X. Details: This section explains the card terms. "
        "If the card is lost, call support immediately. "
        "A replacement is sent within 48 hours. Unrelated sentence."
    )
    params = StructuredParams(target_chars=90, max_chars=120, min_chars=0)

    spans = chunk_structured(text, params)

    condition_chunks = [
        span
        for span in spans
        if "If the card is lost" in span.content
    ]
    assert len(condition_chunks) == 1
    assert "A replacement is sent" in condition_chunks[0].content


def test_structured_splits_an_oversize_sentence_at_a_delimiter() -> None:
    """A single over-size sentence is split and explicitly marked."""
    text = (
        "Policy Code: X. Details: "
        + "A" * 30
        + "; "
        + "B" * 30
        + "; "
        + "C" * 30
        + "."
    )
    params = StructuredParams(target_chars=50, max_chars=50, min_chars=0)

    spans = chunk_structured(text, params)

    assert len(spans) == 3
    assert all(span.oversize_split for span in spans)
    assert all(len(span.content) <= 50 for span in spans)
    assert all(span.sentence_start == span.sentence_end == 0 for span in spans)
    for span in spans:
        assert text[span.char_start : span.char_end] == span.content


def test_structured_falls_back_to_the_whole_document_without_details() -> None:
    """A document without ``Details:`` is chunked with an empty section path."""
    text = "A policy sentence. Another sentence."
    params = StructuredParams(target_chars=20, max_chars=100, min_chars=0)

    spans = chunk_structured(text, params)

    assert spans[0].char_start == 0
    assert spans[-1].char_end == len(text)
    assert all(span.section_path == () for span in spans)
    for span in spans:
        assert text[span.char_start : span.char_end] == span.content


def test_structured_short_body_is_one_span() -> None:
    """A single short body sentence produces one chunk."""
    text = "Policy Code: X. Details: A short sentence."
    spans = chunk_structured(text)

    assert len(spans) == 1
    assert spans[0].content == "A short sentence."
    assert spans[0].section_path == ("Details",)
    assert spans[0].sentence_start == 0
    assert spans[0].sentence_end == 0
    assert not spans[0].oversize_split


def test_structured_merges_a_short_tail() -> None:
    """A tail below ``min_chars`` is merged when the pair fits ``max_chars``."""
    text = "Policy Code: X. Details: " + "A" * 80 + ". " + "B" * 10 + "."
    params = StructuredParams(target_chars=50, max_chars=150, min_chars=50)

    spans = chunk_structured(text, params)

    assert len(spans) == 1
    assert "A" * 80 in spans[0].content
    assert "B" * 10 in spans[0].content


def test_structured_offsets_are_exact_and_ordered() -> None:
    """Every span resolves to the source and spans do not overlap."""
    text = (
        "Policy Code: BT-001-APEX-2024. Applicable Product: Apex Platinum. "
        "Policy Effective Year: 2024. Category: Balance Transfers. Details: "
        "First sentence about fees. Second sentence about eligibility. "
        "Third sentence about reporting."
    )
    params = StructuredParams(target_chars=30, max_chars=100, min_chars=0)
    spans = chunk_structured(text, params)

    assert len(spans) == 3
    previous_end = 0
    for index, span in enumerate(spans):
        assert span.chunk_index == index
        assert text[span.char_start : span.char_end] == span.content
        assert span.char_start >= previous_end
        previous_end = span.char_end


def test_structured_rejects_empty_text() -> None:
    """An empty or whitespace-only document has no valid chunk."""
    for text in ("", "   ", "\n\t"):
        with pytest.raises(IngestionError, match="empty text"):
            chunk_structured(text)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_chars": 0},
        {"target_chars": True},
        {"target_chars": 1.5},
        {"max_chars": 0},
        {"min_chars": -1},
        {"target_chars": 100, "max_chars": 50},
        {"min_chars": 100, "target_chars": 50, "max_chars": 200},
    ],
)
def test_structured_rejects_invalid_params(kwargs: dict[str, object]) -> None:
    """Packing parameters must respect positive and nested bounds."""
    with pytest.raises(ValueError):
        StructuredParams(**kwargs)  # type: ignore[arg-type]


def test_structured_is_deterministic() -> None:
    """Repeated calls with the same inputs return the same spans."""
    text = (
        "Policy Code: X. Details: First sentence. Second sentence. "
        "Third sentence. Fourth sentence."
    )
    params = StructuredParams(target_chars=30, max_chars=100, min_chars=0)
    first = chunk_structured(text, params)
    second = chunk_structured(text, params)
    assert first == second
