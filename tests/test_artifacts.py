"""Artifact overwrite guards.

These tests are pure path checks against ``tmp_path``; they never touch the
historical reports under ``reports/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from src.evaluation.artifacts import (
    ArtifactExistsError,
    ensure_writable,
    existing_artifacts,
)


def test_missing_paths_are_writable(tmp_path: Path) -> None:
    """A fresh path needs no flag."""
    ensure_writable((tmp_path / "a.json", tmp_path / "a.md"))


def test_existing_path_is_refused_by_default(tmp_path: Path) -> None:
    """A default write must not replace a reviewed artifact."""
    target = tmp_path / "report.json"
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactExistsError, match="refusing to overwrite"):
        ensure_writable((target,), overwrite=False)


def test_overwrite_allows_replacement(tmp_path: Path) -> None:
    """``--overwrite`` is the explicit escape hatch."""
    target = tmp_path / "report.json"
    target.write_text("{}", encoding="utf-8")
    ensure_writable((target,), overwrite=True)


def test_one_clash_names_every_existing_target(tmp_path: Path) -> None:
    """The error lists exactly the files that already exist."""
    json_path = tmp_path / "bench.json"
    report_path = tmp_path / "bench.md"
    json_path.write_text("{}", encoding="utf-8")
    report_path.write_text("# report", encoding="utf-8")
    with pytest.raises(ArtifactExistsError) as excinfo:
        ensure_writable((json_path, report_path, tmp_path / "new.md"))
    message = str(excinfo.value)
    assert str(json_path) in message
    assert str(report_path) in message
    assert "new.md" not in message


def test_existing_artifacts_preserves_input_order(tmp_path: Path) -> None:
    """The listing is deterministic and keeps caller order."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    second.write_text("{}", encoding="utf-8")
    first.write_text("{}", encoding="utf-8")
    assert existing_artifacts([first, tmp_path / "missing", second]) == [first, second]


def test_ensure_writable_accepts_an_empty_iterable() -> None:
    """No targets means nothing to guard."""
    ensure_writable(())
