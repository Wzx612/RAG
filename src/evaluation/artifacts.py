"""Artifact path guards shared by every benchmark entry point.

A benchmark report is evidence: re-running a benchmark silently over the same
path would replace the numbers a previous report was reviewed against, with no
record that it happened. Every writer therefore refuses to replace an existing
file unless the caller explicitly passes ``overwrite=True`` (CLI:
``--overwrite``).

Pure functions only -- no I/O beyond ``Path.exists``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


class ArtifactExistsError(RuntimeError):
    """Raised when a write would replace an existing artifact by default."""


def existing_artifacts(paths: Iterable[Path]) -> list[Path]:
    """Return the subset of ``paths`` that already exist, in input order."""
    return [path for path in paths if Path(path).exists()]


def ensure_writable(paths: Iterable[Path], *, overwrite: bool = False) -> None:
    """Refuse to clobber existing artifacts unless ``overwrite`` is set.

    Args:
        paths: Target JSON/Markdown/HTML artifact paths.
        overwrite: Explicit permission to replace existing files.

    Raises:
        ArtifactExistsError: One or more paths exist and ``overwrite`` is false.
    """
    if overwrite:
        return
    clashes = existing_artifacts(paths)
    if clashes:
        listed = ", ".join(str(path) for path in clashes)
        raise ArtifactExistsError(
            f"refusing to overwrite existing artifact(s): {listed}; "
            "pass --overwrite or write to a new path"
        )
