"""Import the knowledge base: ``python scripts/ingest.py``.

The Phase 3 brief names this path, so it exists -- but all it does is hand over
to the same CLI the Makefile drives, which is where the pipeline actually lives
(``src/cli.py``). Keeping the two in sync is a problem this file deliberately
does not create.

Run ``python -m src.cli ingest --help`` for the options.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running a file puts *its own* directory on sys.path, not the repository root,
# and the image has a second copy of the package installed under site-packages.
# Prepending the root makes `src` resolve to the live tree the caller is editing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import app  # noqa: E402 -- the path above must be set first

if __name__ == "__main__":
    # Route through the Typer app rather than calling `ingest()` directly: the
    # parameter defaults are Typer objects, and only Typer resolves them.
    sys.argv = [sys.argv[0], "ingest", *sys.argv[1:]]
    app()
