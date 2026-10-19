"""Consistent logging for the whole project."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_ROOT = "apex"


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Return a namespaced logger, configuring the handler on first use."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger(_ROOT)
        root.addHandler(handler)
        root.setLevel(level or logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(f"{_ROOT}.{name}")
