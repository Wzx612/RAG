"""Retrieval package."""

from __future__ import annotations


class RetrievalError(RuntimeError):
    """A retrieval request could not be completed.

    Wraps embedding-backend and database failures with a message that names the
    stage. The original exception is always kept as ``__cause__`` so callers can
    inspect the SQLSTATE or backend error rather than parsing this message.
    """
