"""ingestion package.

The write path: JSON -> Document -> Chunk -> vector -> PostgreSQL.

``IngestionError`` lives here rather than in one of the four modules so that
``loader``, ``embedder`` and ``indexer`` can all raise the same type without
importing each other.
"""

from __future__ import annotations


class IngestionError(RuntimeError):
    """The knowledge base could not be loaded, embedded or stored.

    Raised for conditions the pipeline must not paper over: a malformed corpus,
    a vector of the wrong width, a non-finite value, or a write that failed.
    The message always names the offending record or batch.
    """
