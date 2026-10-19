"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from src.db import connection
from src.ingestion.loader import CorpusReport, inspect_source
from src.schema.document import Document


@pytest.fixture(scope="module")
def conn() -> Iterator[psycopg.Connection]:
    """A pooled connection, or a skip if PostgreSQL is not reachable."""
    ctx = connection()
    try:
        live = ctx.__enter__()
    except (psycopg.Error, ConnectionError) as exc:
        pytest.skip(f"PostgreSQL is not reachable -- run `make up` first: {exc}")
    try:
        yield live
    finally:
        ctx.__exit__(None, None, None)


@pytest.fixture(scope="session")
def corpus_report() -> CorpusReport:
    """The corpus on disk, read and validated once per session."""
    return inspect_source()


@pytest.fixture(scope="session")
def corpus(corpus_report: CorpusReport) -> list[Document]:
    """Every document in ``data/kb_documents.json``, as models."""
    return corpus_report.documents
