"""Document loader.

Reads data/kb_documents.json into validated Document models. Sole job:
file -> models. No chunking, no embedding, no database.

Implemented in: Phase 3 (ingestion)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.config import Settings, get_settings
from src.ingestion import IngestionError
from src.schema.document import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)

SOURCE_FILENAME = "kb_documents.json"

# Every content field opens with "Policy Code: BT-001-APEX-2024." -- the loader
# keeps the family token (BT-001) because 150 documents share only 78 of them and
# that grouping is what makes the near-duplicate families analysable.
POLICY_CODE_RE = re.compile(r"Policy Code:\s*([A-Z]+-\d+)")

# The columns `documents` actually has. Anything else in the JSON is reported
# rather than silently dropped, so a corpus change cannot pass unnoticed.
KNOWN_FIELDS = frozenset({"doc_id", "title", "category", "card_tier", "effective_year", "content"})


@dataclass(frozen=True)
class DocumentProblem:
    """One reason a record cannot be imported."""

    doc_id: str
    title: str
    issue: str

    def __str__(self) -> str:
        """Render as ``DOC-0001 ('Title'): issue`` so it reads as one log line."""
        return f"{self.doc_id} ({self.title!r}): {self.issue}"


def source_path(settings: Settings | None = None) -> Path:
    """Return the absolute path of the corpus JSON."""
    settings = settings or get_settings()
    return settings.resolve(settings.data_dir) / SOURCE_FILENAME


def extract_policy_code(content: str) -> str | None:
    """Return the ``BT-001``-style family token, or None when absent."""
    match = POLICY_CODE_RE.search(content)
    return match.group(1) if match else None


@dataclass(frozen=True)
class CorpusReport:
    """Result of reading and validating the corpus, problems included.

    Carrying the problems back instead of raising on the first one is what lets
    the CLI print a ``valid / invalid`` count and still fail the run.
    """

    path: Path
    total: int
    documents: list[Document]
    problems: list[DocumentProblem]

    @property
    def ok(self) -> bool:
        """True when every record became a document."""
        return not self.problems


def inspect_source(path: Path | None = None, settings: Settings | None = None) -> CorpusReport:
    """Read and validate the corpus without deciding what to do about problems.

    Raises:
        IngestionError: the file is missing or unreadable, is not a JSON array,
            or holds the wrong number of records. These are properties of the
            file as a whole, so there is nothing per-record to report.
    """
    settings = settings or get_settings()
    path = path or source_path(settings)
    records = _read_records(path)

    expected = settings.expected_document_count
    if len(records) != expected:
        raise IngestionError(
            f"{path} holds {len(records)} records, expected {expected}. "
            "The corpus is fixed; adjust APEX_EXPECTED_DOCUMENT_COUNT if this is deliberate."
        )

    documents, problems = validate_records(records)
    return CorpusReport(path=path, total=len(records), documents=documents, problems=problems)


def load_documents(path: Path | None = None, settings: Settings | None = None) -> list[Document]:
    """Read, validate and return every document in the knowledge base.

    Nothing is repaired, dropped or padded: any problem at all aborts the load
    and the raised message lists every offending record, so one run reports
    everything wrong with the corpus rather than revealing it one row at a time.

    Args:
        path: Override the corpus location. Defaults to ``data/kb_documents.json``.
        settings: Override the settings singleton (tests).

    Raises:
        IngestionError: as for :func:`inspect_source`, or if any record is
            malformed.
    """
    report = inspect_source(path, settings)
    if report.problems:
        listed = "\n  ".join(str(problem) for problem in report.problems)
        raise IngestionError(
            f"{len(report.problems)} invalid document(s) in {report.path}:\n  {listed}"
        )

    logger.info("loaded %d documents from %s", len(report.documents), report.path)
    return report.documents


def validate_records(records: list[Any]) -> tuple[list[Document], list[DocumentProblem]]:
    """Split raw JSON records into validated documents and the problems found.

    Pure: no I/O, no logging. Exposed for tests and for a dry-run report.

    Returns:
        ``(documents, problems)``. A record is either in one or the other,
        never both.
    """
    documents: list[Document] = []
    problems: list[DocumentProblem] = []
    seen: dict[str, int] = {}
    unknown_fields: set[str] = set()

    for index, record in enumerate(records):
        doc_id, title = _identity(record, index)

        if not isinstance(record, dict):
            problems.append(DocumentProblem(doc_id, title, f"record is {type(record).__name__}"))
            continue

        unknown_fields |= set(record) - KNOWN_FIELDS

        content = record.get("content")
        if content is None:
            problems.append(DocumentProblem(doc_id, title, "content is missing"))
            continue
        if not isinstance(content, str) or not content.strip():
            problems.append(DocumentProblem(doc_id, title, "content is empty"))
            continue

        try:
            document = Document.model_validate(record)
        except ValidationError as exc:
            problems.append(DocumentProblem(doc_id, title, _condense(exc)))
            continue

        if document.doc_id in seen:
            problems.append(
                DocumentProblem(
                    document.doc_id,
                    title,
                    f"duplicate doc_id, already used by record #{seen[document.doc_id]}",
                )
            )
            continue

        seen[document.doc_id] = index
        documents.append(document.model_copy(update={"policy_code": extract_policy_code(content)}))

    if unknown_fields:
        # Not fatal -- `documents` simply has no column for them -- but a silent
        # drop would be the wrong default if the corpus ever grows.
        logger.warning(
            "ignoring JSON field(s) with no `documents` column: %s",
            ", ".join(sorted(unknown_fields)),
        )

    return documents, problems


def _read_records(path: Path) -> list[Any]:
    """Parse the corpus file, refusing anything that is not a JSON array."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IngestionError(f"cannot read knowledge base at {path}: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IngestionError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise IngestionError(f"{path} must hold a JSON array, found {type(payload).__name__}")
    return payload


def _identity(record: Any, index: int) -> tuple[str, str]:
    """Best-effort label for a record, so a problem can be traced back."""
    if not isinstance(record, dict):
        return f"record #{index}", "<not an object>"
    return str(record.get("doc_id", f"record #{index}")), str(record.get("title", "<no title>"))


def _condense(exc: ValidationError) -> str:
    """Flatten a pydantic error into one line naming the offending field."""
    parts = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"]) or "<record>"
        parts.append(f"{field}: {error['msg']}")
    return "; ".join(parts)
