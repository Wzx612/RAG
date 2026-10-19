"""Phase 7B reranker benchmark.

The formal comparison keeps the four accepted arms from the Phase 6B artifact
and adds one live arm:

``dense`` / ``sparse_or`` / ``sparse_and`` / ``rrf``
    Reused from ``reports/hybrid_benchmark.json``. Their raw hits are
    independently re-scored with the same ``_measure`` function, and the stored
    metrics plus the RRF formula are checked before they are accepted as a
    baseline. They are marked ``historical``; this module does not re-run the
    embedding model.

``rrf_reranker``
    Live cross-encoder scoring of the exact historical RRF Top-10 candidate
    identities. Candidate text is fetched from PostgreSQL one query at a time,
    so the whole corpus is never held in memory. The candidates are replayed
    through :class:`~src.retrieval.pipeline.RerankedRetriever`, which preserves
    the retriever protocol and the reranker's failure categories.

No silent fallback exists: a model, inference, input or database failure marks
the query unevaluated and fails the run; the pre-rerank list is never reported
as a reranked result. No labels, documents or retrieval semantics are changed.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import psutil  # type: ignore[import-untyped]

from src.config import PROJECT_ROOT, get_settings
from src.db import connection
from src.evaluation.dense_benchmark import (
    BenchmarkError,
    EvalQuery,
    load_eval_queries,
)
from src.evaluation.hybrid_benchmark import _measure, collect_dataset, collect_environment
from src.evaluation.metrics import (
    LABELS_EXHAUSTIVE,
    METRIC_VERSION,
    RELEVANCE_SCHEME,
    percentile,
)
from src.retrieval.pipeline import RerankedRetriever
from src.retrieval.reranker import CrossEncoderReranker, Reranker, RerankError, rerank_text
from src.schema.retrieval import RerankConfig, SearchHit
from src.utils.logger import get_logger

logger = get_logger(__name__)

HISTORICAL_ARMS = ("dense", "sparse_or", "sparse_and", "rrf")
"""Arms reused from the accepted Phase 6B artifact."""

ARM_NAMES = (*HISTORICAL_ARMS, "rrf_reranker")
"""All five arms in report order."""

FULL_QUERY_COUNT = 30
"""The labelled query count required for a complete benchmark."""

DEFAULT_MODEL = "BAAI/bge-reranker-base"
DEFAULT_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
DEFAULT_CACHE_FOLDER = PROJECT_ROOT / ".cache" / "huggingface" / "hub"
DEFAULT_HISTORICAL = PROJECT_ROOT / "reports" / "hybrid_benchmark.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "rerank_benchmark.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "rerank_benchmark.md"

MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer_config.json",
)

ProgressFn = Callable[[str], None]
TextFetcher = Callable[
    [Sequence[tuple[str, str]]],
    Mapping[tuple[str, str], Mapping[str, Any]],
]
ConnectionFactory = Callable[[], AbstractContextManager[Any]]


@dataclass(frozen=True)
class RerankBenchmarkConfig:
    """Fixed parameters for one reranker comparison."""

    top_k: int = 10
    candidate_k: int = 10
    rrf_k: int = 60
    batch_size: int = 16
    max_length: int = 512
    activation: Literal["identity", "sigmoid"] = "identity"
    device: str = "cpu"
    model: str = DEFAULT_MODEL
    revision: str | None = DEFAULT_REVISION
    cache_folder: Path | None = DEFAULT_CACHE_FOLDER
    local_files_only: bool = True
    latency_repetitions: int = 2


def build_config(
    *,
    top_k: int = 10,
    candidate_k: int = 10,
    rrf_k: int = 60,
    batch_size: int = 16,
    max_length: int = 512,
    activation: Literal["identity", "sigmoid"] = "identity",
    device: str = "cpu",
    model: str = DEFAULT_MODEL,
    revision: str | None = DEFAULT_REVISION,
    cache_folder: Path | None = DEFAULT_CACHE_FOLDER,
    local_files_only: bool = True,
    latency_repetitions: int = 2,
) -> RerankBenchmarkConfig:
    """Validate the fixed comparison parameters through ``RerankConfig``.

    The historical RRF artifact stores exactly ten candidates per query, so
    ``candidate_k`` cannot exceed ten. Changing ``rrf_k`` would mean the replayed
    pool was produced by a different formula, so it is rejected rather than
    silently compared.
    """
    if (
        isinstance(candidate_k, bool)
        or not isinstance(candidate_k, int)
        or not 1 <= candidate_k <= 10
    ):
        raise BenchmarkError("candidate_k must be an int between 1 and 10 (historical RRF pool)")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= candidate_k:
        raise BenchmarkError(f"top_k must be an int between 1 and candidate_k, got {top_k!r}")
    if rrf_k != 60:
        raise BenchmarkError("rrf_k must be 60, the value recorded in the Phase 6B artifact")
    if (
        isinstance(latency_repetitions, bool)
        or not isinstance(latency_repetitions, int)
        or latency_repetitions < 1
    ):
        raise BenchmarkError("latency_repetitions must be a positive int")
    try:
        RerankConfig(
            enabled=True,
            model=model,
            revision=revision,
            candidates=candidate_k,
            top_k=top_k,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            activation=activation,
            local_files_only=local_files_only,
            cache_folder=str(cache_folder) if cache_folder is not None else None,
        )
    except ValueError as exc:  # pydantic ValidationError subclasses ValueError
        raise BenchmarkError(f"invalid reranker configuration: {exc}") from exc
    return RerankBenchmarkConfig(
        top_k=top_k,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
        batch_size=batch_size,
        max_length=max_length,
        activation=activation,
        device=device,
        model=model,
        revision=revision,
        cache_folder=cache_folder,
        local_files_only=local_files_only,
        latency_repetitions=latency_repetitions,
    )


def _rerank_config(config: RerankBenchmarkConfig, **overrides: Any) -> RerankConfig:
    """Build the production ``RerankConfig`` for this benchmark."""
    values: dict[str, Any] = {
        "enabled": True,
        "model": config.model,
        "revision": config.revision,
        "candidates": config.candidate_k,
        "top_k": config.top_k,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "device": config.device,
        "activation": config.activation,
        "local_files_only": config.local_files_only,
        "cache_folder": str(config.cache_folder) if config.cache_folder is not None else None,
    }
    values.update(overrides)
    return RerankConfig(**values)


# ------------------------------------------------------------------ history


def load_historical(path: Path | None = None) -> dict[str, Any]:
    """Read the accepted Phase 6B JSON artifact without modifying it."""
    path = path or DEFAULT_HISTORICAL
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkError(f"cannot read historical artifact {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkError(f"{path} must hold a JSON object")
    return payload


def _historical_hit(record: Mapping[str, Any]) -> SearchHit:
    """Rebuild a ``SearchHit`` from a stored Phase 6B hit record."""
    return SearchHit(
        chunk_id=str(record["chunk_id"]),
        doc_id=str(record["doc_id"]),
        score=float(record["score"]),
        distance=record.get("distance"),
        rank=int(record["rank"]),
        source=record["source"],
        dense_rank=record.get("dense_rank"),
        sparse_rank=record.get("sparse_rank"),
        dense_score=record.get("dense_score"),
        sparse_score=record.get("sparse_score"),
        rrf_score=record.get("rrf_score"),
        candidate_sources=tuple(record.get("candidate_sources", ())),
        chunk_strategy=record.get("chunk_strategy"),
        chunk_index=record.get("chunk_index"),
    )


def _same(actual: object, expected: object) -> bool:
    """Compare stored and recomputed values without hiding float drift."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    return actual == expected


def verify_historical(
    artifact: Mapping[str, Any],
    queries: Sequence[EvalQuery],
    config: RerankBenchmarkConfig,
    *,
    check_hashes: bool = True,
) -> dict[str, Any]:
    """Recompute the historical four-arm metrics and check the RRF formula.

    This is an independent check of the stored artifact, not a live retrieval
    run. Any mismatch is recorded and makes the comparison untrustworthy; it is
    never repaired or ignored.
    """
    mismatches: list[str] = []
    parameters = artifact.get("parameters", {})
    records = artifact.get("per_query", [])
    if not isinstance(records, list):
        mismatches.append("historical per_query is not a list")
        records = []
    expected_parameters = {
        "queries": len(records),
        "top_k": config.top_k,
        "candidate_k": config.candidate_k,
        "rrf_k": config.rrf_k,
    }
    for key, expected in expected_parameters.items():
        if not _same(parameters.get(key), expected):
            mismatches.append(
                f"historical parameter {key}={parameters.get(key)!r}, expected {expected!r}"
            )
    if artifact.get("status") not in {"PASS", "PASS WITH MINOR ISSUES"}:
        mismatches.append(f"historical status is {artifact.get('status')!r}")

    by_id = {
        str(record.get("query_id")): record
        for record in records
        if isinstance(record, Mapping)
    }
    missing_ids = {query.query_id for query in queries} - set(by_id)
    if missing_ids:
        mismatches.append(
            f"historical query ids are missing selected queries: {sorted(missing_ids)}"
        )
    if len(queries) > len(records):
        mismatches.append(
            f"selected {len(queries)} queries but the artifact has only {len(records)} records"
        )
    full_selection = len(queries) == len(records)

    if check_hashes:
        settings = get_settings()
        data_dir = settings.resolve(settings.data_dir)
        current_hashes = {
            "kb_documents.json": _sha256(data_dir / "kb_documents.json"),
            "eval_queries.json": _sha256(data_dir / "eval_queries.json"),
        }
        stored_hashes = artifact.get("dataset", {}).get("hashes", {})
        if current_hashes != stored_hashes:
            mismatches.append(
                f"data hashes changed: stored={stored_hashes}, current={current_hashes}"
            )

    recomputed: dict[str, dict[str, dict[str, Any]]] = {}
    stored_aggregates = artifact.get("arms", {}).get("arms", {})
    rrf_formula_checked = 0
    rrf_formula_mismatches = 0
    gold_coverage_queries = 0

    for query in queries:
        record = by_id.get(query.query_id)
        if record is None:
            continue
        recomputed[query.query_id] = {}
        for arm in HISTORICAL_ARMS:
            arm_record = record.get("arms", {}).get(arm)
            if not isinstance(arm_record, Mapping) or not arm_record.get("evaluated"):
                mismatches.append(f"{query.query_id}/{arm} is not evaluated in the artifact")
                continue
            hits = arm_record.get("hits", [])
            if not isinstance(hits, list):
                mismatches.append(f"{query.query_id}/{arm} hits is not a list")
                continue
            measured = _measure(query, [_historical_hit(hit) for hit in hits], config.top_k)
            recomputed[query.query_id][arm] = measured
            for field in (
                "result_count",
                "doc_unique_count",
                "empty_result",
                "doc_ids",
                "gold_ranks",
                "first_gold_rank",
                "reciprocal_rank",
                "hit_at_1",
                "recalled_gold",
                "gold_count",
                "semantic_recall",
            ):
                if not _same(measured.get(field), arm_record.get(field)):
                    mismatches.append(
                        f"{query.query_id}/{arm}.{field}: stored {arm_record.get(field)!r}, "
                        f"recomputed {measured.get(field)!r}"
                    )

        rrf_record = record.get("arms", {}).get("rrf", {})
        rrf_hits = rrf_record.get("hits", []) if isinstance(rrf_record, Mapping) else []
        gold = set(query.relevant_doc_ids)
        rrf_docs = set(rrf_record.get("doc_ids", [])) if isinstance(rrf_record, Mapping) else set()
        gold_coverage_queries += int(gold <= rrf_docs)
        for position, hit in enumerate(rrf_hits, start=1):
            if hit.get("rank") != position:
                mismatches.append(f"{query.query_id}/rrf rank is not 1..N at position {position}")
            dense_rank = hit.get("dense_rank")
            sparse_rank = hit.get("sparse_rank")
            expected_score = 0.0
            if dense_rank is not None:
                expected_score += 1.0 / (config.rrf_k + int(dense_rank))
            if sparse_rank is not None:
                expected_score += 1.0 / (config.rrf_k + int(sparse_rank))
            rrf_formula_checked += 1
            if not _same(hit.get("rrf_score"), expected_score):
                rrf_formula_mismatches += 1
                mismatches.append(
                    f"{query.query_id}/rrf chunk {hit.get('chunk_id')}: rrf_score "
                    f"{hit.get('rrf_score')!r}, formula {expected_score!r}"
                )
            if not _same(hit.get("score"), hit.get("rrf_score")):
                mismatches.append(f"{query.query_id}/rrf score != rrf_score")
        scores = [hit.get("rrf_score") for hit in rrf_hits]
        if any(score is None for score in scores):
            mismatches.append(f"{query.query_id}/rrf has a missing rrf_score")
        elif any(
            float(scores[index]) < float(scores[index + 1]) - 1e-12
            for index in range(len(scores) - 1)
        ):
            mismatches.append(f"{query.query_id}/rrf ordering is not non-increasing")

    recomputed_aggregates: dict[str, Any] = {}
    for arm in HISTORICAL_ARMS:
        arm_records = [
            recomputed[query.query_id][arm]
            for query in queries
            if arm in recomputed.get(query.query_id, {})
        ]
        recomputed_aggregates[arm] = _aggregate_arm(
            arm_records,
            top_k=config.top_k,
            latency=stored_aggregates.get(arm, {}).get("latency_ms"),
            latency_source="historical_⑥-B",
        )
        if not full_selection:
            # A --limit smoke test checks only the selected per-query records;
            # the stored aggregate describes all 30 queries and is not comparable.
            continue
        stored = stored_aggregates.get(arm, {})
        for field in (
            "evaluated_queries",
            "semantic_recall_macro",
            "semantic_recall_micro",
            "mrr",
            "hit_at_1",
            "recalled_gold",
            "gold_total",
        ):
            if field in stored and not _same(
                recomputed_aggregates[arm].get(field), stored.get(field)
            ):
                mismatches.append(
                    f"aggregate {arm}.{field}: stored {stored.get(field)!r}, "
                    f"recomputed {recomputed_aggregates[arm].get(field)!r}"
                )

    return {
        "ok": not mismatches,
        "mismatches": mismatches[:200],
        "mismatch_count": len(mismatches),
        "parameters_expected": expected_parameters,
        "recomputed": recomputed,
        "recomputed_aggregates": recomputed_aggregates,
        "stored_aggregates": {
            arm: stored_aggregates.get(arm, {}) for arm in HISTORICAL_ARMS
        },
        "rrf_formula_checked_hits": rrf_formula_checked,
        "rrf_formula_mismatches": rrf_formula_mismatches,
        "queries_with_all_gold_in_rrf_top_k": gold_coverage_queries,
        "selection_full": full_selection,
        "aggregate_comparison_skipped": not full_selection,
        "artifact_generated_at": artifact.get("generated_at"),
        "artifact_status": artifact.get("status"),
        "source": "historical_⑥-B_recomputed_from_raw_hits_not_a_new_db_run",
    }


# ------------------------------------------------------------------ database


def fetch_chunk_texts(
    identities: Sequence[tuple[str, str]],
    *,
    connection_factory: ConnectionFactory = connection,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Fetch the text and metadata for one query's candidate identities.

    The caller passes at most one candidate pool (normally ten rows), so the
    corpus is never materialised in memory.
    """
    if not identities:
        return {}
    placeholders = ", ".join(["(%s, %s)"] * len(identities))
    parameters = [value for identity in identities for value in identity]
    sql = (
        "SELECT chunk_strategy, chunk_id, doc_id, chunk_index, content, title, "
        "category, card_tier, effective_year, policy_code "
        "FROM chunks WHERE (chunk_strategy, chunk_id) IN (" + placeholders + ")"
    )
    with connection_factory() as conn:
        rows = conn.execute(sql, parameters).fetchall()
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        result[(str(row[0]), str(row[1]))] = {
            "chunk_strategy": str(row[0]),
            "chunk_id": str(row[1]),
            "doc_id": str(row[2]),
            "chunk_index": int(row[3]),
            "content": row[4],
            "title": row[5],
            "category": row[6],
            "card_tier": row[7],
            "effective_year": row[8],
            "policy_code": row[9],
        }
    return result


def _enrich_hit(record: Mapping[str, Any], text: Mapping[str, Any]) -> SearchHit:
    """Combine a historical hit with live PostgreSQL text and metadata."""
    if str(record["doc_id"]) != str(text["doc_id"]):
        raise BenchmarkError(
            f"chunk {record['chunk_id']} maps to doc {text['doc_id']} in the database, "
            f"but the historical hit says {record['doc_id']}"
        )
    if str(record.get("chunk_strategy")) != str(text["chunk_strategy"]):
        raise BenchmarkError(f"chunk {record['chunk_id']} has a different chunk_strategy")
    return SearchHit(
        chunk_id=str(record["chunk_id"]),
        doc_id=str(record["doc_id"]),
        score=float(record["score"]),
        distance=record.get("distance"),
        rank=int(record["rank"]),
        source=record["source"],
        dense_rank=record.get("dense_rank"),
        sparse_rank=record.get("sparse_rank"),
        dense_score=record.get("dense_score"),
        sparse_score=record.get("sparse_score"),
        rrf_score=record.get("rrf_score"),
        candidate_sources=tuple(record.get("candidate_sources", ())),
        content=text.get("content"),
        chunk_strategy=text.get("chunk_strategy"),
        chunk_index=text.get("chunk_index"),
        title=text.get("title"),
        category=text.get("category"),
        card_tier=text.get("card_tier"),
        effective_year=text.get("effective_year"),
        policy_code=text.get("policy_code"),
    )


class _HistoricalRRFBase:
    """A retriever that replays one query's historical RRF candidate pool."""

    name = "historical_rrf_top10"

    def __init__(self, hits: Sequence[SearchHit]) -> None:
        """Store the enriched candidate pool."""
        self._hits = list(hits)

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        """Return the fixed pool, truncated to ``top_k``."""
        return list(self._hits if top_k is None else self._hits[:top_k])


# --------------------------------------------------------------------- helpers


def _sha256(path: Path) -> str:
    """Return a file's SHA256, or ``missing``.

    The model weights are ~1.1 GB, so they are hashed in chunks instead of
    being read into memory in one call.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return "missing"
    return digest.hexdigest()


def _package_version(name: str) -> str:
    """Return an installed distribution version or ``missing``."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def _memory_snapshot() -> dict[str, Any]:
    """Sample current and peak process memory plus system availability."""
    process = psutil.Process()
    info = process.memory_info()
    system = psutil.virtual_memory()
    return {
        "rss_mib": round(info.rss / 2**20, 1),
        "peak_wset_mib": (
            round(info.peak_wset / 2**20, 1) if getattr(info, "peak_wset", None) else None
        ),
        "system_available_gib": round(system.available / 2**30, 3),
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _latency_summary(samples: Sequence[float]) -> dict[str, Any]:
    """Count, min, nearest-rank P50/P95, max and mean."""
    if not samples:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
    return {
        "n": len(samples),
        "min": min(samples),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "max": max(samples),
        "mean": statistics.fmean(samples),
    }


def _aggregate_arm(
    records: Sequence[Mapping[str, Any]],
    *,
    top_k: int | None = None,
    latency: Mapping[str, Any] | None = None,
    latency_source: str = "live",
) -> dict[str, Any]:
    """Aggregate document-level metrics for one arm.

    ``top_k`` is optional so historical records can still be inspected. When
    it is absent, or when any evaluated record predates the supplementary
    metrics, the corresponding aggregate is ``None`` rather than a fabricated
    zero.
    """
    evaluated = [record for record in records if record.get("evaluated")]
    recalls = [float(record["semantic_recall"]) for record in evaluated]
    reciprocal = [float(record["reciprocal_rank"]) for record in evaluated]
    gold_total = sum(int(record["gold_count"]) for record in evaluated)
    recalled_total = sum(int(record["recalled_gold"]) for record in evaluated)
    hit_at_1 = sum(1 for record in evaluated if record["hit_at_1"])
    has_precision = bool(evaluated) and all(
        record.get("precision_at_k") is not None for record in evaluated
    )
    has_ceiling = bool(evaluated) and all(
        record.get("precision_at_k_ceiling") is not None for record in evaluated
    )
    has_ndcg = bool(evaluated) and all(
        record.get("ndcg_at_k") is not None for record in evaluated
    )
    precisions = [float(record["precision_at_k"]) for record in evaluated] if has_precision else []
    ceilings = (
        [float(record["precision_at_k_ceiling"]) for record in evaluated] if has_ceiling else []
    )
    ndcgs = [float(record["ndcg_at_k"]) for record in evaluated] if has_ndcg else []
    return {
        "evaluated_queries": len(evaluated),
        "semantic_recall_macro": statistics.fmean(recalls) if recalls else None,
        "semantic_recall_micro": (recalled_total / gold_total) if gold_total else None,
        "precision_at_k_macro": statistics.fmean(precisions) if precisions else None,
        "precision_at_k_micro": (
            recalled_total / (top_k * len(evaluated))
            if top_k is not None and has_precision and evaluated
            else None
        ),
        "precision_at_k_ceiling_macro": statistics.fmean(ceilings) if ceilings else None,
        "ndcg_at_k_macro": statistics.fmean(ndcgs) if ndcgs else None,
        "recalled_gold": recalled_total,
        "gold_total": gold_total,
        "mrr": statistics.fmean(reciprocal) if reciprocal else None,
        "queries_with_gold_hit": sum(1 for value in reciprocal if value > 0),
        "queries_with_no_gold_hit": sum(1 for value in reciprocal if value == 0),
        "hit_at_1": hit_at_1,
        "hit_at_1_rate": (hit_at_1 / len(evaluated)) if evaluated else None,
        "perfect_recall_queries": sum(1 for value in recalls if value >= 1.0),
        "empty_result_queries": sum(1 for record in evaluated if record["empty_result"]),
        "latency_ms": dict(latency) if latency is not None else _latency_summary([]),
        "latency_source": latency_source,
    }


def _model_files(config: RerankBenchmarkConfig) -> dict[str, Any]:
    """Hash the local snapshot files without trusting a reported checksum."""
    if config.cache_folder is None:
        return {"available": False, "reason": "cache_folder is not configured", "files": {}}
    snapshots = config.cache_folder / "models--BAAI--bge-reranker-base" / "snapshots"
    revisions = (
        [config.revision]
        if config.revision
        else sorted(path.name for path in snapshots.glob("*") if path.is_dir())
    )
    snapshot = (
        snapshots / revisions[0]
        if revisions and (snapshots / revisions[0]).is_dir()
        else None
    )
    if snapshot is None:
        return {
            "available": False,
            "reason": f"snapshot not found under {snapshots}",
            "files": {},
        }
    files: dict[str, Any] = {}
    for name in MODEL_FILES:
        path = snapshot / name
        files[name] = {
            "bytes": path.stat().st_size if path.exists() else None,
            "sha256": _sha256(path),
        }
    return {
        "available": True,
        "snapshot": str(snapshot),
        "revision": snapshot.name,
        "files": files,
    }


def _runtime_facts() -> dict[str, Any]:
    """Record torch/runtime facts without assuming CUDA exists."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 -- diagnostics must not fail the run
        return {"torch": _package_version("torch"), "torch_error": str(exc)}
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "sentence_transformers": _package_version("sentence-transformers"),
        "tokenizers": _package_version("tokenizers"),
        "huggingface_hub": _package_version("huggingface-hub"),
        "safetensors": _package_version("safetensors"),
        "numpy": _package_version("numpy"),
        "psutil": _package_version("psutil"),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _not_evaluated(message: str) -> dict[str, Any]:
    """The shape of a reranked arm record with no successful result."""
    return {
        "evaluated": False,
        "result_count": None,
        "doc_unique_count": None,
        "empty_result": None,
        "doc_ids": [],
        "candidate_doc_ids": [],
        "gold_doc_ids": [],
        "gold_ranks": {},
        "first_gold_rank": None,
        "reciprocal_rank": None,
        "hit_at_1": None,
        "recalled_gold": None,
        "gold_count": None,
        "semantic_recall": None,
        "precision_at_k": None,
        "precision_at_k_ceiling": None,
        "ndcg_at_k": None,
        "hits": [],
        "error": message,
    }


def _hit_record(hit: SearchHit) -> dict[str, Any]:
    """Serialise a reranked hit with provenance and a text fingerprint."""
    text = rerank_text(hit) if hit.content else ""
    return {
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "chunk_strategy": hit.chunk_strategy,
        "chunk_index": hit.chunk_index,
        "score": hit.score,
        "distance": hit.distance,
        "rank": hit.rank,
        "source": hit.source,
        "dense_rank": hit.dense_rank,
        "sparse_rank": hit.sparse_rank,
        "dense_score": hit.dense_score,
        "sparse_score": hit.sparse_score,
        "rrf_score": hit.rrf_score,
        "candidate_sources": list(hit.candidate_sources),
        "rerank_score": hit.rerank_score,
        "pre_rerank_rank": hit.pre_rerank_rank,
        "pre_rerank_score": hit.pre_rerank_score,
        "title": hit.title,
        "content_chars": len(text),
        "pair_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _error_record(
    exc: BaseException,
    *,
    stage: str,
    query_id: str,
) -> dict[str, Any]:
    """Serialise an isolated failure without a traceback."""
    return {
        "stage": stage,
        "query_id": query_id,
        "exception": type(exc).__name__,
        "message": str(exc)[:500],
    }


# ------------------------------------------------------------------ one query


def _rerank_one(
    query: EvalQuery,
    historical_record: Mapping[str, Any],
    config: RerankBenchmarkConfig,
    reranker: Reranker,
    text_fetcher: TextFetcher,
) -> dict[str, Any]:
    """Replay one RRF pool through the reranker and record every outcome."""
    pool_records = list(historical_record["arms"]["rrf"]["hits"])[: config.candidate_k]
    if not pool_records:
        failure = _not_evaluated("historical RRF pool is empty")
        return {
            "query_id": query.query_id,
            "query": query.query,
            "arms": {
                arm: dict(historical_record["arms"][arm]) for arm in HISTORICAL_ARMS
            },
            "rrf_reranker": failure,
            "deltas": {},
            "candidate_pool": [],
            "rerank": {"fallback_used": False, "failure_stage": "input"},
            "error": {
                "stage": "input",
                "query_id": query.query_id,
                "exception": "EmptyCandidatePool",
                "message": "historical RRF pool is empty",
            },
        }

    identities = [
        (str(record["chunk_strategy"]), str(record["chunk_id"])) for record in pool_records
    ]
    try:
        texts = text_fetcher(identities)
    except Exception as exc:  # noqa: BLE001 -- isolate one query's database failure
        return _failed_query(
            query, historical_record, "db_fetch", exc, pool_records
        )

    missing = [identity for identity in identities if identity not in texts]
    if missing:
        failure = _not_evaluated(f"missing text for {missing}")
        return {
            "query_id": query.query_id,
            "query": query.query,
            "arms": {
                arm: dict(historical_record["arms"][arm]) for arm in HISTORICAL_ARMS
            },
            "rrf_reranker": failure,
            "deltas": {},
            "candidate_pool": [],
            "rerank": {},
            "error": {
                "stage": "input",
                "query_id": query.query_id,
                "exception": "MissingCandidateText",
                "message": f"missing text for {missing}",
            },
        }

    try:
        hits = [_enrich_hit(record, texts[(str(record["chunk_strategy"]), str(record["chunk_id"]))])
                for record in pool_records]
        base = _HistoricalRRFBase(hits)
        pipeline = RerankedRetriever(base, reranker)
        result = pipeline.retrieve_with_details(query.query, top_k=config.top_k)
    except RerankError as exc:
        return _failed_query(query, historical_record, exc.stage, exc, pool_records)
    except Exception as exc:  # noqa: BLE001 -- a harness failure must be visible
        return _failed_query(query, historical_record, "unexpected", exc, pool_records)

    measured = _measure(query, result.hits, config.top_k)
    measured["hits"] = [_hit_record(hit) for hit in result.hits]
    measured["evaluated"] = True
    rrf_arm = historical_record["arms"]["rrf"]
    deltas = {
        "rrf_reranker_vs_rrf_reciprocal_rank": (
            float(measured["reciprocal_rank"]) - float(rrf_arm["reciprocal_rank"])
        ),
        "rrf_reranker_vs_rrf_first_gold_rank": (
            None
            if measured["first_gold_rank"] is None or rrf_arm["first_gold_rank"] is None
            else int(measured["first_gold_rank"]) - int(rrf_arm["first_gold_rank"])
        ),
        "rrf_reranker_vs_rrf_hit_at_1": int(bool(measured["hit_at_1"]))
        - int(bool(rrf_arm["hit_at_1"])),
        "recall_preserved": bool(measured["semantic_recall"] == rrf_arm["semantic_recall"]),
        "top1_changed": bool(
            (measured["doc_ids"][:1] or [None]) != (rrf_arm["doc_ids"][:1] or [None])
        ),
    }
    return {
        "query_id": query.query_id,
        "query": query.query,
        "arms": {
            arm: dict(historical_record["arms"][arm]) for arm in HISTORICAL_ARMS
        },
        "rrf_reranker": measured,
        "deltas": deltas,
        "candidate_pool": [_hit_record(hit) for hit in result.base_hits],
        "rerank": {
            "model_id": result.model_id,
            "model_revision": result.model_revision,
            "device": result.device,
            "batch_size": result.batch_size,
            "max_length": result.max_length,
            "activation": result.activation,
            "model_loaded": result.model_loaded,
            "candidate_count": len(result.base_hits),
            "output_count": len(result.hits),
            "base_ms": result.base_ms,
            "rerank_ms": result.rerank_ms,
            "total_ms": result.total_ms,
            "truncated": result.truncated,
            "empty_candidates": result.empty_candidates,
            "fallback_used": result.fallback_used,
        },
        "error": None,
    }


def _failed_query(
    query: EvalQuery,
    historical_record: Mapping[str, Any],
    stage: str,
    exc: BaseException,
    pool_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a failed query record without fabricating a reranked list."""
    return {
        "query_id": query.query_id,
        "query": query.query,
        "arms": {arm: dict(historical_record["arms"][arm]) for arm in HISTORICAL_ARMS},
        "rrf_reranker": _not_evaluated(str(exc)),
        "deltas": {},
        "candidate_pool": [dict(record) for record in pool_records],
        "rerank": {
            "fallback_used": False,
            "failure_stage": stage,
        },
        "error": _error_record(exc, stage=stage, query_id=query.query_id),
    }


# ------------------------------------------------------------------- direction


def run_direction_check(
    reranker: Reranker,
    queries: Sequence[EvalQuery],
    historical: Mapping[str, Any],
    text_fetcher: TextFetcher,
    config: RerankBenchmarkConfig,
    *,
    query_ids: Sequence[str] = ("Q-001", "Q-004", "Q-014"),
) -> dict[str, Any]:
    """Score three pools and record whether gold outranks non-gold."""
    by_id = {query.query_id: query for query in queries}
    historical_by_id = {
        str(record.get("query_id")): record
        for record in historical.get("per_query", [])
        if isinstance(record, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for query_id in query_ids:
        query = by_id.get(query_id)
        record = historical_by_id.get(query_id)
        if query is None or record is None:
            rows.append({"query_id": query_id, "status": "skipped", "reason": "missing query"})
            continue
        pool = list(record["arms"]["rrf"]["hits"])[: config.candidate_k]
        identities = [(str(hit["chunk_strategy"]), str(hit["chunk_id"])) for hit in pool]
        try:
            texts = text_fetcher(identities)
            hits = [
                _enrich_hit(hit, texts[(str(hit["chunk_strategy"]), str(hit["chunk_id"]))])
                for hit in pool
            ]
            scores = reranker.score(query.query, hits)
            repeat = reranker.score(query.query, hits)
            reverse_hits = list(reversed(hits))
            reverse_scores = reranker.score(query.query, reverse_hits)
            reverse_map = {
                hit.chunk_id: score for hit, score in zip(reverse_hits, reverse_scores, strict=True)
            }
            forward_map = {
                hit.chunk_id: score for hit, score in zip(hits, scores, strict=True)
            }
            gold = set(query.relevant_doc_ids)
            gold_scores = [
                score for hit, score in zip(hits, scores, strict=True) if hit.doc_id in gold
            ]
            other_scores = [
                score for hit, score in zip(hits, scores, strict=True) if hit.doc_id not in gold
            ]
            empty = reranker.score(query.query, [])
            invalid_rejected = False
            try:
                reranker.score("", hits)
            except RerankError as exc:
                invalid_rejected = exc.stage == "input"
            long_hit = hits[0].model_copy(update={"content": "token " * 2000})
            long_score = reranker.score(query.query, [long_hit])
            rows.append(
                {
                    "query_id": query_id,
                    "status": "ok",
                    "gold_doc_ids": list(query.relevant_doc_ids),
                    "gold_scores": gold_scores,
                    "non_gold_scores": other_scores,
                    "gold_mean": statistics.fmean(gold_scores) if gold_scores else None,
                    "non_gold_mean": statistics.fmean(other_scores) if other_scores else None,
                    "direction_ok": (
                        statistics.fmean(gold_scores) > statistics.fmean(other_scores)
                        if gold_scores and other_scores
                        else None
                    ),
                    "all_gold_above_all_non_gold": (
                        max(other_scores) < min(gold_scores)
                        if gold_scores and other_scores
                        else None
                    ),
                    "repeat_max_abs_diff": max(
                        (abs(left - right) for left, right in zip(scores, repeat, strict=True)),
                        default=0.0,
                    ),
                    "reverse_mapping_max_abs_diff": max(
                        (
                            abs(forward_map[chunk_id] - reverse_map[chunk_id])
                            for chunk_id in forward_map
                        ),
                        default=0.0,
                    ),
                    "empty_candidates_returns_empty": empty == [],
                    "invalid_query_rejected": invalid_rejected,
                    "overlong_input_score": long_score[0] if long_score else None,
                    "overlong_input_finite": bool(long_score) and math.isfinite(long_score[0]),
                }
            )
        except Exception as exc:  # noqa: BLE001 -- diagnostics are reported, not hidden
            rows.append(
                {
                    "query_id": query_id,
                    "status": "error",
                    "exception": type(exc).__name__,
                    "message": str(exc)[:500],
                }
            )
    return {
        "queries": rows,
        "direction_ok_queries": sum(1 for row in rows if row.get("direction_ok") is True),
        "evaluated_queries": sum(1 for row in rows if row.get("status") == "ok"),
        "note": (
            "Scores are raw cross-encoder logits; only relative order is meaningful. "
            "Gold means the human relevant_doc_ids, not an exhaustive relevance judgement."
        ),
    }


# --------------------------------------------------------------------- latency


def run_latency_probe(
    reranker: CrossEncoderReranker,
    query: EvalQuery,
    config: RerankBenchmarkConfig,
    *,
    repetitions: int | None = None,
) -> dict[str, Any]:
    """Measure 10/50 candidates at batch 1/16 with one loaded model.

    The 50-candidate pool is a live sparse-only retrieval for this query. It is
    a latency fixture, not a quality arm; using sparse avoids loading the
    embedding model alongside the reranker.
    """
    repetitions = repetitions or config.latency_repetitions
    from src.retrieval.sparse import SparseRetriever
    from src.schema.retrieval import SparseConfig

    try:
        sparse = SparseRetriever(SparseConfig(top_n=50))
        pool = list(sparse.retrieve(query.query, top_k=50))
    except Exception as exc:  # noqa: BLE001 -- the probe is diagnostic
        return {
            "status": "skipped",
            "reason": f"sparse latency pool failed: {type(exc).__name__}: {exc}",
            "query_id": query.query_id,
            "combinations": [],
        }
    if len(pool) < 50:
        return {
            "status": "skipped",
            "reason": f"sparse pool has {len(pool)} candidates, need 50",
            "query_id": query.query_id,
            "combinations": [],
        }
    if reranker.backend is None:
        return {
            "status": "skipped",
            "reason": "reranker backend is not loaded",
            "query_id": query.query_id,
            "combinations": [],
        }

    combinations: list[dict[str, Any]] = []
    score_arrays: dict[tuple[int, int], list[float]] = {}
    for candidate_count in (10, 50):
        for batch_size in (1, 16):
            probe = CrossEncoderReranker(
                _rerank_config(
                    config,
                    candidates=candidate_count,
                    top_k=min(config.top_k, candidate_count),
                    batch_size=batch_size,
                ),
                model=reranker.backend,
            )
            candidates = pool[:candidate_count]
            try:
                probe.score(query.query, candidates)
                samples: list[float] = []
                scores: list[float] = []
                for _ in range(repetitions):
                    started = time.perf_counter()
                    scores = probe.score(query.query, candidates)
                    samples.append((time.perf_counter() - started) * 1000.0)
            except Exception as exc:  # noqa: BLE001 -- record the failed combination
                combinations.append(
                    {
                        "candidates": candidate_count,
                        "batch_size": batch_size,
                        "status": "error",
                        "exception": type(exc).__name__,
                        "message": str(exc)[:500],
                    }
                )
                continue
            score_arrays[(candidate_count, batch_size)] = scores
            combinations.append(
                {
                    "candidates": candidate_count,
                    "batch_size": batch_size,
                    "status": "ok",
                    "n": len(samples),
                    "mean_ms": statistics.fmean(samples),
                    "min_ms": min(samples),
                    "max_ms": max(samples),
                    "per_candidate_ms": statistics.fmean(samples) / candidate_count,
                    "score_count": len(scores),
                    "all_finite": all(math.isfinite(value) for value in scores),
                    "scores_first3": scores[:3],
                }
            )

    invariance: dict[str, Any] = {}
    for candidate_count in (10, 50):
        left = score_arrays.get((candidate_count, 1))
        right = score_arrays.get((candidate_count, 16))
        if left is not None and right is not None:
            invariance[str(candidate_count)] = max(
                (abs(a - b) for a, b in zip(left, right, strict=True)), default=0.0
            )
    return {
        "status": "ok" if all(row.get("status") == "ok" for row in combinations) else "partial",
        "query_id": query.query_id,
        "query": query.query,
        "pool_source": "live sparse OR Top-50; latency fixture only, not a quality arm",
        "pool_size": len(pool),
        "gold_doc_ids": list(query.relevant_doc_ids),
        "combinations": combinations,
        "batch_invariance_max_abs_diff": invariance,
        "note": (
            "One warm-up call precedes the timed calls for each combination. "
            "CPU-only, single process, sequential; no concurrency claim."
        ),
    }


# ------------------------------------------------------------------- benchmark


def _load(reranker: Reranker) -> None:
    """Load a reranker through its optional ``load`` method.

    The protocol only promises scoring and reranking; the concrete
    :class:`CrossEncoderReranker` loads lazily. A reranker without ``load`` is
    treated as a model-initialisation failure rather than assumed ready.
    """
    loader = getattr(reranker, "load", None)
    if not callable(loader):
        raise RerankError(
            f"reranker {type(reranker).__name__} has no callable load()",
            stage="model_init",
        )
    loader()


def run_rerank_benchmark(
    config: RerankBenchmarkConfig,
    queries: Sequence[EvalQuery] | None = None,
    *,
    historical: Mapping[str, Any] | None = None,
    reranker: Reranker | None = None,
    environment: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    text_fetcher: TextFetcher | None = None,
    connection_factory: ConnectionFactory = connection,
    progress: ProgressFn | None = None,
    run_probe: bool = True,
    run_direction: bool = True,
    check_hashes: bool = True,
) -> dict[str, Any]:
    """Run the five-arm comparison: four historical arms plus live reranking."""
    if not isinstance(config, RerankBenchmarkConfig):
        raise BenchmarkError("config must be a RerankBenchmarkConfig")
    selected = list(queries) if queries is not None else load_eval_queries()
    if not selected:
        raise BenchmarkError("at least one eval query is required")
    artifact = dict(historical) if historical is not None else load_historical()
    notify = progress or (lambda _message: None)
    started_at = datetime.now(UTC)
    benchmark_started = time.perf_counter()

    verification = verify_historical(
        artifact, selected, config, check_hashes=check_hashes
    )
    environment_facts = dict(environment) if environment is not None else collect_environment()
    dataset_facts = dict(dataset) if dataset is not None else collect_dataset(selected)
    fetcher = text_fetcher or (
        lambda identities: fetch_chunk_texts(
            identities, connection_factory=connection_factory
        )
    )

    active: Reranker = (
        reranker if reranker is not None else CrossEncoderReranker(_rerank_config(config))
    )
    memory_before = _memory_snapshot()
    load_started = time.perf_counter()
    try:
        _load(active)
    except RerankError as exc:
        return _failure_result(
            config,
            selected,
            artifact,
            verification,
            environment_facts,
            dataset_facts,
            exc,
            memory_before,
            started_at,
        )
    measured_load_ms = getattr(active, "load_ms", None)
    load_ms = (
        float(measured_load_ms)
        if measured_load_ms is not None
        else (time.perf_counter() - load_started) * 1000.0
    )
    memory_after_load = _memory_snapshot()

    model_info: dict[str, Any] = {
        "model_id": getattr(active, "model_id", config.model),
        "revision": getattr(active, "revision", config.revision),
        "device": getattr(active, "device", config.device),
        "batch_size": getattr(active, "batch_size", config.batch_size),
        "max_length": getattr(active, "max_length", config.max_length),
        "activation": getattr(active, "activation", config.activation),
        "local_files_only": config.local_files_only,
        "cache_folder": str(config.cache_folder) if config.cache_folder else None,
        "load_ms": load_ms,
        "files": _model_files(config),
        "license": "MIT",
        "license_source": "Hugging Face model card checked during 7-B preflight",
        "commercial_use": "model card states commercial use is free of charge",
        "architecture": "XLM-RoBERTa base, 12 layers, hidden 768, 278,044,931 parameters",
        "max_position_embeddings": 514,
        "tokenizer_max_length": 512,
        "score_semantics": (
            "raw relevance logits with activation=identity; higher is better; "
            "only relative order is meaningful and the scale is not comparable "
            "with cosine similarity or ts_rank_cd"
        ),
        "runtime": _runtime_facts(),
        "memory": {
            "before_load": memory_before,
            "after_load": memory_after_load,
            "peak_during_run": memory_after_load,
        },
    }

    probe: dict[str, Any] | None = None
    if run_probe and selected:
        notify("latency probe: 10/50 candidates, batch 1/16")
        try:
            if isinstance(active, CrossEncoderReranker):
                probe = run_latency_probe(active, selected[0], config)
            else:
                probe = {
                    "status": "skipped",
                    "reason": "injected reranker does not expose a reusable backend",
                    "combinations": [],
                }
        except Exception as exc:  # noqa: BLE001 -- probe failure is recorded, not hidden
            probe = {
                "status": "skipped",
                "reason": f"{type(exc).__name__}: {exc}",
                "combinations": [],
            }

    direction: dict[str, Any] | None = None
    if run_direction:
        notify("score-direction check on Q-001/Q-004/Q-014")
        try:
            direction = run_direction_check(active, selected, artifact, fetcher, config)
        except Exception as exc:  # noqa: BLE001 -- diagnostics are reported
            direction = {
                "queries": [],
                "evaluated_queries": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    # One untimed warm-up on the first query; the timed loop then covers all queries.
    warmup_error: dict[str, Any] | None = None
    if selected:
        first_record = _historical_record(artifact, selected[0].query_id)
        if first_record is not None:
            notify(f"warm-up: {selected[0].query_id}")
            warmup = _rerank_one(selected[0], first_record, config, active, fetcher)
            if warmup.get("error"):
                warmup_error = warmup["error"]

    per_query: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    aborted = False
    loop_started = time.perf_counter()
    for index, query in enumerate(selected, start=1):
        if aborted:
            break
        notify(f"[{index}/{len(selected)}] {query.query_id}")
        record = _historical_record(artifact, query.query_id)
        if record is None:
            errors.append(
                {
                    "stage": "historical_input",
                    "query_id": query.query_id,
                    "exception": "MissingHistoricalRecord",
                    "message": "historical artifact has no record for this query",
                }
            )
            continue
        outcome = _rerank_one(query, record, config, active, fetcher)
        per_query.append(outcome)
        if outcome.get("error"):
            errors.append(outcome["error"])
            if outcome["error"].get("stage") in {"model_init", "resource"}:
                aborted = True
        rerank = outcome.get("rerank", {})
        arm = outcome["rrf_reranker"]
        notify(
            f"  {query.query_id}: rrf_rr={record['arms']['rrf'].get('reciprocal_rank')} "
            f"rerank_rr={arm.get('reciprocal_rank') if arm.get('evaluated') else '-'} "
            f"rerank_ms={rerank.get('rerank_ms')}"
        )
    loop_ms = (time.perf_counter() - loop_started) * 1000.0
    peak_memory = _memory_snapshot()
    memory_after_run = _memory_snapshot()
    model_info["memory"]["peak_during_run"] = {
        "rss_mib": max(
            memory_after_load["rss_mib"], peak_memory["rss_mib"], memory_after_run["rss_mib"]
        ),
        "peak_wset_mib": max(
            (
                value
                for value in (
                    memory_after_load.get("peak_wset_mib"),
                    peak_memory.get("peak_wset_mib"),
                    memory_after_run.get("peak_wset_mib"),
                )
                if value is not None
            ),
            default=None,
        ),
        "system_available_gib_min": min(
            memory_before["system_available_gib"],
            memory_after_load["system_available_gib"],
            peak_memory["system_available_gib"],
            memory_after_run["system_available_gib"],
        ),
    }
    model_info["memory"]["after_run"] = memory_after_run

    reranker_records = [record["rrf_reranker"] for record in per_query]
    rerank_latencies = [
        float(record["rerank"]["total_ms"])
        for record in per_query
        if record.get("rerank", {}).get("total_ms") is not None
    ]
    rerank_only_latencies = [
        float(record["rerank"]["rerank_ms"])
        for record in per_query
        if record.get("rerank", {}).get("rerank_ms") is not None
    ]
    arms = {
        arm: verification["recomputed_aggregates"][arm] for arm in HISTORICAL_ARMS
    }
    arms["rrf_reranker"] = _aggregate_arm(
        reranker_records,
        top_k=config.top_k,
        latency=_latency_summary(rerank_latencies),
        latency_source="live_7B",
    )
    arms["rrf_reranker"]["rerank_only_latency_ms"] = _latency_summary(rerank_only_latencies)
    arms["rrf_reranker"]["source"] = "live_rerank_over_historical_rrf_top10"
    for arm in HISTORICAL_ARMS:
        arms[arm]["source"] = "historical_⑥-B"

    moves = [
        record
        for record in per_query
        if record.get("deltas")
        and record["deltas"].get("rrf_reranker_vs_rrf_reciprocal_rank") != 0
    ]
    delta_rows = [
        record["deltas"]["rrf_reranker_vs_rrf_reciprocal_rank"]
        for record in per_query
        if record.get("deltas")
    ]
    evaluated = [
        record for record in per_query if record["rrf_reranker"].get("evaluated")
    ]
    all_evaluated = len(evaluated) == len(selected) and not errors and not aborted
    partial = len(selected) < FULL_QUERY_COUNT
    if not verification["ok"] or not all_evaluated:
        status = "FAIL"
    elif partial:
        status = "PARTIAL"
    else:
        status = "PASS"
    totals = {
        "queries_requested": len(selected),
        "queries_processed": len(per_query),
        "reranker_evaluated_queries": len(evaluated),
        "historical_arms_reused": list(HISTORICAL_ARMS),
        "errors": len(errors),
        "fallbacks": sum(
            1 for record in per_query if record.get("rerank", {}).get("fallback_used")
        ),
        "model_load_errors": sum(1 for error in errors if error.get("stage") == "model_init"),
        "resource_errors": sum(1 for error in errors if error.get("stage") == "resource"),
        "inference_errors": sum(1 for error in errors if error.get("stage") == "inference"),
        "input_errors": sum(1 for error in errors if error.get("stage") == "input"),
        "db_fetch_errors": sum(1 for error in errors if error.get("stage") == "db_fetch"),
        "aborted": aborted,
    }
    total_ms = (time.perf_counter() - benchmark_started) * 1000.0
    return {
        "schema_version": 1,
        "metric_version": METRIC_VERSION,
        "labels_exhaustive": LABELS_EXHAUSTIVE,
        "relevance_scheme": RELEVANCE_SCHEME,
        "phase": "7B",
        "generated_at": started_at.isoformat(),
        "status": status,
        "partial": partial,
        "parameters": {
            "queries": len(selected),
            "top_k": config.top_k,
            "candidate_k": config.candidate_k,
            "rrf_k": config.rrf_k,
            "batch_size": config.batch_size,
            "max_length": config.max_length,
            "activation": config.activation,
            "device": config.device,
            "model": config.model,
            "revision": config.revision,
            "local_files_only": config.local_files_only,
            "latency_repetitions": config.latency_repetitions,
            "arm_order": list(ARM_NAMES),
        },
        "definitions": _definitions(config),
        "environment": environment_facts,
        "dataset": dataset_facts,
        "model": model_info,
        "historical_verification": verification,
        "probe": probe,
        "direction_check": direction,
        "warmup": {"error": warmup_error},
        "arms": {
            "arms": arms,
            "rrf_reranker_ordering": {
                "improved": sum(1 for value in delta_rows if value > 0),
                "equal": sum(1 for value in delta_rows if value == 0),
                "worse": sum(1 for value in delta_rows if value < 0),
                "mrr_delta_mean": statistics.fmean(delta_rows) if delta_rows else None,
                "first_gold_rank_delta_mean": (
                    statistics.fmean(
                        [
                            record["deltas"]["rrf_reranker_vs_rrf_first_gold_rank"]
                            for record in per_query
                            if record.get("deltas")
                            and record["deltas"].get("rrf_reranker_vs_rrf_first_gold_rank")
                            is not None
                        ]
                    )
                    if any(
                        record.get("deltas")
                        and record["deltas"].get("rrf_reranker_vs_rrf_first_gold_rank")
                        is not None
                        for record in per_query
                    )
                    else None
                ),
                "top1_changed": sum(
                    1 for record in per_query if record.get("deltas", {}).get("top1_changed")
                ),
                "recall_preserved_queries": sum(
                    1
                    for record in per_query
                    if record.get("deltas", {}).get("recall_preserved") is True
                ),
                "moved_queries": [record["query_id"] for record in moves],
            },
        },
        "per_query": per_query,
        "totals": totals,
        "latency": {
            "model_load_ms": load_ms,
            "evaluation_loop_ms": loop_ms,
            "total_wall_ms": total_ms,
            "reranker_total_ms": _latency_summary(rerank_latencies),
            "reranker_inference_ms": _latency_summary(rerank_only_latencies),
            "statistical_basis": (
                f"{len(evaluated)} timed queries, one deterministic rerank pass each; "
                "nearest-rank percentiles; no confidence interval; model load and warm-up "
                "are excluded from per-query latency"
            ),
        },
        "errors": errors,
        "limitations": _limitations(partial),
        "integrity": {
            "historical_artifact": str(DEFAULT_HISTORICAL),
            "historical_verification_ok": verification["ok"],
            "data_hashes": dataset_facts.get("hashes"),
            "missing_gold_docs": dataset_facts.get("missing_gold_docs"),
            "read_only": True,
            "post_run_counts": None,
        },
    }


def _historical_record(
    artifact: Mapping[str, Any], query_id: str
) -> Mapping[str, Any] | None:
    """Find one historical per-query record by id."""
    for record in artifact.get("per_query", []):
        if isinstance(record, Mapping) and str(record.get("query_id")) == query_id:
            return record
    return None


def _failure_result(
    config: RerankBenchmarkConfig,
    queries: Sequence[EvalQuery],
    artifact: Mapping[str, Any],
    verification: Mapping[str, Any],
    environment: Mapping[str, Any],
    dataset: Mapping[str, Any],
    exc: RerankError,
    memory_before: Mapping[str, Any],
    started_at: datetime,
) -> dict[str, Any]:
    """Return a FAIL artifact when the model cannot be loaded."""
    return {
        "schema_version": 1,
        "metric_version": METRIC_VERSION,
        "labels_exhaustive": LABELS_EXHAUSTIVE,
        "relevance_scheme": RELEVANCE_SCHEME,
        "phase": "7B",
        "generated_at": started_at.isoformat(),
        "status": "FAIL",
        "partial": len(queries) < FULL_QUERY_COUNT,
        "parameters": {
            "queries": len(queries),
            "top_k": config.top_k,
            "candidate_k": config.candidate_k,
            "rrf_k": config.rrf_k,
            "batch_size": config.batch_size,
            "max_length": config.max_length,
            "activation": config.activation,
            "device": config.device,
            "model": config.model,
            "revision": config.revision,
            "local_files_only": config.local_files_only,
        },
        "definitions": _definitions(config),
        "environment": dict(environment),
        "dataset": dict(dataset),
        "model": {
            "model_id": config.model,
            "revision": config.revision,
            "device": config.device,
            "files": _model_files(config),
            "memory_before_load": dict(memory_before),
        },
        "historical_verification": dict(verification),
        "failure": {
            "stage": exc.stage,
            "exception": type(exc).__name__,
            "message": str(exc),
        },
        "per_query": [],
        "arms": {"arms": {}, "rrf_reranker_ordering": {}},
        "totals": {
            "queries_requested": len(queries),
            "queries_processed": 0,
            "reranker_evaluated_queries": 0,
            "errors": 1,
            "fallbacks": 0,
            "model_load_errors": int(exc.stage == "model_init"),
            "resource_errors": int(exc.stage == "resource"),
        },
        "errors": [
            {
                "stage": exc.stage,
                "query_id": None,
                "exception": type(exc).__name__,
                "message": str(exc)[:500],
            }
        ],
        "limitations": _limitations(len(queries) < FULL_QUERY_COUNT),
        "integrity": {
            "historical_verification_ok": verification.get("ok"),
            "data_hashes": dataset.get("hashes"),
            "read_only": True,
        },
    }


def _definitions(config: RerankBenchmarkConfig) -> dict[str, str]:
    """Metric, pool and score definitions stored with the raw artifact."""
    return {
        "candidate_granularity": "chunk identity (chunk_strategy, chunk_id)",
        "metric_granularity": "document; distinct doc_id in ranked order, truncated to top_k",
        "semantic_recall": (
            "recall_at_k against human relevant_doc_ids; denominator is the number of "
            "distinct gold docs, not K; not ANN recall"
        ),
        "mrr": (
            "mean reciprocal rank of the first gold doc at doc rank r <= top_k; "
            "no gold in Top-K contributes 0; denominator is evaluated queries"
        ),
        "rrf": (
            f"sum over arms of 1 / (k + rank), rank starts at 1, k={config.rrf_k}; "
            "raw dense/sparse scores are never added"
        ),
        "reranker_pool": (
            "the historical ⑥-B RRF Top-10 candidate identities for the same query; "
            "text is fetched live from PostgreSQL; the dense and sparse arms are not re-run"
        ),
        "reranker_score": (
            "raw cross-encoder logit with activation=identity; higher is better; only "
            "relative order is meaningful and it is not comparable with cosine similarity "
            "or ts_rank_cd"
        ),
        "reranker_order": (
            "rerank_score descending, then pre-rerank rank, then (chunk_strategy, chunk_id)"
        ),
        "final_top_k": f"first {config.top_k} ranked documents/hits",
        "no_gold_handling": "reciprocal rank 0.0 and no fabricated gold document",
        "precision_at_k": (
            "|distinct doc_id in top_k ∩ known gold| / k; the denominator is always k, "
            "even when fewer than k documents are returned; duplicate doc_ids are "
            "collapsed in rank order before the cutoff; supplementary because labels "
            "are not exhaustive"
        ),
        "precision_at_k_ceiling": (
            "min(distinct gold doc count, k) / k per query; the maximum Precision@K "
            "reachable on that query, not an observed ranking score"
        ),
        "ndcg_at_k": (
            "binary nDCG: gain 1 for a known gold doc_id and 0 otherwise; DCG uses "
            "1 / log2(rank + 1); IDCG uses the same k cutoff over the full known gold "
            "set; non-gold ids are unlabelled, not asserted irrelevant"
        ),
        "labels_exhaustive": (
            f"{LABELS_EXHAUSTIVE}; non-gold ids have not been judged irrelevant"
        ),
        "relevance_scheme": (
            f"{RELEVANCE_SCHEME}; only existing gold ids receive relevance 1"
        ),
        "metric_version": f"{METRIC_VERSION}",
    }


def _limitations(partial: bool) -> list[str]:
    """Known limits that this artifact does not establish."""
    items = [
        "The four arms (dense, sparse_or, sparse_and, rrf) are reused from the Phase 6B artifact, "
        "not re-run against PostgreSQL in Phase 7B. Their raw hits are independently "
        "re-scored and the RRF formula is re-checked, but this is offline replay.",
        "The RRF+Reranker arm reranks the historical RRF Top-10 identities; those "
        "candidates are not regenerated by a live dense/sparse query in this run.",
        "Recall cannot increase over RRF because the reranker sees the same Top-10 pool; "
        "only ordering metrics (MRR, Hit@1) can change.",
        "The corpus has one chunk per document, so chunk-level and document-level "
        "candidate sets coincide here.",
        "The 30 queries are English; Chinese and mixed-language reranking are not evaluated.",
        "The embedding model and the reranker were not loaded in the same process; "
        "combined-memory behaviour is not measured, so this run does not claim to solve it.",
        "Host Python execution may differ from the pinned container environment; Docker "
        "was not used for this run and the container path is unverified.",
        "The full pytest suite was not run because the host torch/transformers stack has "
        "a known native crash on an embedding integration test; only module-level tests ran.",
        "Dependency versions on the host differ from the exact pins in pyproject.toml; "
        "that drift is not resolved or verified here.",
        "Latency is CPU-only, single-process, sequential and warm. No GPU, scale, "
        "concurrency or cross-machine reproducibility claim is made.",
        "Precision@K and binary nDCG@K are supplementary and label-limited: non-gold "
        "ids are unlabelled, not confirmed irrelevant.",
        "A database failure was exercised with unit-level connection stubs, not by "
        "stopping the live PostgreSQL server during this benchmark.",
    ]
    if partial:
        items.append(
            "This is a PARTIAL run: fewer than 30 labelled queries were evaluated, so it "
            "is not a complete benchmark."
        )
    return items


# ---------------------------------------------------------------------- output


def write_outputs(
    result: Mapping[str, Any],
    json_path: Path | None = None,
    markdown_path: Path | None = None,
) -> tuple[Path, Path]:
    """Write the raw JSON and the rendered Markdown report."""
    json_path = json_path or DEFAULT_OUTPUT
    markdown_path = markdown_path or DEFAULT_REPORT
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def _md_table(headers: Sequence[object], rows: Sequence[Sequence[object]]) -> str:
    """Render a GitHub-flavoured Markdown table."""

    def cell(value: object) -> str:
        """Escape one cell."""
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _fmt(value: object, digits: int = 4) -> str:
    """Format an optional number without inventing a value for ``None``."""
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(result: Mapping[str, Any]) -> str:
    """Render the 7B artifact as a reviewable Markdown report."""
    parameters = result.get("parameters", {})
    model = result.get("model", {})
    dataset = result.get("dataset", {})
    verification = result.get("historical_verification", {})
    arms = result.get("arms", {}).get("arms", {})
    ordering = result.get("arms", {}).get("rrf_reranker_ordering", {})
    lines = [
        "# Reranker 实现与离线评测（阶段 ⑦-B）",
        "",
        f"- Status: **{result.get('status', 'unknown')}**",
        f"- Generated: {result.get('generated_at', '-')}",
        f"- Metric version / labels_exhaustive / relevance_scheme: "
        f"{result.get('metric_version')} / {result.get('labels_exhaustive')} / "
        f"{result.get('relevance_scheme')}",
        f"- Queries: {parameters.get('queries')}",
        f"- Candidate pool / final Top-K: {parameters.get('candidate_k')} / "
        f"{parameters.get('top_k')}",
        f"- RRF k: {parameters.get('rrf_k')}",
        f"- Model: `{parameters.get('model')}` @ `{parameters.get('revision')}`",
        f"- Batch size / max length / activation: {parameters.get('batch_size')} / "
        f"{parameters.get('max_length')} / {parameters.get('activation')}",
        "",
        "## 1. 结论摘要",
        "",
        _md_table(
            [
                "arm",
                "source",
                "Recall macro",
                "Recall micro",
                "MRR",
                "Hit@1",
                "P50 ms",
                "P95 ms",
            ],
            [
                [
                    arm,
                    record.get("source", "-"),
                    _fmt(record.get("semantic_recall_macro")),
                    _fmt(record.get("semantic_recall_micro")),
                    _fmt(record.get("mrr")),
                    f"{record.get('hit_at_1')}/{record.get('evaluated_queries')}",
                    _fmt(record.get("latency_ms", {}).get("p50")),
                    _fmt(record.get("latency_ms", {}).get("p95")),
                ]
                for arm, record in arms.items()
            ],
        ),
        "",
        f"RRF 的 Recall 为 {_fmt(arms.get('rrf', {}).get('semantic_recall_macro'))}，"
        "候选池完全相同，因此 Recall 不可能因重排而上升；本阶段只判断 MRR 与 Hit@1 "
        "是否改变。若 MRR 没有提升，报告如实记录。",
        "",
        "## 2. 固定条件与评测口径",
        "",
    ]
    for key, value in result.get("definitions", {}).items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(
        [
            "",
            f"- 历史 artifact: `{verification.get('artifact_generated_at', '-')}`，"
            f"status `{verification.get('artifact_status', '-')}`",
            f"- 历史四臂独立复核: **{'PASS' if verification.get('ok') else 'FAIL'}**，"
            f"mismatches={verification.get('mismatch_count')}，"
            f"RRF formula checked={verification.get('rrf_formula_checked_hits')}，"
            f"formula mismatches={verification.get('rrf_formula_mismatches')}",
            f"- RRF Top-K 覆盖全部 gold 的查询数: "
            f"{verification.get('queries_with_all_gold_in_rrf_top_k')}/"
            f"{parameters.get('queries')}",
            f"- 数据哈希: {dataset.get('hashes')}",
            f"- 缺失 gold: {dataset.get('missing_gold_docs')}",
            "",
            "## 3. 模型、来源与资源",
            "",
            _md_table(
                ["item", "value"],
                [
                    ["model id", model.get("model_id")],
                    ["revision", model.get("revision")],
                    ["license", model.get("license")],
                    ["license source", model.get("license_source")],
                    ["commercial use", model.get("commercial_use")],
                    ["architecture", model.get("architecture")],
                    ["device", model.get("device")],
                    ["batch size", model.get("batch_size")],
                    ["max length", model.get("max_length")],
                    ["activation", model.get("activation")],
                    ["local files only", model.get("local_files_only")],
                    ["cache folder", model.get("cache_folder")],
                    ["model load ms", _fmt(model.get("load_ms"), 1)],
                    ["memory before load", model.get("memory", {}).get("before_load")],
                    ["memory after load", model.get("memory", {}).get("after_load")],
                    ["memory peak", model.get("memory", {}).get("peak_during_run")],
                    ["runtime", model.get("runtime")],
                ],
            ),
            "",
            "本地模型文件（实测大小与 SHA256）：",
            "",
            _md_table(
                ["file", "bytes", "sha256"],
                [
                    [name, record.get("bytes"), record.get("sha256")]
                    for name, record in model.get("files", {}).get("files", {}).items()
                ],
            ),
            "",
            "## 4. 最小推理与分数方向",
            "",
        ]
    )
    direction = result.get("direction_check") or {}
    if direction.get("queries"):
        lines.append(
            _md_table(
                [
                    "query",
                    "status",
                    "gold mean",
                    "non-gold mean",
                    "direction ok",
                    "all gold > all non-gold",
                    "repeat max diff",
                    "reverse map diff",
                    "empty -> []",
                    "invalid query rejected",
                    "overlong finite",
                ],
                [
                    [
                        row.get("query_id"),
                        row.get("status"),
                        _fmt(row.get("gold_mean")),
                        _fmt(row.get("non_gold_mean")),
                        row.get("direction_ok"),
                        row.get("all_gold_above_all_non_gold"),
                        row.get("repeat_max_abs_diff"),
                        row.get("reverse_mapping_max_abs_diff"),
                        row.get("empty_candidates_returns_empty"),
                        row.get("invalid_query_rejected"),
                        row.get("overlong_input_finite"),
                    ]
                    for row in direction["queries"]
                ],
            )
        )
        lines.append("")
        lines.append(str(direction.get("note", "")))
    else:
        lines.append(f"- direction check not available: {direction}")
    lines.extend(["", "## 5. 延迟测试（10/50 候选，batch 1/16）", ""])
    probe = result.get("probe") or {}
    if probe.get("combinations"):
        lines.append(
            _md_table(
                [
                    "query",
                    "candidates",
                    "batch",
                    "status",
                    "n",
                    "mean ms",
                    "min ms",
                    "max ms",
                    "ms/candidate",
                    "scores finite",
                ],
                [
                    [
                        probe.get("query_id"),
                        row.get("candidates"),
                        row.get("batch_size"),
                        row.get("status"),
                        row.get("n"),
                        _fmt(row.get("mean_ms"), 1),
                        _fmt(row.get("min_ms"), 1),
                        _fmt(row.get("max_ms"), 1),
                        _fmt(row.get("per_candidate_ms"), 2),
                        row.get("all_finite"),
                    ]
                    for row in probe["combinations"]
                ],
            )
        )
        lines.append("")
        lines.append(f"- pool: {probe.get('pool_source')}")
        lines.append(
            f"- batch invariance max abs diff: {probe.get('batch_invariance_max_abs_diff')}"
        )
        lines.append(f"- {probe.get('note')}")
    else:
        lines.append(f"- skipped: {probe.get('reason', 'not run')}")
    lines.extend(["", "## 6. 五臂对比", ""])
    lines.append(
        _md_table(
            [
                "arm",
                "evaluated",
                "Recall macro",
                "Recall micro",
                "MRR",
                "Hit@1",
                "MRR vs RRF",
                "source",
            ],
            [
                [
                    arm,
                    record.get("evaluated_queries"),
                    _fmt(record.get("semantic_recall_macro")),
                    _fmt(record.get("semantic_recall_micro")),
                    _fmt(record.get("mrr")),
                    f"{record.get('hit_at_1')}/{record.get('evaluated_queries')}",
                    _fmt(
                        None
                        if arm in {"dense", "sparse_or", "sparse_and", "rrf"}
                        else (
                            float(record.get("mrr") or 0.0)
                            - float(arms.get("rrf", {}).get("mrr") or 0.0)
                        )
                    ),
                    record.get("source"),
                ]
                for arm, record in arms.items()
            ],
        )
    )
    lines.extend(
        [
            "",
            f"- RRF vs RRF+Reranker MRR 变化：improved={ordering.get('improved')}，"
            f"equal={ordering.get('equal')}，worse={ordering.get('worse')}，"
            f"mean delta={_fmt(ordering.get('mrr_delta_mean'))}",
            f"- first-gold-rank mean delta: {_fmt(ordering.get('first_gold_rank_delta_mean'))} "
            "（负值表示排名提前）",
            f"- Top-1 改变的查询数: {ordering.get('top1_changed')}",
            f"- Recall 保持不变的查询数: {ordering.get('recall_preserved_queries')}/"
            f"{parameters.get('queries')}",
            f"- 排序发生变化的查询: {', '.join(ordering.get('moved_queries', [])) or '无'}",
            "",
            "### 补充指标：Precision@K / 二元 nDCG@K",
            "",
            f"- labels_exhaustive={result.get('labels_exhaustive')}; "
            f"relevance_scheme={result.get('relevance_scheme')}; "
            f"metric_version={result.get('metric_version')}",
            "- 非 gold 文档只是未标注，不是已确认无关；P@K ceiling 是该 Query 在当前 "
            "gold 数量与 K 下的理论上限。",
            "",
            _md_table(
                [
                    "arm",
                    "P@K macro",
                    "P@K micro",
                    "P@K ceiling macro",
                    "nDCG@K macro",
                ],
                [
                    [
                        arm,
                        _fmt(record.get("precision_at_k_macro")),
                        _fmt(record.get("precision_at_k_micro")),
                        _fmt(record.get("precision_at_k_ceiling_macro")),
                        _fmt(record.get("ndcg_at_k_macro")),
                    ]
                    for arm, record in arms.items()
                ],
            ),
            "",
            "## 7. 逐查询结果",
            "",
            _md_table(
                [
                    "query",
                    "gold",
                    "RRF RR",
                    "rerank RR",
                    "delta RR",
                    "RRF first gold",
                    "rerank first gold",
                    "delta rank",
                    "RRF Hit@1",
                    "rerank Hit@1",
                    "rerank top-1 doc",
                    "fallback",
                ],
                [
                    [
                        record["query_id"],
                        ", ".join(record["arms"]["rrf"].get("gold_doc_ids", [])),
                        _fmt(record["arms"]["rrf"].get("reciprocal_rank")),
                        _fmt(record["rrf_reranker"].get("reciprocal_rank")),
                        _fmt(record["deltas"].get("rrf_reranker_vs_rrf_reciprocal_rank")),
                        _fmt(record["arms"]["rrf"].get("first_gold_rank")),
                        _fmt(record["rrf_reranker"].get("first_gold_rank")),
                        _fmt(record["deltas"].get("rrf_reranker_vs_rrf_first_gold_rank")),
                        record["arms"]["rrf"].get("hit_at_1"),
                        record["rrf_reranker"].get("hit_at_1"),
                        (
                            record["rrf_reranker"].get("doc_ids") or ["-"]
                        )[0],
                        record.get("rerank", {}).get("fallback_used"),
                    ]
                    for record in result.get("per_query", [])
                ],
            ),
            "",
            "## 8. 排序分析",
            "",
            "重排只改变顺序，不改变候选集合。以下按 MRR 变化列出每个查询；未列出的查询"
            "表示 MRR 未变。所有 30 条结果都保留在原始 JSON 中，没有只展示改善的查询。",
            "",
        ]
    )
    moved = [
        record
        for record in result.get("per_query", [])
        if record.get("deltas")
        and record["deltas"].get("rrf_reranker_vs_rrf_reciprocal_rank") != 0
    ]
    if moved:
        lines.append(
            _md_table(
                ["query", "RRF RR", "rerank RR", "delta", "RRF top-1", "rerank top-1"],
                [
                    [
                        record["query_id"],
                        _fmt(record["arms"]["rrf"].get("reciprocal_rank")),
                        _fmt(record["rrf_reranker"].get("reciprocal_rank")),
                        _fmt(record["deltas"].get("rrf_reranker_vs_rrf_reciprocal_rank")),
                        (record["arms"]["rrf"].get("doc_ids") or ["-"])[0],
                        (record["rrf_reranker"].get("doc_ids") or ["-"])[0],
                    ]
                    for record in moved
                ],
            )
        )
    else:
        lines.append("所有查询的 MRR 与 RRF 相同。")
    lines.extend(
        [
            "",
            "## 9. 错误、fallback 与资源限制",
            "",
            f"- errors: {result.get('totals', {}).get('errors')}",
            f"- fallbacks: {result.get('totals', {}).get('fallbacks')} "
            "（实现没有静默 fallback；失败即抛错）",
            f"- model load errors: {result.get('totals', {}).get('model_load_errors')}",
            f"- inference errors: {result.get('totals', {}).get('inference_errors')}",
            f"- input errors: {result.get('totals', {}).get('input_errors')}",
            f"- db fetch errors: {result.get('totals', {}).get('db_fetch_errors')}",
            f"- resource errors: {result.get('totals', {}).get('resource_errors')}",
            f"- aborted: {result.get('totals', {}).get('aborted')}",
            "",
            f"端到端耗时：model load={_fmt(result.get('latency', {}).get('model_load_ms'), 1)} ms，"
            f"evaluation loop={_fmt(result.get('latency', {}).get('evaluation_loop_ms'), 1)} ms，"
            f"total={_fmt(result.get('latency', {}).get('total_wall_ms'), 1)} ms。",
            f"统计口径：{result.get('latency', {}).get('statistical_basis')}",
            "",
            "## 10. 未验证事项",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result.get("limitations", []))
    lines.extend(
        [
            "",
            "## 11. 复现",
            "",
            "Host PowerShell, from the repository root:",
            "",
            "```powershell",
            "$env:APEX_PG_HOST='127.0.0.1'; $env:APEX_PG_PORT='5434';",
            "python scripts/benchmark_rerank.py --top-k 10 --candidate-k 10 --rrf-k 60 "
            "--batch-size 16 --activation identity",
            "```",
            "",
            "Smoke test without writing reports:",
            "",
            "```powershell",
            "python scripts/benchmark_rerank.py --limit 2 --no-latency-probe --no-write",
            "```",
            "",
            "## 12. 数据安全",
            "",
            "本 benchmark 只读：SELECT 查询、内存中的重排、模型推理。没有写入数据库、"
            "没有修改 `data/kb_documents.json`、`data/eval_queries.json` 或 gold 标签，"
            "没有重建索引或更换 Embedding。原始逐查询与候选结果见 "
            f"[`{DEFAULT_OUTPUT.name}`]({DEFAULT_OUTPUT.name})。",
            "",
        ]
    )
    return "\n".join(lines)
