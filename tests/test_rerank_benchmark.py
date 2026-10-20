"""Phase 7B benchmark tests.

They use a two-query fake of the accepted ⑥-B artifact and a deterministic fake
cross-encoder, so no PostgreSQL, Hugging Face download or real model is needed.
The tests check the comparison harness, not the model's quality.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

import numpy as np
import pytest
from src.evaluation.dense_benchmark import BenchmarkError, EvalQuery
from src.evaluation.rerank_benchmark import (
    HISTORICAL_ARMS,
    _aggregate_arm,
    _historical_hit,
    _hit_record,
    build_config,
    fetch_chunk_texts,
    render_markdown,
    run_rerank_benchmark,
    verify_historical,
    write_outputs,
)
from src.retrieval.reranker import CrossEncoderReranker, RerankError
from src.schema.retrieval import RerankConfig, SearchHit

QUERIES = [
    EvalQuery(query_id="Q-001", query="first query", relevant_doc_ids=("D1",)),
    EvalQuery(query_id="Q-002", query="second query", relevant_doc_ids=("D2",)),
]
TOP_K = 3
CANDIDATE_K = 3
RRF_K = 60


def _hit(
    *,
    chunk_id: str,
    doc_id: str,
    rank: int,
    arm: str,
    rrf_score: float | None = None,
) -> SearchHit:
    """Build one historical hit with the provenance the checks expect."""
    if arm == "rrf":
        return SearchHit(
            chunk_id=chunk_id,
            doc_id=doc_id,
            score=rrf_score if rrf_score is not None else 0.0,
            rank=rank,
            source="fused",
            dense_rank=rank,
            sparse_rank=rank,
            dense_score=1.0 / rank,
            sparse_score=10.0 / rank,
            rrf_score=rrf_score,
            candidate_sources=("dense", "sparse"),
            chunk_strategy="document",
            chunk_index=rank - 1,
        )
    if arm == "dense":
        return SearchHit(
            chunk_id=chunk_id,
            doc_id=doc_id,
            score=1.0 / rank,
            distance=1.0 - 1.0 / rank,
            rank=rank,
            source="dense",
            candidate_sources=("dense",),
            chunk_strategy="document",
            chunk_index=rank - 1,
        )
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        score=10.0 / rank,
        rank=rank,
        source="sparse",
        candidate_sources=("sparse",),
        chunk_strategy="document",
        chunk_index=rank - 1,
    )


def _arm_hits(pool: Sequence[tuple[str, str]], arm: str) -> list[SearchHit]:
    """Build an arm's hits in the given pool order."""
    hits: list[SearchHit] = []
    for position, (chunk_id, doc_id) in enumerate(pool, start=1):
        rrf_score = None
        if arm == "rrf":
            rrf_score = 1.0 / (RRF_K + position) + 1.0 / (RRF_K + position)
        hits.append(
            _hit(
                chunk_id=chunk_id,
                doc_id=doc_id,
                rank=position,
                arm=arm,
                rrf_score=rrf_score,
            )
        )
    return hits


def _frozen_arm_record(
    query: EvalQuery, pool: Sequence[tuple[str, str]], arm: str
) -> dict[str, Any]:
    """Build a stored arm record exactly as the accepted artifact would hold it."""
    hits = _arm_hits(pool, arm)
    from src.evaluation.hybrid_benchmark import _measure

    return _measure(query, hits, TOP_K)


def _artifact() -> dict[str, Any]:
    """A two-query fake artifact with valid metrics and RRF provenance."""
    pools = {
        "Q-001": [("C2", "D2"), ("C1", "D1"), ("C3", "D3")],
        "Q-002": [("C1", "D1"), ("C2", "D2"), ("C3", "D3")],
    }
    per_query: list[dict[str, Any]] = []
    for query in QUERIES:
        arms = {
            arm: _frozen_arm_record(query, pools[query.query_id], arm)
            for arm in HISTORICAL_ARMS
        }
        per_query.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "arms": arms,
                "deltas": {},
                "errors": 0,
            }
        )
    stored_aggregates = {
        arm: _aggregate_arm(
            [record["arms"][arm] for record in per_query],
            top_k=TOP_K,
            latency={"n": 2, "min": 1.0, "p50": 1.0, "p95": 1.0, "max": 1.0, "mean": 1.0},
            latency_source="historical_⑥-B",
        )
        for arm in HISTORICAL_ARMS
    }
    return {
        "schema_version": 1,
        "phase": "6B",
        "generated_at": "2026-10-17T00:00:00+00:00",
        "status": "PASS",
        "parameters": {
            "queries": len(QUERIES),
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
        },
        "per_query": per_query,
        "arms": {"arms": stored_aggregates},
        "dataset": {"hashes": {"kb_documents.json": "fake", "eval_queries.json": "fake"}},
    }


def _texts() -> dict[tuple[str, str], dict[str, Any]]:
    """Candidate text and metadata as PostgreSQL would return it."""
    content = {"C1": "alpha", "C2": "beta", "C3": "gamma"}
    docs = {"C1": "D1", "C2": "D2", "C3": "D3"}
    return {
        ("document", chunk_id): {
            "chunk_strategy": "document",
            "chunk_id": chunk_id,
            "doc_id": docs[chunk_id],
            "chunk_index": index,
            "content": content[chunk_id],
            "title": f"Title {chunk_id}",
            "category": "rewards",
            "card_tier": "gold",
            "effective_year": 2026,
            "policy_code": "P-1",
        }
        for index, chunk_id in enumerate(("C1", "C2", "C3"))
    }


def _fetcher(
    mapping: Mapping[tuple[str, str], Mapping[str, Any]],
) -> Any:
    """Return a text fetcher backed by ``mapping``."""

    def fetch(identities: Sequence[tuple[str, str]]) -> Mapping[tuple[str, str], Mapping[str, Any]]:
        return {identity: mapping[identity] for identity in identities if identity in mapping}

    return fetch


class _Backend:
    """Deterministic cross-encoder-shaped backend."""

    def __init__(
        self,
        scores: Mapping[str, float] | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.scores = dict(scores or {"alpha": 3.0, "beta": 1.0, "gamma": 2.0})
        self.error = error
        self.calls = 0
        self.batch_sizes: list[int] = []

    def predict(self, pairs: Sequence[Sequence[str]], **kwargs: Any) -> np.ndarray:
        """Return one scripted score per passage in input order."""
        self.calls += 1
        self.batch_sizes.append(int(kwargs.get("batch_size", 0)))
        if self.error is not None:
            raise self.error
        return np.asarray(
            [self.scores[pair[1].split("\n")[-1]] for pair in pairs],
            dtype=float,
        )


def _reranker(
    backend: _Backend,
    *,
    cache_folder: str | None = None,
) -> CrossEncoderReranker:
    """Wrap the fake backend in the production reranker."""
    config = RerankConfig(
        enabled=True,
        model="fake-reranker",
        candidates=CANDIDATE_K,
        top_k=TOP_K,
        batch_size=2,
        max_length=128,
        local_files_only=True,
        cache_folder=cache_folder,
    )
    return CrossEncoderReranker(config, model=backend)


def _benchmark_config() -> Any:
    """The fixed small configuration used by every test."""
    return build_config(
        top_k=TOP_K,
        candidate_k=CANDIDATE_K,
        rrf_k=RRF_K,
        batch_size=2,
        max_length=128,
        cache_folder=None,
        revision=None,
        latency_repetitions=1,
    )


def _run(
    *,
    artifact: Mapping[str, Any] | None = None,
    reranker: CrossEncoderReranker | None = None,
    fetcher: Any = None,
    check_hashes: bool = False,
    queries: Sequence[EvalQuery] = QUERIES,
) -> dict[str, Any]:
    """Run the benchmark with isolated fakes and no probe."""
    return run_rerank_benchmark(
        _benchmark_config(),
        queries,
        historical=artifact or _artifact(),
        reranker=reranker or _reranker(_Backend()),
        environment={"execution_environment": "test", "documents": 150},
        dataset={"queries": 2, "hashes": {"kb_documents.json": "fake"}},
        text_fetcher=fetcher or _fetcher(_texts()),
        run_probe=False,
        run_direction=False,
        check_hashes=check_hashes,
    )


def test_two_query_run_reranks_and_keeps_five_arms() -> None:
    """The live arm reranks the historical pool and preserves the four baselines."""
    result = _run()

    assert result["status"] == "PARTIAL"  # fewer than the 30 required queries
    assert result["partial"] is True
    assert result["totals"]["errors"] == 0
    assert result["totals"]["fallbacks"] == 0
    assert result["totals"]["reranker_evaluated_queries"] == 2
    assert result["historical_verification"]["ok"] is True
    assert set(result["arms"]["arms"]) == {*HISTORICAL_ARMS, "rrf_reranker"}
    assert result["metric_version"] == 2
    assert result["labels_exhaustive"] is False
    assert result["relevance_scheme"] == "binary_known_gold"

    first = result["per_query"][0]
    assert first["query_id"] == "Q-001"
    assert first["rrf_reranker"]["doc_ids"] == ["D1", "D3", "D2"]
    assert first["rrf_reranker"]["first_gold_rank"] == 1
    assert first["rrf_reranker"]["hit_at_1"] is True
    assert first["deltas"]["rrf_reranker_vs_rrf_reciprocal_rank"] == pytest.approx(0.5)
    assert first["rerank"]["fallback_used"] is False
    assert first["rrf_reranker"]["hits"][0]["rerank_score"] is not None
    assert first["rrf_reranker"]["hits"][0]["pre_rerank_rank"] == 2
    assert first["candidate_pool"][0]["rrf_score"] is not None

    second = result["per_query"][1]
    assert second["rrf_reranker"]["doc_ids"] == ["D1", "D3", "D2"]
    assert second["rrf_reranker"]["first_gold_rank"] == 3
    assert second["deltas"]["rrf_reranker_vs_rrf_reciprocal_rank"] == pytest.approx(
        1.0 / 3.0 - 0.5
    )

    ordering = result["arms"]["rrf_reranker_ordering"]
    assert ordering["improved"] == 1
    assert ordering["worse"] == 1
    assert ordering["equal"] == 0
    assert ordering["moved_queries"] == ["Q-001", "Q-002"]


def test_supplementary_metrics_are_aggregated_for_every_arm() -> None:
    """P@K and nDCG@K use document rank and the same fixed K for each arm."""
    result = _run()
    arms = result["arms"]["arms"]

    historical = arms["rrf"]
    assert historical["precision_at_k_macro"] == pytest.approx(1 / 3)
    assert historical["precision_at_k_micro"] == pytest.approx(1 / 3)
    assert historical["precision_at_k_ceiling_macro"] == pytest.approx(1 / 3)
    assert historical["ndcg_at_k_macro"] == pytest.approx(0.6309297535714574)

    live = arms["rrf_reranker"]
    assert live["precision_at_k_macro"] == pytest.approx(1 / 3)
    assert live["precision_at_k_micro"] == pytest.approx(1 / 3)
    assert live["precision_at_k_ceiling_macro"] == pytest.approx(1 / 3)
    assert live["ndcg_at_k_macro"] == pytest.approx(0.75)

    per_query = result["per_query"][0]["rrf_reranker"]
    assert per_query["precision_at_k"] == pytest.approx(1 / 3)
    assert per_query["precision_at_k_ceiling"] == pytest.approx(1 / 3)
    assert per_query["ndcg_at_k"] == 1.0


def test_aggregate_arm_keeps_missing_supplementary_metrics_as_none() -> None:
    """Old artifacts without the new fields aggregate to None, not KeyError or zero."""
    old_record = {
        "evaluated": True,
        "semantic_recall": 1.0,
        "reciprocal_rank": 1.0,
        "gold_count": 1,
        "recalled_gold": 1,
        "hit_at_1": True,
        "empty_result": False,
    }
    aggregate = _aggregate_arm([old_record], top_k=TOP_K)
    assert aggregate["semantic_recall_macro"] == 1.0
    assert aggregate["precision_at_k_macro"] is None
    assert aggregate["precision_at_k_micro"] is None
    assert aggregate["precision_at_k_ceiling_macro"] is None
    assert aggregate["ndcg_at_k_macro"] is None


def test_aggregate_arm_mixed_new_and_legacy_records_uses_all_or_nothing() -> None:
    """One legacy record suppresses new aggregates without affecting legacy metrics."""
    new_record = {
        "evaluated": True,
        "semantic_recall": 1.0,
        "reciprocal_rank": 1.0,
        "gold_count": 1,
        "recalled_gold": 1,
        "hit_at_1": True,
        "empty_result": False,
        "precision_at_k": 1.0,
        "precision_at_k_ceiling": 1.0,
        "ndcg_at_k": 1.0,
    }
    legacy_record = {
        "evaluated": True,
        "semantic_recall": 0.5,
        "reciprocal_rank": 0.5,
        "gold_count": 2,
        "recalled_gold": 1,
        "hit_at_1": False,
        "empty_result": False,
    }
    new_before = dict(new_record)
    legacy_before = dict(legacy_record)

    aggregate = _aggregate_arm([new_record, legacy_record], top_k=TOP_K)

    assert aggregate["evaluated_queries"] == 2
    assert aggregate["precision_at_k_macro"] is None
    assert aggregate["precision_at_k_micro"] is None
    assert aggregate["precision_at_k_ceiling_macro"] is None
    assert aggregate["ndcg_at_k_macro"] is None
    assert aggregate["semantic_recall_macro"] == pytest.approx(0.75)
    assert aggregate["semantic_recall_micro"] == pytest.approx(2 / 3)
    assert aggregate["mrr"] == pytest.approx(0.75)
    assert aggregate["hit_at_1"] == 1
    assert aggregate["hit_at_1_rate"] == pytest.approx(0.5)
    assert aggregate["recalled_gold"] == 2
    assert aggregate["gold_total"] == 3
    assert new_record == new_before
    assert legacy_record == legacy_before


def test_reranker_hits_record_scores_and_pre_rerank_provenance() -> None:
    """Each reranked hit carries its score and its old rank."""
    result = _run()
    hits = result["per_query"][0]["rrf_reranker"]["hits"]

    assert [hit["rank"] for hit in hits] == [1, 2, 3]
    assert [hit["rerank_score"] for hit in hits] == [3.0, 2.0, 1.0]
    assert [hit["pre_rerank_rank"] for hit in hits] == [2, 3, 1]
    assert all(hit["source"] == "reranked" for hit in hits)
    assert all(hit["distance"] is None for hit in hits)
    assert all(hit["pair_text_sha256"] for hit in hits)


def test_empty_historical_pool_is_an_input_error_not_a_fallback() -> None:
    """An empty pool is reported as input failure with no reranked list."""
    artifact = _artifact()
    artifact["per_query"][0]["arms"]["rrf"]["hits"] = []
    result = _run(artifact=artifact, fetcher=_fetcher(_texts()))

    assert result["status"] == "FAIL"
    assert result["totals"]["input_errors"] == 1
    assert result["per_query"][0]["rrf_reranker"]["evaluated"] is False
    assert result["per_query"][0]["error"]["stage"] == "input"
    assert result["per_query"][0]["rerank"]["fallback_used"] is False


def test_missing_candidate_text_is_an_input_error() -> None:
    """A missing chunks row is not silently dropped from scoring."""
    result = _run(fetcher=_fetcher({}))

    assert result["status"] == "FAIL"
    assert result["totals"]["input_errors"] == 2
    assert all(
        record["rrf_reranker"]["evaluated"] is False for record in result["per_query"]
    )
    assert "missing text" in result["errors"][0]["message"]


def test_database_fetch_failure_is_isolated_per_query() -> None:
    """A database error is categorised separately from an inference error."""

    def broken(_identities: Sequence[tuple[str, str]]) -> Mapping[tuple[str, str], Any]:
        raise RuntimeError("database down")

    result = _run(fetcher=broken)

    assert result["status"] == "FAIL"
    assert result["totals"]["db_fetch_errors"] == 2
    assert result["totals"]["inference_errors"] == 0
    assert result["per_query"][0]["rerank"]["fallback_used"] is False


def test_inference_failure_never_returns_the_pre_rerank_list() -> None:
    """A backend exception fails the query and the run; no silent fallback."""
    result = _run(reranker=_reranker(_Backend(error=RuntimeError("CUDA on fire"))))

    assert result["status"] == "FAIL"
    assert result["totals"]["inference_errors"] == 2
    assert result["totals"]["fallbacks"] == 0
    for record in result["per_query"]:
        assert record["rrf_reranker"]["evaluated"] is False
        assert record["rrf_reranker"]["hits"] == []
        assert record["rrf_reranker"]["precision_at_k"] is None
        assert record["rrf_reranker"]["precision_at_k_ceiling"] is None
        assert record["rrf_reranker"]["ndcg_at_k"] is None
        assert record["rerank"]["fallback_used"] is False


def test_historical_metric_mismatch_fails_verification_and_the_run() -> None:
    """A tampered stored metric is detected, not silently accepted."""
    artifact = _artifact()
    artifact["arms"]["arms"]["rrf"]["mrr"] = 0.123
    result = _run(artifact=artifact)

    assert result["historical_verification"]["ok"] is False
    assert result["historical_verification"]["mismatch_count"] >= 1
    assert result["status"] == "FAIL"


def test_historical_rrf_formula_mismatch_is_detected() -> None:
    """The stored RRF score must equal the standard formula."""
    artifact = _artifact()
    for record in artifact["per_query"]:
        record["arms"]["rrf"]["hits"][0]["rrf_score"] = 0.5
        record["arms"]["rrf"]["hits"][0]["score"] = 0.5

    verification = verify_historical(
        artifact, QUERIES, _benchmark_config(), check_hashes=False
    )

    assert verification["ok"] is False
    assert verification["rrf_formula_mismatches"] == 2


def test_model_load_failure_returns_a_fail_artifact_without_queries() -> None:
    """Model initialisation is reported before any candidate is scored."""

    class _BrokenReranker:
        name = "broken"
        model_id = "broken-model"
        revision = None
        device = "cpu"
        batch_size = 1
        max_length = 8
        activation = "identity"

        def load(self) -> None:
            raise RerankError("weights are missing", stage="model_init")

        def score(self, query: str, candidates: Sequence[SearchHit]) -> list[float]:
            raise AssertionError("must not score")

        def rerank(
            self, query: str, candidates: Sequence[SearchHit], top_k: int | None = None
        ) -> list[SearchHit]:
            raise AssertionError("must not rerank")

    result = _run(reranker=_BrokenReranker())  # type: ignore[arg-type]

    assert result["status"] == "FAIL"
    assert result["failure"]["stage"] == "model_init"
    assert result["totals"]["model_load_errors"] == 1
    assert result["per_query"] == []
    assert result["metric_version"] == 2
    assert result["labels_exhaustive"] is False
    assert result["relevance_scheme"] == "binary_known_gold"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"candidate_k": 11}, "candidate_k"),
        ({"top_k": 4, "candidate_k": 3}, "top_k"),
        ({"rrf_k": 61}, "rrf_k"),
        ({"latency_repetitions": 0}, "latency_repetitions"),
    ],
)
def test_build_config_rejects_fixed_experiment_violations(
    overrides: dict[str, Any], message: str
) -> None:
    """The historical pool and RRF k cannot be changed silently."""
    with pytest.raises(BenchmarkError, match=message):
        build_config(cache_folder=None, revision=None, **overrides)


def test_partial_selection_does_not_compare_a_subset_to_full_aggregates() -> None:
    """A --limit run verifies selected rows but is correctly marked PARTIAL."""
    verification = verify_historical(
        _artifact(), QUERIES[:1], _benchmark_config(), check_hashes=False
    )
    assert verification["ok"] is True
    assert verification["selection_full"] is False
    assert verification["aggregate_comparison_skipped"] is True

    result = _run(queries=QUERIES[:1])
    assert result["status"] == "PARTIAL"
    assert result["partial"] is True
    assert result["historical_verification"]["ok"] is True
    assert len(result["per_query"]) == 1


def test_verification_rejects_an_artifact_from_another_chunk_scope() -> None:
    """Historical evidence is only valid inside the scope it was produced for."""
    config = build_config(
        top_k=TOP_K,
        candidate_k=CANDIDATE_K,
        rrf_k=RRF_K,
        batch_size=2,
        max_length=128,
        cache_folder=None,
        revision=None,
        latency_repetitions=1,
        chunk_strategy="fixed_window",
    )
    verification = verify_historical(_artifact(), QUERIES, config, check_hashes=False)

    assert verification["ok"] is False
    assert any("chunk_strategy" in value for value in verification["mismatches"])


def test_probe_is_absent_when_disabled() -> None:
    """`run_probe=False` performs no retrieval probe."""
    result = _run()
    assert result["probe"] is None


def test_fetch_chunk_texts_uses_one_parameterised_query() -> None:
    """The DB helper fetches only the requested candidate identities."""

    class _Cursor:
        def __init__(self) -> None:
            self.sql = ""
            self.params: list[Any] = []

        def execute(self, sql: str, params: Sequence[Any]) -> _Cursor:
            self.sql = sql
            self.params = list(params)
            return self

        def fetchall(self) -> list[tuple[Any, ...]]:
            return [
                (
                    "document",
                    "C1",
                    "D1",
                    0,
                    "alpha",
                    "Title C1",
                    "rewards",
                    "gold",
                    2026,
                    "P-1",
                ),
                (
                    "document",
                    "C2",
                    "D2",
                    1,
                    "beta",
                    "Title C2",
                    "rewards",
                    "gold",
                    2026,
                    "P-1",
                ),
            ]

    class _Connection:
        def __init__(self) -> None:
            self.cursor = _Cursor()

        def execute(self, sql: str, params: Sequence[Any]) -> _Cursor:
            return self.cursor.execute(sql, params)

        def __enter__(self) -> _Connection:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class _Factory(AbstractContextManager[_Connection]):
        def __init__(self) -> None:
            self.conn = _Connection()

        def __enter__(self) -> _Connection:
            return self.conn

        def __exit__(self, *args: object) -> None:
            return None

    factory = _Factory()
    texts = fetch_chunk_texts(
        [("document", "C1"), ("document", "C2")],
        connection_factory=lambda: factory,
    )

    assert set(texts) == {("document", "C1"), ("document", "C2")}
    assert texts[("document", "C1")]["content"] == "alpha"
    assert "(chunk_strategy, chunk_id) IN ((%s, %s), (%s, %s))" in factory.conn.cursor.sql
    assert factory.conn.cursor.params == ["document", "C1", "document", "C2"]


def test_report_rendering_and_writing_do_not_need_the_real_model(tmp_path: Any) -> None:
    """The Markdown report contains the five arms and all per-query rows."""
    result = _run()
    text = render_markdown(result)
    assert "RRF+Reranker" in text
    assert "Q-001" in text and "Q-002" in text
    assert "### 补充指标：Precision@K / 二元 nDCG@K" in text
    assert "P@K ceiling macro" in text
    assert "labels_exhaustive=False" in text
    assert "未验证事项" in text

    json_path, markdown_path = write_outputs(
        result, tmp_path / "rerank.json", tmp_path / "rerank.md"
    )
    assert json_path.exists() and markdown_path.exists()
    assert json_path.stat().st_size > 1000


def test_historical_hit_round_trip_keeps_rrf_identity() -> None:
    """Stored provenance can be reconstructed without changing IDs or order."""
    query = QUERIES[0]
    record = _frozen_arm_record(query, [("C2", "D2"), ("C1", "D1")], "rrf")
    hits = [_historical_hit(hit) for hit in record["hits"]]
    assert [hit.doc_id for hit in hits] == ["D2", "D1"]
    assert hits[0].dense_rank == 1
    assert hits[0].sparse_rank == 1
    assert hits[0].rrf_score == pytest.approx(2.0 / (RRF_K + 1))


def test_hit_record_includes_the_reranker_provenance() -> None:
    """Serialised reranked hits expose score, rank mapping and text hash."""
    hit = SearchHit(
        chunk_id="C1",
        doc_id="D1",
        score=2.0,
        rank=1,
        source="reranked",
        rerank_score=2.0,
        pre_rerank_rank=2,
        pre_rerank_score=0.01,
        content="alpha",
        title="Title C1",
        chunk_strategy="document",
    )
    record = _hit_record(hit)
    assert record["rerank_score"] == 2.0
    assert record["pre_rerank_rank"] == 2
    assert record["content_chars"] == len("Title C1\nalpha")
    assert record["source"] == "reranked"
