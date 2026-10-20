# Dense Retrieval Benchmark (Phase 4C)

- Generated: `2026-10-20T03:20:52.742325+00:00`
- Status: **ISSUES**
- Queries: 30  |  K: 10  |  Timed passes/config: 5  |  Statement timeout: 5000 ms  |  Chunk strategy: `document`

## 1. Environment and dataset

| Item | Value |
|---|---|
| PostgreSQL | 16.14 (Debian 16.14-1.pgdg12+1) |
| pgvector extension | 0.8.2 |
| HNSW index | CREATE INDEX chunks_embedding_hnsw_idx ON public.chunks USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='64') |
| HNSW parameters | m=16, ef_construction=64 |
| Server address | 172.22.0.2/32:5432 |
| Data directory | /var/lib/postgresql/data |
| Host Python | 3.12.13 |
| Host model stack | sentence-transformers=6.1.0, torch=2.14.1+cpu, transformers=5.20.0, numpy=2.5.4, psycopg=3.3.6, pgvector=0.5.1 |
| Embedding model | BAAI/bge-large-en-v1.5 |
| Docker control plane | not installed |
| Free disk | project=110.59GB free |
| Documents / chunks | 150 / 578 |
| Distinct chunk doc_ids | 150 |
| Scoped chunks / doc_ids | 150 / 150 (`document`) |
| Chunk strategies in table | {'document': 150, 'fixed_window': 267, 'structured': 161} |
| Eval queries | 30 |
| Gold sizes | [1, 2, 3] |
| Missing gold docs | [] |
| Data files | {'kb_documents.json': '7d230e97cbf76d2ca64382e7c0278c3b24bbe27b8a8904abc9553b45a230057d', 'eval_queries.json': '739bd308a90e72437ccbffca658ca950f6a069f2986040cb729694a878d9bbef'} |
| Counts after run | {'documents': 150, 'chunks': 578, 'distinct_chunk_docs': 150, 'strategy_chunks': 150, 'strategy_docs': 150, 'embedding_dimensions': [1024]} |

The numbers below were produced on the host Python fallback against the live PostgreSQL container. The Docker CLI/API was unresponsive during the run, so container-internal execution is **not verified** by this report; the database data plane was healthy and all benchmark statements are read-only.

## 2. Experiment matrix and parameters

| Arm | Mode | ef_search | Top-K | Planner control | Definition |
|---|---|---|---|---|---|
| exact_seqscan | exact | 40 | 10 | index/bitmap/indexonly off, seqscan on | Exact cosine Top-K via a forced Seq Scan on chunks. |
| natural_plan | natural | 40 | 10 | none | Production path: no planner forcing, default ef_search. |
| hnsw_ef_10 | hnsw | 10 | 10 | seqscan off | Forced HNSW path at hnsw.ef_search=10. |
| hnsw_ef_40 | hnsw | 40 | 10 | seqscan off | Forced HNSW path at hnsw.ef_search=40. |
| hnsw_ef_100 | hnsw | 100 | 10 | seqscan off | Forced HNSW path at hnsw.ef_search=100. |

The exact arm is the baseline; the natural arm is the production path; the HNSW arms are controlled diagnostics. `ef_search` values are legal pgvector values in the project-validated range 1--1000.

## 3. Exact baseline definition

The exact baseline uses the **same** `DENSE_SEARCH_SQL`, query vectors, cosine metric, chunk granularity and Top-K as every ANN arm. Only the transaction-local planner settings differ: `enable_indexscan=off`, `enable_bitmapscan=off`, `enable_indexonlyscan=off`, `enable_seqscan=on`. The captured plan must contain `Seq Scan on chunks` or the run refuses to call it exact.

## 4. Plan evidence

| Arm | Plan kind | Uses HNSW | Seq Scan on chunks | EXPLAIN ms | Valid |
|---|---|---|---|---|---|
| exact_seqscan | seq_scan | False | True | 0.803 | True |
| natural_plan | other_index | False | False | 0.459 | True |
| hnsw_ef_10 | missing | - | - | - | False |
| hnsw_ef_40 | missing | - | - | - | False |
| hnsw_ef_100 | missing | - | - | - | False |

Captured plan snippets:

### `exact_seqscan`

```text
Sort  (cost=159.70..159.72 rows=10 width=360) (actual time=0.767..0.769 rows=10 loops=1)
  Sort Key: (omitted: contains the 1024-dim query vector)
  Sort Method: quicksort  Memory: 30kB
  Buffers: shared hit=649
  ->  Hash Join  (cost=132.99..159.53 rows=10 width=360) (actual time=0.717..0.751 rows=10 loops=1)
        Hash Cond: (d.doc_id = chunks.doc_id)
        Buffers: shared hit=646
        ->  Seq Scan on documents d  (cost=0.00..25.50 rows=150 width=123) (actual time=0.005..0.026 rows=150 loops=1)
              Buffers: shared hit=24
        ->  Hash  (cost=132.87..132.87 rows=10 width=246) (actual time=0.694..0.694 rows=10 loops=1)
              Buckets: 1024  Batches: 1  Memory Usage: 13kB
              Buffers: shared hit=622
              ->  Limit  (cost=132.84..132.87 rows=10 width=246) (actual time=0.687..0.689 rows=10 loops=1)
                    Buffers: shared hit=622
                    ->  Sort  (cost=132.84..133.22 rows=150 width=246) (actual time=0.686..0.686 rows=10 loops=1)
  Sort Key: (omitted: contains the 1024-dim query vector)
                          Sort Method: top-N heapsort  Memory: 33kB
                          Buffers: shared hit=622
                          ->  Seq Scan on chunks  (cost=0.00..129.60 rows=150 width=246) (actual time=0.020..0.622 rows=150 loops=1)
                                Filter: (chunk_strategy = 'document'::text)
                                Rows Removed by Filter: 428
                                Buffers: shared hit=622
Planning:
  Buffers: shared hit=105
Planning Time: 0.347 ms
Execution Time: 0.803 ms
```

### `natural_plan`

```text
Nested Loop  (cost=66.55..83.54 rows=10 width=360) (actual time=0.426..0.445 rows=10 loops=1)
  Buffers: shared hit=552
  ->  Limit  (cost=66.39..66.42 rows=10 width=246) (actual time=0.418..0.420 rows=10 loops=1)
        Buffers: shared hit=532
        ->  Sort  (cost=66.39..66.77 rows=150 width=246) (actual time=0.418..0.419 rows=10 loops=1)
  Sort Key: (omitted: contains the 1024-dim query vector)
              Sort Method: top-N heapsort  Memory: 33kB
              Buffers: shared hit=532
              ->  Index Scan using chunks_strategy_idx on chunks  (cost=0.15..63.15 rows=150 width=246) (actual time=0.012..0.382 rows=150 loops=1)
                    Index Cond: (chunk_strategy = 'document'::text)
                    Buffers: shared hit=532
  ->  Memoize  (cost=0.15..4.57 rows=1 width=123) (actual time=0.002..0.002 rows=1 loops=10)
        Cache Key: chunks.doc_id
        Cache Mode: logical
        Hits: 0  Misses: 10  Evictions: 0  Overflows: 0  Memory Usage: 3kB
        Buffers: shared hit=20
        ->  Index Scan using documents_pkey on documents d  (cost=0.14..4.56 rows=1 width=123) (actual time=0.001..0.001 rows=1 loops=10)
              Index Cond: (doc_id = chunks.doc_id)
              Buffers: shared hit=20
Planning:
  Buffers: shared hit=5
Planning Time: 0.118 ms
Execution Time: 0.459 ms
```

### `hnsw_ef_10`

```text
no plan captured
```

### `hnsw_ef_40`

```text
no plan captured
```

### `hnsw_ef_100`

```text
no plan captured
```

## 5. Latency and Recall@K

Latency is reported separately for `query_ms` (execute + fetch only) and `db_ms` (pool borrow + transaction + GUC/planner setup + execute + fetch). Percentiles are nearest-rank over all per-query timed samples.

| Arm | n | query P50 | query P95 | query P99 | db P50 | db P95 | ANN Recall@K mean | ANN Recall@K min | Semantic Recall mean | Order-equal | Top-1 match | Errors | Timeouts | Skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_seqscan | 150 | 0.681 | 0.818 | 1.081 | 1.641 | 2.178 | 1.000 | 1.000 | 0.944 | 30 | 30 | 0 | 0 | 0 |
| natural_plan | 150 | 0.681 | 0.832 | 0.994 | 1.466 | 1.887 | 1.000 | 1.000 | 0.944 | 30 | 30 | 0 | 0 | 0 |
| hnsw_ef_10 | 0 | - | - | - | - | - | - | - | - | 0 | 0 | 0 | 0 | 150 |
| hnsw_ef_40 | 0 | - | - | - | - | - | - | - | - | 0 | 0 | 0 | 0 | 150 |
| hnsw_ef_100 | 0 | - | - | - | - | - | - | - | - | 0 | 0 | 0 | 0 | 150 |

`ANN Recall@K` is against the exact chunk Top-K. `Semantic Recall` is against `relevant_doc_ids` and is a different question: it measures whether the returned documents are the labelled ones, not whether ANN reproduced exact vector order.

## 6. Per-arm detail

| Arm | Evaluated | Missing | Baseline missing | Shortfall | Overflow | Duplicates | ef mismatches | Quality mismatches | Observed ef | Valid | Skip reason | query min | query max | db min | db max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_seqscan | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 40 | True | - | 0.611 | 1.095 | 1.397 | 2.548 |
| natural_plan | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 40 | True | - | 0.591 | 1.059 | 1.234 | 2.455 |
| hnsw_ef_10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | False | forced HNSW arm did not use the HNSW index; the ef_search comparison would be invalid (plan kind='other_index') | - | - | - | - |
| hnsw_ef_40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | False | forced HNSW arm did not use the HNSW index; the ef_search comparison would be invalid (plan kind='other_index') | - | - | - | - |
| hnsw_ef_100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | False | forced HNSW arm did not use the HNSW index; the ef_search comparison would be invalid (plan kind='other_index') | - | - | - | - |

`Shortfall` counts queries returning fewer than K rows (a legal HNSW outcome when `ef_search < K`); `Overflow` and `Duplicates` must be zero for a sound Top-K. `ef mismatches` compares the read-back live GUC to the requested value; it must be zero. `Quality mismatches` counts timed samples whose result signature differs from that query's first sample.

## 7. Interpretation and limitations

- Natural vs exact on 30 queries: mean ANN Recall@K = 1.000, order-equal = 30, top-1 match = 30.
- Natural plan kind: `other_index`. At 150 chunks a Seq Scan is plausible; `ef_search` is then not observable in the natural path.
- The corpus is only 150 chunks. If every ANN arm reproduces exact Top-K, that is a scale observation, not proof that `ef_search` is ineffective at larger N. It also does not establish an HNSW latency advantage.
- Latency here is single-process, sequential and warm; no concurrency or throughput claim is made. Embedding time is measured separately from database retrieval and is not folded into `query_ms`.
- The human-labelled `relevant_doc_ids` measure retrieval quality; ANN Recall@K measures fidelity to the exact vector search. Neither is a substitute for the other.
- `SET LOCAL enable_seqscan=off` is a diagnostic control, never a production or global setting. The natural arm is the production path.

## 8. Errors, timeouts, skips

- Totals: errors=3, timeouts=0, skipped=450, invalid configs=3.
- Exact reference errors: 0.

Every failed timed sample is stored with its stage, query id, pass, error type and timeout flag in the JSON artifact. A timeout is a `QueryCanceled` or pool timeout; it is not silently converted into a zero-latency or zero-recall result.

## 9. Reproduce

Host fallback (PowerShell), from the repository root:

```powershell
$env:APEX_PG_HOST='127.0.0.1'; $env:HF_HUB_CACHE='D:\hf_cache'; $env:HF_HUB_OFFLINE='1';
python scripts/benchmark_dense.py --top-k 10 --ef-search 10 40 100 --repetitions 5 --warmup-passes 1
```

Container path, once the Docker control plane is healthy:

```bash
docker compose run --rm -T app python scripts/benchmark_dense.py --top-k 10 --ef-search 10 40 100 --repetitions 5 --warmup-passes 1
```

Smoke first:

```bash
python scripts/benchmark_dense.py --limit 3 --top-k 10 --ef-search 10 40 --repetitions 1 --warmup-passes 1 --dry-run
```

## 10. Data safety

The benchmark issues SELECT, EXPLAIN and transaction-local `set_config` only. It does not write rows, change schema, rebuild indexes, set a global GUC, or modify `data/kb_documents.json` / `data/eval_queries.json`. Dataset hashes and the post-run row counts are recorded above. The raw per-sample JSON is in `reports/dense_benchmark.json`.
