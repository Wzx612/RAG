# Dense Retrieval Benchmark (Phase 4C)

- Generated: `2026-10-17T06:47:56.816482+00:00`
- Status: **PASS**
- Queries: 30  |  K: 10  |  Timed passes/config: 5  |  Statement timeout: 5000 ms

## 1. Environment and dataset

| Item | Value |
|---|---|
| PostgreSQL | 16.14 (Debian 16.14-1.pgdg12+1) |
| pgvector extension | 0.8.2 |
| HNSW index | CREATE INDEX chunks_embedding_hnsw_idx ON public.chunks USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='64') |
| HNSW parameters | m=16, ef_construction=64 |
| Server address | 172.22.0.2/32:5432 |
| Data directory | /var/lib/postgresql/data |
| Host Python | 3.14.3 |
| Host model stack | sentence-transformers=5.5.1, torch=2.13.0, transformers=5.10.2, numpy=2.4.6, psycopg=3.3.6, pgvector=0.5.1 |
| Embedding model | BAAI/bge-large-en-v1.5 |
| Docker control plane | unresponsive (> 5s); no container command executed |
| Free disk | project=110.91GB free; C=8.48GB free; D=110.91GB free |
| Documents / chunks | 150 / 150 |
| Distinct chunk doc_ids | 150 |
| Eval queries | 30 |
| Gold sizes | [1, 2, 3] |
| Missing gold docs | [] |
| Data files | {'kb_documents.json': '7d230e97cbf76d2ca64382e7c0278c3b24bbe27b8a8904abc9553b45a230057d', 'eval_queries.json': '739bd308a90e72437ccbffca658ca950f6a069f2986040cb729694a878d9bbef'} |
| Counts after run | {'documents': 150, 'chunks': 150, 'distinct_chunk_docs': 150, 'embedding_dimensions': [1024]} |

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
| exact_seqscan | seq_scan | False | True | 0.870 | True |
| natural_plan | seq_scan | False | True | 0.440 | True |
| hnsw_ef_10 | hnsw | True | False | 0.370 | True |
| hnsw_ef_40 | hnsw | True | False | 0.243 | True |
| hnsw_ef_100 | hnsw | True | False | 0.429 | True |

Captured plan snippets:

### `exact_seqscan`

```text
Sort  (cost=77.60..77.62 rows=10 width=448) (actual time=0.829..0.832 rows=10 loops=1)
  Sort Key: (omitted: contains the 1024-dim query vector)
  Sort Method: quicksort  Memory: 30kB
  Buffers: shared hit=573
  ->  Hash Join  (cost=57.27..77.43 rows=10 width=448) (actual time=0.751..0.809 rows=10 loops=1)
        Hash Cond: (d.doc_id = chunks.doc_id)
        Buffers: shared hit=570
        ->  Seq Scan on documents d  (cost=0.00..19.50 rows=150 width=123) (actual time=0.017..0.042 rows=150 loops=1)
              Buffers: shared hit=18
        ->  Hash  (cost=57.14..57.14 rows=10 width=334) (actual time=0.720..0.721 rows=10 loops=1)
              Buckets: 1024  Batches: 1  Memory Usage: 13kB
              Buffers: shared hit=552
              ->  Limit  (cost=57.12..57.14 rows=10 width=334) (actual time=0.710..0.712 rows=10 loops=1)
                    Buffers: shared hit=552
                    ->  Sort  (cost=57.12..57.49 rows=150 width=334) (actual time=0.709..0.710 rows=10 loops=1)
  Sort Key: (omitted: contains the 1024-dim query vector)
                          Sort Method: top-N heapsort  Memory: 33kB
                          Buffers: shared hit=552
                          ->  Seq Scan on chunks  (cost=0.00..53.88 rows=150 width=334) (actual time=0.052..0.644 rows=150 loops=1)
                                Buffers: shared hit=552
Planning:
  Buffers: shared hit=103
Planning Time: 0.335 ms
Execution Time: 0.870 ms
```

### `natural_plan`

```text
Sort  (cost=77.60..77.62 rows=10 width=448) (actual time=0.426..0.427 rows=10 loops=1)
  Sort Key: (omitted: contains the 1024-dim query vector)
  Sort Method: quicksort  Memory: 30kB
  Buffers: shared hit=570
  ->  Hash Join  (cost=57.27..77.43 rows=10 width=448) (actual time=0.398..0.422 rows=10 loops=1)
        Hash Cond: (d.doc_id = chunks.doc_id)
        Buffers: shared hit=570
        ->  Seq Scan on documents d  (cost=0.00..19.50 rows=150 width=123) (actual time=0.003..0.013 rows=150 loops=1)
              Buffers: shared hit=18
        ->  Hash  (cost=57.14..57.14 rows=10 width=334) (actual time=0.384..0.385 rows=10 loops=1)
              Buckets: 1024  Batches: 1  Memory Usage: 13kB
              Buffers: shared hit=552
              ->  Limit  (cost=57.12..57.14 rows=10 width=334) (actual time=0.378..0.380 rows=10 loops=1)
                    Buffers: shared hit=552
                    ->  Sort  (cost=57.12..57.49 rows=150 width=334) (actual time=0.378..0.378 rows=10 loops=1)
  Sort Key: (omitted: contains the 1024-dim query vector)
                          Sort Method: top-N heapsort  Memory: 33kB
                          Buffers: shared hit=552
                          ->  Seq Scan on chunks  (cost=0.00..53.88 rows=150 width=334) (actual time=0.022..0.343 rows=150 loops=1)
                                Buffers: shared hit=552
Planning:
  Buffers: shared hit=5
Planning Time: 0.085 ms
Execution Time: 0.440 ms
```

### `hnsw_ef_10`

```text
Sort  (cost=510.96..510.99 rows=10 width=448) (actual time=0.346..0.347 rows=10 loops=1)
  Sort Key: (omitted: contains the 1024-dim query vector)
  Sort Method: quicksort  Memory: 30kB
  Buffers: shared hit=190
  ->  Merge Join  (cost=477.51..510.79 rows=10 width=448) (actual time=0.309..0.342 rows=10 loops=1)
        Merge Cond: (chunks.doc_id = d.doc_id)
        Buffers: shared hit=190
        ->  Sort  (cost=477.37..477.39 rows=10 width=334) (actual time=0.304..0.305 rows=10 loops=1)
  Sort Key: (omitted: contains the 1024-dim query vector)
              Sort Method: quicksort  Memory: 29kB
              Buffers: shared hit=177
              ->  Limit  (cost=453.07..477.20 rows=10 width=334) (actual time=0.271..0.294 rows=10 loops=1)
                    Buffers: shared hit=177
                    ->  Index Scan using chunks_embedding_hnsw_idx on chunks  (cost=453.07..815.00 rows=150 width=334) (actual time=0.271..0.293 rows=10 loops=1)
                          Order By: (embedding <=> (query vector omitted))
                          Buffers: shared hit=177
        ->  Index Scan using documents_pkey on documents d  (cost=0.14..32.90 rows=150 width=123) (actual time=0.003..0.016 rows=150 loops=1)
              Buffers: shared hit=13
Planning:
  Buffers: shared hit=5
Planning Time: 0.078 ms
Execution Time: 0.370 ms
```

### `hnsw_ef_40`

```text
Sort  (cost=652.88..652.90 rows=10 width=448) (actual time=0.229..0.230 rows=10 loops=1)
  Sort Key: (omitted: contains the 1024-dim query vector)
  Sort Method: quicksort  Memory: 30kB
  Buffers: shared hit=250
  ->  Merge Join  (cost=619.43..652.71 rows=10 width=448) (actual time=0.181..0.223 rows=10 loops=1)
        Merge Cond: (chunks.doc_id = d.doc_id)
        Buffers: shared hit=250
        ->  Sort  (cost=619.28..619.31 rows=10 width=334) (actual time=0.177..0.178 rows=10 loops=1)
  Sort Key: (omitted: contains the 1024-dim query vector)
              Sort Method: quicksort  Memory: 29kB
              Buffers: shared hit=237
              ->  Limit  (cost=605.12..619.12 rows=10 width=334) (actual time=0.139..0.168 rows=10 loops=1)
                    Buffers: shared hit=237
                    ->  Index Scan using chunks_embedding_hnsw_idx on chunks  (cost=605.12..815.00 rows=150 width=334) (actual time=0.138..0.167 rows=10 loops=1)
                          Order By: (embedding <=> (query vector omitted))
                          Buffers: shared hit=237
        ->  Index Scan using documents_pkey on documents d  (cost=0.14..32.90 rows=150 width=123) (actual time=0.002..0.020 rows=150 loops=1)
              Buffers: shared hit=13
Planning:
  Buffers: shared hit=5
Planning Time: 0.083 ms
Execution Time: 0.243 ms
```

### `hnsw_ef_100`

```text
Sort  (cost=652.88..652.90 rows=10 width=448) (actual time=0.411..0.413 rows=10 loops=1)
  Sort Key: (omitted: contains the 1024-dim query vector)
  Sort Method: quicksort  Memory: 30kB
  Buffers: shared hit=321
  ->  Merge Join  (cost=619.43..652.71 rows=10 width=448) (actual time=0.363..0.406 rows=10 loops=1)
        Merge Cond: (chunks.doc_id = d.doc_id)
        Buffers: shared hit=321
        ->  Sort  (cost=619.28..619.31 rows=10 width=334) (actual time=0.356..0.357 rows=10 loops=1)
  Sort Key: (omitted: contains the 1024-dim query vector)
              Sort Method: quicksort  Memory: 29kB
              Buffers: shared hit=308
              ->  Limit  (cost=605.12..619.12 rows=10 width=334) (actual time=0.306..0.346 rows=10 loops=1)
                    Buffers: shared hit=308
                    ->  Index Scan using chunks_embedding_hnsw_idx on chunks  (cost=605.12..815.00 rows=150 width=334) (actual time=0.305..0.345 rows=10 loops=1)
                          Order By: (embedding <=> (query vector omitted))
                          Buffers: shared hit=308
        ->  Index Scan using documents_pkey on documents d  (cost=0.14..32.90 rows=150 width=123) (actual time=0.004..0.026 rows=150 loops=1)
              Buffers: shared hit=13
Planning:
  Buffers: shared hit=5
Planning Time: 0.127 ms
Execution Time: 0.429 ms
```

## 5. Latency and Recall@K

Latency is reported separately for `query_ms` (execute + fetch only) and `db_ms` (pool borrow + transaction + GUC/planner setup + execute + fetch). Percentiles are nearest-rank over all per-query timed samples.

| Arm | n | query P50 | query P95 | query P99 | db P50 | db P95 | ANN Recall@K mean | ANN Recall@K min | Semantic Recall mean | Order-equal | Top-1 match | Errors | Timeouts | Skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_seqscan | 150 | 1.022 | 1.185 | 1.377 | 4.237 | 4.993 | 1.000 | 1.000 | 0.944 | 30 | 30 | 0 | 0 | 0 |
| natural_plan | 150 | 0.920 | 1.146 | 1.425 | 2.935 | 3.480 | 1.000 | 1.000 | 0.944 | 30 | 30 | 0 | 0 | 0 |
| hnsw_ef_10 | 150 | 0.778 | 0.960 | 1.216 | 3.725 | 4.387 | 1.000 | 1.000 | 0.944 | 30 | 30 | 0 | 0 | 0 |
| hnsw_ef_40 | 150 | 0.835 | 1.053 | 1.316 | 3.810 | 4.719 | 1.000 | 1.000 | 0.944 | 30 | 30 | 0 | 0 | 0 |
| hnsw_ef_100 | 150 | 0.819 | 1.112 | 1.598 | 3.707 | 5.625 | 1.000 | 1.000 | 0.944 | 30 | 30 | 0 | 0 | 0 |

`ANN Recall@K` is against the exact chunk Top-K. `Semantic Recall` is against `relevant_doc_ids` and is a different question: it measures whether the returned documents are the labelled ones, not whether ANN reproduced exact vector order.

## 6. Per-arm detail

| Arm | Evaluated | Missing | Baseline missing | Shortfall | Overflow | Duplicates | ef mismatches | Quality mismatches | Observed ef | Valid | Skip reason | query min | query max | db min | db max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_seqscan | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 40 | True | - | 0.870 | 1.450 | 3.551 | 5.390 |
| natural_plan | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 40 | True | - | 0.791 | 1.567 | 2.507 | 4.637 |
| hnsw_ef_10 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | True | - | 0.590 | 1.222 | 3.016 | 5.150 |
| hnsw_ef_40 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 40 | True | - | 0.685 | 1.525 | 3.079 | 5.326 |
| hnsw_ef_100 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | True | - | 0.700 | 1.705 | 3.153 | 7.254 |

`Shortfall` counts queries returning fewer than K rows (a legal HNSW outcome when `ef_search < K`); `Overflow` and `Duplicates` must be zero for a sound Top-K. `ef mismatches` compares the read-back live GUC to the requested value; it must be zero. `Quality mismatches` counts timed samples whose result signature differs from that query's first sample.

## 7. Interpretation and limitations

- Natural vs exact on 30 queries: mean ANN Recall@K = 1.000, order-equal = 30, top-1 match = 30.
- Natural plan kind: `seq_scan`. At 150 chunks a Seq Scan is plausible; `ef_search` is then not observable in the natural path.
- `hnsw_ef_10`: observed ef = [10], ANN Recall@K mean = 1.000, order-equal = 30/30, query P50/P95 = 0.778/0.960 ms.
- `hnsw_ef_40`: observed ef = [40], ANN Recall@K mean = 1.000, order-equal = 30/30, query P50/P95 = 0.835/1.053 ms.
- `hnsw_ef_100`: observed ef = [100], ANN Recall@K mean = 1.000, order-equal = 30/30, query P50/P95 = 0.819/1.112 ms.
- The corpus is only 150 chunks. If every ANN arm reproduces exact Top-K, that is a scale observation, not proof that `ef_search` is ineffective at larger N. It also does not establish an HNSW latency advantage.
- Latency here is single-process, sequential and warm; no concurrency or throughput claim is made. Embedding time is measured separately from database retrieval and is not folded into `query_ms`.
- The human-labelled `relevant_doc_ids` measure retrieval quality; ANN Recall@K measures fidelity to the exact vector search. Neither is a substitute for the other.
- `SET LOCAL enable_seqscan=off` is a diagnostic control, never a production or global setting. The natural arm is the production path.

## 8. Errors, timeouts, skips

- Totals: errors=0, timeouts=0, skipped=0, invalid configs=0.
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
