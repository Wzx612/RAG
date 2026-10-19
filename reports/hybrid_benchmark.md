# Hybrid Search / RRF 离线评测（阶段 ⑥-B）

- Status: **PASS**
- Generated: 2026-10-17T07:49:52.438512+00:00
- Top-K: 10
- Candidate pool per arm: 10
- RRF k: 60
- Queries: 30
- Repetitions / warm-ups: 5 / 1

## 评测口径

- **candidate_granularity**: chunk identity (chunk_strategy, chunk_id)
- **metric_granularity**: document; distinct doc_id in ranked order, truncated to top_k
- **semantic_recall**: recall_at_k against human relevant_doc_ids; denominator is the number of distinct gold docs, not K; not ANN recall
- **mrr**: mean reciprocal rank of the first gold doc at doc rank r <= top_k; no gold in Top-K contributes 0; denominator is evaluated queries
- **rrf**: sum over arms of 1 / (k + rank), rank starts at 1, k=60; raw dense/sparse scores are never added
- **candidate_union**: interleaved dense[0], sparse[0], dense[1], sparse[1], ... after removing repeated chunk identities; score is a 0.0 placeholder because this baseline has no fused score
- **candidate_pool**: first 10 hits from each retrieval arm
- **final_top_k**: first 10 ranked documents/hits
- **no_gold_handling**: reciprocal rank 0.0 and no fabricated gold document
- **precision_ndcg**: not computed: the labels are not exhaustive, so unlabelled non-gold positions cannot be treated as true negatives
- **latency**: wall-clock milliseconds per retrieval call, including query embedding for dense; one warm-up pass precedes timed passes

## 五臂对比

| arm | Recall macro | Recall micro | MRR | Hit@1 | 有 gold 查询 | 无 gold 查询 | 空结果 | P50 ms | P95 ms |
|---|---|---|---|---|---|---|---|---|---|
| dense | 0.944 | 0.884 | 0.917 | 25/30 | 30 | 0 | 0 | 75.062 | 94.624 |
| sparse_or | 1.000 | 1.000 | 0.940 | 27/30 | 30 | 0 | 0 | 3.750 | 5.608 |
| sparse_and | 0.133 | 0.093 | 0.133 | 4/30 | 4 | 26 | 26 | 1.343 | 1.716 |
| candidate_union | 1.000 | 1.000 | 0.917 | 25/30 | 30 | 0 | 0 | 78.733 | 98.924 |
| rrf | 1.000 | 1.000 | 0.950 | 27/30 | 30 | 0 | 0 | 78.714 | 98.862 |

## 逐查询 MRR

| query | gold | dense RR | sparse OR RR | sparse AND RR | union RR | RRF RR | RRF-dense | RRF-sparse |
|---|---|---|---|---|---|---|---|---|
| Q-001 | DOC-0001, DOC-0005 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-002 | DOC-0003 | 1.000 | 0.200 | 0.000 | 1.000 | 0.500 | -0.500 | 0.300 |
| Q-003 | DOC-0005 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-004 | DOC-0016, DOC-0017, DOC-0018 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-005 | DOC-0020, DOC-0021, DOC-0022 | 0.500 | 1.000 | 0.000 | 0.500 | 1.000 | 0.500 | 0.000 |
| Q-006 | DOC-0024, DOC-0025, DOC-0026 | 0.500 | 1.000 | 0.000 | 0.500 | 1.000 | 0.500 | 0.000 |
| Q-007 | DOC-0031 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-008 | DOC-0036 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-009 | DOC-0039, DOC-0040, DOC-0041 | 0.500 | 1.000 | 0.000 | 0.500 | 1.000 | 0.500 | 0.000 |
| Q-010 | DOC-0046 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-011 | DOC-0050, DOC-0051, DOC-0052 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-012 | DOC-0054 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-013 | DOC-0061 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-014 | DOC-0065, DOC-0066, DOC-0067 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-015 | DOC-0065 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-016 | DOC-0076 | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 0.000 | 0.000 |
| Q-017 | DOC-0077 | 0.500 | 1.000 | 1.000 | 0.500 | 0.500 | 0.000 | -0.500 |
| Q-018 | DOC-0094 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-019 | DOC-0058 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-020 | DOC-0099 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-021 | DOC-0037 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-022 | DOC-0043 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-023 | DOC-0057 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-024 | DOC-0073 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-025 | DOC-0024 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-026 | DOC-0061 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-027 | DOC-0106 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-028 | DOC-0149 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Q-029 | DOC-0060 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 0.000 | 0.500 |
| Q-030 | DOC-0039 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |

## 五个 Dense 漏检 gold 的融合排名

| query | doc | dense rank | sparse rank | union rank | RRF rank | RRF score | in RRF Top-K |
|---|---|---|---|---|---|---|---|
| Q-004 | DOC-0017 | - | 3 | 5 | 6 | 0.016 | True |
| Q-005 | DOC-0021 | - | 4 | 6 | 8 | 0.016 | True |
| Q-009 | DOC-0039 | - | 2 | 4 | 5 | 0.016 | True |
| Q-009 | DOC-0040 | - | 1 | 2 | 3 | 0.016 | True |
| Q-014 | DOC-0067 | - | 4 | 6 | 8 | 0.016 | True |

## 候选覆盖

| metric | value |
|---|---|
| queries_with_all_gold_in_dense_candidates | 26 |
| queries_with_all_gold_in_sparse_candidates | 30 |
| queries_with_all_gold_in_union_candidates | 30 |
| queries_with_all_gold_in_rrf_top_k | 30 |
| both_arms | 38 |
| dense_only | 0 |
| sparse_only | 5 |
| neither | 0 |
| queries | 30 |

RRF Top-K 来源：dense_only=77, sparse_only=69, both=154

## 环境与数据

- execution_environment: host Python process
- python_version: 3.14.3
- postgres_version: 16.14 (Debian 16.14-1.pgdg12+1)
- vector_version: 0.8.2
- server_address: 172.22.0.2/32:5432
- documents: 150
- chunks: 150
- distinct_chunk_docs: 150
- embedding_model: BAAI/bge-large-en-v1.5
- embedding_dimensions: [1024]
- ef_search_setting: 40
- hnsw_index: CREATE INDEX chunks_embedding_hnsw_idx ON public.chunks USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='64')
- gin_index: CREATE INDEX chunks_tsv_gin ON public.chunks USING gin (tsv)
- dependency_versions: {'psycopg': '3.3.6', 'psycopg-pool': '3.3.3', 'pgvector': '0.5.1', 'sentence-transformers': '5.5.1', 'torch': '2.13.0', 'transformers': '5.10.2', 'numpy': '2.4.6', 'pydantic': '2.13.4', 'pytest': '9.0.3'}
- dataset: {'queries': 30, 'gold_sizes': [1, 2, 3], 'distinct_gold_docs': 38, 'missing_gold_docs': [], 'hashes': {'kb_documents.json': '7d230e97cbf76d2ca64382e7c0278c3b24bbe27b8a8904abc9553b45a230057d', 'eval_queries.json': '739bd308a90e72437ccbffca658ca950f6a069f2986040cb729694a878d9bbef'}}

## 限制

- The corpus has one chunk per document, so chunk-level and document-level candidate sets coincide here.
- Sparse OR already retrieves all 43 gold documents, so hybrid recall cannot exceed 1.0 on this dataset.
- The 30 queries are English; Chinese and mixed-language hybrid retrieval are not evaluated.
- Precision and nDCG are not reported because non-gold labels are not exhaustive.
- Host execution may differ from the pinned container environment; see environment.dependency_versions.
- This run measures quality and latency at 150 chunks, not scale or concurrency behaviour.

## 执行事实

- evaluated_queries: 30
- missing_arm_queries: 0
- errors: 0
- warmup_errors: 0
- quality_mismatch_samples: 0

原始逐查询与候选结果：[`hybrid_benchmark.json`](hybrid_benchmark.json)
