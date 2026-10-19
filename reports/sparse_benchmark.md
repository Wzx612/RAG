# Sparse Retrieval Benchmark (Phase 5A)

- Generated: `2026-10-17T07:17:11.589849+00:00`
- Status: **PASS**
- Queries: 30  |  K: 10  |  Timed passes: 5  |  Warm-ups: 1  |  Statement timeout: 5000 ms

## 1. Environment and dataset

| Item | Value |
|---|---|
| PostgreSQL | 16.14 (Debian 16.14-1.pgdg12+1) |
| pgvector extension | 0.8.2 |
| Server address | 172.22.0.2/32:5432 |
| Data directory | /var/lib/postgresql/data |
| FTS configuration | english |
| Generated tsv expression | ((setweight(to_tsvector('english'::regconfig, COALESCE(title, ''::text)), 'A'::"char") \|\| setweight(to_tsvector('english'::regconfig, COALESCE(card_tier, ''::text)), 'A'::"char")) \|\| setweight(to_tsvector('english'::regconfig, COALESCE(content, ''::text)), 'B'::"char")) |
| GIN index | CREATE INDEX chunks_tsv_gin ON public.chunks USING gin (tsv) |
| Chunk table | chunk_id, doc_id, chunk_strategy, chunk_index, title, content, embedded_text, embedding, category, card_tier, effective_year, policy_code, tsv |
| Host Python | 3.14.3 |
| psycopg | 3.3.6 |
| Docker control plane | unresponsive (> 5s); no container command executed |
| Execution environment | host Python fallback |
| Free disk | project=110.61GB free; C=7.12GB free; D=110.61GB free |
| Documents / chunks | 150 / 150 |
| Distinct chunk doc_ids | 150 |
| Eval queries | 30 |
| Gold sizes | [1, 2, 3] |
| Missing gold docs | - |
| Data hashes | {'kb_documents.json': '7d230e97cbf76d2ca64382e7c0278c3b24bbe27b8a8904abc9553b45a230057d', 'eval_queries.json': '739bd308a90e72437ccbffca658ca950f6a069f2986040cb729694a878d9bbef'} |
| Counts after run | {'documents': 150, 'chunks': 150, 'distinct_chunk_docs': 150, 'embedding_dimensions': [1024]} |
| Data integrity | True |

The numbers below were produced against the live PostgreSQL/pgvector database, on the execution environment named above. No rows, schema, indexes, eval queries or knowledge-base files were modified.

## 2. Method and score semantics

- Retrieval path counts over timed passes: `{'fts': 150}`
- English queries use `websearch_to_tsquery('english', query)` against the generated `chunks.tsv`, with a GIN index and `ts_rank_cd` ranking.
- Query mode: `or`. Plain terms are OR-joined by default; quoted phrases and `-term` are left untouched, and a bare `or` is treated as prose.
- Rank weights `{D, C, B, A}`: `[0.1, 0.2, 0.4, 1.0]`.
- The English sparse score is **weighted cover density**: higher is better, it is not a probability, and raw values are not comparable across queries.
- The CJK fallback score is **the number of matched two-character substrings**; it is not a segmentation score and is only comparable within one query.
- `distance` is `None` for sparse hits because full-text rank is not a vector distance.
- Semantic Recall uses the human `relevant_doc_ids` labels. It is **not** ANN Recall: sparse retrieval has no vector ANN reference.

## 3. Overall semantic quality

| Metric | Value |
|---|---|
| Evaluated queries | 30 |
| Missing queries | 0 |
| Semantic Recall mean | 1.000 |
| Semantic Recall min | 1.000 |
| Semantic Recall max | 1.000 |
| Perfect-recall queries | 30 |
| Zero-recall queries | 0 |
| Empty-result queries | 0 |
| Empty-result rate | 0.000 |
| Shortfall queries (< K) | 0 |
| Duplicate-result queries | 0 |
| Timed errors | 0 |
| Timed timeouts | 0 |
| Warm-up errors | 0 |
| Quality-mismatch samples | 0 |

Semantic Recall is macro-averaged over evaluated queries; a missing/error query is not silently counted as recall 0.

### 3b. AND-mode control

The same queries with PostgreSQL's bare-term AND semantics. This shows why OR is the default; it is not the primary metric.

| Metric | AND control |
|---|---|
| Evaluated queries | 30 |
| Semantic Recall mean | 0.133 |
| Perfect-recall queries | 4 |
| Zero-recall queries | 26 |
| Empty-result queries | 26 |

## 4. Latency

| Clock | n | min ms | P50 ms | P95 ms | P99 ms | max ms | mean ms |
|---|---|---|---|---|---|---|---|
| query_ms | 150 | 0.880 | 2.054 | 2.698 | 3.918 | 4.356 | 1.980 |
| db_ms | 150 | 1.853 | 3.086 | 3.865 | 5.144 | 5.584 | 3.019 |

P50/P95 use the same nearest-rank definition as Phase 4C. These are descriptive measurements of this run, not a cross-engine speed claim.

## 5. Per-query results

`doc_ids` is deduplicated in rank order. `gold_ranks` maps each retrieved gold doc to its first chunk rank.

| Query | Path | Results | Gold | Semantic Recall | Gold ranks | Top score |
|---|---|---|---|---|---|---|
| Q-001 | fts | 10 | 2 | 1.000 | {'DOC-0001': 1, 'DOC-0005': 2} | 15.600 |
| Q-002 | fts | 10 | 1 | 1.000 | {'DOC-0003': 5} | 8.800 |
| Q-003 | fts | 10 | 1 | 1.000 | {'DOC-0005': 1} | 12.600 |
| Q-004 | fts | 10 | 3 | 1.000 | {'DOC-0016': 1, 'DOC-0018': 2, 'DOC-0017': 3} | 8.000 |
| Q-005 | fts | 10 | 3 | 1.000 | {'DOC-0020': 1, 'DOC-0022': 2, 'DOC-0021': 4} | 4.600 |
| Q-006 | fts | 10 | 3 | 1.000 | {'DOC-0024': 1, 'DOC-0026': 2, 'DOC-0025': 4} | 4.800 |
| Q-007 | fts | 10 | 1 | 1.000 | {'DOC-0031': 1} | 12.400 |
| Q-008 | fts | 10 | 1 | 1.000 | {'DOC-0036': 1} | 13.600 |
| Q-009 | fts | 10 | 3 | 1.000 | {'DOC-0040': 1, 'DOC-0039': 2, 'DOC-0041': 4} | 3.600 |
| Q-010 | fts | 10 | 1 | 1.000 | {'DOC-0046': 1} | 8.000 |
| Q-011 | fts | 10 | 3 | 1.000 | {'DOC-0050': 1, 'DOC-0052': 2, 'DOC-0051': 3} | 8.800 |
| Q-012 | fts | 10 | 1 | 1.000 | {'DOC-0054': 1} | 9.800 |
| Q-013 | fts | 10 | 1 | 1.000 | {'DOC-0061': 1} | 11.200 |
| Q-014 | fts | 10 | 3 | 1.000 | {'DOC-0065': 1, 'DOC-0066': 2, 'DOC-0067': 4} | 7.400 |
| Q-015 | fts | 10 | 1 | 1.000 | {'DOC-0065': 1} | 8.600 |
| Q-016 | fts | 10 | 1 | 1.000 | {'DOC-0076': 2} | 10.200 |
| Q-017 | fts | 10 | 1 | 1.000 | {'DOC-0077': 1} | 13.000 |
| Q-018 | fts | 10 | 1 | 1.000 | {'DOC-0094': 1} | 9.600 |
| Q-019 | fts | 10 | 1 | 1.000 | {'DOC-0058': 1} | 9.200 |
| Q-020 | fts | 10 | 1 | 1.000 | {'DOC-0099': 1} | 10.400 |
| Q-021 | fts | 10 | 1 | 1.000 | {'DOC-0037': 1} | 14.600 |
| Q-022 | fts | 10 | 1 | 1.000 | {'DOC-0043': 1} | 12.800 |
| Q-023 | fts | 10 | 1 | 1.000 | {'DOC-0057': 1} | 12.600 |
| Q-024 | fts | 10 | 1 | 1.000 | {'DOC-0073': 1} | 12.000 |
| Q-025 | fts | 10 | 1 | 1.000 | {'DOC-0024': 1} | 5.600 |
| Q-026 | fts | 10 | 1 | 1.000 | {'DOC-0061': 1} | 8.200 |
| Q-027 | fts | 10 | 1 | 1.000 | {'DOC-0106': 1} | 9.200 |
| Q-028 | fts | 10 | 1 | 1.000 | {'DOC-0149': 1} | 12.200 |
| Q-029 | fts | 10 | 1 | 1.000 | {'DOC-0060': 2} | 11.000 |
| Q-030 | fts | 10 | 1 | 1.000 | {'DOC-0039': 1} | 10.000 |

## 6. Focus queries (dense semantic Recall < 1.0 in Phase 4C)

| Query | Sparse Recall | Results | Gold ranks | Doc IDs |
|---|---|---|---|---|
| Q-004 | 1.000 | 10 | {'DOC-0016': 1, 'DOC-0018': 2, 'DOC-0017': 3} | DOC-0016, DOC-0018, DOC-0017, DOC-0019, DOC-0030, DOC-0028, DOC-0029, DOC-0084, DOC-0025, DOC-0027 |
| Q-005 | 1.000 | 10 | {'DOC-0020': 1, 'DOC-0022': 2, 'DOC-0021': 4} | DOC-0020, DOC-0022, DOC-0023, DOC-0021, DOC-0026, DOC-0130, DOC-0024, DOC-0025, DOC-0027, DOC-0008 |
| Q-009 | 1.000 | 10 | {'DOC-0040': 1, 'DOC-0039': 2, 'DOC-0041': 4} | DOC-0040, DOC-0039, DOC-0009, DOC-0041, DOC-0044, DOC-0147, DOC-0010, DOC-0012, DOC-0140, DOC-0141 |
| Q-014 | 1.000 | 10 | {'DOC-0065': 1, 'DOC-0066': 2, 'DOC-0067': 4} | DOC-0065, DOC-0066, DOC-0114, DOC-0067, DOC-0110, DOC-0068, DOC-0150, DOC-0106, DOC-0107, DOC-0108 |

## 7. Sparse vs Dense candidate overlap

- Source: `D:\信用卡知识库 RAG 检索评估\reports\dense_benchmark.json`, plan `natural_plan`
- Top-K aligned: True (dense=10, sparse=10)
- Compared queries: 30
- Mean chunk Jaccard: 0.362
- Mean doc Jaccard: 0.362
- Gold-doc coverage: both=38, sparse-only=5, dense-only=0, neither=0

| Query | Chunk overlap | Sparse-only | Dense-only | Chunk Jaccard | Doc Jaccard |
|---|---|---|---|---|---|
| Q-001 | 8 | 2 | 2 | 0.667 | 0.667 |
| Q-002 | 8 | 2 | 2 | 0.667 | 0.667 |
| Q-003 | 6 | 4 | 4 | 0.429 | 0.429 |
| Q-004 | 5 | 5 | 5 | 0.333 | 0.333 |
| Q-005 | 7 | 3 | 3 | 0.538 | 0.538 |
| Q-006 | 5 | 5 | 5 | 0.333 | 0.333 |
| Q-007 | 5 | 5 | 5 | 0.333 | 0.333 |
| Q-008 | 7 | 3 | 3 | 0.538 | 0.538 |
| Q-009 | 2 | 8 | 8 | 0.111 | 0.111 |
| Q-010 | 3 | 7 | 7 | 0.176 | 0.176 |
| Q-011 | 5 | 5 | 5 | 0.333 | 0.333 |
| Q-012 | 4 | 6 | 6 | 0.250 | 0.250 |
| Q-013 | 4 | 6 | 6 | 0.250 | 0.250 |
| Q-014 | 5 | 5 | 5 | 0.333 | 0.333 |
| Q-015 | 7 | 3 | 3 | 0.538 | 0.538 |
| Q-016 | 5 | 5 | 5 | 0.333 | 0.333 |
| Q-017 | 7 | 3 | 3 | 0.538 | 0.538 |
| Q-018 | 6 | 4 | 4 | 0.429 | 0.429 |
| Q-019 | 3 | 7 | 7 | 0.176 | 0.176 |
| Q-020 | 3 | 7 | 7 | 0.176 | 0.176 |
| Q-021 | 5 | 5 | 5 | 0.333 | 0.333 |
| Q-022 | 5 | 5 | 5 | 0.333 | 0.333 |
| Q-023 | 4 | 6 | 6 | 0.250 | 0.250 |
| Q-024 | 3 | 7 | 7 | 0.176 | 0.176 |
| Q-025 | 5 | 5 | 5 | 0.333 | 0.333 |
| Q-026 | 3 | 7 | 7 | 0.176 | 0.176 |
| Q-027 | 6 | 4 | 4 | 0.429 | 0.429 |
| Q-028 | 5 | 5 | 5 | 0.333 | 0.333 |
| Q-029 | 5 | 5 | 5 | 0.333 | 0.333 |
| Q-030 | 8 | 2 | 2 | 0.667 | 0.667 |

Candidate overlap is not a quality metric: two lists can overlap heavily and both be wrong, or overlap little and both be right.

## 8. Chinese probe (separate from the 30-query metrics)

These are synthetic Chinese probes, not part of the labelled 30-query set. The corpus is English, so zero hits do **not** show that Chinese retrieval works; they show the fallback's behaviour on this corpus.

| Probe | Path | Results | English FTS matches | Top doc IDs | Top scores |
|---|---|---|---|---|---|
| 余额转账手续费 | cjk | 0 | 0 | - | - |
| 信用卡年费减免政策 | cjk | 0 | 0 | - | - |
| 境外消费返现 | cjk | 0 | 0 | - | - |
| 信用 | cjk | 0 | 0 | - | - |
| balance transfer fee 余额 | cjk | 0 | 0 | - | - |

- English FTS lexeme evidence: `{'text': "'余额转账手续费':1", 'query': "'余额转账手续费'"}`
- CJK bigram example: `['余额', '额转', '转账', '账手', '手续', '续费']`
- The fallback is substring matching over title/content/card_tier. It is not word segmentation, has no Chinese dictionary, and was not validated against Chinese documents because none are in this corpus.

## 9. Errors and integrity

- Timed errors: 0
- Timed timeouts: 0
- Warm-up errors: 0
- Quality-mismatch samples: 0
- Post-run data integrity: True
- Issues: -

## 10. Limitations and next-stage prerequisites

- Chinese support is a minimal substring fallback, not segmentation. A real Chinese corpus needs `zhparser`/`pg_jieba`, a matching generated tsvector, and a rebuilt index before Chinese retrieval quality can be claimed.
- A mixed Chinese+ASCII query currently takes the CJK path only; ASCII terms are not combined with the fallback. The probe `balance transfer fee 余额` therefore returns zero rows on this English corpus.
- The OR default is a recall-oriented choice, not a universal one: the AND control returned 26 empty results out of 30. The mode is recorded in `parameters.query_mode` so later stages can reproduce the exact run.
- The evaluation set has 30 queries and gold docs; Semantic Recall values are sensitive to those labels and are not a generalisation guarantee.
- The sparse score is not calibrated across queries, so it must not be compared with dense cosine similarity or fed to a weighted fusion without normalisation or rank-based fusion (RRF).
- Hybrid Search, RRF and reranking are intentionally **not** implemented in this phase.
