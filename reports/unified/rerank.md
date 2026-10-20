# Reranker 实现与离线评测（阶段 ⑦-B）

- Status: **PASS**
- Generated: 2026-10-20T03:21:25.110163+00:00
- Metric version / labels_exhaustive / relevance_scheme: 2 / False / binary_known_gold
- Queries: 30
- Candidate pool / final Top-K: 10 / 10
- RRF k: 60
- Model: `BAAI/bge-reranker-base` @ `2cfc18c9415c912f9d8155881c133215df768a70`
- Batch size / max length / activation: 16 / 512 / identity

## 1. 结论摘要

| arm | source | Recall macro | Recall micro | MRR | Hit@1 | P50 ms | P95 ms | P99 ms |
|---|---|---|---|---|---|---|---|---|
| dense | historical_⑥-B | 0.9444 | 0.8837 | 0.9167 | 25/30 | 75.0620 | 94.6243 | - |
| sparse_or | historical_⑥-B | 1.0000 | 1.0000 | 0.9400 | 27/30 | 3.7502 | 5.6076 | - |
| sparse_and | historical_⑥-B | 0.1333 | 0.0930 | 0.1333 | 4/30 | 1.3431 | 1.7162 | - |
| rrf | historical_⑥-B | 1.0000 | 1.0000 | 0.9500 | 27/30 | 78.7144 | 98.8624 | - |
| rrf_reranker | live_rerank_over_historical_rrf_top10 | 1.0000 | 1.0000 | 0.9833 | 29/30 | 710.3390 | 852.2723 | 880.7805 |

RRF 的 Recall 为 1.0000，候选池完全相同，因此 Recall 不可能因重排而上升；本阶段只判断 MRR 与 Hit@1 是否改变。若 MRR 没有提升，报告如实记录。

## 2. 固定条件与评测口径

- **candidate_granularity**: chunk identity (chunk_strategy, chunk_id)
- **metric_granularity**: document; distinct doc_id in ranked order, truncated to top_k
- **semantic_recall**: recall_at_k against human relevant_doc_ids; denominator is the number of distinct gold docs, not K; not ANN recall
- **mrr**: mean reciprocal rank of the first gold doc at doc rank r <= top_k; no gold in Top-K contributes 0; denominator is evaluated queries
- **rrf**: sum over arms of 1 / (k + rank), rank starts at 1, k=60; raw dense/sparse scores are never added
- **reranker_pool**: the historical ⑥-B RRF Top-10 candidate identities for the same query; text is fetched live from PostgreSQL; the dense and sparse arms are not re-run
- **reranker_score**: raw cross-encoder logit with activation=identity; higher is better; only relative order is meaningful and it is not comparable with cosine similarity or ts_rank_cd
- **reranker_order**: rerank_score descending, then pre-rerank rank, then (chunk_strategy, chunk_id)
- **final_top_k**: first 10 ranked documents/hits
- **no_gold_handling**: reciprocal rank 0.0 and no fabricated gold document
- **precision_at_k**: |distinct doc_id in top_k ∩ known gold| / k; the denominator is always k, even when fewer than k documents are returned; duplicate doc_ids are collapsed in rank order before the cutoff; supplementary because labels are not exhaustive
- **precision_at_k_ceiling**: min(distinct gold doc count, k) / k per query; the maximum Precision@K reachable on that query, not an observed ranking score
- **ndcg_at_k**: binary nDCG: gain 1 for a known gold doc_id and 0 otherwise; DCG uses 1 / log2(rank + 1); IDCG uses the same k cutoff over the full known gold set; non-gold ids are unlabelled, not asserted irrelevant
- **labels_exhaustive**: False; non-gold ids have not been judged irrelevant
- **relevance_scheme**: binary_known_gold; only existing gold ids receive relevance 1
- **metric_version**: 2

- 历史 artifact: `2026-10-17T07:49:52.438512+00:00`，status `PASS`
- 历史四臂独立复核: **PASS**，mismatches=0，RRF formula checked=300，formula mismatches=0
- RRF Top-K 覆盖全部 gold 的查询数: 30/30
- 数据哈希: {'kb_documents.json': '7d230e97cbf76d2ca64382e7c0278c3b24bbe27b8a8904abc9553b45a230057d', 'eval_queries.json': '739bd308a90e72437ccbffca658ca950f6a069f2986040cb729694a878d9bbef'}
- 缺失 gold: []

## 3. 模型、来源与资源

| item | value |
|---|---|
| model id | BAAI/bge-reranker-base |
| revision | 2cfc18c9415c912f9d8155881c133215df768a70 |
| license | MIT |
| license source | Hugging Face model card checked during 7-B preflight |
| commercial use | model card states commercial use is free of charge |
| architecture | XLM-RoBERTa base, 12 layers, hidden 768, 278,044,931 parameters |
| device | cpu |
| batch size | 16 |
| max length | 512 |
| activation | identity |
| local files only | True |
| cache folder | /app/.cache/huggingface/hub |
| model load ms | 5444.4 |
| memory before load | {'rss_mib': 71.6, 'peak_wset_mib': None, 'system_available_gib': 13.995, 'timestamp': '2026-10-20T03:21:25.119782+00:00'} |
| memory after load | {'rss_mib': 813.6, 'peak_wset_mib': None, 'system_available_gib': 13.36, 'timestamp': '2026-10-20T03:21:30.564468+00:00'} |
| memory peak | {'rss_mib': 1233.7, 'peak_wset_mib': None, 'system_available_gib_min': 13.357} |
| runtime | {'python_version': '3.12.13', 'platform': 'Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.41', 'torch': '2.14.1+cpu', 'transformers': '5.20.0', 'sentence_transformers': '6.1.0', 'tokenizers': '0.23.3', 'huggingface_hub': '1.33.0', 'safetensors': '0.8.0', 'numpy': '2.5.4', 'psutil': '7.2.2', 'torch_threads': 4, 'torch_interop_threads': 4, 'cuda_available': False} |

本地模型文件（实测大小与 SHA256）：

| file | bytes | sha256 |
|---|---|---|
| config.json | 799 | 289adf7ada1eb6b4afa7589a48a032d45a076cf2e46dcdb3b4cabc33be14f708 |
| model.safetensors | 1112206140 | ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd |
| sentencepiece.bpe.model | 5069051 | cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865 |
| tokenizer.json | 17098107 | 9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e |
| tokenizer_config.json | 443 | a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b |

## 4. 最小推理与分数方向

| query | status | gold mean | non-gold mean | direction ok | all gold > all non-gold | repeat max diff | reverse map diff | empty -> [] | invalid query rejected | overlong finite |
|---|---|---|---|---|---|---|---|---|---|---|
| Q-001 | ok | 6.5038 | 1.4236 | True | False | 0.0 | 0.0 | True | True | True |
| Q-004 | ok | 5.1173 | -0.9785 | True | False | 0.0 | 0.0 | True | True | True |
| Q-014 | ok | 2.7634 | -5.7744 | True | False | 0.0 | 0.0 | True | True | True |

Scores are raw cross-encoder logits; only relative order is meaningful. Gold means the human relevant_doc_ids, not an exhaustive relevance judgement.

## 5. 延迟测试（10/50 候选，batch 1/16）

| query | candidates | batch | status | n | mean ms | min ms | max ms | ms/candidate | scores finite |
|---|---|---|---|---|---|---|---|---|---|
| Q-001 | 10 | 1 | ok | 2 | 770.3 | 753.0 | 787.6 | 77.03 | True |
| Q-001 | 10 | 16 | ok | 2 | 934.2 | 925.3 | 943.0 | 93.42 | True |
| Q-001 | 50 | 1 | ok | 2 | 3858.6 | 3849.6 | 3867.6 | 77.17 | True |
| Q-001 | 50 | 16 | ok | 2 | 3555.1 | 3524.6 | 3585.5 | 71.10 | True |

- pool: live sparse OR Top-50; latency fixture only, not a quality arm
- batch invariance max abs diff: {'10': 2.86102294921875e-06, '50': 4.0531158447265625e-06}
- One warm-up call precedes the timed calls for each combination. CPU-only, single process, sequential; no concurrency claim.

## 6. 五臂对比

| arm | evaluated | Recall macro | Recall micro | MRR | Hit@1 | MRR vs RRF | source |
|---|---|---|---|---|---|---|---|
| dense | 30 | 0.9444 | 0.8837 | 0.9167 | 25/30 | - | historical_⑥-B |
| sparse_or | 30 | 1.0000 | 1.0000 | 0.9400 | 27/30 | - | historical_⑥-B |
| sparse_and | 30 | 0.1333 | 0.0930 | 0.1333 | 4/30 | - | historical_⑥-B |
| rrf | 30 | 1.0000 | 1.0000 | 0.9500 | 27/30 | - | historical_⑥-B |
| rrf_reranker | 30 | 1.0000 | 1.0000 | 0.9833 | 29/30 | 0.0333 | live_rerank_over_historical_rrf_top10 |

- RRF vs RRF+Reranker MRR 变化：improved=3，equal=26，worse=1，mean delta=0.0333
- first-gold-rank mean delta: -0.0667 （负值表示排名提前）
- Top-1 改变的查询数: 6
- Recall 保持不变的查询数: 30/30
- 排序发生变化的查询: Q-002, Q-011, Q-016, Q-017

### 补充指标：Precision@K / 二元 nDCG@K

- labels_exhaustive=False; relevance_scheme=binary_known_gold; metric_version=2
- 非 gold 文档只是未标注，不是已确认无关；P@K ceiling 是该 Query 在当前 gold 数量与 K 下的理论上限。

| arm | P@K macro | P@K micro | P@K ceiling macro | nDCG@K macro |
|---|---|---|---|---|
| dense | 0.1267 | 0.1267 | 0.1433 | 0.9004 |
| sparse_or | 0.1433 | 0.1433 | 0.1433 | 0.9506 |
| sparse_and | 0.0133 | 0.0133 | 0.1433 | 0.1333 |
| rrf | 0.1433 | 0.1433 | 0.1433 | 0.9429 |
| rrf_reranker | 0.1433 | 0.1433 | 0.1433 | 0.9777 |

## 7. 逐查询结果

| query | gold | RRF RR | rerank RR | delta RR | RRF first gold | rerank first gold | delta rank | RRF Hit@1 | rerank Hit@1 | rerank top-1 doc | fallback |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Q-001 | DOC-0001, DOC-0005 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0001 | False |
| Q-002 | DOC-0003 | 0.5000 | 1.0000 | 0.5000 | 2 | 1 | -1 | False | True | DOC-0003 | False |
| Q-003 | DOC-0005 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0005 | False |
| Q-004 | DOC-0016, DOC-0017, DOC-0018 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0018 | False |
| Q-005 | DOC-0020, DOC-0021, DOC-0022 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0020 | False |
| Q-006 | DOC-0024, DOC-0025, DOC-0026 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0024 | False |
| Q-007 | DOC-0031 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0031 | False |
| Q-008 | DOC-0036 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0036 | False |
| Q-009 | DOC-0039, DOC-0040, DOC-0041 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0040 | False |
| Q-010 | DOC-0046 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0046 | False |
| Q-011 | DOC-0050, DOC-0051, DOC-0052 | 1.0000 | 0.5000 | -0.5000 | 1 | 2 | 1 | True | False | DOC-0053 | False |
| Q-012 | DOC-0054 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0054 | False |
| Q-013 | DOC-0061 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0061 | False |
| Q-014 | DOC-0065, DOC-0066, DOC-0067 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0065 | False |
| Q-015 | DOC-0065 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0065 | False |
| Q-016 | DOC-0076 | 0.5000 | 1.0000 | 0.5000 | 2 | 1 | -1 | False | True | DOC-0076 | False |
| Q-017 | DOC-0077 | 0.5000 | 1.0000 | 0.5000 | 2 | 1 | -1 | False | True | DOC-0077 | False |
| Q-018 | DOC-0094 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0094 | False |
| Q-019 | DOC-0058 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0058 | False |
| Q-020 | DOC-0099 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0099 | False |
| Q-021 | DOC-0037 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0037 | False |
| Q-022 | DOC-0043 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0043 | False |
| Q-023 | DOC-0057 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0057 | False |
| Q-024 | DOC-0073 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0073 | False |
| Q-025 | DOC-0024 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0024 | False |
| Q-026 | DOC-0061 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0061 | False |
| Q-027 | DOC-0106 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0106 | False |
| Q-028 | DOC-0149 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0149 | False |
| Q-029 | DOC-0060 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0060 | False |
| Q-030 | DOC-0039 | 1.0000 | 1.0000 | 0.0000 | 1 | 1 | 0 | True | True | DOC-0039 | False |

## 8. 排序分析

重排只改变顺序，不改变候选集合。以下按 MRR 变化列出每个查询；未列出的查询表示 MRR 未变。所有 30 条结果都保留在原始 JSON 中，没有只展示改善的查询。

| query | RRF RR | rerank RR | delta | RRF top-1 | rerank top-1 |
|---|---|---|---|---|---|
| Q-002 | 0.5000 | 1.0000 | 0.5000 | DOC-0014 | DOC-0003 |
| Q-011 | 1.0000 | 0.5000 | -0.5000 | DOC-0050 | DOC-0053 |
| Q-016 | 0.5000 | 1.0000 | 0.5000 | DOC-0031 | DOC-0076 |
| Q-017 | 0.5000 | 1.0000 | 0.5000 | DOC-0033 | DOC-0077 |

## 9. 错误、fallback 与资源限制

- errors: 0
- fallbacks: 0 （实现没有静默 fallback；失败即抛错）
- model load errors: 0
- inference errors: 0
- input errors: 0
- db fetch errors: 0
- resource errors: 0
- aborted: False

端到端耗时：model load=5444.4 ms，evaluation loop=21565.7 ms，total=63715.6 ms。
统计口径：30 timed queries, one deterministic rerank pass each; nearest-rank percentiles; no confidence interval; model load and warm-up are excluded from per-query latency

## 10. 未验证事项

- The four arms (dense, sparse_or, sparse_and, rrf) are reused from the Phase 6B artifact, not re-run against PostgreSQL in Phase 7B. Their raw hits are independently re-scored and the RRF formula is re-checked, but this is offline replay.
- The RRF+Reranker arm reranks the historical RRF Top-10 identities; those candidates are not regenerated by a live dense/sparse query in this run.
- Recall cannot increase over RRF because the reranker sees the same Top-10 pool; only ordering metrics (MRR, Hit@1) can change.
- The corpus has one chunk per document, so chunk-level and document-level candidate sets coincide here.
- The 30 queries are English; Chinese and mixed-language reranking are not evaluated.
- The embedding model and the reranker were not loaded in the same process; combined-memory behaviour is not measured, so this run does not claim to solve it.
- Host Python execution may differ from the pinned container environment; Docker was not used for this run and the container path is unverified.
- The full pytest suite was not run because the host torch/transformers stack has a known native crash on an embedding integration test; only module-level tests ran.
- Dependency versions on the host differ from the exact pins in pyproject.toml; that drift is not resolved or verified here.
- Latency is CPU-only, single-process, sequential and warm. No GPU, scale, concurrency or cross-machine reproducibility claim is made.
- Precision@K and binary nDCG@K are supplementary and label-limited: non-gold ids are unlabelled, not confirmed irrelevant.
- A database failure was exercised with unit-level connection stubs, not by stopping the live PostgreSQL server during this benchmark.

## 11. 复现

Host PowerShell, from the repository root:

```powershell
$env:APEX_PG_HOST='127.0.0.1'; $env:APEX_PG_PORT='5434';
python scripts/benchmark_rerank.py --top-k 10 --candidate-k 10 --rrf-k 60 --batch-size 16 --activation identity
```

Smoke test without writing reports:

```powershell
python scripts/benchmark_rerank.py --limit 2 --no-latency-probe --no-write
```

## 12. 数据安全

本 benchmark 只读：SELECT 查询、内存中的重排、模型推理。没有写入数据库、没有修改 `data/kb_documents.json`、`data/eval_queries.json` 或 gold 标签，没有重建索引或更换 Embedding。原始逐查询与候选结果见 [`rerank_benchmark.json`](rerank_benchmark.json)。
