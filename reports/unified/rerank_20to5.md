# 统一检索评测

- Status: **PASS**
- Reason: all arms evaluated every expected query
- Generated: 2026-10-20T03:59:49.129746+00:00
- Queries: 30 / expected 30
- Top-K: 5；candidate_k: 20；RRF k: 60
- Chunk strategy: `document`

> Recall/P@K/nDCG/MRR 都按文档级计算（chunk 列表按 doc_id 去重）。
> Gold 不是穷尽标注：非 gold 位置只是未标注，不是 true negative。

## 1. Arm 指标

| Arm | Status | Recall macro | Recall micro | P@K macro | P@K ceiling | nDCG@K | MRR | Hit@1 | Evaluated |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rrf | PASS | 0.9556 | 0.9070 | 0.2600 | 0.2867 | 0.9204 | 0.9500 | 27/30 | 30 |
| rrf_rerank | PASS | 1.0000 | 1.0000 | 0.2867 | 0.2867 | 0.9777 | 0.9833 | 29/30 | 30 |

## 2. 延迟

| Arm | n | P50 ms | P95 ms | P99 ms | Max ms |
|---|---:|---:|---:|---:|---:|
| rrf | 30 | 89.4622 | 109.5530 | 112.8697 | 112.8697 |
| rrf_rerank | 30 | 1790.9155 | 2038.5050 | 4683.3748 | 4683.3748 |

## 3. 失败与限制

- 没有失败的 Query。

- 指标只衡量检索，不代表答案正确性或生成质量。
- `Precision@K` 的分母固定为 K，且 gold 只有 1–3 篇；绝对值会偏低，应同时看 ceiling 并只做横向比较。
- 延迟是单进程、顺序、已预热的运行；不含并发、GPU 或跨机器。
