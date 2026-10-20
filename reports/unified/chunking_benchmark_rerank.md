# 分块策略对比

- Status: **PASS**
- Reason: every strategy evaluated every expected query
- Generated: 2026-10-20T03:24:39.510585+00:00
- Queries: 30 / expected 30；Top-K: 10；candidate_k: 10
- Strategies: `document`, `fixed_window`, `structured`

> 指标按文档级计算：同一文档的多个 chunk 先归并为一个排名，再计算 Recall/MRR/nDCG。
> Gold 不是穷尽标注；非 gold 位置只是未标注，不是 true negative。

## 1. 分块统计

| Strategy | Chunks | Documents | Avg chars | Min chars | Max chars |
|---|---:|---:|---:|---:|---:|
| document | 150 | 150 | 291.6 | 220 | 540 |
| fixed_window | 267 | 150 | 191.7 | 65 | 256 |
| structured | 161 | 150 | 133.0 | 73 | 276 |

## 2. 检索指标

| Strategy | Arm | Status | Recall macro | Recall micro | P@K | P@K ceiling | nDCG@K | MRR | Hit@1 | Evaluated |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| document | dense | PASS | 0.9444 | 0.8837 | 0.1267 | 0.1433 | 0.9004 | 0.9167 | 25/30 | 30 |
| document | sparse_or | PASS | 1.0000 | 1.0000 | 0.1433 | 0.1433 | 0.9506 | 0.9400 | 27/30 | 30 |
| document | rrf | PASS | 1.0000 | 1.0000 | 0.1433 | 0.1433 | 0.9429 | 0.9500 | 27/30 | 30 |
| document | rrf_rerank | PASS | 1.0000 | 1.0000 | 0.1433 | 0.1433 | 0.9777 | 0.9833 | 29/30 | 30 |
| fixed_window | dense | PASS | 0.9889 | 0.9767 | 0.1400 | 0.1433 | 0.8731 | 0.8589 | 23/30 | 30 |
| fixed_window | sparse_or | PASS | 1.0000 | 1.0000 | 0.1433 | 0.1433 | 0.9448 | 0.9389 | 27/30 | 30 |
| fixed_window | rrf | PASS | 1.0000 | 1.0000 | 0.1433 | 0.1433 | 0.9247 | 0.9122 | 26/30 | 30 |
| fixed_window | rrf_rerank | PASS | 1.0000 | 1.0000 | 0.1433 | 0.1433 | 0.9607 | 0.9500 | 27/30 | 30 |
| structured | dense | PASS | 0.9389 | 0.9302 | 0.1333 | 0.1433 | 0.8129 | 0.7872 | 21/30 | 30 |
| structured | sparse_or | PASS | 1.0000 | 1.0000 | 0.1433 | 0.1433 | 0.9763 | 0.9733 | 29/30 | 30 |
| structured | rrf | PASS | 1.0000 | 1.0000 | 0.1433 | 0.1433 | 0.8992 | 0.8778 | 24/30 | 30 |
| structured | rrf_rerank | PASS | 1.0000 | 1.0000 | 0.1433 | 0.1433 | 0.9596 | 0.9500 | 27/30 | 30 |

## 3. 延迟

| Strategy | Arm | n | P50 ms | P95 ms | P99 ms | Max ms |
|---|---|---:|---:|---:|---:|---:|
| document | dense | 30 | 78.6434 | 131.6010 | 135.9906 | 135.9906 |
| document | sparse_or | 30 | 2.2084 | 3.7804 | 4.1723 | 4.1723 |
| document | rrf | 30 | 81.8028 | 88.9834 | 99.6815 | 99.6815 |
| document | rrf_rerank | 30 | 792.6168 | 919.4077 | 2497.2251 | 2497.2251 |
| fixed_window | dense | 30 | 78.5577 | 86.9108 | 92.3192 | 92.3192 |
| fixed_window | sparse_or | 30 | 3.0544 | 3.9684 | 4.0631 | 4.0631 |
| fixed_window | rrf | 30 | 83.0479 | 95.7101 | 111.4960 | 111.4960 |
| fixed_window | rrf_rerank | 30 | 617.2422 | 674.3213 | 711.2702 | 711.2702 |
| structured | dense | 30 | 84.4717 | 117.5184 | 191.4009 | 191.4009 |
| structured | sparse_or | 30 | 3.0504 | 5.9012 | 6.9236 | 6.9236 |
| structured | rrf | 30 | 97.3334 | 111.5498 | 132.9805 | 132.9805 |
| structured | rrf_rerank | 30 | 511.6815 | 616.1759 | 630.9877 | 630.9877 |

## 4. 限制

- 指标只衡量检索，不代表答案正确性或生成质量。
- 分块策略改变候选粒度与数量；延迟差异部分来自 chunk 数量，而不只是策略本身。
- 所有策略共用同一个 HNSW bootstrap 索引与查询参数；本实验不等同于索引调优。
- P@K 的分母固定为 K，且 gold 只有 1–3 篇；应与 ceiling 一起横向比较。
- 预期 Query 数为 30；少于该数的运行会被标为 PARTIAL。
