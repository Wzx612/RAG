# 索引参数实验（HNSW / IVFFlat）

- Status: **PASS**
- Reason: every index configuration completed
- Generated: 2026-10-20T04:04:26.002123+00:00
- Top-K: 10；查询数: 30；chunk strategy: `document`
- Exact reference: seqscan；queries: 30
- Synthetic scale: 10（**SYNTHETIC**，仅用于索引行为）

> 每个配置在临时表副本上重建索引，不修改 `chunks` 表及其 bootstrap 索引。
> ANN Recall 是相对精确 Seq Scan Top-K 的保真度，不是语义召回。

## 1. 总览

| Method | Params | Query setting | Build ms | Size | Rows | ANN Recall@K | P50 ms | P95 ms | P99 ms | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| hnsw | m=8, ef_construction=64 | ef_search=10 | 90.801 | 1208 kB | 1500 | 1.000 | 0.265 | 0.368 | 0.441 | PASS |
| hnsw | m=8, ef_construction=64 | ef_search=40 | 90.801 | 1208 kB | 1500 | 1.000 | 0.237 | 0.391 | 0.805 | PASS |
| hnsw | m=8, ef_construction=64 | ef_search=100 | 90.801 | 1208 kB | 1500 | 1.000 | 0.332 | 0.422 | 0.540 | PASS |
| hnsw | m=8, ef_construction=128 | ef_search=10 | 118.507 | 1208 kB | 1500 | 1.000 | 0.246 | 0.393 | 0.467 | PASS |
| hnsw | m=8, ef_construction=128 | ef_search=40 | 118.507 | 1208 kB | 1500 | 1.000 | 0.323 | 0.461 | 0.943 | PASS |
| hnsw | m=8, ef_construction=128 | ef_search=100 | 118.507 | 1208 kB | 1500 | 1.000 | 0.353 | 0.502 | 0.596 | PASS |
| hnsw | m=16, ef_construction=64 | ef_search=10 | 85.040 | 1208 kB | 1500 | 1.000 | 0.220 | 0.379 | 0.553 | PASS |
| hnsw | m=16, ef_construction=64 | ef_search=40 | 85.040 | 1208 kB | 1500 | 1.000 | 0.286 | 0.450 | 0.711 | PASS |
| hnsw | m=16, ef_construction=64 | ef_search=100 | 85.040 | 1208 kB | 1500 | 1.000 | 0.349 | 0.492 | 0.804 | PASS |
| hnsw | m=16, ef_construction=128 | ef_search=10 | 135.600 | 1208 kB | 1500 | 1.000 | 0.315 | 0.439 | 0.494 | PASS |
| hnsw | m=16, ef_construction=128 | ef_search=40 | 135.600 | 1208 kB | 1500 | 1.000 | 0.359 | 0.473 | 0.498 | PASS |
| hnsw | m=16, ef_construction=128 | ef_search=100 | 135.600 | 1208 kB | 1500 | 1.000 | 0.371 | 0.477 | 0.601 | PASS |
| hnsw | m=32, ef_construction=64 | ef_search=10 | 478.880 | 12 MB | 1500 | 0.797 | 0.217 | 0.355 | 0.422 | PASS |
| hnsw | m=32, ef_construction=64 | ef_search=40 | 478.880 | 12 MB | 1500 | 0.897 | 0.289 | 0.467 | 0.481 | PASS |
| hnsw | m=32, ef_construction=64 | ef_search=100 | 478.880 | 12 MB | 1500 | 0.963 | 0.417 | 0.598 | 0.756 | PASS |
| hnsw | m=32, ef_construction=128 | ef_search=10 | 147.636 | 1208 kB | 1500 | 1.000 | 0.222 | 0.328 | 0.430 | PASS |
| hnsw | m=32, ef_construction=128 | ef_search=40 | 147.636 | 1208 kB | 1500 | 1.000 | 0.260 | 0.465 | 0.901 | PASS |
| hnsw | m=32, ef_construction=128 | ef_search=100 | 147.636 | 1208 kB | 1500 | 1.000 | 0.321 | 0.467 | 0.862 | PASS |
| ivfflat | lists=1 | probes=1 | 43.527 | 12 MB | 1500 | 1.000 | 1.886 | 2.437 | 2.623 | PASS |
| ivfflat | lists=1 | probes=2 | 43.527 | 12 MB | 1500 | 1.000 | 2.163 | 2.733 | 3.749 | PASS |
| ivfflat | lists=1 | probes=4 | 43.527 | 12 MB | 1500 | 1.000 | 1.947 | 2.525 | 3.124 | PASS |
| ivfflat | lists=2 | probes=1 | 40.614 | 12 MB | 1500 | 0.967 | 1.574 | 2.081 | 2.383 | PASS |
| ivfflat | lists=2 | probes=2 | 40.614 | 12 MB | 1500 | 1.000 | 1.862 | 2.271 | 2.429 | PASS |
| ivfflat | lists=2 | probes=4 | 40.614 | 12 MB | 1500 | 1.000 | 1.765 | 2.229 | 2.512 | PASS |
| ivfflat | lists=4 | probes=1 | 45.899 | 12 MB | 1500 | 0.800 | 0.273 | 0.547 | 0.888 | PASS |
| ivfflat | lists=4 | probes=2 | 45.899 | 12 MB | 1500 | 1.000 | 0.425 | 0.790 | 1.295 | PASS |
| ivfflat | lists=4 | probes=4 | 45.899 | 12 MB | 1500 | 1.000 | 1.877 | 2.382 | 2.804 | PASS |
| ivfflat | lists=8 | probes=1 | 47.213 | 12 MB | 1500 | 0.800 | 0.293 | 0.510 | 0.715 | PASS |
| ivfflat | lists=8 | probes=2 | 47.213 | 12 MB | 1500 | 0.833 | 0.378 | 0.616 | 0.870 | PASS |
| ivfflat | lists=8 | probes=4 | 47.213 | 12 MB | 1500 | 0.967 | 0.518 | 0.868 | 1.144 | PASS |
| ivfflat | lists=16 | probes=1 | 47.837 | 12 MB | 1500 | 0.733 | 0.210 | 0.345 | 0.442 | PASS |
| ivfflat | lists=16 | probes=2 | 47.837 | 12 MB | 1500 | 0.833 | 0.236 | 0.373 | 0.491 | PASS |
| ivfflat | lists=16 | probes=4 | 47.837 | 12 MB | 1500 | 0.967 | 0.279 | 0.502 | 0.715 | PASS |

## 2. 限制

- SYNTHETIC：向量按 10 倍复制，仅用于索引行为，不能用于语义或质量结论。
- 复制产生大量等距向量（tie），ANN 与精确 Top-K 在 tie 中选行不同，因此即使索引正确，ANN Recall 也可能低于 1.0；这不是索引质量下降。
- 每个配置在临时表副本上建索引；不是对服务表 chunks 的直接查询。
- ANN Recall 是相对精确 Seq Scan Top-K 的保真度，不是语义 Recall。
- 延迟为单进程、顺序、已预热；不含并发、GPU 或跨机器。
- IVFFlat 需要先有数据再建索引；本实验按此顺序执行。
