# 索引参数实验（HNSW / IVFFlat）

- Status: **PARTIAL**
- Reason: 1 of 14 configurations failed
- Generated: 2026-10-20T03:19:38.289911+00:00
- Top-K: 10；查询数: 30；chunk strategy: `document`
- Exact reference: seqscan；queries: 30
- Synthetic scale: 1（真实 150 行）

> 每个配置在临时表副本上重建索引，不修改 `chunks` 表及其 bootstrap 索引。
> ANN Recall 是相对精确 Seq Scan Top-K 的保真度，不是语义召回。

## 1. 总览

| Method | Params | Query setting | Build ms | Size | Rows | ANN Recall@K | P50 ms | P95 ms | P99 ms | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| hnsw | m=8, ef_construction=32 | ef_search=10 | 13.167 | 1208 kB | 150 | 0.977 | 0.194 | 0.342 | 0.462 | PASS |
| hnsw | m=8, ef_construction=32 | ef_search=40 | 13.167 | 1208 kB | 150 | 1.000 | 0.223 | 0.439 | 0.536 | PASS |
| hnsw | m=8, ef_construction=32 | ef_search=100 | 13.167 | 1208 kB | 150 | 1.000 | 0.249 | 0.403 | 0.455 | PASS |
| hnsw | m=8, ef_construction=64 | ef_search=10 | 12.835 | 1208 kB | 150 | 0.977 | 0.191 | 0.329 | 0.511 | PASS |
| hnsw | m=8, ef_construction=64 | ef_search=40 | 12.835 | 1208 kB | 150 | 1.000 | 0.205 | 0.334 | 0.515 | PASS |
| hnsw | m=8, ef_construction=64 | ef_search=100 | 12.835 | 1208 kB | 150 | 1.000 | 0.237 | 0.408 | 0.534 | PASS |
| hnsw | m=8, ef_construction=128 | ef_search=10 | 15.735 | 1208 kB | 150 | 0.983 | 0.195 | 0.322 | 0.359 | PASS |
| hnsw | m=8, ef_construction=128 | ef_search=40 | 15.735 | 1208 kB | 150 | 1.000 | 0.222 | 0.351 | 0.408 | PASS |
| hnsw | m=8, ef_construction=128 | ef_search=100 | 15.735 | 1208 kB | 150 | 1.000 | 0.250 | 0.391 | 0.679 | PASS |
| hnsw | m=16, ef_construction=32 | ef_search=10 | 16.480 | 1208 kB | 150 | 1.000 | 0.230 | 0.334 | 0.701 | PASS |
| hnsw | m=16, ef_construction=32 | ef_search=40 | 16.480 | 1208 kB | 150 | 1.000 | 0.231 | 0.383 | 0.473 | PASS |
| hnsw | m=16, ef_construction=32 | ef_search=100 | 16.480 | 1208 kB | 150 | 1.000 | 0.271 | 0.449 | 0.607 | PASS |
| hnsw | m=16, ef_construction=64 | ef_search=10 | 18.134 | 1208 kB | 150 | 1.000 | 0.194 | 0.343 | 0.481 | PASS |
| hnsw | m=16, ef_construction=64 | ef_search=40 | 18.134 | 1208 kB | 150 | 1.000 | 0.226 | 0.392 | 0.708 | PASS |
| hnsw | m=16, ef_construction=64 | ef_search=100 | 18.134 | 1208 kB | 150 | 1.000 | 0.270 | 0.403 | 0.559 | PASS |
| hnsw | m=16, ef_construction=128 | ef_search=10 | 18.447 | 1208 kB | 150 | 1.000 | 0.192 | 0.379 | 0.521 | PASS |
| hnsw | m=16, ef_construction=128 | ef_search=40 | 18.447 | 1208 kB | 150 | 1.000 | 0.224 | 0.383 | 0.610 | PASS |
| hnsw | m=16, ef_construction=128 | ef_search=100 | 18.447 | 1208 kB | 150 | 1.000 | 0.263 | 0.422 | 0.506 | PASS |
| hnsw | m=32, ef_construction=32 | - | - | - | - | - | - | - | - | FAIL |
| hnsw | m=32, ef_construction=64 | ef_search=10 | 32.461 | 1208 kB | 150 | 1.000 | 0.212 | 0.373 | 0.450 | PASS |
| hnsw | m=32, ef_construction=64 | ef_search=40 | 32.461 | 1208 kB | 150 | 1.000 | 0.274 | 0.459 | 0.509 | PASS |
| hnsw | m=32, ef_construction=64 | ef_search=100 | 32.461 | 1208 kB | 150 | 1.000 | 0.346 | 0.482 | 0.782 | PASS |
| hnsw | m=32, ef_construction=128 | ef_search=10 | 35.895 | 1208 kB | 150 | 1.000 | 0.201 | 0.354 | 0.843 | PASS |
| hnsw | m=32, ef_construction=128 | ef_search=40 | 35.895 | 1208 kB | 150 | 1.000 | 0.240 | 0.399 | 0.529 | PASS |
| hnsw | m=32, ef_construction=128 | ef_search=100 | 35.895 | 1208 kB | 150 | 1.000 | 0.306 | 0.435 | 0.594 | PASS |
| ivfflat | lists=1 | probes=1 | 30.814 | 1216 kB | 150 | 1.000 | 0.206 | 0.341 | 0.379 | PASS |
| ivfflat | lists=1 | probes=2 | 30.814 | 1216 kB | 150 | 1.000 | 0.199 | 0.378 | 0.513 | PASS |
| ivfflat | lists=1 | probes=4 | 30.814 | 1216 kB | 150 | 1.000 | 0.212 | 0.403 | 0.477 | PASS |
| ivfflat | lists=2 | probes=1 | 17.249 | 1224 kB | 150 | 0.893 | 0.191 | 0.377 | 0.587 | PASS |
| ivfflat | lists=2 | probes=2 | 17.249 | 1224 kB | 150 | 1.000 | 0.204 | 0.382 | 0.574 | PASS |
| ivfflat | lists=2 | probes=4 | 17.249 | 1224 kB | 150 | 1.000 | 0.199 | 0.379 | 0.455 | PASS |
| ivfflat | lists=4 | probes=1 | 18.156 | 1240 kB | 150 | 0.617 | 0.187 | 0.346 | 0.364 | PASS |
| ivfflat | lists=4 | probes=2 | 18.156 | 1240 kB | 150 | 0.843 | 0.216 | 0.339 | 0.364 | PASS |
| ivfflat | lists=4 | probes=4 | 18.156 | 1240 kB | 150 | 1.000 | 0.205 | 0.373 | 0.627 | PASS |
| ivfflat | lists=8 | probes=1 | 18.745 | 1272 kB | 150 | 0.513 | 0.176 | 0.325 | 0.477 | PASS |
| ivfflat | lists=8 | probes=2 | 18.745 | 1272 kB | 150 | 0.690 | 0.171 | 0.304 | 0.476 | PASS |
| ivfflat | lists=8 | probes=4 | 18.745 | 1272 kB | 150 | 0.900 | 0.179 | 0.303 | 0.351 | PASS |
| ivfflat | lists=16 | probes=1 | 17.608 | 1336 kB | 150 | 0.413 | 0.162 | 0.251 | 0.423 | PASS |
| ivfflat | lists=16 | probes=2 | 17.608 | 1336 kB | 150 | 0.573 | 0.170 | 0.313 | 0.408 | PASS |
| ivfflat | lists=16 | probes=4 | 17.608 | 1336 kB | 150 | 0.810 | 0.204 | 0.346 | 0.476 | PASS |

## 2. 限制

- 每个配置在临时表副本上建索引；不是对服务表 chunks 的直接查询。
- ANN Recall 是相对精确 Seq Scan Top-K 的保真度，不是语义 Recall。
- 延迟为单进程、顺序、已预热；不含并发、GPU 或跨机器。
- IVFFlat 需要先有数据再建索引；本实验按此顺序执行。
- 150 行规模下参数差异可能不可观察；不能外推到 10 万级以上。

## 3. 失败的配置

- `idx_bench_hnsw_m32_efc32`: InvalidParameterValue: ef_construction must be greater than or equal to 2 * m
