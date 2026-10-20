# Apex RAG 评测结果汇总

- 汇总日期：2026-10-19
- 源报告生成日期：2026-10-17（Dense / Sparse / Hybrid / Rerank / RAG）
- 汇总性质：**只读汇总现有报告**；本汇总没有重新运行 Benchmark、pytest、RAG、
  Embedding、Reranker 或真实 LLM，也没有修改任何历史报告、JSON、数据或 gold label。
- 总体状态：Dense / Sparse / Hybrid / Rerank 的报告状态为 **PASS**；RAG 的正式报告
  状态为 **PARTIAL**（mock Provider + historical 重放，正式报告未调用真实 LLM）；
  另有独立的真实 RAG + Judge 全量结果（`reports/judge/`，PASS，30/30），见 §3.5。
- 后续代码变更（2026-10-20，P0）：检索层新增 `chunk_strategy` 作用域，历史 RRF 重放
  会校验 artifact 与命中的策略；索引管理新增 shadow 建索引 API。这些变更没有重跑
  Benchmark，也没有改写任何历史报告；本汇总中的数字仍只属于原始报告的运行环境。
- 后续代码变更（2026-10-20，P0+P1+P2）：新增统一评测/报告层、索引与分块实验、
  RRF 敏感性与 FastAPI。随后已在**容器内真实执行**统一检索、索引、分块、RRF 敏感性、
  Top-20 → Top-5 精排与最终报告生成，产物为 `reports/unified/*.json`、
  `reports/final_report.md` 与 `reports/figures/*.png`。加入 Judge 后报告已按
  30/30 全量结果重生成，`Report status: PASS`，图共 5 张。本汇总新增 §8 记录这些
  容器结果；五份历史报告仍保持只读。
- 后续新增（2026-10-20，Judge）：LLM-as-a-Judge 模块完成宿主 Ruff / mypy / 定向与
  全量测试，以及容器 mock 闭环（`reports/judge-smoke/`）和**真实 30/30 全量运行**
  （`reports/judge/`，PASS）。真实结果：Context Relevance mean 3.9667、
  Faithfulness mean 5.0000、0 失败、0 NOT RUN、`same_model=true`。历史 3/30 真实
  smoke 归档在 `reports/judge-real-3q/`；人工校准与 inter-judge 一致性仍为
  **NOT RUN**，见 §3.5 与 `README.md` 的“LLM-as-a-Judge 自动评估”一节。

> 本文是跨报告索引与摘要，不替代原始报告。每个数字均来自下方的原始报告或数据文件；
> 如果汇总与原始报告冲突，以原始报告为准。复核时必须按原始报告记录的环境、数据和
> 命令执行，不能把本汇总当作一次新的实测。
> 题目评分维度、加分项与提交物对照见 `README.md` 的“技术评估要求对照（题目 PDF）”。

---

## 1. 项目目标与评测范围

项目面向信用卡知识库的检索增强问答，比较朴素稠密检索、混合检索（Dense + PostgreSQL
FTS + RRF）与 Cross-Encoder 重排序，并把最终证据接入可替换的 LLM Provider，生成带
引用的结构化答案。

| 阶段 | 评测问题 | 原始报告 |
|---|---|---|
| Dense | 不同 HNSW 参数与自然查询计划下的召回、排序一致性和延迟 | [`dense_benchmark.md`](dense_benchmark.md) / [`dense_benchmark.json`](dense_benchmark.json) |
| Sparse | PostgreSQL FTS 的 OR/AND 语义、召回与延迟 | [`sparse_benchmark.md`](sparse_benchmark.md) / [`sparse_benchmark.json`](sparse_benchmark.json) |
| Hybrid / RRF | Dense、Sparse、候选并集与 RRF 的五臂对比 | [`hybrid_benchmark.md`](hybrid_benchmark.md) / [`hybrid_benchmark.json`](hybrid_benchmark.json) |
| Reranker | Cross-Encoder 在 RRF Top-10 候选池上的重排收益 | [`rerank_benchmark.md`](rerank_benchmark.md) / [`rerank_benchmark.json`](rerank_benchmark.json) |
| RAG | 上下文组装、证据编号、引用校验与端到端链路 | [`rag_benchmark.md`](rag_benchmark.md) / [`rag_benchmark.json`](rag_benchmark.json) |

**环境边界（所有报告的共同前提）**：实测在宿主 Python 3.14.3 / Windows 11 上完成，
PostgreSQL 16.14 + pgvector 0.8.2 运行在容器中；Dense / Sparse / Hybrid / Rerank / RAG
报告均记录 Docker 控制面在运行时不可用或未用于执行。**容器内完整链路未验证**，宿主
结果不能当作容器结果。

---

## 2. 数据集规模与 gold label

| 项目 | 值 | 来源 |
|---|---|---|
| 文档数 | 150 | Dense / Sparse / Hybrid / Rerank / RAG 报告 |
| chunk 数 | 150 | 同上 |
| 分块策略（本汇总的历史 Benchmark） | `document`（1 篇文档 = 1 个 chunk） | `docs/roadmap.md` §1；`src/ingestion/chunker.py` |
| 评测 Query | 30 条英文 Query | 五份报告 |
| gold 引用总数 | 43 条 | `docs/roadmap.md` §2 的只读统计（`data/eval_queries.json`） |
| 不同 gold 文档数 | 38 篇 | `hybrid_benchmark.json` 的 `dataset.distinct_gold_docs`；`docs/roadmap.md` §2 |
| 单条 Query 的 gold 大小 | 1–3 | Dense / Sparse / Hybrid / Rerank 报告 |
| 缺失 gold 文档 | 无（`[]`） | Dense / Sparse / Rerank 报告 |
| Policy Code 数 | 78 | `docs/roadmap.md` §1 的只读统计（`data/kb_documents.json`） |

> **口径说明**：43 是 **gold 引用条数**，38 是 **不同文档数**。Hybrid 报告的限制段落
> 曾写“all 43 gold documents”，该措辞不准确；正确表述是“43 条 gold 引用 / 38 篇不同
> 文档”。这不改变 Sparse OR Recall 1.000 的结论。历史报告未修改。

**标注限制**：gold label 不是穷尽相关性标注。非 gold 位置不能视为 true negative，
因此 `Precision@K` 与二元 `nDCG@K` 只能作为补充指标；本汇总引用的历史报告未包含
这些字段（当前代码已实现并有单元测试，但不改写历史结果）。现有 Recall / MRR /
Hit@1 也不能替代生成质量。

---

## 3. 各方案结果摘要

### 3.1 Dense（阶段 ④-C）

- 30 条 Query，K=10，每条配置 5 次计时；5 个臂：`exact_seqscan`、`natural_plan`、
  `hnsw_ef_10`、`hnsw_ef_40`、`hnsw_ef_100`。
- **Semantic Recall mean = 0.944**（五个臂相同，对 `relevant_doc_ids`）。
- **ANN Recall@K mean / min = 1.000 / 1.000**（对 exact chunk Top-K；这是 ANN 保真度，
  不是语义召回）。
- `order-equal = 30`、`top-1 match = 30`、`errors = 0`、`timeouts = 0`。
- 自然查询计划为 `Seq Scan`；HNSW 臂只在 `enable_seqscan=off` 的诊断设置下使用。
  150 chunks 下没有可观察的 HNSW 延迟优势，这不是规模结论。

来源：[`dense_benchmark.md`](dense_benchmark.md) / [`dense_benchmark.json`](dense_benchmark.json)。

### 3.2 Sparse（阶段 ⑤-A / ⑤-B）

- 30 条 Query，K=10，5 次计时 + 1 次 warm-up；English FTS，默认 OR；
  `websearch_to_tsquery('english', query)` + GIN + `ts_rank_cd`。
- **OR Semantic Recall mean = 1.000**；perfect-recall 30/30；zero-recall 0；empty 0。
- **AND control Semantic Recall mean = 0.133**；perfect 4/30；zero-recall 26；empty 26。
- 中文探针 5 条均为 0 命中；语料是英文，不能据此判断中文检索质量。CJK fallback 是
  两字符子串匹配，不是分词；混合中英文查询只走 CJK 路径。
- Sparse 与 Dense 候选的平均 chunk Jaccard / doc Jaccard 均为 0.362；gold 覆盖
  `both=38`、`sparse-only=5`、`dense-only=0`、`neither=0`。候选重叠不是质量指标。

来源：[`sparse_benchmark.md`](sparse_benchmark.md) / [`sparse_benchmark.json`](sparse_benchmark.json)。

### 3.3 Hybrid / RRF（阶段 ⑥-B）

- 五臂：Dense、Sparse OR、Sparse AND、候选并集（candidate union）、RRF。
- 30 条 Query；`top_k=10`、`candidate_k=10`、RRF `k=60`；5 次计时 + 1 次 warm-up。
- **RRF：Recall macro 1.000、MRR 0.950、Hit@1 27/30**。
- Sparse OR 已经达到 Recall 1.000；因此在本数据集上，混合检索的 Recall 不可能超过 1.0。
  RRF 的增益主要体现在 MRR（0.940 → 0.950）和补回 Dense 漏检的 5 条 gold。
- 5 条 Dense 漏检 gold 均进入 RRF Top-10：Q-004/DOC-0017、Q-005/DOC-0021、
  Q-009/DOC-0039、Q-009/DOC-0040、Q-014/DOC-0067。
- `Precision@K`、`nDCG@K` 未计算；原因是标签非穷尽。

来源：[`hybrid_benchmark.md`](hybrid_benchmark.md) / [`hybrid_benchmark.json`](hybrid_benchmark.json)。
同口径五臂数字见 §5.1。

### 3.4 Reranker（阶段 ⑦-B）

- 模型：`BAAI/bge-reranker-base` @ `2cfc18c9415c912f9d8155881c133215df768a70`；
  Cross-Encoder，batch 16，max length 512，activation identity，CPU。
- 四个基线臂复用 ⑥-B 的历史 artifact；重排臂使用同一 Query 的历史 RRF Top-10 身份，
  文本由 PostgreSQL **实时读取**，重排**实时执行**。Dense/Sparse/RRF 没有在 ⑦-B 重新查询。
- **RRF baseline：Recall/MRR/Hit@1 = 1.0000 / 0.9500 / 27/30**。
- **RRF + Reranker：Recall/MRR/Hit@1 = 1.0000 / 0.9833 / 29/30**。
- MRR 变化：improved 3 / equal 26 / worse 1；平均 ΔMRR +0.0333；首个 gold 平均名次
  变化 −0.0667（负值表示提前）；Top-1 改变的 Query 数为 6。
- 候选池相同，因此 Recall 不可能因重排上升；本阶段只考察 MRR / Hit@1。
- 模型加载 25,992.8 ms；峰值 RSS 1,288.9 MiB / peak WSET 1,314.6 MiB；系统可用内存
  最低 1.123 GiB。10 候选 batch 1/16：1,163.2 / 1,082.5 ms；50 候选 batch 1/16：
  5,742.2 / 4,599.4 ms；约 92–116 ms/candidate。

来源：[`rerank_benchmark.md`](rerank_benchmark.md) / [`rerank_benchmark.json`](rerank_benchmark.json)。
重排前后数字见 §5.2。

**题面参数复跑（容器，2026-10-20）**：为对齐题目要求的 Top-K 20 → 5，另跑了一次
实时统一评测 `reports/unified/rerank_20to5.json`（30/30，`candidate_k=20`、
`top_k=5`、RRF k=60、`document` 策略）。两个臂在同一候选池上实时执行：

| 臂 | Recall macro | Recall micro | nDCG@5 | MRR | Hit@1 | P50 / P95 / P99 ms |
|---|---:|---:|---:|---:|---:|---:|
| `rrf`（20 → 5） | 0.9556 | 0.9070 | 0.9204 | 0.9500 | 27/30 | 89.46 / 109.55 / 112.87 |
| `rrf_rerank`（20 → 5） | 1.0000 | 1.0000 | 0.9777 | 0.9833 | 29/30 | 1790.92 / 2038.51 / 4683.37 |

这是**实时检索 + 实时重排**，不是历史重放；候选池缩小到 5 后，`rrf` 在 Q 上的
Recall 从 1.000（K=10）降到 0.9556，而 Cross-Encoder 在同一 20 条候选池上把
Recall 拉回 1.0000，并提升 MRR/nDCG。代价是 P50 从约 89 ms 升到约 1791 ms（CPU）。
它不改变 §3.4 历史报告的 10 → 10 口径，两者分开记录。

### 3.5 RAG（阶段 ⑧）

- **Status: PARTIAL**。Provider 是确定性 **mock**，答案以 `MOCK` 开头；正式报告
  记录 `real_llm_calls = False`、`real_llm_evaluation = not_run`。
- 检索模式为 **historical**：重放 ⑥-B 已接受的 RRF candidate identities；Dense、
  Sparse、RRF 没有在 ⑧ 重新执行；证据文本实时读自 PostgreSQL，Reranker 实时运行。
- 30/30 `status=ok`；生成失败率 0.0000；证据不足率 0.0000；引用有效率 1.0000（30/30）；
  证据 Recall macro / micro = 1.0000 / 1.0000；证据 MRR = 0.9833；上下文截断 Query 数 0。
- total P50/P95 = 970.8791 / 1082.2045 ms；retrieval P50/P95 = 970.0994 / 1081.4004 ms；
  LLM mock P50/P95 = 0.0675 / 0.0904 ms；n=30，单进程，未剔除首次模型加载。
- **本报告未计算**：answer correctness、Answer Relevance；Context Relevance /
  Faithfulness 由独立 Judge 模块评估。真实 30/30 全量已执行（`reports/judge/`，
  PASS）：Context Relevance mean 3.9667、Faithfulness mean 5.0000、0 失败、
  `same_model=true`；另有独立 mock 闭环 `reports/judge-smoke/` 与历史 3/30 smoke
  `reports/judge-real-3q/`。人工校准与 inter-judge 一致性仍为 **NOT RUN**；
  Judge 分数不并入本节历史数字。
  证据 Recall/MRR 是检索指标，不是生成准确率。
- 单条 Query 的 total 最大值为 Q-001 的 24,025.2685 ms，包含首次懒加载；因此
  P50/P95 不是稳态延迟。

来源：[`rag_benchmark.md`](rag_benchmark.md) / [`rag_benchmark.json`](rag_benchmark.json)。

---

## 4. 关键指标、延迟与资源开销

### 4.1 延迟

| 方案 | 指标 | 值 | 口径 / 来源 |
|---|---|---|---|
| Dense | query P50 / P95 | 0.778–1.022 / 0.960–1.185 ms | 五个臂的范围；n=150；5 次计时；④-C |
| Dense | db P50 / P95 | 2.935–4.237 / 3.480–5.625 ms | 五个臂的范围；含连接池/事务/GUC；④-C |
| Sparse | query P50 / P95 | 2.054 / 2.698 ms | n=150；5 次计时 + 1 warm-up；⑤-B |
| Sparse | db P50 / P95 | 3.086 / 3.865 ms | n=150；⑤-B |
| Hybrid | 见 §5.1 | 五臂各自 P50/P95 | n=150；5 次计时 + 1 warm-up；⑥-B |
| Rerank | 10 候选 batch 1 / 16 | 1,163.2 / 1,082.5 ms | CPU，单进程；⑦-B |
| Rerank | 50 候选 batch 1 / 16 | 5,742.2 / 4,599.4 ms | CPU，单进程；⑦-B |
| Rerank | ms/candidate | 约 92–116 | 由上表换算的候选池延迟范围；⑦-B |
| RAG | total P50 / P95 | 970.8791 / 1082.2045 ms | n=30；含首次懒加载；⑧ |
| RAG | retrieval P50 / P95 | 970.0994 / 1081.4004 ms | n=30；⑧ |
| RAG | LLM mock P50 / P95 | 0.0675 / 0.0904 ms | mock，不代表真实网络；⑧ |

### 4.2 资源开销

| 项目 | 值 | 来源 |
|---|---|---|
| Reranker 模型加载 | 25,992.8 ms | ⑦-B |
| Reranker 加载后 RSS | 848.1 MiB | ⑦-B |
| Reranker 峰值 RSS / peak WSET | 1,288.9 MiB / 1,314.6 MiB | ⑦-B |
| Reranker 运行期间系统最低可用内存 | 1.123 GiB | ⑦-B |
| Reranker batch 不变性最大分数差 | 5.0067901611328125e-06 | ⑦-B |
| Dense / Sparse / Hybrid 进程级 RSS / CPU | **报告未提供** | ④/⑤/⑥ 只报告 DB/query 延迟 |
| RAG 进程级 RSS / CPU | **报告未提供** | ⑧ 只报告 total/retrieval/llm 延迟 |

> 资源数据只在源报告确实提供时填写。Dense / Sparse / Hybrid 报告没有进程级内存或
> CPU 采样，因此不能从延迟反推内存或吞吐。所有延迟均为单进程、顺序、已预热的宿主
> 运行；不代表 GPU、并发、吞吐或跨机器性能。

---

## 5. 结果对比表

### 5.1 Hybrid 同口径五臂（⑥-B，30 queries，K=10，candidate_k=10，RRF k=60）

| 臂 | Recall macro | Recall micro | MRR | Hit@1 | 有 gold 查询 | 无 gold 查询 | 空结果 | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0.944 | 0.884 | 0.917 | 25/30 | 30 | 0 | 0 | 75.062 | 94.624 |
| Sparse OR | 1.000 | 1.000 | 0.940 | 27/30 | 30 | 0 | 0 | 3.750 | 5.608 |
| Sparse AND | 0.133 | 0.093 | 0.133 | 4/30 | 4 | 26 | 26 | 1.343 | 1.716 |
| Candidate union | 1.000 | 1.000 | 0.917 | 25/30 | 30 | 0 | 0 | 78.733 | 98.924 |
| RRF | 1.000 | 1.000 | 0.950 | 27/30 | 30 | 0 | 0 | 78.714 | 98.862 |

> 这些数字来自同一次 ⑥-B 运行，列间可比。Dense 的 0.944 是 Semantic Recall；
> 与 ④-C 的 ANN Recall 1.000 不是同一个指标。Sparse AND 的 4/30 表示只有 4 条 Query
> 有 gold 结果，不是 4 条 Query 的排序正确率。

### 5.2 Reranker 前后（⑦-B）

| 臂 | Recall macro | Recall micro | MRR | Hit@1 | MRR 变化 | 来源 |
|---|---:|---:|---:|---:|---|---|
| RRF baseline | 1.0000 | 1.0000 | 0.9500 | 27/30 | — | 历史 ⑥-B |
| RRF + Reranker | 1.0000 | 1.0000 | 0.9833 | 29/30 | +0.0333 | 实时重排 |

- MRR：improved 3 / equal 26 / worse 1；首个 gold 平均名次变化 −0.0667。
- 候选池相同，Recall 保持 1.0000；重排只改变顺序。
- 排序变化的 Query：Q-002、Q-011、Q-016、Q-017。Q-011 的 RRF Top-1 从 gold
  DOC-0050 变成非 gold DOC-0053，但三个 gold 仍在 Top-10；这是唯一的 MRR 下降 Query。

### 5.3 RAG（⑧）与检索指标的关系

| 指标 | 值 | 口径 |
|---|---:|---|
| 端到端成功率 | 1.0000 (30/30) | status ∈ {ok, insufficient_evidence}；mock |
| 生成失败率 | 0.0000 (0/30) | Provider 已调用但未产出结构化答案 |
| 证据不足率 | 0.0000 (0/30) | 空检索短路 + 模型声明证据不足 |
| 引用有效率 | 1.0000 (30/30) | 有效 `E<n>` / 全部返回标签 |
| 证据 Recall macro / micro | 1.0000 / 1.0000 | 最终证据集合；检索指标 |
| 证据 MRR | 0.9833 | 最终证据集合；检索指标 |
| answer correctness / Answer Relevance | **未计算** | 无 rubric、人工评分或实现 |
| Context Relevance / Faithfulness | 由 Judge 模块单独评估 | 真实 30/30 全量（PASS）：CR mean 3.9667、Faithfulness mean 5.0000、`same_model=true`；人工校准与 inter-judge 一致性为 **NOT RUN**；不并入本表 |

> RAG 的 1.0000 证据 Recall 与 0.9833 证据 MRR **不是**答案正确率。mock 答案不能
> 用于评价真实 LLM 的生成质量；另有 Judge 30/30 全量真实运行（见 §3.5），
> 不属于本节的 ⑧ 数字。

---

## 6. 已知限制、口径差异与未验证项

1. **规模有限**：只有 150 documents；`document` 策略为 150 chunks，三种分块合计
   578 chunks。Dense 的自然计划是 `Seq Scan`，HNSW 没有可观察延迟优势；这些结果
   不能外推到 10 万、100 万或 1000 万级数据。
2. **分块对比已运行（容器，2026-10-20）**：`document`、`fixed_window`、`structured`
   分别入库 150 / 267 / 161 chunks，同一 30 条 Query 上运行
   `dense` / `sparse_or` / `rrf`，另有独立 artifact 增加 `rrf_rerank`。证据：
   `reports/unified/chunking_benchmark.json`（PASS）与
   `reports/unified/chunking_benchmark_rerank.json`（PASS）。
   **没有单一策略在所有指标上胜出**：`fixed_window` 的 dense Recall 最高
   （0.9889），`structured` 的 `sparse_or` MRR 最高（0.9733）；`document` 的
   `rrf_rerank` MRR 最高（0.9833）。rerank 对三种策略的 MRR/nDCG 都有提升
   （如 `structured` MRR 0.8778 → 0.9500），但 P50 延迟从约 82–97 ms 升到
   512–793 ms，且所有策略共用一个 HNSW 索引，索引大小无法按策略归因。
   另外 `chunk_strategy` 过滤发生在 ANN 扫描之后：`ef_search` 较小时某策略可能
   返回少于 K 条，策略间召回差异因此部分来自其在全局索引中的占比。
   这些是文档级检索指标，不是答案质量。历史 Benchmark 仍只使用 `document`。
3. **语言有限**：30 条 Query 是英文；中文探针 5 条均为 0 命中；CJK fallback 是两字符
   子串匹配，不是分词。中文/混合语言的检索、重排、生成与引用均未验证。
4. **指标口径**：labels 非穷尽；`Precision@K` 与二元 `nDCG@K` 已在代码中实现
   并有单元测试，但历史报告未包含，只能作为补充指标。`Recall@K` / MRR / Hit@1
   是检索指标，不能代表答案正确性。
5. **RAG 是 mock + historical**：Provider 为 mock，答案以 `MOCK` 开头；Dense/Sparse/RRF
   未在 ⑧ 重新执行；证据指标衡量被重放的候选池。**⑧ 未调用真实 LLM**；独立的
   Judge 30/30 全量运行见 §3.5，不能改变本报告的 mock 结论。
6. **Reranker 基线是历史重放**：四个基线臂来自 ⑥-B；重排臂使用历史 RRF Top-10 身份，
   文本实时读取、重排实时执行。候选池相同，Recall 不会因重排上升。
7. **未同进程加载 Embedding + Reranker**：⑦-B 没有测量二者共存的内存行为；RAG live
   链路曾在同进程 reranker 加载阶段触发原生段错误（exit 139），历史
   `0xC0000005` / `torch_cpu.dll` 根因未确认。E-5B（多候选）未执行。
8. **完整 pytest 状态**：容器内全量套件已真实执行；加入 Judge 测试后为
   `691 passed, 1 skipped`（JUnit `reports/container/pytest-full-judge.xml`），
   旧记录为 `664 passed, 1 skipped`（`reports/container/pytest-full.log`）。
   宿主定向 Judge 测试为 36 tests / 3 skipped（缺 `matplotlib`）。依赖版本与
   `pyproject.toml` 的精确 pin 仍存在漂移，没有覆盖传递依赖的锁文件。
   `chunk_strategy` 过滤与索引管理已有真实数据库/运行证据（见 §8）。
9. **Docker 全链路仍非完整**：数据库侧、三策略 ingest、统一检索、索引、分块、
   RRF 敏感性、Top-20 → Top-5 精排、模型离线加载与报告/图均已在容器内运行；
   真实 RAG + Judge 30/30 全量也已执行；FastAPI `api` 服务已启动，`/healthz` 与
   `/acceptance/api/summary` 已验证（浏览器渲染、`/readyz` 与 `/answer` 未验证）。
   题目 Bonus C 的 `POST /api/v1/search` 与 Swagger 已实现并在容器内实际请求：
   `api` 容器 healthy，检索接口返回 200、5 条结果，`/docs` 返回 200，
   `/openapi.json` 含 `/api/v1/search`。重启 `api` 后按最终文件复跑，两个请求仍为
   200。首次请求包含 embedding/reranker 加载（约 15–19 s），不是稳态延迟。
   Ruff / mypy / `tests/test_api.py`（23 个用例）均已通过。详见
   `docs/roadmap.md` §10.8。五个历史 Benchmark 与 RAG 的原始数字仍来自宿主，
   不能当作容器结果。
10. **延迟口径**：所有延迟都是单进程、顺序、宿主机运行，不含并发、GPU 或跨机器；
    RAG 的 total P50/P95 包含首条 Query 的懒加载（Q-001 = 24,025.2685 ms），不是稳态值。
11. **Hybrid 报告措辞**：报告限制段写“all 43 gold documents”，实际为 43 条 gold 引用 /
    38 篇不同文档；这不改变 Recall 1.000 的结论。历史报告未修改。
12. **报告状态不等于生产可用**：Dense/Sparse/Hybrid/Rerank 的 PASS 是针对各自报告
    口径的评测状态；RAG 仍是 PARTIAL。没有任何报告支持“生产级”“高可用”“大规模”
    或“完整生成质量”结论。
13. **索引管理已执行（两轮）**：容器内已执行 shadow HNSW/IVFFlat 建索引、查询与清理。
    `reports/unified/index_benchmark.json`（`synthetic_scale=1`）因
    `m=32, ef_construction=32` 违反 pgvector 的 `ef_construction >= 2*m` 而 PARTIAL；
    随后用合法网格与 `synthetic_scale=10`（1500 行）重跑，得到 PASS 的
    `reports/unified/index_benchmark_scale10.json`。合成向量只反映索引行为，
    不进入语义质量结论；旧 PARTIAL artifact 保留，未被覆盖。
14. **Judge 已完成 30/30 真实全量**：Context Relevance mean 3.9667（2×1、3×4、
    4×20、5×5；低分 1 条 Q-029）、Faithfulness mean 5.0000（5×30），
    `same_model=true`（同模型自评），无人工校准或 inter-judge 一致性；0 失败、
    0 NOT RUN，但不能替代人工质量结论。配置模型字段为 `deepseek-chat`，但响应
    `model` 均为 `deepseek-flash`（同一 endpoint，报告中的模型名应理解为配置名）。
    另外 Q-001 的 Context Relevance 为 4 分，却把 10 条 context 中的 9 条标为
    irrelevant，说明 rubric 执行可能不一致；分数不是客观真值。历史 3/30 smoke
    归档在 `reports/judge-real-3q/`。

---

## 7. 原始报告索引与复核方式

| 报告 | 状态 | 主要内容 | 复核入口 |
|---|---|---|---|
| [`dense_benchmark.md`](dense_benchmark.md) / [`.json`](dense_benchmark.json) | PASS | 5 个臂；Semantic Recall 0.944；ANN Recall 1.000；query/db 延迟；计划证据 | `python scripts/benchmark_dense.py --top-k 10 --ef-search 10 40 100 --repetitions 5 --warmup-passes 1`（源报告命令） |
| [`sparse_benchmark.md`](sparse_benchmark.md) / [`.json`](sparse_benchmark.json) | PASS | OR/AND 召回、query/db 延迟、中文探针、Dense/Sparse 候选重叠 | 源报告未附完整命令行；入口为 `scripts/benchmark_sparse.py`（见 README） |
| [`hybrid_benchmark.md`](hybrid_benchmark.md) / [`.json`](hybrid_benchmark.json) | PASS | 五臂 Recall/MRR/Hit@1/延迟、逐查询 MRR、候选覆盖 | `python scripts/benchmark_hybrid.py --top-k 10 --candidate-k 10 --rrf-k 60 --repetitions 5 --warmup-passes 1`（源报告命令） |
| [`rerank_benchmark.md`](rerank_benchmark.md) / [`.json`](rerank_benchmark.json) | PASS | RRF vs RRF+Reranker、逐查询变化、模型/资源/延迟 | `python scripts/benchmark_rerank.py --top-k 10 --candidate-k 10 --rrf-k 60 --batch-size 16 --activation identity`（源报告命令） |
| [`rag_benchmark.md`](rag_benchmark.md) / [`.json`](rag_benchmark.json) | PARTIAL | mock Provider + historical；引用校验；证据 Recall/MRR；延迟 | `python scripts/benchmark_rag.py --provider mock --retrieval-mode historical`（源报告命令） |

**复核方式**：

1. 先读对应 `.json` 的运行元数据（环境、参数、数据哈希、逐查询结果），再读 `.md` 摘要；
   不要只看汇总数字。
2. 按源报告记录的命令、数据与依赖环境执行；本汇总没有重跑五份历史报告。§8 的
   容器 artifact 是另一次独立运行，只按自己的 `.json` 复核。
3. 报告重新生成会改变耗时、时间戳与哈希；不要覆盖历史报告。报告哈希见
   [`README.md`](../README.md) 的“验收证据哈希”一节。
4. 如果代码、数据、gold label、模型或依赖发生变化，应先更新对应报告和 README 阶段表，
   再更新本汇总；本汇总不能替代原始报告或验收记录。

**报告文件哈希**：见 [`README.md`](../README.md) 的“验收证据哈希”一节。该哈希只在
报告内容不变时有意义；重新运行 Benchmark 会改变耗时等字段，从而改变哈希。

---

## 8. 本轮容器内新增 artifact（2026-10-20）

| Artifact | Status | 口径 | 关键数字 |
|---|---|---|---|
| [`unified/benchmark.json`](unified/benchmark.json) / `.md` | PASS | 30 queries；`top_k=10`、`candidate_k=10`、RRF k=60；6 arms；容器 | dense 0.9444/0.9167/25；sparse_or 1.0000/0.9400/27；sparse_and 0.1333/0.1333/4；union 1.0000/0.9167/25；rrf 1.0000/0.9500/27；rrf_rerank 1.0000/0.9833/29（Recall/MRR/Hit@1） |
| [`unified/rerank_20to5.json`](unified/rerank_20to5.json) / `.md` | PASS | 30 queries；**candidate pool 20 → final Top-5**；实时检索 + 实时重排；容器 | rrf 0.9556/0.9500/27；rrf_rerank 1.0000/0.9833/29；nDCG@5 0.9204 → 0.9777；P50 89.46 → 1790.92 ms |
| [`unified/chunking_benchmark.json`](unified/chunking_benchmark.json) / `.md` | PASS | 3 策略 × 3 arms × 30 queries；578 chunks；容器 | 详见 `final_report.md` §5 |
| [`unified/chunking_benchmark_rerank.json`](unified/chunking_benchmark_rerank.json) / `.md` | PASS | 3 策略 × 4 arms（含 `rrf_rerank`）× 30 queries；容器 | `document` 的 `rrf_rerank` MRR 0.9833 最高 |
| [`unified/fusion_sensitivity.json`](unified/fusion_sensitivity.json) / `.md` | PASS | RRF k / candidate_k 网格；容器 | 6 个网格单元 |
| [`unified/index_benchmark_scale10.json`](unified/index_benchmark_scale10.json) / `.md` | PASS（合成） | `synthetic_scale=10`（1500 行）；6 组 HNSW + 5 组 IVFFlat；容器 | 只反映索引行为：HNSW 多数 ANN Recall 1.000，`m=32/efc=64` 在 `ef_search=10` 为 0.797；IVFFlat `lists=16/probes=1` 为 0.733 |
| [`unified/index_benchmark.json`](unified/index_benchmark.json) / `.md` | PARTIAL（保留） | `synthetic_scale=1`；14 组中 1 组因非法参数失败 | 不作为调参结论 |
| [`judge-smoke/`](judge-smoke/judge_summary.json) / `.md` | PARTIAL（**MOCK fixture**） | 3/30；mock RAG + mock judge；只验证接线、解析、失败隔离与聚合 | 两指标各 3/3、mean/median 4.0；**不是模型判断**；真实结果见下一行 |
| [`judge/`](judge/judge_summary.json) / `.md` | PASS（**真实**） | 真实 RAG + 真实 Judge，30/30 全量；同模型自评；另含 JSONL 30 行 | CR mean 3.9667（2×1、3×4、4×20、5×5）；Faithfulness mean 5.0000（5×30）；0 失败、0 NOT RUN；`judge_is_real=true`、`real_judge_calls=true` |
| [`judge-real-3q/`](judge-real-3q/judge_summary.json) | PARTIAL（**真实，历史归档**） | 真实 RAG + 真实 Judge，3/30 smoke；同模型自评；JSONL 3 行 | CR mean 4.3333（4×2、5×1）；Faithfulness mean 5.0000（5×3）；0 失败、27 NOT RUN |
| [`final_report.md`](final_report.md) + [`figures/`](figures/) | PASS | 由 benchmark / index_scale10 / chunking_rerank 渲染，另含 rerank 20→5 与 30/30 Judge 章节 | 每个表格带 artifact 路径与 SHA256；5 张 PNG（含 `judge_distribution.png`）；`Report status: PASS` |

> 这些数字来自容器内独立运行，与 §1–§6 的五份历史报告（宿主 Python）不是同一环境，
> 不能互相替代。索引实验的合成规模只用于观察索引行为，不进入语义质量结论。
