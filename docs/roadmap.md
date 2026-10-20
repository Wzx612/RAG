# Roadmap 与当前状态

> 阶段 ⑨ 产出（2026-10-19）；2026-10-20 增补第 10 节（P0+P1+P2 迭代）。
> 本文汇总当前实现、可复核证据、未验证事项与后续路线。
> 它不是验收报告：阶段验收结论以 README 的“实现进度与阶段验收”表和 `reports/*` 为准。
> 本文没有重新运行历史 Benchmark；真实 LLM API 在 10.6 的 3 条 smoke 与 30 条
> 全量 Judge 中调用，不改变历史检索指标。历史指标逐字引用现有报告。
> 第 10 节记录的新代码已于 2026-10-20 执行：宿主机 Ruff / mypy / 纯单元 pytest
> 通过；容器内 `db-check`、三策略 ingest、模型离线加载、统一检索、索引、分块、
> RRF 敏感性、Top-20 → Top-5 精排与最终报告/图均已运行并留下 artifact。
> 真实 LLM 已执行 30 条全量 RAG + Judge（10.6，PASS）；API 服务与只读验收接口
> 已在容器内验证，浏览器渲染仍为 `NOT RUN`。
> 题目评分维度、加分项与提交物对照见 `README.md` 的“技术评估要求对照（题目 PDF）”。

---

## 1. 项目快照

- 数据集：150 篇文档 / 30 条英文评测 Query；78 个 Policy Code（`data/kb_documents.json`
  只读统计）；43 条 gold 引用 / 38 篇不同文档；单条 Query 的 gold 大小为 1–3。
- 语料分块：`document`、`fixed_window`、`structured` 三种策略均已实现并入库
  （150 / 267 / 161 chunks，合计 578）；`contextual` 已退役。分块对比已在容器内
  运行：30 条 Query × 3 策略 × `dense/sparse_or/rrf`，另有独立 artifact 增加
  `rrf_rerank`；结果见 `reports/unified/chunking_benchmark.json` 与
  `chunking_benchmark_rerank.json`（均 PASS）。历史 Dense/Sparse/Hybrid/Rerank/RAG
  Benchmark 仍只使用 `document`。
- 检索作用域（P0，2026-10-20）：`DenseConfig` / `SparseConfig` 新增
  `chunk_strategy`（默认 `document`），Dense 与 Sparse 的 SQL 均按该列过滤，
  行映射器再校验一次；各 Benchmark 的环境门槛与数据完整性检查改用按策略统计的
  `strategy_chunks` / `strategy_docs`。已在含三种策略、共 578 chunks 的数据库上
  实际运行，三种策略的 30 条 Query 全部完成。
- 技术栈：PostgreSQL 16 + pgvector HNSW（cosine）、`BAAI/bge-large-en-v1.5`
  （1024 维）、PostgreSQL FTS、RRF（k=60）、`BAAI/bge-reranker-base`
  Cross-Encoder、OpenAI 兼容 Provider。
- 运行环境：五份历史报告（Dense / Sparse / Hybrid / Rerank / RAG）均在宿主
  Python 3.14.3 / Windows 11 上完成，PostgreSQL 在容器中。2026-10-20 的新实验
  （统一检索、索引、分块、RRF 敏感性、20→5 精排、报告/图）已在容器内运行并记录
  容器环境字段；容器内全量 pytest 曾通过（664 passed、1 skipped、`EXIT=0`），
  报告测试扩展后已重跑（统计见第 10 节）；真实 LLM 已做 30 条全量 RAG + Judge，
  API 服务与只读验收接口已在容器内执行。
- 当前状态：⑧ 离线 RAG 已实现、报告 PARTIAL；⑨ 文档已定稿、待人工验收；
  ⑩ 交付检查完成、待人工确认；Git 仓库已初始化（`main` 分支，已有历史提交），
  工作区仍有未提交改动，本轮未 commit / push；FastAPI `api` 服务已在容器内启动，
  `/healthz`、`/acceptance/api/summary`、`/api/v1/search` 与 `/docs` 已实际请求
  （浏览器渲染未验证）；Windows 原生崩溃根因未确认。
- P0 文档整改（Stage ⑩-B）：已新增 `reports/results.md`、修正 `.dockerignore`
  过期陈述、补充生产治理设计；仅文档变更，未执行 P1/P2，待人工确认。
- P0+P1+P2 迭代（2026-10-20，见第 10 节）：新增统一评测/报告层、索引与分块实验、
  RRF 敏感性、完整 CLI、只读 FastAPI 与容器编排。代码已写入；宿主机 Ruff / mypy /
  纯单元 pytest 通过，容器内 `db-check`、三策略 ingest、模型离线加载、统一检索
  benchmark、索引实验（PARTIAL）、分块对比（含 `rrf_rerank`）、RRF 敏感性均已运行
  并留下 artifact。`final_report.md` + 5 张图与真实 30 条全量 Judge 已生成/执行；
  API 服务与只读验收接口已在容器内验证。历史报告未被覆盖；3/30 真实 smoke 归档在
  `reports/judge-real-3q/`。

---

## 2. 阶段状态

| 阶段 | 内容 | 状态 | 依据 |
|---|---|---|---|
| ① | 项目结构、技术选型、模块职责、数据流、实验架构 | 历史阶段 | `docs/architecture.md` |
| ② | PostgreSQL / pgvector / Schema / HNSW / 连接层 | 历史阶段 | `docker/init.sql`、`src/db.py`、`make db-check` |
| ③ | Embedding 与 150 条数据导入 | 历史阶段 | 数据库 150 documents / 150 chunks；数据哈希见 ④/⑤/⑥ 报告 |
| ④ | HNSW 建索引与参数 Benchmark | PASS | `reports/dense_benchmark.md` / `.json` |
| ⑤-A | Sparse Retrieval 实现 | PASS | `src/retrieval/sparse.py`、`tests/test_sparse_retrieval.py` |
| ⑤-B | Sparse 独立评测与误差分析 | PASS | `reports/sparse_benchmark.md` / `.json` |
| ⑥-A | Hybrid 前置核验与实验设计 | PASS WITH MINOR ISSUES | 设计核验已交付；无独立运行报告 |
| ⑥-B | Hybrid / RRF 五臂评测 | PASS WITH MINOR ISSUES | `reports/hybrid_benchmark.md` / `.json` |
| ⑦-A | Reranker 前置核验与设计 | PASS WITH MINOR ISSUES | 只读核验报告（会话交付）；无独立运行报告 |
| ⑦-B | Reranker 实现与评测 | PASS WITH MINOR ISSUES | `reports/rerank_benchmark.md` / `.json` |
| ⑧ | RAG 端到端问答链路集成与评测 | 离线实测完成；**待人工验收**（自评 PARTIAL） | `reports/rag_benchmark.md` / `.json` |
| ⑨ | README / roadmap 定稿 | **文档已定稿；待人工验收** | `README.md`、本文 |
| ⑩ | 项目最终整合与 GitHub 提交准备（当时 FastAPI 加分项未实现；见 ⑪） | **交付检查完成；待人工确认**；Git 已初始化，工作区有未提交改动，未 commit / push | `README.md` 的交付检查、本文章节；只读核对报告、依赖、忽略配置与敏感信息；`git rev-parse` 确认仓库存在 |
| ⑩-B | P0 文档整改（`reports/results.md` / `.dockerignore` 说明 / 生产治理设计） | **文档整改完成；待人工确认**；未执行 P1/P2 | `reports/results.md`、`README.md`、本文 |
| ⑪ | P0+P1+P2 迭代：统一评测/报告、索引与分块实验、CLI、FastAPI、容器编排 | **PARTIAL**：静态检查与纯单元测试通过；容器内统一检索 / 索引 / 分块 / RRF 敏感性 / 全量 pytest（664 passed、1 skipped）已运行；`final_report.md` + 图已由 ⑫ 补上并按 30/30 Judge 重生成（`Report status: PASS`）；API 服务与验收接口已在容器内验证 | 见第 10 节；`reports/unified/*.json`、`reports/judge/`、`reports/container/*.log` |
| ⑫ | LLM-as-a-Judge（Context Relevance / Faithfulness） | **PASS（自动化评估）**：代码、宿主静态检查与定向/全量测试通过；容器 mock smoke 与全量 pytest（`691 passed, 1 skipped`）通过；**真实 30/30 全量 Judge 已执行**（`reports/judge/`，`status=PASS`、0 失败、0 NOT RUN，CR mean 3.9667、Faithfulness mean 5.0000，`same_model=true`）；3/30 真实 smoke 归档在 `reports/judge-real-3q/`；最终报告/分布图已按新 Judge 重生成（`Report status: PASS`）；人工校准仍未运行 | 3.7；`src/evaluation/judge*.py`、`scripts/evaluate_judge.py`、`tests/test_judge*.py`、`reports/judge/`、`reports/judge-real-3q/` |
| ⑬ | 页面验收（只读 artifact 看板） | **PARTIAL**：后端路由与静态页已写入；宿主 Ruff 与 mypy PASS、定向 pytest PASS（0 failed、3 skipped）；容器 `api` healthy，`/healthz` 与 `/acceptance/api/summary` PASS，全量 Judge 后接口返回 `status=PASS`、6/6 检查通过；浏览器渲染/静态页请求仍为 `NOT RUN`。页面不是人工验收结论，不写仓库 | 10.7；`src/api/acceptance.py`、`web/acceptance/`、`tests/test_api_acceptance.py`、`tests/test_acceptance_page.py` |

> **阶段编号说明**：`docs/architecture.md` 是阶段 ① 的设计稿，正文保留设计期的
> 阶段规划；阶段 ⑩ 已在文首加入校正说明，统一采用“⑨ README / roadmap 定稿、
> ⑩ 项目最终整合与 GitHub 提交准备”；FastAPI 在 ⑪ 中写入 `src/api`，其 `api`
> 服务已在容器内启动，`/healthz`、`/acceptance/api/summary`、`/api/v1/search` 与
> `/docs` 已实际请求（见第 10.7、10.8 节）；浏览器渲染、`/readyz` 与 `/answer`
> 仍未验证。状态以 README 和本文为准。

> **Hybrid 报告的措辞不精确（已核实数据）**：`reports/hybrid_benchmark.md` 的
> `dataset.distinct_gold_docs` 为 **38**，限制段落却写
> “Sparse OR already retrieves all **43** gold documents”。本阶段对
> `data/eval_queries.json` 做只读统计：30 条 Query 共 **43 条 gold 引用**、
> **38 篇不同文档**。因此正确的表述是“43 条 gold 引用 / 38 篇不同 gold 文档”，
> 报告原文把引用数写成了文档数；这不改变 Sparse OR Recall 1.000 的结论。
> 历史报告未修改。

---

## 3. 可复核指标

以下数值逐字取自报告；括号内为 Recall / MRR / Hit@1 时，Recall 均为报告中的
Semantic Recall（macro），不是 ANN Recall。

### 3.1 Dense（④）

| 项目 | 值 |
|---|---|
| 状态 / 口径 | PASS；30 queries；K=10；150 docs / 150 chunks；宿主 Python |
| 臂 | exact seq scan、natural plan、forced HNSW `ef_search=10/40/100` |
| Semantic Recall mean | 0.944（五个臂相同） |
| ANN Recall@K mean / min | 1.000 / 1.000（五个臂相同） |
| query P50 / P95 | 0.778–1.022 ms / 0.960–1.185 ms（随臂变化） |
| 计划 | natural plan 为 `Seq Scan`；HNSW 臂仅在强制设置下使用 |
| 限制 | 150 chunks 下 HNSW 没有可观察延迟优势；不是规模结论 |

来源：`reports/dense_benchmark.md` / `.json`。

### 3.2 Sparse（⑤-B）

| 项目 | 值 |
|---|---|
| 状态 / 口径 | PASS；30 queries；K=10；English FTS；OR 默认 |
| OR Semantic Recall | 1.000（min/max 1.000；perfect 30/30；empty 0） |
| AND control Semantic Recall | 0.133（perfect 4/30；zero-recall 26；empty 26） |
| query P50 / P95 | 2.054 / 2.698 ms |
| db P50 / P95 | 3.086 / 3.865 ms |
| 中文探针 | 5 条合成探针均为 0 命中；语料是英文，不能据此判断中文检索质量 |
| 中文 fallback | 两字符子串匹配，不是分词；混合中英文查询只走 CJK 路径 |

来源：`reports/sparse_benchmark.md` / `.json`。

### 3.3 Hybrid / RRF（⑥-B）

| 臂 | Recall macro | Recall micro | MRR | Hit@1 | P50 ms | P95 ms |
|---|---|---|---|---|---|---|
| Dense | 0.944 | 0.884 | 0.917 | 25/30 | 75.062 | 94.624 |
| Sparse OR | 1.000 | 1.000 | 0.940 | 27/30 | 3.750 | 5.608 |
| Sparse AND | 0.133 | 0.093 | 0.133 | 4/30 | 1.343 | 1.716 |
| Candidate union | 1.000 | 1.000 | 0.917 | 25/30 | 78.733 | 98.924 |
| RRF | 1.000 | 1.000 | 0.950 | 27/30 | 78.714 | 98.862 |

- 30 queries；`top_k=10`、`candidate_k=10`、RRF `k=60`；5 次计时 + 1 次 warm-up。
- 五个 Dense 漏检 gold 均进入 RRF Top-10：Q-004 / DOC-0017、Q-005 / DOC-0021、
  Q-009 / DOC-0039、Q-009 / DOC-0040、Q-014 / DOC-0067。
- 43 条 gold 引用 / 38 篇不同 gold 文档，见 §2 注。
- 限制：一个文档一个 chunk；Sparse OR 已 Recall 1.0；英文；历史报告未包含
  Precision/nDCG（代码已实现并测试，标签非穷尽，只能作为补充指标）；宿主运行；
  150 chunks。

来源：`reports/hybrid_benchmark.md` / `.json`。

### 3.4 Reranker（⑦-B）

| 项目 | 值 |
|---|---|
| 状态 / 口径 | PASS；30 queries；candidate pool / final Top-K = 10 / 10；batch 16；max length 512；activation identity；宿主 CPU |
| 模型 | `BAAI/bge-reranker-base` @ `2cfc18c9415c912f9d8155881c133215df768a70` |
| RRF baseline | Recall/MRR/Hit@1 = 1.0000 / 0.9500 / 27/30（历史 ⑥-B） |
| RRF + Reranker | Recall/MRR/Hit@1 = 1.0000 / 0.9833 / 29/30（实时重排） |
| MRR 变化 | improved 3 / equal 26 / worse 1；平均 ΔMRR +0.0333 |
| 首个 gold 平均名次变化 | −0.0667（负值表示提前） |
| 模型加载 | 25,992.8 ms |
| 内存 | 加载后 RSS 848.1 MiB；峰值 RSS 1,288.9 MiB / WSET 1,314.6 MiB；最低可用 1.123 GiB |
| 延迟 | 10 候选 batch 1/16：1,163.2 / 1,082.5 ms；50 候选 batch 1/16：5,742.2 / 4,599.4 ms；约 92–116 ms/candidate |
| 限制 | 四个基线臂是历史 ⑥-B，不是本次重新查询；候选池相同，Recall 不可能上升；Embedding 与 Reranker 未同进程加载 |

来源：`reports/rerank_benchmark.md` / `.json`。

### 3.5 RAG（⑧）

| 项目 | 值 |
|---|---|
| 状态 / 口径 | **PARTIAL**；30 queries；mock Provider；historical RRF 重放 + 实时 PostgreSQL 文本 + 实时 Reranker |
| 端到端成功率 | 1.0000（30/30，status ∈ {ok, insufficient_evidence}） |
| 生成失败率 / 证据不足率 | 0.0000 / 0.0000 |
| 引用有效率 | 1.0000（30/30） |
| 证据 Recall macro / micro | 1.0000 / 1.0000 |
| 证据 MRR | 0.9833 |
| total P50 / P95 | 970.9 / 1082.2 ms |
| retrieval P50 / P95 | 970.1 / 1081.4 ms |
| LLM mock P50 / P95 | 0.07 / 0.09 ms |
| 真实 LLM 调用 | False；`real_llm_evaluation=not_run` |
| 未计算 | answer correctness、Answer Relevance；Context Relevance / Faithfulness 由独立的 Judge 模块评估，见 §3.7 |

来源：`reports/rag_benchmark.md` / `.json`。

### 3.6 Windows 崩溃调查（Stage A）

| 项目 | 值 |
|---|---|
| 审核对象 | E-5A 单候选、E-5C 空候选 |
| E-5A | 1 个候选；1 次 `rerank` / `predict` / `forward`；分数有限；ID 映射保持；无新 WER/minidump；**PASS** |
| E-5C | 空结果；`model_factory=0`、`predict=0`、`forward=0`；模型未加载；torch 未导入；**PASS** |
| 旧证据 | 九项关键哈希 PASS；E-1 完整清单 PASS；E-4R 完整清单 PASS；E-2/E-4 完整目录 **INCONCLUSIVE** |
| 审核时点内存 | 3.842 / 3.848 GiB，低于 E-5B 的 4.5 GiB 门槛，**FAIL** |
| 根因 | **未确认**；历史 `0xC0000005` / `torch_cpu.dll` / `0x8e4a269` 证据仍在 |

来源：`D:\RAG_crash_evidence\E5_REVIEW_2026-10-19_140237\E5_REVIEW_REPORT.md`。

### 3.7 LLM-as-a-Judge（Context Relevance / Faithfulness）

| 项目 | 值 |
|---|---|
| 实现 | `src/schema/judge.py`、`src/evaluation/judge_prompts.py`、`judge.py`、`judge_evaluation.py`、`scripts/evaluate_judge.py`、`make judge`、`python -m src.cli evaluate-judge` |
| 指标 | Context Relevance 1–5（相关/不相关 context ids）；Faithfulness 1–5 + claim 级 `supported` / `partially_supported` / `unsupported` / `contradicted` |
| 输入 | `reports/judge/rag_answers.json` 或 `--rag-artifact` 的 `per_query`（问题、答案、实际送入生成模型的 `E1..En` 证据）；不读 `gold_response` |
| 产物 | `reports/judge/judge_results.jsonl`、`judge_summary.json`、`judge_report.md`；最终报告的 optional `judge` 章节与 `judge_distribution.png` |
| 聚合 | 均值/中位数/1–5 分布只使用 `status=success`；失败按 status / error category 计数，不折算为 0 |
| 状态语义 | PASS = 真实 RAG + 真实 Judge + 30/30 两指标成功；Mock、3/30 smoke、部分失败 = PARTIAL；无有效分数 = FAIL；缺 Key = BLOCKED（退出码 2，不写 artifact） |
| 本轮执行 | **PASS（真实 30/30 全量 Judge 已执行）**：Ruff / mypy / 定向 pytest（36，3 skipped）通过；容器 mock smoke 写入 `reports/judge-smoke/`（3/30，fixture）；容器全量 pytest `691 passed, 1 skipped`。全量真实运行写入 `reports/judge/`：30/30、0 失败、0 NOT RUN、`queries_with_judge_failure=0`、`Integrity=True`；CR mean 3.9667 / median 4（2×1、3×4、4×20、5×5，低分 1 条 Q-029）、Faithfulness mean 5.0000 / median 5（5×30）；`judge_is_real=true`、`same_model=true`。3/30 真实 smoke 归档在 `reports/judge-real-3q/`。`final_report.md` 与 `judge_distribution.png` 需按新 Judge 重生成；人工一致性校准仍未运行；未以 Mock 或历史数据代替真实评分 |

---

## 4. 状态分类

### 4.1 已实现且有可复核报告

- `document` 分块与 150 条数据导入（历史 Benchmark 使用的唯一策略）。
- Dense 检索（pgvector，cosine，HNSW 索引存在）。
- Sparse 检索（PostgreSQL FTS，OR 默认，AND 对照）。
- 候选并集与 RRF 融合。
- Cross-Encoder Reranker。
- RAG 上下文组装、证据编号、提示词、OpenAI 兼容 Provider、引用校验、结构化答案。
- 30 条 Query 的 Dense / Sparse / Hybrid / Rerank / RAG 离线评测。

对应报告：`reports/dense_benchmark.md`（PASS）、`sparse_benchmark.md`（PASS）、
`hybrid_benchmark.md`（PASS）、`rerank_benchmark.md`（PASS）、
`rag_benchmark.md`（PARTIAL）。

### 4.2 已实现但只有单元/模块级证据

- Provider 超时、限流、网络、5xx、401 等错误映射与有界重试。
- 上下文单块截断、总量截断与按 rank 丢弃。
- 引用标签校验；无效标签进入 `invalid_citations`。
- 空候选短路；重复候选身份、非法输入、非有限分数等失败路径。
- 检索 / 重排 / 证据 / 生成失败的分类。
- `fixed_window` / `structured` 分块、`context_header` 与 `ChunkingConfig` 校验
  （三种策略均已在容器内入库：150 / 267 / 161 chunks）。
- Recall@K / Precision@K / MRR / 二元 nDCG@K 的纯函数与记录级实现。
- **检索作用域隔离（P0）**：`chunk_strategy` 作为 SQL 绑定参数出现在 Dense 查询、
  Sparse 英文 FTS 查询与 CJK fallback 中；`_rows_to_hits` 拒绝与请求作用域不一致
  的行。`tests/test_retrieval_scope.py` 用记录连接验证参数位置与过滤条件。
  历史 RRF 重放（`HistoricalRRFRetriever` 与 rerank 的历史校验）也会拒绝
  `parameters.chunk_strategy` 或命中策略不一致的 artifact。
  **已在含三种策略、共 578 chunks 的容器数据库上运行**：分块对比 per-strategy
  30/30 查询完成；benchmark 的命中策略校验未报错（见
  `reports/unified/chunking_benchmark*.json`）。
- **LLM-as-a-Judge（Context Relevance / Faithfulness）**：prompt 构建、JSON 解析、
  1–5 分数与 `E<n>` 标签校验、claim 支持状态、失败隔离与汇总均已写入，且有纯单元
  测试（`tests/test_judge.py`、`tests/test_judge_evaluation.py`）。容器 mock smoke
  （`reports/judge-smoke/`，3/30）、真实 3 条 smoke（归档在 `reports/judge-real-3q/`）
  与真实 30/30 全量运行（`reports/judge/`，PASS、`same_model=true`）均已执行；
  人工一致性仍未校准（见 3.7）。
- **索引管理（P0）**：`src/ingestion/indexer.py` 提供
  `build_hnsw_index()` / `build_ivfflat_index()` / `drop_index()` /
  `list_vector_indexes()` / `collect_index_stats()`，默认创建带版本号的 shadow
  索引并保护 bootstrap 索引 `chunks_embedding_hnsw_idx`。`tests/test_index_management.py`
  覆盖命名、参数边界、标识符校验与删除保护。索引实验已在容器内执行 DDL 并留下
  `reports/unified/index_benchmark.json`，但该 artifact 为 **PARTIAL**
  （`synthetic_scale=1`，14 组中 1 组失败），详见 4.3。

证据：`tests/test_rag_*.py`、`tests/test_reranker.py`、`tests/test_retrieval.py`、
`tests/test_fusion.py`、`tests/test_chunking.py`、`tests/test_chunker.py`、
`tests/test_metrics.py`、`tests/test_retrieval_scope.py`、`tests/test_index_management.py`、
`tests/test_hybrid_benchmark.py`、`tests/test_rerank_benchmark.py`、
`tests/test_judge.py`、`tests/test_judge_evaluation.py`、`tests/test_report.py` 等。
测试分类、命令与实际执行状态见 README 的“测试与静态检查”。Judge 轮实测：宿主
`ruff check src scripts tests` 通过、`mypy src` 54 个文件无错误；宿主定向 Judge /
报告测试 36 个（3 skipped，缺 `matplotlib`）；容器内全量 `691 passed, 1 skipped`。
宿主全量与数据库集成用例仍未全量复跑。

### 4.3 部分验证

- **RAG 端到端**：正式 `rag_benchmark` 仍是 mock Provider + historical RRF 重放；
  另有独立真实 30/30 live 运行（真实检索 + 真实生成 + 真实 Judge，`reports/judge/`，
  PASS）。它评估的是 Judge 指标，不改变正式 `rag_benchmark` 的 mock 口径。
- **Reranker**：宿主 CPU；历史 RRF Top-10 池；未覆盖多候选、长输入、容器。
- **Dense / HNSW**：150 chunks；自然计划为 Seq Scan；无规模、并发或跨机器结论。
- **Docker**：数据库侧与 2026-10-20 的容器内 ingest / benchmark / 全量 pytest
  链路已验证（Judge 轮重跑为 691 passed、1 skipped）；真实 RAG + Judge 30/30 全量
  也在容器内完成；API 服务已在容器内验证，浏览器渲染未验证。
- **检索作用域**：已在 578 chunks 的多策略库上运行三策略评测；但所有策略共用
  同一个 HNSW bootstrap 索引，无法把索引大小或索引行为归因到单个策略。
- **索引管理**：首轮 `reports/unified/index_benchmark.json` 为 **PARTIAL**：
  HNSW / IVFFlat 的 DDL 已真实执行，`synthetic_scale=1`（仅 150 行），14 组配置中
  1 组失败（`m=32, ef_construction=32` 违反 pgvector 约束）。随后用合法网格与
  `synthetic_scale=10`（1500 行）重跑，得到 PASS 的
  `reports/unified/index_benchmark_scale10.json`；旧 PARTIAL artifact 保留。
  合成向量只反映索引行为，不进入语义质量结论；真实语料的参数结论仍未验证。
- **分块对比**：`reports/unified/chunking_benchmark.json`（`dense/sparse_or/rrf`）
  与 `reports/unified/chunking_benchmark_rerank.json`（增加 `rrf_rerank`）均 PASS，
  30/30 查询 × 3 策略；指标为文档级检索指标，不是答案质量。所有策略共用
  一个 HNSW 索引，且 `chunk_strategy` 过滤在 ANN 扫描之后执行：`ef_search=5` 时
  全局 Top-5 中仅 3 条属于 `document`，单策略返回数可少于 K。策略间召回差异
  因此部分来自其在全局索引中的占比。
- **Dense benchmark 的强制 HNSW 臂**：多策略库上仅 `enable_seqscan=off` 会让
  planner 选择 `chunks_strategy_idx` + Sort。已改为同时 `enable_bitmapscan=off`、
  `enable_sort=off`，使 `ORDER BY distance LIMIT k` 只能由 HNSW 满足；plan 中
  仍必须出现 `chunks_embedding_hnsw_idx`，否则该组配置继续标 invalid。

### 4.4 未验证

- 30 条全量 RAG + Judge 已执行并通过（见 10.6）；真实网络错误路径仍只由重试
  代码覆盖，未注入故障验证。
- RAG live 全链路的重复运行稳定性（30 条单次全量已通过；并发、长输入与重复
  运行未验证）。
- 多候选 batch 推理 / E-5B。
- 历史 Windows 原生崩溃的根因。
- Docker 完整生产链路的浏览器渲染（30 条容器内 RAG 端到端与 API 服务已验证）。
- 宿主全量 pytest（目前宿主只有纯单元子集与 4 个指定 DB 用例有统计）。
- 中文与中英混合检索、生成与引用。
- 大规模性能、并发、吞吐、GPU。
- 跨机器复现性。
- FastAPI `/readyz` 与 `/answer` 的运行时请求（`/healthz` 与 `/acceptance/api/summary` 已验证）。
- `final_report.md` 与图表：已生成（5 张图，`Report status: PASS`），见 10.6。
- 30 条全量真实 LLM 评测的人工校准与 inter-judge 一致性（自动评估已完成）；最终提交整合。
- 索引参数扫描的真实规模结论：`synthetic_scale=10`（1500 行）的 PASS 只反映
  索引行为；真实语料上的参数结论未验证；旧的 `synthetic_scale=1` PARTIAL artifact
  保留，见 4.3。

### 4.5 已知问题与风险

1. **原生崩溃未结案**：E-5A/E-5C 只覆盖各自探针范围，不能外推。
2. **依赖漂移**：`pyproject.toml` 精确 pin 与宿主已安装版本不同；没有覆盖传递
   依赖的锁文件。报告中的 torch 为 2.13.0，声明为 2.14.1（其他包也有差异）。
   本轮只澄清声明 / 宿主实测 / 历史记录，不为数字一致而升级或降级 Torch；
   `psutil` 已声明为必需依赖，MIT `LICENSE` 已补齐。
3. **容器路径**：五份历史报告在宿主 Python 运行，不能当作容器结果；2026-10-20 的
   新实验已在容器内运行并记录容器环境字段；容器内全量 pytest 已通过
  （Judge 轮为 691 passed、1 skipped）。真实 RAG + Judge 30 条全量已执行并通过；
  `api` 服务与验收接口已在容器内验证，浏览器渲染仍未验证。
4. **完整 pytest 未在宿主重跑**：宿主 torch/transformers 组合存在已知原生崩溃风险。
   本轮已真实执行 23 个纯单元模块：506 个用例、504 passed（无 F/E）、2 skipped、
   `EXIT=0`。两个跳过是宿主机缺少 matplotlib（`test_report.py` 的两个 fig 用例）。
   **容器内全量套件已执行**：Judge 轮重跑为 691 passed、1 skipped、`EXIT=0`
   （30.42s，JUnit `reports/container/pytest-full-judge.xml`）；此前为 664 passed
   （`pytest-full.log`）。唯一 skip 是固定 150 chunks 的 natural-plan 断言
   （当前库为 578 chunks）。宿主的数据库集成与真实模型用例仍未全量复跑。
   **Ruff 与 mypy 已在宿主机真实执行并通过**（Judge 轮 `All checks passed!`；
   `Success: no issues found in 54 source files`）。见 README 的“测试与静态检查”。
5. **中文能力有限**：Sparse 只有两字符子串 fallback，不是分词；没有中文语料验证。
6. **指标口径**：标签非穷尽；Precision@K / 二元 nDCG@K 已实现并有单元测试，
   但只能作为补充指标，历史报告未包含；Recall/MRR 不能代表生成质量。
7. **RAG 是重放**：Dense/Sparse/RRF 未在 ⑧ 重新执行；证据指标衡量被重放的候选池。
8. **mock 答案**：以 `MOCK` 开头，不是模型输出。
9. **E-2/E-4 完整证据完整性 INCONCLUSIVE**：缺少完整清单，不能推断为 PASS。

---

## 5. 未完成验证（逐项）

以下项目在 README 或报告中仍为未验证；没有新的执行授权前，不应写成“通过”：

1. **多候选批量推理**：E-5B 未执行；现有 `e5a_rerank_single.py` 硬编码单候选，
   需要新的独立多候选探针，且尚未准备/审核。
2. **历史 Windows 原生崩溃根因**：`0xC0000005` / `torch_cpu.dll` 未定位；
   E-5A/E-5C 未复现不等于修复。
3. **真实 LLM 调用**：已完成 30/30 真实 RAG + Judge 全量运行（10.6，PASS）；
   但正式 RAG 报告指标仍使用 mock Provider，未用真实 LLM 重跑该报告。
4. **完整 RAG 稳定性**：live 链路在同进程加载 Embedding + Reranker 时曾原生崩溃；
   30/30 真实 RAG + Judge 已在容器内以 live 模式跑通（一次运行），但重复运行与
   长时间稳定性仍未验证；正式 `rag_benchmark` 仍用 historical 重放。
5. **pytest**：宿主纯单元组合已执行（506 用例、2 skipped、`EXIT=0`），并单独运行
   4 个 DB planner/benchmark 用例；容器内全量套件已执行，Judge 轮为 691 passed、
   1 skipped（JUnit `reports/container/pytest-full-judge.xml`；此前 664 passed，
   日志 `reports/container/pytest-full.log`）。唯一 skip 是固定 150 chunks
   的 natural-plan 断言（当前库为 578 chunks）。`tests/test_dense_benchmark.py` 的
   DB 集成用例在 578 chunks 上暴露并修正了 forced-HNSW planner 设置与 Top-K 断言
   （见 10.4.1）。宿主全量仍未复跑。
6. **Docker 全链路**：容器内 `db-check`、三策略 ingest、模型离线加载、统一检索
   benchmark（6 臂）、索引实验、分块对比、RRF 敏感性、全量 pytest（691 passed、
   1 skipped）与真实 RAG + Judge 30/30 live 运行已执行；API 服务与只读验收接口
   已在容器内验证。五份历史 benchmark 脚本本身仍未在容器内重跑。
7. **中文 / 混合语言**：检索、生成、引用均未评测；Sparse 中文探针 0 命中。
8. **大规模性能**：只有 150 chunks；没有 10 万 / 100 万 / 1000 万级数据。
9. **并发**：所有延迟为单进程顺序执行；无并发、吞吐、连接池压力测试。
10. **跨机器复现**：没有第二台机器或独立环境的复现记录。
11. **P0+P1+P2 迭代的执行验证**：Ruff / mypy 与纯单元 pytest（506 用例、2 skipped、
    `EXIT=0`）在宿主机通过；容器内已完成 `db-check`、三策略 ingest、模型离线加载、
    统一检索 benchmark、索引实验（PARTIAL）、分块对比（含 `rrf_rerank`）、RRF
    敏感性与全量 pytest（664 passed、1 skipped、`EXIT=0`），artifacts 见第 10 节。
    `final_report.md` + 5 张图已生成，真实 30/30 Judge 已执行（报告/图待重生成）；
    API 服务与验收接口已在容器内验证。
12. **依赖精确 pin 的重新安装核验**：未在隔离环境按 `pyproject.toml` 重装并重跑。
13. **E-2/E-4 完整目录完整性**：缺少完整 `SHA256SUMS.txt`，INCONCLUSIVE。
14. **生成质量指标**：answer correctness、Answer Relevance 未计算；Context Relevance /
   Faithfulness 已完成 30/30 真实全量 Judge（PASS），但仍是同模型自评、没有人工
   校准，不能当作正确性真值。

---

## 6. 后续路线（未承诺）

### 6.1 近期

1. **阶段 ⑨ 人工验收**：审阅 README 与本文；只改文档，不执行测试。
2. **决定是否继续崩溃调查**：若继续，先准备并审核独立多候选探针（E-5B），
   再单独申请执行授权；执行前需满足内存门槛、无重型 Python/Torch 进程、
   本地模型 revision 一致、全新证据目录、WER/minidump 监控与明确超时。
3. **真实 LLM 评测**：30/30 RAG + Judge 已完成（10.6，PASS）；若要用真实 LLM 重跑
   正式 `rag_benchmark`，仍需明确预算与网络条件，并继续区分真实结果与 mock。
4. **Docker 全链路**：容器内已完成 `db-check`、三策略 ingest、统一检索 / 索引 /
   分块 / RRF 敏感性、模型离线加载、全量 pytest、真实 RAG + Judge 30/30 与 API
   服务；仍需补依赖 pin 差异记录与真实 LLM 版正式 `rag_benchmark`。
5. **完整 pytest**：在隔离环境或容器内运行；若复现原生崩溃，保留证据后再分析。

### 6.2 中期

- 中文语料接入：`zhparser` / `pg_jieba`、匹配的 `tsvector`、重建索引，再谈中文指标。
- 更大规模评测：10 万 / 100 万级合成或真实语料，测 HNSW `ef_search`、内存、
  建索引时间、召回与延迟；当前 150 chunks 不能作为规模结论。
- 并发与吞吐：连接池、批量查询、reranker 并发、GPU/CPU 对比。
- 生成质量：已完成 30 条全量 LLM-as-a-Judge；仍需 rubric、人工校准与独立模型复核，
  并与检索指标分开报告。
- FastAPI + Docker：`src/api` 与 compose `api` 服务已启动，`/healthz`、
  `/acceptance/api/summary`、`/api/v1/search`、`/docs` 已实际请求；仍需验证
  `/readyz`、`/answer` 与浏览器渲染，并补故障注入下的 API 行为测试。

### 6.3 验收门槛

- 任何“PASS”必须有对应报告或原始证据；缺失证据标 INCONCLUSIVE，不替代 PASS。
- 真实 LLM、Docker 全链路、完整 pytest、规模/并发/跨机器必须分别授权与记录。
- 不修改历史证据；不通过降低内存门槛或跳过 preflight 让测试“跑起来”。
- 不把“未观察到崩溃”写成“根因已修复”。

---

## 7. 生产治理设计（10M 规模，规划设计 / 待验证）

> **状态：规划设计，未实现、未验证。** 当前实测只有 150 documents / 150 chunks；
> 本节不改变 README 的阶段验收结论，也不代表任何生产部署、容量或合规能力已通过。
> 所有容量、延迟、缓存命中率与成本数字都需要在目标规模重新测量。

### 7.1 千万级数据规模与索引演进

- **设计方案**：按租户/卡产品/生效年份做逻辑分区，或按 `tenant_id` 哈希分片；
  每个分区独立维护 pgvector HNSW 索引，查询时按路由键选择分区并在网关合并 Top-K。
  索引使用版本化命名（如 `chunks_embedding_hnsw_vNNN`），新索引以 shadow build
  方式构建，完成后切换读路径；删除旧索引前保留回滚窗口。大规模写入采用批量
  embedding、增量 upsert、`CREATE INDEX CONCURRENTLY`（或等价的离线构建窗口）
  和 `VACUUM/ANALYZE` 维护；重建前先做容量与召回回归。
- **关键风险/失败模式**：分区倾斜导致热点；跨分区 Top-K 合并改变排序；HNSW 构建
  内存不足或超时；重建期间双写不一致；量化/压缩导致 Recall 下降；索引版本切换后
  旧缓存仍指向旧索引。
- **验收指标/监控项**：索引构建时间、索引大小、分区行数偏差、ANN Recall@K 对 exact
  基线、P95/P99 延迟、构建失败率、回滚耗时、双写不一致行数。
- **当前未验证**：没有 10M 级数据；没有分区/分片、并发建索引、量化召回或 shadow
  build 的实测；150 chunks 的 Seq Scan/HNSW 结果不能外推。

### 7.2 Redis 查询缓存

- **设计方案**：缓存键包含 `tenant_id`、权限/角色版本、`policy_version`、
  `index_version`、检索配置与规范化 Query 的哈希，例如
  `rag:{tenant}:{acl_ver}:{policy_ver}:{index_ver}:{config_hash}:{query_hash}`；
  不把原始 Query 或 PII 直接写入键。TTL 按策略类型设置（规划为分钟级，具体值待定），
  并在响应中记录缓存版本。策略更新通过版本号自增主动失效；可选 Redis Pub/Sub 或
  失效频道广播；缓存未命中时使用 single-flight/短锁避免击穿；空结果使用短 TTL 的
  负缓存，避免穿透。
- **关键风险/失败模式**：策略更新后仍返回旧答案；跨租户/跨权限命中；缓存键碰撞；
  缓存击穿或雪崩；Redis 故障时同步拖垮检索；失效广播延迟；缓存中残留已删除文档。
- **验收指标/监控项**：命中率/未命中率、stale hit 率、失效传播延迟、缓存 P95、
  错误率、驱逐率、跨租户命中测试通过率、策略版本分布。
- **当前未验证**：没有 Redis 部署、容量、失效链路或故障注入测试；当前代码没有
  查询缓存实现。

### 7.3 PII 与数据最小化

- **设计方案**：入库前做字段白名单与敏感字段识别（姓名、卡号/账号、证件号、电话、
  邮箱、地址、交易与催收备注等）；对必须保留的字段做脱敏/令牌化，默认不把 PII
  送入 embedding；日志、trace、缓存与评测报告统一脱敏；设定保留期限与删除传播
  任务，删除时同步清理向量、缓存、备份索引与派生日志；对评测样本使用合成或脱敏
  数据。
- **关键风险/失败模式**：PII 进入 embedding 后难以删除；日志/缓存泄露；删除未传播
  到备份；脱敏规则漏掉自由文本；跨租户样本混入评测集。
- **验收指标/监控项**：PII 扫描命中数、脱敏覆盖率、删除 SLA、删除传播成功率、
  日志抽检违规数、评测集 PII 抽检结果。
- **当前未验证**：没有 PII 分类器、脱敏流水线、保留期限任务或删除传播测试；
  当前语料是合成/政策文档，不能证明生产 PII 处理能力。

### 7.4 访问控制与租户隔离

- **设计方案**：接入 SSO/OIDC 做身份认证；授权在服务层按租户、角色、卡产品与数据
  范围判定；数据库使用行级安全（RLS）或强制 `tenant_id` 过滤，服务端不接受客户端
  自报的租户身份；检索、缓存、审计与模型调用都携带同一授权上下文；管理操作使用
  独立角色与最小权限；跨租户访问做显式拒绝与告警。
- **关键风险/失败模式**：漏加 tenant 过滤导致越权；缓存键缺少租户/权限维度；
  embedding 或索引跨租户共享；管理员绕过；错误信息泄露其他租户数据；授权版本
  更新滞后。
- **验收指标/监控项**：越权尝试次数、授权决策延迟、跨租户回归测试通过率、
  RLS 覆盖率、审计完整率、权限变更传播延迟。
- **当前未验证**：没有认证/授权、RLS、租户隔离或跨租户渗透测试；当前是单租户
  本地评测。

### 7.5 审计与可观测性

- **设计方案**：每次检索/问答记录 `request_id`、租户、角色、Query 哈希（不记录
  原始 PII）、检索配置、`index_version`、`policy_version`、证据 ID、引用 ID、
  各阶段延迟、状态/错误分类、Provider/模型版本、Prompt 版本、缓存命中与 token/
  成本。指标按阶段拆分：embedding、dense、sparse、fusion、rerank、generation；
  建立错误预算与告警；审计日志写入不可篡改或受控存储，并设置保留期限。
- **关键风险/失败模式**：审计日志本身含 PII；日志量/成本失控；缺少 `request_id`
  关联；时钟偏差；只记录成功请求；指标口径不一致。
- **验收指标/监控项**：P50/P95/P99、分阶段错误率、空结果/证据不足率、引用有效率、
  缓存命中率、每 Query 成本、索引/策略版本分布、审计写入成功率、告警响应时间。
- **当前未验证**：没有生产级日志聚合、指标看板、审计存储或成本核算；现有报告
  只有离线评测字段。

### 7.6 金融场景的引用可追溯性

- **设计方案**：建立不可变的引用链：`answer_id → citation E<n> → doc_id → chunk_id
  → source_document_version → policy_code → effective_from/effective_to → 原文位置
  （章节/段落/字符偏移）→ index_version → retrieval_run_id`。保存文档内容哈希或
  版本快照，确保 chunk 重切后仍能定位原文。政策更新或失效时，历史答案标记
  “政策版本已过期/已失效”，展示生效时间并要求重新检索；不静默改写历史答案。
  对外展示引用时同时展示文档标题、Policy Code、版本与生效日期。
- **关键风险/失败模式**：引用指向旧版本或已删除文档；chunk 重切导致偏移失效；
  政策更新后答案仍被当作现行有效；无法从审计日志重建引用链；多版本文档混淆。
- **验收指标/监控项**：引用可解析率、版本匹配率、过期政策答案占比、引用链重建
  成功率、政策更新检测延迟、人工抽检一致率。
- **当前未验证**：没有版本化政策库、原文偏移映射、历史答案再校验或引用链审计；
  当前 RAG 只验证 `E<n>` 指向提供的证据，不验证引用是否支持答案每一句话。

### 7.7 失败模式与降级策略

- **设计方案**：缓存故障时绕过缓存直接检索，并对上游限流；检索/数据库故障时返回
  明确的不可用状态，不生成无证据答案；模型不可用时返回 `insufficient_evidence`
  或预定义的抽取式降级，并在响应中标记降级原因；重排失败时回退到 RRF 排序必须
  显式标记并审计，不能静默；策略更新失败时继续提供上一个已知良好版本并展示版本
  警告，过期策略则拒绝回答；索引重建失败时回滚到上一索引版本。
- **关键风险/失败模式**：静默返回旧/错答案；降级路径绕过租户或权限；重试风暴；
  缓存/索引回滚不完整；故障不可见；降级状态被误写成正常。
- **验收指标/监控项**：错误预算、按原因分类的降级率、stale answer 率、恢复时间
  （RTO）、回滚成功率、重试次数、告警覆盖率。
- **当前未验证**：没有故障注入、混沌测试或生产 runbook；现有实现只覆盖部分
  单元级错误分类，未验证真实依赖故障下的端到端行为。

---

## 8. 文档一致性与待核实项

以下差异在本次只读检查中发现；本阶段只记录，不修改业务代码或历史证据。

1. **阶段编号**：`README.md` / 本文采用“⑨ README / roadmap 定稿、⑩ 最终整合与
   GitHub 提交准备”；`docs/architecture.md` 是阶段 ① 设计稿，已在阶段 ⑩ 加入
   校正说明并改写编号与过时路径。三份文档的当前编号以此为准。
2. **报告路径**：`docs/architecture.md` 原设计把 `reports/results.md` 与
   `reports/figures/` 描述为由 `make report` 生成的产物。现在 `reports/results.md`
   已在 Stage ⑩-B 作为**手工只读汇总**创建，不是自动生成，也不替代五份原始报告；
   `make report` / `cli.report` 在 2026-10-20 已写入但未执行验证。实际评测报告是
   `reports/{dense,sparse,hybrid,rerank,rag}_benchmark.*`；`reports/figures/`
   当前不存在（也没有生成的图表）。P1-10 已修正 `.dockerignore`：
   `reports/results.md` 与 `reports/figures/*.png` 不再被排除，Dockerfile 的
   `COPY . .` 会把交付报告带入镜像；仅 `reports/runs/` 作为中间运行日志被排除。
   本轮没有重建镜像，容器内可见性仍需在下次 `docker build` 时验证。
3. **Makefile 计划目标（2026-10-20 已更新，未验证）**：`make index` 已拆分并替换为
   `index-build` / `index-list` / `tune`；`report` / `pipeline` 现在转发给已实现的
   CLI 子命令。目标存在且不再引用占位命令，但**没有执行过**；不能据此写成可运行能力。
4. **API（2026-10-20 已更新，容器内已验证）**：`src/api` 提供 `/healthz`、
   `/readyz`、`/answer`、`/api/v1/search` 与 Swagger；compose 已启用 `api`
   profile 与 `warm-cache` 服务，Dockerfile 安装 `.[dev,report,api]`。
   `api` 服务已在容器内启动，`/healthz`、`/acceptance/api/summary`、
   `/api/v1/search` 与 `/docs` 已实际请求；`/readyz` 与 `/answer` 仍未实际请求。
   `tests/test_api.py` 使用注入的 fake pipeline / retriever，不访问数据库或模型。
5. **Hybrid gold 口径**：经只读统计，`eval_queries.json` 为 43 条 gold 引用 /
   38 篇不同文档；报告“all 43 gold documents”措辞不准确，但 Recall 1.000
   的结论不变。历史报告未修改。
6. **依赖漂移**：`pyproject.toml` 精确 pin 与宿主实际版本不同；报告已记录，
   本轮未重装、未升级、未降级。声明版本、宿主实测版本、历史报告版本与锁定状态
   已在 README 的“依赖、版本与复现性”表中分开列出。宿主 site-packages 还存在
   同一包的多个 dist-info（pydantic、pydantic-settings、fastapi、uvicorn 等），
   说明它不是按当前 pin 干净重建的环境，不能作为锁文件来源。
   `matplotlib==3.11.2` 与 `tabulate==0.10.0` 已移入可选 `report` extra，
   当前源码仍未导入；P1 的 `report.py` 必须实际使用它们。
   `pyproject.toml` 已加入 `[[tool.uv.index]]` + `[tool.uv.sources]`，把
   Linux/Windows 的 Torch 指向 CPU 索引。工作区的 `uv.lock` 生成于该映射之前，
   尚未重新生成或验证；Torch 2.14.1 的 CPU wheel 与 Python/平台兼容性也尚未核实。
   在锁文件验证通过前，不把它纳入版本控制，也不把 Dockerfile 改为 `uv sync`。
7. **内存采样时间差**：Stage A 审核时点 14:10 为 3.842 / 3.848 GiB（FAIL）；
   审核后会话内只读采样 14:32:58 / 14:33:05 为 4,926 / 4,925 MB（约 4.81 GiB，
   高于 4.5 GiB）。后者未写入 Stage A 报告，不是对前者的修正；也没有据此执行 E-5B。
8. **E-2/E-4 完整性**：只有报告列出的关键文件哈希，没有完整清单；完整目录完整性
   为 INCONCLUSIVE，不得写成 PASS。
9. **原始 10 步路线**：除已记录的 ⑧/⑨/⑩ 外，其余编号与范围未在仓库中找到完整
   定义，**待核实**。
10. **Git 仓库**：项目根目录存在 `.git`，当前分支 `main`，已有历史提交；本轮
    检查时工作区仍有未提交改动（包括本轮修改）。本轮未执行 `git init`、
    `git add`、`git commit` 或 `git push`。
11. **本地凭据**：`.env` 中存在非空的 LLM API Key 配置（不在本文或终端输出值）。
    `.env` 已被 `.gitignore` 与 `.dockerignore` 排除；公开提交前仍建议轮换并向
    人工确认没有任何副本、压缩包或日志携带该值。
12. **Docker 构建上下文**：`.dockerignore` 已在 Stage ⑩-A 经授权加入 `.cache/`
    （第 24 行），因此当前 `docker build` 不会把本地约 1.1 GB 的
    `.cache/huggingface` 送入构建上下文。2026-10-20 镜像已重建（安装
    `.[dev,report,api]`），`hf_cache` named volume 中的两个模型可在
    `HF_HUB_OFFLINE=1` 下加载。`api` 服务已启动，`/healthz` 与
    `/acceptance/api/summary` 已返回预期结果；容器内 30 条 RAG + Judge 与全量
    pytest 也已执行。**仍未验证**：浏览器渲染、`/readyz` 与 `/answer` 的运行时请求。
    P1-10 只移除了报告相关的排除行并保留 `.env`、`.cache/`、虚拟环境等敏感/缓存
    排除。
13. **`psutil` 依赖**：`src/evaluation/rerank_benchmark.py` 在模块级导入
    `psutil`，因此它是必需依赖；`scripts/benchmark_rag.py` 的导入是可选/降级用法。
    本轮已把 `psutil==7.2.2` 加入 `pyproject.toml` 的 `[project] dependencies`，
    与历史报告记录的版本一致；未重新安装或重建容器。
14. **许可证文件**：`pyproject.toml` 声明 MIT，仓库已新增标准 MIT `LICENSE`；
    版权行使用项目集体署名 “Apex RAG contributors”（2026）。如果公开时需要使用
    具体的个人或组织作为版权持有人，应由人工确认后替换该行。
15. **报告中的本地路径**：`reports/rerank_benchmark.json`、`sparse_benchmark.json`
    含生成机器的绝对路径与模型缓存路径；属于历史证据，未改写。公开提交前应人工
    检查所有 Markdown/JSON 是否含本地绝对路径、用户名或缓存路径。公开版本可以使用
    清洗副本，但必须保留原始报告不被覆盖、不改变任何结论，并在副本中明确标注
    “公开清洗副本”；是否清洗及具体范围需要人工决定。
16. **Makefile 清理目标**：P1-10 已移除 `reports/results.md` 与
    `reports/figures/*.png` 的删除命令；`clean` 现在只删除宿主 Python 缓存
    （`__pycache__`、`.mypy_cache`、`.ruff_cache`、`.pytest_cache`），并改为在
    宿主执行以避免 `docker compose run` 启动数据库依赖。未运行 `make clean`
    做破坏性验证；清理范围通过阅读目标与静态检查确认。
17. **P0 检索作用域与索引管理（2026-10-20）**：所有检索语句现在带
    `chunk_strategy` 过滤，Benchmark 的环境门槛按策略统计 chunk 数；报告与
    JSON artifact 新增 `chunk_strategy`、`strategy_chunks`、`strategy_docs`
    字段，历史 RRF 重放会拒绝作用域不一致的 artifact。`indexer.py` 的索引
    管理 API 已在容器内真实执行 shadow 索引 DDL（见
    `reports/unified/index_benchmark.json`，状态 PARTIAL）。这些改动已通过
    宿主机 Ruff / mypy / 纯单元测试，并在 578 chunks 的容器库上运行；历史报告
    未改写，新结果全部写入 `reports/unified/` 与 `reports/container/`。
18. **历史报告与当前代码的口径差异**：历史五份报告由不带作用域过滤的代码生成，
    当时 `chunks` 表只有 `document` 策略，因此结论仍然有效；但报告中的
    环境字段没有 `strategy_chunks` / `strategy_docs`。不能把新代码的输出追加到
    历史 JSON 中，也不能用新字段反推历史运行。

---

## 9. 证据位置与更新规则

- 主报告：`reports/dense_benchmark.md` / `.json`、`sparse_benchmark.md` / `.json`、
  `hybrid_benchmark.md` / `.json`、`rerank_benchmark.md` / `.json`、
  `rag_benchmark.md` / `.json`。
- 跨报告汇总：`reports/results.md`（只读汇总，非新评测；不替代原始报告）。
- 2026-10-20 新实验 artifact：`reports/unified/benchmark.json`、
  `fusion_sensitivity.json`、`index_benchmark.json`（PARTIAL）、
  `chunking_benchmark.json`、`chunking_benchmark_rerank.json`、`rerank_20to5.json`；
  Judge 真实全量：`reports/judge/`（PASS，30/30）；3/30 真实 smoke 归档：
  `reports/judge-real-3q/`；mock 闭环：`reports/judge-smoke/`；运行日志与执行前
  哈希在 `reports/container/`。
  五份历史报告未被覆盖。
- 崩溃调查证据：`D:\RAG_crash_evidence\E1_2026-10-19_1055`、
  `E2_2026-10-19_1127`、`E4_2026-10-19_1134`、`E4R_2026-10-19_1141`、
  `E5_2026-10-19_1205\E5A`、`E5_2026-10-19_1207\E5C`、
  `E5_REVIEW_2026-10-19_140237`。
- 历史证据只读；不删除、不覆盖、不重新生成。需要新结论时，新建证据目录并记录
  命令、环境、哈希、内存与退出码。
- 报告重新生成会改变耗时等字段与哈希；README 的哈希块只在内容不变时有意义。
- 如果代码、配置或数据发生变化，先更新对应报告与 README 阶段表，再更新本文；
  本文不替代验收记录。

---

## 10. 本轮迭代（P0+P1+P2，2026-10-20）

> §10.1–10.4 是 P0+P1+P2 轮次的历史快照（当时 Judge 尚未接入）；其后的 §10.6
> 已补上 Judge 模块、真实 30/30 全量运行、最终报告与 5 张图，§10.7 记录容器 API
> 验收。两者冲突时以 §10.6 / §10.7 为准。

### 10.1 范围

按已批准的计划实施 P0 + P1 + P2：统一 arm 注册表与评测 schema、HNSW / IVFFlat
参数扫描、三种分块策略对比、RRF 敏感性、统一报告与图、完整 CLI、只读 FastAPI、
Docker 编排与预热缓存、文档更新。原则保持不变：不重建项目、不改原始数据与历史
报告、不做破坏性 Docker/数据库操作、不伪造实验结果。

### 10.2 实现状态

| 能力 | 实现状态 | 执行验证 | 关键文件 |
|---|---|---|---|
| 统一 arm 注册表（6 臂） | IMPLEMENTED_AND_VERIFIED | 容器内 6 臂 × 30 Query 全部完成（`reports/unified/benchmark.json`，PASS） | `src/retrieval/registry.py` |
| 统一评测 + artifact 安全写入 | IMPLEMENTED_AND_VERIFIED | 上述运行写入新路径；旧 artifact 哈希在执行前记录 | `src/evaluation/benchmark.py`、`src/evaluation/artifacts.py` |
| 索引参数扫描（HNSW 9 组 + IVFFlat lists/probes） | IMPLEMENTED_AND_VERIFIED | **PARTIAL**：DDL 已真实执行，14 组中 1 组失败，`synthetic_scale=1`（`reports/unified/index_benchmark.json`） | `src/evaluation/index_benchmark.py` |
| 分块策略对比（3 策略） | IMPLEMENTED_AND_VERIFIED | 容器内 3 策略 × `dense/sparse_or/rrf`（PASS），另有 3 策略 × 4 臂含 `rrf_rerank`（PASS） | `src/evaluation/chunking_benchmark.py` |
| RRF 敏感性（rrf_k × candidate_k） | IMPLEMENTED_AND_VERIFIED | 容器内 6 组单元格完成（`reports/unified/fusion_sensitivity.json`，PASS） | `src/evaluation/fusion_benchmark.py` |
| 统一报告 + 图 | IMPLEMENTED_NOT_VERIFIED | **未执行**：没有 `final_report.md` 或 `reports/figures/` | `src/evaluation/report.py`、`scripts/report.py` |
| CLI 全命令 | PARTIAL | 仅 `db-check` 与 `ingest --strategy` 实际执行；其余子命令只经 `--help` / 单测 | `src/cli.py` |
| FastAPI | IMPLEMENTED_NOT_VERIFIED（历史快照，见 §10.7） | 当时纯单元测试通过；**未启动服务、未发请求** | `src/api/app.py`、`src/api/schema.py` |
| Docker / compose / Makefile | PARTIAL（历史快照，见 §10.7） | 当时镜像已重建；`warm-cache` 与离线加载通过；`api` 服务未启动；宿主机无 `make` | `Dockerfile`、`docker-compose.yml`、`Makefile` |
| 容器内全链路 | PARTIAL | `db-check`、三策略 ingest、统一检索 / 索引 / 分块 / RRF 敏感性已运行；RAG 端到端、API、pytest 全量未运行 | `docker-compose.yml`、`scripts/warm_cache.py` |
| 真实 LLM smoke | IMPLEMENTED_NOT_VERIFIED（`--provider real` 入口已存在） | **未调用** | `scripts/benchmark_rag.py`、`src/rag/provider.py` |
| 新实验数字 / artifact | IMPLEMENTED_AND_VERIFIED（大部分） | `reports/unified/benchmark.json`、`fusion_sensitivity.json`、`index_benchmark.json`、`chunking_benchmark.json`、`chunking_benchmark_rerank.json`；无 `rerank.json` / `rag.json` / `final_report.md` / 图 | `reports/unified/` |

### 10.3 执行方式与结果

安全检查机制在部分时段不可用，自动化工具无法执行 shell；所有命令改由用户在会话内
以 `!` 前缀手动运行，输出直接落回会话。以下结果均带退出码或 artifact：

**宿主机**

- `python -m ruff check src scripts tests` → `All checks passed!`。
- `python -m mypy src` → `Success: no issues found in 50 source files`。
- 纯单元 pytest（23 个模块）→ 506 用例、504 passed、2 skipped、`EXIT=0`；
  两个跳过均为宿主机缺少 matplotlib（`test_report.py` 的 fig 用例）。

**容器内**

- `db-check` → 9/9 PASS；150 documents / 150 `document` chunks。
- `ingest --strategy fixed_window` / `structured` → 267 / 161 chunks，各 5/5 校验；
  三种策略合计 578 chunks。
- `warm_cache.py` → BGE-large 与 bge-reranker-base 均 PASS；随后
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 离线检查同样 PASS。
- 统一检索 benchmark → `reports/unified/benchmark.json` PASS，6 臂（含
  `rrf_rerank`）各 30 条 Query。
- 索引实验 → `reports/unified/index_benchmark.json` **PARTIAL**（14 组中 1 组失败，
  `synthetic_scale=1`）。
- 分块对比 → `reports/unified/chunking_benchmark.json` PASS；
  补跑 `rrf_rerank` 后写入 `reports/unified/chunking_benchmark_rerank.json` PASS，
  12 个 (策略, 臂) 单元格全部 30/30 完成。
- RRF 敏感性 → `reports/unified/fusion_sensitivity.json` PASS（6 组）。
- 容器环境字段：Linux / WSL2、Python 3.12.13、torch 2.14.1+cpu、`chunks=578`；
  数据哈希与既有 artifact 一致。

**回归（2026-10-20，多策略库）**

- 宿主 4 个 DB 用例通过：`test_dense_benchmark.py` 的 smoke、`test_database.py`
  的 HNSW plan、`test_dense_retrieval.py` 的两个 forced-plan 用例。
- 容器内 `tests/test_dense_retrieval.py`：46 passed、1 skipped（跳过的是固定
  150 chunks 的 natural-plan 断言）；覆盖真实 BGE embedding 与数据库路径。
- 容器内全量 pytest：**664 passed、1 skipped、2 warnings、28.27s、`EXIT=0`**，
  日志 `reports/container/pytest-full.log`；唯一 skip 同上。全量包含
  `test_all_figures_render_from_complete_artifacts`（测试构造的 artifacts，
  不代表正式 `final_report.md` 已生成）。
- 这些运行暴露并修复了 forced-HNSW planner 失效，同时确认了 ANN 后过滤行为
  （见 10.4.1）。

**分块 + rerank 结论（来自新 artifact，非历史报告）**：没有单一策略在所有指标上
胜出；rerank 对三种策略的 MRR / nDCG 都有提升（如 `structured` MRR 0.8778 →
0.9500，`document` 0.9500 → 0.9833），代价是 P50 从约 82–97 ms 升到 512–793 ms。
所有策略共用一个 HNSW bootstrap 索引，索引大小无法按策略归因。

### 10.4 本轮仍未完成（避免误读）

- 没有生成 `reports/final_report.md`、`reports/figures/*.png`，也没有
  `reports/unified/rerank.json` / `rag.json`。
- 没有启动 FastAPI 服务或发送请求；`tests/test_api.py` 只用 fake pipeline。
- 没有调用真实 LLM；mock / historical 结果没有被写成真实调用。
- 索引实验仍是 PARTIAL：只有 150 行（`synthetic_scale=1`），1 组失败未逐项定位。
- 没有运行 `make`（宿主机没有 make），也没有执行 `docker compose down -v` /
  `make reset` 等破坏性操作。
- 没有修改 `data/`，没有覆盖五份历史 `reports/*_benchmark.*`。
- 没有 commit / push。
- `tests/test_dense_benchmark.py` 的 DB 集成用例已适配 578 chunks：计数与向量按
  `chunk_strategy='document'` 收窄；`ef_search=5` 时断言返回数为 1..K，而不是恰好 K。

### 10.4.1 多策略库暴露的两个问题（2026-10-20 发现并处理）

1. **forced-HNSW 臂失效**：`enable_seqscan=off` 不再足以让 planner 使用
   `chunks_embedding_hnsw_idx`；它会选择 `chunks_strategy_idx` + Sort。已在
   `dense_benchmark._configure_planner` 的 HNSW 分支增加
   `enable_bitmapscan=off` 与 `enable_sort=off`，并在 `definitions.hnsw_arm`
   记录该口径。plan 仍必须出现 HNSW 索引名，否则配置继续标 invalid。
   同一设置已同步到 `tests/test_dense_retrieval.py` 与 `tests/test_database.py`
   的 HNSW 诊断用例。
2. **ANN 后过滤**：pgvector 先做 ANN 扫描，再应用 `chunk_strategy` 过滤；多策略
   共存时单策略可能返回少于 K 条（实测 `ef_search=5` → 3 条 `document`）。这不是
   planner 问题，也不能靠 GUC 消除；已写入分块实验的 limitations、README、
   `docs/architecture.md` 与 `reports/results.md`。`hnsw.iterative_scan` 是潜在
   改进方向，但本轮未启用、未验证。

### 10.5 剩余步骤

1. **回归**：重跑 `ruff check src scripts tests` 与 `mypy src`（覆盖本轮对
   `dense_benchmark.py`、`chunking_benchmark.py`、`test_dense_benchmark.py`、
   `test_dense_retrieval.py`、`test_database.py` 的修改），并在数据库上执行
   `tests/test_dense_benchmark.py` 的集成用例与其余 HNSW 诊断用例。
2. ~~容器内全量 pytest~~ 已完成：664 passed、1 skipped、`EXIT=0`
   （`reports/container/pytest-full.log`）。宿主全量仍未跑。
3. **报告**：运行 `scripts/report.py` 生成 `final_report.md` 与图，检查每个数字
   带 `[artifact: path#field sha256=...]`。
4. **索引收尾**：定位 14 组中的 1 组失败；如需规模结论，显式使用
   `--synthetic-scale > 1` 并在 artifact 中保留 `SYNTHETIC` 标记。
5. **API 冒烟**：`docker compose --profile api up -d api` → `/healthz`、`/readyz`、
   `POST /answer`。
6. **真实 LLM smoke**：2–3 条，`--no-write`；只标 `smoke=true`，不评价生成质量。
7. **交付**：验证 `uv.lock` 后再决定是否纳入；提交前检查无密钥。

### 10.6 LLM-as-a-Judge（2026-10-20）

按已批准的计划写入 Context Relevance / Faithfulness 评估层：`src/schema/judge.py`、
`src/evaluation/judge_prompts.py`、`judge.py`、`judge_evaluation.py`、
`scripts/evaluate_judge.py`、`make judge` 与 `python -m src.cli evaluate-judge`；
报告层把 judge 作为 optional artifact，新增 `judge_distribution.png`。

**执行状态：PASS（真实 30/30 全量 Judge 已完成；最终报告/分布图已按该结果重生成，`Report status: PASS`）。**
实际结果如下：

| 检查 | 实际结果 |
|---|---|
| `python -m ruff check src scripts tests`（宿主） | 首轮 5 个问题（D301 / I001 / F541 ×2 / E501），修复后 `All checks passed!` |
| `python -m mypy src`（宿主） | `Success: no issues found in 54 source files` |
| 定向 pytest（宿主） | 36 tests / 0 failed / 0 errors / 3 skipped（缺 `matplotlib`）；JUnit `reports/container/pytest-judge.xml` |
| 容器 mock smoke | `PARTIAL`：3/30、`not_run_queries=27`、两指标各 3/3、mean/median 4.0（fixture）、JSONL 3 行；产物 `reports/judge-smoke/` |
| 容器真实 smoke（历史） | `PARTIAL`（真实执行）：3/30、27 NOT RUN；`judge_is_real=true`、`real_judge_calls=true`、`same_model=true`；CR mean 4.3333（4×2、5×1）、Faithfulness mean 5.0000（5×3）；JSONL 3 行；记录归档在 `reports/judge-real-3q/` |
| 容器真实全量 | **PASS**（真实执行）：30/30、0 NOT RUN、0 Judge 失败、0 RAG 失败；`queries_with_judge_failure=0`；CR mean 3.9667 / median 4（2×1、3×4、4×20、5×5；低分 1 条 Q-029）、Faithfulness mean 5.0000 / median 5（5×30）；JSONL 30 行；`Integrity=True`；产物 `reports/judge/` |
| 报告集成 | **PASS**：已运行 `docker compose run --rm app python scripts/report.py --overwrite`，输出 `Wrote /app/reports/final_report.md`、5 张图（含 `judge_distribution.png`）与 `Report status: PASS` |
| 容器全量 pytest | `691 passed, 1 skipped, 3 warnings`（30.42s）；JUnit `reports/container/pytest-full-judge.xml` |

真实 smoke 前有两次缺 Key 的 `BLOCKED` 尝试（不写 artifact），随后通过
`docker compose run -e APEX_LLM_API_KEY ...` 注入 Key 完成真实调用；全量运行沿用
同一方式，30 条全部成功。Key 未写入代码、日志、artifact 或报告。mock artifact
`reports/judge-smoke/`、历史真实 smoke（`reports/judge-real-3q/`）与当前全量
`reports/judge/` 分开保存，没有混用，也没有把 mock 分数并入最终报告。

本轮实测的限制：配置模型字段为 `deepseek-chat`，但响应 `model` 均为
`deepseek-flash`（同一 endpoint；报告中的模型名应理解为配置名）；Q-001 的 Context
Relevance 为 4 分却把 10 条 context 中的 9 条标为 irrelevant，说明 rubric 执行可能
不一致；全量运行中 Q-029 的 CR 为 2（唯一低分），理由是唯一提到 Global Entry 的
证据属于另一张卡。分数是同模型自评，不是客观真值。不以宿主结果替代容器结果，
也不把“代码存在”写成“已验证”。

### 10.7 页面验收（只读，2026-10-20）

按已批准的计划新增只读验收看板：`src/api/acceptance.py`、`src/api/app.py` 的路由
挂载、`web/acceptance/{index.html,styles.css,app.js}`，以及
`tests/test_api_acceptance.py`、`tests/test_acceptance_page.py`。

| 项目 | 实现 / 验证状态 |
|---|---|
| 接口 | 仅 GET：`/acceptance/`、`/acceptance/app.js`、`/acceptance/styles.css`、`/acceptance/api/summary`、`/acceptance/api/judge/queries`；无写路由 |
| 数据来源 | 固定五份 artifact；总状态复用 `report_status()`；不重算、不补零、不写 `reports/`、`data/` 或数据库 |
| 安全 | 白名单路径；响应剔除 `raw_response`、usage、answer/context/prompt 哈希、prompt 文本与 Provider endpoint（artifact SHA256 保留用于核对）；CSP + `nosniff` + `no-store`；前端只用 `textContent` |
| 静态检查 / 单测 | **PASS（宿主，2026-10-20）**：`ruff check src scripts tests` → `All checks passed!`；`mypy src` → `Success: no issues found in 55 source files`；定向 pytest（两组验收测试 + `test_api.py` + `test_report.py`）→ 0 failed、3 skipped |
| 容器内 API 请求 | **PASS（2026-10-20，全量 Judge 后复核）**：`GET /healthz` → `{"status":"ok","service":"apex-rag-api"}`；`GET /acceptance/api/summary` → `status=PASS`、5/5 artifact `present=true` 且全部 `PASS`、6/6 检查 `pass`（`all_loaded_pass=5/5`、`judge_coverage=30/30`、`judge_real=real`）；`judge` 的 SHA256 为 `9feefa14…`。此前 3/30 smoke 时该接口为 `PARTIAL`，已由全量结果取代 |
| 浏览器人工检查 / 静态页请求 | **NOT RUN**：尚未在浏览器打开页面，也未请求 `/acceptance/`、`app.js`、`styles.css`；不以 API 验证代替渲染验证 |
| 验收结论 | 页面只读，自动检查是机械规则，**不是人工验收结论**；本阶段不写验收结论 |

待执行的验证命令（逐项记录，失败或未执行不得写成 PASS）：

```bash
python -m ruff check src scripts tests
python -m mypy src
python -m pytest tests/test_api_acceptance.py tests/test_acceptance_page.py \
  tests/test_api.py tests/test_report.py -q -p no:cacheprovider
# 宿主页面检查（需在仓库根、已安装 api extra）
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/acceptance/api/summary
# 容器检查（不执行 down -v / reset）
docker compose --profile api up -d api
curl http://127.0.0.1:8000/acceptance/api/judge/queries
```

未执行或失败的项目保持 `NOT RUN` / `BLOCKED`，不以纯单元测试代替浏览器或容器验证。

### 10.8 检索接口 `POST /api/v1/search`（Bonus C，2026-10-20）

为补齐题目 Bonus C 的示例路径，在现有 FastAPI 服务上新增检索端点；不改检索算法，
不改历史 artifact，也不调用 LLM。

| 项目 | 实现 / 验证状态 |
|---|---|
| 端点 | `POST /api/v1/search`：请求 `{query, top_k?}`；返回 `query`、有效 `top_k`、`hit_count`、`retriever`、排序后的 `results[]` 与 `timings_ms` |
| 结果字段 | `chunk_id`、`doc_id`、`rank`、`score`、`source`、`rerank_score`、`title`、`category`、`card_tier`、`effective_year`、`policy_code`、`content`；不返回 prompt、上下文全文拼接、Provider 原始输出或任何密钥 |
| 校验 | `query` 1–2000 字符，去除首尾空白，全空白 422；`top_k` 1–50，且不得超过 `rag_candidates`（默认 10），否则 422；未知字段 422 |
| 复用 | 默认调用 `get_pipeline().retriever`，与 `/answer` 共用 embedding 与 reranker，每个进程只加载一次；测试可注入 `retriever_factory` |
| 错误映射 | `psycopg.Error` / `RetrievalError`（含重排阶段的 `RerankError`）/ `OSError` → 503；其他异常 → 500；响应只含固定文案 |
| Swagger | FastAPI 自动生成 `/docs` 与 `/openapi.json`；端点带 `search` 标签、请求/响应 schema 与 422/503/500 说明 |
| 代码 | `src/api/app.py`、`src/api/schema.py`；Docker 侧无需新服务，`api` profile 已监听 8000 |
| 契约测试 | `tests/test_api.py` 新增 5 组测试：成功投影、空白/越界校验、候选池上限、错误不泄漏、OpenAPI/`/docs` |
| 静态检查与测试 | **PASS（2026-10-20）**：`ruff check src/api tests/test_api.py` → `All checks passed!`；`mypy src` → `Success: no issues found in 55 source files`；`pytest tests/test_api.py -q` → 23 个用例执行完成（23 个点、无 F/E） |
| 容器请求 | **PASS（2026-10-20）**：`api` 容器 healthy；`/healthz` → 200；`POST /api/v1/search` → 200、`hit_count=5`、`retriever=hybrid`，Top-1 `DOC-0035`（Apex Platinum，FEE-202，0 外币手续费）、Top-2 `DOC-0038`（Apex Business Select，1%）；`/docs` → 200；`/openapi.json` 含 `/api/v1/search`。重启 `api` 后按最终文件复跑，`/healthz` 与检索接口仍均为 200（`retrieval_ms=14964.789`）。首次请求的 `retrieval_ms` 含模型加载，不是稳态延迟 |

待执行命令（逐项记录；失败或未执行不得写成 PASS）：

```bash
python -m ruff check src/api tests/test_api.py
python -m mypy src
python -m pytest tests/test_api.py -q -p no:cacheprovider
docker compose --profile api up -d --build api
curl -s http://127.0.0.1:8000/openapi.json
curl -s -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the foreign transaction fee?", "top_k": 5}'
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/docs
```

首次检索请求会触发 embedding 与 reranker 加载，延迟明显高于后续请求；该延迟不应写成
检索稳态性能。截至 2026-10-20，Ruff / mypy / `test_api.py` 与容器请求均已执行，
Bonus C 在 README 中记为 PASS。
