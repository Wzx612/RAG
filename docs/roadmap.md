# Roadmap 与当前状态

> 阶段 ⑨ 产出（2026-10-19）。本文汇总当前实现、可复核证据、未验证事项与后续路线。
> 它不是验收报告：阶段验收结论以 README 的“实现进度与阶段验收”表和 `reports/*` 为准。
> 本文没有重新运行 Benchmark、pytest、模型推理、完整 RAG 链路或真实 LLM API；
> 所有指标均逐字引用现有报告，未计算新指标。

---

## 1. 项目快照

- 数据集：150 篇文档 / 30 条英文评测 Query；78 个 Policy Code（`data/kb_documents.json`
  只读统计）；43 条 gold 引用 / 38 篇不同文档；单条 Query 的 gold 大小为 1–3。
- 语料分块：当前只使用 `document` 策略，1 篇文档 = 1 个 chunk；`fixed_window` /
  `contextual` 未用于评测。
- 技术栈：PostgreSQL 16 + pgvector HNSW（cosine）、`BAAI/bge-large-en-v1.5`
  （1024 维）、PostgreSQL FTS、RRF（k=60）、`BAAI/bge-reranker-base`
  Cross-Encoder、OpenAI 兼容 Provider。
- 运行环境：Dense / Sparse / Hybrid / Rerank / RAG 的实测报告均在宿主 Python
  3.14.3 / Windows 11 上完成，PostgreSQL 在容器中；容器内完整链路未验证。
- 当前状态：⑧ 离线 RAG 已实现、报告 PARTIAL；⑨ 文档已定稿、待人工验收；
  ⑩ 交付检查完成、待人工确认；Git 尚未初始化，未 commit / push；
  FastAPI 未实现；Windows 原生崩溃根因未确认。
- P0 文档整改（Stage ⑩-B）：已新增 `reports/results.md`、修正 `.dockerignore`
  过期陈述、补充生产治理设计；仅文档变更，未执行 P1/P2，待人工确认。

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
| ⑩ | 项目最终整合与 GitHub 提交准备（FastAPI 加分项未实现） | **交付检查完成；待人工确认**；未 commit / push | `README.md` 的交付检查、本文章节；只读核对报告、依赖、忽略配置与敏感信息；无 Git 仓库 |
| ⑩-B | P0 文档整改（`reports/results.md` / `.dockerignore` 说明 / 生产治理设计） | **文档整改完成；待人工确认**；未执行 P1/P2 | `reports/results.md`、`README.md`、本文 |

> **阶段编号说明**：`docs/architecture.md` 是阶段 ① 的设计稿，正文保留设计期的
> 阶段规划；阶段 ⑩ 已在文首加入校正说明，统一采用“⑨ README / roadmap 定稿、
> ⑩ 项目最终整合与 GitHub 提交准备”，FastAPI 仍为未实现的加分项。
> 状态以 README 和本文为准。

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
- 限制：一个文档一个 chunk；Sparse OR 已 Recall 1.0；英文；无 Precision/nDCG；
  宿主运行；150 chunks。

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
| 未计算 | answer correctness、Faithfulness、Answer Relevance |

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

---

## 4. 状态分类

### 4.1 已实现且有可复核报告

- `document` 分块与 150 条数据导入。
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

证据：`tests/test_rag_*.py`、`tests/test_reranker.py`、`tests/test_retrieval.py`、
`tests/test_fusion.py` 等。**完整 pytest 未重跑**，因此这些只是模块级证据。

### 4.3 部分验证

- **RAG 端到端**：mock Provider + historical RRF 重放；真实 LLM 未运行。
- **Reranker**：宿主 CPU；历史 RRF Top-10 池；未覆盖多候选、长输入、容器。
- **Dense / HNSW**：150 chunks；自然计划为 Seq Scan；无规模、并发或跨机器结论。
- **Docker**：只验证了数据库侧；RAG / 模型缓存的容器路径未验证。

### 4.4 未验证

- 真实 LLM API 调用与真实网络错误行为。
- RAG live 全链路（Dense + Sparse + RRF + Embedding + Reranker + 生成同进程）。
- 多候选 batch 推理 / E-5B。
- 历史 Windows 原生崩溃的根因。
- Docker 全链路（容器内 ingest、benchmark、RAG、模型缓存与离线加载）。
- 完整 pytest。
- 中文与中英混合检索、生成与引用。
- 大规模性能、并发、吞吐、GPU。
- 跨机器复现性。
- FastAPI 服务与最终提交整合。

### 4.5 已知问题与风险

1. **原生崩溃未结案**：E-5A/E-5C 只覆盖各自探针范围，不能外推。
2. **依赖漂移**：`pyproject.toml` 精确 pin 与宿主已安装版本不同；没有 `.venv`
   或锁文件。报告中的 torch 为 2.13.0，声明为 2.14.1（其他包也有差异）。
3. **容器路径未验证**：报告在宿主 Python 运行；Docker 控制面曾在部分阶段不可用。
4. **完整 pytest 未重跑**：宿主 torch/transformers 组合存在已知原生崩溃风险。
5. **中文能力有限**：Sparse 只有两字符子串 fallback，不是分词；没有中文语料验证。
6. **指标口径**：标签非穷尽，Precision/nDCG 不计算；Recall/MRR 不能代表生成质量。
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
3. **真实 LLM 调用**：没有 `APEX_LLM_API_KEY` / `DEEPSEEK_API_KEY`；正式 RAG 报告
   使用 mock Provider。
4. **完整 RAG 稳定性**：live 链路在同进程加载 Embedding + Reranker 时曾原生崩溃；
   historical 重放是明确替代路径，不是完整 live 链路验证。
5. **完整 pytest**：只运行模块级测试；完整套件未重跑。
6. **Docker 全链路**：容器内 ingest、benchmark、RAG、模型缓存与离线加载未验证。
7. **中文 / 混合语言**：检索、生成、引用均未评测；Sparse 中文探针 0 命中。
8. **大规模性能**：只有 150 chunks；没有 10 万 / 100 万 / 1000 万级数据。
9. **并发**：所有延迟为单进程顺序执行；无并发、吞吐、连接池压力测试。
10. **跨机器复现**：没有第二台机器或独立环境的复现记录。
11. **FastAPI + Docker 加分项**：未实现、未排期。
12. **依赖精确 pin 的重新安装核验**：未在隔离环境按 `pyproject.toml` 重装并重跑。
13. **E-2/E-4 完整目录完整性**：缺少完整 `SHA256SUMS.txt`，INCONCLUSIVE。
14. **生成质量指标**：answer correctness、Faithfulness、Answer Relevance 未计算。

---

## 6. 后续路线（未承诺）

### 6.1 近期

1. **阶段 ⑨ 人工验收**：审阅 README 与本文；只改文档，不执行测试。
2. **决定是否继续崩溃调查**：若继续，先准备并审核独立多候选探针（E-5B），
   再单独申请执行授权；执行前需满足内存门槛、无重型 Python/Torch 进程、
   本地模型 revision 一致、全新证据目录、WER/minidump 监控与明确超时。
3. **真实 LLM 评测**：在提供合法 API Key、明确预算与网络条件后，先用
   `--limit 2 --no-write` 冒烟，再决定是否跑 30 条；必须区分真实结果与 mock。
4. **Docker 全链路**：在容器内重跑 ingest、五个 benchmark 与 RAG；验证模型缓存、
   离线加载、依赖 pin 与宿主差异。
5. **完整 pytest**：在隔离环境或容器内运行；若复现原生崩溃，保留证据后再分析。

### 6.2 中期

- 中文语料接入：`zhparser` / `pg_jieba`、匹配的 `tsvector`、重建索引，再谈中文指标。
- 更大规模评测：10 万 / 100 万级合成或真实语料，测 HNSW `ef_search`、内存、
  建索引时间、召回与延迟；当前 150 chunks 不能作为规模结论。
- 并发与吞吐：连接池、批量查询、reranker 并发、GPU/CPU 对比。
- 生成质量：rubric、人工评分或已执行的 LLM-as-a-Judge；与检索指标分开报告。
- FastAPI + Docker 加分项：在主线稳定后实现；当前只有 `[api]` 依赖预留与
  compose 注释，没有 `src/api`。

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
   `make report` / `cli.report` 仍未实现。实际评测报告是
   `reports/{dense,sparse,hybrid,rerank,rag}_benchmark.*`；`reports/figures/`
   存在但为空，未生成本阶段图表。另外，`.dockerignore` 仍将
   `reports/results.md` 视为生成物并排除出 Docker 构建上下文；这不影响 Git
   交付，但容器镜像不会包含该汇总。本阶段未修改 `.dockerignore`。
3. **Makefile 计划目标**：`make index` / `tune` / `report` / `pipeline` 目标存在，
   但调用的是未实现的 CLI 子命令；README 已标注“尚未实现”，不是可执行能力。
4. **API 预留**：`pyproject.toml` 有 `[api]` 可选依赖，`docker-compose.yml` 有注释的
   `api` 服务；`src/api` 不存在，FastAPI 未实现。compose 与 pyproject 中的
   “Phase 8 (bonus)”是历史阶段编号，**待核实**是否更新；不影响当前交付。
5. **Hybrid gold 口径**：经只读统计，`eval_queries.json` 为 43 条 gold 引用 /
   38 篇不同文档；报告“all 43 gold documents”措辞不准确，但 Recall 1.000
   的结论不变。历史报告未修改。
6. **依赖漂移**：`pyproject.toml` 精确 pin 与宿主实际版本不同；报告已记录，
   本阶段未重装、未升级、未降级。
7. **内存采样时间差**：Stage A 审核时点 14:10 为 3.842 / 3.848 GiB（FAIL）；
   审核后会话内只读采样 14:32:58 / 14:33:05 为 4,926 / 4,925 MB（约 4.81 GiB，
   高于 4.5 GiB）。后者未写入 Stage A 报告，不是对前者的修正；也没有据此执行 E-5B。
8. **E-2/E-4 完整性**：只有报告列出的关键文件哈希，没有完整清单；完整目录完整性
   为 INCONCLUSIVE，不得写成 PASS。
9. **原始 10 步路线**：除已记录的 ⑧/⑨/⑩ 外，其余编号与范围未在仓库中找到完整
   定义，**待核实**。
10. **Git 仓库**：项目根目录没有 `.git`；无法核对历史提交、已跟踪/未跟踪文件。
    本阶段未执行 `git init`、`git add`、`git commit` 或 `git push`。
11. **本地凭据**：`.env` 中存在非空的 LLM API Key 配置（不在本文或终端输出值）。
    `.env` 已被 `.gitignore` 与 `.dockerignore` 排除；公开提交前仍建议轮换并向
    人工确认没有任何副本、压缩包或日志携带该值。
12. **Docker 构建上下文**：`.dockerignore` 已在 Stage ⑩-A 经授权加入 `.cache/`
    （第 24 行），因此当前 `docker build` 不会把本地约 1.1 GB 的
    `.cache/huggingface` 送入构建上下文。该修正只解决构建上下文排除问题，
    **不代表 Docker 全链路已验证**；Docker Compose 的 `./:/app` bind mount、
    `hf_cache` named volume、容器内模型缓存初始化与离线加载仍需独立验证。
    本阶段未修改 `.dockerignore`。
13. **未声明的依赖**：`src/evaluation/rerank_benchmark.py` 与
    `scripts/benchmark_rag.py` 导入 `psutil`，但 `pyproject.toml` 没有声明它。
    容器内是否能由传递依赖提供未验证；本阶段未安装或修改依赖。
14. **许可证文件**：`pyproject.toml` 声明 MIT，但仓库没有 `LICENSE` 文件；
    公开前需人工补齐版权主体或许可证文本。
15. **报告中的本地路径**：`reports/rerank_benchmark.json`、`sparse_benchmark.json`
    含生成机器的绝对路径与模型缓存路径；属于历史证据，未改写。公开提交前应人工
    检查所有 Markdown/JSON 是否含本地绝对路径、用户名或缓存路径。公开版本可以使用
    清洗副本，但必须保留原始报告不被覆盖、不改变任何结论，并在副本中明确标注
    “公开清洗副本”；是否清洗及具体范围需要人工决定。
16. **Makefile 清理目标**：`clean` 引用 `reports/results.md` 与
    `reports/figures/*.png`。`reports/results.md` 现在已存在（Stage ⑩-B 手工汇总），
    `reports/figures/` 为空；`make clean` 会删除该汇总或图表。本阶段未运行
    `make clean`，也未修改 Makefile；若执行清理，需要先确认是否会删除交付文档。

---

## 9. 证据位置与更新规则

- 主报告：`reports/dense_benchmark.md` / `.json`、`sparse_benchmark.md` / `.json`、
  `hybrid_benchmark.md` / `.json`、`rerank_benchmark.md` / `.json`、
  `rag_benchmark.md` / `.json`。
- 跨报告汇总：`reports/results.md`（只读汇总，非新评测；不替代原始报告）。
- 崩溃调查证据：`D:\RAG_crash_evidence\E1_2026-10-19_1055`、
  `E2_2026-10-19_1127`、`E4_2026-10-19_1134`、`E4R_2026-10-19_1141`、
  `E5_2026-10-19_1205\E5A`、`E5_2026-10-19_1207\E5C`、
  `E5_REVIEW_2026-10-19_140237`。
- 历史证据只读；不删除、不覆盖、不重新生成。需要新结论时，新建证据目录并记录
  命令、环境、哈希、内存与退出码。
- 报告重新生成会改变耗时等字段与哈希；README 的哈希块只在内容不变时有意义。
- 如果代码、配置或数据发生变化，先更新对应报告与 README 阶段表，再更新本文；
  本文不替代验收记录。
