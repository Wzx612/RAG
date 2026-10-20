
面向呼叫中心坐席 Co-Pilot 的检索增强生成引擎。在 150 篇信用卡政策文档、
30 条标注 Query 上，量化对比**朴素稠密检索**与**混合检索 + 重排序**流水线，
并把最终证据接入可替换的 LLM Provider 生成带引用的答案。

> 当前进度：核心任务 ①–⑩ 已完成；GitHub 仓库已同步至 `main`
> （<https://github.com/Wzx612/RAG>）。真实 30/30 Judge 为 PASS，
> Bonus C 的 `POST /api/v1/search` 与 Swagger 已在容器内实际请求通过。
> 仍有明确记录的未验证项（浏览器渲染、真实 LLM 版正式 `rag_benchmark`、
> 人工 Judge 校准与 inter-judge 一致性等），见下文；这些项已被标记为
> `NOT RUN`，不能当作已通过，但不影响本次提交。
> ⑦-B 已验收，判定 PASS WITH MINOR ISSUES。Dense、Sparse OR/AND、
> 候选并集、RRF 与 Cross-Encoder Reranker 均已在真实 PostgreSQL 上跑完 30 条标注
> Query。阶段 ⑧ 新增上下文组装、提示词、OpenAI 兼容 Provider、引用校验、结构化答案
> 与 30 条 Query 的问答评测；结果见
> [`reports/rag_benchmark.md`](reports/rag_benchmark.md)。Reranker 实测见
> [`reports/rerank_benchmark.md`](reports/rerank_benchmark.md)；⑥-B 混合检索结果见
> [`reports/hybrid_benchmark.md`](reports/hybrid_benchmark.md)。
>
> **真实 LLM 全量 Judge 已完成**：2026-10-20 在容器内对 30/30 条真实 RAG 答案执行
> Context Relevance 与 Faithfulness 评分，`status=PASS`、0 失败、0 NOT RUN
> （`reports/judge/`）。正式问答报告生成时仍使用显式标注的 mock Provider；该 Judge
> 结果不改变历史 RAG benchmark 数字，mock 答案不是模型输出。原 3/30 真实 smoke 已
> 归档到 `reports/judge-real-3q/`，历史记录未删除。
> 阶段验收状态与依据见 [实现进度与阶段验收](#实现进度与阶段验收)。
>
> **Windows Reranker 原生崩溃调查仍未结案**：E-5A（单候选）与 E-5C（空候选）
> 探针执行 PASS，其原始证据审核也 PASS；但 Stage A 审核时点的可用内存门槛为
> FAIL，且 PASS 只覆盖各自探针范围。历史 `0xC0000005` / `torch_cpu.dll` 根因
> 未确认。E-5B（多候选）未执行，当前暂停。
> 详见 [Windows Reranker 原生崩溃调查状态](#windows-reranker-原生崩溃调查状态)。
>
> 设计与取舍见 [`docs/architecture.md`](docs/architecture.md)；当前状态、未验证项
> 与后续路线见 [`docs/roadmap.md`](docs/roadmap.md)。
>
> **Stage ⑩-B（P0 文档整改）**：跨阶段结果汇总见
> [`reports/results.md`](reports/results.md)；该汇总只引用现有报告，没有重新运行
> Benchmark、pytest、RAG、模型或真实 LLM。README / roadmap 中关于 `.dockerignore`
> 的过期陈述已修正，roadmap 已补充生产治理设计；仅文档变更，未执行 P1/P2。

---

## 技术评估要求对照（题目 PDF）

> 依据 `AI Engineer 技术评估.pdf`（“企业级向量检索与 RAG 系统”）整理。只引用
> 本仓库已落盘 artifact；实现状态与执行验证分开记录，未执行项保持 `NOT RUN`。

### 评分维度（总分 100 + 加分 20）

| 维度（权重） | 题目要求 | 对应实现与证据 | 当前状态 |
|---|---|---|---|
| 1. 向量存储与导入（25） | 任务 1：Embedding 流水线；文档块、向量与元数据（分类、规则、生效年份等）入库；HNSW / IVFFlat 建索引并调参至 <100 ms | `src/ingestion/`（入库 `category` / `card_tier` / `effective_year` / `policy_code`）、`src/retrieval/dense.py`；`reports/dense_benchmark.md`（PASS，150 chunks）；`reports/unified/index_benchmark_scale10.json`（PASS，合成 1500 行）；`reports/unified/benchmark.json` 的 `dense` 臂 P50 78.90 ms / P95 92.54 ms | PASS（150 条真实数据；索引调参结论来自合成规模，见限制） |
| 2. 混合检索与重排序（25） | 任务 2：Dense + Sparse（`tsvector`/BM25）与 RRF/加权融合；Cross-Encoder / BGE-Reranker 将 Top-N 精排为 Top-K；固定滑窗 vs 语义/层级分块对比 | `src/retrieval/`（`dense.py`、`sparse.py`、`fusion.py`、`reranker.py`）；`reports/unified/benchmark.json`（6 臂 PASS）；`reports/unified/rerank_20to5.json`（PASS，RRF+Rerank MRR 0.9833）；`reports/unified/chunking_benchmark*.json`（PASS，3 策略） | PASS（文档级检索指标；生成质量限制见 Judge / RAG 章节） |
| 3. 定量评估与数据呈现（20） | 任务 3：基线（朴素 dense）vs 改进版（hybrid+rerank）；Recall@K、Precision@K、MRR、nDCG@K、P50/P95；表格、图表或 Notebook | `reports/hybrid_benchmark.md`、`reports/rerank_benchmark.md`、`reports/rag_benchmark.md`；`reports/final_report.md`（PASS，含 P@K/nDCG@K/P50/P95）；`reports/figures/` 5 张 PNG；`reports/results.md` | PASS（历史数字与容器新实验分环境记录，不互相替代） |
| 4. 代码质量与软件工程（20） | 模块化 `src/ingestion`、`src/retrieval`、评价模块；类型提示、PEP 8、Docstrings、异常处理；pydantic-settings/.env；Makefile 或脚本一键导入、建索引、评测 | `src/`（含 `src/evaluation/`）、`tests/`；`pyproject.toml`（Ruff line-length 100、mypy `disallow_untyped_defs`）、`src/config.py`（pydantic-settings）；宿主 Ruff / mypy PASS；容器全量 pytest `691 passed, 1 skipped`；`Makefile`、`docker-compose.yml` | PASS（宿主无 `make`，改用等价 `docker compose` 命令执行） |
| 5. 战略架构路线图（10） | 任务 4：从 150 条扩展到 10,000,000 条；实时政策更新与高频查询缓存（Redis）；PII 防护与访问控制；1 页 `docs/roadmap.md` | [`docs/roadmap.md`](docs/roadmap.md) §7（10M 规模、Redis 缓存、PII、访问控制与租户隔离） | 文档已交付（篇幅超过题目建议的 1 页，但覆盖全部要求）；10M 规模本身 **NOT RUN** |

### 加分 / 拓展挑战（最高 +20）

| 加分项 | 题目要求 | 本仓库状态 | 证据 |
|---|---|---|---|
| A. Apache Iceberg 向量检索 | PyIceberg / DuckDB / PySpark PoC | **NOT RUN** | — |
| B. GraphRAG / 知识图谱遍历 | 实体关系图 + 图遍历结合向量检索 | **NOT RUN** | — |
| C. 容器化与 OpenAPI 接口 | 单个 `docker-compose.yml`、FastAPI `/api/v1/search`、Swagger | **PASS** | 单个 `docker-compose.yml`（`db` + `api` profile）；`POST /api/v1/search` 与 Swagger `/docs` / `/openapi.json` 已实现并**在容器内实际请求成功**（200，5 条结果，`/docs` 200，OpenAPI 含该路径）；Ruff `All checks passed!`、mypy `Success: no issues found in 55 source files`、`test_api.py` 23 个用例无失败 |
| D. LLM-as-a-Judge 自动化评估 | 测量 Context Relevance 与 Faithfulness | **PASS** | `reports/judge/judge_summary.json`（30/30，0 失败，CR mean 3.9667 / Faithfulness mean 5.0000）；同模型自评，人工校准 **NOT RUN** |

### 提交清单

| 提交物 | 仓库位置 | 状态 |
|---|---|---|
| README（环境、运行步骤、结果摘要） | `README.md` | 已定稿，待人工验收 |
| 源代码 | `src/`、`tests/`、`scripts/` | 文件齐全；宿主/容器测试与静态检查见“测试与静态检查”；未 commit / push |
| 标准化数据集与索引配置 | `data/kb_documents.json`、`data/eval_queries.json`（原始数据未修改）；索引配置与实验记录在 `reports/unified/index_benchmark*.json`、`src/evaluation/index_benchmark.py` | 数据未覆盖；索引配置已落盘 |
| 可视化对比报告 | [`reports/results.md`](reports/results.md)、[`reports/final_report.md`](reports/final_report.md)、`reports/figures/*.png` | results.md 为只读汇总；final_report 为渲染报告（PASS），内嵌 5 张 PNG |
| 战略架构路线图 | [`docs/roadmap.md`](docs/roadmap.md) | 已交付（规模化、缓存、PII、访问控制） |
| 一键构建与运行 | `Makefile`、`docker-compose.yml`、`Dockerfile` | 已配置；`make` 目标因宿主无 `make` 未直接执行，等价 `docker compose` 命令已运行 |

---

## 本轮迭代交付状态（P0+P1+P2，2026-10-20）

本轮按批准的 P0+P1+P2 计划补齐了统一评测、索引参数实验、分块对比、CLI、
报告层、FastAPI 与容器编排。**执行验证分三层记录，互不替代**：宿主机静态检查
（Ruff / mypy）与纯单元 pytest 已真实通过；容器内 `db-check`、三策略 ingest、
模型离线加载、统一检索、索引、分块、RRF 敏感性、Top-20 → Top-5 精排与最终报告/图
均已真实运行并留下 artifact；另已完成真实 30 条 RAG 答案 + 全量 Judge（PASS），
并在容器内验证了 API 服务与只读验收接口。
下表的“实现状态”
与“执行验证”必须分开阅读。

| 能力 | 实现状态 | 执行验证 | 证据（代码 / artifact） |
|---|---|---|---|
| 统一 arm 注册表（dense / sparse_or / sparse_and / union / rrf / rrf_rerank） | IMPLEMENTED_AND_VERIFIED | PASS（容器） | `src/retrieval/registry.py`；`reports/unified/benchmark.json`（6 arms、30/30） |
| 统一评测与 artifact schema | IMPLEMENTED_AND_VERIFIED | PASS（容器） | `src/evaluation/benchmark.py`、`src/evaluation/artifacts.py`；`reports/unified/benchmark.json` |
| HNSW `m` / `ef_construction` 与 IVFFlat `lists` / `probes` 扫描（shadow 索引 + 合成规模） | IMPLEMENTED_AND_VERIFIED | PASS（容器；`synthetic_scale=10`，6 组 HNSW + 5 组 IVFFlat） | `src/evaluation/index_benchmark.py`；`reports/unified/index_benchmark_scale10.json`（旧 `index_benchmark.json` 保留为 PARTIAL 记录） |
| 三种分块策略同 Query 集对比 | IMPLEMENTED_AND_VERIFIED | PASS（容器） | `src/evaluation/chunking_benchmark.py`；`reports/unified/chunking_benchmark.json`、`chunking_benchmark_rerank.json`（含 `rrf_rerank`） |
| RRF `k` / `candidate_k` 敏感性 | IMPLEMENTED_AND_VERIFIED | PASS（容器；6 个网格单元） | `src/evaluation/fusion_benchmark.py`；`reports/unified/fusion_sensitivity.json` |
| Cross-Encoder 精排 **Top-20 → Top-5**（题面参数） | IMPLEMENTED_AND_VERIFIED | PASS（容器；30/30，两端同一候选池） | `reports/unified/rerank_20to5.json` / `.md`；RRF Recall 0.9556、MRR 0.9500；RRF+Rerank Recall 1.0000、MRR 0.9833、Hit@1 29/30 |
| 统一报告与图（5 张 PNG） | IMPLEMENTED_AND_VERIFIED | PASS（容器；按 30/30 Judge 重新生成，`Report status: PASS`） | `src/evaluation/report.py`、`scripts/report.py`；`reports/final_report.md`（Status PASS）与 `reports/figures/{arm_metrics,arm_latency,hnsw_heatmap,chunking_metrics,judge_distribution}.png` |
| CLI：`index build\|list\|drop\|tune`、`benchmark`、`evaluate`、`report`、`pipeline`、`serve` | PARTIAL | PARTIAL（容器：仅 `db-check` 与 `ingest --strategy` 已执行；其余子命令未执行） | `src/cli.py`；**`tests/test_cli.py` 尚不存在** |
| 只读 FastAPI：`/healthz`、`/readyz`、`/answer` | IMPLEMENTED_AND_VERIFIED | PARTIAL（容器：`api` 服务已启动，`/healthz` 与 `/acceptance/api/summary` 已实际请求成功；`/readyz` 与 `/answer` 尚无运行时请求记录） | `src/api/app.py`、`src/api/schema.py`、`tests/test_api.py` |
| 检索接口：`POST /api/v1/search` + Swagger | IMPLEMENTED_AND_VERIFIED | **PASS（容器，2026-10-20）**：`POST /api/v1/search` → 200、`hit_count=5`、Top-1 为 `DOC-0035`（FEE-202，Apex Platinum 0 外币手续费）；`/docs` → 200；`/openapi.json` 含 `/api/v1/search`；Ruff PASS、mypy PASS（55 files）、`tests/test_api.py` 23 个用例无失败 | `src/api/app.py`、`src/api/schema.py`、`tests/test_api.py`；返回排序 chunk，不调用 LLM；复用 `build_pipeline().retriever`，每个进程只加载一组模型 |
| 页面验收：`/acceptance/*` + 静态看板 | PARTIAL | PARTIAL（宿主静态检查/定向测试 PASS；容器 `/healthz` 与 `/acceptance/api/summary` PASS，接口返回 `status=PASS`、6/6 检查通过；浏览器渲染仍为 NOT RUN） | `src/api/acceptance.py`、`web/acceptance/`、`tests/test_api_acceptance.py`、`tests/test_acceptance_page.py`；只读，不写验收结论 |
| Dockerfile / compose `api` + `warm-cache` / Makefile 目标 | IMPLEMENTED_AND_VERIFIED | PARTIAL（容器：镜像重建、`warm_cache.py`、离线 `--check-only`、`app` 运行通过；`api` 服务已启动并通过 `/healthz`；`make` 目标因宿主无 `make` 未直接执行，改用等价 `docker compose` 命令） | `Dockerfile`、`docker-compose.yml`、`Makefile`、`scripts/warm_cache.py` |
| 容器内全链路（ingest、benchmark、离线模型、报告） | PARTIAL | PARTIAL（已执行：三策略 ingest、统一检索、索引、分块、RRF 敏感性、Top-20 → Top-5 精排、模型离线加载、最终报告/图、真实 30/30 RAG + Judge、API 服务与验收接口；未执行：浏览器渲染、真实 LLM 版正式 `rag_benchmark`） | `reports/unified/*.json`、`reports/container/*.log`、`reports/final_report.md`、`reports/figures/`、`reports/judge/` |
| 真实 LLM 全量 Judge（30 条） | IMPLEMENTED_AND_VERIFIED | PASS（容器；真实 RAG + 真实 Judge，30/30，0 失败） | `reports/judge/judge_summary.json`、`judge_results.jsonl`（30 行）；CR mean 3.9667、Faithfulness mean 5.0000；`judge_is_real=true`、`same_model=true`；旧 3/30 记录在 `reports/judge-real-3q/` |
| 新的实验数字（统一 benchmark / 索引 / 分块 / 敏感性 / 精排） | IMPLEMENTED_AND_VERIFIED | PASS（容器） | `reports/unified/{benchmark,index_benchmark_scale10,chunking_benchmark,chunking_benchmark_rerank,fusion_sensitivity,rerank_20to5}.{json,md}`；`reports/final_report.md` |

**本轮已在容器内产生新的实测 artifact，且已生成最终报告与图，未覆盖历史报告。** 五份
2026-10-17 的 `reports/*_benchmark.*` 保持只读；新结果只写入 `reports/unified/`。
索引实验在 `synthetic_scale=10` 下 PASS，但它只反映索引行为，不进入语义质量结论
（旧的 `index_benchmark.json` 因一组非法参数保留为 PARTIAL 记录）。
`reports/final_report.md` 在 30/30 全量 Judge 完成后已重新生成，状态为 **PASS**
（`Report status: PASS`），引用 `benchmark` / `index_scale10` / `chunking_rerank`
三份 PASS artifact，并额外包含 `rerank_20to5.json` 的「候选池 20 → 最终 Top-5」
章节与 30/30 Judge 章节；`reports/figures/` 下 5 张 PNG 均由脚本重新生成。
任何“已实现”只表示代码与测试文件存在；执行验证必须对照上表的证据路径与状态词。
执行顺序与门禁见 [`docs/roadmap.md`](docs/roadmap.md) 的“本轮迭代（P0+P1+P2）”。

---

## 前置条件

只需要 **Docker**（含 Docker Compose v2）。不需要在本机安装 Python、PostgreSQL
或任何数据库扩展——除 `make clean` 外，项目目标都在容器内执行，宿主 Python
版本无关紧要。`make clean` 只删除宿主 Python 缓存，不启动数据库。

```bash
docker --version           # 20.10+
docker compose version     # v2
```

> **Windows 用户注意**：Git for Windows **不**附带 `make`。请用
> `choco install make` 或 `scoop install make` 安装，或者直接运行 `make` 目标里
> 的那条命令——大多数目标只是 `docker compose run --rm app <命令>` 的简写
> （`up` / `down` / `reset` / `psql` / `logs` / `api` / `warm-cache` /
> `verify-container` / `clean` 例外），例如 `make db-check` 等价于
> `docker compose run --rm app python -m src.cli db-check`。

---

## 快速开始

```bash
cp .env.example .env      # 可选；所有值都有默认值
make up                   # 启动 PostgreSQL + pgvector 并等待健康检查
make db-check             # 验证扩展、schema 与 HNSW 索引
```

`make up` 完成后应当看到：

```
$ docker compose ps
NAME            IMAGE                    STATUS
apex-rag-db-1   pgvector/pgvector:pg16   Up (healthy)   0.0.0.0:5432->5432/tcp
```

以及 `make db-check` 的逐项结果：

```
[PASS] PostgreSQL connection: SELECT 1
[PASS] PostgreSQL version: 16.14
[PASS] pgvector extension: vector
[PASS] documents / chunks tables: chunks, documents
[PASS] embedding vector type: vector(1024)
[PASS] search_vector generated column: tsv is GENERATED ALWAYS ... STORED
[PASS] HNSW index (cosine): chunks_embedding_hnsw_idx on vector_cosine_ops
[PASS] GIN index on search_vector: chunks_tsv_gin
[PASS] corpus consistent: documents=0 chunks=0 (empty -- run `make ingest`)

All 9 checks passed.
```

> **端口冲突**：若本机 5432 已被占用，在 `.env` 里设置 `APEX_PG_PORT=<空闲端口>`。
> 容器内部始终用 `db:5432`，只有宿主映射端口会变。

数据存放在名为 `pgdata` 的 Docker volume 中：`make down` 后重新 `make up`，
schema 与数据都还在（初始化脚本只在一个空数据目录上执行一次）。
`make reset` 会**删除**该 volume 并重建 schema。

---

## 数据导入

```bash
make ingest                                  # 等价于 python -m src.cli ingest
python scripts/ingest.py                     # 同一个入口，另一种调用方式
docker compose run --rm app python -m src.cli ingest --strategy document
```

流程是 `JSON → 校验 → Embedding → PostgreSQL`：

| 步骤 | 做什么 |
|---|---|
| **Load** | 读 `data/kb_documents.json`，解析出 `policy_code`（`BT-001` 这样的族标识） |
| **Validate** | 必须恰好 150 条；`doc_id` 唯一；`content` 非空。任一条不合格就整体失败并列出**全部**问题，不修补、不丢弃、不补数据 |
| **Chunk** | 命令行默认 `document`（1 篇文档 = 1 个 chunk）。`fixed_window` 与 `structured` 已实现、已测试并**已在容器内入库**（267 / 161 chunks）；`contextual` 已退役，仅保留为带迁移提示的旧配置名。检索配置带有 `chunk_strategy` 作用域，Dense/Sparse 的 SQL 都按该列过滤，避免多种分块在同一张表中混合。历史 Benchmark 仍只使用 `document`；三策略对比见 `reports/unified/chunking_benchmark.json` 与 `chunking_benchmark_rerank.json`。限制：三策略共用一个 HNSW 索引，且 `chunk_strategy` 过滤在 ANN 扫描之后执行——`ef_search=5` 时全局 Top-5 中仅 3 条属于 `document`，因此某策略可能返回少于 K 条，策略间召回差异也部分来自其在全局索引中的占比 |
| **Embed** | `BAAI/bge-large-en-v1.5` 本地编码，32 条一批。逐批校验维度与有限性；失败按 1s/2s/4s 退避重试 3 次，仍失败则中止 |
| **Write** | `documents` 与 `chunks` 在**同一个事务**内 upsert，要么全成要么全不写 |

导入完成后自检并打印：

```
Verification:
  [PASS] documents in database: count = 150
  [PASS] chunks for strategy 'document': count = 150
  [PASS] chunks with a non-null embedding: 150 / 150
  [PASS] distinct doc_ids among chunks: count = 150
  [PASS] embedding dimension: min = 1024, max = 1024, expected = 1024
```

自己核对：

```bash
make psql
```
```sql
SELECT count(*) FROM documents;                                  -- 150
SELECT count(*) FROM chunks WHERE embedding IS NOT NULL;         -- 150
SELECT count(DISTINCT doc_id) FROM chunks;                       -- 150
SELECT count(*) FROM chunks WHERE embedding <#> embedding = 0;   -- 0（没有零向量）
```

**可以重复执行。** 写入是按主键 upsert 的，第二次运行会把同样 150 行重写一遍
（`0 new, 150 updated`），不会变成 300 行。失败重跑也从已知状态开始——事务保证
不会留下半成品。

> 首次运行会下载约 1.3 GB 的模型权重到 `hf_cache` 卷，之后各次运行复用。
> 数据导入本身不需要任何 API Key：Embedding 模型在容器内本地运行。只有阶段 ⑧
> 的可选真实 LLM Provider 需要环境变量中的 Key（默认 mock，不调用外部 API）。

---

## 常用命令

| 命令 | 作用 |
|---|---|
| `make up` / `make down` | 启停数据库 |
| `make db-check` | 验证 PostgreSQL + pgvector + schema + HNSW |
| `make psql` / `make logs` | 数据库 shell / 日志 |
| `make reset` | 停止并**删除**数据卷（schema 会重建） |
| `make shell` | 进入 app 容器 |
| `make ingest` | 加载、分块、向量化、入库 |
| `make benchmark-dense` | 阶段 ④ 稠密检索 Benchmark |
| `make benchmark-sparse` | 阶段 ⑤ Sparse Retrieval Benchmark |
| `make benchmark-hybrid` | 阶段 ⑥-B 五臂评测（Dense / Sparse OR / Sparse AND / 候选并集 / RRF） |
| `make benchmark-rerank` | 阶段 ⑦-B 五臂评测（历史 RRF Top-10 + Cross-Encoder 重排） |
| `make benchmark-rag` | 阶段 ⑧ 端到端问答评测（默认 mock Provider，不调用真实 API） |
| `make benchmark-retrieval` | 统一六臂评测，写入 `reports/unified/benchmark.json` / `.md` |
| `make benchmark-index` | HNSW / IVFFlat 参数扫描（shadow 索引；默认 150 chunks） |
| `make benchmark-chunking` | `document` / `fixed_window` / `structured` 分块对比（前提：三策略已 ingest） |
| `make benchmark-fusion` | RRF `k` 与候选池敏感性 |
| `make benchmark-all` | 依次运行 retrieval + index + chunking |
| `make ask-rag` | 单问题诊断入口（默认 mock；真实 Provider 需 API Key） |
| `make index-build` / `make index-list` | 建一个带版本号的 shadow 索引 / 列出索引 |
| `make tune` | `index tune`：HNSW / IVFFlat 参数扫描（`ARGS` 可传 `--synthetic-scale N`） |
| `make evaluate` | 读取统一 artifact 并打印指标（不重新计算，不改写文件） |
| `make report` | 汇总 benchmark / index / chunking 三个必需 artifact，并在存在时加入 `rerank` 20→5 artifact，生成 `reports/final_report.md` 与图（`fusion_sensitivity.json` 单独保留） |
| `make pipeline` | 运行 retrieval + index + chunking，然后生成报告 |
| `make warm-cache` | 下载 embedding / reranker 权重到 `hf_cache` 卷 |
| `make api` | 启动只读 FastAPI 服务（`--profile api`，默认 8000） |
| `make verify-container` | 容器内 db-check → benchmark-all → report，日志写入 `reports/container/` |
| `make test` / `make lint` / `make typecheck` | 测试与静态检查（在容器内执行；`make test` 会通过 compose 启动数据库依赖） |
| `make clean` | 只删除宿主 Python 缓存（`__pycache__`、`.mypy_cache`、`.ruff_cache`、`.pytest_cache`），**不删除** `reports/` 下的交付文件，也不启动数据库 |
| `make help` | 列出全部目标 |

---

## GitHub 交付与复现状态

> 本节记录项目交付与 GitHub 同步状态；它不表示项目已通过生产级验收。

- **Git 仓库已同步**：`main` 已推送至 `origin`
  （<https://github.com/Wzx612/RAG>）。2026-10-20 的同步提交为 `b31e053`
  （`Sync complete RAG assessment: unified benchmarks, Judge 30/30, FastAPI search and reports`），
  包含 136 个文件变更；此后如有文档或图表修订，以仓库最新提交为准。
  当前工作区仅保留图表渲染改进的未提交修改，不影响已推送版本；是否把该改进
  纳入提交仍由人工确认。
- **`.env` 是本地文件，可能含真实凭据**，已在 `.gitignore` 与 `.dockerignore` 中
  排除；不要提交、打包或上传。若曾把 `.env` 复制到别处，建议轮换其中的 Key。
- **模型缓存**：`.cache/huggingface` 约 1.1 GB，已被 `.gitignore` 排除；
  `.dockerignore` 已排除 `.cache/`，因此 `docker build` 不会把本地模型缓存送入
  构建上下文。该排除是 Stage ⑩-A 经授权的配置修正，不影响历史报告。
  Docker Compose 的 bind mount、`hf_cache` named volume 与模型缓存初始化已在
  容器内验证（模型可在 `HF_HUB_OFFLINE=1` 下加载）；**浏览器渲染与真实 LLM 版
  正式 `rag_benchmark` 仍未执行，不等于 Docker 全链路全部通过**。
- **不应提交的生成物**：`.mypy_cache`（约 217 MB）、`.pytest_cache`、`.ruff_cache`、
  `__pycache__`、日志与临时文件；均已由 `.gitignore` 覆盖。
- **可以提交的交付物**：`src/`、`tests/`、`scripts/`、`data/`、`reports/`（五个
  Benchmark 的 Markdown 与 JSON，以及汇总 `reports/results.md`）、`docs/`、
  `pyproject.toml`、`LICENSE`、`Dockerfile`、`docker-compose.yml`、`Makefile`、
  `.env.example`、`.gitignore`、`.dockerignore`、`.github/workflows/ci.yml`。
- **CI**：`.github/workflows/ci.yml` 在 Ubuntu + Python 3.12 上运行 ruff、mypy 与
  纯单元测试（跳过真实模型下载；数据库用例在无 PostgreSQL 时由 `conftest.py` skip）。
  该工作流已随 `b31e053` 推送触发；本次未查看 GitHub Actions 的运行结果，
  因此不宣称 CI 已通过。
- **复现性状态**：`pyproject.toml` 对直接依赖使用精确 pin，容器构建按该声明安装；
  `uv.lock` 已提交，但容器仍按 `pyproject.toml` 安装，未验证锁文件与镜像依赖完全一致。
  宿主已安装版本与声明存在漂移（见下一节），
  历史报告记录的是宿主实测版本，不能当作 pin 的安装验证。`psutil` 已声明为必需
  依赖（`src/evaluation/rerank_benchmark.py` 在模块级导入）；MIT `LICENSE` 文件已补齐。
  其版权行目前使用集体署名 “Apex RAG contributors”（2026）；若公开时要求填写具体
  个人或组织，应由人工确认后替换，不能由本轮自行编造。
- **Docker 中的报告**：`.dockerignore` 不再排除 `reports/results.md` 与
  `reports/figures/*.png`，Dockerfile 的 `COPY . .` 会把交付报告带入镜像；
  仅 `reports/runs/`（中间运行日志）被排除。`make clean` 也不再删除这些交付文件。
- **报告路径**：`reports/*_benchmark.json` 与部分 Markdown 报告保留了生成时的本地
  绝对路径（例如项目目录、模型缓存路径与仓库外崩溃证据路径）。这些是历史证据的
  一部分，不应直接改写或覆盖；公开提交前需人工审查。若需要公开版本，可另行生成
  脱敏副本，但必须保留原始报告、原始指标与结论不变。
- **公开提交前检查**：确认 `.env`、`.cache/`、`.venv/`、日志、压缩包与临时文件均
  未进入提交；若 `.env` 或其他本地配置中存在 API Key，应在仓库外轮换，并确认
  Git 历史、日志与公开归档中没有密钥值。不要读取、打印或复制密钥内容。
- **Docker 边界**：五份历史 Benchmark 的数字仍来自宿主 Python，不能当作容器结果；
  容器内已另行执行三策略 ingest、统一检索、索引、分块、RRF 敏感性、20→5 精排、
  模型离线加载、报告/图、全量 pytest、真实 30/30 RAG + Judge 与 API 服务。浏览器
  渲染与真实 LLM 版正式 `rag_benchmark` 仍未执行。

### 依赖、版本与复现性

依赖只在一个地方声明：`pyproject.toml`。Dockerfile 先单独安装 CPU 版 Torch，
再执行 `pip install ".[dev,report,api]"`；`torch` 的版本必须与 `pyproject.toml`
保持一致。本地开发（可选）使用 Python 3.12+ 虚拟环境执行
`pip install -e ".[dev,report,api]"`；editable install 与容器安装读取同一份依赖声明。
`report` extra 只在生成表格与图表时需要，检索与 Benchmark 不依赖它；
`api` extra（FastAPI/uvicorn）只被 `serve` 与 `tests/test_api.py` 使用。
宿主结果不能当作容器验证。

| 包 | 项目声明（pyproject / Dockerfile） | 当前宿主实测 | 历史报告记录 | 锁定版本 |
|---|---|---|---|---|
| Python | `>=3.12`；镜像 `3.12` | 3.14.3 | 3.14.3 | 镜像 tag `python:3.12-slim` |
| torch | `2.14.1`（CPU 索引） | `2.13.0+cpu` | `2.13.0` | 无 |
| transformers | `5.20.0` | `5.10.2` | `5.10.2` | 无 |
| sentence-transformers | `6.1.0` | `5.5.1` | `5.5.1` | 无 |
| numpy | `2.5.4` | `2.4.6` | `2.4.6` | 无 |
| psutil | `7.2.2` | `7.2.2` | `7.2.2` | 无 |
| pytest | `9.1.1` | `9.0.3` | `9.0.3` | 无 |
| ruff / mypy | `0.17.1` / `2.4.0` | `0.17.1` / `2.4.0` | 未记录 | 无 |

**如何阅读这张表：**

- **项目声明版本**是复现目标；容器按它安装。
- **锁定版本尚未验证**：工作区存在未纳入版本控制的 `uv.lock`，但它生成于
  CPU 索引映射加入之前，不能视为当前 `pyproject.toml` 的有效锁。`pyproject.toml`
  现已声明 `[[tool.uv.index]]` + `[tool.uv.sources]`，把 Linux/Windows 的
  `torch` 指向 PyTorch CPU 索引；锁文件必须在有网络的环境下重新生成，并核实
  `torch` 来源为 CPU、依赖树中没有 `nvidia-*` / `cuda-*` 之后才能提交。
  在完成该验证前，不把现有 `uv.lock` 当作复现依据。
- **当前宿主实测版本**是历史 Benchmark 与既有模块级测试所在环境的版本；它与声明
  不同，说明宿主环境不是按当前 pin 重建的。本轮未能重新执行测试，不能把它当作
  本轮测试结果。
- **历史报告记录**来自 `reports/*.json` 的 `dependency_versions`；报告数字只属于
  该历史环境，不能改写成声明版本的测量结果。
- **Torch 不做盲改**：没有证据表明声明的 `2.14.1` 与代码不兼容，也没有完成
  2.14.1 的安装与回归，因此本轮保留声明版本，只澄清漂移；不为了数字一致而
  升级或降级宿主环境。
- **宿主环境不是干净重装**：`%APPDATA%\Python\Python314\site-packages` 同时存在
  多个 dist-info，例如 `pydantic` 2.13.4 / 2.14.1、`pydantic-settings` 2.14.1 /
  2.15.0、`fastapi` 0.136.3 / 0.143.0、`uvicorn` 0.48.0 / 0.54.0，以及
  `typing-extensions`、`annotated-types`、`typing-inspection`、`pydantic-core`
  的双版本。包内 `__version__` 显示当前生效的是 pydantic 2.14.1、
  pydantic-settings 2.15.0、fastapi 0.143.0、uvicorn 0.54.0。该环境不能作为
  锁文件来源，也不能据此声称依赖已按 `pyproject.toml` 重建。
- **约束关系（按已安装包的 METADATA 静态核对）**：`sentence-transformers 5.5.1`
  要求 `transformers<6.0.0,>=4.41.0` 与 `torch>=1.11.0`；宿主
  `transformers 5.10.2`、`torch 2.13.0` 满足。项目声明的
  `transformers 5.20.0` 仍落在 `<6.0.0` 内，声明的 `sentence-transformers 6.1.0`
  的 METADATA 未安装、未下载，无法离线核对。`psutil 7.2.2` 只要求 Python
  `>=3.6`。`matplotlib==3.11.2` 与 `tabulate==0.10.0` 位于可选 `report` extra；
  本轮 `src/evaluation/report.py` 已延迟导入并实际使用 matplotlib 生成图，
  `scripts/report.py` 负责写文件。该导入路径已在容器内执行验证（生成
  `reports/final_report.md` 与 5 张图）。

### 锁文件方案

`pyproject.toml` 现在是唯一声明源，并已配置 Torch 的 CPU 索引映射。
工作区的 `uv.lock` 生成于该映射之前，**尚未重新生成或验证**，不能作为复现依据。
确定的做法：

1. **`uv lock` + `uv.lock`**：在有网络时重新生成，核实 `torch` 来自
   `https://download.pytorch.org/whl/cpu`，且依赖树中没有 `nvidia-*` / `cuda-*`。
   验证通过后才纳入版本控制，并可把 Dockerfile 改为 `uv sync --frozen`
   （当前仍用 pip + CPU wheel 的显式安装）。
2. **不采用 `pip freeze`**：宿主环境已漂移且存在重复 dist-info，冻结结果会与
   `pyproject.toml` 冲突。
3. **不手写锁文件**：没有完整解析结果时手写会产生假锁。

尚未解决的问题：

- 尚未核实 `torch 2.14.1` 是否存在于官方 CPU wheel 索引，以及 cp312 / cp314、
  win_amd64 / linux_x86_64 的 wheel 是否齐全。
- 声明的 `transformers 5.20.0`、`sentence-transformers 6.1.0`、`pytest 9.1.1`
  等版本尚未在离线条件下验证可解析性。
- 锁文件需要同时覆盖 Windows 宿主与 Linux 镜像。
- `matplotlib` 与 `tabulate` 位于 `report` extra，且已被 `src/evaluation/report.py`
  使用；若最终报告不生成图，`--no-figures` 会跳过 matplotlib 导入。

### 测试与静态检查

测试按依赖分为三类，运行前必须区分，不能把“没有连数据库”说成“完整套件通过”：

| 类别 | 覆盖内容 | 外部依赖 | 说明 |
|---|---|---|---|
| 纯单元测试 | 分块（`document` / `fixed_window` / `structured`）、指标、融合、RAG 上下文/Provider/历史重放、Hybrid/Rerank 评测逻辑、检索 `chunk_strategy` 作用域、索引命名与参数校验、统一 benchmark/registry/artifacts、索引/分块/融合敏感性评测逻辑、报告渲染与 artifact 哈希、API 传输层、LLM-as-a-Judge prompt/解析/失败隔离/聚合 | 无 | 不连 PostgreSQL、不加载真实模型；`test_hybrid_benchmark.py` / `test_rerank_benchmark.py` 使用 fake retriever / fake backend；`test_retrieval_scope.py` 使用记录连接，`test_index_management.py` 只验证在连接前就应失败的路径；`test_api.py` 注入 fake pipeline 与 monkeypatched `check_database`（`importorskip` fastapi/httpx）；`test_report.py` 只读 `tmp_path` 中的 artifact；`test_judge.py` / `test_judge_evaluation.py` 只注入 fake / mock provider，不访问网络 |
| 数据库集成测试 | `test_database.py`、`test_dense_retrieval.py`、`test_sparse_retrieval.py`、`test_ingestion.py` 的存储部分、`test_dense_benchmark.py` / `test_sparse_benchmark.py` 的集成用例 | PostgreSQL + pgvector + 已导入语料 | `tests/conftest.py` 的 `conn` fixture 在数据库不可达时 `pytest.skip`；运行这些测试会连接数据库 |
| 真实模型测试 | `test_ingestion.py` 末尾两个测试 | `BAAI/bge-large-en-v1.5` 权重（首次运行约 1.3 GB） | 冷缓存时需要网络/模型下载；不能算作纯单元测试 |

预期命令（在仓库根目录、已安装 `.[dev]` 的环境中；`-p no:cacheprovider` 避免写
`.pytest_cache`）：

```bash
python -m pytest tests/test_hybrid_benchmark.py tests/test_rerank_benchmark.py \
  -q -p no:cacheprovider

python -m pytest tests/test_chunking.py tests/test_chunker.py tests/test_metrics.py \
  tests/test_fusion.py tests/test_retrieval.py tests/test_reranker.py \
  tests/test_rag_context.py tests/test_rag_provider.py tests/test_rag_pipeline.py \
  tests/test_rag_benchmark.py tests/test_rag_historical.py \
  tests/test_retrieval_scope.py tests/test_index_management.py \
  tests/test_dense_benchmark.py tests/test_sparse_benchmark.py tests/test_ingestion.py \
  -q -p no:cacheprovider \
  --deselect tests/test_dense_benchmark.py::test_real_db_smoke_run_verifies_plans_guc_and_metrics \
  --deselect tests/test_sparse_benchmark.py::test_run_sparse_benchmark_on_two_real_queries \
  --deselect tests/test_sparse_benchmark.py::test_focus_queries_are_extracted_by_id \
  --deselect tests/test_ingestion.py::test_the_real_model_emits_1024_finite_dimensions \
  --deselect tests/test_ingestion.py::test_query_and_document_embeddings_differ_by_the_instruction_prefix \
  --deselect tests/test_ingestion.py::test_every_stored_embedding_has_the_configured_dimension \
  --deselect tests/test_ingestion.py::test_no_stored_embedding_is_a_zero_vector \
  --deselect tests/test_ingestion.py::test_the_document_strategy_covers_the_whole_corpus_exactly_once

# 本轮新增的纯单元测试（不连数据库、不加载模型）
python -m pytest tests/test_benchmark.py tests/test_registry.py tests/test_artifacts.py \
  tests/test_index_benchmark.py tests/test_chunking_benchmark.py \
  tests/test_fusion_benchmark.py tests/test_report.py tests/test_api.py \
  tests/test_judge.py tests/test_judge_evaluation.py \
  -q -p no:cacheprovider

python -m ruff check src scripts tests
python -m mypy src
```

`tests/test_api.py` 用 `pytest.importorskip` 跳过缺失的 FastAPI / httpx；Dockerfile
现在安装 `.[dev,report,api]`，容器内会执行而不是跳过。**`tests/test_cli.py` 尚不存在**，
CLI 目前只有静态审阅，没有自动化测试。

> **Bonus C 增补（2026-10-20）**：`tests/test_api.py` 新增了 5 组
> `POST /api/v1/search` 契约测试（成功投影、校验、候选池上限、错误不泄漏、
> OpenAPI/`/docs`）。该文件已单独执行，23 个用例无失败；Ruff `All checks passed!`；
> mypy `Success: no issues found in 55 source files`。见 `docs/roadmap.md` §10.8。

**实际执行状态（P0+P1+P2 实现轮次，2026-10-20，宿主机 + 容器）**

| 检查 | 命令 | 实际输出 | 状态 |
|---|---|---|---|
| Ruff | `python -m ruff check src scripts tests` | `All checks passed!` | PASS |
| mypy | `python -m mypy src` | `Success: no issues found in 50 source files` | PASS |
| pytest（纯单元组合） | 23 个模块，见上方命令 | 506 个用例全部收尾：504 passed（由 506−2 推算，无 F/E）、2 skipped、`EXIT=0` | PASS（仅纯单元） |
| pytest（DB + 真实模型，单文件） | 容器内 `docker compose run --rm app python -m pytest tests/test_dense_retrieval.py -q` | 46 passed、1 skipped、无失败；跳过的是 `test_natural_plan_at_150_chunks_is_a_seq_scan`（库内为 578 chunks，该断言固定于 150） | PASS（容器，单文件） |
| pytest（全量） | 容器内 `python -m pytest -q -o addopts=""`，日志 `reports/container/pytest-full.log` | `664 passed, 1 skipped, 2 warnings in 28.27s`，`EXIT=0`；跳过同上 | PASS（容器，全量） |
| pytest（DB，4 个 planner/benchmark 用例） | 宿主指定 4 个用例，见下方说明 | 4 passed；Ruff / mypy 同时通过 | PASS（宿主，指定用例） |

Ruff 与 mypy 的通过记录来自本机真实执行（三轮修复：17 条 Ruff、18 条 mypy、
随后 1 条 Ruff 与 5 条 mypy 全部清零）。它只证明**静态检查**通过，**不**证明
数据库、容器或真实 LLM 链路可用。mypy 只检查 `src`，不检查 `scripts` 与 `tests`。

纯单元 pytest 已真实执行：506 个用例、2 skipped、退出码 0、无 F/E。两个跳过都来自
`tests/test_report.py:306/317`，原因均为 `No module named 'matplotlib'`——宿主环境
未安装 `[report]` extra，因此**宿主机上的 fig 生成未验证**，需要容器内补跑。
`tests/test_api.py` 未被跳过，说明 fastapi / httpx 在宿主可用，fake pipeline 的
传输层用例全部通过。此外，容器内已单独运行 `tests/test_dense_retrieval.py`：
46 passed、1 skipped，覆盖数据库读取与真实 BGE embedding 路径；被跳过的是固定
150 chunks 的 natural-plan 断言（当前库为 578 chunks）。宿主还单独运行了 4 个
planner / benchmark 用例：
`test_dense_benchmark.py::test_real_db_smoke_run_verifies_plans_guc_and_metrics`、
`test_database.py::test_a_query_plan_uses_the_hnsw_index`、
`test_dense_retrieval.py::test_ef_search_reaches_the_live_guc_and_the_hnsw_scan`、
`test_dense_retrieval.py::test_forced_plan_uses_hnsw_and_returns_top_k`，均通过。
本次还执行了容器内全量套件：**664 passed、1 skipped、2 warnings、28.27s、
`EXIT=0`**，日志保留在 `reports/container/pytest-full.log`；唯一的 skip 仍是固定
150 chunks 的 natural-plan 断言。全量中包含
`test_all_figures_render_from_complete_artifacts`，说明 report 模块在容器内可用
matplotlib 渲染图（用的是测试构造的 artifacts，**不是** `reports/final_report.md`
的正式产物）。该时点仍未运行的是 CLI（无测试文件）与 API 服务冒烟；真实 LLM 仅做
了 3 条 RAG + Judge smoke。此后已完成 30/30 全量 Judge、API/验收接口与报告重生成，
见“LLM-as-a-Judge 自动评估”的执行记录。

**Judge 轮次复跑（2026-10-20，接入 Judge 测试之后）**：宿主 Ruff
`All checks passed!`、mypy `54 source files` 通过、定向 36 tests / 0 failed /
3 skipped（缺 matplotlib）；容器全量 **691 passed, 1 skipped**（30.42s，
JUnit `reports/container/pytest-full-judge.xml`）。相对上一轮 664 passed 增加 27 个
用例，来自 `tests/test_judge.py`、`tests/test_judge_evaluation.py` 与
`tests/test_report.py` 的 Judge 分支。测试通过不等于真实 Judge 质量已验证；真实
3 条 smoke 随后在容器内执行（记录现归档于 `reports/judge-real-3q/`，PARTIAL），
之后又完成了 30/30 全量真实 Judge（`reports/judge/`，PASS），见“LLM-as-a-Judge
自动评估”的执行记录。

上述选择已做静态副作用审查：被保留的测试只使用 fake retriever / fake backend、
`httpx.MockTransport` 与 `tmp_path`，不读取或写入真实报告；模块级 `import psycopg`
本身不会建立连接。`test_reranker.py` 会在模块级 `import torch`（不下载权重、
不加载 Cross-Encoder），在历史上出现过原生崩溃的宿主上仍属于需要留意的风险。
被 `--deselect` 的用例才会连接 PostgreSQL、读取真实模型或加载权重。完整 `pytest`
还会连接数据库并加载真实模型，按本轮约束不应直接运行；需要在获得授权、数据库
可用且模型缓存就绪的独立步骤中执行，并记录真实计数与失败输出。

---

## 项目结构

```
src/
├── config.py        集中配置（pydantic-settings，APEX_ 前缀）
├── cli.py           唯一入口：db-check / ingest / index / benchmark / evaluate / report / pipeline / serve
├── db.py            连接池 + schema 诊断
├── schema/          数据形状：Document / Chunk / SearchHit / RAG 答案与配置
├── api/             只读 HTTP 传输层：/healthz、/readyz、/answer；acceptance.py 投影验收 artifact
├── ingestion/       loader → chunker → embedder → indexer
├── retrieval/       dense / sparse / fusion / reranker / pipeline / registry（评测 arm 注册表）
├── rag/             context / prompt / provider / pipeline / factory / historical
├── evaluation/      metrics / benchmark / report / artifacts / *_benchmark
└── utils/           logger / timing

data/               数据集（随仓库提交）
docs/               architecture.md（阶段 ① 设计稿）/ roadmap.md（当前状态与路线）
reports/            dense/sparse/hybrid/rerank/rag 的历史 JSON 与 Markdown 报告
reports/unified/    统一评测 artifact（benchmark / index_scale10 / chunking* / fusion / rerank_20to5）
reports/figures/    报告图（5 张 PNG，由 scripts/report.py 生成）
reports/container/  容器验证日志（ingest、chunking、pytest、精排、索引、报告）
scripts/ingest.py   数据导入入口（转发给 src/cli.py，不含逻辑）
scripts/benchmark_dense.py / benchmark_sparse.py / benchmark_hybrid.py / benchmark_rerank.py  各阶段只读评测入口
scripts/benchmark_rag.py  阶段 ⑧ 问答评测入口（mock 默认；真实 Provider 需 API Key）
scripts/benchmark_retrieval.py  统一六臂评测（registry）
scripts/benchmark_index.py       HNSW / IVFFlat 参数扫描
scripts/benchmark_chunking.py    三种分块策略对比
scripts/benchmark_fusion.py      RRF 敏感性
scripts/benchmark_all.py         子进程编排 + 前置门禁
scripts/report.py                聚合 artifact → final_report.md + figures
scripts/warm_cache.py            预热模型缓存（不写报告、不连数据库）
scripts/ask_rag.py               阶段 ⑧ 单问题诊断入口
scripts/diagnose_dense.py        只读诊断：GUC、自然/强制 HNSW 计划、Top-K
web/acceptance/     无构建步骤的只读验收页（index.html / styles.css / app.js）
docker/init.sql     扩展、表结构、GIN 与 HNSW 索引
docker/init.sh      首次初始化时把 HNSW 参数注入 init.sql
```

---

## 数据库

`docker compose up -d db` 会在一个空数据目录上依次创建：

| 对象 | 说明 |
|---|---|
| `vector` 扩展 | pgvector |
| `documents` | 与 `kb_documents.json` 一一对应，**不含向量** |
| `chunks` | 向量与检索向量所在处；主键 `(chunk_strategy, chunk_id)`，三种分块策略可共存 |
| `chunks.tsv` | `GENERATED ALWAYS ... STORED`，title/card_tier 权重 A、正文权重 B |
| `chunks_tsv_gin` | 稀疏检索用（阶段 ⑤） |
| `chunks_embedding_hnsw_idx` | HNSW，`vector_cosine_ops`，默认 `m=16, ef_construction=64` |

HNSW 参数可用 `APEX_HNSW_M` / `APEX_HNSW_EF_CONSTRUCTION` 覆盖，但**只影响空数据目录上
执行的 `init.sql`**；已有数据库不应通过 `make reset`（会删除 volume）来改参数，而应使用
shadow 索引构建。`indexer.py` 提供显式的 `build_hnsw_index()` /
`build_ivfflat_index()` / `drop_index()` / `list_vector_indexes()`：默认创建带版本号的
shadow 索引（`chunks_embedding_hnsw_vNNN`），不会删除或替换 `chunks_embedding_hnsw_idx`；
`ingest` 不会隐式重建索引。索引创建/删除尚未在本轮执行，只有纯单元测试覆盖命名与参数校验
（见“测试与静态检查”）。

`embedding` 列是 `NOT NULL`：pgvector 会把 `NULL` 向量的行静默排除在 HNSW 之外，
那样的行对顺序扫描可见、对 ANN 检索不可见，会污染 Recall@K 而没有任何测试能发现。

---

## 稠密检索（阶段 ④-B）

`src.retrieval.dense.DenseRetriever` 是后续 Benchmark 的稠密检索入口：

```python
from src.retrieval.dense import DenseRetriever
from src.schema.retrieval import DenseConfig

retriever = DenseRetriever(DenseConfig(top_n=10, ef_search=40))
hits = retriever.retrieve("How many days do I have to request a 0% intro balance transfer?")
```

- 查询文本复用阶段 ③ 的 `SentenceTransformerEmbedder`（同一模型、查询指令与 1024 维）。
- SQL 先在内层子查询 `ORDER BY embedding <=> %s LIMIT %s` 固定 Top-K，再 JOIN `documents`；
  JOIN 只补元数据，不会放大或重复结果。每个结果同时携带 `chunk_id` 与 `doc_id`。
- `SearchHit.distance` 是 pgvector 余弦距离（越小越好），`score = 1 - distance`（越大越好）。
- `DenseConfig.chunk_strategy`（默认 `document`）作为 SQL 参数过滤 `chunks.chunk_strategy`；
  返回行若携带其他策略，映射器会报错而不是静默混入。Sparse 的英文与 CJK 查询同样受该作用域约束。
  `Settings.retrieval_chunk_strategy`（`APEX_RETRIEVAL_CHUNK_STRATEGY`）决定 RAG 链路使用哪一策略；
  四个检索 Benchmark 脚本另有 `--chunk-strategy`（`document` / `fixed_window` / `structured`，默认 `document`）。
  历史 RRF 重放（rerank / RAG）也会校验 artifact 的 `parameters.chunk_strategy` 与每条命中，
  作用域不一致即报错，不会把其他策略的 chunk 混入重放结果。
- `DenseConfig.ef_search` 通过 `src.db.ef_search()` 在查询事务内用 `set_config(..., true)` 设置；
  `m` / `ef_construction` 是建索引参数，不会作用于已存在的索引。
- 当前 150 chunks 下自然计划是 `Seq Scan`，所以 `ef_search` 对自然查询没有可观察影响。
  用只读脚本分别查看真实结果、GUC 与计划：

```bash
docker compose run --rm -T app python scripts/diagnose_dense.py --ef-search 7 --top-k 50 --force-hnsw
```

`--force-hnsw` 只在单个事务内执行 `SET LOCAL enable_seqscan = off` 用于诊断，不是生产默认。

---

## 混合检索与评测（阶段 ⑥-B）

`src.retrieval.fusion.RRFFusion` 实现标准 RRF：

```
RRF(d) = Σ_r 1 / (k + rank_r(d))
```

- `rank_r(d)` 是 chunk 在单路候选截断后的 **1-based** 名次；两路都命中的 chunk 累加两项，
  只命中一路的只累加该项；稠密分数与稀疏分数**从不相加**。
- 融合身份是 `(chunk_strategy, chunk_id)`；指标按 `doc_id` 去重后截断到 Top-K。
- 候选并集基线使用可复现的交错顺序 `dense[0], sparse[0], dense[1], sparse[1], ...`，
  去重后截断；它没有融合分数，`score` 固定为 `0.0` 占位。
- 单臂为空时只返回另一臂的贡献；两臂都空返回 `[]`；单路重复 chunk、非法参数、
  非有限分数会在融合前报错。

推荐的完整评测命令（输出原始逐查询结果与汇总指标）：

```bash
docker compose run --rm app python scripts/benchmark_hybrid.py \
  --top-k 10 --candidate-k 10 --rrf-k 60 --repetitions 5 --warmup-passes 1
```

Docker 不可用但宿主能连接数据库时，可用同一脚本在宿主运行（本次 ⑥-B 实测即这种方式）：

```bash
APEX_PG_HOST=127.0.0.1 python scripts/benchmark_hybrid.py --top-k 10 --candidate-k 10 \
  --rrf-k 60 --repetitions 5 --warmup-passes 1
```

本次 30 条 Query 的结果（完整表格与口径见 [`reports/hybrid_benchmark.md`](reports/hybrid_benchmark.md)）：

| 臂 | Recall macro | MRR | Hit@1 |
|---|---|---|---|
| Dense | 0.944 | 0.917 | 25/30 |
| Sparse OR | 1.000 | 0.940 | 27/30 |
| Sparse AND | 0.133 | 0.133 | 4/30 |
| 候选并集 | 1.000 | 0.917 | 25/30 |
| RRF | 1.000 | 0.950 | 27/30 |

报告称 Sparse OR 已经覆盖全部 43 个 gold；同一报告的 `dataset.distinct_gold_docs`
字段为 **38**。本阶段对 `data/eval_queries.json` 做只读统计后确认：30 条 Query 共
包含 **43 条 gold 引用**、对应 **38 篇不同文档**（部分文档被多条 Query 引用）。
因此“43 gold documents”的措辞不准确，应为“43 条 gold 引用”；在此数据集上
Recall 无法超过 1.0 的结论不受影响。RRF 的增益体现在 MRR
（0.940 → 0.950）与五个 Dense 漏检 gold 的补回（均进入 RRF Top-10）。

---

## 重排序（阶段 ⑦-B）

`src.retrieval.reranker.CrossEncoderReranker` 是独立的第二阶段：它不检索、不融合、
不访问数据库，只对传入的 `(query, passage)` 对打分并重排。模型在一个进程内只加载
一次并复用；空候选返回 `[]` 且不加载模型。

- 输入文本为 `title + "\n" + content`；正文缺失是输入错误，不会被静默跳过。
- 候选身份是 `(chunk_strategy, chunk_id)`；重复身份直接报错，避免分数与 ID 映射歧义。
- 排序键：`rerank_score` 降序 → 重排前 rank → `(chunk_strategy, chunk_id)`。
- 返回的每个 hit 保留原元数据与融合来源，并记录 `rerank_score`、`pre_rerank_rank`、
  `pre_rerank_score`；`source="reranked"`，`distance` 清空。
- 没有静默 fallback：模型初始化、推理、资源不足、输入、数据库读取失败分别归类为
  `model_init` / `inference` / `resource` / `input` / `db_fetch`，失败即抛错。

Makefile 提供容器内目标 `make benchmark-rerank`；本阶段 **未验证** 容器内的模型缓存
与离线加载路径，因此不能把宿主机结果当成容器验证。本次实测在宿主运行（Docker 不可用），
使用同一脚本并复用 ⑥-B 的历史四臂原始结果：

```powershell
$env:APEX_PG_HOST='127.0.0.1'; $env:APEX_PG_PORT='5434';
python scripts/benchmark_rerank.py --top-k 10 --candidate-k 10 --rrf-k 60 `
  --batch-size 16 --activation identity
```

模型来源与本地缓存：

| 项 | 值 |
|---|---|
| 模型 / revision | `BAAI/bge-reranker-base` / `2cfc18c9415c912f9d8155881c133215df768a70` |
| 许可证 | MIT（模型卡注明可商用） |
| 缓存位置 | `.cache/huggingface/hub/models--BAAI--bge-reranker-base` |
| 权重文件 | `model.safetensors` 1,112,206,140 bytes，SHA256 `ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd` |
| 5 个模型文件合计 | 1,134,374,540 bytes |
| 缓存写入窗口 | 约 58 s（16:23:03–16:24:01，来自文件 mtime，未单独计时下载命令；无独立下载日志） |

30 条 Query 的五臂结果（完整表格与逐查询行见
[`reports/rerank_benchmark.md`](reports/rerank_benchmark.md)）：

| 臂 | Recall macro | MRR@10 | Hit@1 | 来源 |
|---|---|---|---|---|
| Dense | 0.9444 | 0.9167 | 25/30 | ⑥-B 历史结果（独立复核） |
| Sparse OR | 1.0000 | 0.9400 | 27/30 | ⑥-B 历史结果（独立复核） |
| Sparse AND | 0.1333 | 0.1333 | 4/30 | ⑥-B 历史结果（独立复核） |
| RRF | 1.0000 | 0.9500 | 27/30 | ⑥-B 历史结果（独立复核） |
| RRF + Reranker | 1.0000 | **0.9833** | **29/30** | 本次实时重排 |

RRF → RRF+Reranker：MRR improved 3 / equal 26 / worse 1，平均 ΔMRR +0.0333；
首个 gold 的平均名次变化 −0.0667；30/30 查询的 Recall 保持不变。排序变化的查询为
Q-002、Q-011、Q-016、Q-017：其中 Q-002、Q-016、Q-017 的倒数排名从 0.5 升到 1.0，
Q-011 从 1.0 降到 0.5（三个 gold 仍全部留在 Top-10）。没有为了指标而调整 gold、查询或池子。

资源与延迟（host CPU，非容器）：模型加载 25,992.8 ms；加载后 RSS 848.1 MiB；
运行峰值 RSS 1,288.9 MiB / peak WSET 1,314.6 MiB；系统可用内存最低 1.123 GiB。
10 候选 batch 1/16：1,163.2 / 1,082.5 ms；50 候选 batch 1/16：5,742.2 / 4,599.4 ms。
每候选约 92–116 ms；batch 1 与 16 的最大分数差 5.0e-6。CPU 单进程顺序执行，
不代表 GPU、并发或跨机器性能。

**口径限制**：四个基线臂是 ⑥-B 的历史结果，不是本次重新查询数据库；重排臂使用同一
查询的历史 RRF Top-10 身份，文本由 PostgreSQL 实时读取。由于候选池相同，Recall 不可能
因重排上升，本阶段只考察 MRR/Hit@1。Embedding 与 Reranker 未在同一进程同时加载，
因此本阶段不声称已解决二者共存的内存问题。

---

## RAG 端到端问答（阶段 ⑧）

阶段 ⑧ 不改动已验收的 Dense、Sparse、RRF 与 Reranker，只在它们之后增加生成层：

```
question
  │
  ▼  RAGPipeline.answer()
  输入校验（空串/非字符串 → invalid_request）
  │
  ▼  Provider 配置检查（缺 Key → config_error，在检索前失败）
  │
  ▼  RerankedRetriever（候选拉取 → Cross-Encoder → 最终 Top-K）
  │
  ▼  build_context()
  PostgreSQL chunks.content 真实文本 → E1..En（按最终 rank）
  单块上限 + 总字符上限；截断/丢弃显式记录；SHA256 固定实际上下文
  │
  ▼  build_messages()  prompt rag-answer-v1
  │
  ▼  OpenAI-compatible Provider（DeepSeek 默认端点）
  │
  ▼  JSON 解析 → 引用校验 → RAGAnswer
```

关键行为：

| 场景 | 行为 |
|---|---|
| 检索为空 / 上下文为空 | 不调用 Provider，返回确定性的 `insufficient_evidence` |
| 模型声明证据不足 | 保留 `insufficient_evidence`，不强迫编造 |
| 模型引用未提供的标签 | 进入 `invalid_citations`，绝不进入 `citations` |
| 上下文超预算 | 先截断单块，再按 rank 丢弃后续候选；记录 `truncated` / `dropped` |
| Provider 超时 / 限流 / 网络 / 5xx / 401 | 分别映射为独立 `llm_*` 状态；只对超时、网络、限流、5xx 做有界重试 |
| 空响应 / 非 JSON / 长度截断 | `llm_invalid_response`，不把半截文本当答案 |
| 缺少 API Key | `config_error`，不会静默回退到 mock |
| 检索 / 重排 / 证据异常 | `retrieval_error` / `rerank_error` / `evidence_error`，与生成失败区分 |

Provider 复用现有配置：`APEX_LLM_API_KEY`（或 `DEEPSEEK_API_KEY`）、
`APEX_LLM_BASE_URL`（默认 `https://api.deepseek.com`）、`APEX_LLM_MODEL`，
以及独立的 connect/read timeout 与有界重试次数。Key 只写入 Authorization 头，
不进入 README、日志、报告或测试；测试默认注入 mock Provider 与 `httpx.MockTransport`。
生成模型、Embedding 模型与 Reranker 是三个独立组件。

本次正式评测命令（宿主，30 条 Query，mock Provider）：

```powershell
$env:APEX_PG_HOST='127.0.0.1'
python scripts/benchmark_rag.py --provider mock --retrieval-mode historical
```

`APEX_PG_HOST=127.0.0.1` 是本机绕过：`localhost` 同时解析到 IPv6/IPv4 时连接池会
耗满 `connect_timeout`；它不是代码默认值。宿主运行需要仓库 Python 依赖已安装；
容器内的 RAG live 路径已在后续轮次以真实 30/30 运行验证。

结果见 [`reports/rag_benchmark.md`](reports/rag_benchmark.md)（**Status: PARTIAL**，
原因是这份正式报告没有调用真实 LLM；独立的 30/30 真实 RAG + Judge 见 Judge 一节）：

| 指标 | 值 | 口径 |
|---|---|---|
| 端到端成功率 | 1.0000 (30/30) | status ∈ {ok, insufficient_evidence} |
| 生成失败率 | 0.0000 (0/30) | Provider 已调用但未产出结构化答案 |
| 证据不足率 | 0.0000 (0/30) | 空检索短路 + 模型声明证据不足 |
| 引用有效率 | 1.0000 (30/30) | 有效 `E<n>` / 全部返回标签 |
| 证据 Recall macro / micro | 1.0000 / 1.0000 | 最终证据集合，检索指标 |
| 证据 MRR | 0.9833 | 最终证据集合，检索指标 |
| 端到端延迟 P50 / P95 | 970.9 / 1082.2 ms | n=30，单进程，未剔除首次模型加载 |
| LLM 延迟 P50 / P95 | 0.07 / 0.09 ms | n=30，mock Provider，不代表真实网络 |

**限制（必须与结果一起阅读）**：

- Provider 是确定性的 mock，答案以 `MOCK` 开头，不是真实模型输出；不能据此声称
  答案正确、Faithful 或 Relevant。正式报告使用 mock Provider，没有调用真实 API；
  独立执行的 30/30 真实 RAG + Judge（`reports/judge/`）不改变本报告数字。
- 检索是 **historical 重放**：候选身份与 RRF 名次来自 ⑥-B artifact，文本实时读自
  PostgreSQL，Reranker 实时运行；Dense/Sparse/RRF 未在本阶段重新执行。因此
  证据指标衡量的是被重放的候选池，不是一次新的检索运行。
- 在本机把 Embedding 与 Reranker 放入同一进程时，live 链路在 reranker 加载阶段
  触发原生段错误（exit 139）。这不是测试失败，也没有被隐藏；historical 模式是
  为避免该崩溃而明确标注的替代路径。
- 本报告的答案正确性、Answer Relevance 未计算：数据集只有 `gold_response` 参考文本，
  且该报告生成时没有 Judge。Context Relevance / Faithfulness 现由独立的 Judge 模块
  评估，其分数只存在于 `reports/judge/*`；本轮已执行真实 30/30 全量 Judge（PASS，
  `same_model=true`），另有历史 3/30 smoke（`reports/judge-real-3q/`）与独立 mock
  闭环 `reports/judge-smoke/*`。三者都不并入本报告的历史数字；30/30 是自动 Judge
  的全量结果，但仍不是人工校准的质量结论。

单问题诊断（historical 模式只接受 `data/eval_queries.json` 中记录的原问题文本）：

```powershell
python scripts/ask_rag.py "What is the balance transfer fee for the Everyday Freedom card?" `
  --mock --retrieval-mode historical
```

真实 Provider（需要环境变量中的 Key，禁止写进代码/报告）：

```powershell
$env:APEX_LLM_API_KEY='<set-in-your-shell>'
python scripts/benchmark_rag.py --provider real --retrieval-mode historical --limit 2 --no-write
```

---

## LLM-as-a-Judge 自动评估（Context Relevance / Faithfulness）

在既有 RAG 结果上评分，**不重新检索、不改变答案**。输入是 RAG artifact 的
`per_query`（问题、答案、本次实际送入生成模型的证据 `E1..En`）；不使用
`gold_response`、不写数据库、不修改 `data/`。

| 指标 | 评分对象 | 输出 |
|---|---|---|
| Context Relevance | 检索到的证据集合 | 1–5 分、理由、relevant / irrelevant 标签 |
| Faithfulness | 生成的答案 | 1–5 分、逐 claim 支持状态（`supported` / `partially_supported` / `unsupported` / `contradicted`）、unsupported claims |

### 命令

```bash
# 默认 mock：只验证接线、解析、失败隔离与聚合，不是模型判断
make judge ARGS="--limit 3"
python scripts/evaluate_judge.py --provider mock --judge-provider mock --limit 3 \
  --output-dir reports/judge-smoke

# 真实 smoke：真实 RAG + 真实 Judge，只跑 3 条
python scripts/evaluate_judge.py --provider real --judge-provider real --limit 3 \
  --retrieval-mode live --allow-network --output-dir reports/judge

# 只重评已有 RAG 答案，不重复生成
python scripts/evaluate_judge.py --rag-artifact reports/judge/rag_answers.json \
  --judge-provider real --output-dir reports/judge-rejudge

python -m src.cli evaluate-judge --help
```

### 产物

`reports/judge/`：`rag_answers.json` / `.md`（RAG 中间产物）、
`judge_results.jsonl`（每 Query 一行）、`judge_summary.json`（汇总、失败、限制与
provenance）、`judge_report.md`。`make report` 把 `judge_summary.json` 作为 optional
artifact 并入 `reports/final_report.md`，并新增 `reports/figures/judge_distribution.png`。
`reports/judge/` 现为 30 条全量真实运行（`PASS`）；3/30 真实 smoke 归档在
`reports/judge-real-3q/`。`reports/judge-smoke/` 是独立的 mock 闭环，不与真实
artifact 混用。全量完成后已运行 `python scripts/report.py --overwrite` 重新生成
`reports/final_report.md` 与 `judge_distribution.png`，报告输出 `Report status: PASS`。

### 状态语义

| 状态 | 条件 |
|---|---|
| PASS | 真实 RAG + 真实 Judge，30/30 Query 两个指标都成功 |
| PARTIAL | 覆盖少于预期查询数（如 3/30 smoke）、任一指标有失败、Mock、或 RAG 答案为 mock |
| FAIL | 两个指标都没有有效分数，或全部 RAG Query 失败 |
| BLOCKED | 请求真实调用但 `APEX_LLM_API_KEY` / `DEEPSEEK_API_KEY` 缺失：退出码 2，不写 artifact |

聚合口径：均值/中位数/1–5 分布只使用 `status=success` 的分数；失败按 status /
error category 单独计数，**不按 0 分计入**，也不静默忽略。summary 与最终报告都写明
`queries / expected_queries`、`NOT RUN` 条数与 `same_model`。

Judge 复用 `APEX_LLM_MODEL`（可用 `APEX_JUDGE_MODEL` 覆盖），因此默认是**同模型自评**，
存在未量化的自评偏差；分数是模型判断，不是人工真值。Prompt 版本与 SHA256 按
`judge-context-relevance-v1` / `judge-faithfulness-v1` 记录。API Key 只从环境读取，
artifact、日志与报告只记录 `api_key_present` 布尔值。

**本轮执行状态（2026-10-20）**：实现、静态检查、定向/全量测试、容器 mock 闭环与
**真实 30 条全量 Judge** 均已实际执行。全量结果：`status=PASS`（30/30、0 NOT RUN）、
`judge_is_real=true`、`real_judge_calls=true`、`same_model=true`。Context Relevance
mean **3.9667** / median 4（分布 2×1、3×4、4×20、5×5；低分 1 条，Q-029）；
Faithfulness mean **5.0000** / median 5（5×30）；两指标各 30/30 成功、0 失败；
`queries_with_judge_failure=0`，数据完整性检查为 `True`。此前的 3/30 真实 smoke
记录完整保存在 `reports/judge-real-3q/`。

### 执行记录（2026-10-20）

| 检查 | 命令 | 实际结果 |
|---|---|---|
| Ruff | `python -m ruff check src scripts tests` | **PASS**：`All checks passed!`（宿主） |
| mypy | `python -m mypy src` | **PASS**：`Success: no issues found in 54 source files`（宿主） |
| 定向 pytest | `python -m pytest tests/test_judge.py tests/test_judge_evaluation.py tests/test_report.py -q -p no:cacheprovider` | **PASS**：36 tests / 0 failed / 0 errors / 3 skipped（宿主缺 `matplotlib`；JUnit 见 `reports/container/pytest-judge.xml`） |
| 容器 mock smoke | `docker compose run --rm app python scripts/evaluate_judge.py --provider mock --judge-provider mock --limit 3 --output-dir reports/judge-smoke` | **PARTIAL（符合预期）**：3/30、`not_run_queries=27`、两指标各 3/3 成功、mean/median 4.0（fixture）、`judge_is_real=false`、JSONL 3 行；产物在 `reports/judge-smoke/` |
| 容器真实 smoke（历史） | `docker compose run --rm app python scripts/evaluate_judge.py --provider real --judge-provider real --limit 3 --retrieval-mode live --allow-network --output-dir reports/judge --overwrite` | **PARTIAL（真实执行）**：3/30、27 NOT RUN；`judge_is_real=true` / `real_judge_calls=true` / `same_model=true`；CR mean 4.3333、Faithfulness mean 5.0000；JSONL 3 行；记录已归档到 `reports/judge-real-3q/` |
| 容器真实全量 | `docker compose run --rm -e APEX_LLM_API_KEY app python scripts/evaluate_judge.py --provider real --judge-provider real --retrieval-mode live --allow-network --output-dir reports/judge --overwrite` | **PASS（真实执行，2026-10-20）**：30/30、0 NOT RUN、0 Judge 失败、0 RAG 失败；CR mean 3.9667、Faithfulness mean 5.0000；JSONL 30 行；`Integrity: True`；产物 `reports/judge/` |
| 报告集成（全量后重生成） | `docker compose run --rm app python scripts/report.py --overwrite` | **PASS**：`Wrote /app/reports/final_report.md`、5 张 PNG（含 `judge_distribution.png`）重新生成，`Report status: PASS` |
| 容器全量 pytest | `docker compose run --rm app python -m pytest -q -o addopts="" -p no:cacheprovider --junitxml=reports/container/pytest-full-judge.xml` | **PASS**：`691 passed, 1 skipped`（30.42s） |

真实调用需要环境中的 Key：缺 Key 时脚本打印 `BLOCKED` 且不写 artifact（本轮实测
两次）；用 `docker compose run -e APEX_LLM_API_KEY ...` 注入后完成 3 条 smoke 与
30 条全量真实调用。不得用 mock 或历史 replay 结果替代真实评分。

本轮实测暴露的限制：

- **模型标识不一致**：artifact 的配置字段是 `deepseek-chat`，但真实 smoke 的 6 次
  响应 `model` 都是 `deepseek-flash`。两者来自同一 endpoint，`same_model` 仍为 true；
  但报告里的模型名应理解为“配置名”，真实响应以 per-metric `model` 字段为准。
- **rubric 执行可能不一致**：Q-001 的 Context Relevance 得 4 分，却把 10 条 context
  中的 9 条标为 irrelevant（只保留 E1）。全量运行中 Q-029 得 2 分，理由是唯一提到
  Global Entry 的证据属于另一张卡。分数与标签列表出现张力，说明 Judge 对评分标准
  的执行未被人工校准；这类分数不能当作客观真值。
- **同模型自评**：Judge 与答案生成复用同一模型，`same_model=true`；自评偏差未量化，
  也没有独立模型或人工复核。

---

## 统一评测、报告与服务（2026-10-20 新增；容器内已执行）

### 统一评测

```bash
make benchmark-retrieval     # 6 臂：dense / sparse_or / sparse_and / union / rrf / rrf_rerank
make benchmark-index         # HNSW m×ef_construction；IVFFlat lists×probes
make benchmark-chunking      # document / fixed_window / structured
make benchmark-fusion        # RRF k × candidate_k
make benchmark-all           # 以上前三项顺序执行
make report                  # reports/final_report.md + reports/figures/*.png
```

产物路径固定为 `reports/unified/*.json` 与 `.md`；默认**拒绝覆盖**已有文件，
必须显式 `--overwrite`。`verify-container` 与历史报告路径不重叠。

规则（写入代码，并已在容器执行中验证）：

- 指标在 benchmark 中计算一次；`report` 只读取 artifact，不查库、不重算、不补零。
- 报告中的每个表格都带 `[artifact: path#field sha256=...]`，可回到 JSON 字段。
- 状态只有 `PASS` / `PARTIAL` / `FAIL` / `BLOCKED`；`PARTIAL` 必须写明缺多少。
- 索引实验使用临时表副本（`CREATE TEMP TABLE ... AS SELECT`），不触碰服务中的
  `chunks` 表与 bootstrap HNSW 索引；`--synthetic-scale N` 的产物标记 `SYNTHETIC`，
  **不进入质量指标**。
- 分块对比按 `doc_id` 去重后计算文档级指标；三策略必须都已 ingest。

### 预览命令（不写文件）

```bash
make benchmark-retrieval ARGS="--dry-run"
make tune ARGS="--dry-run --synthetic-scale 100"
make report ARGS="--no-write"
```

### 只读 API

```bash
make api                     # docker compose --profile api up -d api
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the foreign transaction fee?"}'
```

- `POST /answer` 只接受 `query`（1–2000 字符）与可选 `query_id`；Top-K 是服务端配置。
- 响应不包含 prompt、`context_text` 或 Provider 原始输出，也不包含任何密钥。
- `insufficient_evidence` 是 200；检索/重排/配置错误、数据库不可用与模型缓存缺失
  为 503，超时 504，限流 429，非法 Provider 响应 502；未列出的状态为 500。
- `/readyz` 只检查数据库；模型在首次 `/answer` 时懒加载。

**搜索接口（题目 Bonus C）**

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the foreign transaction fee?", "top_k": 5}'
```

- `POST /api/v1/search` 只做检索，**不调用 LLM**：返回 `chunk_id`、`doc_id`、`rank`、
  `score`、`rerank_score`、元数据与 `content`，以及分阶段 `timings_ms`。
- `query` 为 1–2000 字符，首尾空白会被去除，全空白拒绝；`top_k` 可选（1–50），
  但不得超过服务端候选池 `rag_candidates`（默认 10），否则返回 422。
- 查询复用 `build_pipeline().retriever`，因此 embedding 与 reranker 在每个进程只加载一次；
  数据库/检索阶段失败返回 503，未预期异常返回 500，均不回显路径、SQL 或密钥。
- Swagger UI 在 `http://localhost:8000/docs`，OpenAPI 描述在
  `http://localhost:8000/openapi.json`；端点声明了 `search` 标签、请求/响应模型与
  422/503/500 响应说明。
- **运行时验证状态（2026-10-20，容器）**：`api` 容器 healthy；`POST /api/v1/search`
  返回 200，`hit_count=5`，Top-1 为 `DOC-0035`（Apex Platinum，FEE-202，0 外币手续费），
  Top-2 为 `DOC-0038`（Apex Business Select，1%）；`/docs` 返回 200，
  `/openapi.json` 含 `/api/v1/search`。首次请求包含 embedding 与 reranker 加载，
  `retrieval_ms=19428.86`、`base_ms=14834.811`、`rerank_ms=4593.996`，
  **不能当作稳态延迟**。重启 `api` 后已按最终文件复跑：`/healthz` 与
  `POST /api/v1/search` 均返回 200（`retrieval_ms=14964.789`，仍含模型加载）。
  Ruff `All checks passed!`；mypy
  `Success: no issues found in 55 source files`；`tests/test_api.py` 23 个用例
  执行完成、无失败（见 `docs/roadmap.md` 第 10.8 节）。

### 页面验收（只读）

```bash
make acceptance              # 复用 api profile；页面 http://localhost:8000/acceptance/
curl http://localhost:8000/acceptance/api/summary
curl "http://localhost:8000/acceptance/api/judge/queries?limit=50&offset=0"
```

- 页面与接口**只有 GET**，不提供验收结论提交、审批或写库入口；POST/PUT/PATCH/DELETE
  返回 405。
- 数据只来自 `reports/unified/benchmark.json`、`index_benchmark_scale10.json`、
  `chunking_benchmark_rerank.json`、`rerank_20to5.json` 与
  `reports/judge/judge_summary.json`；路径固定，不接受客户端传入文件路径。
- 总状态与原因调用 `src/evaluation/report.py` 的 `report_status()`，与最终报告同源；
  页面不重算指标，缺失值显示 `—`，`NOT RUN` 不按 0 分处理。
- Judge 区显示覆盖范围（当前 30/30、0 NOT RUN）、`is_real`、`real_judge_calls`、
  `retrieval_mode` 与 `same_model`；响应剔除 `raw_response`、usage、answer/context
  与 prompt 哈希、prompt 文本及 Provider endpoint。artifact 文件 SHA256 与数据集
  哈希保留，用于核对页面读到的就是仓库中的那份文件。
- 页面固定声明“只读”和“不是人工验收结论”；自动检查只是机械规则，不能替代原始报告。
- 静态文件在 `web/acceptance/`（原生 HTML/CSS/JS，无构建步骤、无 CDN、无新依赖），
  脚本只使用 `textContent` 写入 artifact 文本。

**验证状态**：宿主机 `ruff check src scripts tests` 为 **PASS**（`All checks passed!`），
`mypy src` 为 **PASS**（55 source files），定向 pytest（本页两组测试 + `test_api.py` +
`test_report.py`）为 **PASS**（0 failed、3 skipped）。**该记录早于 Bonus C 新增的
`/api/v1/search` 用例与代码**；搜索端点随后单独复跑：Ruff PASS、mypy PASS、
`test_api.py` 23 个用例无失败（见 §10.8）。容器内 `api` 为 healthy：
`/healthz` 返回 `{"status":"ok","service":"apex-rag-api"}`，
`/acceptance/api/summary` 返回 `status=PASS`，五个 artifact 全部 `present=true` 且
全部 `PASS`，6/6 自动检查为 `pass`（`all_loaded_pass=5/5`、`judge_coverage=30/30`）。
**浏览器渲染与静态页请求仍为 `NOT RUN`**，
分别记录见 `docs/roadmap.md` 第 10.7 节；未执行的项不会写成 PASS。

### 容器与离线模型

```bash
make warm-cache              # 下载 embedding + reranker 到 hf_cache 卷
docker compose run --rm -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  app python scripts/benchmark_rerank.py --limit 2 --no-write   # 离线复现检查
make verify-container        # db-check → benchmark-all → report，日志写 reports/container/
```

上述统一评测 / 报告 / 全量 pytest、真实 30/30 RAG + Judge 与 `make api` 服务已在
容器内直接执行（日志在 `reports/container/`）；`make verify-container` 一键目标
仍未执行。Dockerfile 安装 `.[dev,report,api]`；compose 增加了
`api`（profile `api`）与 `warm-cache`（profile `tools`）服务。

---

## Windows Reranker 原生崩溃调查状态

历史现象：在本机把 Embedding 与 Reranker 放入同一进程运行 RAG live 链路时，
曾在 reranker 加载阶段触发 Windows 原生崩溃。已保留的证据签名为
`python.exe` / `torch_cpu.dll` / 异常 `0xC0000005` / 偏移 `0x8e4a269`，
包含 5 份 WER 报告与 5 份 minidump。这些历史证据没有被删除或改写。

已完成的受控探针均使用独立子进程、独立证据目录、运行前后 WER 快照：

| 探针 | 覆盖范围 | 结果 | 证据目录 |
|---|---|---|---|
| E-5A | 1 个候选、`batch_size=1`、`top_k=1`、单次 `rerank()`；36 token；1 次 predict / forward | 未复现崩溃；分数有限；ID 映射保持；无新 WER/minidump | `D:\RAG_crash_evidence\E5_2026-10-19_1205\E5A` |
| E-5C | 空候选短路；不加载模型、不导入 torch | `model_factory=0`、`predict=0`、`forward=0` | `D:\RAG_crash_evidence\E5_2026-10-19_1207\E5C` |

Stage A 原始证据审核对上述两份报告、`probe_result.json`、stdout/stderr、
`memory_samples.csv`、WER 快照和 `SHA256SUMS.txt` 做了逐项核对，判定
**E-5A PASS、E-5C PASS**；九项旧证据关键哈希、E-1 完整副本/原件清单与
E-4R 完整清单均 PASS；E-2/E-4 因缺少完整清单，完整目录完整性为
**INCONCLUSIVE**。审核报告位于
`D:\RAG_crash_evidence\E5_REVIEW_2026-10-19_140237\E5_REVIEW_REPORT.md`，
SHA256 为 `7f2a79f3dc2e785a570f319b87ce7dd9dbe954d5186c5f84ca593220347d6327`。

**边界（必须与 PASS 一起阅读）：**

- E-5A/E-5C 的 PASS 只表示“这两次探针的报告与原始证据一致，且本次未观察到新崩溃”，
  **不表示历史原生崩溃的根因已确认或已修复**。
- E-5A 只覆盖单候选、batch=1、36 token；E-5C 只覆盖空候选。两者都不覆盖多候选、
  多 batch、长输入、Embedding 与 Reranker 同进程共存。
- **E-5B（多候选）未执行**。现有 `e5a_rerank_single.py` 硬编码单候选，不能直接
  改参数用于多候选；新的独立多候选探针尚未准备和审核。本阶段决定暂停 E-5B
  及其他崩溃专项测试，优先完成文档。
- Stage A 审核时点（2026-10-19 14:10:15 / 14:10:17）两次可用内存为
  **3.842 / 3.848 GiB**，低于 E-5B 的 4.5 GiB 门槛，环境判定 FAIL。审核后会话内
  另行只读采样（14:32:58 / 14:33:05）为 **4,926 / 4,925 MB（约 4.81 GiB）**，
  高于门槛；这是较晚的系统状态，不是对审核报告的修正，也未据此执行 E-5B。
- 历史 `0xC0000005` / `torch_cpu.dll` 证据仍在；E-5A/E-5C 没有新增 WER/minidump
  不等于“历史崩溃从未发生”，也不等于“问题已修复”。

---

## 技术选型

| 领域 | 选择 |
|---|---|
| 数据库 | PostgreSQL 16 + pgvector（HNSW，余弦） |
| 稠密检索 | `BAAI/bge-large-en-v1.5`（1024 维，**本地运行，无需 API Key**） |
| 稀疏检索 | PostgreSQL 全文检索（加权 `tsvector` + GIN + `ts_rank_cd`） |
| 融合 | RRF（k=60，已实现）+ 可复现的候选并集基线；加权归一化**未实现** |
| 重排序 | `BAAI/bge-reranker-base` Cross-Encoder（⑦-B 已实现并在宿主 CPU 实测；MIT） |
| 生成 | OpenAI 兼容 HTTP Provider（默认 DeepSeek 端点）；直接使用 `httpx`，不引入新 SDK；mock/真实 Provider 可替换 |
| 服务 | FastAPI（只读：`/healthz`、`/readyz`、`/answer`、`POST /api/v1/search`；Swagger `/docs`），复用 `build_pipeline()`；模型懒加载，`/readyz` 只检查数据库 |
| 评测 | 自实现 Semantic Recall@K / Precision@K / MRR / 二元 nDCG@K / Hit@1 + P50/P95/P99（nearest-rank）；统一 artifact schema 与带哈希引用的报告；标签非穷尽，Precision/nDCG 只作补充指标；生成质量指标未计算 |

选型理由、被否决的备选方案，以及数据集本身的坑，都写在
[`docs/architecture.md`](docs/architecture.md) 里。

---

## 实现进度与阶段验收

> 本表是 README 中的阶段验收记录；`docs/roadmap.md` 只汇总当前状态与路线，不替代本表。
> 验收结论只依据仓库内报告、代码/测试与已记录的只读核验；没有报告支撑的项不会写成“通过”。

| 阶段 | 内容 | 实现状态 | 验收结论 | 依据 |
|---|---|---|---|---|
| ① | 项目结构、技术选型、模块职责、数据流、实验架构 | ✅ 完成 | 历史阶段，未在本记录中重新验收 | `docs/architecture.md` |
| ② | PostgreSQL / pgvector / Schema / HNSW / 连接层 | ✅ 完成 | 历史阶段，未在本记录中重新验收 | `docker/init.sql`、`src/db.py`、`make db-check` |
| ③ | Embedding 与 150 条数据导入 | ✅ 完成 | 历史阶段，未在本记录中重新验收 | 数据库 150 documents / 150 chunks；数据文件哈希见 ⑥-B 报告 |
| ④ | HNSW 建索引与参数 Benchmark | ✅ 完成 | 历史阶段，未在本记录中重新验收 | `reports/dense_benchmark.md`（Status: PASS） |
| ⑤-A | Sparse Retrieval 实现 | ✅ 完成 | **PASS** | `src/retrieval/sparse.py`、`tests/test_sparse_retrieval.py`；⑤-B 在真实数据库上运行通过 |
| ⑤-B | Sparse 独立评测与误差分析 | ✅ 完成 | **PASS** | `reports/sparse_benchmark.md` / `.json`（Status: PASS；30 queries；Sparse OR Recall 1.000；Sparse AND Recall 0.133、empty 26；MRR 0.940/0.133 与五臂同口径数据见 `reports/hybrid_benchmark.md`） |
| ⑥-A | Hybrid 前置核验与实验设计 | ✅ 完成（设计） | **PASS WITH MINOR ISSUES** | 设计核验已交付；⑥-B 指令确认该判定；无独立运行报告 |
| ⑥-B | Hybrid/RRF 五臂评测 | ✅ 完成（实测） | **PASS WITH MINOR ISSUES** | `reports/hybrid_benchmark.md` / `.json`（Status: PASS；30/30 queries，0 errors；RRF Recall 1.000、MRR 0.950、Hit@1 27/30；Dense Recall 0.944、MRR 0.917；Sparse OR Recall 1.000、MRR 0.940） |
| ⑦-A | Reranker 前置核验与设计 | ✅ 完成（只读核验与设计） | **PASS WITH MINOR ISSUES** | ⑦-A 只读核验报告（会话交付，结论与当前记录一致）；`db-check` 9 项 PASS；150 documents / 150 chunks；⑦-A 时点无 reranker 实现、模型未下载；Docker 不可用；依赖漂移。未验证项见下 |
| ⑦-B | Reranker 实现与评测 | ✅ 完成（实测，宿主 CPU） | **PASS WITH MINOR ISSUES** | `reports/rerank_benchmark.md` / `.json`（Status: PASS；30/30 queries，0 errors，0 fallbacks；RRF+Reranker Recall 1.0000、MRR 0.9833、Hit@1 29/30；RRF MRR 0.9500；历史四臂复核 0 mismatch，RRF 公式复核 300 hits / 0 mismatch）；未验证项保留在下方 |
| ⑧ | RAG 端到端问答链路集成与评测 | ✅ 完成（离线链路实测；正式报告未调用真实 LLM） | **待人工验收**（自评 PARTIAL，不替代验收） | `reports/rag_benchmark.md` / `.json`（Status: PARTIAL；30/30 status=ok，引用有效 30/30，证据 Recall 1.0000、MRR 0.9833；Provider=mock，historical 重放）；真实 30/30 RAG + Judge 见 ⑫ |
| ⑨ | README / roadmap 定稿 | 🟡 文档已定稿 | **待人工验收** | `README.md`、`docs/roadmap.md`；指标仅引用 `reports/*` 与 Stage A 审核报告，未重新运行任何测试 |
| ⑩ | 项目最终整合与 GitHub 提交准备 | ✅ 已提交并推送 | **已同步 `origin/main`**（`b31e053`） | 本阶段只读检查 README、roadmap、architecture、报告、依赖与忽略配置；提交本身未重新运行测试，测试结果单列于“测试与静态检查” |
| ⑪ | P0+P1+P2 迭代：统一评测/报告、索引与分块实验、CLI、FastAPI、容器编排 | 🟡 部分实现已验证 | **PARTIAL（容器 + 宿主）** | 宿主：Ruff / mypy / 纯单元 pytest（506 用例、2 skipped、`EXIT=0`）通过；4 个 DB planner/benchmark 用例通过。容器：`db-check` 9/9、三策略 ingest（150/267/161）、模型离线加载、统一检索（6 arms PASS）、索引（PARTIAL，`synthetic_scale=1`）、分块（PASS，含 rerank）、RRF 敏感性（PASS）、`test_dense_retrieval.py`（46 passed、1 skipped）、全量 pytest（664 passed、1 skipped、`EXIT=0`）；artifact 在 `reports/unified/`，日志在 `reports/container/`。**已由 ⑫/⑬ 补上**：`final_report.md`（PASS）、figures、真实 30/30 Judge、API 服务与验收接口；**仍未完成**：浏览器渲染、真实 LLM 版正式 `rag_benchmark` |
| ⑫ | LLM-as-a-Judge：Context Relevance / Faithfulness | ✅ 真实 30/30 全量 Judge 已执行 | **PASS（自动评估；人工校准未做）** | `src/schema/judge.py`、`src/evaluation/judge_prompts.py` / `judge.py` / `judge_evaluation.py`、`scripts/evaluate_judge.py`、`make judge`；测试 `tests/test_judge.py`、`tests/test_judge_evaluation.py`。宿主 Ruff / mypy / 定向 pytest（36，3 skipped）通过；容器 mock smoke 与全量 pytest（`691 passed, 1 skipped`）通过；**真实 30/30 已执行**（`reports/judge/`，`status=PASS`、0 失败、0 NOT RUN，CR mean 3.9667、Faithfulness mean 5.0000，`real_judge_calls=true`、`same_model=true`）；`final_report.md` 已重生成并输出 `Report status: PASS`；历史 3/30 smoke 归档在 `reports/judge-real-3q/`。人工校准与 inter-judge 一致性仍 `NOT RUN` |

> 原 10 步路线中的“指标报告生成”已随 ⑥-B 交付：Recall / MRR / Hit@1 / P50 / P95
> 的表格与逐查询原始结果见 `reports/hybrid_benchmark.md` / `.json`；因此不再单列
> 一个独立的 ⑦ 报告阶段。

### ⑦-A 未验证项在 ⑦-B 的状态

以下保留 ⑦-A 的全部项目，并标注 ⑦-B 实际验证到哪一步；未标注“已验证”的仍是未验证项：

- 模型下载 — **已验证**：本地缓存存在 5 个模型文件，大小与 SHA256 已实测记录
- 宿主模型加载 — **已验证**：CPU 加载成功，加载耗时 25,992.8 ms
- 内存峰值 — **已验证（仅宿主单进程）**：峰值 RSS 1,288.9 MiB；未测 Embedding + Reranker 共存
- 实际推理延迟 — **已验证（仅宿主 CPU）**：10/50 候选 × batch 1/16 已实测
- 排序质量 — **已验证（30 条英文 Query，历史 RRF Top-10 池）**：MRR 0.9500 → 0.9833
- Docker（容器内模型缓存、离线加载与运行未验证）— **仍未验证**：本次为宿主运行
- 完整 pytest（此前存在原生崩溃，未重跑）— **仍未验证**：只跑模块级测试
- 中文与混合语言 — **仍未验证**
- 规模与并发 — **仍未验证**
- 许可证 — **已验证**：模型卡为 MIT，注明可商用
- 跨机器复现性 — **仍未验证**

### ⑦-B 仍未验证事项

- 四个基线臂与 RRF 候选来自 ⑥-B 历史 artifact（已独立复核指标与 RRF 公式），不是本次重新查询数据库。
- Reranker 与 Embedding 未同进程加载，二者共存时的内存行为未测量，不声称已解决。
- 依赖版本仍与 `pyproject.toml` 的精确 pin 存在漂移；容器路径未验证。
- 延迟只覆盖 CPU 单进程顺序执行，不含 GPU、并发、跨机器复现。

### ⑧ 未验证事项

- **正式 RAG 报告仍为 mock Provider**：该报告是 PARTIAL 运行，mock 文本以 `MOCK`
  开头，不是模型输出。另有独立的 30/30 真实 RAG + Judge（见下一条），但不改变
  正式报告的 mock 口径。
- 答案正确性、Answer Relevance 未计算；Context Relevance / Faithfulness 由 Judge
  模块单独评估。已执行真实 30/30 全量运行（`reports/judge/`，PASS）：Context
  Relevance mean 3.9667、Faithfulness mean 5.0000，`same_model=true`。它覆盖全部
  查询，但仍是同模型自评；人工校准与 inter-judge 一致性校验为 **NOT RUN**。
  Judge 分数是模型判断，不是真值；证据 Recall/MRR 只是检索指标，不能当作生成准确率。
- live 链路（Dense + Sparse + RRF + Embedding + Reranker + 生成）曾在宿主同一进程
  触发原生段错误（exit 139）；容器内的 30/30 真实运行已以 `live` 模式跑通一次，
  但正式评测仍使用 `historical` 候选重放 + 实时 PostgreSQL 文本 + 实时 Reranker。
  重复运行的 live 稳定性未验证。
- 真实 Provider 的 30 条 RAG + 60 次 Judge 调用成功；超时、限流、5xx、401 与网络
  错误的重试分类仍只在单元测试中通过 `httpx.MockTransport` 验证，未注入真实故障。
- Docker/容器内的 RAG 路径已验证到 30/30 live 运行；模型缓存、离线加载、API 服务
  与验收接口已验证；浏览器渲染仍未验证。
- 宿主全量 pytest 未重跑（已知原生崩溃风险）；容器内全量套件已执行
  （`691 passed, 1 skipped`，JUnit `reports/container/pytest-full-judge.xml`）。
- 依赖固定未重新安装核验；宿主版本仍与 `pyproject.toml` 存在漂移。
- 中文与中英混合语言的检索、生成与引用未评测。
- 规模、并发、GPU、跨机器复现性未验证。
- 延迟为单进程宿主 CPU，首次查询包含模型懒加载；LLM 延迟为 mock，不代表真实网络。

> 路线编号说明：阶段 ⑧ 的实际内容是 RAG 端到端问答；**⑨ 为 README / roadmap
> 定稿**，**⑩ 为项目最终整合与 GitHub 提交准备**。FastAPI 已在 ⑪ 中写入
> `src/api`，并已在容器内启动与验证（浏览器渲染除外）；代码存在不等于可用能力。
> `docs/architecture.md` 是阶段 ① 设计稿，已加入阶段 ⑩ 的
> 编号与路径校正说明；状态以 README / roadmap 为准。

### 实现状态分类

| 分类 | 内容 | 依据 |
|---|---|---|
| **已实现且有可复核报告** | `document` 分块与 150 条导入；三种分块策略对比（`document` / `fixed_window` / `structured`，含 rerank 臂）；Dense 检索；Sparse OR/AND；候选并集与 RRF；RRF 参数敏感性；Cross-Encoder Reranker（历史 Top-10 与题面 Top-20 → Top-5）；最终 Markdown 报告与 5 张图（含 `judge_distribution.png`）；RAG 上下文、提示词、Provider、引用校验与结构化答案；LLM-as-a-Judge 30/30 全量真实运行 | `reports/dense_benchmark.md`（PASS）、`sparse_benchmark.md`（PASS）、`hybrid_benchmark.md`（PASS）、`rerank_benchmark.md`（PASS）、`rag_benchmark.md`（PARTIAL）；容器内 `reports/unified/chunking_benchmark*.json`（PASS）、`fusion_sensitivity.json`（PASS）、`rerank_20to5.json`（PASS）、`index_benchmark_scale10.json`（PASS）；`reports/judge/judge_summary.json`（PASS，30/30）；`reports/final_report.md`（PASS） |
| **已实现但只有单元/模块级证据** | Provider 超时/限流/5xx/401/网络错误映射与重试；上下文截断/丢弃；引用标签校验；空候选；重复身份/非法输入；失败分类；`fixed_window` / `structured` 分块实现与配置校验；Recall@K / Precision@K / MRR / 二元 nDCG@K；索引管理（shadow 命名、参数校验、bootstrap 索引保护）；LLM-as-a-Judge prompt/解析/失败隔离/聚合（mock 闭环 `reports/judge-smoke/`、历史 3/30 smoke `reports/judge-real-3q/` 与当前 30/30 全量 `reports/judge/` 均已跑） | `tests/test_rag_*.py`、`tests/test_reranker.py`、`tests/test_retrieval.py`、`tests/test_chunking.py`、`tests/test_chunker.py`、`tests/test_metrics.py`、`tests/test_retrieval_scope.py`、`tests/test_index_management.py`、`tests/test_hybrid_benchmark.py`、`tests/test_rerank_benchmark.py`、`tests/test_judge.py`、`tests/test_judge_evaluation.py` 等；完整 pytest 的执行结果见下方“测试与静态检查” |
| **部分验证** | RAG 端到端（mock Provider + historical RRF 重放）；LLM-as-a-Judge（真实 RAG + 真实 Judge，30/30 全量，同模型自评，无人工校准）；Reranker（宿主 CPU + 容器内实时 `rrf_rerank` 与 20→5）；Dense/HNSW（150 chunks，无规模结论）；Docker（数据库 + `app` tools 链路 + `api` 服务；浏览器渲染未验证）；检索作用域（已在含三种分块的 578 chunks 数据库上运行）；索引管理（已执行 shadow DDL；`synthetic_scale=10` 的 PASS artifact 与 `synthetic_scale=1` 的 PARTIAL artifact 都保留） | `reports/unified/*.json`、`reports/container/*.log`、`reports/final_report.md` 与本页限制 |
| **未验证** | 真实 LLM 全量评测中的人工校准与 inter-judge 一致性；RAG live 全链路重复运行稳定性；多候选 rerank / E-5B；原生崩溃根因；页面浏览器渲染（API 与验收接口已验证）；中文与混合语言；大规模性能；并发；跨机器复现；真实 LLM 版正式 `rag_benchmark` | 见 [`docs/roadmap.md`](docs/roadmap.md) 的“未完成验证” |
| **已知问题/风险** | 原生崩溃未结案；依赖声明与宿主实测存在漂移；标签非穷尽使 Precision/nDCG 只能作为补充指标；historical 重放不是新检索；mock 答案不代表生成质量；E-2/E-4 完整证据完整性 INCONCLUSIVE | 见下节与 [`docs/roadmap.md`](docs/roadmap.md) |

### 已知问题与风险

1. **原生崩溃根因未确认**：历史 `0xC0000005` / `torch_cpu.dll` 证据仍在；
   E-5A/E-5C 只覆盖单候选与空候选，不等于修复。
2. **正式 RAG 报告仍使用 mock Provider**：Context Relevance / Faithfulness 已由 Judge
   模块完成 30/30 真实全量评估（同模型自评，无人工校准或 inter-judge 一致性）；
   答案正确性与 Answer Relevance 未实现。
3. **依赖漂移**：`pyproject.toml` 是精确 pin，但宿主实测与历史报告版本不同
   （例如 torch 声明 2.14.1、宿主与历史报告均为 2.13.0）；没有覆盖传递依赖的
   锁文件。声明、宿主实测与历史记录已在“依赖、版本与复现性”表中分开列出。
   `psutil` 已声明为必需依赖，MIT `LICENSE` 已补齐。
4. **容器路径已部分验证**：数据库、三策略 ingest、统一检索、索引、分块、RRF 敏感性、
   Top-20 → Top-5 精排、报告/图与真实 30/30 RAG + Judge 均已在容器内运行；FastAPI
   `api` 服务已启动，`/healthz`、`/acceptance/api/summary`、`/api/v1/search` 与
   `/docs` 已实际请求。仍未执行的是真实 LLM 版正式 `rag_benchmark`（历史 §3.5 的
   mock + historical 报告保持原样）。宿主历史报告不能当作容器结果。
5. **完整 pytest 记录分层**：容器内全量套件已执行，加入 Judge 测试后为
   `691 passed, 1 skipped`（JUnit `reports/container/pytest-full-judge.xml`）；
   本轮 Bonus C 另单独执行 `tests/test_api.py`，23 个用例无失败。宿主全量套件
   未执行；宿主 torch/transformers 组合存在已知原生崩溃风险。
6. **中文/混合语言未评测**：Sparse 的中文探针在英文语料上 0 命中；CJK fallback
   是两字符子串匹配，不是分词。
7. **规模与并发未验证**：150 chunks 下 HNSW 没有延迟优势，自然计划是 Seq Scan；
   不能外推到大规模、GPU、并发或跨机器。
8. **指标口径限制**：标注非穷尽，Precision@K / 二元 nDCG@K 已在代码中实现并有
   单元测试，但只能作为补充指标；历史报告未包含这些字段。Recall/MRR 是检索指标，
   不能代表生成质量。
9. **E-2/E-4 完整目录完整性 INCONCLUSIVE**：只有报告列出的关键文件哈希，
   没有完整 `SHA256SUMS.txt`；不得推断为 PASS。

### 报告与证据来源

| 报告 | 状态 | 主要口径 | 关键结果（原文数值） |
|---|---|---|---|
| [`reports/dense_benchmark.md`](reports/dense_benchmark.md) / `.json` | PASS | 30 queries, K=10, 150 docs/chunks, host Python | Semantic Recall 0.944；ANN Recall@K 1.000；query P50 0.778–1.022 ms / P95 0.960–1.185 ms；自然计划为 Seq Scan |
| [`reports/sparse_benchmark.md`](reports/sparse_benchmark.md) / `.json` | PASS | 30 queries, English FTS, OR default; AND control | OR Semantic Recall 1.000，perfect 30/30，empty 0；AND Recall 0.133，perfect 4/30，empty 26；query P50/P95 2.054/2.698 ms；中文探针 0 命中 |
| [`reports/hybrid_benchmark.md`](reports/hybrid_benchmark.md) / `.json` | PASS | 30 queries, top_k/candidate_k=10, RRF k=60, 5 reps + warm-up | Recall/MRR/Hit@1：Dense 0.944/0.917/25；Sparse OR 1.000/0.940/27；Sparse AND 0.133/0.133/4；Union 1.000/0.917/25；RRF 1.000/0.950/27 |
| [`reports/rerank_benchmark.md`](reports/rerank_benchmark.md) / `.json` | PASS | 30 queries, candidate pool 10, batch 16, max length 512, identity, host CPU | Recall/MRR/Hit@1：RRF+Reranker 1.0000/0.9833/29；RRF 1.0000/0.9500/27；MRR improved 3 / equal 26 / worse 1；平均 ΔMRR +0.0333；模型加载 25,992.8 ms；峰值 RSS 1,288.9 MiB |
| [`reports/rag_benchmark.md`](reports/rag_benchmark.md) / `.json` | PARTIAL | 30 queries, mock Provider, historical RRF replay + live PostgreSQL text + live reranker | 30/30 status=ok；引用有效 30/30；证据 Recall macro/micro 1.0000/1.0000；MRR 0.9833； total P50/P95 970.9/1082.2 ms；real LLM calls False |
| [`reports/unified/rerank_20to5.md`](reports/unified/rerank_20to5.md) / `.json` | PASS | 容器；30 queries；RRF candidate pool 20 → 最终 Top-5；CPU Cross-Encoder | RRF Recall/MRR/Hit@1 0.9556/0.9500/27；RRF+Rerank 1.0000/0.9833/29；nDCG@5 0.9204 → 0.9777；P50 89.5 → 1790.9 ms |
| [`reports/unified/index_benchmark_scale10.md`](reports/unified/index_benchmark_scale10.md) / `.json` | PASS（合成） | 容器；`synthetic_scale=10`（1500 行）；6 组 HNSW + 5 组 IVFFlat | 仅反映索引行为：HNSW `m=32/efc=64` 在 `ef_search=10` 为 0.797，其余多数 1.000；IVFFlat `lists=16/probes=1` 为 0.733；不进入语义质量结论 |
| [`reports/unified/chunking_benchmark_rerank.md`](reports/unified/chunking_benchmark_rerank.md) / `.json` | PASS | 容器；3 策略 × 4 臂 × 30 queries | 见 `reports/results.md` §6；`document` 的 `rrf_rerank` MRR 最高 0.9833 |
| [`reports/final_report.md`](reports/final_report.md) | PASS | 由 3 个必需 artifact + 可选 20→5 与 Judge artifact 渲染；每个表格带路径与 SHA256 | 报告状态 PASS（Judge artifact 为 30/30 真实全量，`status=PASS`）；5 张图位于 `reports/figures/`；不重算指标 |
| [`reports/results.md`](reports/results.md) | 汇总（非新评测） | 五份历史报告 + 本轮容器 artifact 的索引与摘要 | 只引用现有报告与 artifact，不新增指标；不替代原始报告或验收记录 |
| `D:\RAG_crash_evidence\E5_REVIEW_2026-10-19_140237\E5_REVIEW_REPORT.md` | PASS（证据审核）/ 环境 FAIL（审核时点） | E-5A/E-5C 原始证据 | E-5A 1 候选 1 次 rerank/predict/forward；E-5C 空候选、模型未加载、predict/forward 0；内存 3.842/3.848 GiB < 4.5 GiB |

### 验收证据哈希

以下哈希固定本次记录所依据的报告文件；若报告被重新生成，应同步更新验收记录。

```text
reports/dense_benchmark.json   a451844a282e2a32ed6c940ffb2a188083ed3f820b25aa6cc0b9fdc99e473d3a
reports/dense_benchmark.md     2b11c458c460333d104708521b508f8a4b055dcdaa63b99e68998273890d62a2
reports/sparse_benchmark.json  133d67d250572f661f492f96647dd88dac9fc77574a2aca7deeea6f882517766
reports/sparse_benchmark.md    ae50d7fbe57dfa7b7f95038449d88aa0703be4fe90896fb1593e2d68c382ffd8
reports/hybrid_benchmark.json  63ba2a164554b5a4c8c631b7c68592be525f3440de62ea68e3ab6752ae9bbb37
reports/hybrid_benchmark.md    a5763f2ab1988635fb2cdef6101186b5d1f91ff6295c711f12a9fb81eb737e82
reports/rerank_benchmark.json  32a4fc5d3ea0a83277bebe339e3281c2ce292f6a6ab3f16f0e3ea55f9a6a2858
reports/rerank_benchmark.md    bff247fcbe07ba92f48963a106457316a9698bec8c645e1287344c84fe6f38a1
reports/rag_benchmark.json     dc715fa9291fef582a65f2f392401ed5568d29b80ca8c02666d10dcb54d589c4
reports/rag_benchmark.md       26f4240dad78657a3f708b05647d26e41e6b9ef4d07769071ceb5d6c893e641c
```

崩溃调查证据：

```text
D:\RAG_crash_evidence\E5_REVIEW_2026-10-19_140237\E5_REVIEW_REPORT.md
  7f2a79f3dc2e785a570f319b87ce7dd9dbe954d5186c5f84ca593220347d6327
D:\RAG_crash_evidence\E5_2026-10-19_1205\E5A\E5A_REPORT.md
  236ced0bcbe8eb383c4703390e9c7e1c30e9ea0525597224f1ac90d4c8688ee7
D:\RAG_crash_evidence\E5_2026-10-19_1207\E5C\E5C_REPORT.md
  5c868ffaedefb420bc5d4d7bf512c9f84a9968defbd92d0c833590c1a7519be9
```

> `rag_benchmark.*` 由 mock + historical 重放生成；重新运行会改变耗时等字段，
> 因而改变哈希。哈希只在报告内容不变时有意义。上述崩溃证据位于仓库外，
> 本阶段只引用，没有修改或重新生成。
