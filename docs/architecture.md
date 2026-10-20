# 架构设计

> 阶段 ① 的产出。本文档描述**目录结构、技术选型、模块职责、数据流、实验架构**。
> 业务代码在后续阶段实现；每个模块文件顶部都写明了它的单一职责与实现阶段。
>
> **阶段 ⑩ 校正说明（2026-10-19）**：本文保留阶段 ① 的设计内容，但以下设计目标
> 与当前实现不同，实际状态以 `README.md` 和 `docs/roadmap.md` 为准：
> `reports/results.md`、`reports/figures/*.png`、`make report`、`make pipeline`、
> `cli.report` / `cli.evaluate` 在 2026-10-20 之前均未实现；本轮已写入
> `cli.report` / `cli.evaluate` / `make report` / `make pipeline`。这些入口其后已在
> 容器内实际执行：`scripts/report.py` 生成 `reports/final_report.md` 与
> `reports/figures/*.png`（5 张，`Report status: PASS`）。五份历史报告
> `reports/{dense,sparse,hybrid,rerank,rag}_benchmark.md` / `.json` 仍由
> `scripts/*_benchmark.py` 生成并保持原样。阶段编号已统一为 **⑨ README / roadmap 定稿**、
> **⑩ 项目最终整合与 GitHub 提交准备**、**⑪ P0+P1+P2 迭代**；FastAPI 已写入
> `src/api`，`api` 服务已在容器内启动并通过 `/healthz` 与
> `/acceptance/api/summary` 请求。分块策略已由设计期的
> `contextual` 调整为 `structured`（`contextual` 退役，只保留迁移提示）；依赖以
> `pyproject.toml` 的声明为准，宿主实测版本与该声明存在漂移（见 README
> “依赖、版本与复现性”），本文第 8 节的旧版本陈述不构成安装验证。
>
> **2026-10-20 增补**：P0+P1+P2 迭代已写入统一评测层（`evaluation/benchmark.py`、
> `artifacts.py`）、报告层（`evaluation/report.py`、`scripts/report.py`）、arm
> 注册表（`retrieval/registry.py`）、索引与分块实验、完整 CLI 与只读 FastAPI
> （`src/api`）。这些模块其后已在容器内执行并留下 artifact；本文正文仍是设计目标，
> 实际执行状态以 README 的“本轮迭代交付状态”和 `docs/roadmap.md` 第 10 节为准。

---

## 1. 约束与优先级

题目数据规模：**150 篇文档 / 30 条评测 Query**。

据此确定优先级：**正确性 > 可复现性 > 可评测性 > 工程结构 > 性能 > 扩展性**。

由此推出一条硬约束：**不引入任何只在大规模下才有意义的组件**。不用 Elasticsearch、不用向量数据库独立服务、不用消息队列、不做微服务。唯一的基础设施是 PostgreSQL（题目指定）+ pgvector（题目指定），而这两者本来就要跑。

架构上真正为"未来"预留的只有一处：`docs/roadmap.md` 里讨论 1000 万条规模时该换什么。

---

## 2. 数据集的三个结构性事实

这三个事实决定了后面几乎所有的设计取舍，因此先讲。

**① 数据集是模板复制的。** 150 篇文档只有 **78 个 Policy Code**，平均 1.9 篇/族。同一条规则被复制成 4 个卡种版本，正文除了卡名和几个数字几乎逐字相同。`BT-001-APEX-2024` 同时存在于 Platinum / Preferred Travel / Everyday Freedom / Business Select 四篇文档里。

**② 精确数字是主要干扰项。** `$800 / $600 / $1,000`、`60 / 45 / 30 天`、`15 / 12 / 18 / 9 个账单周期` 密集分布。稠密向量对这类微小数值差异最不敏感。

**③ 元数据是强信号。** `card_tier` 是显式字段，30 条 Query 里 20 条点名了卡种。

**但有一个反例必须先讲清楚**：`Q-029` 问"Apex Preferred Travel 有没有 Global Entry 报销"，唯一的相关文档 `DOC-0060` 属于 **Platinum**（答案是否定的）。**任何"先按 card_tier 硬过滤"的策略都会在这一条上直接失败。** 因此本设计选择**软加权而非硬过滤**——把 card_tier 以最高权重写进 tsvector（见 §7），让排序去解决，而不是让过滤去截断。

---

## 3. 最终目录结构

```
.
├── data/
│   ├── kb_documents.json            # 150 篇，随仓库提交
│   └── eval_queries.json            # 30 条，随仓库提交
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # pydantic-settings 集中配置
│   ├── cli.py                       # 唯一入口
│   ├── db.py                        # 连接管理
│   │
│   ├── schema/                      # 数据形状（Pydantic），无行为
│   │   ├── document.py              #   Document, Chunk
│   │   └── retrieval.py             #   SearchHit + 各类 Config
│   │
│   ├── ingestion/
│   │   ├── loader.py                # JSON -> Document
│   │   ├── chunker.py               # Document -> Chunk
│   │   ├── embedder.py              # Text -> Vector
│   │   └── indexer.py               # Chunk + Vector -> PostgreSQL
│   │
│   ├── retrieval/
│   │   ├── base.py                  # Retriever Protocol
│   │   ├── dense.py                 # pgvector ANN
│   │   ├── sparse.py                # PostgreSQL FTS
│   │   ├── fusion.py                # RRF / 加权
│   │   ├── reranker.py              # Cross-Encoder
│   │   ├── pipeline.py              # 组合成检索策略
│   │   └── registry.py              # 评测 arm 注册表（名称 -> 构造器 + 参数）
│   │
│   ├── api/                         # 只读 HTTP 传输层（2026-10-20 新增）
│   │   ├── app.py                   #   /healthz、/readyz、/answer、/api/v1/search
│   │   ├── acceptance.py            #   /acceptance/* 只读 artifact 投影
│   │   └── schema.py                #   请求/响应投影（不含 prompt / raw payload）
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # 纯函数指标
│   │   ├── benchmark.py             # 统一编排与 schema
│   │   ├── artifacts.py             # 覆盖保护、哈希
│   │   ├── report.py                # artifact -> Markdown + 图（不重算指标）
│   │   ├── index_benchmark.py       # HNSW / IVFFlat 扫描（2026-10-20 新增）
│   │   ├── chunking_benchmark.py    # 分块策略对比（2026-10-20 新增）
│   │   └── fusion_benchmark.py      # RRF 敏感性（2026-10-20 新增）
│   │
│   └── utils/
│       ├── logger.py
│       └── timing.py                # 只采集，不算分位数
│
├── web/acceptance/                  # 无构建步骤的静态验收页（2026-10-20 新增）
│   ├── index.html                   #   tab 结构、CSP meta、只读声明
│   ├── styles.css                   #   同源样式，支持深色与窄屏
│   └── app.js                       #   fetch 两个 GET API，只用 textContent
│
├── tests/
│   ├── conftest.py
│   ├── test_chunking.py
│   ├── test_fusion.py
│   ├── test_metrics.py
│   └── test_retrieval.py
│
├── reports/
│   ├── dense_benchmark.md / .json   # 实际交付：阶段 ④
│   ├── sparse_benchmark.md / .json  # 实际交付：阶段 ⑤
│   ├── hybrid_benchmark.md / .json  # 实际交付：阶段 ⑥
│   ├── rerank_benchmark.md / .json  # 实际交付：阶段 ⑦
│   ├── rag_benchmark.md / .json     # 实际交付：阶段 ⑧
│   ├── unified/                     # 统一评测 artifact（目标路径；尚未生成）
│   ├── container/                   # 容器验证日志（目标路径；尚未生成）
│   ├── figures/                     # 报告图（目标路径；当前为空）
│   └── final_report.md              # 汇总报告（目标产物；尚未生成）
│
├── docs/
│   ├── architecture.md              # 本文档（阶段 ① 设计稿）
│   └── roadmap.md                   # 阶段 ⑨ 产出（当前状态与路线）
│
├── docker/
│   └── init.sql                     # 扩展 + 表 + GIN 索引
│
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

### 相对"标准答案"的四处改动及理由

| 改动 | 理由 |
|---|---|
| `src/models/` → **`src/schema/`** | `models/` 在 ML 项目里默认指**模型权重**，而这里放的是 Pydantic 数据模型。改名消除歧义，零成本。 |
| `src/main.py` → **`src/cli.py`** | `main.py` 不说明它是什么。这就是个 CLI 入口，名字应当直说。 |
| ~~删除 `scripts/`~~ → **恢复入口脚本** | 阶段 ① 删掉了 `scripts/`，理由是它只能是对 `cli.py` 的薄包装。阶段 ③ 先恢复了 `scripts/ingest.py`（6 行转发，不含逻辑）。阶段 ④–⑧ 实际使用 `scripts/benchmark_*.py` 与 `scripts/ask_rag.py` 作为评测/问答入口；它们直接编排 benchmark 模块，不是 `cli.py` 的转发。`Makefile` 的 `ingest` 仍走 `python -m src.cli ingest`。 |
| **新增 `src/db.py`** | `indexer`（写）与 `dense`/`sparse`（读）都要连接。放在任一侧都会造成不对称依赖，独立出来最干净。 |
| **新增 `src/retrieval/base.py` 与 `registry.py`** | 这是 5 个实验能"不写死"的关键，见 §6。 |
| **不加 `notebooks/`** | 实际报告由 `scripts/*_benchmark.py` 生成 Markdown/JSON；`make report` 已写入、尚未执行验证。Notebook 依赖手工执行顺序，与"可复现"优先级冲突。 |

---

## 4. 技术选型表

| 技术领域 | 最终方案 | 备选方案 | 选择原因 |
|---|---|---|---|
| Language | **Python 3.12**（容器内）<br>本地 ≥3.12 | 3.12 / 3.13 / 3.14 | 项目声明、Ruff、Mypy 与镜像统一为 3.12；3.12 的 wheel 覆盖最完整。**本机 3.14.3 也能跑**（设计时核实 torch 2.14.1 提供 cp314 wheel；实际评测环境的版本存在漂移，见 `docs/roadmap.md` §4.5），但容器固定 3.12 以消除环境漂移。 |
| Database | **PostgreSQL 16** (`pgvector/pgvector:pg16`) | SQLite / 独立向量库 | 题目指定；且 FTS 与向量检索同库，省掉一整套同步逻辑。 |
| Vector | **pgvector** | FAISS / Qdrant | 题目指定。 |
| Vector Index | **HNSW**（余弦） | IVFFlat | IVFFlat 需要足够数据量才能训练出有意义的聚类；150 行下几乎没有可训练的分布，且需要预先指定 `lists`。HNSW 无需训练、增量友好、召回更稳。**IVFFlat 对照实验的扫描代码已写入（`evaluation/index_benchmark.py`），尚未执行**。 |
| Embedding | **`BAAI/bge-large-en-v1.5`**（1024 维，本地） | `text-embedding-3-small`（API）<br>`all-MiniLM-L6-v2`（384 维） | 见下方说明。 |
| Sparse | **PostgreSQL FTS**（`tsvector` + GIN + `ts_rank_cd`） | Elasticsearch / BM25 库 | 见下方说明。 |
| Fusion | **RRF**（k=60） | 加权得分归一化 | 见下方说明。 |
| Reranker | **`BAAI/bge-reranker-base`**（本地 Cross-Encoder） | `cross-encoder/ms-marco-MiniLM-L-6-v2`<br>Cohere Rerank（API） | 见下方说明。 |
| Evaluation | **纯 Python + numpy** | ranx / pytrec_eval | 4 个指标 + 3 个延迟分位数，自己实现约 60 行，且**可被单元测试逐一验证**。引入评测库反而增加依赖和黑盒。 |
| Container | **Docker Compose**（db + app） | 裸机安装 | 题目要求"一键运行"；且见 §8 环境决策。 |
| API | **FastAPI**（`src/api`，只读） | — | 复用 `build_pipeline()`，不复制检索逻辑；`/answer` 只投影 answer / citations / evidence，`/api/v1/search` 只返回排序 chunk 并复用同一 retriever（每个进程一组模型）；响应不含 prompt 与 provider 原始输出；模型懒加载，`/readyz` 只检查数据库。Swagger 由 FastAPI 生成（`/docs`、`/openapi.json`）。`[api]` extra 与 compose `api` profile 已配置。`api` 服务已在容器内启动；`/healthz`、`/acceptance/api/summary`、`/api/v1/search` 与 `/docs` 已实际请求，`/readyz`、`/answer` 与浏览器渲染尚未验证。 |

### Embedding：为什么选本地 bge-large，而不是题目示例里的 OpenAI

**决定性因素是评审者的复现成本。** `text-embedding-3-small` 需要 API Key——评审者手上大概率没有，于是"一条命令跑通"直接变成不可能。本地模型把这一整类问题消掉。

在题目给的两个本地选项之间选 `bge-large-en-v1.5`：语料是纯英文，bge 系列的英文检索质量明显高于 `all-MiniLM-L6-v2`（384 维 vs 1024 维），而 150 篇文档的编码成本本来就只有几秒。

代价如实说明：镜像因此增加约 1.3 GB。`all-MiniLM-L6-v2` 作为配置项保留（`APEX_EMBEDDING_MODEL`），供资源受限时切换。原计划在阶段 ④ 把两者都跑一遍作为额外消融，**但该消融未执行**；现有报告只使用 `bge-large-en-v1.5`。

> ⚠️ 实现注意：bge-en-v1.5 系列要求在 **Query 侧**加指令前缀
> `"Represent this sentence for searching relevant passages: "`，
> **文档侧不加**。这一条已经写进 `Settings.query_instruction`，`embedder.py` 必须区分 `embed_documents()` 与 `embed_query()`。

### Sparse：为什么是 PostgreSQL FTS 而不是 Elasticsearch

150 行的语料，引入 Elasticsearch 意味着多一个 JVM 服务、多一套索引同步、多一份 docker-compose 复杂度——**而收益是零**，因为在 150 行上 Postgres 的 GIN 索引响应在亚毫秒级，与 ES 无法区分。

更关键的是它**恰好补上了稠密向量的短板**：`to_tsvector('english', '$800 per claim')` 会保留 `800` 这个 token，`'60 days'` 会保留 `60`。也就是说，语料里那些`$800/$600/$1,000`、`60/45/30` 的精确数值差异，正是词法检索最擅长、稠密向量最不擅长的部分。

已知局限（如实记录）：FTS 会做词干还原并丢弃停用词，纯符号形式（如 `3%`）的切分不理想。这类残差交给重排序阶段兜底。

### Fusion：RRF 与加权归一化并存，而非二选一

> **实现状态（阶段 ⑥-B）**：`RRFFusion` 与候选并集基线 `interleaved_union` 已实现并有单元测试；
> 本节讨论的加权归一化 **尚未实现**。阶段 ⑥-B 的实测参数是 `candidate_k=10`、`top_k=10`、
> `RRF k=60`（不是下文的 50/10 生产默认值），结果见 `reports/hybrid_benchmark.md`。

不是因为"RRF 更好"，而是因为**题目要求对比**，而且这两者回答的是不同问题：

- **加权归一化**需要把稠密（余弦相似度，0~1）和稀疏（`ts_rank_cd`，无界）的分数放到同一尺度。这本身就是一次有信息量的实验：两种分数量纲根本不可比，min-max 归一化后加权，`alpha` 该怎么取。
- **RRF** 只用排名不用分数，天然免疫量纲问题，`k=60` 是 Cormack 等 2009 年的原始默认值。

于是：**实验 2 用加权，实验 3 用 RRF**，直接对应题目要求的两个阶段，且构成一组干净的对照。

### Reranker：本地 Cross-Encoder

> **实现状态（阶段 ⑦-B）**：`src/retrieval/reranker.py` 已实现为独立的 Cross-Encoder
> 重排阶段，并在宿主 CPU 上完成 30 条 Query 的五臂评测；没有静默 fallback。
> 容器内运行与跨机器复现仍未验证。

同样是复现性优先（Cohere 需要 Key）。`bge-reranker-base` 英文表现好、CPU 可跑，与 `bge-large-en-v1.5` 同源，语义空间一致。

**必须如实报告的一点**：重排序是整个链路里最慢的一环。在 CPU 上对 50 个候选打分约需 1–3 秒，而向量检索本身只要几毫秒。因此**延迟必须分阶段报告**（embed / dense / sparse / fusion / rerank / total），不能只报一个总数——否则"重排序把 P95 拉高了两个数量级"这个关键结论会被藏起来。

---

## 5. 模块职责

每个模块**只有一个职责**，且不越界。

### 配置与基础设施

| 模块 | 做什么 | 输入 | 输出 | 被谁调用 | 阶段 |
|---|---|---|---|---|---|
| `config.py` | 集中读取环境变量 / `.env` | 环境 | `Settings` 单例 | 所有模块 | ①✅ |
| `db.py` | 连接工厂 | `Settings.dsn` | 连接 / 游标上下文 | `indexer`, `dense`, `sparse` | ②✅ |

### 数据形状

| 模块 | 做什么 | 输入 | 输出 | 被谁调用 | 阶段 |
|---|---|---|---|---|---|
| `schema/document.py` | 定义 `Document`, `Chunk` | — | Pydantic 模型 | `loader`, `chunker`, `indexer` | ①✅ |
| `schema/retrieval.py` | 定义 `SearchHit` 与各类 `*Config` | — | Pydantic 模型 | `retrieval/*`, `benchmark` | ①✅ |

### 写入链路

| 模块 | 做什么 | 输入 | 输出 | 被谁调用 | 阶段 |
|---|---|---|---|---|---|
| `ingestion/loader.py` | 读 JSON，校验，抽 `policy_code` | `data/kb_documents.json` | `list[Document]` | `cli.ingest` | ③✅ |
| `ingestion/chunker.py` | 按策略切块 | `Document` | `list[Chunk]` | `cli.ingest` | ③✅（仅 `document`；另两种归 ⑤） |
| `ingestion/embedder.py` | 文本 → 向量；区分 query/document 前缀 | `list[str]` | `list[list[float]]` | `cli.ingest`, `pipeline` | ③✅ |
| `ingestion/indexer.py` | 批量写入；HNSW / IVFFlat shadow 索引管理（未执行 DDL） | `list[Chunk]` + 向量 | 数据库行 / `IndexStats` | `cli.ingest`（写入）；`cli.index build/list/drop` | ③✅ 写入 / 索引管理已实现、未运行 |

> 关于 `embedder` 的返回类型：阶段 ① 这里写的是 `np.ndarray`，实际实现返回
> `list[list[float]]`。向量越过模块边界后就只是一串浮点数，让 numpy 出现在接口上
> 会把数组语义泄漏到 `indexer` 和阶段 ④ 的 `dense.py`，而它们都不需要。
> `pgvector` 的适配器对 list 有专门的快速路径，所以这样写也更快。

### 检索链路

| 模块 | 做什么 | 输入 | 输出 | 被谁调用 | 阶段 |
|---|---|---|---|---|---|
| `retrieval/base.py` | 定义 `Retriever` Protocol | — | 接口 | 所有 retriever | ⑤ |
| `retrieval/dense.py` | pgvector ANN 检索 | query 向量 | `list[SearchHit]` | `pipeline` | ⑤ |
| `retrieval/sparse.py` | FTS 检索 | query 文本 | `list[SearchHit]` | `pipeline` | ⑤ |
| `retrieval/fusion.py` | 合并多个有序列表（RRF / 候选并集） | `list[list[SearchHit]]` | `list[SearchHit]` | `pipeline` | ⑥-B 已实现 |
| `retrieval/reranker.py` | Cross-Encoder 重排 | query + 候选 | `list[SearchHit]` | `pipeline` | ⑦-B 已实现 |
| `retrieval/pipeline.py` | 把各阶段组合成策略 | `RetrievalConfig` | `Retriever` 实例 | `registry` | ⑥-B（HybridRetriever） |
| `retrieval/registry.py` | 按名字构造 retriever | 实验名 | `Retriever` | `benchmark` | ⑤ |

**明确的边界**：`dense.py` 只做稠密检索——它不负责 embedding、不负责融合、不负责重排。同理 `fusion.py` 只接收若干有序列表、只返回一个有序列表，它不知道这些列表从哪来。

### 评测链路

> 下表保留阶段 ① 的设计职责。历史报告的实际入口是 `scripts/benchmark_*.py`；
> 2026-10-20 新增的统一入口是 `scripts/benchmark_{retrieval,index,chunking,fusion}.py`
> 与 `scripts/benchmark_all.py`，由 `cli.benchmark` 转发。`cli.evaluate` 读取 artifact、
> `cli.report` 渲染报告，二者都**不重算指标**；它们与新增脚本均未执行验证。

| 模块 | 做什么 | 输入 | 输出 | 被谁调用 | 阶段 |
|---|---|---|---|---|---|
| `evaluation/metrics.py` | 纯函数：`recall_at_k`, `precision_at_k`, `mrr`, `ndcg_at_k`, `percentile` | 相关 ID 列表 + 命中列表 | `float` | `benchmark` | ④–⑧ |
| `evaluation/benchmark.py` | 遍历「实验 × Query」，记录指标与分阶段延迟 | `list[RetrievalConfig]` + `eval_queries.json` | `BenchmarkResult` | 设计目标；当前由具体 benchmark 模块编排 | ④–⑧ |
| `evaluation/report.py` | 渲染表格与图表 | `BenchmarkResult` | `reports/*_benchmark.md` / `.json` | `scripts/*_benchmark.py` | ④–⑧ |

**关于 `utils/timing.py` 与 `metrics.py` 的边界**（易踩的重复）：`timing.py` **只采集**耗时（`timer()` / `StageLatencies`），**不做统计**；把毫秒列表变成 P50/P95/P99 是 `metrics.percentile()` 的职责。两者不重叠。

### RAG 问答链路（阶段 ⑧）

阶段 ⑧ 不修改任何检索或重排模块；生成层是独立包，只消费 `Retriever` 返回的
`SearchHit`，并复用其中的真实 chunk 文本与稳定 ID。

| 模块 | 做什么 | 输入 | 输出 | 被谁调用 | 阶段 |
|---|---|---|---|---|---|
| `schema/rag.py` | 定义 `RAGAnswer`、`Evidence`、`Citation`、`LLMConfig`、`RAGStatus` | — | Pydantic 模型 | `rag/*`, `evaluation/rag_benchmark` | ⑧ |
| `rag/context.py` | 证据去重、`E1..En` 编号、单块/总预算截断、SHA256 | `list[SearchHit]` | `ContextBundle` | `RAGPipeline` | ⑧ |
| `rag/prompt.py` | 证据约束提示词与版本/哈希 | `ContextBundle` + query | `messages`, `prompt_sha256` | `RAGPipeline` | ⑧ |
| `rag/provider.py` | OpenAI 兼容 HTTP Provider、错误分类、有界重试、mock | messages | `LLMResponse` | `RAGPipeline` | ⑧ |
| `rag/pipeline.py` | 校验 → 检索 → 上下文 → 生成 → 引用校验 → 结构化答案 | query | `RAGAnswer` | CLI / benchmark / `ask_rag` | ⑧ |
| `rag/factory.py` | 由 `Settings` 构造 retriever / provider / pipeline | `Settings` | 已装配对象 | CLI / benchmark | ⑧ |
| `rag/historical.py` | 重放 ⑥-B RRF 候选身份，连接实时 DB 文本 | query | `list[SearchHit]` | `RAGPipeline` | ⑧（评测替代路径） |
| `evaluation/rag_benchmark.py` | 30 条 Query 的端到端记录、指标、Markdown/JSON | pipeline + queries | artifact | `scripts/benchmark_rag.py` | ⑧ |

**边界**：`context.py` 只组装文本，不调用模型；`provider.py` 只负责一次（或有限次）
HTTP 交互，不读数据库、不知道检索存在；`pipeline.py` 是唯一把两者串起来的地方。
生成模型、Embedding 模型、Reranker 是三个独立组件，不能互相替代。

### LLM-as-a-Judge（2026-10-20 新增）

Judge 只消费 RAG 已经产出的结构化结果。它不重新检索、不修改答案、不读取
`gold_response`，因此同一份 RAG artifact 可以被重复评分，且评分成本与生成成本解耦。

| 模块 | 做什么 | 输入 | 输出 | 被谁调用 | 阶段 |
|---|---|---|---|---|---|
| `schema/judge.py` | Context Relevance / Faithfulness 结果、claim、错误与 artifact 契约 | — | Pydantic 模型 | `evaluation/judge*` | 新增 |
| `evaluation/judge_prompts.py` | 两套 rubric、untrusted-data 分隔符、版本号与 prompt SHA256 | query + `JudgeContext` (+ answer) | messages、hash | `evaluation/judge.py` | 新增 |
| `evaluation/judge.py` | 调用 Provider、解析 JSON、校验分数/标签/claim；失败返回 `score=None` 与稳定 status | messages | `ContextRelevanceResult` / `FaithfulnessResult` | `judge_evaluation` | 新增 |
| `evaluation/judge_evaluation.py` | 批量评分、失败隔离、均值/中位数/分布、状态判定、JSONL/JSON/Markdown | RAG artifact + Judge | judge artifact | `scripts/evaluate_judge.py` | 新增 |
| `scripts/evaluate_judge.py` | 参数解析、真实调用的 key/网络门禁、写盘保护 | CLI | `reports/judge/*` | `make judge` / `src.cli` | 新增 |

**数据流**：`RAGPipeline` → `run_rag_benchmark()`（或 `--rag-artifact`）→
`per_query[].evidence`（`E1..En`）与 `answer` → 两个 Judge prompt → 逐条结果 →
JSONL + summary + Markdown → `src/evaluation/report.py` 的 optional `judge` 章节与
`judge_distribution.png`。

**边界**：`judge.py` 不知道 RAG artifact 或文件布局；`judge_evaluation.py` 不构造
prompt、不解析模型文本；`report.py` 只读已落盘的 summary，不重新评分。失败请求
（超时、限流、非法 JSON、未知标签）只影响该条 Query，绝不折算成 0 分；Mock Judge
固定 `is_real=False`，其运行只能是 `PARTIAL`。

### HTTP 传输层（2026-10-20 新增）

| 模块 | 做什么 | 输入 | 输出 | 被谁调用 | 阶段 |
|---|---|---|---|---|---|
| `api/schema.py` | 请求校验（1–2000 字符，`extra="forbid"`）与响应投影；含 `SearchRequest` / `SearchResult` / `SearchResponse` | JSON body | Pydantic 模型 | `api/app.py` | ⑪ |
| `api/app.py` | 进程内缓存一个 `RAGPipeline`；状态到 HTTP 的映射；`/api/v1/search` 复用 pipeline 的 retriever | request | `JSONResponse` | uvicorn | ⑪ |
| `api/acceptance.py` | 固定白名单加载 artifact，投影为只读 summary 与逐 Query Judge 页 | 已落盘 JSON | `JSONResponse` / `FileResponse` | `api/app.py` | ⑬ |
| `web/acceptance/*` | 原生 HTML/CSS/JS 看板；fetch 两个 GET API，`textContent` 渲染 | summary JSON | 页面 | 浏览器 | ⑬ |

**边界**：API 只是传输层。它不构造检索参数（Top-K 是服务端配置，不接受客户端输入），
不重试 Provider（`RAGPipeline` 已有有界重试），不返回 `context_text` 或
`raw_response`。`/api/v1/search` 是例外的最小参数面：调用方只能给 `query` 与可选
`top_k`，且 `top_k` 不得超过服务端候选池；该端点不调用 LLM。未列出的 `RAGStatus`
映射为 500；`/readyz` 失败返回 503 但不会因为模型未加载而报错——模型在首次
`/answer` 或 `/api/v1/search` 时懒加载。

**验收数据流**：五份固定 artifact → `evaluation/report.load_artifacts()` /
`report_status()` → `api/acceptance.build_summary()`（只选字段，不重算）→
`GET /acceptance/api/summary`；`judge_summary.per_query[]` → 逐条投影（剔除
`raw_response`、usage、answer/context/prompt 哈希、prompt 文本、endpoint；
artifact 文件 SHA256 与数据集哈希保留用于追溯）→
`GET /acceptance/api/judge/queries`。路由只注册 GET，不查数据库、不加载模型；
artifact 存在但损坏时 fail-closed 返回 503，不展示部分数字。

---

## 6. 核心数据流

### 写入链路（实际由 `make ingest` 触发；`make pipeline` 已实现但未执行）

```
data/kb_documents.json
        │
        ▼  loader.load_documents()
   list[Document]                     ← 校验字段，抽取 policy_code
        │
        ▼  chunker.chunk(documents, ChunkingConfig)
   list[Chunk]                        ← 每个 Chunk 携带 doc_id
        │
        ▼  embedder.embed_documents(chunk.embedded_text)
   list[list[float]] (N × 1024)
        │
        ▼  indexer.upsert_corpus(documents, chunks, vectors)
   PostgreSQL: documents + chunks     ← 单事务；chunks.tsv 由 GENERATED 列自动维护
        │
        ▼  indexer.build_hnsw_index(m=..., ef_construction=...)   # 显式调用；ingest 不会自动建
   shadow HNSW 索引（chunks_embedding_hnsw_vNNN）
```

### 查询链路

```
query (str)
   │
   ▼  embedder.embed_query()          ← 加 bge 指令前缀
   │
   ├──────────────────┬───────────────────┐
   ▼                  ▼                   │
dense.search()    sparse.search()         │
(<=> 余弦, HNSW)  (tsquery + ts_rank_cd)  │
top 50            top 50                  │
   │                  │                   │
   └────────┬─────────┘                   │
            ▼  fusion.fuse(method)
       候选 top 50                        │
            ▼  reranker.rerank()          │
       最终 top 10  ───────────────────────┘
            ▼
      list[SearchHit]（doc_id 级别去重后）
```

### RAG 问答链路（阶段 ⑧）

```
Retriever 最终 Top-K SearchHit（稳定 chunk_id / doc_id，真实 DB content）
        │
        ▼  build_context()
   ContextBundle：E1..En、单块/总字符预算、truncated/dropped、context_sha256
        │
        ▼  build_messages()                  ← prompt rag-answer-v1
   system + user messages（只包含 E<n> 证据，要求 JSON 输出）
        │
        ▼  LLMProvider.complete()
   LLMResponse（文本 / usage，若 Provider 返回）
        │
        ▼  JSON 解析 + 引用校验
   citations：只接受实际提供的 E 标签；invalid_citations 单独记录
        │
        ▼  RAGAnswer
   status | answer | citations | evidence | retrieval trace |
   prompt/context 哈希 | usage | timings | error
```

短路规则：证据为空时不调用 Provider，直接返回确定性的
`insufficient_evidence`。模型自身的“证据不足”回答同样保留为
`insufficient_evidence`，不强迫生成。缺 Key 在检索前失败；检索、重排、
上下文与 Provider 的错误各有独立状态，不能伪装成 `ok`。

### 评测链路

```
data/eval_queries.json
        │
        ▼  benchmark.run(configs, queries)
   ┌──────────────────────────────────────────┐
   │ 对每个实验配置 × 每条 Query：              │
   │   1. pipeline.retrieve(query, top_k)      │
   │   2. StageLatencies 记录各阶段耗时          │
   │   3. 命中 chunk → 折叠为 doc_id            │
   │   4. metrics.* 对比 relevant_doc_ids      │
   └──────────────────────────────────────────┘
        │
        ▼  BenchmarkResult
        ▼  benchmark 模块写报告
   reports/*_benchmark.md  +  reports/*_benchmark.json
```

> ⚠️ **第 3 步是必须的**：金牌标注是 **doc 级**的（`relevant_doc_ids`），而检索返回的是 **chunk 级**的。若分块策略把一篇文档切成多块，必须先按 `doc_id` 折叠（取最高分、保留最优排名）再算指标，否则同一篇文档会重复计数，Recall 会被系统性高估。这就是 `SearchHit` 必须携带 `doc_id` 的原因。

---

## 7. 数据库 Schema 要点

完整 DDL 见 `docker/init.sql`。五处设计值得说明：

**① 加权 tsvector 是对"卡种混淆"的正面回答。**

```sql
tsv TSVECTOR GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(title, '')),     'A') ||
    setweight(to_tsvector('english', coalesce(card_tier, '')), 'A') ||
    setweight(to_tsvector('english', coalesce(content, '')),   'B')
) STORED
```

`title` 和 `card_tier` 给最高权重 A，正文给 B。查询 "Apex Platinum" 时，标题里写着 Platinum 的文档会显著靠前。这是**用排序解决歧义，而不是用过滤**——从而不会在 `Q-029` 那种"答案是另一个卡种"的场景上翻车。

**② `chunks` 反范式化了 `title`。** GENERATED 列不能跨表引用，而加权 tsvector 需要 title。150 行的冗余成本为零。

**③ HNSW 索引在 `init.sql` 里建一个引导索引，最终仍由 `indexer.py` 拥有。**

> 此处推翻了阶段 ① 的原设计（原本 HNSW 完全不在 `init.sql` 中）。理由是阶段 ② 要求"`docker compose up` 之后 schema 必须自证完整"——一个建不出索引的 schema 是不完整的，而且 HNSW 参数（`m = 16`、`ef_construction = 64`）与 `DenseConfig` 的默认值本就一致。
>
> 两个索引不会打架：`indexer.py` 现在提供显式的 `build_hnsw_index()` / `build_ivfflat_index()`，默认创建带版本号的 shadow 索引（`chunks_embedding_hnsw_vNNN`），不删除也不替换 `chunks_embedding_hnsw_idx`；`drop_index()` 只有在显式调用并传入 `allow_production=True` 时才能删除引导索引。`ingest` 不会隐式重建索引。`init.sql` 只在数据目录为空时执行一次，此后不再介入。
>
> 参数不写死在 SQL 里：`docker/init.sh` 用 psql 的 `:variable` 把 `APEX_HNSW_M` / `APEX_HNSW_EF_CONSTRUCTION` 注入进去。改这两个值只影响新初始化的数据目录；已有数据库应使用 shadow 索引构建，而不是 `make reset`。
>
> 上述索引函数只有纯单元测试覆盖命名与参数校验；**尚未执行任何建索引或删索引 DDL**。

**④ `chunks.embedding` 是 `NOT NULL`。** 阶段 ② 的指令建议允许 `NULL`，这里没有采纳：pgvector 会把 `NULL` 向量的行**静默排除**在 HNSW 索引之外，于是这样的行对顺序扫描可见、对 ANN 检索不可见——一个没有任何测试能发现的 Recall@K 偏差。把这个状态变得不可表达，比事后检测它便宜得多。

**⑤ `documents` 表没有 `embedding` 列，这是刻意的。**

阶段 ③ 的指令写的是"写入 `documents.embedding` 字段"，但那张表确实没有这一列，向量在 `chunks` 上。这不是遗漏，而是阶段 ① 的双表设计：分块策略是**实验变量**（实验 5 要对比 document / fixed_window / structured 三种；设计期写的 `contextual` 已退役），把向量挂在 `documents` 上会让一篇文档只能有一个向量，实验 5 直接无法进行。

代价是阶段 ③ 的验收 SQL 需要换个写法。二者等价：

```sql
-- 阶段 ③ 指令里的写法（本 schema 下不存在）
SELECT count(*) FROM documents WHERE embedding IS NOT NULL;
-- 本 schema 下的等价写法：document 策略下 1 篇文档恰好 1 个 chunk
SELECT count(*) FROM chunks WHERE chunk_strategy = 'document' AND embedding IS NOT NULL;
```

如果评审标准要求 `documents.embedding` 这一列必须存在，改动是局部的：加一列 + 把 150 行从 `chunks` 复制过去，向量已经在库里、不需要重新编码。

---

## 8. 环境决策：容器是唯一契约

**已核实**：本机 Python 3.14.3。宿主实测 torch 为 `2.13.0+cpu`；`pyproject.toml`
声明 `2.14.1`，该声明版本尚未在本机安装与回归，不能据此写成"已核实可运行"。
**除 `clean` 外，Makefile 目标仍统一走 `docker compose run`**；`clean` 只清理宿主
Python 缓存，避免为删除缓存而启动数据库。

理由是消除"在我机器上能跑"这一类问题：评审者只需有 Docker，宿主 Python 版本、系统库、CUDA 有没有，全部无关。宿主 `python -m venv` 仅作为开发便利，不是交付契约。

另外注意：`Dockerfile` 把 torch 从 PyTorch 的 CPU 专用源单独安装——Linux 上的默认 wheel 会连带拉入约 2.5 GB 的 CUDA 运行时，而这个项目永远不会用到 GPU。

---

## 9. 实验架构

### 唯一的抽象：`Retriever`

```python
class Retriever(Protocol):
    name: str
    def retrieve(self, query: str, top_k: int) -> list[SearchHit]: ...
```

> **实现状态（2026-10-20 更新）**：实验 1（Dense）、实验 3（RRF）、候选并集
> 基线与实验 4（Reranker）已实测；实验 5（分块策略对比）已在容器内完成：三种策略
> 分别入库（`document=150`、`fixed_window=267`、`structured=161`，合计 578 chunks），
> 30 条 Query × 3 策略 × 4 arms（含 `rrf_rerank`）全部 PASS；结果见
> `reports/unified/chunking_benchmark.json` 与 `chunking_benchmark_rerank.json`。
> 实验 2（加权融合）仍未实现。
> 新增的 `retrieval/registry.py` 把 6 个评测臂（dense / sparse_or / sparse_and /
> union / rrf / rrf_rerank）收敛为一个注册表，`evaluation/benchmark.py` 只负责运行与
> 记录，因此新增实验是新增注册项，而不是分支逻辑。下表是设计目标，不是当前全部
> 可运行的实验清单。

五个实验**全部由组合得出，没有任何实验分支写在函数体里**：

| 实验 | 配置 | 构造方式 |
|---|---|---|
| **1. Dense**（基线） | `use_dense=True, use_sparse=False, rerank.enabled=False` | `DenseRetriever` |
| **2. Dense + Sparse 加权** | `fusion.method="weighted"` | `HybridRetriever(dense, sparse, WeightedFusion)` |
| **3. + RRF** | `fusion.method="rrf"` | `HybridRetriever(dense, sparse, RRFFusion)` |
| **4. + Reranker** | `rerank.enabled=True` | `RerankedRetriever(HybridRetriever(...), CrossEncoder)` |
| **5. Chunking 对比** | `ChunkingConfig.strategy ∈ {document, fixed_window, structured}`（`contextual` 已退役） | 相同 retriever × 重新入库的策略；已在容器内运行（3 策略 × 4 arms，含 rerank） |

实验 5 的设计方式（**检索隔离与对比实验均已运行**）：`chunks` 表的主键是
`(chunk_strategy, chunk_id)`，三种策略的分块**可以共存于同一张表**。`DenseConfig` /
`SparseConfig` 带有 `chunk_strategy`（默认 `document`），Dense 与 Sparse 的 SQL 都把它
作为绑定参数过滤，行映射器再校验一次；`Settings.retrieval_chunk_strategy` 决定 RAG
链路的作用域，四个检索 Benchmark 脚本提供 `--chunk-strategy`。历史 RRF 重放还会校验
artifact 的 `parameters.chunk_strategy` 与每条命中，作用域不一致即报错。因此不需要建三套
数据库，一次入库、并行对比。已验证：578 chunks 上三种策略各自 30/30 查询完成，
`chunking_benchmark.json`（dense / sparse_or / rrf）与
`chunking_benchmark_rerank.json`（增加 `rrf_rerank`）均为 `status=PASS`。
限制：所有策略共用同一个 HNSW bootstrap 索引，无法把索引大小按策略归因；
`chunk_strategy` 过滤在 ANN 扫描之后执行，`ef_search` 较小时单策略可能返回少于 K 条
（2026-10-20 在 `ef_search=5` 下实测：全局 Top-5 中仅 3 条属于 `document`），
因此策略间召回差异部分来自各策略在全局索引中的占比；`candidate_k=10`；
延迟为单进程顺序执行。

**实验列表是配置，不是代码。** `registry.py` 把实验名映射到 `RetrievalConfig`，`benchmark.py` 只负责遍历——新增一个实验 = 新增一条配置，不改任何检索逻辑。

### 默认参数

| 参数 | 取值 | 理由 |
|---|---|---|
| Dense `top_n` | 50 | 语料仅 150 篇。取 50 意味着一路召回覆盖 1/3 语料，为 Recall 留足余量。 |
| Sparse `top_n` | 50 | 同上，且两侧候选需要可比才能做融合对照。 |
| `candidate_k` | 50 | RRF 合并去重后交给重排序的候选数。 |
| Reranker `top_k` | 10 | 金牌集合最大为 3 篇，K=10 有充分余量。指标报告 K ∈ {1,3,5,10}。 |
| RRF `k` | 60 | Cormack 等 2009 原始默认值。 |
| 加权 `alpha` | 0.5 | 起点，实验 2 中做敏感性分析。 |
| HNSW `m` / `ef_construction` / `ef_search` | 16 / 64 / 40 | pgvector 默认值，实验 4 扫描 {8,16,32} × {32,64,128,200} × {10,40,100,200}。 |

> 阶段 ⑥-B 的实际评测没有使用上表的 50/10 生产默认值，而是固定
> `candidate_k=10`、`top_k=10`、`RRF k=60`，每条 Query 5 次计时 + 1 次预热；
> 这些参数同时写入 `reports/hybrid_benchmark.json`。

### 必须如实报告的一条反直觉结论

**在 150 行的规模上，HNSW 不会比顺序扫描快，甚至在 `ef_search` 较大时更慢。** 这是正常的：HNSW 的价值在于次线性复杂度，而 150 行的顺序扫描在内存里只需微秒级。

阶段 ④ 应当**同时测量 `enable_seqscan=on/off` 的对照**，并把这条结论明写进评测报告（当前为 `reports/dense_benchmark.md`）。把它藏起来、只报一个好看的 HNSW 数字，反而会被评审者一眼看穿。正确的表述是：HNSW 此刻是为 `roadmap.md` 里 1000 万条的规模做的架构准备，而不是当前数据量下的性能优化。

---

## 10. 已知的评测陷阱

**Precision@K 在这个数据集上信息量很低。** 每条 Query 的相关文档只有 1–3 篇，于是 Precision@10 的理论上限就是 0.1–0.3。报告里应当把它列出来（题目要求），但**结论必须以 Recall@K / MRR / nDCG@K 为主**，并显式说明 P@K 的天花板效应——否则读者会把"P@10 = 0.2"误读成系统很差。

**延迟的构成与直觉相反。** 在 150 行规模上，耗时几乎全部来自 Python 开销与模型推理，数据库本身可以忽略。报告必须分阶段给出，并注明测量硬件与是否包含预热。

---

## 11. 报告与复现架构（2026-10-20 新增，未验证）

### 11.1 一条单向的数据流

```
scripts/benchmark_retrieval.py ─┐
scripts/benchmark_index.py     ─┼─► reports/unified/*.json  (原始 artifact)
scripts/benchmark_chunking.py  ─┤          │
scripts/benchmark_fusion.py    ─┘          │  只读
                                           ▼
                              src/evaluation/report.py
                                           │
                              ┌────────────┴────────────┐
                              ▼                         ▼
                    reports/final_report.md     reports/figures/*.png
```

- **指标只在 benchmark 中计算一次。** `report.py` 读取已落盘的 artifact，不重新查询
  数据库、不重新计算指标、不补零；缺失指标显示 `-`。
- **每个表格都带来源。** 统一 artifact 的 `sha256` 与字段路径写入表头
  （`[artifact: path#field sha256=...]`），每行引用 `benchmark#arms[i]`，读者可以从
  报告数字回到 JSON 字段。
- **覆盖保护。** `artifacts.ensure_writable(paths, overwrite=False)` 在所有写入口
  （benchmark 脚本、report 脚本）执行；已有文件默认拒绝写入，必须显式
  `--overwrite`。历史 `reports/*_benchmark.*` 不在默认输出路径上。
- **合成规模与真实规模分离。** 索引扫描可用 `--synthetic-scale N` 复制向量以观察
  HNSW / IVFFlat 的参数行为；artifact 标记 `SYNTHETIC`，且**不进入任何质量指标**。
  150 行的真实数据下“参数无差异”本身就是需要如实报告的结论。
- **进程隔离。** 各 benchmark 由 `scripts/benchmark_all.py` 以子进程顺序执行：
  避免 Embedding 与 Reranker 在同一进程加载（Windows 原生崩溃风险），也让每个
  artifact 对应一个可独立重跑的命令。

### 11.2 状态语义

`PASS` 只表示该 artifact 覆盖全部预期 Query 且前置门禁通过；缺 Query、部分臂失败
是 `PARTIAL`；有失败是 `FAIL`；外部条件（数据库、Docker、Key）不满足是 `BLOCKED`。
`None` 表示未计算，绝不写成 `0`。最终报告的状态是三个必需 artifact 状态的合取，
缺任何一个即为 `PARTIAL`。
