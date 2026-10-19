# 架构设计

> 阶段 ① 的产出。本文档描述**目录结构、技术选型、模块职责、数据流、实验架构**。
> 业务代码在后续阶段实现；每个模块文件顶部都写明了它的单一职责与实现阶段。
>
> **阶段 ⑩ 校正说明（2026-10-19）**：本文保留阶段 ① 的设计内容，但以下设计目标
> 与当前实现不同，实际状态以 `README.md` 和 `docs/roadmap.md` 为准：
> `reports/results.md`、`reports/figures/*.png`、`make report`、`make pipeline`、
> `cli.report` / `cli.evaluate` 均未实现；实际报告是
> `reports/{dense,sparse,hybrid,rerank,rag}_benchmark.md` / `.json`，由
> `scripts/*_benchmark.py` 生成。阶段编号已统一为 **⑨ README / roadmap 定稿**、
> **⑩ 项目最终整合与 GitHub 提交准备**；FastAPI 仍未实现。

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
│   │   └── registry.py              # Config -> Retriever 工厂
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # 纯函数指标
│   │   ├── benchmark.py             # 编排
│   │   └── report.py                # 渲染
│   │
│   └── utils/
│       ├── logger.py
│       └── timing.py                # 只采集，不算分位数
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
│   └── figures/                     # 预留；当前为空
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
| **不加 `notebooks/`** | 实际报告由 `scripts/*_benchmark.py` 生成 Markdown/JSON；`make report` 尚未实现。Notebook 依赖手工执行顺序，与"可复现"优先级冲突。 |

---

## 4. 技术选型表

| 技术领域 | 最终方案 | 备选方案 | 选择原因 |
|---|---|---|---|
| Language | **Python 3.12**（容器内）<br>本地 ≥3.11 | 3.11 / 3.13 / 3.14 | 3.11 是题目下限；3.12 的 wheel 覆盖最完整。**本机 3.14.3 也能跑**（设计时核实 torch 2.14.1 提供 cp314 wheel；实际评测环境的版本存在漂移，见 `docs/roadmap.md` §4.5），但容器固定 3.12 以消除环境漂移。 |
| Database | **PostgreSQL 16** (`pgvector/pgvector:pg16`) | SQLite / 独立向量库 | 题目指定；且 FTS 与向量检索同库，省掉一整套同步逻辑。 |
| Vector | **pgvector** | FAISS / Qdrant | 题目指定。 |
| Vector Index | **HNSW**（余弦） | IVFFlat | IVFFlat 需要足够数据量才能训练出有意义的聚类；150 行下几乎没有可训练的分布，且需要预先指定 `lists`。HNSW 无需训练、增量友好、召回更稳。**IVFFlat 原计划作为对照实验，但未执行**。 |
| Embedding | **`BAAI/bge-large-en-v1.5`**（1024 维，本地） | `text-embedding-3-small`（API）<br>`all-MiniLM-L6-v2`（384 维） | 见下方说明。 |
| Sparse | **PostgreSQL FTS**（`tsvector` + GIN + `ts_rank_cd`） | Elasticsearch / BM25 库 | 见下方说明。 |
| Fusion | **RRF**（k=60） | 加权得分归一化 | 见下方说明。 |
| Reranker | **`BAAI/bge-reranker-base`**（本地 Cross-Encoder） | `cross-encoder/ms-marco-MiniLM-L-6-v2`<br>Cohere Rerank（API） | 见下方说明。 |
| Evaluation | **纯 Python + numpy** | ranx / pytrec_eval | 4 个指标 + 3 个延迟分位数，自己实现约 60 行，且**可被单元测试逐一验证**。引入评测库反而增加依赖和黑盒。 |
| Container | **Docker Compose**（db + app） | 裸机安装 | 题目要求"一键运行"；且见 §8 环境决策。 |
| API | 后续阶段（FastAPI，未实现） | — | 加分项，不阻塞主线。当前 `pyproject.toml` 已预留 `[api]` 可选依赖组。阶段 ⑧ 交付的是进程内 RAG 问答链路与 CLI，不是 HTTP 服务；`src/api` 不存在。 |

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
| `ingestion/indexer.py` | 批量写入；建 HNSW 索引 | `list[Chunk]` + 向量 | 数据库行 | `cli.ingest`, `cli.index` | ③✅ 写入 / ④ 索引重建 |

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

> 下表保留阶段 ① 的设计职责。实际入口是 `scripts/benchmark_*.py`（直接调用
> 具体 benchmark 模块并写 `reports/*_benchmark.md` / `.json`）；`cli.evaluate`
> 与 `cli.report` 未实现。

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

---

## 6. 核心数据流

### 写入链路（实际由 `make ingest` 触发；`make pipeline` 未实现）

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
        ▼  indexer.build_hnsw(m, ef_construction)
   HNSW 索引就绪
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
> 两个索引不会打架：`indexer.py` 在实验 4 里 `DROP INDEX IF EXISTS` 后按扫描到的参数重建，并从那一刻起拥有它。`init.sql` 只在数据目录为空时执行一次，此后不再介入。
>
> 参数不写死在 SQL 里：`docker/init.sh` 用 psql 的 `:variable` 把 `APEX_HNSW_M` / `APEX_HNSW_EF_CONSTRUCTION` 注入进去。改这两个值需要 `make reset`（初始化脚本只在空数据目录上跑）。

**④ `chunks.embedding` 是 `NOT NULL`。** 阶段 ② 的指令建议允许 `NULL`，这里没有采纳：pgvector 会把 `NULL` 向量的行**静默排除**在 HNSW 索引之外，于是这样的行对顺序扫描可见、对 ANN 检索不可见——一个没有任何测试能发现的 Recall@K 偏差。把这个状态变得不可表达，比事后检测它便宜得多。

**⑤ `documents` 表没有 `embedding` 列，这是刻意的。**

阶段 ③ 的指令写的是"写入 `documents.embedding` 字段"，但那张表确实没有这一列，向量在 `chunks` 上。这不是遗漏，而是阶段 ① 的双表设计：分块策略是**实验变量**（实验 5 要对比 document / fixed_window / contextual 三种），把向量挂在 `documents` 上会让一篇文档只能有一个向量，实验 5 直接无法进行。

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

**已核实**：本机 Python 3.14.3，torch 2.14.1 提供 `cp314` wheel，本地跑得动。**但所有 Makefile 目标仍统一走 `docker compose run`。**

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

> **实现状态（阶段 ⑦-B）**：实验 1（Dense）、实验 3（RRF）、候选并集基线与
> 实验 4（Reranker）已实测；实验 2（加权）与实验 5（分块策略对比）仍未实现。
> 下表是设计目标，不是当前全部可运行的实验清单。

五个实验**全部由组合得出，没有任何实验分支写在函数体里**：

| 实验 | 配置 | 构造方式 |
|---|---|---|
| **1. Dense**（基线） | `use_dense=True, use_sparse=False, rerank.enabled=False` | `DenseRetriever` |
| **2. Dense + Sparse 加权** | `fusion.method="weighted"` | `HybridRetriever(dense, sparse, WeightedFusion)` |
| **3. + RRF** | `fusion.method="rrf"` | `HybridRetriever(dense, sparse, RRFFusion)` |
| **4. + Reranker** | `rerank.enabled=True` | `RerankedRetriever(HybridRetriever(...), CrossEncoder)` |
| **5. Chunking 对比** | `ChunkingConfig.strategy ∈ {document, fixed_window, contextual}` | 相同 retriever × 重新入库的策略 |

实验 5 的实现方式：`chunks` 表的主键是 `(chunk_strategy, chunk_id)`，三种策略的分块**可以共存于同一张表**，检索时按 `chunk_strategy` 过滤即可。因此不需要建三套数据库，一次入库、并行对比。

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
