# Apex RAG — 信用卡知识库检索增强问答

面向呼叫中心坐席 Co-Pilot 的检索增强生成引擎。在 150 篇信用卡政策文档、
30 条标注 Query 上，量化对比**朴素稠密检索**与**混合检索 + 重排序**流水线，
并把最终证据接入可替换的 LLM Provider 生成带引用的答案。

> 当前进度：**阶段 ⑧（RAG 端到端问答链路集成与评测）已实现并完成离线实测，
> 等待人工验收**；**阶段 ⑨（README / roadmap 定稿）文档已定稿，等待人工验收**；
> **阶段 ⑩（项目最终整合与 GitHub 提交准备）交付检查完成，等待人工确认**，
> 尚未 `git init` / commit / push。
> ⑦-B 已验收，判定 PASS WITH MINOR ISSUES。Dense、Sparse OR/AND、
> 候选并集、RRF 与 Cross-Encoder Reranker 均已在真实 PostgreSQL 上跑完 30 条标注
> Query。阶段 ⑧ 新增上下文组装、提示词、OpenAI 兼容 Provider、引用校验、结构化答案
> 与 30 条 Query 的问答评测；结果见
> [`reports/rag_benchmark.md`](reports/rag_benchmark.md)。Reranker 实测见
> [`reports/rerank_benchmark.md`](reports/rerank_benchmark.md)；⑥-B 混合检索结果见
> [`reports/hybrid_benchmark.md`](reports/hybrid_benchmark.md)。
>
> **真实 LLM 评测尚未完成**：正式问答报告生成时使用显式标注的 mock Provider，
> 本阶段也没有调用真实 LLM API；mock 答案不是模型输出，不能代表生成质量。
> 本地 `.env` 是否配置 Key 不作为评测依据，报告中的 `real_llm_calls` 仍为 `false`。
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

## 前置条件

只需要 **Docker**（含 Docker Compose v2）。不需要在本机安装 Python、PostgreSQL
或任何数据库扩展——所有 `make` 目标都在容器内执行，宿主 Python 版本无关紧要。

```bash
docker --version           # 20.10+
docker compose version     # v2
```

> **Windows 用户注意**：Git for Windows **不**附带 `make`。请用
> `choco install make` 或 `scoop install make` 安装，或者直接运行 `make` 目标里
> 的那条命令——每个目标都只是 `docker compose run --rm app <命令>` 的简写，
> 例如 `make db-check` 等价于
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
| **Chunk** | 本阶段只用 `document` 策略（1 篇文档 = 1 个 chunk）。`fixed_window` / `contextual` 属于阶段 ⑤ 的实验 5 |
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
| `make ask-rag` | 单问题诊断入口（默认 mock；真实 Provider 需 API Key） |
| `make evaluate` | `benchmark-hybrid` 的别名 |
| `make test` / `make lint` / `make typecheck` | 测试与静态检查 |
| `make index` / `make tune` / `make report` / `make pipeline` | **尚未实现**（对应的 CLI 子命令仍待后续阶段） |
| `make help` | 列出全部目标 |

---

## GitHub 交付与复现状态

> 本节记录阶段 ⑩ 的只读交付检查结论；它不表示项目已通过生产级验收，也不表示
> 已经执行 `git commit` 或 `git push`。

- **当前目录不是 Git 仓库**：没有 `.git`，因此无法核对历史提交、已跟踪文件或
  未跟踪文件。首次 `git init`、选择提交范围和远程仓库都需要人工执行。
- **`.env` 是本地文件，可能含真实凭据**，已在 `.gitignore` 与 `.dockerignore` 中
  排除；不要提交、打包或上传。若曾把 `.env` 复制到别处，建议轮换其中的 Key。
- **模型缓存**：`.cache/huggingface` 约 1.1 GB，已被 `.gitignore` 排除；
  `.dockerignore` 已排除 `.cache/`，因此 `docker build` 不会把本地模型缓存送入
  构建上下文。该排除是 Stage ⑩-A 经授权的配置修正，不影响历史报告。
  Docker Compose 的 bind mount、`hf_cache` named volume 与模型缓存初始化仍需
  独立验证；**排除 `.cache/` 不等于 Docker 全链路通过**。
- **不应提交的生成物**：`.mypy_cache`（约 217 MB）、`.pytest_cache`、`.ruff_cache`、
  `__pycache__`、日志与临时文件；均已由 `.gitignore` 覆盖。
- **可以提交的交付物**：`src/`、`tests/`、`scripts/`、`data/`、`reports/`（五个
  Benchmark 的 Markdown 与 JSON，以及汇总 `reports/results.md`）、`docs/`、
  `pyproject.toml`、`Dockerfile`、`docker-compose.yml`、`Makefile`、`.env.example`、
  `.gitignore`、`.dockerignore`。
- **复现性尚未完全验证**：没有锁文件；`pyproject.toml` 的精确 pin 与历史报告中的
  实际安装版本存在漂移；`psutil` 在代码中被导入但未写入依赖声明；仓库声明 MIT
  但没有 `LICENSE` 文件。详见 `docs/roadmap.md` 的“文档一致性与待核实项”。
- **报告路径**：`reports/*_benchmark.json` 与部分 Markdown 报告保留了生成时的本地
  绝对路径（例如项目目录、模型缓存路径与仓库外崩溃证据路径）。这些是历史证据的
  一部分，不应直接改写或覆盖；公开提交前需人工审查。若需要公开版本，可另行生成
  脱敏副本，但必须保留原始报告、原始指标与结论不变。
- **公开提交前检查**：确认 `.env`、`.cache/`、`.venv/`、日志、压缩包与临时文件均
  未进入提交；若 `.env` 或其他本地配置中存在 API Key，应在仓库外轮换，并确认
  Git 历史、日志与公开归档中没有密钥值。不要读取、打印或复制密钥内容。
- **Docker 边界**：Docker 只验证过数据库侧；容器内的 ingest、五个 Benchmark、
  RAG、模型缓存与离线加载均未验证。宿主 Python 结果不能当作容器验证。

---

## 项目结构

```
src/
├── config.py        集中配置（pydantic-settings，APEX_ 前缀）
├── cli.py           唯一入口
├── db.py            连接池 + schema 诊断
├── schema/          数据形状：Document / Chunk / SearchHit / RAG 答案与配置
├── ingestion/       loader → chunker → embedder → indexer
├── retrieval/       dense / sparse / fusion / reranker / pipeline / registry
├── rag/             context / prompt / provider / pipeline / factory / historical
├── evaluation/      metrics / benchmark / report / rag_benchmark
└── utils/           logger / timing

data/               数据集（随仓库提交）
docs/               architecture.md（阶段 ① 设计稿）/ roadmap.md（当前状态与路线）
reports/            dense/sparse/hybrid/rerank/rag 的 JSON 原始结果与 Markdown 报告
scripts/ingest.py   数据导入入口（转发给 src/cli.py，不含逻辑）
scripts/benchmark_dense.py / benchmark_sparse.py / benchmark_hybrid.py / benchmark_rerank.py  各阶段只读评测入口
scripts/benchmark_rag.py  阶段 ⑧ 问答评测入口（mock 默认；真实 Provider 需 API Key）
scripts/ask_rag.py        阶段 ⑧ 单问题诊断入口
scripts/diagnose_dense.py  只读诊断：GUC、自然/强制 HNSW 计划、Top-K
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

HNSW 参数可用 `APEX_HNSW_M` / `APEX_HNSW_EF_CONSTRUCTION` 覆盖，但**只在空数据目录上生效**，
改动后需要 `make reset`。阶段 ④ 的 `indexer.py` 会用扫描出来的参数重建该索引。

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
容器内的 RAG 路径本轮未验证。

结果见 [`reports/rag_benchmark.md`](reports/rag_benchmark.md)（**Status: PARTIAL**，
原因是真实 LLM 评测未运行）：

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
  答案正确、Faithful 或 Relevant。真实 LLM 评测未执行：报告使用 mock Provider，
  本阶段也没有调用真实 API。
- 检索是 **historical 重放**：候选身份与 RRF 名次来自 ⑥-B artifact，文本实时读自
  PostgreSQL，Reranker 实时运行；Dense/Sparse/RRF 未在本阶段重新执行。因此
  证据指标衡量的是被重放的候选池，不是一次新的检索运行。
- 在本机把 Embedding 与 Reranker 放入同一进程时，live 链路在 reranker 加载阶段
  触发原生段错误（exit 139）。这不是测试失败，也没有被隐藏；historical 模式是
  为避免该崩溃而明确标注的替代路径。
- 答案正确性、Faithfulness、Answer Relevance 未计算：数据集只有 `gold_response`
  参考文本，没有评分 rubric、人工评分或已运行的 LLM-as-a-Judge。

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
| 评测 | 自实现 Semantic Recall@K / MRR / Hit@1 + P50/P95；Precision/nDCG 因标签非穷尽暂不计算；生成质量指标未计算 |

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
| ⑧ | RAG 端到端问答链路集成与评测 | ✅ 完成（离线链路实测；真实 LLM 未运行） | **待人工验收**（自评 PARTIAL，不替代验收） | `reports/rag_benchmark.md` / `.json`（Status: PARTIAL；30/30 status=ok，引用有效 30/30，证据 Recall 1.0000、MRR 0.9833；Provider=mock，historical 重放；真实 LLM 因缺 Key 未运行） |
| ⑨ | README / roadmap 定稿 | 🟡 文档已定稿 | **待人工验收** | `README.md`、`docs/roadmap.md`；指标仅引用 `reports/*` 与 Stage A 审核报告，未重新运行任何测试 |
| ⑩ | 项目最终整合与 GitHub 提交准备（FastAPI 加分项仍未实现） | 🟡 交付检查完成 | **待人工确认**；未 commit / push | 本阶段只读检查 README、roadmap、architecture、报告、依赖与忽略配置；`src/api` 不存在；未重新运行任何测试 |

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

- **真实 LLM 端到端评测未运行**：正式报告为 mock Provider 的 PARTIAL 运行；
  mock 文本以 `MOCK` 开头，不是模型输出。报告生成时未配置真实 Key，本阶段也未
  调用真实 API；本地 `.env` 不作为评测依据。
- 答案正确性、Faithfulness、Answer Relevance 未计算；没有 rubric、人工评分或已执行的
  LLM-as-a-Judge。证据 Recall/MRR 只是检索指标，不能当作生成准确率。
- live 链路（Dense + Sparse + RRF + Embedding + Reranker + 生成）在本机同一进程触发
  原生段错误（exit 139），未完成；正式评测使用 `historical` 候选重放 + 实时 PostgreSQL
  文本 + 实时 Reranker。
- 真实 Provider 的超时、限流、5xx、401、网络错误与重试分类只在单元测试中通过
  `httpx.MockTransport` 验证，未对真实网络或真实 DeepSeek 服务实测。
- Docker/容器内的 RAG 路径、模型缓存与离线加载未验证。
- 完整 pytest 未重跑（已知原生崩溃风险）；只运行模块级 RAG 测试。
- 依赖固定未重新安装核验；宿主版本仍与 `pyproject.toml` 存在漂移。
- 中文与中英混合语言的检索、生成与引用未评测。
- 规模、并发、GPU、跨机器复现性未验证。
- 延迟为单进程宿主 CPU，首次查询包含模型懒加载；LLM 延迟为 mock，不代表真实网络。

> 路线编号说明：阶段 ⑧ 的实际内容是 RAG 端到端问答；**⑨ 为 README / roadmap
> 定稿**，**⑩ 为项目最终整合与 GitHub 提交准备**。FastAPI 仍是未实现的加分项，
> 不属于本阶段交付。`docs/architecture.md` 是阶段 ① 设计稿，已加入阶段 ⑩ 的
> 编号与路径校正说明；状态以 README / roadmap 为准。

### 实现状态分类

| 分类 | 内容 | 依据 |
|---|---|---|
| **已实现且有可复核报告** | `document` 分块与 150 条导入；Dense 检索；Sparse OR/AND；候选并集与 RRF；Cross-Encoder Reranker；RAG 上下文、提示词、Provider、引用校验与结构化答案 | `reports/dense_benchmark.md`（PASS）、`sparse_benchmark.md`（PASS）、`hybrid_benchmark.md`（PASS）、`rerank_benchmark.md`（PASS）、`rag_benchmark.md`（PARTIAL） |
| **已实现但只有单元/模块级证据** | Provider 超时/限流/5xx/401/网络错误映射与重试；上下文截断/丢弃；引用标签校验；空候选；重复身份/非法输入；失败分类 | `tests/test_rag_*.py`、`tests/test_reranker.py`、`tests/test_retrieval.py` 等；完整 pytest 未重跑 |
| **部分验证** | RAG 端到端（mock Provider + historical RRF 重放）；Reranker（宿主 CPU、历史 RRF Top-10 池）；Dense/HNSW（150 chunks，无规模结论）；Docker（仅数据库路径） | 对应报告中的“未验证事项”与本页限制 |
| **未验证** | 真实 LLM API；RAG live 全链路稳定性；多候选 rerank / E-5B；原生崩溃根因；Docker 全链路；完整 pytest；中文与混合语言；大规模性能；并发；跨机器复现；FastAPI | 见 [`docs/roadmap.md`](docs/roadmap.md) 的“未完成验证” |
| **已知问题/风险** | 原生崩溃未结案；依赖漂移；标签非穷尽导致 Precision/nDCG 不可算；historical 重放不是新检索；mock 答案不代表生成质量；E-2/E-4 完整证据完整性 INCONCLUSIVE | 见下节与 [`docs/roadmap.md`](docs/roadmap.md) |

### 已知问题与风险

1. **原生崩溃根因未确认**：历史 `0xC0000005` / `torch_cpu.dll` 证据仍在；
   E-5A/E-5C 只覆盖单候选与空候选，不等于修复。
2. **真实 LLM 未评测**：正式 RAG 报告使用 mock Provider；答案正确性、
   Faithfulness、Answer Relevance 未计算。
3. **依赖漂移**：`pyproject.toml` 是精确 pin，但报告中的宿主实际版本不同
   （例如 torch 声明 2.14.1、实测 2.13.0）；没有 `.venv` 或锁文件。
4. **容器路径未验证**：Dense/Sparse/Hybrid/Rerank/RAG 实测均在宿主 Python 运行；
   Docker 仅验证了数据库侧，RAG 与模型缓存的容器路径未验证。
5. **完整 pytest 未重跑**：只运行模块级测试；宿主 torch/transformers 组合存在
   已知原生崩溃风险。
6. **中文/混合语言未评测**：Sparse 的中文探针在英文语料上 0 命中；CJK fallback
   是两字符子串匹配，不是分词。
7. **规模与并发未验证**：150 chunks 下 HNSW 没有延迟优势，自然计划是 Seq Scan；
   不能外推到大规模、GPU、并发或跨机器。
8. **指标口径限制**：标注非穷尽，Precision/nDCG 不计算；Recall/MRR 是检索指标，
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
| [`reports/rag_benchmark.md`](reports/rag_benchmark.md) / `.json` | PARTIAL | 30 queries, mock Provider, historical RRF replay + live PostgreSQL text + live reranker | 30/30 status=ok；引用有效 30/30；证据 Recall macro/micro 1.0000/1.0000；MRR 0.9833；total P50/P95 970.9/1082.2 ms；real LLM calls False |
| [`reports/results.md`](reports/results.md) | 汇总（非新评测） | 五份报告索引、口径差异与限制 | 只引用现有报告，不新增指标；不替代原始报告或验收记录 |
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
#   R A G  
 #   R A G  
 