# 信用卡智能客服助手 · 前端 Demo

银行企业视觉风格的 ChatGPT 式信用卡客服演示页。首页以对话为核心，展示真实的
`POST /api/v1/search` 检索证据与 `POST /answer` 生成结果；聊天中的回答、引用、分数、实时请求耗时和原始
JSON 都来自本次网络请求，不使用前端 mock 或预存成功响应。

> 这是面试演示环境，不是生产系统。知识库为英文信用卡政策语料（150 篇文档），英文提问更稳定。

## 技术栈

- React 18.3 + TypeScript 5.7 + Vite 6
- 普通 CSS（`src/styles/global.css`），无 UI 框架、无第三方图标库
- 浏览器内多轮对话，无登录、无服务端会话、无持久化
- 通过 Vite 同源代理访问 `http://127.0.0.1:8000`，后端无需新增 CORS

## 目录

```text
frontend-demo/
├─ src/api/          # 基址/超时、类型、unknown 安全解析、fetch 客户端
├─ src/hooks/        # 连接状态、对话与串行 search→answer、自动滚动
├─ src/components/   # 页头、消息、证据卡片、原始响应、输入框、技术详情抽屉
├─ src/data/         # 静态离线实验指标（来自 reports/unified/rerank_20to5.md）
├─ src/styles/       # 设计 token、布局、响应式、减少动画
├─ vite.config.ts    # dev/preview 代理到 127.0.0.1:8000
└─ .env.example      # VITE_API_BASE_URL 默认留空（走代理）
```

## 启动

前置要求：Node.js 18+（建议 20 LTS）与 npm；后端需要 Docker Desktop（或已配置好的宿主
Python 环境）。

### 1. 后端（仓库根目录）

```powershell
cd "D:\信用卡知识库 RAG 检索评估"
docker compose up -d db
docker compose --profile api up -d api
docker compose ps
Invoke-RestMethod http://localhost:8000/healthz
Invoke-RestMethod http://localhost:8000/readyz
```

`/healthz` 只表示进程存活；`/readyz` 才检查数据库。若 Docker 不可用，可在已配置好的宿主
Python 环境运行：

```powershell
cd "D:\信用卡知识库 RAG 检索评估"
.\.venv\Scripts\python.exe -m src.cli serve
```

### 2. 前端

```powershell
cd "D:\信用卡知识库 RAG 检索评估\frontend-demo"
npm install
npm run dev
```

打开 <http://127.0.0.1:5173/>。Vite 会将 `/healthz`、`/readyz`、`/api`、`/answer`、
`/acceptance`、`/docs` 代理到 `http://127.0.0.1:8000`。

生产构建与类型检查：

```powershell
npm run typecheck
npm run build
npm run preview   # http://127.0.0.1:4173/，同样带代理
```

## 实际调用的接口

| 接口 | 用途 | 说明 |
| --- | --- | --- |
| `GET /healthz` | 进程存活 | 不检查数据库或模型 |
| `GET /readyz` | 数据库就绪 | 失败返回 503 / `degraded` |
| `POST /api/v1/search` | 检索证据 | 请求 `{query, top_k}`；Top-K 可选 3/5/10 |
| `POST /answer` | 真实生成回答 | 请求 `{query}`；不接受 `top_k` |
| `GET /acceptance/api/summary` | 只读验收摘要 | 失败显示「验收摘要暂不可用」 |

### 聊天流程

1. 用户提问后先调用 `POST /api/v1/search`，展示真实的命中数、Top-K、Top-1、文档卡片、
   服务端 `timings_ms` 和原始响应。
2. 检索请求结束后再调用 `POST /answer`，展示生成回答、引用、`provider/model/usage`、
   生成证据与生成耗时。
3. 检索网络失败或超时：不调用 `/answer`，明确标注「本次没有生成式回答」。
4. `/answer` 失败、超时或返回错误：保留检索证据，显示后端错误，不伪造答案。

Top-K 只影响 `/api/v1/search` 返回的证据条数；`/answer` 的证据条数由服务端
`APEX_RAG_TOP_K` 配置决定，界面中已明确说明。

## 数据来源与真实性

| 类型 | 来源 | 界面标注 |
| --- | --- | --- |
| 实时连接 | `/healthz`、`/readyz` | 「实时」 |
| 实时检索 | `/api/v1/search` 响应 | 检索摘要、文档卡片、原始响应 |
| 实时生成 | `/answer` 响应 | 生成回答、引用、生成证据、耗时 |
| 实时验收 | `/acceptance/api/summary` | 「只读摘要」 |
| 离线指标 | `reports/unified/rerank_20to5.md` | 「离线实验报告，不代表当前这次请求的实时结果。」 |
| 静态架构 | 仓库检索链路说明 | 「静态架构说明」 |

- 不读取 `.env`、不包含任何 API Key，前端不直接调用外部 AI 服务。
- 不使用 `dangerouslySetInnerHTML`；文档正文按纯文本渲染。
- `raw` 面板直接序列化本次 `fetch` 的响应体；解析失败时显示实际文本。
- 后端未提供 Dense、Sparse、RRF 的分项实时耗时，因此架构图只做静态说明。
- `/healthz` 成功不等于数据库或模型可用；`PASS` 也不等于生产认证。

## 连接状态

| 状态 | 判定 | 展示 |
| --- | --- | --- |
| `checking` | 请求进行中 | 黄色「连接检查中」，不会默认显示绿色 |
| `connected` | `/healthz` 200 且 `/readyz` `ready` | 绿色 |
| `degraded` | `/healthz` 200，但 `/readyz` 未就绪 | 黄色提示横幅 |
| `unreachable` | 网络错误或超时 | 红色横幅，提示检查 Docker/8000 端口 |
| `error` | `/healthz` 非 200 或响应非法 | 红色「健康接口异常」 |

验收摘要独立加载，失败时不影响聊天区，显示「验收摘要暂不可用」和重试按钮。

## 已知限制

- 知识库与评测查询均为英文，Embedding 模型为 `BAAI/bge-large-en-v1.5`；
  接口不会拒绝中文，但当前仅验证英文查询，中文检索相关性和回答质量未经验证。
- `/answer` 不接受 Top-K，前端不能改变生成阶段的证据条数。
- 未实现流式输出、服务端会话、登录、权限、跨设备历史或页面刷新后的历史恢复。
- 检索接口超时设为 120 s、生成接口 150 s，以覆盖首次加载本地模型的耗时。
- 离线 `Recall 1.0000` 仅针对 30 条英文评测查询，不代表线上准确率。

## 验证记录

### 已执行：依赖安装、类型检查与生产构建（2026-10-21）

通过会话内 `!` 命令执行（工具侧 auto-mode 分类器仍在阻断非只读调用）：

```powershell
npm --prefix "D:\信用卡知识库 RAG 检索评估\frontend-demo" install
npm --prefix "D:\信用卡知识库 RAG 检索评估\frontend-demo" run typecheck
npm --prefix "D:\信用卡知识库 RAG 检索评估\frontend-demo" run build
```

实际结果：

- `npm install`：`added 71 packages in 13s`；生成 `package-lock.json`。
- `npm run typecheck`：`tsc --noEmit` 无报错。
- `npm run build`：Vite `6.4.5`，56 modules transformed，`built in 513ms`；生成
  `dist/index.html`、`dist/assets/index-9ESBD1IK.css`（39.05 kB，gzip 8.23 kB）和
  `dist/assets/index-DbO61YXX.js`（215.36 kB，gzip 66.51 kB）。
- 修正 `embedding_dimensions` 数组解析后复跑：`typecheck` 通过；
  `build` 再次成功（Vite `6.4.5`，56 modules，`built in 519ms`），生成
  `dist/assets/index-1TPp1_-f.js`（215.52 kB，gzip 66.56 kB）。
- `node_modules/`、`dist/` 已由 `.gitignore` 排除；`package-lock.json` 已生成。

### 已执行：只读静态核对

- 逐文件核对前端请求/解析逻辑与后端契约：`src/api/schema.py`、`src/api/app.py`、
  `src/api/acceptance.py` 的字段、HTTP 状态与错误体映射一致。
- 核对 `src/data/offlineMetrics.ts` 的数字与 `reports/unified/rerank_20to5.md` 逐项一致。
- 只读统计 `data/kb_documents.json` 为 150 篇文档。
- 核对 Vite 代理路径、`tsconfig.json` 的 strict 配置、组件导入与所有按钮的实际动作。
- 只读 `git status --short` 确认工作区只有本目录（未跟踪）与原有的
  `src/evaluation/report.py` 未提交修改；未改动后端检索、评测数据或 API 行为。

### 已执行：运行时部分验证（2026-10-21）

工具侧非只读命令仍被 auto-mode 分类器阻断；以下结果由用户在会话内通过 `!` 命令和
浏览器实际执行：

- `GET /healthz`：`HTTP 200`，返回 `{"status":"ok","service":"apex-rag-api"}`。
- 浏览器真实查询「什么是外汇交易费」：
  - `POST /api/v1/search`：`HTTP 200`。
  - `hit_count = 5`，`results.length = 5`，实际 `top_k = 5`，`retriever = hybrid`。
  - 浏览器端到端 `4702.00 ms`；服务端 `retrieval_ms = 4682.78`、`total_ms = 4682.13`。
  - 页面已渲染真实的命中数、结果数组、Top-K、retriever、HTTP 状态和耗时。
- `POST /answer`：`HTTP 503`，`status = config_error`，`error.stage = llm_config`；
  错误信息为 `LLM API key is not configured; set APEX_LLM_API_KEY (or DEEPSEEK_API_KEY)
  in the environment or .env`。
  - 页面显示「本次没有生成式回答」和后端错误详情，没有生成或展示任何伪造答案。
- 前端 Vite 页面已在浏览器中打开，真实检索结果与 `/answer` 失败状态均已显示。

### 已执行：真实生成与就绪验证（2026-10-21）

确认 `.env` 中存在 `APEX_LLM_API_KEY` 后，重建 API 容器使运行中的进程重新加载
环境变量（未读取、未打印 Key 值）：

- `GET /readyz`：`HTTP 200`，`status = ready`；9 项检查全部 `ok`，包括
  PostgreSQL 16.14、pgvector、`vector(1024)`、HNSW/GIN 索引和
  `documents=150 / chunks=578`。
- `POST /answer`：`HTTP 200`，`status = ok`。
  - `provider = openai_compatible`，`model = deepseek-flash`，
    `provider_attempted = true`，`error = null`。
  - `usage`：prompt 1695 / completion 133 / total 1828 tokens。
  - `citations = E1, E3, E2, E4`；`citation_status = ok`，
    `invalid_citations = []`，`insufficient_evidence = false`。
  - `timings_ms`：`retrieval_ms = 18470.5700`、`llm_ms = 1308.2729`、
    `total_ms = 19779.8389`。
  - 回答基于 E1–E4 的四类卡产品费用差异；引用校验通过，未出现无效引用。
- 这是一次真实 Provider 调用，不是 mock 或预存答案。

### 已执行：验收摘要与 Top-K 3/10（2026-10-21）

- `GET /acceptance/api/summary`：返回 `status = PASS`、`read_only = true`；
  5/5 artifact 为 `PASS`，6 项自动检查全部 `pass`，Judge 覆盖 30/30 且为真实模型调用；
  `documents = 150`、`chunks = 578`、`embedding_dimensions = [1024]`。
- `POST /api/v1/search`，`top_k = 3`：`HTTP 200`，`hit_count = 3`，
  `results.length = 3`，`retriever = hybrid`；
  `timings_ms`：`retrieval_ms = 882.37`、`base_ms = 97.023`、`rerank_ms = 785.292`、`total_ms = 882.329`。
- `POST /api/v1/search`，`top_k = 10`：`HTTP 200`，`hit_count = 10`，
  `results.length = 10`，`retriever = hybrid`；
  `timings_ms`：`retrieval_ms = 860.285`、`base_ms = 67.008`、`rerank_ms = 793.246`、`total_ms = 860.261`。
- 两次请求的 `top_k` 与结果数组长度一致，证明界面选择的 Top-K 会真实传到后端；
  `top_k = 3` 的结果是同一查询 `top_k = 10` 的前三名。
- 实时响应中的 `embedding_dimensions` 是数组 `[1024]`；据此修正了前端解析，
  现在按数组展示，而不是误显示为「未返回」。

### 仍待验证（NOT RUN）

- 浏览器中真实生成回答、引用和生成证据的展示
- 验收摘要、真实耗时和 Top-K 在技术详情抽屉中的展示
- 原始响应展开、复制、文档卡片展开全文、多轮对话、清空、IME 回车
- 后端下线、超时、404/422/500、空结果等异常状态
- 1366×768、1920×1080、约 390px 的完整布局检查

仓库中的 `reports/judge-real-3q/` 是 2026-10-20 的历史真实 `/answer` 记录，只能作为
先前的后端能力证据，不代表本次前端联调已经验证。
