# RAG 端到端问答评测（阶段 ⑧）

- Status: **PARTIAL**
- Reason: offline/mock pipeline completed; real LLM evaluation was not run
- Generated: 2026-10-17T09:48:25.898943+00:00
- Queries: 30
- Provider: `mock` / `mock` / model `mock-evidence-v1`
- Real LLM calls: **False**
- Retrieval mode: **historical**
- Retrieval provenance: accepted Phase 6B RRF candidate identities replayed; live PostgreSQL text and live cross-encoder reranking
- Final evidence / candidates: 10 / 10
- Prompt: `rag-answer-v1` (hash per query)

> **MOCK RUN.** The provider is deterministic and offline. Its answer text begins with `MOCK`. This report validates pipeline wiring, evidence transfer, citation mapping and failure handling; it is **not** a real model-quality result.

## 1. 结论摘要

| 指标 | 值 | 定义 / 样本 |
|---|---|---|
| 端到端成功率 | 1.0000 (30/30) | status ∈ {ok, insufficient_evidence} |
| 生成失败率 | 0.0000 (0/30) | Provider 已调用但未产出结构化答案 |
| 证据不足率 | 0.0000 (0/30) | 无检索结果/空上下文短路 + 模型声明证据不足 |
| 引用有效率 | 1.0000 (30/30) | 有效 E 标签 / 全部返回标签 |
| 证据 Recall macro | 1.0000 | final evidence, 30 queries |
| 证据 MRR | 0.9833 | final evidence, retrieval metric only |
| 上下文截断查询数 | 0 | max_context_chars=8000 |

| 延迟（ms） | P50 | P95 | 样本数 | 口径 |
|---|---|---|---|---|
| total | 970.8791 | 1082.2045 | 30 | wall-clock; nearest-rank percentiles; no warm-up exclusion |
| retrieval | 970.0994 | 1081.4004 | 30 | wall-clock; nearest-rank percentiles; no warm-up exclusion |
| llm | 0.0675 | 0.0904 | 30 | wall-clock; nearest-rank percentiles; no warm-up exclusion |

**未计算：** answer correctness、Faithfulness、Answer Relevance。数据集只有 `gold_response` 参考文本，没有答案评分 rubric 或人工评分；本阶段也没有运行 LLM-as-a-Judge。Recall/MRR 是检索指标，不能替代生成质量。

## 2. 固定条件与评测口径

- top_k / candidates: 10 / 10
- max_context_chars / max_chunk_chars: 8000 / 2000
- retrieval_mode: `historical` — accepted Phase 6B RRF candidate identities replayed; live PostgreSQL text and live cross-encoder reranking
- temperature / max_tokens: 0.0 / 512
- prompt version: `rag-answer-v1`
- 检索链路：Dense + Sparse OR → RRF(k=60) → Cross-Encoder → 最终 Top-K。
- 证据文本来自 PostgreSQL `chunks.content`；未使用占位文本或伪造文档。
- 引用标签只允许模型引用上下文中实际出现的 `E<n>`；其他标签进入 `invalid_citations`，不会出现在 `citations` 中。
- 证据 Recall/MRR 以 `gold_doc_ids` 为分母，按最终证据顺序去重；这是检索指标。
- 逐查询原始结果、上下文、证据、引用、耗时与错误全部保存在 JSON。

## 3. Provider、Prompt 与安全

| 项 | 值 |
|---|---|
| provider kind | `mock` |
| provider name | `mock` |
| is_real | False |
| model | `mock-evidence-v1` |
| real_llm_calls | False |
| real_llm_evaluation | not_run |
| API key present | False (value never recorded) |
| LLM judge | False |

- Prompt 要求只依据证据回答、允许“无法从现有材料确认”、只返回一个 JSON 对象。
- 上下文有总字符上限与单块上限；截断状态逐查询记录。
- 缺失 API Key 时在检索前失败，错误为 `config_error`；不会回退到 mock。
- 日志不包含 API Key、请求头、Prompt 或模型响应正文。

## 4. 逐查询结果

| query | status | evidence | citations | invalid | gold recall | first gold | total ms |
|---|---|---|---|---|---|---|---|
| Q-001 | ok | 10 | 1 | 0 | 1.0000 | 1 | 24025.2685 |
| Q-002 | ok | 10 | 1 | 0 | 1.0000 | 1 | 883.0377 |
| Q-003 | ok | 10 | 1 | 0 | 1.0000 | 1 | 1066.1194 |
| Q-004 | ok | 10 | 1 | 0 | 1.0000 | 1 | 932.3779 |
| Q-005 | ok | 10 | 1 | 0 | 1.0000 | 1 | 997.7158 |
| Q-006 | ok | 10 | 1 | 0 | 1.0000 | 1 | 1039.4554 |
| Q-007 | ok | 10 | 1 | 0 | 1.0000 | 1 | 970.8791 |
| Q-008 | ok | 10 | 1 | 0 | 1.0000 | 1 | 891.2367 |
| Q-009 | ok | 10 | 1 | 0 | 1.0000 | 1 | 913.2562 |
| Q-010 | ok | 10 | 1 | 0 | 1.0000 | 1 | 972.6551 |
| Q-011 | ok | 10 | 1 | 0 | 1.0000 | 2 | 1072.0752 |
| Q-012 | ok | 10 | 1 | 0 | 1.0000 | 1 | 1007.6493 |
| Q-013 | ok | 10 | 1 | 0 | 1.0000 | 1 | 940.8283 |
| Q-014 | ok | 10 | 1 | 0 | 1.0000 | 1 | 1020.5954 |
| Q-015 | ok | 10 | 1 | 0 | 1.0000 | 1 | 1063.7731 |
| Q-016 | ok | 10 | 1 | 0 | 1.0000 | 1 | 915.1745 |
| Q-017 | ok | 10 | 1 | 0 | 1.0000 | 1 | 905.7904 |
| Q-018 | ok | 10 | 1 | 0 | 1.0000 | 1 | 961.0182 |
| Q-019 | ok | 10 | 1 | 0 | 1.0000 | 1 | 938.5821 |
| Q-020 | ok | 10 | 1 | 0 | 1.0000 | 1 | 960.4398 |
| Q-021 | ok | 10 | 1 | 0 | 1.0000 | 1 | 933.6649 |
| Q-022 | ok | 10 | 1 | 0 | 1.0000 | 1 | 1069.1838 |
| Q-023 | ok | 10 | 1 | 0 | 1.0000 | 1 | 975.7045 |
| Q-024 | ok | 10 | 1 | 0 | 1.0000 | 1 | 1071.8854 |
| Q-025 | ok | 10 | 1 | 0 | 1.0000 | 1 | 1054.2584 |
| Q-026 | ok | 10 | 1 | 0 | 1.0000 | 1 | 1006.7265 |
| Q-027 | ok | 10 | 1 | 0 | 1.0000 | 1 | 918.7247 |
| Q-028 | ok | 10 | 1 | 0 | 1.0000 | 1 | 883.9505 |
| Q-029 | ok | 10 | 1 | 0 | 1.0000 | 1 | 894.1441 |
| Q-030 | ok | 10 | 1 | 0 | 1.0000 | 1 | 1082.2045 |

## 5. 状态与失败分类

| status | count |
|---|---|
| ok | 30 |

## 6. 引用校验

- 有效引用标签：30
- 无效引用标签：0
- 含引用的查询：30 / 30
- 含无效引用的查询：0
- `status=ok` 但无引用的查询：0

无效标签没有被静默丢弃：它们保存在逐查询的 `invalid_citations` 中，且不计入 `citations`。本阶段只验证“引用是否指向实际提供的证据”，不验证引用是否真正支持答案中的每一句话；后者需要人工或已执行的 Judge。

## 7. 证据与检索

- 证据 Recall macro / micro：1.0000 / 1.0000
- 证据 MRR：0.9833
- 至少命中一个 gold 的查询：30 / 30
- 全部 gold 都进入最终证据的查询：30 / 30

这些是最终证据集合上的检索指标，不是生成答案的准确率。

## 8. 未验证事项

- The provider is a deterministic mock. Its answers begin with MOCK and are not evidence that a real LLM produces correct or faithful output.
- The 30 queries are English; Chinese and mixed-language generation are not evaluated.
- Retrieval is historical: Dense, Sparse and RRF were not rerun. The accepted Phase 6B RRF Top-10 identities and ranks were replayed, then joined to live PostgreSQL text and reranked live. Evidence metrics therefore measure the replayed pool, not a new retrieval run.
- Answer correctness, Faithfulness and Answer Relevance are not computed: no rubric, human score or executed judge exists.
- The stored gold_response is reference text for human review, not a scored label.
- Latency is a single-process host run with no concurrency, GPU or cross-machine claim.
- The first query includes lazy model loading (and, in live mode, embedding model loading); it is not a steady-state latency sample.
- The reranker runs with local_files_only=true; a fresh machine without the cached model must pass --allow-network or populate the cache first.
- Docker/容器路径未验证；本阶段在宿主 Python 运行。
- 完整 pytest 未重跑（存在已知原生崩溃风险）；只运行模块级测试。
- 依赖固定未重新安装核验；宿主版本可能仍与 `pyproject.toml` 漂移。
- Embedding 与 Reranker 同进程共存的内存行为按本次实测记录，不推广到其他机器。

## 9. 复现

本次正式评测使用的命令（`retrieval_mode=historical`）：

```bash
APEX_PG_HOST=127.0.0.1 python scripts/benchmark_rag.py --provider mock --retrieval-mode historical
```

`APEX_PG_HOST=127.0.0.1` 是本机绕过：`localhost` 同时解析到 IPv6/IPv4 时，连接池会在每个地址上耗满 `connect_timeout`。它不是代码默认值。若不加 `--retrieval-mode historical`，脚本默认走 live 链路（Dense + Sparse + RRF + Cross-Encoder 同进程）。该链路在本机 reranker 加载阶段触发原生段错误（exit 139），因此本次改用 historical 重放；这不是断言失败，也不是检索结果失败。

Mock live 链路（默认，不调用真实 API）：

```bash
python scripts/benchmark_rag.py --provider mock
```

真实 Provider（需要 `APEX_LLM_API_KEY` 或 `DEEPSEEK_API_KEY`）：

```bash
python scripts/benchmark_rag.py --provider real --limit 2 --no-write
python scripts/benchmark_rag.py --provider real
```

冒烟测试（不写报告）：

```bash
python scripts/benchmark_rag.py --provider mock --limit 2 --no-write
```

## 10. 数据安全

本评测只读知识库并调用生成 Provider：不写入数据库、不修改 `data/kb_documents.json`、`data/eval_queries.json` 或 gold 标签，不把生成答案写回知识库。原始结果见同目录 JSON。
