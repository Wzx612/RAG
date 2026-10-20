# LLM-as-a-Judge 评测

- Status: **PARTIAL**
- Reason: real judge executed on 3/30 queries (smoke); this is not a full-set quality evaluation
- Run ID: `judge-20261020T053133Z-c2c1d6be`
- Generated: 2026-10-20T05:31:33.618153+00:00
- Queries judged: **3 / 30**
- Not run: **27** (NOT RUN, not scored and not counted as zero)
- RAG provider: `real` / `openai_compatible` / model `deepseek-chat`
- Judge provider: `real` / `openai_compatible` / model `deepseek-chat`
- Same model as RAG: **True**
- Prompt versions: `{'context_relevance': 'judge-context-relevance-v1', 'faithfulness': 'judge-faithfulness-v1'}`
- Score aggregation: only successful scores enter mean/median/distribution; failures are reported separately and never scored as 0.

## 1. 指标汇总

| Metric | Attempted | Success | Failed | Success rate | Valid n | Mean | Median | Low score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Context Relevance | 3 | 3 | 0 | 1.0000 | 3 | 4.3333 | 4.0000 | 0 |
| Faithfulness | 3 | 3 | 0 | 1.0000 | 3 | 5.0000 | 5.0000 | 0 |

## 2. 分数分布（仅成功评分）

| Metric | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Context Relevance | 0 | 0 | 0 | 2 | 1 |
| Faithfulness | 0 | 0 | 0 | 0 | 3 |

## 3. 低分样例

- 没有低于阈值的成功评分。

## 4. 失败分类

- 没有 Judge 失败。

## 5. 逐查询结果

| Query | RAG status | Context Relevance | Faithfulness | Unsupported claims |
|---|---|---|---|---:|
| Q-001 | ok | success (4) | success (5) | 0 |
| Q-002 | ok | success (4) | success (5) | 0 |
| Q-003 | ok | success (5) | success (5) | 0 |

## 6. 限制

- Only 3 of 30 queries were judged; this smoke run does not describe the full evaluation set.
- Judge scores are model-generated judgements, not ground truth; no human review or inter-judge agreement was performed.
- Context Relevance scores the supplied evidence only; it is not retrieval Recall and does not show that the right document was retrieved.
- Faithfulness uses only the evidence actually sent to the answer model. A claim that is true but absent from that evidence is still unsupported here.
- The judge may be sensitive to prompt order, context length and language; repeat-run variance was not measured.
- The 30 queries are English; Chinese and mixed-language judging are not evaluated.
- The judge reuses the same model id as the answer generator; self-evaluation bias is unquantified and the scores must not be treated as independent.

## 7. 复现

```bash
python scripts/evaluate_judge.py --provider real --judge-provider real --limit 3 --retrieval-mode live --allow-network --output-dir reports/judge --overwrite
```

## 8. 数据安全

- Judge 只读取 RAG artifact 中的问题、答案与证据；不修改数据库、`data/` 或 gold 标签。
- API Key 只从环境/.env 读取；artifact、日志和本报告只记录 `api_key_present` 布尔值。
- 失败评分不写入 0；均值/中位数/分布只使用成功评分。
