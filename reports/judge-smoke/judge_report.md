# LLM-as-a-Judge 评测

- Status: **PARTIAL**
- Reason: mock judge completed; scores are deterministic fixtures
- Run ID: `judge-20261020T052218Z-39abaaca`
- Generated: 2026-10-20T05:22:18.187931+00:00
- Queries judged: **3 / 30**
- Not run: **27** (NOT RUN, not scored and not counted as zero)
- RAG provider: `mock` / `mock` / model `mock-evidence-v1`
- Judge provider: `mock` / `mock_judge` / model `mock-judge-v1`
- Same model as RAG: **False**
- Prompt versions: `{'context_relevance': 'judge-context-relevance-v1', 'faithfulness': 'judge-faithfulness-v1'}`
- Score aggregation: only successful scores enter mean/median/distribution; failures are reported separately and never scored as 0.

> **MOCK JUDGE RUN.** Scores are deterministic fixtures, not a model's judgement. This report validates wiring, parsing, failure isolation and aggregation only.

## 1. 指标汇总

| Metric | Attempted | Success | Failed | Success rate | Valid n | Mean | Median | Low score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Context Relevance | 3 | 3 | 0 | 1.0000 | 3 | 4.0000 | 4.0000 | 0 |
| Faithfulness | 3 | 3 | 0 | 1.0000 | 3 | 4.0000 | 4.0000 | 0 |

## 2. 分数分布（仅成功评分）

| Metric | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| Context Relevance | 0 | 0 | 0 | 3 | 0 |
| Faithfulness | 0 | 0 | 0 | 3 | 0 |

## 3. 低分样例

- 没有低于阈值的成功评分。

## 4. 失败分类

- 没有 Judge 失败。

## 5. 逐查询结果

| Query | RAG status | Context Relevance | Faithfulness | Unsupported claims |
|---|---|---|---|---:|
| Q-001 | ok | success (4) | success (4) | 0 |
| Q-002 | ok | success (4) | success (4) | 0 |
| Q-003 | ok | success (4) | success (4) | 0 |

## 6. 限制

- The RAG answers are mock output, so judged answer quality is not real model output.
- The judge is a deterministic mock. Its scores validate the pipeline, not the quality of a real model's judgement.
- Only 3 of 30 queries were judged; this smoke run does not describe the full evaluation set.
- Judge scores are model-generated judgements, not ground truth; no human review or inter-judge agreement was performed.
- Context Relevance scores the supplied evidence only; it is not retrieval Recall and does not show that the right document was retrieved.
- Faithfulness uses only the evidence actually sent to the answer model. A claim that is true but absent from that evidence is still unsupported here.
- The judge may be sensitive to prompt order, context length and language; repeat-run variance was not measured.
- The 30 queries are English; Chinese and mixed-language judging are not evaluated.

## 7. 复现

```bash
python scripts/evaluate_judge.py --provider mock --judge-provider mock --limit 3 --output-dir reports/judge-smoke
```

## 8. 数据安全

- Judge 只读取 RAG artifact 中的问题、答案与证据；不修改数据库、`data/` 或 gold 标签。
- API Key 只从环境/.env 读取；artifact、日志和本报告只记录 `api_key_present` 布尔值。
- 失败评分不写入 0；均值/中位数/分布只使用成功评分。
