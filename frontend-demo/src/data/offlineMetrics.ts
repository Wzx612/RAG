/**
 * 静态离线实验数据，逐字取自 reports/unified/rerank_20to5.md。
 * 这些数字不是当前请求的实时结果，界面必须始终这样标注。
 */
export const OFFLINE_SOURCE = 'reports/unified/rerank_20to5.md';

export const OFFLINE_DISCLAIMER = '离线实验报告，不代表当前这次请求的实时结果。';

export const OFFLINE_SETUP: string[] = [
  '30 条评测查询（报告：Queries 30 / expected 30）',
  'Top-K = 5，candidate_k = 20，RRF k = 60',
  'Chunk strategy = document；指标按文档级计算（chunk 按 doc_id 去重）',
  'RRF 与 RRF + Cross-Encoder Reranker 对比',
];

export interface OfflineMetricRow {
  label: string;
  rrf: string;
  rerank: string;
}

export const OFFLINE_METRICS: OfflineMetricRow[] = [
  { label: 'Recall macro', rrf: '0.9556', rerank: '1.0000' },
  { label: 'Recall micro', rrf: '0.9070', rerank: '1.0000' },
  { label: 'P@K macro', rrf: '0.2600', rerank: '0.2867' },
  { label: 'P@K ceiling（固定 K）', rrf: '0.2867', rerank: '0.2867' },
  { label: 'nDCG@K', rrf: '0.9204', rerank: '0.9777' },
  { label: 'MRR', rrf: '0.9500', rerank: '0.9833' },
  { label: 'Hit@1', rrf: '27 / 30', rerank: '29 / 30' },
];

export const OFFLINE_LATENCY: OfflineMetricRow[] = [
  { label: 'P50', rrf: '89.4622 ms', rerank: '1790.9155 ms' },
  { label: 'P95', rrf: '109.5530 ms', rerank: '2038.5050 ms' },
  { label: 'P99', rrf: '112.8697 ms', rerank: '4683.3748 ms' },
  { label: 'Max', rrf: '112.8697 ms', rerank: '4683.3748 ms' },
];

export const OFFLINE_LIMITATIONS: string[] = [
  '仅评估检索与精排，不评估最终生成答案的质量。',
  'gold 文档由评测集给出且非穷尽，未列出的相关文档不会被记为命中。',
  'P@K 的分母固定为 K，Recall 为 1.0000 不代表线上生产准确率。',
  '延迟为单进程顺序执行的预热后测量，不等同于并发或线上服务性能。',
];
