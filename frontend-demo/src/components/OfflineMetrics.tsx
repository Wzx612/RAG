import {
  OFFLINE_DISCLAIMER,
  OFFLINE_LATENCY,
  OFFLINE_LIMITATIONS,
  OFFLINE_METRICS,
  OFFLINE_SETUP,
  OFFLINE_SOURCE,
} from '../data/offlineMetrics';
import { AlertIcon } from './icons';

export function OfflineMetrics() {
  return (
    <div className="offline-metrics">
      <p className="offline-warning" role="note">
        <AlertIcon />
        <strong>{OFFLINE_DISCLAIMER}</strong>
      </p>

      <p className="offline-source">
        来源：<code>{OFFLINE_SOURCE}</code>
      </p>
      <ul className="offline-setup">
        {OFFLINE_SETUP.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>

      <div className="table-scroll">
        <table className="metrics-table">
          <caption className="sr-only">RRF 与 RRF + Reranker 离线指标对比</caption>
          <thead>
            <tr>
              <th scope="col">指标</th>
              <th scope="col">RRF</th>
              <th scope="col">RRF + Reranker</th>
            </tr>
          </thead>
          <tbody>
            {OFFLINE_METRICS.map((row) => (
              <tr key={row.label}>
                <th scope="row">{row.label}</th>
                <td>{row.rrf}</td>
                <td>{row.rerank}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <details className="tech-details">
        <summary>离线延迟分位（同一报告）</summary>
        <div className="table-scroll">
          <table className="metrics-table metrics-table--compact">
            <thead>
              <tr>
                <th scope="col">分位</th>
                <th scope="col">RRF</th>
                <th scope="col">RRF + Reranker</th>
              </tr>
            </thead>
            <tbody>
              {OFFLINE_LATENCY.map((row) => (
                <tr key={row.label}>
                  <th scope="row">{row.label}</th>
                  <td>{row.rrf}</td>
                  <td>{row.rerank}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      <details className="tech-details">
        <summary>报告声明的限制</summary>
        <ul className="limitation-list">
          {OFFLINE_LIMITATIONS.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </details>

      <p className="tech-note">
        这些指标仅针对 30 条英文评测查询，Recall 1.0000 不代表线上准确率，也不能与本次请求的
        timings_ms 直接比较。
      </p>
    </div>
  );
}
