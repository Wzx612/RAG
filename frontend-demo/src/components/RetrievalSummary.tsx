import { outcomeErrorText } from '../api/client';
import type { ApiOutcome, SearchResponse } from '../api/types';
import { formatInt, formatMs, textOrMissing } from '../utils/format';

interface RetrievalSummaryProps {
  outcome: ApiOutcome<SearchResponse>;
  requestedTopK: number;
}

export function RetrievalSummary({ outcome, requestedTopK }: RetrievalSummaryProps) {
  if (outcome.kind === 'success' && outcome.data) {
    const data = outcome.data;
    const top1 = data.results.find((item) => item.rank === 1) ?? data.results[0];
    const countMismatch =
      data.hitCount !== null && data.hitCount !== data.results.length;
    const serverRetrieval = data.timingsMs.retrieval_ms;
    const serverTotal = data.timingsMs.total_ms;

    return (
      <div className="retrieval-summary">
        <div className="metric-grid">
          <div className="metric">
            <span className="metric__label">命中文档</span>
            <span className="metric__value">{formatInt(data.hitCount)}</span>
          </div>
          <div className="metric">
            <span className="metric__label">结果数组</span>
            <span className="metric__value">{formatInt(data.results.length)}</span>
          </div>
          <div className="metric">
            <span className="metric__label">实际 Top-K</span>
            <span className="metric__value">{formatInt(data.topK)}</span>
          </div>
          <div className="metric">
            <span className="metric__label">retriever</span>
            <span className="metric__value metric__value--text">
              {textOrMissing(data.retriever)}
            </span>
          </div>
          <div className="metric">
            <span className="metric__label">HTTP</span>
            <span className="metric__value">
              {outcome.httpStatus !== undefined ? outcome.httpStatus : '未返回'}
            </span>
          </div>
          <div className="metric">
            <span className="metric__label">浏览器端到端</span>
            <span className="metric__value">{formatMs(outcome.durationMs)}</span>
          </div>
          <div className="metric">
            <span className="metric__label">服务端 retrieval_ms</span>
            <span className="metric__value">{formatMs(serverRetrieval)}</span>
          </div>
          {serverTotal !== undefined ? (
            <div className="metric">
              <span className="metric__label">服务端 total_ms</span>
              <span className="metric__value">{formatMs(serverTotal)}</span>
            </div>
          ) : null}
        </div>

        {countMismatch ? (
          <p className="inline-warning">
            hit_count（{data.hitCount}）与 results.length（{data.results.length}）不一致；
            页面按实际返回的 results 数组展示。
          </p>
        ) : null}

        {data.topK !== null && data.topK !== requestedTopK ? (
          <p className="inline-warning">
            请求的 Top-K（{requestedTopK}）与后端实际生效值（{data.topK}）不同，页面以实际
            返回值为准。
          </p>
        ) : null}

        {data.results.length === 0 ? (
          <div className="empty-retrieval" role="status">
            <strong>本次查询未返回匹配文档。你可以尝试使用更具体的卡种名称、政策编号或问题描述。</strong>
            <p>这不代表知识库中一定不存在相关文档；检索 Top-K 为 {formatInt(data.topK)}。</p>
          </div>
        ) : (
          <div className="top1-line">
            <span className="tag tag--top1">Top-1</span>
            <span className="top1-line__ids">
              {textOrMissing(top1?.docId)}
              {top1?.chunkId ? ` · ${top1.chunkId}` : ''}
            </span>
            {top1?.title ? <span className="top1-line__title">{top1.title}</span> : null}
          </div>
        )}

        <p className="retrieval-summary__note">
          请求的 Top-K = {requestedTopK}。检索成功不代表结果一定相关；服务端 retrieval_ms
          与浏览器端到端耗时的口径不同，不做相加比较。
        </p>
      </div>
    );
  }

  const label =
    outcome.kind === 'timeout'
      ? '检索请求超时'
      : outcome.kind === 'network_error'
        ? '检索请求失败'
        : outcome.kind === 'aborted'
          ? '检索请求已取消'
          : outcome.kind === 'parse_error'
            ? '检索响应解析失败'
            : outcome.httpStatus
              ? `检索接口返回 HTTP ${outcome.httpStatus}`
              : '检索失败';

  return (
    <div className="retrieval-summary">
      <div className="search-error" role="alert">
        <strong>{label}。</strong>
        <p>{outcomeErrorText(outcome)}</p>
        <p>
          浏览器端到端：{formatMs(outcome.durationMs)}
          {outcome.httpStatus !== undefined ? ` · HTTP ${outcome.httpStatus}` : ' · 无 HTTP 状态'}
        </p>
        {outcome.httpStatus === 422 ? (
          <p>请求参数未通过校验。可尝试降低 Top-K 或缩短查询内容后重试。</p>
        ) : null}
      </div>
    </div>
  );
}
