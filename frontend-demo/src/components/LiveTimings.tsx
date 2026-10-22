import type { ApiOutcome, AnswerResponse } from '../api/types';
import type { LastSearchAttempt } from '../hooks/useConversation';
import { formatMs, formatTimeOnly } from '../utils/format';

interface LiveTimingsProps {
  lastSearchAttempt: LastSearchAttempt | null;
  lastAnswerOutcome: ApiOutcome<AnswerResponse> | null;
}

const SEARCH_LABELS: Record<string, string> = {
  retrieval_ms: '检索阶段总计（retrieval_ms）',
  base_ms: '基础检索 Dense + Sparse + RRF（base_ms）',
  rerank_ms: 'Cross-Encoder 精排（rerank_ms）',
  total_ms: '服务端合计（total_ms）',
};

const ANSWER_LABELS: Record<string, string> = {
  retrieval_ms: '证据检索（retrieval_ms）',
  context_ms: '上下文组装（context_ms）',
  prompt_ms: 'Prompt 构建（prompt_ms）',
  llm_ms: 'LLM 调用（llm_ms）',
  total_ms: '服务端合计（total_ms）',
};

interface TimingListProps {
  timings: Record<string, number>;
  labels: Record<string, string>;
  emptyText: string;
}

function TimingList({ timings, labels, emptyText }: TimingListProps) {
  const knownKeys = Object.keys(labels);
  const orderedKeys = [
    ...knownKeys.filter((key) => key in timings),
    ...Object.keys(timings).filter((key) => !knownKeys.includes(key)),
  ];

  if (orderedKeys.length === 0) {
    return <p className="timing-empty">{emptyText}</p>;
  }

  return (
    <dl className="timing-list">
      {orderedKeys.map((key) => (
        <div className="timing-row" key={key}>
          <dt>{labels[key] ?? key}</dt>
          <dd>{formatMs(timings[key])}</dd>
        </div>
      ))}
    </dl>
  );
}

export function LiveTimings({ lastSearchAttempt, lastAnswerOutcome }: LiveTimingsProps) {
  const searchOutcome = lastSearchAttempt?.outcome ?? null;
  const searchData =
    searchOutcome?.kind === 'success' ? searchOutcome.data ?? null : null;
  const answerData = lastAnswerOutcome?.data ?? null;

  return (
    <div className="live-timings">
      <div className="timing-block">
        <div className="timing-block__head">
          <span className="timing-block__title">最近一次检索</span>
          {lastSearchAttempt ? (
            <span className="timing-block__meta">
              {formatTimeOnly(lastSearchAttempt.at)} · Top-K {lastSearchAttempt.topK}
            </span>
          ) : null}
        </div>

        {!lastSearchAttempt ? (
          <p className="timing-empty">尚未发起检索请求。</p>
        ) : (
          <>
            <p className="timing-query" title={lastSearchAttempt.query}>
              {lastSearchAttempt.query}
            </p>
            {searchData ? (
              <TimingList
                timings={searchData.timingsMs}
                labels={SEARCH_LABELS}
                emptyText="本次请求未返回阶段耗时。"
              />
            ) : (
              <p className="timing-empty">
                最近一次检索未成功（{searchOutcome?.kind ?? 'unknown'}），没有可用的服务端 timings_ms。
              </p>
            )}
            <dl className="timing-list">
              <div className="timing-row timing-row--browser">
                <dt>浏览器端到端（前端计时，含网络）</dt>
                <dd>{formatMs(searchOutcome?.durationMs)}</dd>
              </div>
            </dl>
          </>
        )}
      </div>

      <div className="timing-block">
        <div className="timing-block__head">
          <span className="timing-block__title">最新一轮生成</span>
          {lastAnswerOutcome?.httpStatus ? (
            <span className="timing-block__meta">HTTP {lastAnswerOutcome.httpStatus}</span>
          ) : null}
        </div>

        {!lastAnswerOutcome ? (
          <p className="timing-empty">本轮尚未收到 /answer 响应。</p>
        ) : answerData ? (
          <>
            <TimingList
              timings={answerData.timingsMs}
              labels={ANSWER_LABELS}
              emptyText="本次 /answer 未返回阶段耗时。"
            />
            <dl className="timing-list">
              <div className="timing-row timing-row--browser">
                <dt>浏览器端到端（前端计时，含网络）</dt>
                <dd>{formatMs(lastAnswerOutcome.durationMs)}</dd>
              </div>
            </dl>
          </>
        ) : (
          <>
            <p className="timing-empty">
              /answer 未返回可解析的 timings_ms（{lastAnswerOutcome.kind}）。
            </p>
            <dl className="timing-list">
              <div className="timing-row timing-row--browser">
                <dt>浏览器端到端（前端计时，含网络）</dt>
                <dd>{formatMs(lastAnswerOutcome.durationMs)}</dd>
              </div>
            </dl>
          </>
        )}
      </div>

      <p className="tech-note">
        所有阶段耗时按后端返回值原样展示，单位均为毫秒；retrieval_ms、base_ms、rerank_ms
        的口径由后端定义，前端不做相加，也不把相加结果宣称为 HTTP 端到端耗时。
      </p>
    </div>
  );
}
