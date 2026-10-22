import { outcomeErrorText } from '../api/client';
import type { AnswerResponse, ApiOutcome } from '../api/types';
import type { TurnStage } from '../hooks/useConversation';
import { formatMs } from '../utils/format';
import { AnswerEvidence } from './AnswerEvidence';
import { CitationList } from './CitationList';
import { LoadingLine } from './LoadingLine';
import { StatusBadge, type BadgeTone } from './StatusBadge';

interface AnswerPanelProps {
  outcome: ApiOutcome<AnswerResponse> | null;
  stage: TurnStage;
  skippedReason: string | null;
}

const RAG_STATUS_LABELS: Record<string, string> = {
  ok: '生成成功',
  insufficient_evidence: '证据不足',
  invalid_request: '请求无效',
  retrieval_error: '检索失败',
  rerank_error: '精排失败',
  evidence_error: '证据组装失败',
  config_error: '配置错误',
  llm_timeout: '生成超时',
  llm_rate_limited: '生成限流',
  llm_network_error: '生成网络错误',
  llm_server_error: '生成服务错误',
  llm_auth_error: '生成鉴权失败',
  llm_invalid_response: '生成响应无效',
  llm_request_error: '生成请求错误',
  llm_error: '生成失败',
};

function statusLabel(status: string): string {
  return RAG_STATUS_LABELS[status] ?? `状态 ${status}`;
}

function AnswerErrorDetails({ data }: { data: AnswerResponse }) {
  const error = data.error;
  if (!error) return null;
  return (
    <dl className="error-details">
      {error.stage ? (
        <>
          <dt>stage</dt>
          <dd>{error.stage}</dd>
        </>
      ) : null}
      {error.category ? (
        <>
          <dt>category</dt>
          <dd>{error.category}</dd>
        </>
      ) : null}
      {error.message ? (
        <>
          <dt>message</dt>
          <dd>{error.message}</dd>
        </>
      ) : null}
      <dt>retryable</dt>
      <dd>
        {error.retryable === null ? '未返回' : error.retryable ? 'true' : 'false'}
      </dd>
      {error.statusCode !== null ? (
        <>
          <dt>status_code</dt>
          <dd>{error.statusCode}</dd>
        </>
      ) : null}
      {error.attempts !== null ? (
        <>
          <dt>attempts</dt>
          <dd>{error.attempts}</dd>
        </>
      ) : null}
    </dl>
  );
}

function AnswerMeta({
  data,
  durationMs,
  httpStatus,
}: {
  data: AnswerResponse;
  durationMs: number;
  httpStatus?: number;
}) {
  const fields: Array<{ key: string; value: string }> = [
    { key: 'http', value: httpStatus !== undefined ? String(httpStatus) : '未返回' },
    { key: 'browser_e2e', value: formatMs(durationMs) },
  ];
  if (data.provider) fields.push({ key: 'provider', value: data.provider });
  if (data.model) fields.push({ key: 'model', value: data.model });
  if (data.promptVersion) fields.push({ key: 'prompt_version', value: data.promptVersion });
  const usage = data.usage;
  if (usage && usage.totalTokens !== null) {
    fields.push({
      key: 'tokens',
      value: `${usage.totalTokens}（prompt ${usage.promptTokens ?? '未返回'} / completion ${usage.completionTokens ?? '未返回'}）`,
    });
  }
  if (data.providerAttempted === false) {
    fields.push({ key: 'provider_attempted', value: 'false（未调用外部生成服务）' });
  }
  if (fields.length === 0) return null;
  return (
    <div className="answer-meta">
      {fields.map((field) => (
        <span key={field.key} className="answer-meta__item">
          <span className="answer-meta__key">{field.key}</span>
          <span className="answer-meta__value">{field.value}</span>
        </span>
      ))}
    </div>
  );
}

function PanelHead({ tone, badge }: { tone: BadgeTone; badge: string }) {
  return (
    <div className="answer-panel__head">
      <div className="answer-panel__heading">
        <span className="section-kicker">生成回答</span>
        <span className="endpoint-chip">POST /answer</span>
      </div>
      <StatusBadge tone={tone} label={badge} />
    </div>
  );
}

function OutcomeMeta({ outcome }: { outcome: ApiOutcome<unknown> }) {
  return (
    <div className="answer-meta">
      <span className="answer-meta__item">
        <span className="answer-meta__key">http</span>
        <span className="answer-meta__value">
          {outcome.httpStatus !== undefined ? String(outcome.httpStatus) : '未返回'}
        </span>
      </span>
      <span className="answer-meta__item">
        <span className="answer-meta__key">browser_e2e</span>
        <span className="answer-meta__value">{formatMs(outcome.durationMs)}</span>
      </span>
    </div>
  );
}

export function AnswerPanel({ outcome, stage, skippedReason }: AnswerPanelProps) {
  if (skippedReason) {
    return (
      <section className="answer-panel answer-panel--warning" aria-label="生成回答">
        <PanelHead tone="warning" badge="未调用 /answer" />
        <p className="answer-notice">
          <strong>本次没有生成式回答。</strong>
          {skippedReason}
        </p>
      </section>
    );
  }

  if (stage === 'searching' && !outcome) {
    return (
      <section className="answer-panel answer-panel--neutral" aria-label="生成回答">
        <PanelHead tone="neutral" badge="等待检索" />
        <p className="answer-notice muted">检索完成后将调用 /answer 生成回答。</p>
      </section>
    );
  }

  if (stage === 'generating') {
    return (
      <section className="answer-panel answer-panel--neutral" aria-label="生成回答">
        <PanelHead tone="warning" badge="生成中" />
        <LoadingLine label="正在生成回答…" detail="POST /answer · 检索证据已获取" />
      </section>
    );
  }

  if (!outcome) {
    return (
      <section className="answer-panel answer-panel--neutral" aria-label="生成回答">
        <PanelHead tone="neutral" badge="无生成结果" />
        <p className="answer-notice muted">未收到 /answer 的响应。</p>
      </section>
    );
  }

  if (outcome.kind === 'aborted') {
    return (
      <section className="answer-panel answer-panel--neutral" aria-label="生成回答">
        <PanelHead tone="neutral" badge="已取消" />
        <p className="answer-notice">请求已取消，本次没有生成式回答。</p>
        <OutcomeMeta outcome={outcome} />
      </section>
    );
  }

  if (outcome.kind === 'success' && outcome.data) {
    const data = outcome.data;

    if (data.status === 'ok') {
      return (
        <section className="answer-panel answer-panel--success" aria-label="生成回答">
          <PanelHead tone="success" badge={statusLabel(data.status)} />
          {data.answer.trim() ? (
            <div className="answer-text">{data.answer}</div>
          ) : (
            <p className="answer-notice answer-notice--warning">后端返回了空回答。</p>
          )}
          {data.insufficientEvidence === true ? (
            <p className="answer-notice answer-notice--warning">
              后端同时标记 insufficient_evidence = true。
            </p>
          ) : null}
          <CitationList
            citations={data.citations}
            invalidCitations={data.invalidCitations}
            citationStatus={data.citationStatus}
          />
          <AnswerMeta
            data={data}
            durationMs={outcome.durationMs}
            httpStatus={outcome.httpStatus}
          />
          <AnswerEvidence evidence={data.evidence} />
        </section>
      );
    }

    if (data.status === 'insufficient_evidence') {
      return (
        <section className="answer-panel answer-panel--warning" aria-label="生成回答">
          <PanelHead tone="warning" badge={statusLabel(data.status)} />
          <p className="answer-notice">
            <strong>后端判定证据不足。</strong>
            这是后端的检索/证据判定结果，不是生成式结论。
          </p>
          {data.answer.trim() ? <div className="answer-text">{data.answer}</div> : null}
          <CitationList
            citations={data.citations}
            invalidCitations={data.invalidCitations}
            citationStatus={data.citationStatus}
          />
          <AnswerMeta
            data={data}
            durationMs={outcome.durationMs}
            httpStatus={outcome.httpStatus}
          />
          <AnswerEvidence evidence={data.evidence} />
        </section>
      );
    }

    return (
      <section className="answer-panel answer-panel--danger" aria-label="生成回答">
        <PanelHead tone="danger" badge={statusLabel(data.status)} />
        <p className="answer-notice">
          <strong>本次没有生成式回答。</strong>
          后端返回状态 <code>{data.status}</code>
          {data.error?.message ? `：${data.error.message}` : '。'}
        </p>
        <AnswerErrorDetails data={data} />
        {data.invalidCitations.length > 0 ? (
          <p className="answer-notice answer-notice--warning">
            无效引用：{data.invalidCitations.join('、')}
          </p>
        ) : null}
        {data.evidence.length > 0 ? <AnswerEvidence evidence={data.evidence} /> : null}
        <OutcomeMeta outcome={outcome} />
      </section>
    );
  }

  const data = outcome.data;
  const httpLabel =
    outcome.kind === 'timeout'
      ? '请求超时'
      : outcome.kind === 'network_error'
        ? '网络错误'
        : outcome.kind === 'parse_error'
          ? outcome.httpStatus !== undefined
            ? `HTTP ${outcome.httpStatus}（解析失败）`
            : '响应解析失败'
          : outcome.httpStatus !== undefined
            ? `HTTP ${outcome.httpStatus}`
            : '请求失败';
  const reason = data?.error?.message || outcomeErrorText(outcome);

  return (
    <section className="answer-panel answer-panel--danger" aria-label="生成回答">
      <PanelHead tone="danger" badge={httpLabel} />
      <p className="answer-notice">
        <strong>本次没有生成式回答。</strong>
        {reason}
      </p>
      {data?.status ? (
        <p className="answer-notice answer-notice--subtle">
          后端状态：<code>{data.status}</code>
        </p>
      ) : null}
      {data?.error ? <AnswerErrorDetails data={data} /> : null}
      {data && data.evidence.length > 0 ? <AnswerEvidence evidence={data.evidence} /> : null}
      <OutcomeMeta outcome={outcome} />
    </section>
  );
}
