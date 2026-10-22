import { outcomeErrorText } from '../api/client';
import { API_BASE_URL } from '../api/config';
import type { ApiOutcome, AcceptanceSummary } from '../api/types';
import {
  formatDateTime,
  formatInt,
  isSafeExternalUrl,
  textOrMissing,
} from '../utils/format';
import { ExternalLinkIcon } from './icons';
import { LoadingLine } from './LoadingLine';
import { StatusBadge, type BadgeTone } from './StatusBadge';

interface AcceptancePanelProps {
  outcome: ApiOutcome<AcceptanceSummary> | null;
  loading: boolean;
  onRetry: () => void;
}

function statusTone(status: string | null | undefined): BadgeTone {
  if (!status) return 'neutral';
  const normalized = status.toUpperCase();
  if (normalized === 'PASS') return 'success';
  if (normalized === 'FAIL') return 'danger';
  if (normalized === 'PARTIAL' || normalized === 'WARN') return 'warning';
  return 'neutral';
}

function checkTone(state: string): BadgeTone {
  const normalized = state.toLowerCase();
  if (normalized === 'pass') return 'success';
  if (normalized === 'fail') return 'danger';
  if (normalized === 'warn') return 'warning';
  return 'neutral';
}

const CHECK_ICONS: Record<string, string> = {
  pass: '✓',
  fail: '✕',
  warn: '!',
  skip: '–',
};

function boolText(value: boolean | null): string {
  if (value === null) return '未返回';
  return value ? '是' : '否';
}

export function AcceptancePanel({ outcome, loading, onRetry }: AcceptancePanelProps) {
  const backendOrigin = API_BASE_URL || 'http://localhost:8000';
  const linksEnabled = isSafeExternalUrl(`${backendOrigin}/acceptance/`);

  if (loading && !outcome) {
    return <LoadingLine label="正在读取验收摘要…" detail="GET /acceptance/api/summary" />;
  }

  const data = outcome?.kind === 'success' ? (outcome.data ?? null) : null;

  if (!outcome || !data) {
    return (
      <div className="acceptance-unavailable" role="status">
        <strong>验收摘要暂不可用</strong>
        <p>{outcome ? outcomeErrorText(outcome) : '尚未获取验收摘要。'}</p>
        <p className="muted">
          验收摘要只读取离线 artifact，不影响聊天与检索。可稍后重试或直接打开验收看板。
        </p>
        <div className="acceptance-unavailable__actions">
          <button
            type="button"
            className="btn btn--navy btn--sm"
            onClick={onRetry}
            disabled={loading}
          >
            {loading ? '读取中…' : '重新读取验收摘要'}
          </button>
          {linksEnabled ? (
            <>
              <a
                className="btn btn--ghost btn--sm"
                href={`${backendOrigin}/acceptance/`}
                target="_blank"
                rel="noreferrer noopener"
              >
                验收看板
                <ExternalLinkIcon />
              </a>
              <a
                className="btn btn--ghost btn--sm"
                href={`${backendOrigin}/docs`}
                target="_blank"
                rel="noreferrer noopener"
              >
                API 文档
                <ExternalLinkIcon />
              </a>
            </>
          ) : null}
        </div>
      </div>
    );
  }

  const environmentRows: Array<{ label: string; value: string }> = [
    { label: 'python_version', value: textOrMissing(data.environment.pythonVersion) },
    { label: 'postgres_version', value: textOrMissing(data.environment.postgresVersion) },
    { label: 'vector_version', value: textOrMissing(data.environment.vectorVersion) },
    { label: 'embedding_model', value: textOrMissing(data.environment.embeddingModel) },
    { label: 'embedding_device', value: textOrMissing(data.environment.embeddingDevice) },
    {
      label: 'embedding_dimensions',
      value: data.environment.embeddingDimensions
        ? data.environment.embeddingDimensions.join(', ')
        : '未返回',
    },
    { label: 'documents', value: formatInt(data.environment.documents) },
    { label: 'chunks', value: formatInt(data.environment.chunks) },
    { label: 'reranker_model', value: textOrMissing(data.environment.rerankerModel) },
    {
      label: 'execution_environment',
      value: textOrMissing(data.environment.executionEnvironment),
    },
  ];

  return (
    <div className="acceptance-panel">
      <div className="acceptance-status">
        <StatusBadge
          tone={statusTone(data.status)}
          label={data.status ?? '未返回状态'}
          dot
        />
        <div className="acceptance-status__meta">
          <span>generated_at：{formatDateTime(data.generatedAt)}</span>
          <span>read_only：{data.readOnly === null ? '未返回' : String(data.readOnly)}</span>
        </div>
      </div>

      <p className="acceptance-reason">
        {data.statusReason ? data.statusReason : '后端未返回 status_reason。'}
      </p>

      <p className="tech-note tech-note--warning">
        PASS 仅代表该摘要描述的 artifact 检查通过，不等同于生产认证、安全审计或大规模并发性能验证。
        完整看板：
        {linksEnabled ? (
          <>
            {' '}
            <a href={`${backendOrigin}/acceptance/`} target="_blank" rel="noreferrer noopener">
              /acceptance/
              <ExternalLinkIcon />
            </a>
            {' · '}
            <a href={`${backendOrigin}/docs`} target="_blank" rel="noreferrer noopener">
              /docs
              <ExternalLinkIcon />
            </a>
          </>
        ) : (
          ' 链接不可用（后端地址配置无效）'
        )}
      </p>

      <div className="acceptance-block">
        <h3 className="acceptance-block__title">检查项（{data.checks.length}）</h3>
        <ul className="acceptance-checks">
          {data.checks.map((check) => (
            <li key={check.id} className={`acceptance-check acceptance-check--${checkTone(check.state)}`}>
              <span className="acceptance-check__icon" aria-hidden="true">
                {CHECK_ICONS[check.state.toLowerCase()] ?? '·'}
              </span>
              <span className="acceptance-check__body">
                <span className="acceptance-check__label">{check.label}</span>
                <span className="acceptance-check__detail">{check.detail}</span>
              </span>
              <span className="acceptance-check__state">{check.state}</span>
            </li>
          ))}
        </ul>
      </div>

      <details className="tech-details" open>
        <summary>Artifact 状态（{data.artifacts.length}）</summary>
        <div className="table-scroll">
          <table className="metrics-table acceptance-artifacts">
            <thead>
              <tr>
                <th scope="col">Artifact</th>
                <th scope="col">必需</th>
                <th scope="col">存在</th>
                <th scope="col">状态</th>
              </tr>
            </thead>
            <tbody>
              {data.artifacts.map((artifact) => (
                <tr key={artifact.key}>
                  <th scope="row">
                    <span className="artifact-label">{artifact.label}</span>
                    <code className="artifact-key" title={artifact.path ?? undefined}>
                      {artifact.key}
                    </code>
                  </th>
                  <td>{boolText(artifact.required)}</td>
                  <td>{boolText(artifact.present)}</td>
                  <td>
                    <StatusBadge
                      tone={statusTone(artifact.status)}
                      label={
                        artifact.status ??
                        (artifact.present === false ? '缺失' : '未返回')
                      }
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      <details className="tech-details">
        <summary>数据集与环境</summary>
        <dl className="acceptance-dataset">
          <div>
            <dt>queries</dt>
            <dd>{formatInt(data.dataset.queries)}</dd>
          </div>
          <div>
            <dt>eval_queries_on_disk</dt>
            <dd>{formatInt(data.dataset.evalQueriesOnDisk)}</dd>
          </div>
          <div>
            <dt>gold_docs</dt>
            <dd>{formatInt(data.dataset.goldDocs)}</dd>
          </div>
        </dl>
        <dl className="metadata-grid">
          {environmentRows.map((row) => (
            <div key={row.label}>
              <dt>{row.label}</dt>
              <dd>{row.value}</dd>
            </div>
          ))}
        </dl>
      </details>

      {data.limitations.length > 0 ? (
        <details className="tech-details">
          <summary>Artifact 声明的限制（{data.limitations.length}）</summary>
          <ul className="limitation-list">
            {data.limitations.map((item, index) => (
              <li key={`${item.source}-${index}`}>
                <span className="tag tag--neutral">{item.source}</span>
                {item.text}
              </li>
            ))}
          </ul>
        </details>
      ) : null}

      {data.caveats.length > 0 ? (
        <details className="tech-details">
          <summary>Caveats（{data.caveats.length}）</summary>
          <ul className="limitation-list">
            {data.caveats.map((item, index) => (
              <li key={`${item}-${index}`}>{item}</li>
            ))}
          </ul>
        </details>
      ) : null}

      {data.artifacts.some((artifact) => artifact.synthetic === true) ? (
        <p className="inline-warning">
          该摘要包含 synthetic = true 的 artifact，请勿将其视为真实数据。
        </p>
      ) : null}

      <button
        type="button"
        className="btn btn--ghost btn--sm acceptance-refresh"
        onClick={onRetry}
        disabled={loading}
      >
        {loading ? '读取中…' : '重新读取验收摘要'}
      </button>
    </div>
  );
}
