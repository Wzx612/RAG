import { BACKEND_DISPLAY_ADDRESS } from '../api/config';
import type { ApiOutcome, HealthResponse, ReadinessResponse } from '../api/types';
import type { ConnectionState } from '../hooks/useBackendStatus';
import { AlertIcon } from './icons';

interface ConnectionBannerProps {
  connection: ConnectionState;
  healthOutcome: ApiOutcome<HealthResponse> | null;
  readinessOutcome: ApiOutcome<ReadinessResponse> | null;
  onRetry: () => void;
  onOpenTech: () => void;
}

export function ConnectionBanner({
  connection,
  healthOutcome,
  readinessOutcome,
  onRetry,
  onOpenTech,
}: ConnectionBannerProps) {
  if (connection === 'connected' || connection === 'checking') return null;

  const unreachable = connection === 'unreachable';
  const degraded = connection === 'degraded';
  const title = unreachable
    ? '无法连接后端'
    : degraded
      ? '后端进程可访问，但尚未就绪'
      : '后端健康接口异常';

  const detail = unreachable
    ? `无法访问 ${BACKEND_DISPLAY_ADDRESS}。请确认 Docker Compose 的 db 与 api 服务已启动；后端不可用时，页面不会展示任何模拟检索结果或模拟回答。`
    : degraded
      ? 'GET /readyz 未通过，数据库或依赖可能仍在启动。检索与生成可能失败；已返回的结果仍会按真实响应展示。'
      : 'GET /healthz 未返回预期响应，请检查后端日志与代理配置。';

  const backendMessage =
    healthOutcome?.message ?? readinessOutcome?.message ?? null;

  const readinessChecks = degraded ? readinessOutcome?.data?.checks ?? [] : [];

  return (
    <section
      className={`connection-banner connection-banner--${connection}`}
      role="alert"
      aria-live="assertive"
    >
      <span className="connection-banner__icon" aria-hidden="true">
        <AlertIcon />
      </span>
      <div className="connection-banner__content">
        <strong className="connection-banner__title">{title}</strong>
        <p className="connection-banner__detail">{detail}</p>
        {backendMessage ? (
          <p className="connection-banner__message">
            最近一次错误：<code>{backendMessage}</code>
          </p>
        ) : null}
        {readinessChecks.length > 0 ? (
          <ul className="connection-banner__checks">
            {readinessChecks.map((check) => (
              <li
                key={check.name}
                className={check.ok === true ? 'is-ok' : check.ok === false ? 'is-fail' : ''}
              >
                <span aria-hidden="true">
                  {check.ok === true ? '✓' : check.ok === false ? '✕' : '·'}
                </span>
                {check.name}
                {check.ok === null ? <span className="muted">（ok 未返回）</span> : null}
                {check.detail ? <span className="muted">（{check.detail}）</span> : null}
              </li>
            ))}
          </ul>
        ) : null}
      </div>
      <div className="connection-banner__actions">
        <button type="button" className="btn btn--light btn--sm" onClick={onRetry}>
          重新检查
        </button>
        <button type="button" className="btn btn--light btn--sm" onClick={onOpenTech}>
          技术详情
        </button>
      </div>
    </section>
  );
}
