import { BACKEND_DISPLAY_ADDRESS } from '../api/config';
import type { ConnectionState } from '../hooks/useBackendStatus';
import { formatTimeOnly } from '../utils/format';
import { BrandMark, RefreshIcon } from './icons';

interface AppHeaderProps {
  connection: ConnectionState;
  isChecking: boolean;
  lastCheckedAt: Date | null;
  onRefresh: () => void;
  onOpenTech: () => void;
}

const CONNECTION_LABELS: Record<ConnectionState, string> = {
  checking: '连接检查中',
  connected: '后端已连接',
  degraded: '就绪检查未通过',
  unreachable: '无法连接后端',
  error: '健康接口异常',
};

export function AppHeader({
  connection,
  isChecking,
  lastCheckedAt,
  onRefresh,
  onOpenTech,
}: AppHeaderProps) {
  return (
    <header className="app-header">
      <div className="app-header__inner">
        <div className="brand">
          <BrandMark className="brand__icon" />
          <div className="brand__text">
            <span className="brand__title">信用卡智能客服助手</span>
            <span className="brand__subtitle">Credit Card AI Assistant · Powered by RAG</span>
          </div>
        </div>

        <div className="app-header__actions">
          <div
            className={`conn-status conn-status--${connection}`}
            role="status"
            aria-live="polite"
            title={
              lastCheckedAt
                ? `最后检查：${formatTimeOnly(lastCheckedAt)}`
                : '正在检查后端连接'
            }
          >
            <span className="conn-status__dot" aria-hidden="true" />
            <span className="conn-status__label">{CONNECTION_LABELS[connection]}</span>
          </div>

          <span className="backend-address" title="后端服务地址">
            <span className="backend-address__key">后端</span>
            <code>{BACKEND_DISPLAY_ADDRESS}</code>
          </span>

          <button
            type="button"
            className="btn btn--ghost btn--sm"
            onClick={onRefresh}
            disabled={isChecking}
            title="重新检查 /healthz、/readyz 与验收摘要"
          >
            <RefreshIcon className={isChecking ? 'spin' : undefined} />
            {isChecking ? '检查中…' : '重新检查'}
          </button>

          <button
            type="button"
            className="btn btn--navy btn--sm header-tech-button"
            onClick={onOpenTech}
          >
            技术详情
          </button>
        </div>
      </div>
    </header>
  );
}
