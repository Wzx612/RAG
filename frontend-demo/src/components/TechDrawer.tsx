import {
  useEffect,
  useRef,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from 'react';
import { API_BASE_URL, BACKEND_DISPLAY_ADDRESS } from '../api/config';
import type { AnswerResponse, ApiOutcome } from '../api/types';
import type { BackendStatus } from '../hooks/useBackendStatus';
import type { LastSearchAttempt } from '../hooks/useConversation';
import { formatDateTime, formatInt, formatMs, formatTimeOnly } from '../utils/format';
import { AcceptancePanel } from './AcceptancePanel';
import { CloseIcon } from './icons';
import { LiveTimings } from './LiveTimings';
import { OfflineMetrics } from './OfflineMetrics';
import { StatusBadge, type BadgeTone } from './StatusBadge';
import { TechArchitecture } from './TechArchitecture';

interface TechDrawerProps {
  open: boolean;
  onClose: () => void;
  backend: BackendStatus;
  lastSearchAttempt: LastSearchAttempt | null;
  lastAnswerOutcome: ApiOutcome<AnswerResponse> | null;
}

function outcomeTone(outcome: ApiOutcome<unknown> | null): BadgeTone {
  if (!outcome) return 'neutral';
  if (outcome.kind === 'success') return 'success';
  if (outcome.kind === 'http_error' || outcome.kind === 'network_error') return 'danger';
  if (outcome.kind === 'timeout') return 'warning';
  if (outcome.kind === 'parse_error') return 'danger';
  return 'neutral';
}

function outcomeLabel(outcome: ApiOutcome<unknown> | null): string {
  if (!outcome) return '未检查';
  if (outcome.kind === 'success') {
    return `成功 · HTTP ${outcome.httpStatus ?? '未返回'} · ${formatMs(outcome.durationMs)}`;
  }
  if (outcome.kind === 'http_error') {
    return `HTTP ${outcome.httpStatus ?? '?'} · ${formatMs(outcome.durationMs)}`;
  }
  if (outcome.kind === 'network_error') return '网络错误';
  if (outcome.kind === 'timeout') return '请求超时';
  if (outcome.kind === 'parse_error') return '响应解析失败';
  return '已取消';
}

function EndpointRow({
  endpoint,
  outcome,
}: {
  endpoint: string;
  outcome: ApiOutcome<unknown> | null;
}) {
  return (
    <div className="endpoint-row">
      <code className="endpoint-row__path">{endpoint}</code>
      <StatusBadge tone={outcomeTone(outcome)} label={outcomeLabel(outcome)} />
      {outcome?.httpStatus ? (
        <span className="endpoint-row__http">HTTP {outcome.httpStatus}</span>
      ) : null}
    </div>
  );
}

interface TechSectionProps {
  kicker: string;
  title: string;
  children: ReactNode;
}

function TechSection({ kicker, title, children }: TechSectionProps) {
  return (
    <section className="tech-section">
      <div className="tech-section__head">
        <span className="tech-section__kicker">{kicker}</span>
        <h3 className="tech-section__title">{title}</h3>
      </div>
      {children}
    </section>
  );
}

export function TechDrawer({
  open,
  onClose,
  backend,
  lastSearchAttempt,
  lastAnswerOutcome,
}: TechDrawerProps) {
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const dialogRef = useRef<HTMLElement | null>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!open) return;
    const previousActive =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    closeRef.current?.focus();

    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onCloseRef.current();
      }
    };
    document.addEventListener('keydown', handleKey);

    return () => {
      document.removeEventListener('keydown', handleKey);
      document.body.style.overflow = previousOverflow;
      previousActive?.focus();
    };
  }, [open]);

  if (!open) return null;

  const handleTab = (event: ReactKeyboardEvent<HTMLElement>) => {
    if (event.key !== 'Tab') return;
    const root = dialogRef.current;
    if (!root) return;
    const focusables = Array.from(
      root.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), textarea, input, select, summary, [tabindex]:not([tabindex="-1"])',
      ),
    ).filter(
      (element) =>
        !element.hasAttribute('hidden') && element.getClientRects().length > 0,
    );
    if (focusables.length === 0) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  const searchOutcome = lastSearchAttempt?.outcome ?? null;
  const searchData =
    searchOutcome?.kind === 'success' ? searchOutcome.data ?? null : null;
  const readinessChecks = backend.readinessOutcome?.data?.checks ?? [];

  return (
    <div className="drawer-layer">
      <div className="drawer-backdrop" onClick={onClose} aria-hidden="true" />
      <aside
        className="tech-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="tech-drawer-title"
        ref={dialogRef}
        onKeyDown={handleTab}
      >
        <header className="tech-drawer__head">
          <div>
            <span className="tech-drawer__kicker">技术详情</span>
            <h2 className="tech-drawer__title" id="tech-drawer-title">
              RAG 检索链路与工程状态
            </h2>
          </div>
          <button
            type="button"
            className="icon-button"
            onClick={onClose}
            aria-label="关闭技术详情"
            ref={closeRef}
          >
            <CloseIcon />
          </button>
        </header>

        <div className="tech-drawer__body">
          <TechSection kicker="实时" title="连接与最近一次请求">
            <dl className="tech-kv">
              <div>
                <dt>后端地址</dt>
                <dd>
                  <code>{BACKEND_DISPLAY_ADDRESS}</code>
                </dd>
              </div>
              <div>
                <dt>请求路径</dt>
                <dd>
                  {API_BASE_URL
                    ? `直连 ${BACKEND_DISPLAY_ADDRESS}（需后端配置 CORS）`
                    : '经 Vite 同源代理转发：/healthz、/readyz、/api、/answer、/acceptance'}
                </dd>
              </div>
              <div>
                <dt>最后检查</dt>
                <dd>{backend.lastCheckedAt ? formatDateTime(backend.lastCheckedAt) : '尚未完成'}</dd>
              </div>
            </dl>

            <div className="endpoint-list">
              <EndpointRow endpoint="GET /healthz" outcome={backend.healthOutcome} />
              <EndpointRow endpoint="GET /readyz" outcome={backend.readinessOutcome} />
            </div>

            {readinessChecks.length > 0 ? (
              <ul className="endpoint-checks">
                {readinessChecks.map((check) => (
                  <li
                    key={check.name}
                    className={check.ok === true ? 'is-ok' : check.ok === false ? 'is-fail' : ''}
                  >
                    <span aria-hidden="true">
                      {check.ok === true ? '✓' : check.ok === false ? '✕' : '·'}
                    </span>
                    {check.name}
                    {check.ok === null ? (
                      <span className="muted"> · ok 未返回</span>
                    ) : null}
                    {check.detail ? <span className="muted"> · {check.detail}</span> : null}
                  </li>
                ))}
              </ul>
            ) : null}

            <div className="latest-search">
              <span className="latest-search__title">最近一次检索</span>
              {lastSearchAttempt ? (
                <>
                  <p className="latest-search__query">{lastSearchAttempt.query}</p>
                  <div className="latest-search__meta">
                    <span>{formatTimeOnly(lastSearchAttempt.at)}</span>
                    <span>请求 Top-K {lastSearchAttempt.topK}</span>
                    {searchData ? (
                      <>
                        <span>实际 Top-K {formatInt(searchData.topK)}</span>
                        <span>命中 {formatInt(searchData.hitCount)}</span>
                        <span>retriever {searchData.retriever || '未返回'}</span>
                      </>
                    ) : (
                      <span>未获得可解析的检索响应</span>
                    )}
                  </div>
                </>
              ) : (
                <p className="muted">尚未发起检索请求。</p>
              )}
            </div>

            <p className="tech-note">
              /healthz 只代表后端进程存活，不代表数据库、向量索引或模型可用；就绪状态请以
              /readyz 与真实检索结果为准。
            </p>
          </TechSection>

          <TechSection kicker="静态说明" title="检索与生成架构">
            <TechArchitecture />
          </TechSection>

          <TechSection kicker="实时" title="本次请求耗时">
            <LiveTimings
              lastSearchAttempt={lastSearchAttempt}
              lastAnswerOutcome={lastAnswerOutcome}
            />
          </TechSection>

          <TechSection kicker="离线" title="离线实验数据">
            <OfflineMetrics />
          </TechSection>

          <TechSection kicker="只读摘要" title="工程验收状态">
            <AcceptancePanel
              outcome={backend.acceptance}
              loading={backend.acceptanceLoading}
              onRetry={backend.refreshAcceptance}
            />
          </TechSection>
        </div>
      </aside>
    </div>
  );
}
