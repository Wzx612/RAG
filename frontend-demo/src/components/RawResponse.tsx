import { useEffect, useRef, useState } from 'react';
import type { ApiOutcome } from '../api/types';
import type { ChatTurn } from '../hooks/useConversation';
import { formatJson } from '../utils/format';
import { CheckIcon, ChevronDownIcon, CopyIcon } from './icons';

interface RawResponseProps {
  turn: ChatTurn;
}

function outcomeStatus(outcome: ApiOutcome<unknown> | null, pending: boolean): string {
  if (!outcome) return pending ? '请求已发送，等待响应' : '尚未调用';
  if (outcome.kind === 'success') return `HTTP ${outcome.httpStatus ?? '未返回'}`;
  if (outcome.kind === 'http_error') return `HTTP ${outcome.httpStatus ?? '?'}（错误响应）`;
  if (outcome.kind === 'network_error') {
    return outcome.httpStatus !== undefined
      ? `网络错误（代理 HTTP ${outcome.httpStatus}）`
      : '网络错误（无 HTTP 状态）';
  }
  if (outcome.kind === 'timeout') return '请求超时（无 HTTP 状态）';
  if (outcome.kind === 'parse_error') return `HTTP ${outcome.httpStatus ?? '?'}（响应无法解析）`;
  return '请求已取消';
}

function responseTextOf(outcome: ApiOutcome<unknown> | null, pending: boolean): string {
  if (!outcome) return pending ? '（等待响应）' : '（未发送请求，无响应体）';
  if (outcome.raw !== undefined) return formatJson(outcome.raw);
  if (outcome.rawText !== undefined && outcome.rawText !== '') return outcome.rawText;
  return '（无响应体）';
}

async function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {
      // Permission can be denied (e.g. insecure context); try the legacy path.
    }
  }
  const area = document.createElement('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  area.style.position = 'fixed';
  area.style.opacity = '0';
  document.body.appendChild(area);
  try {
    area.select();
    const ok = document.execCommand('copy');
    if (!ok) throw new Error('copy command failed');
  } finally {
    document.body.removeChild(area);
  }
}

function CopyButton({ getText }: { getText: () => string }) {
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle');
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    };
  }, []);

  const handleClick = async () => {
    try {
      await copyToClipboard(getText());
      setState('copied');
    } catch {
      setState('failed');
    }
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => setState('idle'), 2000);
  };

  return (
    <button type="button" className="copy-button" onClick={() => void handleClick()}>
      {state === 'copied' ? <CheckIcon /> : <CopyIcon />}
      {state === 'copied' ? '已复制' : state === 'failed' ? '复制失败' : '复制请求与响应'}
    </button>
  );
}

interface RawBlockProps {
  title: string;
  request: unknown;
  outcome: ApiOutcome<unknown> | null;
  requestSent: boolean;
  pending: boolean;
}

function RawBlock({ title, request, outcome, requestSent, pending }: RawBlockProps) {
  const responseText = responseTextOf(outcome, pending);
  const requestText = requestSent ? formatJson(request) : '（本次未发送该请求）';
  const requestLabel = requestSent ? '实际请求体' : '请求体（未发送）';
  const responseLabel = requestSent ? '实际响应体' : '响应体（无）';
  const buildText = () =>
    `${title}\n\n${requestLabel}\n${requestText}\n\n响应（${outcomeStatus(outcome, pending)}）\n${responseText}`;

  return (
    <div className="raw-block">
      <div className="raw-block__head">
        <div>
          <h4 className="raw-block__title">{title}</h4>
          <span className="raw-block__status">{outcomeStatus(outcome, pending)}</span>
        </div>
        <CopyButton getText={buildText} />
      </div>
      <div className="raw-block__label">{requestLabel}</div>
      <pre className="raw-block__pre">{requestText}</pre>
      <div className="raw-block__label">{responseLabel}</div>
      <pre className="raw-block__pre">{responseText}</pre>
    </div>
  );
}

export function RawResponse({ turn }: RawResponseProps) {
  const answerRequestSent =
    turn.answerOutcome !== null ||
    turn.stage === 'generating' ||
    turn.stage === 'done';

  return (
    <details className="raw-panel">
      <summary className="raw-panel__summary">
        <span>原始响应（本次网络请求）</span>
        <span className="raw-panel__hint">
          已发送请求的请求体与响应体均来自本次 fetch，不使用预存 JSON
        </span>
        <ChevronDownIcon className="raw-panel__chevron" />
      </summary>
      <div className="raw-panel__body">
        <RawBlock
          title="POST /api/v1/search"
          request={{ query: turn.query, top_k: turn.topK }}
          outcome={turn.searchOutcome}
          requestSent
          pending={turn.stage === 'searching' && !turn.searchOutcome}
        />
        <RawBlock
          title="POST /answer"
          request={{ query: turn.query }}
          outcome={turn.answerOutcome}
          requestSent={answerRequestSent}
          pending={turn.stage === 'generating' && !turn.answerOutcome}
        />
      </div>
    </details>
  );
}
