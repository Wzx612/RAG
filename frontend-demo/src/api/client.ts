import {
  ACCEPTANCE_TIMEOUT_MS,
  ANSWER_TIMEOUT_MS,
  API_BASE_URL,
  BACKEND_DISPLAY_ADDRESS,
  HEALTH_TIMEOUT_MS,
  READINESS_TIMEOUT_MS,
  SEARCH_TIMEOUT_MS,
} from './config';
import {
  parseAcceptanceSummary,
  parseAnswerResponse,
  parseHealthResponse,
  parseReadinessResponse,
  parseSearchResponse,
  isRecord,
} from './parse';
import type {
  AcceptanceSummary,
  AnswerResponse,
  ApiOutcome,
  HealthResponse,
  ReadinessResponse,
  SearchResponse,
} from './types';

type Parser<T> = (raw: unknown) => T | null;

interface RequestOptions<T> {
  method?: 'GET' | 'POST';
  body?: unknown;
  timeoutMs: number;
  signal?: AbortSignal;
  parse: Parser<T>;
}

function requestUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

function extractBackendMessage(raw: unknown): string | null {
  if (!isRecord(raw)) return null;
  const detail = raw.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    // FastAPI 422 validation errors: [{loc, msg, type}, ...]
    const parts = detail
      .map((item) => {
        if (!isRecord(item)) return null;
        const loc = Array.isArray(item.loc) ? item.loc.join('.') : '';
        const msg = typeof item.msg === 'string' ? item.msg : '';
        if (!msg) return null;
        return loc ? `${loc}: ${msg}` : msg;
      })
      .filter((item): item is string => Boolean(item));
    if (parts.length > 0) return parts.join('; ');
  }
  if (isRecord(raw.error) && typeof raw.error.message === 'string') {
    return raw.error.message;
  }
  if (typeof raw.message === 'string') return raw.message;
  return null;
}

/**
 * Vite's proxy answers with a plain-text 500 when it cannot reach the target.
 * Treat that as a connection failure, not as a backend HTTP error, so the UI
 * can show the "backend unreachable" state and skip a pointless /answer call.
 */
function isProxyConnectionFailure(status: number, rawParsed: boolean, rawText: string): boolean {
  if (rawParsed || status < 500) return false;
  return /ECONNREFUSED|EHOSTUNREACH|ENOTFOUND|Error occurred while trying to proxy/i.test(
    rawText,
  );
}

const UNREACHABLE_MESSAGE = `无法连接后端。请确认 Docker Compose 的 db 与 api 服务已启动，并检查 ${BACKEND_DISPLAY_ADDRESS} 是否可访问。`;

export function describeHttpStatus(status: number): string {
  if (status === 404) return '接口不存在或未启用（HTTP 404）。';
  if (status === 422) return '请求参数未通过后端校验（HTTP 422）。';
  if (status === 429) return '请求过于频繁（HTTP 429）。';
  if (status === 500) return '后端内部错误（HTTP 500）。';
  if (status === 502) return '生成服务返回无效响应或网关错误（HTTP 502）。';
  if (status === 503) return '后端依赖不可用（HTTP 503）。';
  if (status === 504) return '生成请求超时（HTTP 504）。';
  return `后端返回 HTTP ${status}。`;
}

async function requestJson<T>(path: string, options: RequestOptions<T>): Promise<ApiOutcome<T>> {
  const controller = new AbortController();
  const startedAt = performance.now();
  let timedOut = false;

  const onExternalAbort = () => {
    controller.abort();
  };

  if (options.signal) {
    if (options.signal.aborted) {
      return { kind: 'aborted', durationMs: 0, message: '请求已取消。' };
    }
    options.signal.addEventListener('abort', onExternalAbort, { once: true });
  }

  const timer = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, options.timeoutMs);

  try {
    const response = await fetch(requestUrl(path), {
      method: options.method ?? 'GET',
      headers: options.body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      signal: controller.signal,
      cache: 'no-store',
    });

    const rawText = await response.text();
    const durationMs = Math.round(performance.now() - startedAt);
    let raw: unknown = undefined;
    let rawParsed = false;
    if (rawText.trim() !== '') {
      try {
        raw = JSON.parse(rawText);
        rawParsed = true;
      } catch {
        raw = undefined;
      }
    }

    const parsed = rawParsed ? options.parse(raw) : null;

    if (!response.ok) {
      if (isProxyConnectionFailure(response.status, rawParsed, rawText)) {
        return {
          kind: 'network_error',
          httpStatus: response.status,
          rawText,
          message: UNREACHABLE_MESSAGE,
          durationMs,
        };
      }
      const backendMessage = extractBackendMessage(raw);
      return {
        kind: 'http_error',
        httpStatus: response.status,
        data: parsed ?? undefined,
        raw,
        rawText,
        message: backendMessage ?? describeHttpStatus(response.status),
        durationMs,
      };
    }

    if (!rawParsed) {
      return {
        kind: 'parse_error',
        httpStatus: response.status,
        rawText,
        message: '后端返回的内容不是合法 JSON。',
        durationMs,
      };
    }

    if (parsed === null) {
      return {
        kind: 'parse_error',
        httpStatus: response.status,
        raw,
        rawText,
        message: '后端响应缺少预期字段，无法安全解析。',
        durationMs,
      };
    }

    return {
      kind: 'success',
      httpStatus: response.status,
      data: parsed,
      raw,
      rawText,
      durationMs,
    };
  } catch (error) {
    const durationMs = Math.round(performance.now() - startedAt);
    if (controller.signal.aborted) {
      if (timedOut) {
        return {
          kind: 'timeout',
          durationMs,
          message: `请求超时（${options.timeoutMs} ms）。后端可能正在加载模型，请稍后重试。`,
        };
      }
      return {
        kind: 'aborted',
        durationMs,
        message: '请求已取消。',
      };
    }
    const detail = error instanceof Error ? error.message : String(error);
    return {
      kind: 'network_error',
      durationMs,
      message: `${UNREACHABLE_MESSAGE}（${detail}）`,
    };
  } finally {
    window.clearTimeout(timer);
    options.signal?.removeEventListener('abort', onExternalAbort);
  }
}

export function searchKnowledge(
  query: string,
  topK: number,
  signal?: AbortSignal,
): Promise<ApiOutcome<SearchResponse>> {
  return requestJson('/api/v1/search', {
    method: 'POST',
    body: { query, top_k: topK },
    timeoutMs: SEARCH_TIMEOUT_MS,
    signal,
    parse: parseSearchResponse,
  });
}

export function generateAnswer(
  query: string,
  queryId?: string,
  signal?: AbortSignal,
): Promise<ApiOutcome<AnswerResponse>> {
  const body: Record<string, string> = { query };
  if (queryId) body.query_id = queryId;
  return requestJson('/answer', {
    method: 'POST',
    body,
    timeoutMs: ANSWER_TIMEOUT_MS,
    signal,
    parse: parseAnswerResponse,
  });
}

export function checkHealth(signal?: AbortSignal): Promise<ApiOutcome<HealthResponse>> {
  return requestJson('/healthz', {
    timeoutMs: HEALTH_TIMEOUT_MS,
    signal,
    parse: parseHealthResponse,
  });
}

export function checkReadiness(signal?: AbortSignal): Promise<ApiOutcome<ReadinessResponse>> {
  return requestJson('/readyz', {
    timeoutMs: READINESS_TIMEOUT_MS,
    signal,
    parse: parseReadinessResponse,
  });
}

export function fetchAcceptanceSummary(
  signal?: AbortSignal,
): Promise<ApiOutcome<AcceptanceSummary>> {
  return requestJson('/acceptance/api/summary', {
    timeoutMs: ACCEPTANCE_TIMEOUT_MS,
    signal,
    parse: parseAcceptanceSummary,
  });
}

export function outcomeErrorText(outcome: ApiOutcome<unknown>): string {
  if (outcome.kind === 'http_error') {
    return outcome.message ?? describeHttpStatus(outcome.httpStatus ?? 0);
  }
  return outcome.message ?? '请求失败。';
}
