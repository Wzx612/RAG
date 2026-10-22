const configuredBase = (import.meta.env.VITE_API_BASE_URL ?? '').trim().replace(/\/+$/, '');

/**
 * Empty by default: requests are relative and handled by the Vite proxy.
 * Do not put secrets in VITE_* variables; this project never needs one.
 */
export const API_BASE_URL = configuredBase;

/**
 * Displayed/default backend address. When no override is configured requests go
 * through the Vite proxy to localhost:8000, so that is what the UI reports.
 */
export const BACKEND_DISPLAY_ADDRESS = configuredBase
  ? configuredBase.replace(/^https?:\/\//i, '')
  : 'localhost:8000';

export const TOP_K_OPTIONS = [3, 5, 10] as const;
export const DEFAULT_TOP_K = 5;

export const SEARCH_TIMEOUT_MS = 120_000;
export const ANSWER_TIMEOUT_MS = 150_000;
export const HEALTH_TIMEOUT_MS = 8_000;
export const READINESS_TIMEOUT_MS = 10_000;
export const ACCEPTANCE_TIMEOUT_MS = 20_000;

export const MAX_QUERY_CHARS = 2000;
