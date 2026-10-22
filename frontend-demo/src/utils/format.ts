export function formatMs(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '未返回';
  return `${value.toLocaleString('zh-CN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} ms`;
}

export function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '未返回';
  return value.toFixed(4);
}

export function formatInt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '未返回';
  return value.toLocaleString('zh-CN');
}

export function formatDateTime(value: Date | string | null | undefined): string {
  if (!value) return '未返回';
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString('zh-CN', { hour12: false });
}

export function formatTimeOnly(value: Date | null | undefined): string {
  if (!value) return '未返回';
  return value.toLocaleTimeString('zh-CN', { hour12: false });
}

export function formatJson(value: unknown): string {
  if (value === undefined) return '（无响应体）';
  try {
    const json = JSON.stringify(value, null, 2);
    return json === undefined ? String(value) : json;
  } catch {
    return String(value);
  }
}

export function truncateText(text: string, max = 220): string {
  const normalized = text.replace(/\s+/g, ' ').trim();
  if (normalized.length <= max) return normalized;
  return `${normalized.slice(0, max)}…`;
}

export function textOrMissing(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return '未返回';
  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(value) : '未返回';
  }
  return value.trim() === '' ? '未返回' : value;
}

export function isSafeExternalUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}
