import type {
  AcceptanceArtifact,
  AcceptanceCheck,
  AcceptanceDataset,
  AcceptanceEnvironment,
  AcceptanceLimitation,
  AcceptanceSummary,
  AnswerError,
  AnswerResponse,
  Citation,
  Evidence,
  HealthResponse,
  LLMUsage,
  ReadinessCheck,
  ReadinessResponse,
  SearchResponse,
  SearchResult,
} from './types';

type JsonRecord = Record<string, unknown>;

export function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function read(record: JsonRecord, ...keys: string[]): unknown {
  for (const key of keys) {
    if (Object.prototype.hasOwnProperty.call(record, key) && record[key] !== undefined) {
      return record[key];
    }
  }
  return undefined;
}

function asString(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function asText(value: unknown): string | null {
  if (typeof value === 'string') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return String(value);
  return null;
}

function asNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function asBooleanOrNull(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

function asNumberArray(value: unknown): number[] | null {
  const values = Array.isArray(value) ? value : [value];
  const numbers = values
    .map(asNumber)
    .filter((item): item is number => item !== null);
  return numbers.length === 0 ? null : numbers;
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === 'string');
}

function asRecordArray(value: unknown): JsonRecord[] {
  if (!Array.isArray(value)) return [];
  return value.filter(isRecord);
}

/** Keep only finite numeric entries; timings are displayed as returned. */
export function parseTimings(value: unknown): Record<string, number> {
  const result: Record<string, number> = {};
  if (!isRecord(value)) return result;
  for (const [key, raw] of Object.entries(value)) {
    if (typeof raw === 'number' && Number.isFinite(raw)) {
      result[key] = raw;
    }
  }
  return result;
}

function parseSearchResult(record: JsonRecord): SearchResult {
  return {
    chunkId: asString(read(record, 'chunk_id', 'chunkId')),
    docId: asString(read(record, 'doc_id', 'docId')),
    rank: asNumber(read(record, 'rank')),
    score: asNumber(read(record, 'score')),
    source: asString(read(record, 'source')),
    rerankScore: asNumber(read(record, 'rerank_score', 'rerankScore')),
    title: asString(read(record, 'title')),
    category: asString(read(record, 'category')),
    cardTier: asString(read(record, 'card_tier', 'cardTier')),
    effectiveYear: asText(read(record, 'effective_year', 'effectiveYear')),
    policyCode: asString(read(record, 'policy_code', 'policyCode')),
    content: asString(read(record, 'content')),
  };
}

export function parseSearchResponse(raw: unknown): SearchResponse | null {
  if (!isRecord(raw)) return null;
  const resultsRaw = read(raw, 'results');
  if (!Array.isArray(resultsRaw)) return null;
  const results = resultsRaw.filter(isRecord).map(parseSearchResult);
  const hitCount = asNumber(read(raw, 'hit_count', 'hitCount'));
  const topK = asNumber(read(raw, 'top_k', 'topK'));
  return {
    query: asString(read(raw, 'query')) ?? '',
    topK,
    hitCount,
    retriever: asString(read(raw, 'retriever')) ?? '',
    results,
    timingsMs: parseTimings(read(raw, 'timings_ms', 'timingsMs')),
  };
}

function parseCitation(record: JsonRecord): Citation {
  return {
    label: asString(read(record, 'label')),
    chunkId: asString(read(record, 'chunk_id', 'chunkId')),
    docId: asString(read(record, 'doc_id', 'docId')),
    title: asString(read(record, 'title')),
  };
}

function parseEvidence(record: JsonRecord): Evidence {
  return {
    label: asString(read(record, 'label')),
    chunkId: asString(read(record, 'chunk_id', 'chunkId')),
    docId: asString(read(record, 'doc_id', 'docId')),
    chunkStrategy: asString(read(record, 'chunk_strategy', 'chunkStrategy')),
    chunkIndex: asNumber(read(record, 'chunk_index', 'chunkIndex')),
    retrievalRank: asNumber(read(record, 'retrieval_rank', 'retrievalRank')),
    score: asNumber(read(record, 'score')),
    rerankScore: asNumber(read(record, 'rerank_score', 'rerankScore')),
    title: asString(read(record, 'title')),
    category: asString(read(record, 'category')),
    cardTier: asString(read(record, 'card_tier', 'cardTier')),
    effectiveYear: asText(read(record, 'effective_year', 'effectiveYear')),
    policyCode: asString(read(record, 'policy_code', 'policyCode')),
    content: asString(read(record, 'content')),
    contextBlock: asString(read(record, 'context_block', 'contextBlock')),
    originalChars: asNumber(read(record, 'original_chars', 'originalChars')),
    includedChars: asNumber(read(record, 'included_chars', 'includedChars')),
    truncated: asBooleanOrNull(read(record, 'truncated')),
  };
}

function parseUsage(value: unknown): LLMUsage | null {
  if (!isRecord(value)) return null;
  return {
    promptTokens: asNumber(read(value, 'prompt_tokens', 'promptTokens')),
    completionTokens: asNumber(read(value, 'completion_tokens', 'completionTokens')),
    totalTokens: asNumber(read(value, 'total_tokens', 'totalTokens')),
  };
}

function parseAnswerError(value: unknown): AnswerError | null {
  if (!isRecord(value)) return null;
  return {
    stage: asString(read(value, 'stage')) ?? '',
    category: asString(read(value, 'category')) ?? '',
    message: asString(read(value, 'message')) ?? '',
    retryable: asBooleanOrNull(read(value, 'retryable')),
    statusCode: asNumber(read(value, 'status_code', 'statusCode')),
    attempts: asNumber(read(value, 'attempts')),
  };
}

export function parseAnswerResponse(raw: unknown): AnswerResponse | null {
  if (!isRecord(raw)) return null;
  const status = asString(read(raw, 'status'));
  if (status === null) return null;
  return {
    queryId: asString(read(raw, 'query_id', 'queryId')),
    status: status as AnswerResponse['status'],
    answer: asString(read(raw, 'answer')) ?? '',
    citations: asRecordArray(read(raw, 'citations')).map(parseCitation),
    invalidCitations: asStringArray(read(raw, 'invalid_citations', 'invalidCitations')),
    citationStatus: asString(read(raw, 'citation_status', 'citationStatus')) ?? '',
    insufficientEvidence: asBooleanOrNull(
      read(raw, 'insufficient_evidence', 'insufficientEvidence'),
    ),
    evidence: asRecordArray(read(raw, 'evidence')).map(parseEvidence),
    promptVersion: asString(read(raw, 'prompt_version', 'promptVersion')),
    provider: asString(read(raw, 'provider')),
    model: asString(read(raw, 'model')),
    usage: parseUsage(read(raw, 'usage')),
    timingsMs: parseTimings(read(raw, 'timings_ms', 'timingsMs')),
    providerAttempted: asBooleanOrNull(read(raw, 'provider_attempted', 'providerAttempted')),
    error: parseAnswerError(read(raw, 'error')),
  };
}

export function parseHealthResponse(raw: unknown): HealthResponse | null {
  if (!isRecord(raw)) return null;
  const status = asString(read(raw, 'status'));
  if (status === null) return null;
  return { status, service: asString(read(raw, 'service')) };
}

function parseReadinessCheck(record: JsonRecord): ReadinessCheck {
  return {
    name: asString(read(record, 'name')) ?? '',
    ok: asBooleanOrNull(read(record, 'ok')),
    detail: asString(read(record, 'detail')) ?? '',
  };
}

export function parseReadinessResponse(raw: unknown): ReadinessResponse | null {
  if (!isRecord(raw)) return null;
  const status = asString(read(raw, 'status'));
  if (status === null) return null;
  return {
    status,
    checks: asRecordArray(read(raw, 'checks')).map(parseReadinessCheck),
  };
}

function parseArtifact(record: JsonRecord): AcceptanceArtifact {
  return {
    key: asString(read(record, 'key')) ?? '',
    label: asString(read(record, 'label')) ?? '',
    required: asBooleanOrNull(read(record, 'required')),
    present: asBooleanOrNull(read(record, 'present')),
    status: asString(read(record, 'status')),
    statusReason: asString(read(record, 'status_reason', 'statusReason')),
    generatedAt: asString(read(record, 'generated_at', 'generatedAt')),
    path: asString(read(record, 'path')),
    synthetic: asBooleanOrNull(read(record, 'synthetic')),
  };
}

function parseCheck(record: JsonRecord): AcceptanceCheck {
  return {
    id: asString(read(record, 'id')) ?? '',
    label: asString(read(record, 'label')) ?? '',
    state: asString(read(record, 'state')) ?? '',
    detail: asString(read(record, 'detail')) ?? '',
  };
}

function parseDataset(value: unknown): AcceptanceDataset {
  const record: JsonRecord = isRecord(value) ? value : {};
  return {
    queries: asNumber(read(record, 'queries')),
    evalQueriesOnDisk: asNumber(read(record, 'eval_queries_on_disk', 'evalQueriesOnDisk')),
    goldDocs: asNumber(read(record, 'gold_docs', 'goldDocs')),
  };
}

function parseEnvironment(value: unknown): AcceptanceEnvironment {
  const record: JsonRecord = isRecord(value) ? value : {};
  return {
    pythonVersion: asString(read(record, 'python_version', 'pythonVersion')),
    postgresVersion: asString(read(record, 'postgres_version', 'postgresVersion')),
    vectorVersion: asString(read(record, 'vector_version', 'vectorVersion')),
    embeddingModel: asString(read(record, 'embedding_model', 'embeddingModel')),
    embeddingDevice: asString(read(record, 'embedding_device', 'embeddingDevice')),
    embeddingDimensions: asNumberArray(
      read(record, 'embedding_dimensions', 'embeddingDimensions'),
    ),
    documents: asNumber(read(record, 'documents')),
    chunks: asNumber(read(record, 'chunks')),
    rerankerModel: asString(read(record, 'reranker_model', 'rerankerModel')),
    executionEnvironment: asString(
      read(record, 'execution_environment', 'executionEnvironment'),
    ),
  };
}

function parseLimitations(value: unknown): AcceptanceLimitation[] {
  return asRecordArray(value).map((record) => ({
    source: asString(read(record, 'source')) ?? '',
    text: asString(read(record, 'text')) ?? '',
  }));
}

export function parseAcceptanceSummary(raw: unknown): AcceptanceSummary | null {
  if (!isRecord(raw)) return null;
  const status = asString(read(raw, 'status'));
  if (status === null) return null;
  return {
    schemaVersion: asNumber(read(raw, 'schema_version', 'schemaVersion')),
    generatedAt: asString(read(raw, 'generated_at', 'generatedAt')),
    readOnly:
      typeof read(raw, 'read_only', 'readOnly') === 'boolean'
        ? (read(raw, 'read_only', 'readOnly') as boolean)
        : null,
    status,
    statusReason: asString(read(raw, 'status_reason', 'statusReason')),
    artifacts: asRecordArray(read(raw, 'artifacts')).map(parseArtifact),
    dataset: parseDataset(read(raw, 'dataset')),
    environment: parseEnvironment(read(raw, 'environment')),
    checks: asRecordArray(read(raw, 'checks')).map(parseCheck),
    limitations: parseLimitations(read(raw, 'limitations')),
    caveats: asStringArray(read(raw, 'caveats')),
  };
}
