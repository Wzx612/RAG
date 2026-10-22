export type ApiOutcomeKind =
  | 'success'
  | 'http_error'
  | 'network_error'
  | 'timeout'
  | 'parse_error'
  | 'aborted';

export interface ApiOutcome<T> {
  kind: ApiOutcomeKind;
  httpStatus?: number;
  data?: T;
  raw?: unknown;
  rawText?: string;
  message?: string;
  durationMs: number;
}

export interface SearchRequest {
  query: string;
  top_k: number;
}

export interface SearchResult {
  chunkId: string | null;
  docId: string | null;
  rank: number | null;
  score: number | null;
  source: string | null;
  rerankScore: number | null;
  title: string | null;
  category: string | null;
  cardTier: string | null;
  effectiveYear: number | string | null;
  policyCode: string | null;
  content: string | null;
}

export interface SearchResponse {
  query: string;
  /** null when the backend omitted the field; the UI shows 未返回 instead of guessing. */
  topK: number | null;
  hitCount: number | null;
  retriever: string;
  results: SearchResult[];
  timingsMs: Record<string, number>;
}

export type RagStatus =
  | 'ok'
  | 'insufficient_evidence'
  | 'invalid_request'
  | 'retrieval_error'
  | 'rerank_error'
  | 'evidence_error'
  | 'config_error'
  | 'llm_timeout'
  | 'llm_rate_limited'
  | 'llm_network_error'
  | 'llm_server_error'
  | 'llm_auth_error'
  | 'llm_invalid_response'
  | 'llm_request_error'
  | 'llm_error';

export interface Citation {
  label: string | null;
  chunkId: string | null;
  docId: string | null;
  title: string | null;
}

export interface Evidence {
  label: string | null;
  chunkId: string | null;
  docId: string | null;
  chunkStrategy: string | null;
  chunkIndex: number | null;
  retrievalRank: number | null;
  score: number | null;
  rerankScore: number | null;
  title: string | null;
  category: string | null;
  cardTier: string | null;
  effectiveYear: number | string | null;
  policyCode: string | null;
  content: string | null;
  contextBlock: string | null;
  originalChars: number | null;
  includedChars: number | null;
  truncated: boolean | null;
}

export interface LLMUsage {
  promptTokens: number | null;
  completionTokens: number | null;
  totalTokens: number | null;
}

export interface AnswerError {
  stage: string;
  category: string;
  message: string;
  retryable: boolean | null;
  statusCode: number | null;
  attempts: number | null;
}

export interface AnswerResponse {
  queryId: string | null;
  status: RagStatus;
  answer: string;
  citations: Citation[];
  invalidCitations: string[];
  citationStatus: 'ok' | 'no_citations' | 'invalid' | string;
  insufficientEvidence: boolean | null;
  evidence: Evidence[];
  promptVersion: string | null;
  provider: string | null;
  model: string | null;
  usage: LLMUsage | null;
  timingsMs: Record<string, number>;
  /** null when the backend omitted the field. */
  providerAttempted: boolean | null;
  error: AnswerError | null;
}

export interface HealthResponse {
  status: string;
  service: string | null;
}

export interface ReadinessCheck {
  name: string;
  ok: boolean | null;
  detail: string;
}

export interface ReadinessResponse {
  status: string;
  checks: ReadinessCheck[];
}

export interface AcceptanceArtifact {
  key: string;
  label: string;
  required: boolean | null;
  present: boolean | null;
  status: string | null;
  statusReason: string | null;
  generatedAt: string | null;
  path: string | null;
  synthetic: boolean | null;
}

export interface AcceptanceCheck {
  id: string;
  label: string;
  state: string;
  detail: string;
}

export interface AcceptanceDataset {
  queries: number | null;
  evalQueriesOnDisk: number | null;
  goldDocs: number | null;
}

export interface AcceptanceEnvironment {
  pythonVersion: string | null;
  postgresVersion: string | null;
  vectorVersion: string | null;
  embeddingModel: string | null;
  embeddingDevice: string | null;
  embeddingDimensions: number[] | null;
  documents: number | null;
  chunks: number | null;
  rerankerModel: string | null;
  executionEnvironment: string | null;
}

export interface AcceptanceLimitation {
  source: string;
  text: string;
}

export interface AcceptanceSummary {
  schemaVersion: number | null;
  generatedAt: string | null;
  readOnly: boolean | null;
  status: string | null;
  statusReason: string | null;
  artifacts: AcceptanceArtifact[];
  dataset: AcceptanceDataset;
  environment: AcceptanceEnvironment;
  checks: AcceptanceCheck[];
  limitations: AcceptanceLimitation[];
  caveats: string[];
}
