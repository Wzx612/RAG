import type { Evidence } from '../api/types';
import { formatScore, textOrMissing } from '../utils/format';
import { ChevronDownIcon } from './icons';

interface AnswerEvidenceProps {
  evidence: Evidence[];
}

export function AnswerEvidence({ evidence }: AnswerEvidenceProps) {
  return (
    <details className="answer-evidence">
      <summary className="answer-evidence__summary">
        <span>查看生成所用证据（来自 /answer，共 {evidence.length} 条）</span>
        <ChevronDownIcon className="answer-evidence__chevron" />
      </summary>
      {evidence.length === 0 ? (
        <p className="answer-evidence__empty">后端未返回生成证据。</p>
      ) : (
        <ol className="answer-evidence__list">
          {evidence.map((item, index) => (
            <li key={`${item.label ?? 'evidence'}-${item.chunkId ?? index}`} className="answer-evidence__item">
              <div className="answer-evidence__head">
                <span className="answer-evidence__label">
                  {item.label ?? `E${index + 1}`}
                </span>
                <span className="answer-evidence__ids">
                  {textOrMissing(item.docId)}
                  {item.chunkId ? ` · ${item.chunkId}` : ''}
                </span>
                {item.truncated === true ? (
                  <span className="tag tag--warning">正文已截断</span>
                ) : null}
              </div>
              <div className="answer-evidence__scores">
                <span>retrieval_rank：{item.retrievalRank ?? '未返回'}</span>
                <span>score：{formatScore(item.score)}</span>
                <span>rerank_score：{formatScore(item.rerankScore)}</span>
                {item.originalChars !== null ? (
                  <span>
                    chars：{item.includedChars ?? '未返回'} / {item.originalChars}
                  </span>
                ) : null}
              </div>
              {item.title ? <p className="answer-evidence__title">{item.title}</p> : null}
              <pre className="answer-evidence__content">
                {item.content ?? '未返回正文内容。'}
              </pre>
            </li>
          ))}
        </ol>
      )}
    </details>
  );
}
