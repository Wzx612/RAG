import { useId, useState } from 'react';
import type { SearchResult } from '../api/types';
import { formatScore, textOrMissing, truncateText } from '../utils/format';
import { ChevronDownIcon } from './icons';

interface EvidenceCardProps {
  result: SearchResult;
  index: number;
  isTop1: boolean;
}

export function EvidenceCard({ result, index, isTop1 }: EvidenceCardProps) {
  const [expanded, setExpanded] = useState(false);
  const detailId = useId();
  const rankLabel = result.rank !== null ? `Rank ${result.rank}` : `结果 ${index + 1}`;
  const content = result.content;

  return (
    <article className={expanded ? 'doc-card is-expanded' : 'doc-card'}>
      <div className="doc-card__head">
        <span className="rank-badge">{rankLabel}</span>
        {isTop1 ? <span className="tag tag--top1">Top-1</span> : null}
        <span className="doc-card__ids">
          {textOrMissing(result.docId)}
          {result.chunkId ? ` · ${result.chunkId}` : ''}
        </span>
        <button
          type="button"
          className="link-button"
          aria-expanded={expanded}
          aria-controls={detailId}
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? '收起' : '展开全文与字段'}
          <ChevronDownIcon className={expanded ? 'chevron is-open' : 'chevron'} />
        </button>
      </div>

      <h3 className="doc-card__title">{textOrMissing(result.title)}</h3>

      <div className="doc-card__scores">
        <span>
          <span className="score-key">score</span> {formatScore(result.score)}
        </span>
        <span>
          <span className="score-key">rerank_score</span>{' '}
          {formatScore(result.rerankScore)}
        </span>
        <span>
          <span className="score-key">source</span> {textOrMissing(result.source)}
        </span>
      </div>

      <p className="doc-card__preview">
        {content ? truncateText(content, 240) : '未返回正文内容。'}
      </p>

      <div className="doc-card__detail" id={detailId} hidden={!expanded}>
        <div className="doc-card__section-label">完整正文</div>
        <pre className="doc-card__content">{content ?? '未返回正文内容。'}</pre>

        <div className="doc-card__section-label">元数据（后端实际返回字段）</div>
        <dl className="metadata-grid">
          <div>
            <dt>doc_id</dt>
            <dd>{textOrMissing(result.docId)}</dd>
          </div>
          <div>
            <dt>chunk_id</dt>
            <dd>{textOrMissing(result.chunkId)}</dd>
          </div>
          <div>
            <dt>rank</dt>
            <dd>{result.rank ?? '未返回'}</dd>
          </div>
          <div>
            <dt>source</dt>
            <dd>{textOrMissing(result.source)}</dd>
          </div>
          <div>
            <dt>category</dt>
            <dd>{textOrMissing(result.category)}</dd>
          </div>
          <div>
            <dt>card_tier</dt>
            <dd>{textOrMissing(result.cardTier)}</dd>
          </div>
          <div>
            <dt>effective_year</dt>
            <dd>{textOrMissing(result.effectiveYear)}</dd>
          </div>
          <div>
            <dt>policy_code</dt>
            <dd>{textOrMissing(result.policyCode)}</dd>
          </div>
          <div>
            <dt>title</dt>
            <dd>{textOrMissing(result.title)}</dd>
          </div>
          <div>
            <dt>score</dt>
            <dd>{formatScore(result.score)}</dd>
          </div>
          <div>
            <dt>rerank_score</dt>
            <dd>{formatScore(result.rerankScore)}</dd>
          </div>
        </dl>
      </div>
    </article>
  );
}
