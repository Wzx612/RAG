import type { ChatTurn } from '../hooks/useConversation';
import { formatTimeOnly } from '../utils/format';
import { AnswerPanel } from './AnswerPanel';
import { BrandMark } from './icons';
import { EvidenceCard } from './EvidenceCard';
import { LoadingLine } from './LoadingLine';
import { RawResponse } from './RawResponse';
import { RetrievalSummary } from './RetrievalSummary';

interface AssistantMessageProps {
  turn: ChatTurn;
}

export function AssistantMessage({ turn }: AssistantMessageProps) {
  const answerData = turn.answerOutcome?.data ?? null;
  const searchData =
    turn.searchOutcome?.kind === 'success' ? turn.searchOutcome.data ?? null : null;
  const results = searchData?.results ?? [];
  const top1 = results.find((item) => item.rank === 1) ?? results[0] ?? null;

  return (
    <article className="msg msg--assistant" aria-label="助手回答">
      <div className="msg__avatar" aria-hidden="true">
        <BrandMark />
      </div>
      <div className="msg__body">
        <div className="msg__meta">
          <span className="msg__author">信用卡智能客服助手</span>
          <span className="msg__time">{formatTimeOnly(turn.createdAt)}</span>
          {turn.stage === 'searching' ? (
            <span className="phase-pill">检索中</span>
          ) : turn.stage === 'generating' ? (
            <span className="phase-pill">生成中</span>
          ) : null}
          {answerData?.provider ? (
            <span className="tag tag--neutral">
              {answerData.provider}
              {answerData.model ? ` / ${answerData.model}` : ''}
            </span>
          ) : null}
        </div>

        <AnswerPanel
          outcome={turn.answerOutcome}
          stage={turn.stage}
          skippedReason={turn.answerSkippedReason}
        />

        <section className="retrieval-section" aria-label="检索证据">
          <div className="retrieval-section__head">
            <span className="section-kicker">检索证据</span>
            <span className="endpoint-chip">POST /api/v1/search</span>
            <span className="section-note">请求 Top-K {turn.topK}</span>
          </div>

          {turn.stage === 'searching' && !turn.searchOutcome ? (
            <LoadingLine
              label="正在检索知识库…"
              detail={`Top-K ${turn.topK} · Dense + Sparse → RRF → Reranker`}
            />
          ) : turn.searchOutcome ? (
            <>
              <RetrievalSummary outcome={turn.searchOutcome} requestedTopK={turn.topK} />
              {results.length > 0 ? (
                <>
                  <div className="doc-list" aria-label="Top-K 检索文档">
                    {results.map((result, index) => (
                      <EvidenceCard
                        key={`${result.chunkId ?? 'chunk'}-${index}`}
                        result={result}
                        index={index}
                        isTop1={result === top1}
                      />
                    ))}
                  </div>
                  <p className="retrieval-section__footnote">
                    文档按后端返回顺序展示，前端未重新排序。分数与 rerank_score 均为原始返回值。
                  </p>
                </>
              ) : null}
            </>
          ) : (
            <p className="muted">暂无检索响应。</p>
          )}
        </section>

        <RawResponse turn={turn} />
      </div>
    </article>
  );
}
