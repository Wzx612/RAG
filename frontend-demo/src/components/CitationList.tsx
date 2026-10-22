import type { Citation } from '../api/types';

interface CitationListProps {
  citations: Citation[];
  invalidCitations: string[];
  citationStatus: string;
}

const CITATION_STATUS_LABELS: Record<string, string> = {
  ok: '引用校验通过',
  no_citations: '后端未返回引用',
  invalid: '存在无效引用',
};

export function CitationList({
  citations,
  invalidCitations,
  citationStatus,
}: CitationListProps) {
  if (citations.length === 0) {
    return (
      <>
        <p className="citation-empty">
          后端未返回引用
          {citationStatus ? `（citation_status = ${citationStatus}）` : ''}。
        </p>
        {invalidCitations.length > 0 ? (
          <p className="citation-warning">
            后端标记的无效引用：{invalidCitations.join('、')}
          </p>
        ) : null}
      </>
    );
  }

  return (
    <div className="citation-block">
      <div className="citation-block__head">
        <span className="citation-block__title">引用</span>
        <span className="citation-block__status">
          {CITATION_STATUS_LABELS[citationStatus] ??
            (citationStatus ? `citation_status = ${citationStatus}` : 'citation_status 未返回')}
        </span>
      </div>
      <ul className="citation-list">
        {citations.map((citation, index) => {
          const invalid = citation.label ? invalidCitations.includes(citation.label) : false;
          return (
            <li
              key={`${citation.label ?? 'citation'}-${index}`}
              className={invalid ? 'citation-item is-invalid' : 'citation-item'}
            >
              <span className="citation-item__label">{citation.label ?? `#${index + 1}`}</span>
              <span className="citation-item__body">
                <span className="citation-item__doc">
                  {citation.docId ?? 'doc_id 未返回'}
                  {citation.chunkId ? ` · ${citation.chunkId}` : ''}
                </span>
                {citation.title ? (
                  <span className="citation-item__title">{citation.title}</span>
                ) : null}
              </span>
              {invalid ? <span className="citation-item__invalid">无效引用</span> : null}
            </li>
          );
        })}
      </ul>
      {invalidCitations.length > 0 ? (
        <p className="citation-warning">
          后端标记的无效引用：{invalidCitations.join('、')}
        </p>
      ) : null}
    </div>
  );
}
