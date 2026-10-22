import { InfoIcon } from './icons';

const STEPS = [
  {
    name: 'Query',
    description: '用户问题作为检索输入，原样调用后端接口。',
  },
  {
    name: 'RRF',
    description: 'Reciprocal Rank Fusion 按排名融合 Dense 与 Sparse 候选。',
  },
  {
    name: 'Reranker',
    description: 'Cross-Encoder 精排候选；仓库默认模型为 BAAI/bge-reranker-base。',
  },
  {
    name: 'Top-K Results',
    description: '按精排结果返回 Top-K 文档，作为回答与展示依据。',
  },
];

export function TechArchitecture() {
  return (
    <div className="architecture">
      <p className="arch-formula">
        Query → Dense Retrieval + Sparse Retrieval → RRF → Reranker → Top-K Results
      </p>
      <ol className="arch-flow">
        <li className="arch-node">
          <span className="arch-node__index">1</span>
          <div>
            <strong>{STEPS[0].name}</strong>
            <p>{STEPS[0].description}</p>
          </div>
        </li>
        <li className="arch-split" aria-label="Dense Retrieval 与 Sparse Retrieval 并行召回">
          <div className="arch-node arch-node--half">
            <span className="arch-node__index">2a</span>
            <div>
              <strong>Dense Retrieval</strong>
              <p>pgvector 向量相似度召回。</p>
            </div>
          </div>
          <span className="arch-split__plus" aria-hidden="true">
            +
          </span>
          <div className="arch-node arch-node--half">
            <span className="arch-node__index">2b</span>
            <div>
              <strong>Sparse Retrieval</strong>
              <p>PostgreSQL 全文检索关键词召回。</p>
            </div>
          </div>
        </li>
        {STEPS.slice(1).map((step, index) => (
          <li className="arch-node" key={step.name}>
            <span className="arch-node__index">{index + 3}</span>
            <div>
              <strong>{step.name}</strong>
              <p>{step.description}</p>
            </div>
          </li>
        ))}
      </ol>

      <p className="tech-note tech-note--info">
        <InfoIcon />
        静态架构说明，不是本次请求的逐阶段执行明细。/api/v1/search 目前只返回
        retrieval_ms、base_ms、rerank_ms、total_ms，未提供 Dense、Sparse、RRF
        各自的分项实时耗时，因此这里不显示这些节点的实时状态。
      </p>
    </div>
  );
}
