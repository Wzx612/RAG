import { BrandMark, ArrowUpRightIcon } from './icons';

const EXAMPLES = [
  'What is the foreign transaction fee?',
  'What are the annual fees for this credit card?',
  'What is the interest-free period?',
  'Which benefits are available for this card?',
];

interface WelcomePanelProps {
  disabled: boolean;
  onPickExample: (question: string) => void;
}

export function WelcomePanel({ disabled, onPickExample }: WelcomePanelProps) {
  return (
    <section className="welcome" aria-labelledby="welcome-title">
      <div className="welcome__mark" aria-hidden="true">
        <BrandMark />
      </div>
      <h1 className="welcome__title" id="welcome-title">
        你好，我是信用卡智能客服助手。你可以询问信用卡费用、权益、免息期及相关政策，我会检索知识库并展示对应的依据。
      </h1>
      <p className="welcome__lede">
        每次提问都会真实调用本地 RAG 后端：先检索 Top-K 文档，再尝试生成回答。没有命中或生成失败时，页面会如实说明。
      </p>

      <div className="welcome__examples" role="list" aria-label="示例问题">
        {EXAMPLES.map((question) => (
          <div key={question} role="listitem" className="example-chip-wrap">
            <button
              type="button"
              className="example-chip"
              disabled={disabled}
              onClick={() => onPickExample(question)}
            >
              <span className="example-chip__text">{question}</span>
              <ArrowUpRightIcon className="example-chip__icon" />
            </button>
          </div>
        ))}
      </div>

      <p className="welcome__note">
        当前知识库为英文信用卡政策语料（150 篇文档），目前仅验证英文查询效果；中文问题可以提交，但检索相关性和回答质量未经验证。示例按钮会直接发起真实请求，不会填充预设答案。
      </p>

      <div className="welcome__pipeline" aria-label="检索链路">
        <span>Dense</span>
        <span aria-hidden="true">+</span>
        <span>Sparse</span>
        <span aria-hidden="true">→</span>
        <span>RRF</span>
        <span aria-hidden="true">→</span>
        <span>Reranker</span>
        <span aria-hidden="true">→</span>
        <span>Top-K</span>
      </div>
    </section>
  );
}
