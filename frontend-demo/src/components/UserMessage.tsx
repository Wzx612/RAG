import type { ChatTurn } from '../hooks/useConversation';
import { formatTimeOnly } from '../utils/format';

interface UserMessageProps {
  turn: ChatTurn;
}

export function UserMessage({ turn }: UserMessageProps) {
  return (
    <article className="msg msg--user" aria-label="你的问题">
      <div className="msg__user-meta">
        <span>你</span>
        <span className="msg__dot" aria-hidden="true">·</span>
        <span>{formatTimeOnly(turn.createdAt)}</span>
        <span className="msg__dot" aria-hidden="true">·</span>
        <span>Top-K {turn.topK}</span>
      </div>
      <div className="msg__user-bubble">{turn.query}</div>
    </article>
  );
}
