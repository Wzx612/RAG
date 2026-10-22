import type { ChatPhase, ChatTurn } from '../hooks/useConversation';
import { useAutoScroll } from '../hooks/useAutoScroll';
import { AssistantMessage } from './AssistantMessage';
import { UserMessage } from './UserMessage';
import { WelcomePanel } from './WelcomePanel';

interface MessageListProps {
  turns: ChatTurn[];
  phase: ChatPhase;
  onPickExample: (question: string) => void;
}

export function MessageList({ turns, phase, onPickExample }: MessageListProps) {
  const lastTurn = turns.length > 0 ? turns[turns.length - 1] : null;
  const scrollKey = [
    turns.length,
    phase,
    lastTurn?.stage ?? '',
    lastTurn?.searchOutcome ? 'search' : '',
    lastTurn?.answerOutcome ? 'answer' : '',
  ].join('|');

  const scrollRef = useAutoScroll<HTMLDivElement>(scrollKey, turns.length > 0);

  return (
    <div className="chat-scroll" ref={scrollRef}>
      <div className="chat-scroll__inner">
        {turns.length === 0 ? (
          <WelcomePanel disabled={phase !== 'idle'} onPickExample={onPickExample} />
        ) : (
          <div className="turns" role="log" aria-live="polite" aria-label="对话记录">
            {turns.map((turn) => (
              <div className="turn" key={turn.id}>
                <UserMessage turn={turn} />
                <AssistantMessage turn={turn} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
