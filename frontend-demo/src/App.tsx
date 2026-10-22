import { useMemo, useState } from 'react';
import { API_BASE_URL, DEFAULT_TOP_K } from './api/config';
import type { AnswerResponse, ApiOutcome } from './api/types';
import { AppHeader } from './components/AppHeader';
import { Composer } from './components/Composer';
import { ConnectionBanner } from './components/ConnectionBanner';
import { ExternalLinkIcon } from './components/icons';
import { MessageList } from './components/MessageList';
import { TechDrawer } from './components/TechDrawer';
import { useBackendStatus } from './hooks/useBackendStatus';
import { useConversation } from './hooks/useConversation';
import { isSafeExternalUrl } from './utils/format';

const BACKEND_ORIGIN = API_BASE_URL || 'http://localhost:8000';
const BACKEND_LINKS_ENABLED = isSafeExternalUrl(`${BACKEND_ORIGIN}/acceptance/`);

export default function App() {
  const backend = useBackendStatus();
  const conversation = useConversation();
  const [techOpen, setTechOpen] = useState(false);
  const [topK, setTopK] = useState(DEFAULT_TOP_K);

  const lastAnswerOutcome = useMemo<ApiOutcome<AnswerResponse> | null>(() => {
    const lastTurn = conversation.turns[conversation.turns.length - 1];
    return lastTurn?.answerOutcome ?? null;
  }, [conversation.turns]);

  const handleSend = (query: string) => {
    conversation.send(query, topK);
  };

  return (
    <div className="app-shell">
      <AppHeader
        connection={backend.connection}
        isChecking={backend.isChecking}
        lastCheckedAt={backend.lastCheckedAt}
        onRefresh={backend.refreshAll}
        onOpenTech={() => setTechOpen(true)}
      />

      <main className="app-main">
        <ConnectionBanner
          connection={backend.connection}
          healthOutcome={backend.healthOutcome}
          readinessOutcome={backend.readinessOutcome}
          onRetry={backend.refreshAll}
          onOpenTech={() => setTechOpen(true)}
        />

        <section
          className="chat-card"
          aria-label="信用卡智能客服对话"
          aria-busy={conversation.isBusy}
        >
          <div className="chat-card__head">
            <div className="chat-card__heading">
              <span className="chat-card__dot" data-state={conversation.phase} aria-hidden="true" />
              <div>
                <strong className="chat-card__title">在线咨询</strong>
                <span className="chat-card__subtitle">
                  生成回答 + 检索证据 · 真实调用本地 RAG 后端
                </span>
              </div>
            </div>
            <span className="chat-card__meta">
              {conversation.turns.length > 0
                ? `${conversation.turns.length} 轮对话 · 仅保存在当前页面`
                : '新对话 · 无服务端会话'}
            </span>
          </div>

          <MessageList
            turns={conversation.turns}
            phase={conversation.phase}
            onPickExample={handleSend}
          />

          <Composer
            isBusy={conversation.isBusy}
            topK={topK}
            onTopKChange={setTopK}
            onSend={handleSend}
            onClear={conversation.clear}
            hasTurns={conversation.turns.length > 0}
          />
        </section>

        <footer className="app-footer">
          <span>
            面试 Demo · 非生产系统 · 知识库为英文信用卡政策语料 · 回答与引用均来自本地后端
          </span>
          <span className="app-footer__links">
            {BACKEND_LINKS_ENABLED ? (
              <>
                <a
                  href={`${BACKEND_ORIGIN}/acceptance/`}
                  target="_blank"
                  rel="noreferrer noopener"
                >
                  验收看板
                  <ExternalLinkIcon />
                </a>
                <a href={`${BACKEND_ORIGIN}/docs`} target="_blank" rel="noreferrer noopener">
                  API 文档
                  <ExternalLinkIcon />
                </a>
              </>
            ) : null}
            <button type="button" className="footer-link-button" onClick={() => setTechOpen(true)}>
              技术详情
            </button>
          </span>
        </footer>
      </main>

      <TechDrawer
        open={techOpen}
        onClose={() => setTechOpen(false)}
        backend={backend}
        lastSearchAttempt={conversation.lastSearchAttempt}
        lastAnswerOutcome={lastAnswerOutcome}
      />
    </div>
  );
}
