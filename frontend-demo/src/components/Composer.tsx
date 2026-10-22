import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { MAX_QUERY_CHARS, TOP_K_OPTIONS } from '../api/config';
import { SendIcon } from './icons';

interface ComposerProps {
  isBusy: boolean;
  topK: number;
  onTopKChange: (value: number) => void;
  onSend: (query: string) => void;
  onClear: () => void;
  hasTurns: boolean;
}

export function Composer({
  isBusy,
  topK,
  onTopKChange,
  onSend,
  onClear,
  hasTurns,
}: ComposerProps) {
  const [value, setValue] = useState('');
  const composingRef = useRef(false);
  const compositionEndAtRef = useRef(0);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = 'auto';
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
  }, [value]);

  const submit = () => {
    const query = value.trim();
    if (!query || isBusy) return;
    onSend(query);
    setValue('');
    requestAnimationFrame(() => textareaRef.current?.focus());
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter') return;
    if (event.shiftKey) return;
    const native = event.nativeEvent;
    if (composingRef.current || native.isComposing || native.keyCode === 229) return;
    // Some browsers fire compositionend before the Enter keydown that confirms
    // the IME candidate; ignore that same keystroke instead of sending.
    if (Date.now() - compositionEndAtRef.current < 80) return;
    event.preventDefault();
    submit();
  };

  return (
    <div className="composer">
      <div className="composer__inner">
        <div className="composer__toolbar">
          <div className="topk-control">
            <span className="topk-control__label">检索 Top-K</span>
            <div className="segmented" role="group" aria-label="检索 Top-K">
              {TOP_K_OPTIONS.map((option) => (
                <button
                  key={option}
                  type="button"
                  className={option === topK ? 'segmented__item is-active' : 'segmented__item'}
                  aria-pressed={option === topK}
                  disabled={isBusy}
                  onClick={() => onTopKChange(option)}
                >
                  {option}
                </button>
              ))}
            </div>
            <span
              className="topk-control__hint"
              title="/answer 不接受 Top-K 参数，其证据条数由服务端 APEX_RAG_TOP_K 配置决定。"
            >
              仅影响检索证据；/answer 使用服务端配置
            </span>
          </div>

          <button
            type="button"
            className="link-button link-button--muted"
            onClick={onClear}
            disabled={!hasTurns}
            title="清空当前页面的全部对话，并取消进行中的请求"
          >
            清空对话
          </button>
        </div>

        <div className="composer__box">
          <label className="sr-only" htmlFor="chat-input">
            输入你的信用卡问题
          </label>
          <textarea
            id="chat-input"
            ref={textareaRef}
            className="composer__textarea"
            rows={1}
            value={value}
            maxLength={MAX_QUERY_CHARS}
            placeholder="询问信用卡费用、权益、免息期或相关政策…"
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={handleKeyDown}
            onCompositionStart={() => {
              composingRef.current = true;
            }}
            onCompositionEnd={() => {
              composingRef.current = false;
              compositionEndAtRef.current = Date.now();
            }}
            aria-describedby="composer-help"
          />
          <button
            type="button"
            className="send-button"
            onClick={submit}
            disabled={isBusy || value.trim() === ''}
            aria-label="发送问题"
            title={isBusy ? '正在处理上一条问题' : '发送问题'}
          >
            <SendIcon />
          </button>
        </div>

        <div className="composer__footer" id="composer-help">
          <span>{isBusy ? '正在处理上一条问题…' : 'Enter 发送 · Shift + Enter 换行'}</span>
          <span className={value.length > MAX_QUERY_CHARS - 100 ? 'char-count is-near' : 'char-count'}>
            {value.length} / {MAX_QUERY_CHARS}
          </span>
        </div>
      </div>
    </div>
  );
}
