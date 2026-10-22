import { useCallback, useEffect, useRef, useState } from 'react';
import { generateAnswer, searchKnowledge } from '../api/client';
import { MAX_QUERY_CHARS } from '../api/config';
import type { ApiOutcome, AnswerResponse, SearchResponse } from '../api/types';

export type TurnStage =
  | 'searching'
  | 'generating'
  | 'done'
  | 'search_failed';

export interface ChatTurn {
  id: string;
  query: string;
  topK: number;
  createdAt: Date;
  stage: TurnStage;
  searchOutcome: ApiOutcome<SearchResponse> | null;
  answerOutcome: ApiOutcome<AnswerResponse> | null;
  answerSkippedReason: string | null;
}

export interface LastSearchAttempt {
  query: string;
  topK: number;
  at: Date;
  outcome: ApiOutcome<SearchResponse>;
}

export type ChatPhase = 'idle' | 'searching' | 'generating';

export interface Conversation {
  turns: ChatTurn[];
  phase: ChatPhase;
  isBusy: boolean;
  lastSearchAttempt: LastSearchAttempt | null;
  send: (query: string, topK: number) => void;
  clear: () => void;
}

function makeId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `turn-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function shouldSkipAnswer(outcome: ApiOutcome<SearchResponse>): boolean {
  return (
    outcome.kind === 'network_error' ||
    outcome.kind === 'timeout' ||
    outcome.kind === 'aborted'
  );
}

export function useConversation(): Conversation {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [phase, setPhase] = useState<ChatPhase>('idle');
  const [lastSearchAttempt, setLastSearchAttempt] = useState<LastSearchAttempt | null>(null);

  const controllerRef = useRef<AbortController | null>(null);
  const runRef = useRef(0);
  const busyRef = useRef(false);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      runRef.current += 1;
      controllerRef.current?.abort();
    };
  }, []);

  const updateTurn = useCallback((id: string, patch: Partial<ChatTurn>) => {
    setTurns((current) =>
      current.map((turn) => (turn.id === id ? { ...turn, ...patch } : turn)),
    );
  }, []);

  const send = useCallback(
    (rawQuery: string, topK: number) => {
      const query = rawQuery.trim();
      if (!query || query.length > MAX_QUERY_CHARS || busyRef.current) return;

      busyRef.current = true;
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;
      const run = ++runRef.current;
      const id = makeId();
      const isCurrent = () => mountedRef.current && run === runRef.current;

      const turn: ChatTurn = {
        id,
        query,
        topK,
        createdAt: new Date(),
        stage: 'searching',
        searchOutcome: null,
        answerOutcome: null,
        answerSkippedReason: null,
      };

      setTurns((current) => [...current, turn]);
      setPhase('searching');

      void (async () => {
        const searchOutcome = await searchKnowledge(query, topK, controller.signal);
        if (!isCurrent()) return;

        setLastSearchAttempt({ query, topK, at: new Date(), outcome: searchOutcome });

        if (searchOutcome.kind === 'aborted') {
          updateTurn(id, {
            stage: 'search_failed',
            searchOutcome,
            answerSkippedReason: '检索请求已取消，未调用 /answer。',
          });
          busyRef.current = false;
          setPhase('idle');
          return;
        }

        if (shouldSkipAnswer(searchOutcome)) {
          updateTurn(id, {
            stage: 'search_failed',
            searchOutcome,
            answerSkippedReason: '检索请求未完成，未调用 /answer。',
          });
          busyRef.current = false;
          setPhase('idle');
          return;
        }

        updateTurn(id, { stage: 'generating', searchOutcome });
        setPhase('generating');

        const answerOutcome = await generateAnswer(query, undefined, controller.signal);
        if (!isCurrent()) return;

        updateTurn(id, { stage: 'done', answerOutcome });
        busyRef.current = false;
        setPhase('idle');
      })();
    },
    [updateTurn],
  );

  const clear = useCallback(() => {
    runRef.current += 1;
    controllerRef.current?.abort();
    controllerRef.current = null;
    busyRef.current = false;
    setTurns([]);
    setPhase('idle');
    setLastSearchAttempt(null);
  }, []);

  return {
    turns,
    phase,
    isBusy: phase !== 'idle',
    lastSearchAttempt,
    send,
    clear,
  };
}
