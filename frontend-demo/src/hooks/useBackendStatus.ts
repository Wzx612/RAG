import { useCallback, useEffect, useRef, useState } from 'react';
import {
  checkHealth,
  checkReadiness,
  fetchAcceptanceSummary,
} from '../api/client';
import type {
  AcceptanceSummary,
  ApiOutcome,
  HealthResponse,
  ReadinessResponse,
} from '../api/types';

export type ConnectionState =
  | 'checking'
  | 'connected'
  | 'degraded'
  | 'unreachable'
  | 'error';

export interface BackendStatus {
  connection: ConnectionState;
  healthOutcome: ApiOutcome<HealthResponse> | null;
  readinessOutcome: ApiOutcome<ReadinessResponse> | null;
  lastCheckedAt: Date | null;
  isChecking: boolean;
  acceptance: ApiOutcome<AcceptanceSummary> | null;
  acceptanceLoading: boolean;
  refreshAll: () => void;
  refreshAcceptance: () => void;
}

function resolveConnection(
  health: ApiOutcome<HealthResponse>,
  readiness: ApiOutcome<ReadinessResponse>,
): ConnectionState {
  if (health.kind === 'network_error' || health.kind === 'timeout') {
    return 'unreachable';
  }
  if (health.kind !== 'success' || !health.data) {
    return 'error';
  }
  if (health.data.status !== 'ok') {
    return 'error';
  }
  if (readiness.kind === 'success' && readiness.data?.status === 'ready') {
    return 'connected';
  }
  return 'degraded';
}

export function useBackendStatus(): BackendStatus {
  const [connection, setConnection] = useState<ConnectionState>('checking');
  const [healthOutcome, setHealthOutcome] = useState<ApiOutcome<HealthResponse> | null>(null);
  const [readinessOutcome, setReadinessOutcome] =
    useState<ApiOutcome<ReadinessResponse> | null>(null);
  const [lastCheckedAt, setLastCheckedAt] = useState<Date | null>(null);
  const [isChecking, setIsChecking] = useState(true);

  const [acceptance, setAcceptance] = useState<ApiOutcome<AcceptanceSummary> | null>(null);
  const [acceptanceLoading, setAcceptanceLoading] = useState(true);

  const connectionController = useRef<AbortController | null>(null);
  const acceptanceController = useRef<AbortController | null>(null);
  const connectionRun = useRef(0);
  const acceptanceRun = useRef(0);
  const mounted = useRef(true);

  const refreshAcceptance = useCallback(() => {
    acceptanceController.current?.abort();
    const controller = new AbortController();
    acceptanceController.current = controller;
    const run = ++acceptanceRun.current;
    setAcceptanceLoading(true);

    void fetchAcceptanceSummary(controller.signal).then((outcome) => {
      if (!mounted.current || run !== acceptanceRun.current) return;
      setAcceptance(outcome);
      setAcceptanceLoading(false);
    });
  }, []);

  const refreshAll = useCallback(() => {
    connectionController.current?.abort();
    const controller = new AbortController();
    connectionController.current = controller;
    const run = ++connectionRun.current;

    setIsChecking(true);
    setConnection('checking');

    void Promise.all([
      checkHealth(controller.signal),
      checkReadiness(controller.signal),
    ] as const).then(([health, readiness]) => {
      if (!mounted.current || run !== connectionRun.current) return;
      setHealthOutcome(health);
      setReadinessOutcome(readiness);
      setConnection(
        health.kind === 'aborted' || readiness.kind === 'aborted'
          ? 'checking'
          : resolveConnection(health, readiness),
      );
      setLastCheckedAt(new Date());
      setIsChecking(false);
    });

    refreshAcceptance();
  }, [refreshAcceptance]);

  useEffect(() => {
    mounted.current = true;
    refreshAll();
    return () => {
      mounted.current = false;
      connectionController.current?.abort();
      acceptanceController.current?.abort();
    };
  }, [refreshAll]);

  return {
    connection,
    healthOutcome,
    readinessOutcome,
    lastCheckedAt,
    isChecking,
    acceptance,
    acceptanceLoading,
    refreshAll,
    refreshAcceptance,
  };
}
