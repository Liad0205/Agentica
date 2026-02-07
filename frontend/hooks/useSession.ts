/**
 * Custom hook for session lifecycle management.
 * Handles creating, canceling, and resetting sessions,
 * along with WebSocket connection management.
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useStore } from "@/lib/store";
import { ApiError, api } from "@/lib/api";
import {
  wsClient,
  type ConnectionStatus,
} from "@/lib/websocket";
import type {
  Mode,
  ModelConfig,
  SessionDetailResponse,
  SessionStatus,
  SessionSummary,
} from "@/lib/types";

const ACTIVE_SESSION_STORAGE_KEY =
  "agent_arena.active_session_id";
const CONTINUABLE_SESSION_STATUSES: SessionStatus[] = [
  "complete",
  "error",
];
const ACTIVE_SESSION_STATUSES: SessionStatus[] = [
  "started",
  "running",
];

class SessionActionRequiredError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SessionActionRequiredError";
  }
}

let unsubscribeEventGlobal: (() => void) | null = null;
let unsubscribeStatusGlobal: (() => void) | null = null;
let restoreInFlight: Promise<void> | null = null;
let mountedConsumers = 0;

function getPersistedSessionId(): string | null {
  if (typeof window === "undefined") {
    return null;
  }
  return window.localStorage.getItem(
    ACTIVE_SESSION_STORAGE_KEY,
  );
}

function persistSessionId(sessionId: string): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(
    ACTIVE_SESSION_STORAGE_KEY,
    sessionId,
  );
}

function clearPersistedSessionId(): void {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.removeItem(
    ACTIVE_SESSION_STORAGE_KEY,
  );
}

function ensureSubscriptions(): void {
  if (unsubscribeEventGlobal && unsubscribeStatusGlobal) {
    return;
  }

  unsubscribeEventGlobal = wsClient.onEvent((event) => {
    useStore.getState().processEvent(event);
  });

  unsubscribeStatusGlobal = wsClient.onStatusChange(
    (connectionStatus: ConnectionStatus) => {
      const store = useStore.getState();
      store.setConnectionStatus(connectionStatus);
      if (connectionStatus === "error") {
        store.setConnectionError(
          "WebSocket connection failed",
        );
      } else if (connectionStatus === "connected") {
        store.setConnectionError(null);
      }
    },
  );
}

function cleanupSubscriptions(): void {
  if (unsubscribeEventGlobal) {
    unsubscribeEventGlobal();
    unsubscribeEventGlobal = null;
  }
  if (unsubscribeStatusGlobal) {
    unsubscribeStatusGlobal();
    unsubscribeStatusGlobal = null;
  }
}

function applySessionDetail(
  detail: SessionDetailResponse,
  options?: { allowTerminalReplay?: boolean },
): void {
  const store = useStore.getState();

  store.setMode(detail.mode);
  store.setTask(detail.task);
  store.startSession(detail.session_id);
  if (detail.status !== "started") {
    store.endSession(detail.status);
  }

  persistSessionId(detail.session_id);

  // Connect the WebSocket for active sessions and optional terminal replay.
  // We do not auto-replay terminal sessions during storage restore because
  // those sessions may exist only in persistence (not in-memory WS stream),
  // which causes immediate WS errors.
  const isActive =
    detail.status === "started" ||
    detail.status === "running";
  const allowTerminalReplay =
    options?.allowTerminalReplay ?? false;
  const needsReplay =
    allowTerminalReplay &&
    useStore.getState().events.length === 0 &&
    (detail.status === "complete" ||
      detail.status === "error" ||
      detail.status === "cancelled");

  if (isActive || needsReplay) {
    ensureSubscriptions();
    wsClient.connect(detail.session_id);

    // Handle WebSocket connection failure by clearing stale session.
    // This can happen if the session exists in the database but the backend
    // has no in-memory state for it (e.g., after backend restart).
    const unsubscribeError = wsClient.onStatusChange(
      (status) => {
        if (status === "error") {
          console.warn(
            "[Session] WebSocket connection failed, clearing stale session",
            { sessionId: detail.session_id },
          );
          clearPersistedSessionId();
          useStore.getState().resetSession();
          useStore.getState().setConnectionError(null);
          unsubscribeError();
        } else if (status === "connected") {
          // Connection succeeded, no cleanup needed
          unsubscribeError();
        }
      },
    );
  }
}

function sortSessionsByCreatedAt(
  sessions: SessionSummary[],
): SessionSummary[] {
  return [...sessions].sort(
    (a, b) => b.created_at - a.created_at,
  );
}

async function restoreSessionFromStorage(): Promise<void> {
  if (useStore.getState().sessionId) {
    return;
  }

  const persistedSessionId = getPersistedSessionId();
  if (!persistedSessionId) {
    return;
  }

  try {
    const detail = await api.getSession(persistedSessionId);
    applySessionDetail(detail, {
      allowTerminalReplay: false,
    });
  } catch (error) {
    clearPersistedSessionId();
    if (
      !(error instanceof ApiError && error.status === 404)
    ) {
      console.warn(
        "Failed to restore persisted session:",
        error,
      );
    }
  }
}

/**
 * Return type for the useSession hook.
 */
export interface UseSessionReturn {
  sessionId: string | null;
  status: SessionStatus;
  isRunning: boolean;
  mode: Mode;
  task: string;
  recentSessions: SessionSummary[];
  setMode: (mode: Mode) => void;
  startSession: (
    task: string,
    modelConfig?: ModelConfig,
  ) => Promise<void>;
  switchSession: (sessionId: string) => Promise<void>;
  refreshSessions: () => Promise<void>;
  clearSessions: (force?: boolean) => Promise<void>;
  cancelSession: () => Promise<void>;
  resetSession: () => void;
}

export interface UseSessionOptions {
  withSessionList?: boolean;
}

/**
 * Hook for managing session lifecycle including creation, cancellation,
 * and WebSocket event streaming.
 */
export function useSession(
  options: UseSessionOptions = {},
): UseSessionReturn {
  const withSessionList = options.withSessionList ?? false;
  const sessionId = useStore((state) => state.sessionId);
  const status = useStore((state) => state.sessionStatus);
  const mode = useStore((state) => state.mode);
  const task = useStore((state) => state.task);
  const [recentSessions, setRecentSessions] = useState<
    SessionSummary[]
  >([]);

  const setMode = useStore((state) => state.setMode);
  const setTask = useStore((state) => state.setTask);
  const storeStartSession = useStore(
    (state) => state.startSession,
  );
  const storeEndSession = useStore(
    (state) => state.endSession,
  );
  const storeResetSession = useStore(
    (state) => state.resetSession,
  );
  const setConnectionError = useStore(
    (state) => state.setConnectionError,
  );
  const isMountedRef = useRef(false);

  const refreshSessions =
    useCallback(async (): Promise<void> => {
      if (!withSessionList) {
        return;
      }

      try {
        const sessions = await api.listSessions(25);
        setRecentSessions(
          sortSessionsByCreatedAt(sessions),
        );
      } catch (error) {
        console.warn(
          "Failed to refresh session list:",
          error,
        );
      }
    }, [withSessionList]);

  // Initialize one-time restore and clean global subscriptions when last consumer unmounts.
  useEffect(() => {
    if (!isMountedRef.current) {
      mountedConsumers += 1;
      isMountedRef.current = true;
    }

    if (!restoreInFlight) {
      restoreInFlight = restoreSessionFromStorage().finally(
        () => {
          restoreInFlight = null;
        },
      );
    }

    return () => {
      if (isMountedRef.current) {
        mountedConsumers = Math.max(
          0,
          mountedConsumers - 1,
        );
        isMountedRef.current = false;
      }

      if (mountedConsumers === 0) {
        cleanupSubscriptions();
      }
    };
  }, []);

  // Terminal session states should not keep trying to reconnect.
  // We use an idle detection strategy: after the terminal state is reached,
  // we wait for events to stop arriving for 500ms (indicating replay is done).
  // This is capped at 10s as a safety net for sessions with many events.
  useEffect(() => {
    if (
      status === "complete" ||
      status === "error" ||
      status === "cancelled"
    ) {
      const lastEventTimeRef = { current: Date.now() };
      const IDLE_WINDOW_MS = 500;
      const MAX_WAIT_MS = 10000;
      const CHECK_INTERVAL_MS = 200;
      const startTime = Date.now();

      let previousEventCount =
        useStore.getState().events.length;

      const checkIdle = (): void => {
        const now = Date.now();
        const currentEventCount =
          useStore.getState().events.length;

        // Update last event time if new events arrived
        if (currentEventCount > previousEventCount) {
          lastEventTimeRef.current = now;
          previousEventCount = currentEventCount;
        }

        const timeSinceLastEvent =
          now - lastEventTimeRef.current;
        const totalElapsed = now - startTime;

        // Disconnect if idle window passed OR max wait reached
        if (
          timeSinceLastEvent >= IDLE_WINDOW_MS ||
          totalElapsed >= MAX_WAIT_MS
        ) {
          wsClient.disconnect();
          cleanupSubscriptions();
          return;
        }

        // Continue checking
        timeoutIdRef.current = setTimeout(
          checkIdle,
          CHECK_INTERVAL_MS,
        );
      };

      const timeoutIdRef: {
        current: ReturnType<typeof setTimeout> | null;
      } = {
        current: setTimeout(checkIdle, CHECK_INTERVAL_MS),
      };

      return () => {
        if (timeoutIdRef.current !== null) {
          clearTimeout(timeoutIdRef.current);
        }
      };
    }
    return;
  }, [status]);

  useEffect(() => {
    if (!withSessionList) {
      return;
    }
    void refreshSessions();
  }, [withSessionList, refreshSessions, sessionId]);

  /**
   * Start a session run with the given task and optional model configuration.
   * Creates a new session by default, or continues the existing one when possible.
   */
  const startSession = useCallback(
    async (
      taskText: string,
      modelConfig?: ModelConfig,
    ): Promise<void> => {
      // Store the task
      setTask(taskText);

      try {
        // Read state at call time to avoid stale closure values.
        const store = useStore.getState();
        const currentMode = store.mode;
        const existingSessionId = store.sessionId;
        let shouldContinue = false;

        if (existingSessionId) {
          let existingSession: SessionDetailResponse;
          try {
            existingSession = await api.getSession(
              existingSessionId,
            );
          } catch (error) {
            if (
              error instanceof ApiError &&
              error.status === 404
            ) {
              throw new SessionActionRequiredError(
                'The selected session no longer exists. Click "New Session" to start a new one.',
              );
            }
            if (error instanceof Error) {
              throw error;
            }
            throw new Error(
              "Failed to resolve current session",
            );
          }

          if (existingSession.mode !== currentMode) {
            // Mode changed — discard old session reference so we create a
            // fresh one below.  Do NOT call storeResetSession() here; the
            // workspace state will be replaced when startSession() sets the
            // new session via storeStartSession(..., preserveWorkspace: false).
            // Resetting eagerly would wipe the visible workspace before the
            // new session is even created, making it look like the project
            // was deleted.
            clearPersistedSessionId();
          } else {
            if (
              ACTIVE_SESSION_STATUSES.includes(
                existingSession.status,
              )
            ) {
              throw new SessionActionRequiredError(
                "The current session is still running. Wait for completion or cancel it before sending another prompt.",
              );
            }

            if (existingSession.status === "cancelled") {
              throw new SessionActionRequiredError(
                'Cancelled sessions cannot be resumed. Click "New Session" to start a fresh run.',
              );
            }

            shouldContinue =
              CONTINUABLE_SESSION_STATUSES.includes(
                existingSession.status,
              );

            if (!shouldContinue) {
              throw new SessionActionRequiredError(
                'This session cannot be resumed from its current state. Click "New Session" to start a fresh run.',
              );
            }

            const response = await api.continueSession(
              existingSessionId,
              {
                task: taskText,
                modelConfig,
              },
            );

            // Update store with session ID
            storeStartSession(response.session_id, {
              preserveWorkspace:
                response.session_id === existingSessionId,
            });
            persistSessionId(response.session_id);

            // Ensure exactly one global set of subscriptions.
            ensureSubscriptions();
            wsClient.connect(response.session_id);

            if (withSessionList) {
              void refreshSessions();
            }
            return;
          }
        }

        const response = await api.createSession({
          mode: currentMode,
          task: taskText,
          modelConfig,
        });

        // Update store with session ID
        storeStartSession(response.session_id, {
          preserveWorkspace: false,
        });
        persistSessionId(response.session_id);

        // Ensure exactly one global set of subscriptions.
        ensureSubscriptions();
        wsClient.connect(response.session_id);

        if (withSessionList) {
          void refreshSessions();
        }
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : "Failed to start session";
        setConnectionError(message);
        if (
          !(error instanceof SessionActionRequiredError)
        ) {
          storeEndSession("error");
        }
        throw error;
      }
    },
    [
      setTask,
      storeStartSession,
      setConnectionError,
      storeEndSession,
      withSessionList,
      refreshSessions,
    ],
  );

  const switchSession = useCallback(
    async (targetSessionId: string): Promise<void> => {
      if (!targetSessionId) {
        return;
      }

      const detail = await api.getSession(targetSessionId);
      applySessionDetail(detail, {
        allowTerminalReplay: true,
      });

      if (withSessionList) {
        await refreshSessions();
      }
    },
    [withSessionList, refreshSessions],
  );

  /**
   * Cancel the current running session.
   * Sends cancel command via both WebSocket and HTTP API.
   */
  const cancelSession =
    useCallback(async (): Promise<void> => {
      if (!sessionId) {
        return;
      }

      // Send cancel via WebSocket (for immediate response)
      wsClient.send({ type: "cancel" });

      // Also send via HTTP API (for reliability)
      try {
        await api.cancelSession(sessionId);
      } catch (error) {
        // Ignore already-terminal responses, surface other failures.
        if (
          error instanceof ApiError &&
          (error.status === 400 || error.status === 404)
        ) {
          // Session was already terminal or unavailable.
        } else {
          const message =
            error instanceof Error
              ? error.message
              : "Failed to cancel session";
          setConnectionError(message);
        }
      }

      // Update session status
      storeEndSession("cancelled");

      // Disconnect WebSocket
      wsClient.disconnect();

      cleanupSubscriptions();

      if (withSessionList) {
        void refreshSessions();
      }
    }, [
      sessionId,
      storeEndSession,
      setConnectionError,
      withSessionList,
      refreshSessions,
    ]);

  const clearSessions = useCallback(
    async (force: boolean = true): Promise<void> => {
      await api.clearSessions(force);
      wsClient.disconnect();
      cleanupSubscriptions();
      clearPersistedSessionId();
      storeResetSession();
      if (withSessionList) {
        await refreshSessions();
      }
    },
    [storeResetSession, withSessionList, refreshSessions],
  );

  /**
   * Reset the session state to start fresh.
   */
  const resetSession = useCallback((): void => {
    // Disconnect WebSocket if connected
    wsClient.disconnect();

    cleanupSubscriptions();
    clearPersistedSessionId();

    // Reset store state
    storeResetSession();
    if (withSessionList) {
      void refreshSessions();
    }
  }, [storeResetSession, withSessionList, refreshSessions]);

  const isRunning =
    status === "started" || status === "running";

  return {
    sessionId,
    status,
    isRunning,
    mode,
    task,
    recentSessions,
    setMode,
    startSession,
    switchSession,
    refreshSessions,
    clearSessions,
    cancelSession,
    resetSession,
  };
}
