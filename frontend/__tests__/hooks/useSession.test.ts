/**
 * Tests for the useSession hook (hooks/useSession.ts).
 * Verifies session lifecycle, stale closure prevention, and WebSocket interactions.
 */

import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useSession } from "@/hooks/useSession";
import { useStore } from "@/lib/store";
import { wsClient } from "@/lib/websocket";
import { api } from "@/lib/api";

const ACTIVE_SESSION_STORAGE_KEY = "agent_arena.active_session_id";

function createLocalStorageMock(): Storage {
  const storage = new Map<string, string>();
  return {
    get length(): number {
      return storage.size;
    },
    clear(): void {
      storage.clear();
    },
    getItem(key: string): string | null {
      return storage.get(key) ?? null;
    },
    key(index: number): string | null {
      return Array.from(storage.keys())[index] ?? null;
    },
    removeItem(key: string): void {
      storage.delete(key);
    },
    setItem(key: string, value: string): void {
      storage.set(key, value);
    },
  };
}

// Mock the API module
vi.mock("@/lib/api", () => ({
  api: {
    createSession: vi.fn(),
    continueSession: vi.fn(),
    listSessions: vi.fn(),
    clearSessions: vi.fn(),
    cancelSession: vi.fn(),
    getSession: vi.fn(),
  },
  ApiError: class ApiError extends Error {
    status: number;
    detail?: string;
    constructor(message: string, status: number, detail?: string) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.detail = detail;
    }
  },
}));

// Mock the WebSocket client
vi.mock("@/lib/websocket", () => ({
  wsClient: {
    connect: vi.fn(),
    disconnect: vi.fn(),
    send: vi.fn(),
    onEvent: vi.fn(() => vi.fn()),
    onStatusChange: vi.fn(() => vi.fn()),
    getStatus: vi.fn(() => "disconnected"),
  },
}));

beforeEach(() => {
  Object.defineProperty(window, "localStorage", {
    value: createLocalStorageMock(),
    configurable: true,
  });
  vi.clearAllMocks();
  vi.mocked(api.listSessions).mockResolvedValue([]);
  vi.mocked(api.clearSessions).mockResolvedValue({
    message: "All sessions cleared",
    in_memory_cleared: 0,
    running_cleared: 0,
    persisted_cleared: 0,
    forced: true,
  });
  useStore.getState().resetSession();
  window.localStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY);
});

// ---------------------------------------------------------------------------
// startSession
// ---------------------------------------------------------------------------
describe("startSession", () => {
  it("reads mode from store at call time (not stale closure)", async () => {
    // This is the key regression test: mode should be read via useStore.getState()
    // at call time, not captured as a stale closure value.
    const mockCreateSession = vi.mocked(api.createSession);
    mockCreateSession.mockResolvedValue({
      session_id: "sess-1",
      websocket_url: "/ws/sess-1",
      status: "started",
    });

    const { result } = renderHook(() => useSession());

    // Set mode to "decomposition" AFTER the hook renders
    act(() => {
      useStore.getState().setMode("decomposition");
    });

    // Start session - should use the current mode, not the stale one
    await act(async () => {
      await result.current.startSession("build app");
    });

    expect(mockCreateSession).toHaveBeenCalledWith(
      expect.objectContaining({
        mode: "decomposition",
        task: "build app",
      })
    );
  });

  it("creates session via API and connects WebSocket", async () => {
    const mockCreateSession = vi.mocked(api.createSession);
    mockCreateSession.mockResolvedValue({
      session_id: "sess-new",
      websocket_url: "/ws/sess-new",
      status: "started",
    });

    const { result } = renderHook(() => useSession());

    await act(async () => {
      await result.current.startSession("test task");
    });

    expect(mockCreateSession).toHaveBeenCalledOnce();
    expect(wsClient.connect).toHaveBeenCalledWith("sess-new");
    expect(wsClient.onEvent).toHaveBeenCalledOnce();
    expect(wsClient.onStatusChange).toHaveBeenCalledOnce();
  });

  it("continues an existing terminal session instead of creating a new one", async () => {
    useStore.setState({
      sessionId: "sess-existing",
      sessionStatus: "complete",
      mode: "react",
    });

    const mockContinueSession = vi.mocked(api.continueSession);
    mockContinueSession.mockResolvedValue({
      session_id: "sess-existing",
      websocket_url: "/ws/sess-existing",
      status: "started",
    });
    vi.mocked(api.getSession).mockResolvedValue({
      session_id: "sess-existing",
      mode: "react",
      status: "complete",
      task: "previous task",
      created_at: 1,
      started_at: 1,
      completed_at: 2,
      error_message: null,
      llm_config: null,
    });

    const { result } = renderHook(() => useSession());

    await act(async () => {
      await result.current.startSession("add auth flow");
    });

    expect(mockContinueSession).toHaveBeenCalledWith("sess-existing", {
      task: "add auth flow",
      modelConfig: undefined,
    });
    expect(api.createSession).not.toHaveBeenCalled();
    expect(wsClient.connect).toHaveBeenCalledWith("sess-existing");
  });

  it("requires explicit new session when existing session is not continuable", async () => {
    useStore.setState({
      sessionId: "sess-cancelled",
      sessionStatus: "cancelled",
      mode: "react",
    });

    vi.mocked(api.getSession).mockResolvedValue({
      session_id: "sess-cancelled",
      mode: "react",
      status: "cancelled",
      task: "prior task",
      created_at: 1,
      started_at: 1,
      completed_at: 2,
      error_message: null,
      llm_config: null,
    });

    const { result } = renderHook(() => useSession());

    await expect(
      act(async () => {
        await result.current.startSession("restart after cancel");
      })
    ).rejects.toThrow("Cancelled sessions cannot be resumed");

    expect(api.createSession).not.toHaveBeenCalled();
    expect(api.continueSession).not.toHaveBeenCalled();
    expect(useStore.getState().sessionStatus).toBe("cancelled");
  });

  it("auto-creates new session when selected mode differs from existing session mode", async () => {
    useStore.setState({
      sessionId: "sess-existing",
      sessionStatus: "complete",
      mode: "hypothesis",
    });

    vi.mocked(api.getSession).mockResolvedValue({
      session_id: "sess-existing",
      mode: "react",
      status: "complete",
      task: "previous task",
      created_at: 1,
      started_at: 1,
      completed_at: 2,
      error_message: null,
      llm_config: null,
    });

    vi.mocked(api.createSession).mockResolvedValue({
      session_id: "sess-new-hypothesis",
      status: "started",
    });

    const { result } = renderHook(() => useSession());

    await act(async () => {
      await result.current.startSession("run in hypothesis mode");
    });

    expect(api.createSession).toHaveBeenCalledWith({
      mode: "hypothesis",
      task: "run in hypothesis mode",
      modelConfig: undefined,
    });
    expect(api.continueSession).not.toHaveBeenCalled();
    expect(useStore.getState().sessionId).toBe("sess-new-hypothesis");
  });

  it("requires explicit new session when continue endpoint returns 404", async () => {
    useStore.setState({
      sessionId: "sess-gone",
      sessionStatus: "complete",
      mode: "react",
    });

    vi.mocked(api.getSession).mockResolvedValue({
      session_id: "sess-gone",
      mode: "react",
      status: "complete",
      task: "old task",
      created_at: 1,
      started_at: 1,
      completed_at: 2,
      error_message: null,
      llm_config: null,
    });

    const { ApiError } = await import("@/lib/api");
    vi.mocked(api.continueSession).mockRejectedValue(
      new ApiError("Session not found", 404, "Session not found")
    );

    const { result } = renderHook(() => useSession());

    await expect(
      act(async () => {
        await result.current.startSession("new task after restore");
      })
    ).rejects.toThrow("Session not found");

    expect(api.continueSession).toHaveBeenCalledWith("sess-gone", {
      task: "new task after restore",
      modelConfig: undefined,
    });
    expect(api.createSession).not.toHaveBeenCalled();
  });

  it("does not silently create a new session when continue returns 400", async () => {
    useStore.setState({
      sessionId: "sess-bad-state",
      sessionStatus: "complete",
      mode: "react",
    });

    vi.mocked(api.getSession).mockResolvedValue({
      session_id: "sess-bad-state",
      mode: "react",
      status: "complete",
      task: "old task",
      created_at: 1,
      started_at: 1,
      completed_at: 2,
      error_message: null,
      llm_config: null,
    });

    const { ApiError } = await import("@/lib/api");
    vi.mocked(api.continueSession).mockRejectedValue(
      new ApiError("invalid state", 400, "invalid state")
    );

    const { result } = renderHook(() => useSession());

    await expect(
      act(async () => {
        await result.current.startSession("try to continue");
      })
    ).rejects.toThrow("invalid state");

    expect(api.createSession).not.toHaveBeenCalled();
  });

  it("updates store with session ID on success", async () => {
    vi.mocked(api.createSession).mockResolvedValue({
      session_id: "sess-42",
      websocket_url: "/ws/sess-42",
      status: "started",
    });

    const { result } = renderHook(() => useSession());

    await act(async () => {
      await result.current.startSession("build a website");
    });

    expect(useStore.getState().sessionId).toBe("sess-42");
    expect(useStore.getState().sessionStatus).toBe("started");
  });

  it("sets error state if API call fails", async () => {
    vi.mocked(api.createSession).mockRejectedValue(new Error("Network error"));

    const { result } = renderHook(() => useSession());

    await expect(
      act(async () => {
        await result.current.startSession("fail task");
      })
    ).rejects.toThrow("Network error");

    expect(useStore.getState().sessionStatus).toBe("error");
    expect(useStore.getState().connectionError).toBe("Network error");
  });

  it("passes model config to API", async () => {
    const mockCreateSession = vi.mocked(api.createSession);
    mockCreateSession.mockResolvedValue({
      session_id: "sess-config",
      websocket_url: "/ws/sess-config",
      status: "started",
    });

    const { result } = renderHook(() => useSession());

    const modelConfig = {
      orchestrator: "gpt-4",
      subAgent: "gpt-3.5",
      temperature: 0.5,
    };

    await act(async () => {
      await result.current.startSession("build app", modelConfig);
    });

    expect(mockCreateSession).toHaveBeenCalledWith(
      expect.objectContaining({
        modelConfig,
      })
    );
  });
});

// ---------------------------------------------------------------------------
// cancelSession
// ---------------------------------------------------------------------------
describe("cancelSession", () => {
  it("sends cancel via both WebSocket and HTTP API", async () => {
    // Set up a running session
    useStore.setState({
      sessionId: "sess-running",
      sessionStatus: "running",
    });

    vi.mocked(api.cancelSession).mockResolvedValue(undefined);

    const { result } = renderHook(() => useSession());

    await act(async () => {
      await result.current.cancelSession();
    });

    expect(wsClient.send).toHaveBeenCalledWith({ type: "cancel" });
    expect(api.cancelSession).toHaveBeenCalledWith("sess-running");
  });

  it("disconnects WebSocket after cancel", async () => {
    useStore.setState({
      sessionId: "sess-running",
      sessionStatus: "running",
    });

    vi.mocked(api.cancelSession).mockResolvedValue(undefined);

    const { result } = renderHook(() => useSession());

    await act(async () => {
      await result.current.cancelSession();
    });

    expect(wsClient.disconnect).toHaveBeenCalled();
    expect(useStore.getState().sessionStatus).toBe("cancelled");
  });

  it("does nothing if no session ID", async () => {
    useStore.setState({ sessionId: null });

    const { result } = renderHook(() => useSession());

    await act(async () => {
      await result.current.cancelSession();
    });

    expect(wsClient.send).not.toHaveBeenCalled();
    expect(api.cancelSession).not.toHaveBeenCalled();
  });

  it("tolerates 400/404 errors from cancel API (session already terminal)", async () => {
    useStore.setState({
      sessionId: "sess-done",
      sessionStatus: "running",
    });

    const { ApiError } = await import("@/lib/api");
    vi.mocked(api.cancelSession).mockRejectedValue(
      new ApiError("already done", 400, "already done")
    );

    const { result } = renderHook(() => useSession());

    // Should NOT throw
    await act(async () => {
      await result.current.cancelSession();
    });

    // Should still mark as cancelled
    expect(useStore.getState().sessionStatus).toBe("cancelled");
    // Should NOT set a connection error for expected 400/404
    expect(useStore.getState().connectionError).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// resetSession
// ---------------------------------------------------------------------------
describe("resetSession", () => {
  it("disconnects WebSocket and resets store", () => {
    useStore.setState({
      sessionId: "sess-old",
      sessionStatus: "complete",
    });

    const { result } = renderHook(() => useSession());

    act(() => {
      result.current.resetSession();
    });

    expect(wsClient.disconnect).toHaveBeenCalled();
    expect(useStore.getState().sessionId).toBeNull();
    expect(useStore.getState().sessionStatus).toBe("idle");
  });

  it("preserves mode preference", () => {
    useStore.setState({
      mode: "hypothesis",
      sessionId: "sess-old",
    });

    const { result } = renderHook(() => useSession());

    act(() => {
      result.current.resetSession();
    });

    expect(useStore.getState().mode).toBe("hypothesis");
  });
});

// ---------------------------------------------------------------------------
// isRunning
// ---------------------------------------------------------------------------
describe("isRunning", () => {
  it("returns true when status is started or running", () => {
    act(() => {
      useStore.setState({ sessionStatus: "started" });
    });
    const { result: r1 } = renderHook(() => useSession());
    expect(r1.current.isRunning).toBe(true);

    act(() => {
      useStore.setState({ sessionStatus: "running" });
    });
    const { result: r2 } = renderHook(() => useSession());
    expect(r2.current.isRunning).toBe(true);
  });

  it("returns false for other statuses", () => {
    act(() => {
      useStore.setState({ sessionStatus: "idle" });
    });
    const { result: r1 } = renderHook(() => useSession());
    expect(r1.current.isRunning).toBe(false);

    act(() => {
      useStore.setState({ sessionStatus: "complete" });
    });
    const { result: r2 } = renderHook(() => useSession());
    expect(r2.current.isRunning).toBe(false);

    act(() => {
      useStore.setState({ sessionStatus: "error" });
    });
    const { result: r3 } = renderHook(() => useSession());
    expect(r3.current.isRunning).toBe(false);

    act(() => {
      useStore.setState({ sessionStatus: "cancelled" });
    });
    const { result: r4 } = renderHook(() => useSession());
    expect(r4.current.isRunning).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// setMode
// ---------------------------------------------------------------------------
describe("setMode", () => {
  it("updates the mode in the store", () => {
    const { result } = renderHook(() => useSession());

    act(() => {
      result.current.setMode("hypothesis");
    });

    expect(useStore.getState().mode).toBe("hypothesis");
  });
});

// ---------------------------------------------------------------------------
// Session list / switch
// ---------------------------------------------------------------------------
describe("session list and switching", () => {
  it("loads recent sessions when withSessionList is enabled", async () => {
    vi.mocked(api.listSessions).mockResolvedValue([
      {
        session_id: "sess-newer",
        mode: "react",
        status: "complete",
        task: "newer",
        created_at: 20,
        started_at: 20,
        completed_at: 25,
        error_message: null,
        metrics: {
          total_input_tokens: 1,
          total_output_tokens: 1,
          total_llm_calls: 1,
          total_tool_calls: 1,
          execution_time_seconds: 1,
        },
      },
      {
        session_id: "sess-older",
        mode: "react",
        status: "complete",
        task: "older",
        created_at: 10,
        started_at: 10,
        completed_at: 15,
        error_message: null,
        metrics: {
          total_input_tokens: 1,
          total_output_tokens: 1,
          total_llm_calls: 1,
          total_tool_calls: 1,
          execution_time_seconds: 1,
        },
      },
    ]);

    const { result } = renderHook(() => useSession({ withSessionList: true }));

    await waitFor(() => {
      expect(result.current.recentSessions.map((s) => s.session_id)).toEqual([
        "sess-newer",
        "sess-older",
      ]);
    });
  });

  it("switchSession hydrates session state and reconnects for active sessions", async () => {
    vi.mocked(api.getSession).mockResolvedValue({
      session_id: "sess-switch",
      mode: "decomposition",
      status: "running",
      task: "continue switched session",
      created_at: 1,
      started_at: 1,
      completed_at: null,
      error_message: null,
      llm_config: null,
    });

    const { result } = renderHook(() => useSession({ withSessionList: true }));

    await act(async () => {
      await result.current.switchSession("sess-switch");
    });

    expect(useStore.getState().sessionId).toBe("sess-switch");
    expect(useStore.getState().mode).toBe("decomposition");
    expect(useStore.getState().task).toBe("continue switched session");
    expect(useStore.getState().sessionStatus).toBe("running");
    expect(wsClient.connect).toHaveBeenCalledWith("sess-switch");
  });

  it("clearSessions clears local state and calls backend clear endpoint", async () => {
    useStore.setState({
      sessionId: "sess-old",
      sessionStatus: "complete",
      mode: "react",
      task: "old task",
    });

    const { result } = renderHook(() => useSession({ withSessionList: true }));

    await act(async () => {
      await result.current.clearSessions(true);
    });

    expect(api.clearSessions).toHaveBeenCalledWith(true);
    expect(useStore.getState().sessionId).toBeNull();
    expect(useStore.getState().sessionStatus).toBe("idle");
    expect(wsClient.disconnect).toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// restoreSessionFromStorage
// ---------------------------------------------------------------------------
describe("restoreSessionFromStorage", () => {
  it("rehydrates active session and reconnects websocket when persisted session is running", async () => {
    window.localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, "sess-running");
    vi.mocked(api.getSession).mockResolvedValue({
      session_id: "sess-running",
      mode: "decomposition",
      status: "running",
      task: "continue previous run",
      created_at: Date.now() / 1000,
      started_at: Date.now() / 1000,
      completed_at: null,
      error_message: null,
      llm_config: null,
    });

    renderHook(() => useSession());

    await waitFor(() => {
      expect(api.getSession).toHaveBeenCalledWith("sess-running");
    });

    expect(useStore.getState().sessionId).toBe("sess-running");
    expect(useStore.getState().mode).toBe("decomposition");
    expect(useStore.getState().task).toBe("continue previous run");
    expect(useStore.getState().sessionStatus).toBe("running");
    expect(wsClient.connect).toHaveBeenCalledWith("sess-running");
  });

  it("rehydrates terminal persisted session without websocket replay", async () => {
    window.localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, "sess-done");
    vi.mocked(api.getSession).mockResolvedValue({
      session_id: "sess-done",
      mode: "react",
      status: "complete",
      task: "finished run",
      created_at: Date.now() / 1000,
      started_at: Date.now() / 1000,
      completed_at: Date.now() / 1000,
      error_message: null,
      llm_config: null,
    });

    renderHook(() => useSession());

    await waitFor(() => {
      expect(api.getSession).toHaveBeenCalledWith("sess-done");
    });

    expect(useStore.getState().sessionId).toBe("sess-done");
    expect(useStore.getState().sessionStatus).toBe("complete");
    expect(wsClient.connect).not.toHaveBeenCalled();
  });

  it("clears stale persisted session id when backend returns 404", async () => {
    window.localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, "sess-missing");
    const { ApiError } = await import("@/lib/api");
    vi.mocked(api.getSession).mockRejectedValue(
      new ApiError("not found", 404, "not found")
    );

    renderHook(() => useSession());

    await waitFor(() => {
      expect(api.getSession).toHaveBeenCalledWith("sess-missing");
    });

    expect(window.localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY)).toBeNull();
  });
});
