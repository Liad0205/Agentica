/**
 * WebSocket client for real-time event streaming from the backend.
 * Handles connection management, auto-reconnect with exponential backoff,
 * keep-alive pings, and event distribution to registered handlers.
 */

import type { AgentEvent } from "./types";
import {
  BACKEND_URL,
  CONFIG_DEFAULTS,
  API_ENDPOINTS,
} from "./constants";

// Connection status states
export type ConnectionStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  | "error";

// Commands that can be sent to the server
export type WebSocketCommand =
  | { type: "cancel" }
  | { type: "ping"; timestamp: number };

type IncomingMessage =
  | AgentEvent
  | { type: "pong"; timestamp?: number };

function isAgentEvent(
  message: unknown,
): message is AgentEvent {
  if (!message || typeof message !== "object") {
    return false;
  }

  const candidate = message as Partial<AgentEvent>;
  return (
    typeof candidate.type === "string" &&
    typeof candidate.timestamp === "number" &&
    typeof candidate.session_id === "string" &&
    candidate.data !== undefined
  );
}

// WebSocket client interface
export interface WebSocketClient {
  connect(sessionId: string): void;
  disconnect(): void;
  send(message: WebSocketCommand): void;
  onEvent(handler: (event: AgentEvent) => void): () => void;
  onStatusChange(
    handler: (status: ConnectionStatus) => void,
  ): () => void;
  getStatus(): ConnectionStatus;
}

// Keep-alive ping interval in milliseconds
const PING_INTERVAL_MS = 30000;

/**
 * Converts an HTTP(S) URL to a WebSocket URL (WS/WSS).
 */
function httpToWsUrl(httpUrl: string): string {
  return httpUrl.replace(/^http/, "ws");
}

/**
 * Build a websocket URL from an environment-configured base URL.
 *
 * Supported NEXT_PUBLIC_WS_URL formats:
 * - ws://host:port
 * - ws://host:port/ws
 * - ws://host:port/ws/{session_id}
 */
function buildSocketUrl(sessionId: string): string {
  const configuredBase =
    process.env.NEXT_PUBLIC_WS_URL?.trim();
  const defaultBase = httpToWsUrl(BACKEND_URL);
  const rawBase = configuredBase || defaultBase;

  try {
    const parsed = new URL(rawBase);
    const wsPath = API_ENDPOINTS.websocket(sessionId);

    if (
      configuredBase &&
      parsed.pathname.includes("{session_id}")
    ) {
      parsed.pathname = parsed.pathname.replace(
        "{session_id}",
        sessionId,
      );
      return parsed.toString();
    }

    if (
      configuredBase &&
      /\/ws\/?$/.test(parsed.pathname)
    ) {
      parsed.pathname = `${parsed.pathname.replace(/\/$/, "")}/${sessionId}`;
      return parsed.toString();
    }

    parsed.pathname = wsPath;
    return parsed.toString();
  } catch {
    const wsPath = API_ENDPOINTS.websocket(sessionId);
    return `${rawBase.replace(/\/$/, "")}${wsPath}`;
  }
}

/**
 * WebSocket client implementation with auto-reconnect and keep-alive support.
 */
class WebSocketClientImpl implements WebSocketClient {
  private socket: WebSocket | null = null;
  private status: ConnectionStatus = "disconnected";
  private currentSessionId: string | null = null;
  private reconnectAttempts = 0;
  private reconnectTimeoutId: ReturnType<
    typeof setTimeout
  > | null = null;
  private pingIntervalId: ReturnType<
    typeof setInterval
  > | null = null;
  private isCleanClose = false;
  private currentSocketUrl: string | null = null;

  private eventHandlers: Set<(event: AgentEvent) => void> =
    new Set();
  private statusHandlers: Set<
    (status: ConnectionStatus) => void
  > = new Set();

  /**
   * Connect to the WebSocket endpoint for a given session.
   */
  connect(sessionId: string): void {
    // Clean up any existing connection
    this.cleanupConnection();

    this.currentSessionId = sessionId;
    this.isCleanClose = false;
    this.reconnectAttempts = 0;

    this.establishConnection();
  }

  /**
   * Disconnect from the WebSocket and stop reconnection attempts.
   */
  disconnect(): void {
    this.isCleanClose = true;
    this.cleanupConnection();
    this.currentSessionId = null;
    this.setStatus("disconnected");
  }

  /**
   * Send a command to the server.
   */
  send(message: WebSocketCommand): void {
    if (
      this.socket &&
      this.socket.readyState === WebSocket.OPEN
    ) {
      this.socket.send(JSON.stringify(message));
    }
  }

  /**
   * Register an event handler. Returns an unsubscribe function.
   */
  onEvent(
    handler: (event: AgentEvent) => void,
  ): () => void {
    this.eventHandlers.add(handler);
    return () => {
      this.eventHandlers.delete(handler);
    };
  }

  /**
   * Register a status change handler. Returns an unsubscribe function.
   */
  onStatusChange(
    handler: (status: ConnectionStatus) => void,
  ): () => void {
    this.statusHandlers.add(handler);
    // Immediately notify the handler of the current status
    handler(this.status);
    return () => {
      this.statusHandlers.delete(handler);
    };
  }

  /**
   * Get the current connection status.
   */
  getStatus(): ConnectionStatus {
    return this.status;
  }

  /**
   * Establish the WebSocket connection.
   */
  private establishConnection(): void {
    if (!this.currentSessionId) {
      return;
    }

    this.setStatus("connecting");
    const wsUrl = buildSocketUrl(this.currentSessionId);
    this.currentSocketUrl = wsUrl;

    try {
      this.socket = new WebSocket(wsUrl);
      this.setupSocketHandlers();
    } catch (error) {
      console.error(
        "[WS] WebSocket connection error:",
        error,
      );
      this.setStatus("error");
      this.scheduleReconnect();
    }
  }

  /**
   * Set up WebSocket event handlers.
   */
  private setupSocketHandlers(): void {
    if (!this.socket) {
      return;
    }

    // Track if connection was ever successfully opened.
    // Used to detect early rejection (e.g., 4404 session not found before accept).
    let wasOpened = false;

    this.socket.onopen = (): void => {
      wasOpened = true;
      this.setStatus("connected");
      this.reconnectAttempts = 0;
      this.startPingInterval();
    };

    this.socket.onclose = (event: CloseEvent): void => {
      this.stopPingInterval();

      if (this.isCleanClose) {
        this.setStatus("disconnected");
        return;
      }

      // If connection was rejected before onopen fired, treat as non-recoverable.
      // This happens when backend closes the WebSocket before accepting (e.g., session not found).
      // Reconnecting would hit the same error, so avoid the retry loop.
      if (!wasOpened) {
        console.warn(
          "[WS] Connection rejected before open",
          {
            code: event.code,
            reason: event.reason,
            url: this.currentSocketUrl,
            sessionId: this.currentSessionId,
          },
        );
        this.setStatus("error");
        return;
      }

      const nonRecoverable =
        event.code === 4400 || // invalid session id format
        event.code === 4404 || // session not found
        event.code === 1008; // policy violation

      console.warn("[WS] WebSocket closed", {
        code: event.code,
        reason: event.reason,
        wasClean: event.wasClean,
        url: this.currentSocketUrl,
        sessionId: this.currentSessionId,
      });

      if (nonRecoverable) {
        // Permanent failure for this session; avoid reconnect loops.
        this.setStatus("error");
        return;
      }

      // Unexpected but recoverable close - attempt reconnection.
      this.setStatus("disconnected");
      this.scheduleReconnect();
    };

    this.socket.onerror = (event: Event): void => {
      console.error("[WS] WebSocket transport error", {
        eventType: event.type,
        url: this.currentSocketUrl,
        sessionId: this.currentSessionId,
        readyState: this.socket?.readyState,
      });
      // Don't set error status here; let onclose handle it.
      // The onerror event is always followed by onclose, and onclose
      // has better context (wasOpened, close code) for error handling.
    };

    this.socket.onmessage = (
      event: MessageEvent<string>,
    ): void => {
      this.handleMessage(event.data);
    };
  }

  /**
   * Handle incoming WebSocket message.
   */
  private handleMessage(data: string): void {
    try {
      const message = JSON.parse(data) as IncomingMessage;
      if ((message as { type: string }).type === "pong") {
        return;
      }
      if (isAgentEvent(message)) {
        this.notifyEventHandlers(message);
        return;
      }

      console.warn(
        "Ignoring unrecognized WebSocket payload",
        message,
      );
    } catch (error) {
      console.error(
        "Failed to parse WebSocket message:",
        error,
        data,
      );
    }
  }

  /**
   * Notify all registered event handlers.
   */
  private notifyEventHandlers(event: AgentEvent): void {
    this.eventHandlers.forEach((handler) => {
      try {
        handler(event);
      } catch (error) {
        console.error("Error in event handler:", error);
      }
    });
  }

  /**
   * Update status and notify handlers.
   */
  private setStatus(newStatus: ConnectionStatus): void {
    if (this.status !== newStatus) {
      this.status = newStatus;
      this.statusHandlers.forEach((handler) => {
        try {
          handler(newStatus);
        } catch (error) {
          console.error("Error in status handler:", error);
        }
      });
    }
  }

  /**
   * Schedule a reconnection attempt with exponential backoff.
   */
  private scheduleReconnect(): void {
    if (this.isCleanClose || !this.currentSessionId) {
      return;
    }

    if (
      this.reconnectAttempts >=
      CONFIG_DEFAULTS.wsMaxReconnectAttempts
    ) {
      console.error("Max reconnect attempts reached");
      this.setStatus("error");
      return;
    }

    // Exponential backoff: baseDelay * 2^attempts
    const delay =
      CONFIG_DEFAULTS.wsReconnectDelayMs *
      Math.pow(2, this.reconnectAttempts);
    this.reconnectAttempts++;

    this.reconnectTimeoutId = setTimeout(() => {
      this.establishConnection();
    }, delay);
  }

  /**
   * Start the keep-alive ping interval.
   */
  private startPingInterval(): void {
    this.stopPingInterval();
    this.pingIntervalId = setInterval(() => {
      this.send({ type: "ping", timestamp: Date.now() });
    }, PING_INTERVAL_MS);
  }

  /**
   * Stop the keep-alive ping interval.
   */
  private stopPingInterval(): void {
    if (this.pingIntervalId !== null) {
      clearInterval(this.pingIntervalId);
      this.pingIntervalId = null;
    }
  }

  /**
   * Clean up the current connection and any pending timers.
   */
  private cleanupConnection(): void {
    this.stopPingInterval();

    if (this.reconnectTimeoutId !== null) {
      clearTimeout(this.reconnectTimeoutId);
      this.reconnectTimeoutId = null;
    }

    if (this.socket) {
      // Remove handlers to prevent any callbacks during close
      this.socket.onopen = null;
      this.socket.onclose = null;
      this.socket.onerror = null;
      this.socket.onmessage = null;

      if (
        this.socket.readyState === WebSocket.OPEN ||
        this.socket.readyState === WebSocket.CONNECTING
      ) {
        this.socket.close();
      }
      this.socket = null;
    }
    this.currentSocketUrl = null;
  }
}

// Export singleton instance
export const wsClient = new WebSocketClientImpl();
