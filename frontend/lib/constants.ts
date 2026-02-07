/**
 * Application constants including colors, status mappings, and configuration defaults.
 */

import type {
  AgentStatus,
  Mode,
  NodeStatus,
  SessionStatus,
} from "./types";

// Theme colors (matching CSS variables)
export const COLORS = {
  background: "hsl(var(--background))",
  card: "hsl(var(--card))",
  border: "hsl(var(--border))",
  accent: "hsl(var(--accent))",
  warning: "var(--color-warning)",
  success: "var(--color-success)",
  error: "var(--color-error)",
  active: "hsl(var(--primary))", // V0 style active is usually primary (black/white)
  muted: "hsl(var(--muted-foreground))",
  foreground: "hsl(var(--foreground))",
  foregroundMuted: "hsl(var(--muted-foreground))",
} as const;

// Agent color palette for visual differentiation
export const AGENT_COLORS = [
  "#00d4ff", // cyan
  "#a855f7", // purple
  "#ec4899", // pink
  "#f97316", // orange
  "#14b8a6", // teal
  "#84cc16", // lime
  "#06b6d4", // sky
  "#f43f5e", // rose
] as const;

// Get a deterministic color for an agent based on its ID
export function getAgentColor(agentId: string): string {
  let hash = 0;
  for (let i = 0; i < agentId.length; i++) {
    const char = agentId.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  const index = Math.abs(hash) % AGENT_COLORS.length;
  const color = AGENT_COLORS[index];
  return color ?? AGENT_COLORS[0] ?? "#00d4ff";
}

// Status to color mappings
export const STATUS_COLORS: Record<
  SessionStatus | AgentStatus | NodeStatus,
  string
> = {
  idle: COLORS.muted,
  started: COLORS.accent,
  running: COLORS.active,
  complete: COLORS.success,
  cancelled: COLORS.warning,
  error: COLORS.error,
  reasoning: COLORS.accent,
  executing: COLORS.active,
  reviewing: COLORS.warning,
  failed: COLORS.error,
  timeout: COLORS.warning,
  active: COLORS.active,
} as const;

// Mode display names and descriptions
export const MODE_INFO: Record<
  Mode,
  { name: string; description: string }
> = {
  react: {
    name: "ReAct",
    description: "Single agent with reason-act-review loop",
  },
  decomposition: {
    name: "Decomposition",
    description: "Task split into parallel sub-agents",
  },
  hypothesis: {
    name: "Hypothesis",
    description:
      "Multiple agents compete, best solution selected",
  },
} as const;

// Configuration defaults
export const CONFIG_DEFAULTS = {
  // Agent limits
  maxReactIterations: 15,
  maxSubtaskIterations: 8,
  numHypothesisAgents: 3,
  toolTimeoutSeconds: 60,
  agentTimeoutSeconds: 300,

  // Model defaults (with LiteLLM provider prefix)
  defaultModel: "gemini/gemini-3-flash-preview",
  orchestratorModel: "xai/grok-4-1-fast-reasoning",
  evaluatorModel: "xai/grok-4-1-fast-reasoning",
  defaultTemperature: 0.8,

  // Sandbox settings
  sandboxBasePort: 5173,
  maxConcurrentSandboxes: 10,

  // WebSocket reconnection
  wsReconnectDelayMs: 1000,
  wsMaxReconnectAttempts: 5,
} as const;

// API endpoints
export const API_ENDPOINTS = {
  sessions: "/api/sessions",
  health: "/health",
  websocket: (sessionId: string): string =>
    `/ws/${sessionId}`,
  sessionFiles: (sessionId: string): string =>
    `/api/sessions/${sessionId}/files`,
  sessionFilesArchive: (sessionId: string): string =>
    `/api/sessions/${sessionId}/files/archive`,
  sessionMetrics: (sessionId: string): string =>
    `/api/sessions/${sessionId}/metrics`,
  sessionCancel: (sessionId: string): string =>
    `/api/sessions/${sessionId}/cancel`,
  sessionContinue: (sessionId: string): string =>
    `/api/sessions/${sessionId}/continue`,
  sessionSandboxDevServer: (sessionId: string): string =>
    `/api/sessions/${sessionId}/sandbox/dev-server`,
} as const;

// Backend URL (configurable via environment)
export const BACKEND_URL =
  process.env.NEXT_PUBLIC_API_URL ??
  process.env.NEXT_PUBLIC_BACKEND_URL ??
  "http://localhost:8000";

// Tool names for display
export const TOOL_DISPLAY_NAMES: Record<string, string> = {
  write_file: "Write File",
  read_file: "Read File",
  list_files: "List Files",
  execute_command: "Execute Command",
  search_files: "Search Files",
} as const;

// File type to icon mappings (for file tree)
export const FILE_ICONS: Record<string, string> = {
  // Directories
  directory: "folder",
  // TypeScript/JavaScript
  ts: "typescript",
  tsx: "react",
  js: "javascript",
  jsx: "react",
  // Configuration
  json: "json",
  yaml: "yaml",
  yml: "yaml",
  toml: "toml",
  // Styles
  css: "css",
  scss: "sass",
  less: "less",
  // Web
  html: "html",
  svg: "svg",
  // Other
  md: "markdown",
  txt: "text",
  default: "file",
} as const;

// Get file icon based on extension
export function getFileIcon(
  filename: string,
  isDirectory: boolean,
): string {
  if (isDirectory) {
    return FILE_ICONS.directory ?? "folder";
  }
  const ext =
    filename.split(".").pop()?.toLowerCase() ?? "";
  return FILE_ICONS[ext] ?? FILE_ICONS.default ?? "file";
}

// Monaco editor language mappings
export const FILE_LANGUAGES: Record<string, string> = {
  ts: "typescript",
  tsx: "typescript",
  js: "javascript",
  jsx: "javascript",
  json: "json",
  css: "css",
  scss: "scss",
  html: "html",
  md: "markdown",
  yaml: "yaml",
  yml: "yaml",
} as const;

// Get Monaco language for a file
export function getFileLanguage(filename: string): string {
  const ext =
    filename.split(".").pop()?.toLowerCase() ?? "";
  return FILE_LANGUAGES[ext] ?? "plaintext";
}

// Detect macOS for shortcut display
const isMac =
  typeof navigator !== "undefined"
    ? /mac/i.test(navigator.userAgent)
    : false;

// Keyboard shortcuts
export const SHORTCUTS = {
  submitPrompt: "Ctrl+Enter",
  cancelSession: "Escape",
  togglePreview: "Ctrl+P",
  toggleTerminal: "Ctrl+`",
  toggleSidebar: isMac ? "Cmd+B" : "Ctrl+B",
  toggleGraphPanel: isMac ? "Cmd+G" : "Ctrl+G",
} as const;
