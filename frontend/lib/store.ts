/**
 * Zustand store for the Agent Swarm POC frontend application.
 * Manages all application state including sessions, agents, files, and UI state.
 */

import { create } from "zustand";
import type {
  AgentEvent,
  AgentInfo,
  AgentStatus,
  EvaluationScore,
  FileNode,
  Mode,
  SandboxInfo,
  SessionStatus,
  SubTask,
} from "./types";

// Connection status type
type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

// Workflow status for decomposition and hypothesis modes
type WorkflowStatus = "idle" | "active" | "complete";

/**
 * Application state interface
 */
interface AppState {
  // Session
  sessionId: string | null;
  sessionStatus: SessionStatus;
  mode: Mode;
  task: string;

  // Connection
  connectionStatus: ConnectionStatus;
  connectionError: string | null;

  // Events and agents
  events: AgentEvent[];
  agents: Map<string, AgentInfo>;

  // ReAct mode
  reactAgentInfo: AgentInfo | null;

  // Decomposition mode
  subtasks: SubTask[];
  orchestratorStatus: WorkflowStatus;
  aggregatorStatus: WorkflowStatus;
  integrationStatus: WorkflowStatus;

  // Hypothesis mode
  numSolvers: number;
  evaluatorStatus: WorkflowStatus;
  synthesizerStatus: WorkflowStatus;
  selectedSolverId: string | null;
  evaluationScores: EvaluationScore[];
  evaluationReasoning: string | null;

  // Files
  fileTree: FileNode[];
  openFilePath: string | null;
  fileContent: string | null;
  activeFilePath: string | null;

  // Sandbox
  sandboxes: SandboxInfo[];
  selectedSandboxId: string | null;
  previewUrl: string | null;
  previewLoading: boolean;
  previewError: string | null;

  // Terminal (keyed by sandbox_id, plus "all" for the combined stream)
  terminalOutputs: Record<string, string[]>;

  // Prompt
  promptValue: string;
}

/**
 * Actions interface for the store
 */
interface AppActions {
  // Main event dispatcher
  processEvent: (event: AgentEvent) => void;

  // Session actions
  setMode: (mode: Mode) => void;
  setTask: (task: string) => void;
  startSession: (
    sessionId: string,
    options?: { preserveWorkspace?: boolean }
  ) => void;
  endSession: (status: SessionStatus) => void;
  resetSession: () => void;

  // Connection actions
  setConnectionStatus: (status: ConnectionStatus) => void;
  setConnectionError: (error: string | null) => void;

  // Event actions
  addEvent: (event: AgentEvent) => void;
  clearEvents: () => void;

  // Agent actions
  updateAgentStatus: (agentId: string, status: AgentStatus) => void;

  // File actions
  setFileTree: (tree: FileNode[]) => void;
  selectFile: (path: string | null) => void;
  setFileContent: (content: string | null) => void;
  updateFile: (path: string) => void;
  deleteFile: (path: string) => void;

  // Sandbox actions
  addSandbox: (sandbox: SandboxInfo) => void;
  selectSandbox: (sandboxId: string | null) => void;
  setPreviewUrl: (url: string | null) => void;
  setPreviewLoading: (loading: boolean) => void;
  setPreviewError: (error: string | null) => void;

  // Terminal actions
  appendTerminalOutput: (output: string, sandboxId?: string) => void;
  clearTerminal: (sandboxId?: string) => void;

  // Prompt actions
  setPromptValue: (value: string) => void;
}

/**
 * Initial state values
 */
const initialState: AppState = {
  // Session
  sessionId: null,
  sessionStatus: "idle",
  mode: "react",
  task: "",

  // Connection
  connectionStatus: "disconnected",
  connectionError: null,

  // Events and agents
  events: [],
  agents: new Map(),

  // ReAct mode
  reactAgentInfo: null,

  // Decomposition mode
  subtasks: [],
  orchestratorStatus: "idle",
  aggregatorStatus: "idle",
  integrationStatus: "idle",

  // Hypothesis mode
  numSolvers: 3,
  evaluatorStatus: "idle",
  synthesizerStatus: "idle",
  selectedSolverId: null,
  evaluationScores: [],
  evaluationReasoning: null,

  // Files
  fileTree: [],
  openFilePath: null,
  fileContent: null,
  activeFilePath: null,

  // Sandbox
  sandboxes: [],
  selectedSandboxId: null,
  previewUrl: null,
  previewLoading: false,
  previewError: null,

  // Terminal
  terminalOutputs: { all: [] },

  // Prompt
  promptValue: "",
};

/**
 * Helper function to update a file in the file tree
 */
function updateFileInTree(
  nodes: FileNode[],
  path: string
): FileNode[] {
  const parts = path.split("/").filter(Boolean);
  if (parts.length === 0) {
    return nodes;
  }

  function insert(
    currentNodes: FileNode[],
    remainingParts: string[],
    prefix: string
  ): FileNode[] {
    const [segment, ...rest] = remainingParts;
    if (!segment) {
      return currentNodes;
    }

    const currentPath = prefix ? `${prefix}/${segment}` : segment;

    // Leaf: file
    if (rest.length === 0) {
      if (currentNodes.some((n) => n.path === currentPath)) {
        return currentNodes;
      }
      const newFile: FileNode = {
        name: segment,
        path: currentPath,
        type: "file",
      };
      return [
        ...currentNodes,
        newFile,
      ];
    }

    // Directory
    const existingDir = currentNodes.find(
      (n) => n.type === "directory" && n.path === currentPath
    );

    const ensuredNodes: FileNode[] = existingDir
      ? currentNodes
      : [
          ...currentNodes,
          {
            name: segment,
            path: currentPath,
            type: "directory",
            children: [],
          } satisfies FileNode,
        ];

    return ensuredNodes.map((n) => {
      if (n.type !== "directory" || n.path !== currentPath) {
        return n;
      }
      return {
        ...n,
        children: insert(n.children ?? [], rest, currentPath),
      };
    });
  }

  return insert(nodes, parts, "");
}

/**
 * Helper function to remove a file from the file tree
 */
function removeFileFromTree(nodes: FileNode[], path: string): FileNode[] {
  return nodes
    .filter((n) => n.path !== path)
    .map((n) => {
      if (n.type === "directory" && n.children) {
        return {
          ...n,
          children: removeFileFromTree(n.children, path),
        };
      }
      return n;
    });
}

function createSandboxInfo(sandboxId: string): SandboxInfo {
  return {
    id: sandboxId,
    containerId: sandboxId,
    port: 0,
    workspacePath: "/workspace",
  };
}

function upsertSandbox(
  sandboxes: SandboxInfo[],
  sandboxId: string,
  previewUrl?: string
): SandboxInfo[] {
  const index = sandboxes.findIndex((sandbox) => sandbox.id === sandboxId);
  const existing = index >= 0 ? sandboxes[index] : null;

  const next: SandboxInfo = {
    ...(existing ?? createSandboxInfo(sandboxId)),
    id: sandboxId,
    previewUrl: previewUrl ?? existing?.previewUrl,
  };

  if (index < 0) {
    return [...sandboxes, next];
  }

  return sandboxes.map((sandbox, currentIndex) =>
    currentIndex === index ? next : sandbox
  );
}

function toRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

function toStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

function normalizeSubtask(raw: unknown, index: number): SubTask {
  const source = toRecord(raw);
  const fallback = index + 1;

  const id =
    typeof source.id === "string" && source.id.length > 0
      ? source.id
      : `subtask_${fallback}`;

  const role =
    typeof source.role === "string" && source.role.length > 0
      ? source.role
      : `Subtask ${fallback}`;

  const description =
    typeof source.description === "string" ? source.description : "";

  const filesResponsible = toStringArray(
    source.filesResponsible ?? source.files_responsible
  );

  return {
    id,
    role,
    description,
    filesResponsible,
    dependencies: toStringArray(source.dependencies),
  };
}

function toNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function normalizeEvaluationScore(raw: unknown): EvaluationScore {
  const source = toRecord(raw);
  const agentId =
    typeof source.agentId === "string"
      ? source.agentId
      : typeof source.agent_id === "string"
        ? source.agent_id
        : "";

  return {
    agentId,
    build: toNumber(source.build),
    lint: toNumber(source.lint),
    quality: toNumber(source.quality),
    completeness: toNumber(source.completeness),
    ux: toNumber(source.ux),
    total: toNumber(source.total),
  };
}

function settleActiveAgents(
  agents: Map<string, AgentInfo>,
  reactAgentInfo: AgentInfo | null,
  terminalStatus: "complete" | "failed"
): {
  agents: Map<string, AgentInfo>;
  reactAgentInfo: AgentInfo | null;
} {
  let changed = false;
  const nextAgents = new Map<string, AgentInfo>();

  for (const [agentId, agent] of agents.entries()) {
    if (
      agent.status === "reasoning" ||
      agent.status === "executing" ||
      agent.status === "reviewing"
    ) {
      changed = true;
      nextAgents.set(agentId, { ...agent, status: terminalStatus });
      continue;
    }
    nextAgents.set(agentId, agent);
  }

  const nextReactAgent =
    reactAgentInfo &&
    (reactAgentInfo.status === "reasoning" ||
      reactAgentInfo.status === "executing" ||
      reactAgentInfo.status === "reviewing")
      ? { ...reactAgentInfo, status: terminalStatus }
      : reactAgentInfo;

  if (!changed && nextReactAgent === reactAgentInfo) {
    return { agents, reactAgentInfo };
  }

  if (nextReactAgent && nextAgents.has(nextReactAgent.id)) {
    nextAgents.set(nextReactAgent.id, nextReactAgent);
  }

  return {
    agents: nextAgents,
    reactAgentInfo: nextReactAgent,
  };
}

/**
 * Combined store type
 */
type AppStore = AppState & AppActions;

/**
 * Zustand store for the application
 */
export const useStore = create<AppStore>((set, get) => ({
  ...initialState,

  /**
   * Main event dispatcher - processes WebSocket events and updates state accordingly
   */
  processEvent: (event: AgentEvent): void => {
    // Always add the event to the log
    get().addEvent(event);

    switch (event.type) {
      // Session lifecycle events
      case "session_started": {
        const primarySandboxId = event.data.sandbox_id as string | undefined;
        set((state) => ({
          sessionStatus: "running",
          sandboxes:
            primarySandboxId
              ? upsertSandbox(state.sandboxes, primarySandboxId)
              : state.sandboxes,
          selectedSandboxId:
            state.selectedSandboxId ?? primarySandboxId ?? null,
        }));
        break;
      }

      case "session_complete":
        set((state) => {
          const settled = settleActiveAgents(
            state.agents,
            state.reactAgentInfo,
            "complete"
          );
          return {
            sessionStatus: "complete",
            previewLoading: false,
            integrationStatus:
              state.integrationStatus === "active"
                ? "complete"
                : state.integrationStatus,
            evaluatorStatus:
              state.evaluatorStatus === "active"
                ? "complete"
                : state.evaluatorStatus,
            agents: settled.agents,
            reactAgentInfo: settled.reactAgentInfo,
          };
        });
        break;

      case "session_error":
        set((state) => {
          const settled = settleActiveAgents(
            state.agents,
            state.reactAgentInfo,
            "failed"
          );
          return {
            sessionStatus: "error",
            previewLoading: false,
            agents: settled.agents,
            reactAgentInfo: settled.reactAgentInfo,
          };
        });
        break;

      case "session_cancelled":
        set((state) => {
          const settled = settleActiveAgents(
            state.agents,
            state.reactAgentInfo,
            "failed"
          );
          return {
            sessionStatus: "cancelled",
            previewLoading: false,
            agents: settled.agents,
            reactAgentInfo: settled.reactAgentInfo,
          };
        });
        break;

      // Agent lifecycle events
      case "agent_spawned": {
        const sandboxId = (event.data.sandbox_id as string) ?? "";
        const agentInfo: AgentInfo = {
          id: event.agent_id ?? crypto.randomUUID(),
          role: event.agent_role ?? "unknown",
          status: "idle",
          sandboxId,
          parentId: event.data.parent_id as string | undefined,
          iterations: 0,
          maxIterations: (event.data.max_iterations as number) ?? 10,
          createdAt: event.timestamp,
        };

        set((state) => {
          const newAgents = new Map(state.agents);
          newAgents.set(agentInfo.id, agentInfo);

          const isReactAgent = state.mode === "react";

          const sandboxes = sandboxId
            ? upsertSandbox(state.sandboxes, sandboxId)
            : state.sandboxes;

          return {
            agents: newAgents,
            reactAgentInfo: isReactAgent ? agentInfo : state.reactAgentInfo,
            sandboxes,
            selectedSandboxId:
              state.selectedSandboxId ?? (sandboxId || null),
          };
        });
        break;
      }

      case "agent_thinking":
        get().updateAgentStatus(event.agent_id ?? "", "reasoning");
        break;

      case "agent_tool_call":
        get().updateAgentStatus(event.agent_id ?? "", "executing");
        break;

      case "agent_tool_result":
        // After tool result, agent typically goes back to reasoning
        get().updateAgentStatus(event.agent_id ?? "", "reasoning");
        break;

      case "agent_complete":
        get().updateAgentStatus(event.agent_id ?? "", "complete");
        break;

      case "agent_error":
        get().updateAgentStatus(event.agent_id ?? "", "failed");
        break;

      case "agent_timeout":
        get().updateAgentStatus(event.agent_id ?? "", "timeout");
        break;

      // File events
      case "file_changed": {
        const filePath = event.data.path as string;
        set((state) => ({
          fileTree: updateFileInTree(state.fileTree, filePath),
          activeFilePath: filePath,
        }));
        break;
      }

      case "file_deleted": {
        const deletedPath = event.data.path as string;

        set((state) => ({
          fileTree: removeFileFromTree(state.fileTree, deletedPath),
          activeFilePath:
            state.activeFilePath === deletedPath
              ? null
              : state.activeFilePath,
          openFilePath:
            state.openFilePath === deletedPath
              ? null
              : state.openFilePath,
        }));
        break;
      }

      // Command events
      case "command_output": {
        const output = event.data.output as string;
        const sandboxId = event.data.sandbox_id as string | undefined;
        if (output) {
          get().appendTerminalOutput(output, sandboxId);
        }
        break;
      }

      case "command_started": {
        const command = event.data.command as string;
        const sandboxId = event.data.sandbox_id as string | undefined;
        if (command) {
          const role = event.agent_role;
          const prefix = role ? `\x1b[36m[${role}]\x1b[0m ` : "";
          get().appendTerminalOutput(`${prefix}$ ${command}\n`, sandboxId);
        }
        break;
      }

      case "command_complete": {
        const exitCode = event.data.exit_code as number;
        const sandboxId = event.data.sandbox_id as string | undefined;
        if (exitCode !== undefined) {
          get().appendTerminalOutput(`\n[Exit code: ${exitCode}]\n`, sandboxId);
        }
        break;
      }

      // Preview events
      case "preview_starting": {
        const sandboxId = event.data.sandbox_id as string | undefined;
        set((state) => ({
          previewLoading: true,
          previewError: null,
          sandboxes:
            sandboxId
              ? upsertSandbox(state.sandboxes, sandboxId)
              : state.sandboxes,
          selectedSandboxId:
            state.selectedSandboxId ?? sandboxId ?? null,
        }));
        break;
      }

      case "preview_ready": {
        const rawUrl = event.data.url as string;
        const sandboxId = event.data.sandbox_id as string | undefined;

        // Validate URL to prevent XSS via javascript: or data: schemes
        let validUrl: string | null = null;
        try {
          const parsed = new URL(rawUrl);
          if (parsed.protocol === "http:" || parsed.protocol === "https:") {
            validUrl = parsed.href;
          }
        } catch {
          // invalid URL
        }

        if (!validUrl) {
          set({ previewError: "Invalid preview URL received", previewLoading: false });
          break;
        }

        set((state) => {
          const updatedSandboxes = sandboxId
            ? upsertSandbox(state.sandboxes, sandboxId, validUrl)
            : state.sandboxes;
          const effectiveSelected = state.selectedSandboxId ?? sandboxId ?? null;

          // Only update global previewUrl if the event's sandbox is the currently selected one
          const isSelectedSandbox = !sandboxId || sandboxId === effectiveSelected;

          return {
            previewUrl: isSelectedSandbox ? validUrl : state.previewUrl,
            previewLoading: isSelectedSandbox ? false : state.previewLoading,
            previewError: isSelectedSandbox ? null : state.previewError,
            sandboxes: updatedSandboxes,
            selectedSandboxId: effectiveSelected,
          };
        });
        break;
      }

      case "preview_error": {
        const errorMessage = event.data.error as string;
        set({
          previewLoading: false,
          previewError: errorMessage ?? "Preview failed to start",
        });
        break;
      }

      // Decomposition mode events
      case "orchestrator_plan": {
        const rawSubtasks = Array.isArray(event.data.subtasks)
          ? event.data.subtasks
          : [];
        const subtasks = rawSubtasks.map((subtask, index) =>
          normalizeSubtask(subtask, index)
        );
        set({
          subtasks,
          orchestratorStatus: "complete",
        });
        break;
      }

      case "aggregation_started":
        set({ aggregatorStatus: "active" });
        break;

      case "aggregation_complete":
        set({
          aggregatorStatus: "complete",
          integrationStatus: "active",
        });
        break;

      // Hypothesis mode events
      case "evaluation_started":
        set({ evaluatorStatus: "active" });
        break;

      case "evaluation_result": {
        const scores = Array.isArray(event.data.scores)
          ? event.data.scores.map((score) => normalizeEvaluationScore(score))
          : [];
        const selected =
          typeof event.data.selected === "string" ? event.data.selected : null;
        const reasoning =
          typeof event.data.reasoning === "string"
            ? event.data.reasoning
            : null;

        set({
          evaluationScores: scores,
          selectedSolverId: selected,
          evaluationReasoning: reasoning,
          evaluatorStatus: "complete",
        });
        break;
      }

      // Graph events - can be used for visualization updates
      case "graph_initialized":
        break;

      case "graph_node_active":
      case "graph_node_complete": {
        const nodeId =
          typeof event.data.node_id === "string" ? event.data.node_id : "";
        const isActiveEvent = event.type === "graph_node_active";
        const workflowStatus: WorkflowStatus = isActiveEvent
          ? "active"
          : "complete";

        if (isActiveEvent && nodeId === "reason") {
          get().updateAgentStatus(event.agent_id ?? "", "reasoning");
        } else if (isActiveEvent && nodeId === "execute") {
          get().updateAgentStatus(event.agent_id ?? "", "executing");
        } else if (isActiveEvent && nodeId === "review") {
          get().updateAgentStatus(event.agent_id ?? "", "reviewing");
        }

        const workflowPatch: Partial<AppState> = {};
        if (nodeId === "orchestrate") {
          workflowPatch.orchestratorStatus = workflowStatus;
        } else if (nodeId === "aggregate") {
          workflowPatch.aggregatorStatus = workflowStatus;
        } else if (nodeId === "integration_review") {
          workflowPatch.integrationStatus = workflowStatus;
        } else if (nodeId === "evaluate") {
          workflowPatch.evaluatorStatus = workflowStatus;
        } else if (nodeId === "synthesize") {
          workflowPatch.synthesizerStatus = workflowStatus;
        }

        if (Object.keys(workflowPatch).length > 0) {
          set(workflowPatch);
        }
        break;
      }

      case "graph_edge_traversed":
        // These events can be handled by graph visualization components
        // that subscribe to the events array
        break;

      // Observability events
      case "llm_call_complete":
        // Can be used for metrics tracking
        break;

      // Message events (already stored in events array)
      case "agent_message":
        break;

      default:
        // Unknown event type - already logged to events
        break;
    }
  },

  // Session actions
  setMode: (mode: Mode): void => {
    set({ mode });
  },

  setTask: (task: string): void => {
    set({ task });
  },

  startSession: (
    sessionId: string,
    options?: { preserveWorkspace?: boolean }
  ): void => {
    set((state) => {
      if (options?.preserveWorkspace && state.sessionId === sessionId) {
        return {
          ...state,
          // Reset execution/graph state for the next run while preserving workspace context.
          agents: new Map(),
          reactAgentInfo: null,
          subtasks: [],
          orchestratorStatus: "idle",
          aggregatorStatus: "idle",
          integrationStatus: "idle",
          evaluatorStatus: "idle",
          synthesizerStatus: "idle",
          selectedSolverId: null,
          evaluationScores: [],
          evaluationReasoning: null,
          sessionId,
          sessionStatus: "started",
          connectionStatus: "connecting",
          connectionError: null,
          previewLoading: false,
          previewError: null,
        };
      }

      return {
        ...initialState,
        mode: state.mode,
        numSolvers: state.numSolvers,
        promptValue: state.promptValue,
        task: state.task,
        sessionId,
        sessionStatus: "started",
        connectionStatus: "connecting",
        connectionError: null,
      };
    });
  },

  endSession: (status: SessionStatus): void => {
    set({ sessionStatus: status });
  },

  resetSession: (): void => {
    set({
      ...initialState,
      // Preserve user preferences
      mode: get().mode,
      numSolvers: get().numSolvers,
    });
  },

  // Connection actions
  setConnectionStatus: (status: ConnectionStatus): void => {
    set({
      connectionStatus: status,
      connectionError: status === "connected" ? null : get().connectionError,
    });
  },

  setConnectionError: (error: string | null): void => {
    set({
      connectionError: error,
      connectionStatus: error ? "error" : get().connectionStatus,
    });
  },

  // Event actions
  addEvent: (event: AgentEvent): void => {
    set((state) => {
      const MAX_EVENTS = 2000;
      // Skip duplicate events (e.g. from event replay after reconnect).
      // Events are uniquely identified by their type + timestamp + agent_id
      // combination which is extremely unlikely to collide for distinct events.
      const lastEvents = state.events;
      if (
        lastEvents.length > 0 &&
        lastEvents[lastEvents.length - 1]?.timestamp === event.timestamp &&
        lastEvents[lastEvents.length - 1]?.type === event.type &&
        lastEvents[lastEvents.length - 1]?.agent_id === event.agent_id
      ) {
        return state;
      }
      const events = [...lastEvents, event];
      return {
        events: events.length > MAX_EVENTS ? events.slice(-MAX_EVENTS) : events,
      };
    });
  },

  clearEvents: (): void => {
    set({ events: [] });
  },

  // Agent actions
  updateAgentStatus: (agentId: string, status: AgentStatus): void => {
    set((state) => {
      const agent = state.agents.get(agentId);
      if (!agent) {
        return state;
      }

      const updatedAgent: AgentInfo = {
        ...agent,
        status,
        iterations:
          status === "reasoning" ? agent.iterations + 1 : agent.iterations,
      };

      const newAgents = new Map(state.agents);
      newAgents.set(agentId, updatedAgent);

      // Update reactAgentInfo if this is the react agent
      const newReactAgentInfo =
        state.reactAgentInfo?.id === agentId
          ? updatedAgent
          : state.reactAgentInfo;

      return {
        agents: newAgents,
        reactAgentInfo: newReactAgentInfo,
      };
    });
  },

  // File actions
  setFileTree: (tree: FileNode[]): void => {
    set({ fileTree: tree });
  },

  selectFile: (path: string | null): void => {
    set({
      openFilePath: path,
      fileContent: null, // Content will be loaded separately
    });
  },

  setFileContent: (content: string | null): void => {
    set({ fileContent: content });
  },

  updateFile: (path: string): void => {
    set((state) => ({
      fileTree: updateFileInTree(state.fileTree, path),
      activeFilePath: path,
    }));
  },

  deleteFile: (path: string): void => {
    set((state) => ({
      fileTree: removeFileFromTree(state.fileTree, path),
      activeFilePath:
        state.activeFilePath === path ? null : state.activeFilePath,
      openFilePath:
        state.openFilePath === path ? null : state.openFilePath,
    }));
  },

  // Sandbox actions
  addSandbox: (sandbox: SandboxInfo): void => {
    set((state) => {
      const existingIndex = state.sandboxes.findIndex((s) => s.id === sandbox.id);
      if (existingIndex < 0) {
        return {
          sandboxes: [...state.sandboxes, sandbox],
          selectedSandboxId: state.selectedSandboxId ?? sandbox.id,
        };
      }

      return {
        sandboxes: state.sandboxes.map((current) =>
          current.id === sandbox.id ? { ...current, ...sandbox } : current
        ),
      };
    });
  },

  selectSandbox: (sandboxId: string | null): void => {
    set((state) => {
      const sandbox = sandboxId
        ? state.sandboxes.find((s) => s.id === sandboxId)
        : undefined;
      return {
        selectedSandboxId: sandboxId,
        previewUrl: sandbox?.previewUrl ?? state.previewUrl,
      };
    });
  },

  setPreviewUrl: (url: string | null): void => {
    set({
      previewUrl: url,
      previewLoading: false,
      previewError: null,
    });
  },

  setPreviewLoading: (loading: boolean): void => {
    set({ previewLoading: loading });
  },

  setPreviewError: (error: string | null): void => {
    set({
      previewError: error,
      previewLoading: false,
    });
  },

  // Terminal actions
  appendTerminalOutput: (output: string, sandboxId?: string): void => {
    set((state) => {
      const MAX_LINES = 5000;
      const next = { ...state.terminalOutputs };

      // Always append to the "all" stream
      const allStream = [...(next.all ?? []), output];
      next.all = allStream.length > MAX_LINES ? allStream.slice(-MAX_LINES) : allStream;

      // Also append to sandbox-specific stream when sandboxId is provided
      if (sandboxId) {
        const sbStream = [...(next[sandboxId] ?? []), output];
        next[sandboxId] = sbStream.length > MAX_LINES ? sbStream.slice(-MAX_LINES) : sbStream;
      }

      return { terminalOutputs: next };
    });
  },

  clearTerminal: (sandboxId?: string): void => {
    set((state) => {
      if (sandboxId) {
        return {
          terminalOutputs: { ...state.terminalOutputs, [sandboxId]: [] },
        };
      }
      // Clear all streams
      const next: Record<string, string[]> = {};
      for (const key of Object.keys(state.terminalOutputs)) {
        next[key] = [];
      }
      return { terminalOutputs: next };
    });
  },

  // Prompt actions
  setPromptValue: (value: string): void => {
    set({ promptValue: value });
  },
}));

/**
 * Type export for the store
 */
export type { AppState, AppActions, AppStore };
