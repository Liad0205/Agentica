/**
 * TypeScript types mirroring backend event types and domain models.
 * Keep these in sync with backend/events/types.py
 */

// Event types matching the backend EventType enum
export type EventType =
  // Session lifecycle
  | "session_started"
  | "session_complete"
  | "session_error"
  | "session_cancelled"
  // Agent lifecycle
  | "agent_spawned"
  | "agent_thinking"
  | "agent_tool_call"
  | "agent_tool_result"
  | "agent_message"
  | "agent_complete"
  | "agent_error"
  | "agent_timeout"
  // Graph structure
  | "graph_initialized"
  | "graph_node_active"
  | "graph_node_complete"
  | "graph_edge_traversed"
  // Workspace
  | "file_changed"
  | "file_deleted"
  | "command_started"
  | "command_output"
  | "command_complete"
  // Preview
  | "preview_starting"
  | "preview_ready"
  | "preview_error"
  // Decomposition-specific
  | "orchestrator_plan"
  | "aggregation_started"
  | "aggregation_complete"
  // Hypothesis-specific
  | "evaluation_started"
  | "evaluation_result"
  // Observability
  | "llm_call_complete";

// Main event interface for WebSocket messages
export interface AgentEvent {
  type: EventType;
  timestamp: number;
  session_id: string;
  agent_id?: string;
  agent_role?: string;
  data: Record<string, unknown>;
}

// Agent operation modes
export type Mode = "react" | "decomposition" | "hypothesis";

// Session status
export type SessionStatus =
  | "idle"
  | "started"
  | "running"
  | "complete"
  | "error"
  | "cancelled";

// Agent status
export type AgentStatus =
  | "idle"
  | "reasoning"
  | "executing"
  | "reviewing"
  | "complete"
  | "failed"
  | "timeout";

// Graph node status for visualization
export type NodeStatus = "idle" | "active" | "reasoning" | "executing" | "complete" | "error";

// Agent information for tracking active agents
export interface AgentInfo {
  id: string;
  role: string;
  status: AgentStatus;
  sandboxId: string;
  parentId?: string;
  iterations: number;
  maxIterations: number;
  createdAt: number;
}

// Graph visualization types
export interface GraphNode {
  id: string;
  type: "agent" | "orchestrator" | "aggregator" | "evaluator" | "broadcast" | "synthesizer";
  label: string;
  status: NodeStatus;
  data?: Record<string, unknown>;
  position: { x: number; y: number };
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  animated?: boolean;
  label?: string;
}

// File system types
export interface FileNode {
  name: string;
  path: string;
  type: "file" | "directory";
  children?: FileNode[];
}

// Tool call information
export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface ToolResult {
  toolCallId: string;
  name: string;
  result: string;
  success: boolean;
}

// Message types for the session log
export type MessageType =
  | "thinking"
  | "tool_call"
  | "tool_result"
  | "system"
  | "error"
  | "plan";

export interface Message {
  id: string;
  type: MessageType;
  agentId?: string;
  agentRole?: string;
  content: string;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

// Session configuration
export interface SessionConfig {
  mode: Mode;
  task: string;
  modelConfig?: ModelConfig;
}

export interface ModelConfig {
  orchestrator?: string;
  subAgent?: string;
  evaluator?: string;
  temperature?: number;
}

// API response types
export interface CreateSessionResponse {
  session_id: string;
  websocket_url: string;
  status: string;
}

export interface SessionDetailResponse {
  session_id: string;
  mode: Mode;
  status: SessionStatus;
  task: string;
  created_at: number;
  started_at?: number | null;
  completed_at?: number | null;
  error_message?: string | null;
  llm_config?: {
    orchestrator?: string;
    sub_agent?: string;
    evaluator?: string;
    temperature?: number;
  } | null;
}

export interface SessionMetrics {
  total_input_tokens: number;
  total_output_tokens: number;
  total_llm_calls: number;
  total_tool_calls: number;
  execution_time_seconds: number;
}

export interface SessionSummary {
  session_id: string;
  mode: Mode;
  task: string;
  status: SessionStatus;
  created_at: number;
  started_at?: number | null;
  completed_at?: number | null;
  error_message?: string | null;
  metrics: SessionMetrics;
}

// Subtask types for decomposition mode
export interface SubTask {
  id: string;
  role: string;
  description: string;
  filesResponsible: string[];
  dependencies: string[];
}

export interface SubTaskResult {
  subtaskId: string;
  filesProduced: Record<string, string>;
  buildSuccess: boolean;
  lintOutput: string;
  summary: string;
}

// Hypothesis mode types
export interface HypothesisResult {
  agentId: string;
  files: Record<string, string>;
  buildSuccess: boolean;
  buildOutput: string;
  lintErrors: number;
  iterationsUsed: number;
  agentSummary: string;
  screenshot?: string; // base64 encoded PNG
}

export interface EvaluationScore {
  agentId: string;
  build: number;
  lint: number;
  quality: number;
  completeness: number;
  ux: number;
  total: number;
}

export interface EvaluationResult {
  scores: EvaluationScore[];
  selected: string;
  reasoning: string;
  improvements?: string;
}

// Command execution types
export interface CommandResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

// Sandbox information
export interface SandboxInfo {
  id: string;
  containerId: string;
  port: number;
  workspacePath: string;
  previewUrl?: string;
}
