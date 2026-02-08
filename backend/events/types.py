"""Event type definitions for the Agentica event system.

This module defines all event types that flow from the backend agent execution
to the frontend visualization. Every meaningful state change produces an event.
"""

import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """All event types in the Agentica system.

    Events are categorized by:
    - Session lifecycle: Start, completion, and error states
    - Agent lifecycle: Spawning, thinking, tool calls, and completion
    - Graph structure: Node and edge state changes for visualization
    - Workspace: File and command operations in sandboxes
    - Preview: Dev server state changes
    - Mode-specific: Decomposition and hypothesis-specific events
    - Observability: Metrics and timeout events
    """

    # Session lifecycle
    SESSION_STARTED = "session_started"
    SESSION_COMPLETE = "session_complete"
    SESSION_ERROR = "session_error"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_CLOSED = "session_closed"

    # Agent lifecycle
    AGENT_SPAWNED = "agent_spawned"
    AGENT_THINKING = "agent_thinking"
    AGENT_TOOL_CALL = "agent_tool_call"
    AGENT_TOOL_RESULT = "agent_tool_result"
    AGENT_MESSAGE = "agent_message"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"

    # Graph structure
    GRAPH_INITIALIZED = "graph_initialized"
    GRAPH_NODE_ACTIVE = "graph_node_active"
    GRAPH_NODE_COMPLETE = "graph_node_complete"
    GRAPH_EDGE_TRAVERSED = "graph_edge_traversed"

    # Workspace
    FILE_CHANGED = "file_changed"
    FILE_DELETED = "file_deleted"
    COMMAND_STARTED = "command_started"
    COMMAND_OUTPUT = "command_output"
    COMMAND_COMPLETE = "command_complete"

    # Preview
    PREVIEW_STARTING = "preview_starting"
    PREVIEW_READY = "preview_ready"
    PREVIEW_ERROR = "preview_error"

    # Decomposition-specific
    ORCHESTRATOR_PLAN = "orchestrator_plan"
    AGGREGATION_STARTED = "aggregation_started"
    AGGREGATION_COMPLETE = "aggregation_complete"

    # Hypothesis-specific
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_RESULT = "evaluation_result"

    # Observability
    AGENT_TIMEOUT = "agent_timeout"
    LLM_CALL_COMPLETE = "llm_call_complete"


class AgentEvent(BaseModel):
    """An event emitted during agent execution.

    This is the primary data structure that flows from backend to frontend.
    Each event includes:
    - type: The category of event (from EventType enum)
    - timestamp: Unix timestamp when the event occurred
    - session_id: Which session this event belongs to
    - agent_id: Which agent produced this event (if applicable)
    - agent_role: Human-readable role name (e.g., "Orchestrator", "Solver 1")
    - data: Event-specific payload

    Payload schemas by event type:

    AGENT_SPAWNED:
        - role: str - The agent's role
        - sandbox_id: str - Associated sandbox container
        - parent_id: Optional[str] - Parent agent if this is a sub-agent

    AGENT_THINKING:
        - content: str - The reasoning/planning text
        - streaming: bool - Whether this is a streaming chunk

    AGENT_TOOL_CALL:
        - tool: str - Tool name being called
        - args: dict - Arguments passed to the tool

    AGENT_TOOL_RESULT:
        - tool: str - Tool that was called
        - result: str - Result returned from the tool
        - success: bool - Whether the tool call succeeded

    AGENT_TIMEOUT:
        - reason: str - Why the agent timed out
        - iterations_completed: int - How many iterations were completed

    FILE_CHANGED:
        - path: str - Relative file path
        - content: str - File content
        - sandbox_id: str - Which sandbox the file is in

    FILE_DELETED:
        - path: str - Relative file path
        - sandbox_id: str - Which sandbox

    COMMAND_STARTED:
        - command: str - The command being executed
        - sandbox_id: str - Which sandbox

    COMMAND_OUTPUT:
        - output: str - Output chunk
        - stream: str - "stdout" or "stderr"

    COMMAND_COMPLETE:
        - command: str - The command that completed
        - exit_code: int - Exit code
        - sandbox_id: str - Which sandbox

    PREVIEW_READY:
        - url: str - Preview URL
        - sandbox_id: str - Which sandbox

    GRAPH_INITIALIZED:
        - nodes: list - Initial graph nodes
        - edges: list - Initial graph edges

    GRAPH_NODE_ACTIVE:
        - node_id: str - Which node became active

    GRAPH_NODE_COMPLETE:
        - node_id: str - Which node completed

    GRAPH_EDGE_TRAVERSED:
        - source: str - Source node
        - target: str - Target node

    ORCHESTRATOR_PLAN:
        - subtasks: list - The decomposition plan
        - scaffold_files: list - Shared scaffold file paths

    AGGREGATION_STARTED:
        - subtask_count: int - Number of subtasks being aggregated

    AGGREGATION_COMPLETE:
        - files_merged: list - Files that were merged
        - conflicts_resolved: int - Number of conflicts resolved

    EVALUATION_STARTED:
        - solver_count: int - Number of solvers being evaluated

    EVALUATION_RESULT:
        - selected: int - Index of selected solver
        - scores: list - Score breakdown per solver
        - reasoning: str - Evaluator's explanation
        - screenshots: Optional[list] - Base64 encoded screenshots

    LLM_CALL_COMPLETE:
        - model: str - Model used
        - input_tokens: int - Input token count
        - output_tokens: int - Output token count
        - latency_ms: int - Latency in milliseconds
    """

    type: EventType
    timestamp: float = Field(default_factory=time.time)
    session_id: str
    agent_id: str | None = None
    agent_role: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "agent_spawned",
                    "timestamp": 1699876543.123,
                    "session_id": "sess_abc123",
                    "agent_id": "agent_1",
                    "agent_role": "ReAct Agent",
                    "data": {
                        "role": "ReAct Agent",
                        "sandbox_id": "sandbox_abc123",
                    },
                }
            ]
        }
    }


class LLMMetrics(BaseModel):
    """Token and latency metrics for a single LLM call.

    Attributes:
        model: The model identifier (e.g., "gemini-3.0-flash-preview")
        input_tokens: Number of tokens in the prompt
        output_tokens: Number of tokens in the response
        latency_ms: Time taken for the LLM call in milliseconds
    """

    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int

    @property
    def total_tokens(self) -> int:
        """Total tokens used in this call."""
        return self.input_tokens + self.output_tokens


class SessionMetrics(BaseModel):
    """Aggregate metrics for an entire session.

    Tracks cumulative token usage and execution statistics across
    all agents and LLM calls within a session.

    Attributes:
        total_input_tokens: Sum of all input tokens across LLM calls
        total_output_tokens: Sum of all output tokens across LLM calls
        total_llm_calls: Number of LLM invocations
        total_tool_calls: Number of tool executions
        execution_time_seconds: Total wall-clock execution time
    """

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    execution_time_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens used in the session."""
        return self.total_input_tokens + self.total_output_tokens

    def add_llm_call(self, metrics: LLMMetrics) -> None:
        """Add metrics from an LLM call to the session totals.

        Args:
            metrics: The LLMMetrics from a single call
        """
        self.total_input_tokens += metrics.input_tokens
        self.total_output_tokens += metrics.output_tokens
        self.total_llm_calls += 1

    def add_tool_call(self) -> None:
        """Increment the tool call counter."""
        self.total_tool_calls += 1
