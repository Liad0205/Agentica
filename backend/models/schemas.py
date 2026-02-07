"""Pydantic schemas for API request/response models.

This module defines all the data models used by the HTTP API and WebSocket handlers.
All models use Pydantic v2 with strict type validation.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentMode(StrEnum):
    """Available agent execution modes."""

    REACT = "react"
    DECOMPOSITION = "decomposition"
    HYPOTHESIS = "hypothesis"


class SessionStatus(StrEnum):
    """Session lifecycle status."""

    IDLE = "idle"
    STARTED = "started"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


class ModelConfig(BaseModel):
    """Model configuration for overriding defaults per session.

    Allows users to customize which models are used for different roles
    within a session.
    """

    orchestrator: str | None = Field(
        default=None,
        description="Model for orchestration tasks (decomposition mode)",
        examples=["grok-4-1-fast-reasoning", "gemini-3.0-flash-preview"],
    )
    sub_agent: str | None = Field(
        default=None,
        description="Model for sub-agent execution",
        examples=["gemini-3.0-flash-preview"],
    )
    evaluator: str | None = Field(
        default=None,
        description="Model for evaluation tasks (hypothesis mode)",
        examples=["grok-4-1-fast-reasoning"],
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation (0.0-2.0)",
    )


class CreateSessionRequest(BaseModel):
    """Request body for creating a new session."""

    model_config = ConfigDict(populate_by_name=True)

    mode: AgentMode = Field(
        description="The agent execution mode",
        examples=["react", "decomposition", "hypothesis"],
    )
    task: str = Field(
        min_length=10,
        max_length=10000,
        description="The coding task to accomplish",
        examples=[
            "Build a todo app with React and Tailwind CSS"
            " that supports adding, completing, and deleting tasks."
        ],
    )
    model_config_override: ModelConfig | None = Field(
        default=None,
        alias="llm_config",
        description="Optional model configuration overrides",
    )


class ContinueSessionRequest(BaseModel):
    """Request body for continuing an existing session."""

    model_config = ConfigDict(populate_by_name=True)

    task: str = Field(
        min_length=10,
        max_length=10000,
        description="The follow-up coding task to continue in the same session",
        examples=["Now add filtering and persistence to the todo app."],
    )
    model_config_override: ModelConfig | None = Field(
        default=None,
        alias="llm_config",
        description="Optional model configuration overrides for this follow-up run",
    )


class SessionResponse(BaseModel):
    """Response for session creation."""

    session_id: str = Field(
        description="Unique session identifier",
        examples=["sess_abc123def456"],
    )
    websocket_url: str = Field(
        description="WebSocket URL for real-time event streaming",
        examples=["/ws/sess_abc123def456"],
    )
    status: SessionStatus = Field(
        description="Current session status",
    )


class SessionDetailResponse(BaseModel):
    """Detailed session information."""

    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(description="Unique session identifier")
    mode: str = Field(description="Agent execution mode")
    task: str = Field(description="The coding task")
    status: SessionStatus = Field(description="Current session status")
    created_at: float = Field(description="Unix timestamp of session creation")
    started_at: float | None = Field(
        default=None,
        description="Unix timestamp when execution started",
    )
    completed_at: float | None = Field(
        default=None,
        description="Unix timestamp when execution completed",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if the session failed",
    )
    model_config_data: ModelConfig | None = Field(
        default=None,
        alias="llm_config",
        description="Model configuration used for this session",
    )


class SessionSummaryResponse(BaseModel):
    """Summary information for listing sessions."""

    session_id: str = Field(description="Unique session identifier")
    mode: str = Field(description="Agent execution mode")
    task: str = Field(description="The coding task")
    status: SessionStatus = Field(description="Current session status")
    created_at: float = Field(description="Unix timestamp of session creation")
    started_at: float | None = Field(
        default=None,
        description="Unix timestamp when execution started",
    )
    completed_at: float | None = Field(
        default=None,
        description="Unix timestamp when execution completed",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if the session failed",
    )
    metrics: "SessionMetrics" = Field(
        default_factory=lambda: SessionMetrics(),
        description="Aggregated metrics for this session",
    )


class FileInfo(BaseModel):
    """Information about a file or directory in the sandbox."""

    name: str = Field(
        description="File or directory name",
        examples=["App.tsx", "src"],
    )
    path: str = Field(
        description="Relative path from workspace root",
        examples=["src/App.tsx", "src/components"],
    )
    is_directory: bool = Field(
        description="True if this is a directory",
    )
    size: int | None = Field(
        default=None,
        description="File size in bytes (None for directories)",
    )
    modified_at: float | None = Field(
        default=None,
        description="Last modification timestamp",
    )


class FileContentResponse(BaseModel):
    """Response containing file content."""

    path: str = Field(
        description="Relative file path",
        examples=["src/App.tsx"],
    )
    content: str = Field(
        description="File content as UTF-8 string",
    )
    size: int = Field(
        description="Content size in bytes",
    )


class LLMMetrics(BaseModel):
    """Token and latency metrics for a single LLM call."""

    model: str = Field(
        description="Model identifier used",
        examples=["gemini-3.0-flash-preview"],
    )
    input_tokens: int = Field(
        ge=0,
        description="Number of input tokens",
    )
    output_tokens: int = Field(
        ge=0,
        description="Number of output tokens",
    )
    latency_ms: int = Field(
        ge=0,
        description="Latency in milliseconds",
    )


class SessionMetrics(BaseModel):
    """Aggregate metrics for an entire session."""

    total_input_tokens: int = Field(
        default=0,
        ge=0,
        description="Total input tokens across all LLM calls",
    )
    total_output_tokens: int = Field(
        default=0,
        ge=0,
        description="Total output tokens across all LLM calls",
    )
    total_llm_calls: int = Field(
        default=0,
        ge=0,
        description="Total number of LLM invocations",
    )
    total_tool_calls: int = Field(
        default=0,
        ge=0,
        description="Total number of tool invocations",
    )
    execution_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total execution time in seconds",
    )


class HealthResponse(BaseModel):
    """Health check response with infrastructure status."""

    status: Literal["healthy", "unhealthy"] = Field(
        description="Overall health status",
    )
    timestamp: float = Field(
        description="Current server timestamp",
    )
    version: str = Field(
        default="0.1.0",
        description="API version",
    )
    docker_available: bool = Field(
        default=False,
        description="Whether the Docker daemon is reachable",
    )
    active_sandboxes: int = Field(
        default=0,
        description="Number of currently active sandbox containers",
    )
