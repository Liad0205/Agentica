"""Application configuration using Pydantic Settings.

This module provides centralized configuration management for the Agentica backend.
All settings can be overridden via environment variables or a .env file.
"""

import json
import logging
import os
from typing import Any

import structlog
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        xai_api_key: API key for xAI (Grok) models.
        gemini_api_key: API key for Google Gemini models.
        lm_studio_api_base: Base URL for LM Studio local server (e.g. http://localhost:1234/v1).
        default_model: Default model for sub-agents.
        orchestrator_model: Model used for orchestration and planning tasks.
        evaluator_model: Model used for evaluation in hypothesis mode.
        use_mock_llm: If True, use mock LLM responses for testing.
        max_react_iterations: Maximum iterations for single ReAct agent.
        max_subtask_iterations: Maximum iterations per subtask in decomposition mode.
        num_hypothesis_agents: Number of parallel solvers in hypothesis mode.
        tool_timeout_seconds: Timeout for individual tool executions.
        agent_timeout_seconds: Timeout for entire agent execution.
        llm_retry_attempts: Number of retries for malformed LLM output.
        sandbox_image: Docker image for sandbox containers.
        sandbox_base_port: Base port for sandbox dev servers.
        max_concurrent_sandboxes: Maximum number of concurrent sandbox containers.
        sandbox_cleanup_interval_minutes: Interval for background cleanup.
        sandbox_ttl_minutes: Maximum sandbox lifetime.
        sandbox_allow_unrestricted_commands: If True, agents can execute any
            shell command inside the sandbox (recommended for autonomous coding).
        session_ttl_minutes: Session expiry for cleanup.
        backend_port: Port for the FastAPI server.
        frontend_port: Port for the frontend (for CORS).
        cors_origins: Allowed origins for CORS.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_format: Log format (json or text).
    """

    # LLM Configuration
    xai_api_key: str = ""
    gemini_api_key: str = ""
    # LM Studio local server URL (OpenAI-compatible API)
    lm_studio_api_base: str = "http://localhost:1234/v1"
    # Model names must include provider prefix for LiteLLM (e.g., gemini/, xai/, lm_studio/)
    default_model: str = "lm_studio/zai-org/glm-4.7-flash"
    orchestrator_model: str = "xai/grok-4-1-fast-reasoning"
    evaluator_model: str = "xai/grok-4-1-fast-reasoning"
    use_mock_llm: bool = False

    # Agent Limits
    max_react_iterations: int = 30
    max_subtask_iterations: int = 8
    num_hypothesis_agents: int = 3
    tool_timeout_seconds: int = 60
    # Separate timeout for LLM API calls (reasoning models need more time)
    llm_request_timeout_seconds: int = 120
    agent_timeout_seconds: int = 300
    llm_retry_attempts: int = 2

    # Context Window Management
    context_prune_threshold_tokens: int = 40000
    context_max_messages: int = 20

    # Hypothesis Mode Options
    enable_hypothesis_synthesis: bool = False
    multi_round_evaluation_threshold: float = 0.5

    # LLM Rate Limiting
    llm_rate_limit_rpm: int = 30  # Requests per minute
    llm_rate_limit_tpm: int = 100000  # Tokens per minute

    # LLM Fallback & Degradation
    llm_fallback_model: str | None = None  # Fallback model if primary fails after retries
    llm_max_retries: int = 3  # Max retries before giving up or falling back

    # Sandbox Configuration
    sandbox_image: str = "agentica-sandbox:latest"
    sandbox_base_port: int = 5173
    max_concurrent_sandboxes: int = 10
    sandbox_cleanup_interval_minutes: int = 5
    sandbox_ttl_minutes: int = 30
    sandbox_allow_unrestricted_commands: bool = True

    # Session Configuration
    session_ttl_minutes: int = 60

    # Database Configuration
    database_path: str = "./data/sessions.db"

    # Server Configuration
    backend_port: int = 8000
    frontend_port: int = 3000
    cors_origins: str | list[str] = ["http://localhost:3000"]
    log_level: str = "INFO"
    log_format: str = "json"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        """Parse CORS origins from string or list.

        Accepts:
        - JSON array: '["http://localhost:3000"]'
        - Comma-separated: 'http://localhost:3000,http://localhost:8080'
        - Single value: 'http://localhost:3000'
        - Already a list: ["http://localhost:3000"]
        """
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            # Try JSON first
            if v.startswith("["):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            # Fallback to comma-separated
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return ["http://localhost:3000"]

    model_config = SettingsConfigDict(
        # Support running `uvicorn` from either the repo root or `backend/`
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    def model_post_init(self, __context: Any) -> None:
        """Export LM Studio API base to os.environ for LiteLLM discovery."""
        if self.lm_studio_api_base:
            os.environ.setdefault("LM_STUDIO_API_BASE", self.lm_studio_api_base)


def configure_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """Configure structured logging for the application.

    Sets up structlog with appropriate processors for either JSON or console output.

    Args:
        log_level: The minimum log level to emit (DEBUG, INFO, WARNING, ERROR).
        log_format: Output format - 'json' for production, 'text' for development.
    """
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Global settings instance
settings = Settings()

# Configure logging on module import
configure_logging(settings.log_level, settings.log_format)

# Create a logger for this module
logger = structlog.get_logger(__name__)
