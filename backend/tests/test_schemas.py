"""Tests for models/schemas.py -- Pydantic request/response models.

Validates model construction, validation rules, enum values, and
the ``llm_config`` alias that was introduced to avoid clashing with
Pydantic v2's reserved ``model_config`` attribute.
"""

import pytest
from pydantic import ValidationError

from models.schemas import (
    AgentMode,
    CreateSessionRequest,
    FileContentResponse,
    FileInfo,
    HealthResponse,
    LLMMetrics,
    ModelConfig,
    SessionDetailResponse,
    SessionMetrics,
    SessionResponse,
    SessionStatus,
)

# =========================================================================
# AgentMode enum
# =========================================================================


class TestAgentMode:
    """Agent mode enum values."""

    def test_values(self) -> None:
        assert AgentMode.REACT == "react"
        assert AgentMode.DECOMPOSITION == "decomposition"
        assert AgentMode.HYPOTHESIS == "hypothesis"

    def test_all_values(self) -> None:
        assert set(AgentMode) == {
            AgentMode.REACT,
            AgentMode.DECOMPOSITION,
            AgentMode.HYPOTHESIS,
        }

    def test_string_coercion(self) -> None:
        assert AgentMode("react") == AgentMode.REACT


# =========================================================================
# SessionStatus enum
# =========================================================================


class TestSessionStatus:
    """Session lifecycle status enum."""

    def test_all_statuses(self) -> None:
        expected = {"idle", "started", "running", "complete", "error", "cancelled"}
        assert {s.value for s in SessionStatus} == expected


# =========================================================================
# CreateSessionRequest
# =========================================================================


class TestCreateSessionRequest:
    """Request model for creating a session."""

    def test_valid_request(self) -> None:
        req = CreateSessionRequest(
            mode=AgentMode.REACT,
            task="Build a todo app with React and Tailwind CSS",
        )
        assert req.mode == AgentMode.REACT
        assert "todo" in req.task

    def test_task_min_length(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            CreateSessionRequest(mode=AgentMode.REACT, task="short")
        errors = exc_info.value.errors()
        assert any("task" in str(e.get("loc")) for e in errors)

    def test_task_max_length(self) -> None:
        with pytest.raises(ValidationError):
            CreateSessionRequest(
                mode=AgentMode.REACT,
                task="x" * 10001,
            )

    def test_llm_config_alias(self) -> None:
        """The field ``model_config_override`` can be set via the
        ``llm_config`` alias to avoid the Pydantic v2 ``model_config``
        reserved name conflict."""
        req = CreateSessionRequest(
            mode=AgentMode.DECOMPOSITION,
            task="Build a complex multi-page web application",
            **{"llm_config": ModelConfig(temperature=0.5)},
        )
        assert req.model_config_override is not None
        assert req.model_config_override.temperature == 0.5

    def test_llm_config_from_dict(self) -> None:
        """Construct via dict with alias name."""
        data = {
            "mode": "react",
            "task": "Build a React counter component",
            "llm_config": {"temperature": 1.0},
        }
        req = CreateSessionRequest.model_validate(data)
        assert req.model_config_override is not None
        assert req.model_config_override.temperature == 1.0

    def test_no_config_override(self) -> None:
        req = CreateSessionRequest(
            mode=AgentMode.REACT,
            task="Build a todo app with React and Tailwind CSS",
        )
        assert req.model_config_override is None

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValidationError):
            CreateSessionRequest(
                mode="invalid_mode",  # type: ignore[arg-type]
                task="Build a todo app with React and Tailwind CSS",
            )


# =========================================================================
# ModelConfig
# =========================================================================


class TestModelConfig:
    """Model configuration validation."""

    def test_defaults_are_none(self) -> None:
        mc = ModelConfig()
        assert mc.orchestrator is None
        assert mc.sub_agent is None
        assert mc.evaluator is None
        assert mc.temperature is None

    def test_temperature_bounds(self) -> None:
        mc = ModelConfig(temperature=0.0)
        assert mc.temperature == 0.0

        mc = ModelConfig(temperature=2.0)
        assert mc.temperature == 2.0

    def test_temperature_too_high(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(temperature=2.5)

    def test_temperature_negative(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(temperature=-0.1)

    def test_full_config(self) -> None:
        mc = ModelConfig(
            orchestrator="grok-4-1-fast-reasoning",
            sub_agent="gemini-3.0-flash-preview",
            evaluator="grok-4-1-fast-reasoning",
            temperature=0.7,
        )
        assert mc.orchestrator == "grok-4-1-fast-reasoning"
        assert mc.sub_agent == "gemini-3.0-flash-preview"


# =========================================================================
# SessionResponse
# =========================================================================


class TestSessionResponse:
    """Session creation response."""

    def test_construction(self) -> None:
        resp = SessionResponse(
            session_id="sess_abc123",
            websocket_url="/ws/sess_abc123",
            status=SessionStatus.STARTED,
        )
        assert resp.session_id == "sess_abc123"
        assert resp.websocket_url == "/ws/sess_abc123"
        assert resp.status == SessionStatus.STARTED


# =========================================================================
# SessionDetailResponse
# =========================================================================


class TestSessionDetailResponse:
    """Detailed session information response."""

    def test_construction(self) -> None:
        resp = SessionDetailResponse(
            session_id="sess_abc123",
            mode="react",
            task="Build a todo app",
            status=SessionStatus.RUNNING,
            created_at=1700000000.0,
        )
        assert resp.session_id == "sess_abc123"
        assert resp.status == SessionStatus.RUNNING

    def test_optional_fields(self) -> None:
        resp = SessionDetailResponse(
            session_id="sess_abc123",
            mode="react",
            task="Build a todo app",
            status=SessionStatus.COMPLETE,
            created_at=1700000000.0,
            started_at=1700000001.0,
            completed_at=1700000060.0,
        )
        assert resp.started_at == 1700000001.0
        assert resp.completed_at == 1700000060.0

    def test_llm_config_alias_in_detail(self) -> None:
        """SessionDetailResponse also uses the ``llm_config`` alias."""
        resp = SessionDetailResponse(
            session_id="sess_abc123",
            mode="react",
            task="Build a todo app",
            status=SessionStatus.RUNNING,
            created_at=1700000000.0,
            **{"llm_config": ModelConfig(temperature=0.3)},
        )
        assert resp.model_config_data is not None
        assert resp.model_config_data.temperature == 0.3


# =========================================================================
# SessionMetrics
# =========================================================================


class TestSessionMetrics:
    """Aggregate metrics for a session."""

    def test_defaults(self) -> None:
        m = SessionMetrics()
        assert m.total_input_tokens == 0
        assert m.total_output_tokens == 0
        assert m.total_llm_calls == 0
        assert m.total_tool_calls == 0
        assert m.execution_time_seconds == 0.0

    def test_construction_with_values(self) -> None:
        m = SessionMetrics(
            total_input_tokens=1000,
            total_output_tokens=500,
            total_llm_calls=5,
            total_tool_calls=10,
            execution_time_seconds=45.5,
        )
        assert m.total_input_tokens == 1000
        assert m.total_output_tokens == 500

    def test_negative_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SessionMetrics(total_input_tokens=-1)


# =========================================================================
# LLMMetrics
# =========================================================================


class TestLLMMetrics:
    """Per-call LLM metrics."""

    def test_construction(self) -> None:
        m = LLMMetrics(
            model="gemini-3.0-flash-preview",
            input_tokens=100,
            output_tokens=50,
            latency_ms=250,
        )
        assert m.model == "gemini-3.0-flash-preview"
        assert m.latency_ms == 250

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMMetrics(
                model="test",
                input_tokens=-1,
                output_tokens=0,
                latency_ms=0,
            )


# =========================================================================
# FileInfo & FileContentResponse
# =========================================================================


class TestFileInfo:
    """File information model."""

    def test_file(self) -> None:
        fi = FileInfo(name="App.tsx", path="src/App.tsx", is_directory=False, size=512)
        assert fi.name == "App.tsx"
        assert fi.is_directory is False

    def test_directory(self) -> None:
        fi = FileInfo(name="src", path="src", is_directory=True)
        assert fi.is_directory is True
        assert fi.size is None  # default


class TestFileContentResponse:
    """File content response."""

    def test_construction(self) -> None:
        resp = FileContentResponse(
            path="src/App.tsx",
            content="export default function App() {}",
            size=31,
        )
        assert resp.path == "src/App.tsx"


# =========================================================================
# HealthResponse
# =========================================================================


class TestHealthResponse:
    """Health check response."""

    def test_healthy(self) -> None:
        resp = HealthResponse(status="healthy", timestamp=1700000000.0)
        assert resp.status == "healthy"
        assert resp.version == "0.1.0"

    def test_invalid_status(self) -> None:
        with pytest.raises(ValidationError):
            HealthResponse(status="degraded", timestamp=1700000000.0)  # type: ignore[arg-type]
