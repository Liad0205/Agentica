"""Shared test fixtures for backend tests.

Provides mock objects for SandboxManager, EventBus, LLM clients,
and configuration overrides to ensure tests never touch real Docker
containers or LLM APIs.
"""

import sys
from collections import defaultdict
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Ensure the backend package root is on sys.path so that absolute imports
# like ``from sandbox.security import ...`` resolve correctly when running
# pytest from the repository root.
_backend_root = str(
    __import__("pathlib").Path(__file__).resolve().parent.parent
)
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

from agents.utils import LLMResponse, MockLLMClient, ToolCallData  # noqa: E402
from events.bus import EventBus, reset_event_bus  # noqa: E402
from events.types import AgentEvent, LLMMetrics  # noqa: E402
from sandbox.docker_sandbox import CommandResult, FileInfo  # noqa: E402

# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------


@pytest.fixture()
def event_bus() -> EventBus:
    """Return a fresh EventBus instance for each test."""
    reset_event_bus()
    bus = EventBus()
    return bus


# ---------------------------------------------------------------------------
# Mock Sandbox Manager
# ---------------------------------------------------------------------------


def _make_mock_sandbox_manager() -> AsyncMock:
    """Create a fully-typed mock SandboxManager.

    All methods are AsyncMock by default.  Callers can override
    return values per-test.
    """
    mgr = AsyncMock()
    mgr.create_sandbox = AsyncMock(return_value=MagicMock(
        sandbox_id="sandbox_test123",
        container_id="container_abc",
        port=5173,
        workspace_path="/workspace",
        status="running",
    ))
    mgr.destroy_sandbox = AsyncMock()
    mgr.cleanup_all = AsyncMock()
    mgr.write_file = AsyncMock()
    mgr.read_file = AsyncMock(return_value="file content")
    mgr.list_files = AsyncMock(return_value=[
        FileInfo(name="index.tsx", path="src/index.tsx", is_directory=False, size=100),
        FileInfo(name="components", path="src/components", is_directory=True, size=0),
    ])
    mgr.list_files_recursive = AsyncMock(return_value=[
        FileInfo(name="index.tsx", path="src/index.tsx", is_directory=False, size=100),
    ])
    mgr.execute_command = AsyncMock(return_value=CommandResult(
        stdout="OK", stderr="", exit_code=0, timed_out=False,
    ))
    mgr.start_dev_server = AsyncMock(return_value="http://localhost:5173")
    return mgr


@pytest.fixture()
def mock_sandbox_manager() -> AsyncMock:
    """Provide a mock SandboxManager for each test."""
    return _make_mock_sandbox_manager()


# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set USE_MOCK_LLM=true in the environment."""
    monkeypatch.setenv("USE_MOCK_LLM", "true")


# ---------------------------------------------------------------------------
# Response Factories
# ---------------------------------------------------------------------------


def make_llm_response(
    content: str = "",
    tool_calls: list[ToolCallData] | None = None,
    finish_reason: str = "stop",
) -> LLMResponse:
    """Create an LLMResponse with sensible defaults."""
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="tool_calls" if tool_calls else finish_reason,
        metrics=LLMMetrics(model="mock", input_tokens=10, output_tokens=20, latency_ms=100),
    )


def make_tool_call(name: str, args: dict[str, Any], call_id: str = "tc_1") -> ToolCallData:
    """Create a ToolCallData."""
    return ToolCallData(id=call_id, name=name, args=args)


# ---------------------------------------------------------------------------
# Event Collection Helper
# ---------------------------------------------------------------------------


async def collect_events(event_bus: EventBus, session_id: str) -> list[AgentEvent]:
    """Subscribe to a session and drain all buffered events after graph.run()."""
    queue = event_bus.subscribe(session_id)
    events: list[AgentEvent] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


# ---------------------------------------------------------------------------
# RoutingMockLLMClient
# ---------------------------------------------------------------------------


class RoutingMockLLMClient(MockLLMClient):
    """Mock LLM that routes responses by agent_id for fan-out tests.

    The fan-out graphs (Decomposition, Hypothesis) run multiple agents
    concurrently sharing one ``MockLLMClient``.  Since asyncio interleaving
    makes response ordering non-deterministic, this mock routes responses
    by ``agent_id``.

    Args:
        response_map: Dict mapping agent_id -> list of LLMResponses.
                      Use ``"default"`` for calls without a matching agent_id.
    """

    def __init__(
        self,
        response_map: dict[str, list[LLMResponse]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._response_map: dict[str, list[LLMResponse]] = {
            k: list(v) for k, v in response_map.items()
        }
        self._indexes: dict[str, int] = defaultdict(int)

    async def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> LLMResponse:
        self.call_history.append({"agent_id": agent_id, "messages": messages})
        key = agent_id if agent_id in self._response_map else "default"
        idx = self._indexes[key]
        self._indexes[key] = idx + 1
        return self._response_map[key][idx]
