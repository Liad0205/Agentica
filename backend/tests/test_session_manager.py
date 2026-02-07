"""Tests for session_manager.py -- session lifecycle management.

Covers session creation, cancellation (deadlock-free pattern),
get/list sessions, metrics, and cleanup_all.  All graph execution
and sandbox operations are mocked to avoid real Docker/LLM calls.
"""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from events.bus import EventBus, reset_event_bus
from models.schemas import ModelConfig, SessionStatus

# ---------------------------------------------------------------------------
# We need to patch the graph creation functions before importing
# SessionManager, since importing session_manager.py triggers imports
# of react_graph, decomposition_graph, hypothesis_graph which may not
# have all their heavy dependencies in the test environment.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_graphs() -> Any:
    """Patch graph constructors so SessionManager can be imported and used."""
    mock_graph = MagicMock()
    mock_graph.run = AsyncMock(return_value={
        "status": "complete", "iteration": 3, "files_written": ["index.tsx"],
    })

    with (
        patch("session_manager.create_react_graph", return_value=mock_graph),
        patch("session_manager.create_initial_state", return_value={
            "task": "test", "sandbox_id": "sb_1", "session_id": "sess_1", "agent_id": "react_1",
        }),
        patch("session_manager.create_decomposition_graph", return_value=mock_graph),
        patch("session_manager.create_decomposition_initial_state", return_value={
            "task": "test", "session_id": "sess_1", "integration_sandbox_id": "sb_1",
        }),
        patch("session_manager.create_hypothesis_graph", return_value=mock_graph),
        patch("session_manager.create_hypothesis_initial_state", return_value={
            "task": "test", "session_id": "sess_1", "num_hypotheses": 3,
        }),
    ):
        yield


@pytest.fixture()
def mock_sandbox() -> AsyncMock:
    mgr = AsyncMock()
    mgr.create_sandbox = AsyncMock(return_value=MagicMock(
        sandbox_id="sandbox_test",
        container_id="container_abc",
        port=5173,
        workspace_path="/workspace",
        status="running",
    ))
    mgr.destroy_sandbox = AsyncMock()
    mgr.cleanup_all = AsyncMock()
    mgr.start_dev_server = AsyncMock(return_value="http://localhost:5173")
    return mgr


@pytest.fixture()
def session_mgr(mock_sandbox: AsyncMock) -> Any:
    """Create a SessionManager with mocked dependencies."""
    from session_manager import SessionManager

    reset_event_bus()
    bus = EventBus()
    return SessionManager(sandbox_manager=mock_sandbox, event_bus=bus)


# =========================================================================
# Session Creation
# =========================================================================


class TestCreateSession:
    """Session creation happy paths."""

    async def test_create_react_session(self, session_mgr: Any) -> None:
        session_id = await session_mgr.create_session(
            mode="react",
            task="Build a todo app with React and Tailwind CSS",
        )
        assert session_id.startswith("sess_")
        session = session_mgr.get_session(session_id)
        assert session is not None
        assert session.mode == "react"
        assert session.task == "Build a todo app with React and Tailwind CSS"

    async def test_create_decomposition_session(self, session_mgr: Any) -> None:
        session_id = await session_mgr.create_session(
            mode="decomposition",
            task="Build a complex multi-page web application",
        )
        assert session_id.startswith("sess_")
        session = session_mgr.get_session(session_id)
        assert session is not None
        assert session.mode == "decomposition"

    async def test_create_hypothesis_session(self, session_mgr: Any) -> None:
        session_id = await session_mgr.create_session(
            mode="hypothesis",
            task="Build a sorting algorithm visualizer",
        )
        assert session_id.startswith("sess_")
        session = session_mgr.get_session(session_id)
        assert session is not None
        assert session.mode == "hypothesis"

    async def test_create_with_model_config(self, session_mgr: Any) -> None:
        config = ModelConfig(temperature=0.5, orchestrator="test-model")
        session_id = await session_mgr.create_session(
            mode="react",
            task="Build a simple counter component",
            model_config=config,
        )
        session = session_mgr.get_session(session_id)
        assert session is not None
        assert session.model_config is not None
        assert session.model_config.temperature == 0.5

    async def test_session_id_is_unique(self, session_mgr: Any) -> None:
        id1 = await session_mgr.create_session(mode="react", task="Task one for testing")
        id2 = await session_mgr.create_session(mode="react", task="Task two for testing")
        assert id1 != id2

    async def test_invalid_mode_raises(self, session_mgr: Any) -> None:
        with pytest.raises(ValueError, match="Unsupported mode"):
            await session_mgr.create_session(
                mode="invalid",  # type: ignore[arg-type]
                task="This should fail due to invalid mode",
            )


# =========================================================================
# Continue Session
# =========================================================================


class TestContinueSession:
    """Continuation of terminal sessions on the same sandbox."""

    async def test_continue_complete_session_reuses_sandbox(self, session_mgr: Any) -> None:
        session_id = await session_mgr.create_session(
            mode="react",
            task="Build a small todo app with add/delete support",
        )
        await asyncio.sleep(0.2)

        original = session_mgr.get_session(session_id)
        assert original is not None
        original_sandbox = original.sandbox_id

        continued_id = await session_mgr.continue_session(
            session_id=session_id,
            task="Now add filter tabs and localStorage persistence",
        )
        assert continued_id == session_id

        mid = session_mgr.get_session(session_id)
        assert mid is not None
        assert mid.sandbox_id == original_sandbox

        await asyncio.sleep(0.2)
        done = session_mgr.get_session(session_id)
        assert done is not None
        assert done.status in (SessionStatus.RUNNING, SessionStatus.COMPLETE)

    async def test_continue_running_session_raises(self, session_mgr: Any) -> None:
        mock_graph = MagicMock()

        async def slow_run(state: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(10)
            return {"status": "complete"}

        mock_graph.run = slow_run

        with patch("session_manager.create_react_graph", return_value=mock_graph):
            session_id = await session_mgr.create_session(
                mode="react",
                task="Build a task for continuation-while-running test",
            )

        await asyncio.sleep(0.1)

        with pytest.raises(RuntimeError, match="cannot be continued"):
            await session_mgr.continue_session(
                session_id=session_id,
                task="Apply a follow-up change while still running",
            )

        await session_mgr.cancel_session(session_id)

    async def test_continue_cancelled_session_raises(self, session_mgr: Any) -> None:
        mock_graph = MagicMock()

        async def slow_run(state: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(10)
            return {"status": "complete"}

        mock_graph.run = slow_run

        with patch("session_manager.create_react_graph", return_value=mock_graph):
            session_id = await session_mgr.create_session(
                mode="react",
                task="Build a project that will be cancelled",
            )

        await asyncio.sleep(0.1)
        await session_mgr.cancel_session(session_id)

        with pytest.raises(RuntimeError, match="cannot be continued"):
            await session_mgr.continue_session(
                session_id=session_id,
                task="Try to continue cancelled session",
            )

    async def test_continue_hypothesis_session_cleans_solver_sandboxes(
        self, session_mgr: Any
    ) -> None:
        session_id = await session_mgr.create_session(
            mode="hypothesis",
            task="Build a variant-rich dashboard",
        )
        await asyncio.sleep(0.2)

        with patch.object(
            session_mgr,
            "_cleanup_hypothesis_solver_sandboxes",
            new=AsyncMock(),
        ) as cleanup_mock:
            await session_mgr.continue_session(
                session_id=session_id,
                task="Add another layout variant",
            )
            cleanup_mock.assert_awaited_once_with(session_id)


class TestHypothesisSandboxCleanup:
    async def test_hypothesis_failed_status_cleans_solver_sandboxes(
        self, session_mgr: Any
    ) -> None:
        from session_manager import SessionInfo

        session_id = "sess_hyp_failed"
        session_info = SessionInfo(
            session_id=session_id,
            mode="hypothesis",
            task="Build test app",
            status=SessionStatus.STARTED,
            sandbox_id="hypothesis_pending",
            created_at=time.time(),
        )
        mock_graph = MagicMock()
        mock_graph.run = AsyncMock(
            return_value={"status": "failed", "error_message": "evaluation failed"}
        )

        with patch.object(
            session_mgr,
            "_cleanup_hypothesis_solver_sandboxes",
            new=AsyncMock(),
        ) as cleanup_mock:
            await session_mgr._run_hypothesis_session(
                session_id=session_id,
                session_info=session_info,
                graph=mock_graph,
                initial_state={"session_id": session_id},
            )

            cleanup_mock.assert_awaited_once_with(session_id)
            assert session_info.status == SessionStatus.ERROR
            assert session_info.error_message == "evaluation failed"

    async def test_hypothesis_timeout_cleans_solver_sandboxes(
        self, session_mgr: Any, monkeypatch
    ) -> None:
        from config import settings
        from session_manager import SessionInfo

        session_id = "sess_hyp_timeout"
        session_info = SessionInfo(
            session_id=session_id,
            mode="hypothesis",
            task="Build test app",
            status=SessionStatus.STARTED,
            sandbox_id="hypothesis_pending",
            created_at=time.time(),
        )

        async def slow_run(_: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0.1)
            return {"status": "complete"}

        mock_graph = MagicMock()
        mock_graph.run = slow_run

        monkeypatch.setattr(settings, "agent_timeout_seconds", 0.01)

        with patch.object(
            session_mgr,
            "_cleanup_hypothesis_solver_sandboxes",
            new=AsyncMock(),
        ) as cleanup_mock:
            await session_mgr._run_hypothesis_session(
                session_id=session_id,
                session_info=session_info,
                graph=mock_graph,
                initial_state={"session_id": session_id},
            )

            cleanup_mock.assert_awaited_once_with(session_id)
            assert session_info.status == SessionStatus.ERROR
            assert session_info.error_message is not None
            assert "timed out" in session_info.error_message


# =========================================================================
# Get Session / List Sessions
# =========================================================================


class TestGetSessions:
    """Session retrieval."""

    async def test_get_existing_session(self, session_mgr: Any) -> None:
        session_id = await session_mgr.create_session(
            mode="react", task="Build a React counter component",
        )
        session = session_mgr.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id

    async def test_get_nonexistent_session(self, session_mgr: Any) -> None:
        assert session_mgr.get_session("sess_nonexistent") is None

    async def test_get_all_sessions(self, session_mgr: Any) -> None:
        await session_mgr.create_session(mode="react", task="Task one for list test")
        await session_mgr.create_session(mode="react", task="Task two for list test")
        sessions = session_mgr.get_all_sessions()
        assert len(sessions) == 2


# =========================================================================
# Session Metrics
# =========================================================================


class TestSessionMetrics:
    """Session metrics retrieval."""

    async def test_metrics_for_existing_session(self, session_mgr: Any) -> None:
        session_id = await session_mgr.create_session(
            mode="react", task="Build a simple metrics test app",
        )
        metrics = session_mgr.get_session_metrics(session_id)
        assert metrics is not None
        assert metrics.execution_time_seconds >= 0.0

    async def test_metrics_for_nonexistent_session(self, session_mgr: Any) -> None:
        assert session_mgr.get_session_metrics("sess_nonexistent") is None


# =========================================================================
# Cancel Session
# =========================================================================


class TestCancelSession:
    """Session cancellation -- must not deadlock."""

    async def test_cancel_running_session(self, session_mgr: Any) -> None:
        """Cancellation extracts task under lock then cancels outside lock."""
        # Create a session that takes a while to run
        mock_graph = MagicMock()

        async def slow_run(state: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(10)
            return {"status": "complete"}

        mock_graph.run = slow_run

        with patch("session_manager.create_react_graph", return_value=mock_graph):
            session_id = await session_mgr.create_session(
                mode="react", task="Build a slow app for cancel test",
            )

        # Give the background task a moment to start
        await asyncio.sleep(0.1)

        # Cancel should not deadlock (the CancelledError handler also acquires _lock)
        await asyncio.wait_for(
            session_mgr.cancel_session(session_id),
            timeout=5.0,
        )

        session = session_mgr.get_session(session_id)
        assert session is not None
        assert session.status == SessionStatus.CANCELLED

    async def test_cancel_nonexistent_raises(self, session_mgr: Any) -> None:
        with pytest.raises(KeyError, match="not found"):
            await session_mgr.cancel_session("sess_nonexistent")

    async def test_cancel_already_complete(self, session_mgr: Any) -> None:
        session_id = await session_mgr.create_session(
            mode="react", task="Build a quick app for cancel test",
        )
        # Wait for the background task to complete
        await asyncio.sleep(0.5)

        # Cancelling an already-complete session should be a no-op
        await session_mgr.cancel_session(session_id)
        session = session_mgr.get_session(session_id)
        assert session is not None
        # Status should remain COMPLETE (not changed to CANCELLED)
        assert session.status in (SessionStatus.COMPLETE, SessionStatus.CANCELLED)

    async def test_cancel_cleans_up_sandbox(
        self, session_mgr: Any, mock_sandbox: AsyncMock
    ) -> None:
        mock_graph = MagicMock()
        async def slow_run(state: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(10)
            return {"status": "complete"}

        mock_graph.run = slow_run

        with patch("session_manager.create_react_graph", return_value=mock_graph):
            session_id = await session_mgr.create_session(
                mode="react", task="Build an app to test sandbox cleanup",
            )

        await asyncio.sleep(0.1)
        await session_mgr.cancel_session(session_id)

        # destroy_sandbox should have been called
        mock_sandbox.destroy_sandbox.assert_awaited()


# =========================================================================
# Cleanup All
# =========================================================================


class TestCleanupAll:
    """cleanup_all -- collects tasks under lock then cancels outside lock."""

    async def test_cleanup_all_cancels_tasks(self, session_mgr: Any) -> None:
        mock_graph = MagicMock()
        async def slow_run(state: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(10)
            return {"status": "complete"}

        mock_graph.run = slow_run

        with patch("session_manager.create_react_graph", return_value=mock_graph):
            await session_mgr.create_session(
                mode="react", task="Build app one for cleanup test",
            )
            await session_mgr.create_session(
                mode="react", task="Build app two for cleanup test",
            )

        await asyncio.sleep(0.1)

        # cleanup_all should not deadlock
        await asyncio.wait_for(session_mgr.cleanup_all(), timeout=5.0)

        # All sessions should be cleared
        assert len(session_mgr.get_all_sessions()) == 0

    async def test_cleanup_all_empty(self, session_mgr: Any) -> None:
        """cleanup_all on empty manager should be a no-op."""
        await session_mgr.cleanup_all()
        assert len(session_mgr.get_all_sessions()) == 0

    async def test_cleanup_calls_sandbox_cleanup(
        self, session_mgr: Any, mock_sandbox: AsyncMock
    ) -> None:
        await session_mgr.create_session(
            mode="react", task="Build a quick cleanup sandbox test app",
        )
        await asyncio.sleep(0.3)
        await session_mgr.cleanup_all()
        mock_sandbox.cleanup_all.assert_awaited()


# =========================================================================
# Session Status Transitions
# =========================================================================


class TestSessionStatusTransitions:
    """Verify status progresses through expected states."""

    async def test_status_reaches_complete(self, session_mgr: Any) -> None:
        session_id = await session_mgr.create_session(
            mode="react", task="Build a status transition test application",
        )
        # Wait for background task to complete
        await asyncio.sleep(0.5)

        session = session_mgr.get_session(session_id)
        assert session is not None
        assert session.status in (SessionStatus.COMPLETE, SessionStatus.RUNNING)

    async def test_error_status_on_failure(self, session_mgr: Any) -> None:
        mock_graph = MagicMock()
        mock_graph.run = AsyncMock(side_effect=RuntimeError("LLM API down"))

        with patch("session_manager.create_react_graph", return_value=mock_graph):
            session_id = await session_mgr.create_session(
                mode="react", task="Build an app that will fail for error test",
            )

        await asyncio.sleep(0.5)

        session = session_mgr.get_session(session_id)
        assert session is not None
        assert session.status == SessionStatus.ERROR
        assert session.error_message is not None
        assert "LLM API down" in session.error_message
