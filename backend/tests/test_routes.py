"""Tests for api/routes.py -- HTTP endpoint handlers.

Uses FastAPI TestClient (backed by httpx) with a mocked SessionManager.
No real Docker or LLM calls are made.
"""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router, set_session_manager
from models.schemas import ModelConfig, SessionMetrics, SessionStatus
from sandbox.docker_sandbox import FileInfo

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_session_info(
    session_id: str = "sess_aabb11223344",
    mode: str = "react",
    task: str = "Build a todo app with React",
    status: SessionStatus = SessionStatus.RUNNING,
    sandbox_id: str = "sandbox_test123456",
    created_at: float = 1700000000.0,
    started_at: float | None = 1700000001.0,
    completed_at: float | None = None,
    error_message: str | None = None,
    model_config: ModelConfig | None = None,
) -> MagicMock:
    """Create a mock SessionInfo object."""
    session = MagicMock()
    session.session_id = session_id
    session.mode = mode
    session.task = task
    session.status = status
    session.sandbox_id = sandbox_id
    session.created_at = created_at
    session.started_at = started_at
    session.completed_at = completed_at
    session.error_message = error_message
    session.model_config = model_config
    return session


@pytest.fixture()
def mock_session_manager() -> MagicMock:
    """Create a mock SessionManager."""
    mgr = MagicMock()
    mgr.create_session = AsyncMock(return_value="sess_aabb11223344")
    mgr.continue_session = AsyncMock(return_value="sess_aabb11223344")
    mgr.get_session = MagicMock(return_value=_make_session_info())
    mgr.get_all_sessions = MagicMock(return_value=[_make_session_info()])
    mgr.get_session_metrics = MagicMock(return_value=SessionMetrics(
        execution_time_seconds=10.5,
    ))
    mgr.cancel_session = AsyncMock()
    mgr.cleanup_all = AsyncMock()
    mgr.sandbox_manager = AsyncMock()
    mgr.sandbox_manager.list_files_recursive = AsyncMock(return_value=[
        FileInfo(name="index.tsx", path="src/index.tsx", is_directory=False, size=200),
    ])
    mgr.sandbox_manager.read_file = AsyncMock(return_value="export default function App() {}")
    mgr.sandbox_manager.export_archive = AsyncMock(return_value=b"fake-tar-content")
    mgr.sandbox_manager.is_docker_available = MagicMock(return_value=True)
    mgr.sandbox_manager.get_active_sandbox_count = MagicMock(return_value=0)
    mgr.session_store = None
    return mgr


@pytest.fixture()
def client(mock_session_manager: MagicMock) -> Generator[TestClient, None, None]:
    """Create a FastAPI TestClient with mocked session manager."""
    app = FastAPI()
    app.include_router(router)
    set_session_manager(mock_session_manager)  # type: ignore[arg-type]
    with TestClient(app) as c:
        yield c


# =========================================================================
# Health Check
# =========================================================================


class TestHealthCheck:
    """GET /health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "0.1.0"


# =========================================================================
# Create Session
# =========================================================================


class TestCreateSession:
    """POST /api/sessions."""

    def test_create_session_success(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        resp = client.post(
            "/api/sessions",
            json={
                "mode": "react",
                "task": "Build a todo app with React and Tailwind CSS",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["session_id"] == "sess_aabb11223344"
        assert data["websocket_url"] == "/ws/sess_aabb11223344"

    def test_create_session_with_llm_config(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        resp = client.post(
            "/api/sessions",
            json={
                "mode": "decomposition",
                "task": "Build a multi-page web application with routing",
                "llm_config": {"temperature": 0.5},
            },
        )
        assert resp.status_code == 201

    def test_create_session_task_too_short(self, client: TestClient) -> None:
        resp = client.post(
            "/api/sessions",
            json={"mode": "react", "task": "short"},
        )
        assert resp.status_code == 422  # Validation error

    def test_create_session_invalid_mode(self, client: TestClient) -> None:
        resp = client.post(
            "/api/sessions",
            json={"mode": "invalid", "task": "Build a todo app with React and Tailwind CSS"},
        )
        assert resp.status_code == 422

    def test_create_session_missing_task(self, client: TestClient) -> None:
        resp = client.post("/api/sessions", json={"mode": "react"})
        assert resp.status_code == 422

    def test_create_session_failure(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.create_session = AsyncMock(
            side_effect=RuntimeError("Docker not available")
        )
        resp = client.post(
            "/api/sessions",
            json={
                "mode": "react",
                "task": "Build a todo app that will fail",
            },
        )
        assert resp.status_code == 500
        # Error detail should be generic (no internal info leakage)
        assert resp.json()["detail"] == "Failed to create session"


# =========================================================================
# Clear Sessions
# =========================================================================


class TestClearSessions:
    """DELETE /api/sessions."""

    def test_clear_sessions_requires_force_for_running(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.get_all_sessions.return_value = [
            _make_session_info(status=SessionStatus.RUNNING)
        ]

        resp = client.delete("/api/sessions")
        assert resp.status_code == 409
        assert "force=true" in resp.json()["detail"]
        mock_session_manager.cleanup_all.assert_not_awaited()

    def test_clear_sessions_success(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.get_all_sessions.return_value = [
            _make_session_info(status=SessionStatus.COMPLETE)
        ]

        resp = client.delete("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["message"] == "All sessions cleared"
        assert data["in_memory_cleared"] == 1
        assert data["running_cleared"] == 0
        mock_session_manager.cleanup_all.assert_awaited_once()

    def test_clear_sessions_with_persistence(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.get_all_sessions.return_value = [
            _make_session_info(status=SessionStatus.COMPLETE)
        ]
        store = AsyncMock()
        store.clear_all = AsyncMock(return_value=7)
        mock_session_manager.session_store = store

        resp = client.delete("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["persisted_cleared"] == 7
        store.clear_all.assert_awaited_once()


# =========================================================================
# Continue Session
# =========================================================================


class TestContinueSession:
    """POST /api/sessions/{session_id}/continue."""

    def test_continue_session_success(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        # After continue_session succeeds, the route calls get_session for response
        mock_session_manager.get_session.return_value = _make_session_info(
            status=SessionStatus.STARTED
        )

        resp = client.post(
            "/api/sessions/sess_aabb11223344/continue",
            json={"task": "Now add persistence and filtering."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess_aabb11223344"
        assert data["websocket_url"] == "/ws/sess_aabb11223344"
        mock_session_manager.continue_session.assert_awaited_once()

    def test_continue_session_not_found(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        # SessionManager raises KeyError when session is not found
        mock_session_manager.continue_session = AsyncMock(
            side_effect=KeyError("Session 'sess_111111111111' not found")
        )

        resp = client.post(
            "/api/sessions/sess_111111111111/continue",
            json={"task": "Add tests for edge cases in forms."},
        )
        assert resp.status_code == 404

    def test_continue_session_stale(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        # SessionManager raises RuntimeError with specific message for stale sessions
        mock_session_manager.continue_session = AsyncMock(
            side_effect=RuntimeError(
                "Session 'sess_aabb11223344' exists in history but cannot be "
                "continued because the server was restarted and the "
                "sandbox environment is no longer available. "
                "Please start a new session."
            )
        )

        resp = client.post(
            "/api/sessions/sess_aabb11223344/continue",
            json={"task": "Follow up on previous work."},
        )
        assert resp.status_code == 410  # HTTP Gone
        assert "server was restarted" in resp.json()["detail"]

    def test_continue_session_invalid_state(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.continue_session = AsyncMock(
            side_effect=RuntimeError("Session is currently running and cannot be continued")
        )

        resp = client.post(
            "/api/sessions/sess_aabb11223344/continue",
            json={"task": "Apply follow-up improvements to the UI."},
        )
        assert resp.status_code == 400
        assert "cannot be continued" in resp.json()["detail"]


# =========================================================================
# Get Session
# =========================================================================


class TestGetSession:
    """GET /api/sessions/{session_id}."""

    def test_get_session_success(self, client: TestClient) -> None:
        resp = client.get("/api/sessions/sess_aabb11223344")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess_aabb11223344"
        assert data["mode"] == "react"
        assert data["status"] == "running"

    def test_get_session_not_found(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.get_session.return_value = None
        resp = client.get("/api/sessions/sess_000000000000")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_get_session_falls_back_to_persisted_store(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.get_session.return_value = None
        store = MagicMock()
        store.get_session = AsyncMock(return_value={
            "id": "sess_ddbb00112300",
            "mode": "decomposition",
            "task": "Persisted task",
            "status": "running",
            "llm_config": {"temperature": 0.4},
            "result_summary": None,
            "created_at": 1700000100.0,
            "updated_at": 1700000200.0,
        })
        mock_session_manager.session_store = store

        resp = client.get("/api/sessions/sess_ddbb00112300")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess_ddbb00112300"
        assert data["mode"] == "decomposition"
        assert data["task"] == "Persisted task"
        assert data["status"] == "running"


# =========================================================================
# List Sessions
# =========================================================================


class TestListSessions:
    """GET /api/sessions."""

    def test_list_sessions(self, client: TestClient) -> None:
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_list_sessions_with_limit(self, client: TestClient) -> None:
        resp = client.get("/api/sessions?limit=5")
        assert resp.status_code == 200

    def test_list_sessions_falls_back_to_persisted_store(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.get_all_sessions.return_value = []

        store = MagicMock()
        store.list_sessions = AsyncMock(return_value=[
            {
                "id": "sess_ddbb00000001",
                "mode": "react",
                "task": "Persisted task one",
                "status": "complete",
                "result_summary": "ok",
                "created_at": 1700000000.0,
                "updated_at": 1700000300.0,
            },
        ])
        store.get_metrics = AsyncMock(return_value={
            "prompt_tokens": 120,
            "completion_tokens": 80,
            "llm_calls": 3,
            "tool_calls": 4,
            "duration_ms": 9000,
        })
        mock_session_manager.session_store = store

        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["session_id"] == "sess_ddbb00000001"
        assert data[0]["status"] == "complete"
        assert data[0]["metrics"]["total_input_tokens"] == 120
        assert data[0]["metrics"]["total_output_tokens"] == 80
        assert data[0]["metrics"]["execution_time_seconds"] == 9.0


# =========================================================================
# Cancel Session
# =========================================================================


class TestCancelSession:
    """POST /api/sessions/{session_id}/cancel."""

    def test_cancel_session_success(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.get_session.return_value = _make_session_info(
            status=SessionStatus.RUNNING
        )
        resp = client.post("/api/sessions/sess_aabb11223344/cancel")
        assert resp.status_code == 200
        assert "cancelled" in resp.json()["message"]

    def test_cancel_session_not_found(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.get_session.return_value = None
        resp = client.post("/api/sessions/sess_000000000000/cancel")
        assert resp.status_code == 404

    def test_cancel_already_complete(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.get_session.return_value = _make_session_info(
            status=SessionStatus.COMPLETE
        )
        resp = client.post("/api/sessions/sess_aabb11223344/cancel")
        assert resp.status_code == 400
        assert "already" in resp.json()["detail"]

    def test_cancel_already_errored(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.get_session.return_value = _make_session_info(
            status=SessionStatus.ERROR
        )
        resp = client.post("/api/sessions/sess_aabb11223344/cancel")
        assert resp.status_code == 400

    def test_cancel_already_cancelled(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.get_session.return_value = _make_session_info(
            status=SessionStatus.CANCELLED
        )
        resp = client.post("/api/sessions/sess_aabb11223344/cancel")
        assert resp.status_code == 400


# =========================================================================
# Get Session Metrics
# =========================================================================


class TestGetSessionMetrics:
    """GET /api/sessions/{session_id}/metrics."""

    def test_get_metrics_success(self, client: TestClient) -> None:
        resp = client.get("/api/sessions/sess_aabb11223344/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_input_tokens" in data
        assert "execution_time_seconds" in data

    def test_get_metrics_not_found(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.get_session_metrics.return_value = None
        resp = client.get("/api/sessions/sess_000000000000/metrics")
        assert resp.status_code == 404

    def test_get_metrics_falls_back_to_persisted_store(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.get_session_metrics.return_value = None
        store = MagicMock()
        store.get_session = AsyncMock(return_value={
            "id": "sess_ddbb00112233",
            "status": "complete",
            "created_at": 1700000000.0,
            "updated_at": 1700000200.0,
        })
        store.get_metrics = AsyncMock(return_value={
            "prompt_tokens": 40,
            "completion_tokens": 30,
            "llm_calls": 2,
            "tool_calls": 1,
            "duration_ms": 5000,
        })
        mock_session_manager.session_store = store

        resp = client.get("/api/sessions/sess_ddbb00112233/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_input_tokens"] == 40
        assert data["total_output_tokens"] == 30
        assert data["total_llm_calls"] == 2
        assert data["total_tool_calls"] == 1
        assert data["execution_time_seconds"] == 5.0


# =========================================================================
# Get Files
# =========================================================================


class TestGetFiles:
    """GET /api/sessions/{session_id}/files."""

    def test_get_files_success(self, client: TestClient) -> None:
        resp = client.get("/api/sessions/sess_aabb11223344/files")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["name"] == "index.tsx"

    def test_get_files_session_not_found(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.get_session.return_value = None
        resp = client.get("/api/sessions/sess_000000000000/files")
        assert resp.status_code == 404

    def test_get_files_path_traversal_blocked(self, client: TestClient) -> None:
        resp = client.get("/api/sessions/sess_aabb11223344/files?path=../etc")
        assert resp.status_code == 400
        assert "traversal" in resp.json()["detail"].lower()


# =========================================================================
# Get File Content
# =========================================================================


class TestGetFileContent:
    """GET /api/sessions/{session_id}/files/content."""

    def test_get_file_content_success(self, client: TestClient) -> None:
        resp = client.get(
            "/api/sessions/sess_aabb11223344/files/content?path=src/App.tsx"
        )
        assert resp.status_code == 200
        assert "export default" in resp.text

    def test_get_file_content_session_not_found(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.get_session.return_value = None
        resp = client.get(
            "/api/sessions/sess_000000000000/files/content?path=a.txt"
        )
        assert resp.status_code == 404

    def test_get_file_content_path_traversal(self, client: TestClient) -> None:
        resp = client.get(
            "/api/sessions/sess_aabb11223344/files/content?path=../../etc/passwd"
        )
        assert resp.status_code == 400

    def test_get_file_content_not_found(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.sandbox_manager.read_file = AsyncMock(
            side_effect=FileNotFoundError("File not found")
        )
        resp = client.get(
            "/api/sessions/sess_aabb11223344/files/content?path=nonexistent.ts"
        )
        assert resp.status_code == 404


# =========================================================================
# Download Files Archive
# =========================================================================


class TestDownloadFilesArchive:
    """GET /api/sessions/{session_id}/files/archive."""

    def test_download_archive_success(self, client: TestClient) -> None:
        resp = client.get("/api/sessions/sess_aabb11223344/files/archive")
        assert resp.status_code == 200
        assert resp.content == b"fake-tar-content"
        assert resp.headers["content-type"].startswith("application/x-tar")
        assert "attachment;" in resp.headers["content-disposition"]
        assert resp.headers["content-disposition"].endswith(".tar\"")

    def test_download_archive_session_not_found(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.get_session.return_value = None
        resp = client.get("/api/sessions/sess_000000000000/files/archive")
        assert resp.status_code == 404

    def test_download_archive_path_traversal_blocked(self, client: TestClient) -> None:
        resp = client.get("/api/sessions/sess_aabb11223344/files/archive?path=../etc")
        assert resp.status_code == 400
        assert "traversal" in resp.json()["detail"].lower()

    def test_download_archive_path_not_found(
        self, client: TestClient, mock_session_manager: MagicMock
    ) -> None:
        mock_session_manager.sandbox_manager.export_archive = AsyncMock(
            side_effect=FileNotFoundError("Path not found")
        )
        resp = client.get(
            "/api/sessions/sess_aabb11223344/files/archive?path=missing-dir"
        )
        assert resp.status_code == 404


# =========================================================================
# Session Manager not configured
# =========================================================================


class TestSessionManagerNotConfigured:
    """Routes should fail gracefully if SessionManager is not set."""

    def test_unconfigured_raises_500(self) -> None:
        app = FastAPI()
        app.include_router(router)
        # Deliberately DO NOT set session manager
        set_session_manager(None)  # type: ignore[arg-type]

        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.get("/api/sessions/sess_aabb11223344")
            # Should get 500 or a runtime error
            assert resp.status_code == 500
