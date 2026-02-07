"""HTTP API routes for the Agent Swarm backend.

This module defines all HTTP endpoints for session management, file access,
and health checks. Real-time events are handled via WebSocket in websocket.py.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Annotated

import structlog
from fastapi import APIRouter, Body, HTTPException, Path, Query, status
from fastapi.responses import PlainTextResponse, Response

from models.schemas import (
    ContinueSessionRequest,
    CreateSessionRequest,
    FileInfo,
    HealthResponse,
    SessionDetailResponse,
    SessionMetrics,
    SessionResponse,
    SessionStatus,
    SessionSummaryResponse,
)

if TYPE_CHECKING:
    from session_manager import SessionManager

logger = structlog.get_logger(__name__)

router = APIRouter()

_TERMINAL_SESSION_STATUSES = {
    SessionStatus.COMPLETE,
    SessionStatus.ERROR,
    SessionStatus.CANCELLED,
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _resolve_sandbox_id(
    *,
    session_id: str,
    session_sandbox_id: str,
    requested_sandbox_id: str,
) -> str:
    """Resolve a sandbox id while preventing cross-session access."""
    if requested_sandbox_id in ("", "primary"):
        return session_sandbox_id

    suffix = f"_{session_id[5:]}"  # strip "sess_"
    if not requested_sandbox_id.endswith(suffix):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid sandbox_id for this session",
        )

    return requested_sandbox_id


def _to_int(value: object) -> int:
    """Safely coerce a value to int for metric conversion."""
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _to_float(value: object) -> float:
    """Safely coerce a value to float for timestamp conversion."""
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _to_status(raw_status: object) -> SessionStatus:
    """Convert an untrusted status value into SessionStatus."""
    if isinstance(raw_status, str):
        try:
            return SessionStatus(raw_status)
        except ValueError:
            logger.warning("invalid_persisted_status", status=raw_status)
    return SessionStatus.STARTED


def _to_string(value: object, *, default: str = "") -> str:
    """Safely coerce value to string while avoiding 'None' artifacts."""
    if isinstance(value, str):
        return value
    return default


def _to_optional_string(value: object) -> str | None:
    """Return non-empty strings as-is, otherwise None."""
    if isinstance(value, str) and value:
        return value
    return None


def _to_metrics(raw_metrics: dict[str, object] | None) -> SessionMetrics:
    """Convert persisted DB metrics row to API metrics schema."""
    if raw_metrics is None:
        return SessionMetrics()

    duration_ms = _to_float(raw_metrics.get("duration_ms"))
    return SessionMetrics(
        total_input_tokens=_to_int(raw_metrics.get("prompt_tokens")),
        total_output_tokens=_to_int(raw_metrics.get("completion_tokens")),
        total_llm_calls=_to_int(raw_metrics.get("llm_calls")),
        total_tool_calls=_to_int(raw_metrics.get("tool_calls")),
        execution_time_seconds=max(0.0, duration_ms / 1000.0),
    )


# Session manager dependency (set during application startup)
_session_manager: SessionManager | None = None


def set_session_manager(manager: SessionManager) -> None:
    """Set the session manager instance for the routes.

    This should be called during application startup to inject the session
    manager dependency.

    Args:
        manager: The SessionManager instance to use for all routes.
    """
    global _session_manager
    _session_manager = manager
    logger.info("session_manager_configured")


def get_session_manager() -> SessionManager:
    """Get the session manager instance.

    Returns:
        The configured SessionManager instance.

    Raises:
        RuntimeError: If the session manager has not been configured.
    """
    if _session_manager is None:
        logger.error("session_manager_not_configured")
        raise RuntimeError(
            "SessionManager not configured. Call set_session_manager() during startup."
        )
    return _session_manager


@router.post(
    "/api/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new session",
    description="Create a new agent session with the specified mode and task.",
)
async def create_session(request: CreateSessionRequest) -> SessionResponse:
    """Create a new session and start agent execution.

    This endpoint creates a new session, initializes the appropriate agent graph
    based on the selected mode, and returns connection details for WebSocket streaming.

    Args:
        request: The session creation request containing mode, task, and optional config.

    Returns:
        SessionResponse with session_id, websocket_url, and initial status.

    Raises:
        HTTPException: If session creation fails.
    """
    session_manager = get_session_manager()

    try:
        session_id = await session_manager.create_session(
            mode=request.mode,
            task=request.task,
            model_config=request.model_config_override,
        )

        # Get the session info to return status
        session = session_manager.get_session(session_id)
        session_status = session.status if session else SessionStatus.STARTED

        logger.info(
            "session_created",
            session_id=session_id,
            mode=request.mode,
            task_length=len(request.task),
        )

        return SessionResponse(
            session_id=session_id,
            websocket_url=f"/ws/{session_id}",
            status=session_status,
        )
    except Exception as e:
        logger.error(
            "session_creation_failed",
            error=str(e),
            mode=request.mode,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {e}",
        ) from e


@router.post(
    "/api/sessions/{session_id}/continue",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Continue an existing session",
    description=(
        "Continue a terminal session (complete/error) on the same sandbox "
        "with a follow-up task."
    ),
)
async def continue_session(
    session_id: Annotated[str, Path(description="The session ID")],
    request: ContinueSessionRequest,
) -> SessionResponse:
    """Continue an existing session with a follow-up task."""
    session_manager = get_session_manager()

    try:
        continued_session_id = await session_manager.continue_session(
            session_id=session_id,
            task=request.task,
            model_config=request.model_config_override,
        )
    except RuntimeError as e:
        error_message = str(e)
        # Stale session (exists in DB but sandbox is gone) gets 410 Gone
        if "server was restarted" in error_message:
            logger.warning(
                "continue_session_stale",
                session_id=session_id,
                error=error_message,
            )
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail=error_message,
            ) from e
        # Other RuntimeErrors (invalid state) get 400 Bad Request
        logger.warning(
            "continue_session_invalid_state",
            session_id=session_id,
            error=error_message,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_message,
        ) from e
    except KeyError:
        logger.warning("continue_session_not_found_after_lookup", session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        ) from None
    except Exception as e:
        logger.error(
            "continue_session_failed",
            session_id=session_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to continue session: {e}",
        ) from e

    updated = session_manager.get_session(continued_session_id)
    session_status = updated.status if updated else SessionStatus.STARTED

    logger.info(
        "session_continued",
        session_id=continued_session_id,
        mode=updated.mode if updated else "unknown",
        task_length=len(request.task),
    )

    return SessionResponse(
        session_id=continued_session_id,
        websocket_url=f"/ws/{continued_session_id}",
        status=session_status,
    )


@router.get(
    "/api/sessions",
    response_model=list[SessionSummaryResponse],
    summary="List sessions",
    description="List recent sessions with lifecycle metadata and metrics.",
)
async def list_sessions(
    limit: Annotated[int, Query(description="Maximum sessions to return", ge=1, le=200)] = 25,
) -> list[SessionSummaryResponse]:
    """List sessions with summary details and aggregate metrics."""
    session_manager = get_session_manager()
    sessions = sorted(
        session_manager.get_all_sessions(),
        key=lambda s: s.created_at,
        reverse=True,
    )[:limit]

    response: list[SessionSummaryResponse] = []
    for session in sessions:
        metrics = session_manager.get_session_metrics(session.session_id) or SessionMetrics()
        response.append(
            SessionSummaryResponse(
                session_id=session.session_id,
                mode=session.mode,
                task=session.task,
                status=session.status,
                created_at=session.created_at,
                started_at=session.started_at,
                completed_at=session.completed_at,
                error_message=session.error_message,
                metrics=metrics,
            )
        )

    if len(response) >= limit:
        return response

    # Fallback to persisted sessions to keep history after process restarts.
    session_store = session_manager.session_store
    if session_store is None:
        return response

    persisted_sessions = await session_store.list_sessions(limit=limit)
    existing_session_ids = {item.session_id for item in response}

    for persisted in persisted_sessions:
        session_id = persisted.get("id")
        if not isinstance(session_id, str) or session_id in existing_session_ids:
            continue

        status_value = _to_status(persisted.get("status"))
        created_at = _to_float(persisted.get("created_at"))
        updated_at = _to_float(persisted.get("updated_at"))
        metrics = _to_metrics(await session_store.get_metrics(session_id))

        response.append(
            SessionSummaryResponse(
                session_id=session_id,
                mode=_to_string(persisted.get("mode"), default="react"),
                task=_to_string(persisted.get("task")),
                status=status_value,
                created_at=created_at,
                started_at=created_at if status_value != SessionStatus.STARTED else None,
                completed_at=updated_at if status_value in _TERMINAL_SESSION_STATUSES else None,
                error_message=(
                    _to_optional_string(persisted.get("result_summary"))
                    if status_value == SessionStatus.ERROR
                    else None
                ),
                metrics=metrics,
            )
        )
        existing_session_ids.add(session_id)

        if len(response) >= limit:
            break

    return response


@router.delete(
    "/api/sessions",
    status_code=status.HTTP_200_OK,
    summary="Clear all sessions",
    description=(
        "Clear all in-memory and persisted sessions. "
        "Set force=true to also clear currently running sessions."
    ),
)
async def clear_sessions(
    force: Annotated[bool, Query(description="Also clear running sessions")] = False,
) -> dict[str, object]:
    """Clear all sessions from memory and persistence."""
    session_manager = get_session_manager()
    sessions = session_manager.get_all_sessions()
    running_session_ids = [
        session.session_id
        for session in sessions
        if session.status in (SessionStatus.STARTED, SessionStatus.RUNNING)
    ]

    if running_session_ids and not force:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Running sessions exist. Retry with force=true to clear all sessions."
            ),
        )

    try:
        await session_manager.cleanup_all()
    except Exception as e:
        logger.error("clear_sessions_cleanup_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear sessions: {e}",
        ) from e

    persisted_deleted = 0
    session_store = session_manager.session_store
    if session_store is not None:
        persisted_deleted = await session_store.clear_all()

    logger.info(
        "sessions_cleared",
        in_memory_cleared=len(sessions),
        running_cleared=len(running_session_ids),
        persisted_cleared=persisted_deleted,
        forced=force,
    )

    return {
        "message": "All sessions cleared",
        "in_memory_cleared": len(sessions),
        "running_cleared": len(running_session_ids),
        "persisted_cleared": persisted_deleted,
        "forced": force,
    }


@router.get(
    "/api/sessions/{session_id}",
    response_model=SessionDetailResponse,
    summary="Get session details",
    description="Get current session state including status and configuration.",
)
async def get_session(
    session_id: Annotated[str, Path(description="The session ID")]
) -> SessionDetailResponse:
    """Get current session state including all events so far.

    Args:
        session_id: The unique session identifier.

    Returns:
        SessionDetailResponse with full session details.

    Raises:
        HTTPException: If session is not found.
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if session is not None:
        logger.debug("session_retrieved", session_id=session_id)

        return SessionDetailResponse(
            session_id=session.session_id,
            mode=session.mode,
            task=session.task,
            status=session.status,
            created_at=session.created_at,
            started_at=session.started_at,
            completed_at=session.completed_at,
            error_message=session.error_message,
            model_config_data=session.model_config,
        )

    session_store = session_manager.session_store
    if session_store is None:
        logger.warning("session_not_found", session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    persisted = await session_store.get_session(session_id)
    if persisted is None:
        logger.warning("session_not_found", session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    status_value = _to_status(persisted.get("status"))
    created_at = _to_float(persisted.get("created_at"))
    updated_at = _to_float(persisted.get("updated_at"))
    llm_config = persisted.get("llm_config")

    return SessionDetailResponse(
        session_id=session_id,
        mode=_to_string(persisted.get("mode"), default="react"),
        task=_to_string(persisted.get("task")),
        status=status_value,
        created_at=created_at,
        started_at=created_at if status_value != SessionStatus.STARTED else None,
        completed_at=updated_at if status_value in _TERMINAL_SESSION_STATUSES else None,
        error_message=(
            _to_optional_string(persisted.get("result_summary"))
            if status_value == SessionStatus.ERROR
            else None
        ),
        model_config_data=llm_config if isinstance(llm_config, dict) else None,
    )


@router.post(
    "/api/sessions/{session_id}/cancel",
    status_code=status.HTTP_200_OK,
    summary="Cancel a running session",
    description="Cancel a running session and clean up resources.",
)
async def cancel_session(
    session_id: Annotated[str, Path(description="The session ID")]
) -> dict[str, str]:
    """Cancel a running session.

    Stops all agent execution, cleans up sandbox containers, and marks
    the session as cancelled.

    Args:
        session_id: The unique session identifier.

    Returns:
        Confirmation message.

    Raises:
        HTTPException: If session is not found or cannot be cancelled.
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if session is None:
        logger.warning("cancel_session_not_found", session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    if session.status in (
        SessionStatus.COMPLETE,
        SessionStatus.ERROR,
        SessionStatus.CANCELLED,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session {session_id} is already {session.status}",
        )

    await session_manager.cancel_session(session_id)
    logger.info("session_cancelled", session_id=session_id)

    return {"message": f"Session {session_id} cancelled"}


@router.get(
    "/api/sessions/{session_id}/files",
    response_model=list[FileInfo],
    summary="Get file tree",
    description="Get the current file tree for a sandbox in the session.",
)
async def get_files(
    session_id: Annotated[str, Path(description="The session ID")],
    sandbox_id: Annotated[str, Query(description="Sandbox ID to query")] = "primary",
    path: Annotated[str, Query(description="Directory path to list")] = ".",
) -> list[FileInfo]:
    """Get current file tree for a sandbox.

    Returns a list of files and directories in the specified path within
    the sandbox's workspace.

    Args:
        session_id: The unique session identifier.
        sandbox_id: The sandbox to query (default: "primary").
        path: The directory path to list (default: ".").

    Returns:
        List of FileInfo objects representing files and directories.

    Raises:
        HTTPException: If session or sandbox is not found.
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if session is None:
        logger.warning("get_files_session_not_found", session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Basic traversal guard (SandboxManager also validates)
    if ".." in path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal is not allowed",
        )

    logger.debug(
        "files_requested",
        session_id=session_id,
        sandbox_id=sandbox_id,
        path=path,
    )

    resolved_sandbox_id = _resolve_sandbox_id(
        session_id=session_id,
        session_sandbox_id=session.sandbox_id,
        requested_sandbox_id=sandbox_id,
    )

    if resolved_sandbox_id == "hypothesis_pending":
        return []

    try:
        entries = await session_manager.sandbox_manager.list_files_recursive(
            resolved_sandbox_id,
            path=path,
        )
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sandbox {sandbox_id} not found",
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(
            "list_files_failed",
            session_id=session_id,
            sandbox_id=resolved_sandbox_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list files: {e}",
        ) from e

    return [
        FileInfo(
            name=e.name,
            path=e.path,
            is_directory=e.is_directory,
            size=None if e.is_directory else e.size,
            modified_at=None,
        )
        for e in entries
    ]


@router.get(
    "/api/sessions/{session_id}/files/content",
    response_class=PlainTextResponse,
    summary="Get file content",
    description="Get the content of a specific file in a sandbox.",
)
async def get_file_content(
    session_id: Annotated[str, Path(description="The session ID")],
    path: Annotated[str, Query(description="File path relative to workspace")],
    sandbox_id: Annotated[str, Query(description="Sandbox ID to query")] = "primary",
) -> str:
    """Get content of a specific file.

    Args:
        session_id: The unique session identifier.
        path: The file path relative to the workspace.
        sandbox_id: The sandbox to query (default: "primary").

    Returns:
        The file content as plain text.

    Raises:
        HTTPException: If session, sandbox, or file is not found.
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if session is None:
        logger.warning("get_file_session_not_found", session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Validate path for security (no path traversal)
    if ".." in path:
        logger.warning(
            "path_traversal_blocked",
            session_id=session_id,
            path=path,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal is not allowed",
        )

    logger.debug(
        "file_content_requested",
        session_id=session_id,
        sandbox_id=sandbox_id,
        path=path,
    )

    resolved_sandbox_id = _resolve_sandbox_id(
        session_id=session_id,
        session_sandbox_id=session.sandbox_id,
        requested_sandbox_id=sandbox_id,
    )

    if resolved_sandbox_id == "hypothesis_pending":
        raise HTTPException(
            status_code=409,
            detail=(
                "Hypothesis solvers are still working."
                " File content will be available after evaluation completes."
            )
        )

    try:
        return await session_manager.sandbox_manager.read_file(
            resolved_sandbox_id,
            path,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sandbox {sandbox_id} not found",
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(
            "read_file_failed",
            session_id=session_id,
            sandbox_id=resolved_sandbox_id,
            path=path,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read file: {e}",
        ) from e


@router.put(
    "/api/sessions/{session_id}/files/content",
    status_code=status.HTTP_200_OK,
    summary="Write file content",
    description="Write content to a file in a sandbox workspace.",
)
async def put_file_content(
    session_id: Annotated[str, Path(description="The session ID")],
    path: Annotated[str, Query(description="File path relative to workspace")],
    content: Annotated[str, Body(media_type="text/plain")],
    sandbox_id: Annotated[str, Query(description="Sandbox ID to target")] = "primary",
) -> dict[str, str]:
    """Write content to a file in a sandbox.

    Args:
        session_id: The unique session identifier.
        path: The file path relative to the workspace.
        content: The file content to write (plain text body).
        sandbox_id: The sandbox to target (default: "primary").

    Returns:
        Status message confirming the write.

    Raises:
        HTTPException: If session or sandbox is not found, or path is invalid.
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    if ".." in path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal is not allowed",
        )

    resolved_sandbox_id = _resolve_sandbox_id(
        session_id=session_id,
        session_sandbox_id=session.sandbox_id,
        requested_sandbox_id=sandbox_id,
    )

    try:
        await session_manager.sandbox_manager.write_file(
            resolved_sandbox_id,
            path,
            content,
        )
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sandbox {sandbox_id} not found",
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(
            "write_file_failed",
            session_id=session_id,
            sandbox_id=resolved_sandbox_id,
            path=path,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write file: {e}",
        ) from e

    return {"message": "File saved", "path": path}


@router.get(
    "/api/sessions/{session_id}/files/archive",
    response_class=Response,
    summary="Download sandbox files archive",
    description="Download a tar archive of files from a sandbox workspace path.",
)
async def download_files_archive(
    session_id: Annotated[str, Path(description="The session ID")],
    sandbox_id: Annotated[str, Query(description="Sandbox ID to query")] = "primary",
    path: Annotated[str, Query(description="Directory path to archive")] = ".",
) -> Response:
    """Download a tar archive of sandbox files.

    Args:
        session_id: The unique session identifier.
        sandbox_id: The sandbox to query (default: "primary").
        path: Directory path relative to workspace (default: ".").

    Returns:
        Tar archive bytes as an attachment response.

    Raises:
        HTTPException: If session, sandbox, or requested path is invalid.
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if session is None:
        logger.warning("download_archive_session_not_found", session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    # Basic traversal guard (SandboxManager also validates).
    if ".." in path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal is not allowed",
        )

    resolved_sandbox_id = _resolve_sandbox_id(
        session_id=session_id,
        session_sandbox_id=session.sandbox_id,
        requested_sandbox_id=sandbox_id,
    )

    try:
        archive_bytes = await session_manager.sandbox_manager.export_archive(
            resolved_sandbox_id,
            path,
        )
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sandbox {sandbox_id} not found",
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(
            "download_archive_failed",
            session_id=session_id,
            sandbox_id=resolved_sandbox_id,
            path=path,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export archive: {e}",
        ) from e

    # Determine filename
    filename = f"{session_id}_files.tar"
    if path != ".":
        filename = f"{path.strip('/').replace('/', '_')}.tar"

    return Response(
        content=archive_bytes,
        media_type="application/x-tar",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


@router.post(
    "/api/sessions/{session_id}/sandbox/dev-server",
    status_code=status.HTTP_200_OK,
    summary="Control dev server",
    description="Start or stop the dev server in a sandbox.",
)
async def manage_dev_server(
    session_id: Annotated[str, Path(description="The session ID")],
    action: Annotated[str, Query(description="Action to perform: 'start' or 'stop'")],
    sandbox_id: Annotated[str, Query(description="Sandbox ID to target")] = "primary",
) -> dict[str, object]:
    """Manually start or stop the dev server.

    Args:
        session_id: The unique session identifier.
        action: 'start' or 'stop'.
        sandbox_id: The sandbox to target (default: "primary").

    Returns:
        Status message and details.

    Raises:
        HTTPException: If session/sandbox not found or action failed.
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    resolved_sandbox_id = _resolve_sandbox_id(
        session_id=session_id,
        session_sandbox_id=session.sandbox_id,
        requested_sandbox_id=sandbox_id,
    )

    if action == "start":
        try:
            url = await session_manager.sandbox_manager.start_dev_server(
                resolved_sandbox_id
            )
            return {"message": "Dev server started", "url": url, "status": "running"}
        except Exception as e:
            logger.error("dev_server_start_failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start dev server: {e}",
            ) from e

    elif action == "stop":
        try:
            stopped = await session_manager.sandbox_manager.stop_dev_server(
                resolved_sandbox_id
            )
            msg = "Dev server stopped" if stopped else "Dev server was not running"
            return {"message": msg, "status": "stopped"}
        except Exception as e:
            logger.error("dev_server_stop_failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to stop dev server: {e}",
            ) from e

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action: {action}. Must be 'start' or 'stop'.",
        )


@router.get(
    "/api/sessions/{session_id}/metrics",
    response_model=SessionMetrics,
    summary="Get session metrics",
    description="Get token usage and execution metrics for a session.",
)
async def get_session_metrics(
    session_id: Annotated[str, Path(description="The session ID")]
) -> SessionMetrics:
    """Get token usage and execution metrics for a session.

    Returns aggregate metrics including total tokens used, LLM calls,
    tool invocations, and execution time.

    Args:
        session_id: The unique session identifier.

    Returns:
        SessionMetrics with aggregate statistics.

    Raises:
        HTTPException: If session is not found.
    """
    session_manager = get_session_manager()
    metrics = session_manager.get_session_metrics(session_id)

    if metrics is not None:
        logger.debug("metrics_retrieved", session_id=session_id)
        return metrics

    session_store = session_manager.session_store
    if session_store is not None:
        persisted = await session_store.get_session(session_id)
        if persisted is not None:
            persisted_metrics = await session_store.get_metrics(session_id)
            if persisted_metrics is not None:
                logger.debug("metrics_retrieved_from_store", session_id=session_id)
                return _to_metrics(persisted_metrics)
            # Session exists but no metrics persisted yet.
            logger.debug("metrics_default_for_persisted_session", session_id=session_id)
            return SessionMetrics()

    logger.warning("get_metrics_session_not_found", session_id=session_id)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Session {session_id} not found",
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Health check endpoint with Docker and sandbox status.",
)
async def health_check() -> HealthResponse:
    """Health check endpoint with infrastructure status.

    Returns Docker daemon connectivity and active sandbox count in addition
    to the basic health status and timestamp.

    Returns:
        HealthResponse with status, Docker availability, and sandbox count.
    """
    docker_available = False
    active_sandboxes = 0

    try:
        session_manager = get_session_manager()
        sandbox_mgr = session_manager.sandbox_manager
        docker_available = sandbox_mgr.is_docker_available()
        active_sandboxes = sandbox_mgr.get_active_sandbox_count()
    except RuntimeError:
        # SessionManager not configured yet (e.g., during startup)
        pass
    except Exception as e:
        logger.warning("health_check_partial_failure", error=str(e))

    overall_status: str = "healthy" if docker_available else "unhealthy"

    return HealthResponse(
        status=overall_status,
        timestamp=time.time(),
        docker_available=docker_available,
        active_sandboxes=active_sandboxes,
    )
