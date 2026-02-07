"""WebSocket handler for real-time event streaming.

This module handles WebSocket connections for streaming agent events to the frontend
and receiving commands (like cancel) from clients.
"""

import asyncio
import contextlib
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from events import AgentEvent, EventType, get_event_bus

if TYPE_CHECKING:
    from session_manager import SessionManager

logger = structlog.get_logger(__name__)

websocket_router = APIRouter()

_session_manager: "SessionManager | None" = None


def set_session_manager(manager: "SessionManager") -> None:
    """Set the session manager used by WebSocket command handlers."""
    global _session_manager
    _session_manager = manager
    logger.info("websocket_session_manager_configured")


def get_session_manager() -> "SessionManager":
    """Return configured session manager for WebSocket command handlers."""
    if _session_manager is None:
        raise RuntimeError(
            "SessionManager not configured for WebSocket handlers. "
            "Call set_session_manager() during startup."
        )
    return _session_manager


@websocket_router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time event streaming.

    This endpoint handles bidirectional communication:
    - Server -> Client: Agent events (thinking, tool calls, file changes, etc.)
    - Client -> Server: Commands (cancel, etc.)

    Args:
        websocket: The WebSocket connection.
        session_id: The session ID to stream events for.
    """
    await websocket.accept()

    logger.info("websocket_connected", session_id=session_id)

    # Get the global event bus and subscribe to session events
    event_bus = get_event_bus()

    # IMPORTANT: Subscribe FIRST to start capturing live events, then replay
    # history. This prevents a race condition where events published between
    # get_event_history() and subscribe() would be missed or duplicated.
    queue = event_bus.subscribe(session_id)

    try:
        # Replay historical events so reconnecting clients (e.g. after page
        # refresh) see the full session history.
        last_replay_timestamp: float = 0.0
        history = event_bus.get_event_history(session_id)
        if history:
            logger.info(
                "replaying_event_history",
                session_id=session_id,
                event_count=len(history),
            )
            for event in history:
                try:
                    await websocket.send_json(event.model_dump(mode="json"))
                    last_replay_timestamp = event.timestamp
                except WebSocketDisconnect:
                    logger.info("websocket_disconnect_during_replay", session_id=session_id)
                    return
                except Exception as e:
                    logger.error("websocket_replay_error", session_id=session_id, error=str(e))
                    return

        async def send_events() -> None:
            """Forward events from the event bus to the WebSocket client.

            Skip events that were already replayed from history to prevent
            duplicates. Events with timestamp <= last_replay_timestamp are
            considered duplicates from the subscription buffer.
            """
            try:
                while True:
                    event = await queue.get()
                    # SESSION_CLOSED is a sentinel from close_session; stop sending.
                    if event.type == EventType.SESSION_CLOSED:
                        logger.info("session_closed_sentinel", session_id=session_id)
                        break

                    # Skip events that were already replayed from history.
                    # Events published between get_event_history() and subscribe()
                    # will appear in both history and the subscription queue's buffer.
                    if event.timestamp <= last_replay_timestamp:
                        logger.debug(
                            "event_skipped_duplicate",
                            session_id=session_id,
                            event_type=event.type.value,
                            event_timestamp=event.timestamp,
                            last_replay_timestamp=last_replay_timestamp,
                        )
                        continue

                    await websocket.send_json(event.model_dump(mode="json"))
                    logger.debug(
                        "event_sent",
                        session_id=session_id,
                        event_type=event.type.value,
                    )
            except WebSocketDisconnect:
                logger.info("websocket_disconnect_during_send", session_id=session_id)
            except Exception as e:
                logger.error("websocket_send_error", session_id=session_id, error=str(e))

        async def receive_commands() -> None:
            """Receive and process commands from the WebSocket client."""
            try:
                while True:
                    data = await websocket.receive_json()
                    if not isinstance(data, dict):
                        logger.warning("invalid_ws_message", session_id=session_id)
                        continue
                    command_type = data.get("type")

                    logger.info(
                        "command_received",
                        session_id=session_id,
                        command_type=command_type,
                    )

                    if command_type == "cancel":
                        # Handle cancel command
                        await handle_cancel_command(session_id)
                    elif command_type == "ping":
                        # Respond to ping with pong
                        await websocket.send_json(
                            {"type": "pong", "timestamp": data.get("timestamp")}
                        )
                    else:
                        logger.warning(
                            "unknown_command",
                            session_id=session_id,
                            command_type=command_type,
                        )
            except WebSocketDisconnect:
                logger.info("websocket_disconnect_during_receive", session_id=session_id)
            except Exception as e:
                logger.error("websocket_receive_error", session_id=session_id, error=str(e))

        # Run both tasks concurrently
        send_task = asyncio.create_task(send_events())
        receive_task = asyncio.create_task(receive_commands())

        # Wait for either task to complete (usually due to disconnect)
        done, pending = await asyncio.wait(
            [send_task, receive_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel any pending tasks
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    except WebSocketDisconnect:
        logger.info("websocket_disconnected", session_id=session_id)
    except Exception as e:
        logger.error("websocket_error", session_id=session_id, error=str(e))
    finally:
        # Clean up subscription
        event_bus.unsubscribe(session_id, queue)
        logger.info("websocket_cleanup_complete", session_id=session_id)


async def handle_cancel_command(session_id: str) -> None:
    """Handle a cancel command from the WebSocket client.

    Args:
        session_id: The session to cancel.
    """
    logger.info("cancel_command_processing", session_id=session_id)

    session_manager = get_session_manager()
    event_bus = get_event_bus()

    session = session_manager.get_session(session_id)
    if session is None:
        logger.warning("cancel_command_session_not_found", session_id=session_id)
        await event_bus.publish(
            AgentEvent(
                type=EventType.SESSION_ERROR,
                session_id=session_id,
                data={
                    "error": f"Session {session_id} not found",
                    "phase": "cancellation",
                },
            )
        )
        return

    try:
        await session_manager.cancel_session(session_id)
    except Exception as e:
        logger.error("cancel_command_failed", session_id=session_id, error=str(e))
        await event_bus.publish(
            AgentEvent(
                type=EventType.SESSION_ERROR,
                session_id=session_id,
                data={
                    "error": str(e),
                    "phase": "cancellation",
                },
            )
        )
