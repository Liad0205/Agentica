"""Session manager for orchestrating AI coding sessions.

This module provides the SessionManager class that manages the lifecycle of
coding sessions, including sandbox creation, graph execution, and cleanup.

The SessionManager coordinates between:
- SandboxManager: For isolated code execution environments
- EventBus: For real-time event streaming to the frontend
- LangGraph agents: For AI-driven code generation

Usage:
    >>> from backend.sandbox import SandboxManager
    >>> from backend.events import get_event_bus
    >>> from backend.session_manager import SessionManager
    >>>
    >>> sandbox_manager = SandboxManager()
    >>> event_bus = get_event_bus()
    >>> session_manager = SessionManager(sandbox_manager, event_bus)
    >>>
    >>> # Create and start a session
    >>> session_id = await session_manager.create_session(
    ...     mode="react",
    ...     task="Build a todo app with React"
    ... )
    >>>
    >>> # Get session info
    >>> info = session_manager.get_session(session_id)
    >>> print(info.status)
    >>>
    >>> # Cleanup when done
    >>> await session_manager.cleanup_all()
"""

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import structlog

from agents.decomposition_graph import (
    DecompositionGraph,
    create_decomposition_graph,
    create_decomposition_initial_state,
)
from agents.hypothesis_graph import (
    HypothesisGraph,
    create_hypothesis_graph,
    create_hypothesis_initial_state,
)
from agents.react_graph import ReactGraph, create_initial_state, create_react_graph
from agents.utils import LLMClient
from config import settings
from events import EventBus
from events.types import AgentEvent, EventType
from metrics import MetricsCollector
from models.database import SessionStore
from models.schemas import ModelConfig, SessionMetrics, SessionStatus
from sandbox import SandboxManager

logger = structlog.get_logger()


# Type alias for agent mode
AgentMode = Literal["react", "decomposition", "hypothesis"]


@dataclass
class SessionInfo:
    """Information about a coding session.

    This dataclass holds all metadata and state for a session,
    tracking its lifecycle from creation to completion.

    Attributes:
        session_id: Unique identifier for the session (e.g., "sess_abc123def456")
        mode: The agent execution mode (react, decomposition, hypothesis)
        task: The user's original coding task/prompt
        status: Current session status (started, running, complete, error, cancelled)
        sandbox_id: The sandbox container identifier
        created_at: Unix timestamp when the session was created
        started_at: Unix timestamp when execution began (None if not started)
        completed_at: Unix timestamp when execution finished (None if not complete)
        error_message: Error message if status is "error" (None otherwise)
        model_config: Optional model configuration overrides
    """

    session_id: str
    mode: AgentMode
    task: str
    status: SessionStatus
    sandbox_id: str
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    error_message: str | None = None
    model_config: ModelConfig | None = None


class SessionManager:
    """Manages the lifecycle of AI coding sessions.

    The SessionManager is the central coordinator for coding sessions. It handles:
    - Session creation and ID generation
    - Sandbox provisioning
    - Graph execution in background tasks
    - Session cancellation
    - Resource cleanup

    Thread Safety:
        All operations use asyncio.Lock to ensure safe concurrent access
        to the session registry.

    Attributes:
        sandbox_manager: Manager for Docker sandbox containers
        event_bus: Event bus for real-time event streaming
    """

    def __init__(
        self,
        sandbox_manager: SandboxManager,
        event_bus: EventBus,
        session_store: SessionStore | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        """Initialize the SessionManager.

        Args:
            sandbox_manager: Manager for sandbox operations
            event_bus: Event bus for emitting events
            session_store: Optional SQLite store for session persistence
            metrics_collector: Optional collector for token/timing metrics
        """
        self.sandbox_manager = sandbox_manager
        self.event_bus = event_bus
        self.session_store = session_store
        self.metrics_collector = metrics_collector
        self._sessions: dict[str, SessionInfo] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        logger.info("session_manager_initialized")

    def _generate_session_id(self) -> str:
        """Generate a unique session identifier.

        Returns:
            A session ID in the format "sess_{12 hex chars}"
        """
        return f"sess_{uuid.uuid4().hex[:12]}"

    async def _persist_session_status(
        self,
        session_id: str,
        status: str,
        result_summary: str | None = None,
    ) -> None:
        """Persist a session status change to the database.

        This is a fire-and-forget helper. DB failures are logged but never
        propagated to the caller.

        Args:
            session_id: The session to update.
            status: New status value.
            result_summary: Optional result summary text.
        """
        if self.session_store is None:
            return
        try:
            await self.session_store.update_status(
                session_id, status, result_summary
            )
        except Exception as e:
            logger.error(
                "persist_session_status_failed",
                session_id=session_id,
                status=status,
                error=str(e),
            )

    def _create_llm_client(
        self,
        *,
        default_model: str | None = None,
    ) -> LLMClient:
        """Create an LLMClient configured with the metrics collector.

        Args:
            default_model: Optional default model override for the session.

        Returns:
            An LLMClient instance with event_bus and metrics_collector set.
        """
        return LLMClient(
            event_bus=self.event_bus,
            default_model=default_model,
            metrics_collector=self.metrics_collector,
        )

    def _resolve_model_config(
        self,
        model_config: ModelConfig | None,
    ) -> tuple[str | None, str, str, float]:
        """Resolve per-session model/temperature configuration.

        Returns:
            Tuple of:
            - sub_agent_model (or None for default)
            - orchestrator_model
            - evaluator_model
            - temperature
        """
        sub_agent_model: str | None = None
        orchestrator_model = settings.orchestrator_model
        evaluator_model = settings.evaluator_model
        temperature = 0.7

        if model_config is not None:
            if model_config.sub_agent:
                sub_agent_model = model_config.sub_agent
            if model_config.orchestrator:
                orchestrator_model = model_config.orchestrator
            if model_config.evaluator:
                evaluator_model = model_config.evaluator
            if model_config.temperature is not None:
                temperature = model_config.temperature

        return sub_agent_model, orchestrator_model, evaluator_model, temperature

    async def _finalize_session_metrics(self, session_id: str) -> None:
        """Finalize and persist metrics for a completed session.

        Retrieves the running totals from MetricsCollector, finalizes them
        (calculating duration), and saves to the database.

        Args:
            session_id: The session to finalize metrics for.
        """
        if self.metrics_collector is None:
            return
        try:
            final_metrics = self.metrics_collector.finish(session_id)
            if final_metrics is not None and self.session_store is not None:
                await self.session_store.save_metrics(
                    session_id, final_metrics.to_dict()
                )
        except Exception as e:
            logger.error(
                "finalize_session_metrics_failed",
                session_id=session_id,
                error=str(e),
            )

    def _build_session_coro(
        self,
        *,
        session_id: str,
        session_info: SessionInfo,
        model_config: ModelConfig | None,
    ) -> Any:
        """Build the mode-specific background coroutine for a session run."""
        (
            sub_agent_model,
            orchestrator_model,
            evaluator_model,
            temperature,
        ) = self._resolve_model_config(model_config)

        mode = session_info.mode

        if mode == "react":
            llm_client = self._create_llm_client(
                default_model=sub_agent_model
            )
            graph = create_react_graph(
                sandbox_manager=self.sandbox_manager,
                event_bus=self.event_bus,
                llm_client=llm_client,
                temperature=temperature,
            )
            initial_state = create_initial_state(
                task=session_info.task,
                sandbox_id=session_info.sandbox_id,
                session_id=session_id,
                agent_id=f"react_{session_id[5:]}",
                max_iterations=settings.max_react_iterations,
            )
            return self._run_react_session(
                session_id,
                session_info,
                graph=graph,
                initial_state=initial_state,
            )

        if mode == "decomposition":
            llm_client = self._create_llm_client(
                default_model=sub_agent_model
            )
            graph = create_decomposition_graph(
                sandbox_manager=self.sandbox_manager,
                event_bus=self.event_bus,
                llm_client=llm_client,
                orchestrator_model=orchestrator_model,
                sub_agent_temperature=temperature,
            )
            initial_state = create_decomposition_initial_state(
                task=session_info.task,
                session_id=session_id,
                integration_sandbox_id=session_info.sandbox_id,
            )
            return self._run_decomposition_session(
                session_id,
                session_info,
                graph=graph,
                initial_state=initial_state,
            )

        if mode == "hypothesis":
            llm_client = self._create_llm_client(
                default_model=sub_agent_model
            )
            graph = create_hypothesis_graph(
                sandbox_manager=self.sandbox_manager,
                event_bus=self.event_bus,
                llm_client=llm_client,
                evaluator_model=evaluator_model,
            )
            initial_state = create_hypothesis_initial_state(
                task=session_info.task,
                session_id=session_id,
                num_hypotheses=settings.num_hypothesis_agents,
            )
            return self._run_hypothesis_session(
                session_id,
                session_info,
                graph=graph,
                initial_state=initial_state,
            )

        raise ValueError(f"Unsupported mode: {mode}")

    async def _schedule_session_task(
        self,
        *,
        session_id: str,
        coro: Any,
    ) -> None:
        """Create and register the background task for a session run."""
        async with self._lock:
            background_task = asyncio.create_task(
                coro, name=f"session_{session_id}"
            )
            self._tasks[session_id] = background_task

            # Clean up task reference when it completes
            def _remove_task(
                t: asyncio.Task[None], sid: str = session_id
            ) -> None:
                self._tasks.pop(sid, None)

            background_task.add_done_callback(_remove_task)

    async def create_session(
        self,
        mode: AgentMode,
        task: str,
        model_config: ModelConfig | None = None,
    ) -> str:
        """Create and start a new coding session.

        This method:
        1. Generates a unique session ID
        2. Creates a sandbox container
        3. Initializes the appropriate graph (based on mode)
        4. Starts graph execution in a background task
        5. Emits SESSION_STARTED event

        Args:
            mode: The agent execution mode ("react", "decomposition", "hypothesis")
            task: The coding task/prompt from the user
            model_config: Optional model configuration overrides

        Returns:
            The unique session ID

        Raises:
            ValueError: If mode is not supported.
            RuntimeError: If sandbox creation fails
        """
        if mode not in ("react", "decomposition", "hypothesis"):
            raise ValueError(f"Unsupported mode: {mode}")

        # Generate unique session ID
        session_id = self._generate_session_id()
        sandbox_id = f"sandbox_{session_id[5:]}"  # Remove "sess_" prefix

        logger.info(
            "create_session_start",
            session_id=session_id,
            mode=mode,
            task_length=len(task),
        )

        # Create session info
        session_info = SessionInfo(
            session_id=session_id,
            mode=mode,
            task=task,
            status=SessionStatus.STARTED,
            sandbox_id=sandbox_id,
            created_at=time.time(),
            model_config=model_config,
        )

        # Register session
        async with self._lock:
            self._sessions[session_id] = session_info

        # Persist to database
        if self.session_store is not None:
            llm_config_dict = (
                model_config.model_dump() if model_config else None
            )
            try:
                await self.session_store.save_session(
                    session_id=session_id,
                    mode=mode,
                    task=task,
                    status=SessionStatus.STARTED.value,
                    llm_config=llm_config_dict,
                    created_at=session_info.created_at,
                )
            except Exception as e:
                logger.error(
                    "persist_session_create_failed",
                    session_id=session_id,
                    error=str(e),
                )

        # Start metrics tracking
        if self.metrics_collector is not None:
            self.metrics_collector.start(session_id)

        try:
            # Create sandbox (hypothesis mode creates per-solver sandboxes
            # instead, so skip the shared sandbox to avoid wasting a container)
            if mode != "hypothesis":
                await self.sandbox_manager.create_sandbox(sandbox_id)
            else:
                sandbox_id = "hypothesis_pending"
                session_info.sandbox_id = sandbox_id

            # Emit SESSION_STARTED event
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.SESSION_STARTED,
                    session_id=session_id,
                    data={
                        "mode": mode,
                        "task": task,
                        "sandbox_id": sandbox_id,
                    },
                )
            )

            # Start appropriate graph based on mode.
            coro = self._build_session_coro(
                session_id=session_id,
                session_info=session_info,
                model_config=model_config,
            )
            await self._schedule_session_task(
                session_id=session_id,
                coro=coro,
            )

            logger.info(
                "create_session_complete",
                session_id=session_id,
                sandbox_id=sandbox_id,
            )

            return session_id

        except Exception as e:
            # Clean up on failure
            logger.error(
                "create_session_failed",
                session_id=session_id,
                error=str(e),
            )

            # Update session status
            async with self._lock:
                if session_id in self._sessions:
                    self._sessions[session_id].status = SessionStatus.ERROR
                    self._sessions[session_id].error_message = str(e)

            # Emit error event
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.SESSION_ERROR,
                    session_id=session_id,
                    data={
                        "error": str(e),
                        "phase": "initialization",
                    },
                )
            )

            # Clean up sandbox if it was created
            with contextlib.suppress(KeyError):
                await self.sandbox_manager.destroy_sandbox(sandbox_id)

            raise

    async def continue_session(
        self,
        session_id: str,
        task: str,
        model_config: ModelConfig | None = None,
    ) -> str:
        """Continue a terminal session on its existing sandbox.

        The same session_id and sandbox are reused. The new task is appended
        as follow-up context so subsequent runs retain prior intent.

        Args:
            session_id: Existing session identifier.
            task: Follow-up task/prompt.
            model_config: Optional model overrides for this run.

        Returns:
            The same session_id.

        Raises:
            KeyError: If the session does not exist.
            RuntimeError: If the session is not in a continuable state or
                the sandbox is no longer available (e.g., after server restart).
        """
        effective_model_config: ModelConfig | None

        # First, check in-memory sessions (fast path)
        async with self._lock:
            session_info = self._sessions.get(session_id)

        # If not in memory, check if it exists in the database (stale session)
        if session_info is None:
            if self.session_store is not None:
                try:
                    persisted = await self.session_store.get_session(session_id)
                    if persisted is not None:
                        logger.warning(
                            "continue_session_stale",
                            session_id=session_id,
                            persisted_status=persisted.get("status"),
                        )
                        raise RuntimeError(
                            f"Session '{session_id}' exists in history but cannot be "
                            f"continued because the server was restarted and the "
                            f"sandbox environment is no longer available. "
                            f"Please start a new session."
                        )
                except RuntimeError:
                    raise
                except Exception as e:
                    logger.error(
                        "continue_session_db_lookup_failed",
                        session_id=session_id,
                        error=str(e),
                    )
            raise KeyError(f"Session '{session_id}' not found")

        # Re-acquire lock for mutation operations
        async with self._lock:
            # Re-check that session still exists (could have been cleaned up)
            if session_id not in self._sessions:
                raise KeyError(f"Session '{session_id}' was removed during processing")

            existing_task = self._tasks.get(session_id)
            if existing_task is not None and existing_task.done():
                self._tasks.pop(session_id, None)
                existing_task = None

            if existing_task is not None:
                raise RuntimeError(
                    f"Session '{session_id}' is currently running and cannot be continued"
                )

            if session_info.status in (SessionStatus.STARTED, SessionStatus.RUNNING):
                raise RuntimeError(
                    f"Session '{session_id}' is currently running and cannot be continued"
                )

            if session_info.status == SessionStatus.CANCELLED:
                raise RuntimeError(
                    f"Session '{session_id}' was cancelled and cannot be continued"
                )

            previous_task = session_info.task.strip()
            follow_up_task = task.strip()
            session_info.task = (
                f"{previous_task}\n\nFollow-up request:\n{follow_up_task}"
                if previous_task
                else follow_up_task
            )
            session_info.status = SessionStatus.STARTED
            session_info.started_at = None
            session_info.completed_at = None
            session_info.error_message = None

            if model_config is not None:
                session_info.model_config = model_config

            # For hypothesis mode, clean up the old winning sandbox and reset
            # to placeholder. The new graph run will create fresh solver sandboxes.
            if (
                session_info.mode == "hypothesis"
                and session_info.sandbox_id != "hypothesis_pending"
            ):
                old_sandbox = session_info.sandbox_id
                session_info.sandbox_id = "hypothesis_pending"
            else:
                old_sandbox = None

            effective_model_config = session_info.model_config

        # Destroy old hypothesis winning sandbox outside the lock
        if old_sandbox is not None:
            with contextlib.suppress(KeyError):
                await self.sandbox_manager.destroy_sandbox(old_sandbox)

        # Hypothesis sessions may leave stale per-solver sandboxes after
        # timeout/error exits; clean them before scheduling the next run.
        if session_info.mode == "hypothesis":
            await self._cleanup_hypothesis_solver_sandboxes(session_id)

        logger.info(
            "continue_session_start",
            session_id=session_id,
            mode=session_info.mode,
            task_length=len(task),
        )

        await self._persist_session_status(
            session_id, SessionStatus.STARTED.value
        )

        # Start metrics tracking for the follow-up run.
        if self.metrics_collector is not None:
            self.metrics_collector.start(session_id)

        await self.event_bus.publish(
            AgentEvent(
                type=EventType.SESSION_STARTED,
                session_id=session_id,
                data={
                    "mode": session_info.mode,
                    "task": task,
                    "sandbox_id": session_info.sandbox_id,
                    "continued": True,
                },
            )
        )

        coro = self._build_session_coro(
            session_id=session_id,
            session_info=session_info,
            model_config=effective_model_config,
        )
        await self._schedule_session_task(
            session_id=session_id,
            coro=coro,
        )

        logger.info(
            "continue_session_complete",
            session_id=session_id,
            sandbox_id=session_info.sandbox_id,
        )

        return session_id

    async def _run_react_session(
        self,
        session_id: str,
        session_info: SessionInfo,
        *,
        graph: ReactGraph,
        initial_state: dict[str, Any],
    ) -> None:
        """Run a ReAct mode session in the background.

        This method executes the ReAct graph and handles session lifecycle
        events (running, complete, error).

        Args:
            session_id: The session identifier
            session_info: The session information object
        """
        try:
            # Update status to running
            async with self._lock:
                session_info.status = SessionStatus.RUNNING
                session_info.started_at = time.time()

            # Run the graph with a session-level timeout
            final_state = await asyncio.wait_for(
                graph.run(initial_state),
                timeout=settings.agent_timeout_seconds,
            )

            # Update session status based on result
            async with self._lock:
                session_info.completed_at = time.time()
                if final_state.get("status") == "complete":
                    session_info.status = SessionStatus.COMPLETE
                elif final_state.get("status") == "failed":
                    session_info.status = SessionStatus.ERROR
                    session_info.error_message = "Agent execution failed"
                else:
                    # Unknown/intermediate status means graph exited abnormally
                    final_status = final_state.get("status", "unknown")
                    session_info.status = SessionStatus.ERROR
                    session_info.error_message = (
                        f"Graph exited with non-terminal status: {final_status}"
                    )

            preview_url: str | None = None
            if session_info.status == SessionStatus.COMPLETE:
                # Start preview server for the primary sandbox (with timeout)
                await self.event_bus.publish(
                    AgentEvent(
                        type=EventType.PREVIEW_STARTING,
                        session_id=session_id,
                        agent_id=initial_state["agent_id"],
                        agent_role="ReAct Agent",
                        data={"sandbox_id": session_info.sandbox_id},
                    )
                )

                try:
                    preview_url = await asyncio.wait_for(
                        self.sandbox_manager.start_dev_server(
                            session_info.sandbox_id
                        ),
                        timeout=30,
                    )
                    await self.event_bus.publish(
                        AgentEvent(
                            type=EventType.PREVIEW_READY,
                            session_id=session_id,
                            agent_id=initial_state["agent_id"],
                            agent_role="ReAct Agent",
                            data={
                                "url": preview_url,
                                "sandbox_id": session_info.sandbox_id,
                            },
                        )
                    )
                except Exception as e:
                    logger.error(
                        "react_preview_failed",
                        session_id=session_id,
                        error=str(e),
                    )
                    await self.event_bus.publish(
                        AgentEvent(
                            type=EventType.PREVIEW_ERROR,
                            session_id=session_id,
                            agent_id=initial_state["agent_id"],
                            agent_role="ReAct Agent",
                            data={
                                "error": str(e),
                                "sandbox_id": session_info.sandbox_id,
                            },
                        )
                    )

            # Emit SESSION_COMPLETE event
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.SESSION_COMPLETE,
                    session_id=session_id,
                    data={
                        "status": final_state.get("status", "unknown"),
                        "iterations": final_state.get("iteration", 0),
                        "files_written": final_state.get("files_written", []),
                        "preview_url": preview_url,
                    },
                )
            )

            logger.info(
                "react_session_complete",
                session_id=session_id,
                status=final_state.get("status"),
                iterations=final_state.get("iteration", 0),
            )

            # Persist final status and metrics to DB
            await self._persist_session_status(
                session_id, session_info.status.value
            )
            await self._finalize_session_metrics(session_id)

        except asyncio.CancelledError:
            # Session was cancelled
            logger.info(
                "react_session_cancelled",
                session_id=session_id,
            )

            async with self._lock:
                session_info.status = SessionStatus.CANCELLED
                session_info.completed_at = time.time()

            await self._persist_session_status(
                session_id, SessionStatus.CANCELLED.value
            )
            await self._finalize_session_metrics(session_id)

            raise

        except TimeoutError:
            # Session timed out
            logger.error(
                "react_session_timeout",
                session_id=session_id,
                timeout_seconds=settings.agent_timeout_seconds,
            )

            async with self._lock:
                session_info.status = SessionStatus.ERROR
                session_info.error_message = (
                    f"Session timed out after {settings.agent_timeout_seconds}s"
                )
                session_info.completed_at = time.time()

            await self._persist_session_status(
                session_id,
                SessionStatus.ERROR.value,
                result_summary=f"Timeout after {settings.agent_timeout_seconds}s",
            )
            await self._finalize_session_metrics(session_id)

            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.SESSION_ERROR,
                    session_id=session_id,
                    data={
                        "error": f"Session timed out after {settings.agent_timeout_seconds}s",
                        "phase": "timeout",
                    },
                )
            )

        except Exception as e:
            # Session errored
            logger.error(
                "react_session_error",
                session_id=session_id,
                error=str(e),
            )

            async with self._lock:
                session_info.status = SessionStatus.ERROR
                session_info.error_message = str(e)
                session_info.completed_at = time.time()

            # Persist error status and metrics to DB
            await self._persist_session_status(
                session_id, SessionStatus.ERROR.value, result_summary=str(e)
            )
            await self._finalize_session_metrics(session_id)

            # Emit SESSION_ERROR event
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.SESSION_ERROR,
                    session_id=session_id,
                    data={
                        "error": str(e),
                        "phase": "execution",
                    },
                )
            )

    async def _run_decomposition_session(
        self,
        session_id: str,
        session_info: SessionInfo,
        *,
        graph: DecompositionGraph,
        initial_state: dict[str, Any],
    ) -> None:
        """Run a Decomposition mode session in the background.

        This method executes the Decomposition graph (orchestrator -> sub-agents
        -> aggregator -> integration) and handles session lifecycle events.

        Args:
            session_id: The session identifier
            session_info: The session information object
        """
        try:
            # Update status to running
            async with self._lock:
                session_info.status = SessionStatus.RUNNING
                session_info.started_at = time.time()

            # Run the graph with a session-level timeout
            final_state = await asyncio.wait_for(
                graph.run(initial_state),
                timeout=settings.agent_timeout_seconds,
            )

            # Update session status based on result
            async with self._lock:
                session_info.completed_at = time.time()
                if final_state.get("status") == "complete":
                    session_info.status = SessionStatus.COMPLETE
                elif final_state.get("status") == "failed":
                    session_info.status = SessionStatus.ERROR
                    session_info.error_message = final_state.get(
                        "error_message", "Decomposition failed"
                    )
                else:
                    # Unknown/intermediate status means graph exited abnormally
                    final_status = final_state.get("status", "unknown")
                    session_info.status = SessionStatus.ERROR
                    session_info.error_message = (
                        f"Graph exited with non-terminal status: {final_status}"
                    )

            # Emit SESSION_COMPLETE event
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.SESSION_COMPLETE,
                    session_id=session_id,
                    data={
                        "status": final_state.get("status", "unknown"),
                        "files_merged": final_state.get("merged_files", []),
                        "failed_agents": final_state.get("failed_agents", []),
                        "preview_url": final_state.get("final_preview_url"),
                    },
                )
            )

            logger.info(
                "decomposition_session_complete",
                session_id=session_id,
                status=final_state.get("status"),
                files_merged=len(final_state.get("merged_files", [])),
            )

            # Persist final status and metrics to DB
            await self._persist_session_status(
                session_id, session_info.status.value
            )
            await self._finalize_session_metrics(session_id)

        except asyncio.CancelledError:
            # Session was cancelled
            logger.info(
                "decomposition_session_cancelled",
                session_id=session_id,
            )

            async with self._lock:
                session_info.status = SessionStatus.CANCELLED
                session_info.completed_at = time.time()

            await self._persist_session_status(
                session_id, SessionStatus.CANCELLED.value
            )
            await self._finalize_session_metrics(session_id)

            raise

        except TimeoutError:
            # Session timed out
            logger.error(
                "decomposition_session_timeout",
                session_id=session_id,
                timeout_seconds=settings.agent_timeout_seconds,
            )

            async with self._lock:
                session_info.status = SessionStatus.ERROR
                session_info.error_message = (
                    f"Session timed out after {settings.agent_timeout_seconds}s"
                )
                session_info.completed_at = time.time()

            await self._persist_session_status(
                session_id,
                SessionStatus.ERROR.value,
                result_summary=f"Timeout after {settings.agent_timeout_seconds}s",
            )
            await self._finalize_session_metrics(session_id)

            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.SESSION_ERROR,
                    session_id=session_id,
                    data={
                        "error": f"Session timed out after {settings.agent_timeout_seconds}s",
                        "phase": "timeout",
                    },
                )
            )

        except Exception as e:
            # Session errored
            logger.error(
                "decomposition_session_error",
                session_id=session_id,
                error=str(e),
            )

            async with self._lock:
                session_info.status = SessionStatus.ERROR
                session_info.error_message = str(e)
                session_info.completed_at = time.time()

            # Persist error status and metrics to DB
            await self._persist_session_status(
                session_id, SessionStatus.ERROR.value, result_summary=str(e)
            )
            await self._finalize_session_metrics(session_id)

            # Emit SESSION_ERROR event
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.SESSION_ERROR,
                    session_id=session_id,
                    data={
                        "error": str(e),
                        "phase": "execution",
                    },
                )
            )

    async def _run_hypothesis_session(
        self,
        session_id: str,
        session_info: SessionInfo,
        *,
        graph: HypothesisGraph,
        initial_state: dict[str, Any],
    ) -> None:
        """Run a Hypothesis mode session in the background.

        This method executes the Hypothesis graph (broadcast -> parallel solvers
        -> evaluator -> finalize) and handles session lifecycle events.

        Args:
            session_id: The session identifier
            session_info: The session information object
            graph: Pre-built hypothesis graph for this session
            initial_state: Initial graph state
        """
        try:
            should_cleanup_solver_sandboxes = False

            # Update status to running
            async with self._lock:
                session_info.status = SessionStatus.RUNNING
                session_info.started_at = time.time()

            # Run the graph with a session-level timeout
            final_state = await asyncio.wait_for(
                graph.run(initial_state),
                timeout=settings.agent_timeout_seconds,
            )

            # Update session status based on result
            async with self._lock:
                session_info.completed_at = time.time()
                if final_state.get("status") == "complete":
                    session_info.status = SessionStatus.COMPLETE
                elif final_state.get("status") == "failed":
                    session_info.status = SessionStatus.ERROR
                    session_info.error_message = final_state.get(
                        "error_message", "Hypothesis testing failed"
                    )
                    should_cleanup_solver_sandboxes = True
                else:
                    # Unknown/intermediate status means graph exited abnormally
                    final_status = final_state.get("status", "unknown")
                    session_info.status = SessionStatus.ERROR
                    session_info.error_message = (
                        f"Graph exited with non-terminal status: {final_status}"
                    )
                    should_cleanup_solver_sandboxes = True

                # Point session at the winning solver's sandbox so the
                # file API endpoints (/files, /files/content) work.
                winning_sandbox = final_state.get("final_sandbox_id")
                if winning_sandbox:
                    session_info.sandbox_id = winning_sandbox

            if should_cleanup_solver_sandboxes:
                await self._cleanup_hypothesis_solver_sandboxes(session_id)

            # Emit SESSION_COMPLETE event
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.SESSION_COMPLETE,
                    session_id=session_id,
                    data={
                        "status": final_state.get("status", "unknown"),
                        "selected_agent": final_state.get("selected_agent_id"),
                        "evaluation_reasoning": final_state.get("evaluation_reasoning"),
                        "preview_url": final_state.get("final_preview_url"),
                    },
                )
            )

            logger.info(
                "hypothesis_session_complete",
                session_id=session_id,
                status=final_state.get("status"),
                selected_agent=final_state.get("selected_agent_id"),
            )

            # Persist final status and metrics to DB
            await self._persist_session_status(
                session_id, session_info.status.value
            )
            await self._finalize_session_metrics(session_id)

        except asyncio.CancelledError:
            # Session was cancelled
            logger.info(
                "hypothesis_session_cancelled",
                session_id=session_id,
            )

            async with self._lock:
                session_info.status = SessionStatus.CANCELLED
                session_info.completed_at = time.time()

            await self._persist_session_status(
                session_id, SessionStatus.CANCELLED.value
            )
            await self._finalize_session_metrics(session_id)

            raise

        except TimeoutError:
            # Session timed out
            logger.error(
                "hypothesis_session_timeout",
                session_id=session_id,
                timeout_seconds=settings.agent_timeout_seconds,
            )

            async with self._lock:
                session_info.status = SessionStatus.ERROR
                session_info.error_message = (
                    f"Session timed out after {settings.agent_timeout_seconds}s"
                )
                session_info.completed_at = time.time()

            await self._persist_session_status(
                session_id,
                SessionStatus.ERROR.value,
                result_summary=f"Timeout after {settings.agent_timeout_seconds}s",
            )
            await self._finalize_session_metrics(session_id)

            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.SESSION_ERROR,
                    session_id=session_id,
                    data={
                        "error": f"Session timed out after {settings.agent_timeout_seconds}s",
                        "phase": "timeout",
                    },
                )
            )

            await self._cleanup_hypothesis_solver_sandboxes(session_id)

        except Exception as e:
            # Session errored
            logger.error(
                "hypothesis_session_error",
                session_id=session_id,
                error=str(e),
            )

            async with self._lock:
                session_info.status = SessionStatus.ERROR
                session_info.error_message = str(e)
                session_info.completed_at = time.time()

            # Persist error status and metrics to DB
            await self._persist_session_status(
                session_id, SessionStatus.ERROR.value, result_summary=str(e)
            )
            await self._finalize_session_metrics(session_id)

            # Emit SESSION_ERROR event
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.SESSION_ERROR,
                    session_id=session_id,
                    data={
                        "error": str(e),
                        "phase": "execution",
                    },
                )
            )

            await self._cleanup_hypothesis_solver_sandboxes(session_id)

    async def _cleanup_hypothesis_solver_sandboxes(self, session_id: str) -> None:
        """Destroy per-solver sandboxes for a hypothesis session."""
        suffix = session_id[5:]  # Remove "sess_" prefix
        for i in range(settings.num_hypothesis_agents):
            solver_sandbox_id = f"sandbox_solver{i + 1}_{suffix}"
            try:
                await self.sandbox_manager.destroy_sandbox(solver_sandbox_id)
            except KeyError:
                pass  # Solver may not have started yet
            except Exception as e:
                logger.warning(
                    "cleanup_hypothesis_solver_sandbox_failed",
                    session_id=session_id,
                    sandbox_id=solver_sandbox_id,
                    error=str(e),
                )

    async def cancel_session(self, session_id: str) -> None:
        """Cancel a running session.

        This method cancels the background task running the session
        and cleans up associated resources (sandbox, event bus session).

        Args:
            session_id: The session to cancel

        Raises:
            KeyError: If session doesn't exist
        """
        terminal_status: SessionStatus | None = None
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session '{session_id}' not found")

            session_info = self._sessions[session_id]

            if session_info.status == SessionStatus.COMPLETE:
                logger.info(
                    "cancel_session_noop_terminal_state",
                    session_id=session_id,
                    status=session_info.status.value,
                )
                return

            if session_info.status in (SessionStatus.ERROR, SessionStatus.CANCELLED):
                terminal_status = session_info.status

        # Session is already terminal but may still have resources attached
        # (e.g. graph errored before cleanup). Perform idempotent cleanup.
        if terminal_status is not None:
            if session_info.sandbox_id != "hypothesis_pending":
                try:
                    await self.sandbox_manager.destroy_sandbox(session_info.sandbox_id)
                except KeyError:
                    logger.warning(
                        "cancel_session_terminal_cleanup_sandbox_not_found",
                        session_id=session_id,
                        sandbox_id=session_info.sandbox_id,
                    )

            # For hypothesis mode, also clean up solver sandboxes
            if session_info.mode == "hypothesis":
                await self._cleanup_hypothesis_solver_sandboxes(session_id)

            await self.event_bus.close_session(session_id)
            logger.info(
                "cancel_session_terminal_cleanup_complete",
                session_id=session_id,
                status=terminal_status.value,
            )
            return

        logger.info(
            "cancel_session_start",
            session_id=session_id,
            current_status=session_info.status.value,
        )

        # Extract task reference under the lock, then cancel outside to
        # avoid deadlock (the task's CancelledError handler also acquires _lock).
        task: asyncio.Task[None] | None = None
        async with self._lock:
            if session_id in self._tasks:
                task = self._tasks.pop(session_id)

        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Update session status (the CancelledError handler may have already
        # set this, but we ensure it's set in case the task was already done).
        async with self._lock:
            if session_info.status not in (SessionStatus.CANCELLED,):
                session_info.status = SessionStatus.CANCELLED
                session_info.completed_at = time.time()

        # Persist cancel status to DB
        await self._persist_session_status(
            session_id, SessionStatus.CANCELLED.value
        )
        await self._finalize_session_metrics(session_id)

        # Emit SESSION_CANCELLED before closing the event bus session so
        # connected clients can update UI state immediately.
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.SESSION_CANCELLED,
                session_id=session_id,
                data={
                    "reason": "user_cancelled",
                    "status": SessionStatus.CANCELLED.value,
                },
            )
        )

        # Destroy the sandbox
        if session_info.sandbox_id != "hypothesis_pending":
            try:
                await self.sandbox_manager.destroy_sandbox(session_info.sandbox_id)
            except KeyError:
                logger.warning(
                    "cancel_session_sandbox_not_found",
                    session_id=session_id,
                    sandbox_id=session_info.sandbox_id,
                )

        # For hypothesis mode, also destroy per-solver sandboxes which are
        # not tracked by session_info.sandbox_id.
        if session_info.mode == "hypothesis":
            await self._cleanup_hypothesis_solver_sandboxes(session_id)

        # Close event bus session
        await self.event_bus.close_session(session_id)

        logger.info(
            "cancel_session_complete",
            session_id=session_id,
        )

    def get_session(self, session_id: str) -> SessionInfo | None:
        """Get information about a session.

        Args:
            session_id: The session to retrieve

        Returns:
            SessionInfo if the session exists, otherwise None.
        """
        return self._sessions.get(session_id)

    def get_session_metrics(self, session_id: str) -> SessionMetrics | None:
        """Get execution metrics for a session.

        Combines timing data from the in-memory SessionInfo with live token
        counts from MetricsCollector (for active sessions).

        Args:
            session_id: The session to retrieve metrics for.

        Returns:
            SessionMetrics if the session exists, otherwise None.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None

        if session.started_at is None:
            execution_time_seconds = 0.0
        else:
            end_time = session.completed_at or time.time()
            execution_time_seconds = max(0.0, end_time - session.started_at)

        # Populate token/tool metrics from the collector if available
        total_input_tokens = 0
        total_output_tokens = 0
        total_llm_calls = 0
        total_tool_calls = 0

        if self.metrics_collector is not None:
            live_data = self.metrics_collector.get(session_id)
            if live_data is not None:
                total_input_tokens = live_data.prompt_tokens
                total_output_tokens = live_data.completion_tokens
                total_llm_calls = live_data.llm_calls
                total_tool_calls = live_data.tool_calls

        return SessionMetrics(
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_llm_calls=total_llm_calls,
            total_tool_calls=total_tool_calls,
            execution_time_seconds=execution_time_seconds,
        )

    def get_all_sessions(self) -> list[SessionInfo]:
        """Get information about all sessions.

        Returns:
            List of SessionInfo objects for all sessions
        """
        return list(self._sessions.values())

    async def run_sandbox_health_checks(self) -> None:
        """Check the health of all active sandboxes for running sessions.

        For each session that is currently RUNNING, checks whether its
        sandbox container is healthy. If a sandbox is unhealthy, emits
        a SESSION_ERROR event so the frontend can display a warning.

        This method is intended to be called periodically from a
        background task (see ``start_health_check_loop``).
        """
        running_sessions: list[SessionInfo] = []
        async with self._lock:
            running_sessions = [
                s for s in self._sessions.values()
                if s.status == SessionStatus.RUNNING
            ]

        for session in running_sessions:
            # Hypothesis mode uses per-solver sandboxes; the main sandbox_id
            # is a placeholder until the winner is chosen.
            if session.sandbox_id == "hypothesis_pending":
                continue

            try:
                healthy = await self.sandbox_manager.is_healthy(session.sandbox_id)
                if not healthy:
                    logger.warning(
                        "sandbox_unhealthy",
                        session_id=session.session_id,
                        sandbox_id=session.sandbox_id,
                    )
                    await self.event_bus.publish(
                        AgentEvent(
                            type=EventType.SESSION_ERROR,
                            session_id=session.session_id,
                            data={
                                "error": f"Sandbox {session.sandbox_id} is unhealthy",
                                "phase": "health_check",
                                "sandbox_id": session.sandbox_id,
                            },
                        )
                    )
            except Exception as e:
                logger.error(
                    "sandbox_health_check_failed",
                    session_id=session.session_id,
                    sandbox_id=session.sandbox_id,
                    error=str(e),
                )

    async def start_health_check_loop(
        self, interval_seconds: float = 30.0
    ) -> asyncio.Task[None]:
        """Start a background task that periodically checks sandbox health.

        The task runs until cancelled (typically at application shutdown).

        Args:
            interval_seconds: Seconds between health check rounds (default: 30).

        Returns:
            The background asyncio.Task that can be cancelled on shutdown.
        """

        async def _loop() -> None:
            logger.info(
                "health_check_loop_started",
                interval_seconds=interval_seconds,
            )
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    await self.run_sandbox_health_checks()
                except asyncio.CancelledError:
                    logger.info("health_check_loop_stopped")
                    return
                except Exception as e:
                    logger.error(
                        "health_check_loop_error",
                        error=str(e),
                    )

        task = asyncio.create_task(_loop(), name="sandbox_health_checks")
        return task

    async def cleanup_all(self) -> None:
        """Clean up all sessions and their resources.

        This method should be called during application shutdown to ensure
        all background tasks are cancelled and sandboxes are destroyed.
        """
        logger.info(
            "cleanup_all_start",
            session_count=len(self._sessions),
        )

        # Get all session IDs
        session_ids = list(self._sessions.keys())

        # Collect all tasks under the lock, then cancel outside to avoid
        # deadlock (task CancelledError handlers also acquire _lock).
        tasks_to_cancel: list[tuple[str, asyncio.Task[None]]] = []
        async with self._lock:
            for session_id in session_ids:
                if session_id in self._tasks:
                    tasks_to_cancel.append((session_id, self._tasks[session_id]))
            self._tasks.clear()

        # Cancel all tasks outside the lock
        for session_id, task in tasks_to_cancel:
            if not task.done():
                task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception) as e:
                if not isinstance(e, asyncio.CancelledError):
                    logger.error(
                        "cleanup_task_cancel_failed",
                        session_id=session_id,
                        error=str(e),
                    )

        # Clean up all sandboxes
        await self.sandbox_manager.cleanup_all()

        # Close event streams for all sessions so WebSocket subscribers can exit.
        for session_id in session_ids:
            try:
                await self.event_bus.close_session(session_id)
            except Exception as e:
                logger.warning(
                    "cleanup_close_session_failed",
                    session_id=session_id,
                    error=str(e),
                )
            # Clear event history since session is fully destroyed
            self.event_bus.clear_event_history(session_id)

        # Clear sessions
        async with self._lock:
            self._sessions.clear()

        logger.info("cleanup_all_complete")
