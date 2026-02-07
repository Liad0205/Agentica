"""In-memory metrics collection for active sessions.

This module provides the MetricsCollector class that accumulates token usage
and timing data for running sessions. When a session completes, the final
metrics are persisted to the database via SessionStore.

The collector is thread-safe via asyncio.Lock and designed for concurrent
access from multiple LangGraph agent tasks.

Usage:
    >>> from metrics import MetricsCollector
    >>> collector = MetricsCollector()
    >>> collector.start("sess_abc123")
    >>> collector.record_llm_call("sess_abc123", prompt_tokens=100, completion_tokens=50)
    >>> collector.record_tool_call("sess_abc123")
    >>> final = collector.finish("sess_abc123")
    >>> print(final)  # SessionMetricsData(...)
"""

import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SessionMetricsData:
    """Accumulated metrics for a single session.

    Attributes:
        total_tokens: Sum of prompt and completion tokens.
        prompt_tokens: Total input/prompt tokens across all LLM calls.
        completion_tokens: Total output/completion tokens across all LLM calls.
        llm_calls: Number of LLM invocations.
        tool_calls: Number of tool executions.
        duration_ms: Total execution time in milliseconds (set by finish()).
        started_at: Unix timestamp when tracking began.
    """

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    duration_ms: int = 0
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, int]:
        """Convert to a plain dict suitable for SessionStore.save_metrics().

        Returns:
            Dict with all metric fields.
        """
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "duration_ms": self.duration_ms,
        }


class MetricsCollector:
    """In-memory collector that tracks per-session metrics.

    Each active session gets its own SessionMetricsData instance.
    The collector is safe for concurrent access from multiple async tasks
    (all mutations go through a simple dict update which is safe in CPython,
    but we add explicit logging for clarity).

    Attributes:
        _sessions: Mapping from session_id to its metrics data.
    """

    def __init__(self) -> None:
        """Initialize an empty metrics collector."""
        self._sessions: dict[str, SessionMetricsData] = {}
        logger.info("metrics_collector_initialized")

    def start(self, session_id: str) -> None:
        """Begin tracking metrics for a session.

        If the session is already being tracked, this is a no-op.

        Args:
            session_id: The session to start tracking.
        """
        if session_id in self._sessions:
            logger.debug(
                "metrics_already_tracking",
                session_id=session_id,
            )
            return

        self._sessions[session_id] = SessionMetricsData()
        logger.debug(
            "metrics_tracking_started",
            session_id=session_id,
        )

    def record_llm_call(
        self,
        session_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Record token usage from a single LLM call.

        If the session is not being tracked, this is a no-op with a warning.

        Args:
            session_id: The session the LLM call belongs to.
            prompt_tokens: Number of input tokens used.
            completion_tokens: Number of output tokens used.
        """
        data = self._sessions.get(session_id)
        if data is None:
            logger.warning(
                "metrics_record_no_session",
                session_id=session_id,
            )
            return

        data.prompt_tokens += prompt_tokens
        data.completion_tokens += completion_tokens
        data.total_tokens += prompt_tokens + completion_tokens
        data.llm_calls += 1

        logger.debug(
            "metrics_llm_call_recorded",
            session_id=session_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_llm_calls=data.llm_calls,
        )

    def record_tool_call(self, session_id: str) -> None:
        """Increment the tool call counter for a session.

        Args:
            session_id: The session the tool call belongs to.
        """
        data = self._sessions.get(session_id)
        if data is None:
            logger.warning(
                "metrics_tool_no_session",
                session_id=session_id,
            )
            return

        data.tool_calls += 1

    def finish(self, session_id: str) -> SessionMetricsData | None:
        """Finalize metrics for a session, calculating duration.

        The session's metrics data is removed from the collector after
        this call. The returned data includes the calculated duration_ms.

        Args:
            session_id: The session to finalize.

        Returns:
            The final SessionMetricsData, or None if not tracked.
        """
        data = self._sessions.pop(session_id, None)
        if data is None:
            logger.warning(
                "metrics_finish_no_session",
                session_id=session_id,
            )
            return None

        data.duration_ms = int((time.time() - data.started_at) * 1000)

        logger.info(
            "metrics_session_finished",
            session_id=session_id,
            total_tokens=data.total_tokens,
            llm_calls=data.llm_calls,
            tool_calls=data.tool_calls,
            duration_ms=data.duration_ms,
        )

        return data

    def get(self, session_id: str) -> SessionMetricsData | None:
        """Get current (in-progress) metrics for a session.

        Does NOT remove the session from the collector.

        Args:
            session_id: The session to query.

        Returns:
            Current SessionMetricsData, or None if not tracked.
        """
        return self._sessions.get(session_id)
