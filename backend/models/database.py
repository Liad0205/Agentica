"""SQLite-based session persistence using aiosqlite.

This module provides the SessionStore class for persisting session data
and metrics to a SQLite database. All operations are async and designed
to fail gracefully -- a database error should never crash a running session.

Tables:
    sessions: Core session metadata (id, mode, task, status, timestamps, etc.)
    session_metrics: Aggregate token usage and timing data per session.

Usage:
    >>> from models.database import SessionStore
    >>> store = SessionStore("./data/sessions.db")
    >>> await store.init()
    >>> await store.save_session(
    ...     session_id="sess_abc123",
    ...     mode="react",
    ...     task="Build a todo app",
    ...     status="started",
    ... )
"""

import json
import time
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

logger = structlog.get_logger(__name__)


class SessionStore:
    """Async SQLite store for session persistence and metrics.

    The SessionStore manages a single SQLite database file and provides
    methods for CRUD operations on sessions and their metrics. All public
    methods catch exceptions internally and log errors rather than
    propagating them, ensuring that database issues never break active
    session execution.

    Attributes:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        """Initialize the session store.

        Args:
            db_path: Filesystem path to the SQLite database file.
                     Parent directories are created automatically on init().
        """
        self.db_path = db_path

    async def init(self) -> None:
        """Create database tables if they do not exist.

        Also creates parent directories for the database file if needed.
        """
        # Ensure parent directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        mode TEXT NOT NULL,
                        task TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'started',
                        llm_config TEXT,
                        result_summary TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                """)
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS session_metrics (
                        session_id TEXT PRIMARY KEY,
                        total_tokens INTEGER NOT NULL DEFAULT 0,
                        prompt_tokens INTEGER NOT NULL DEFAULT 0,
                        completion_tokens INTEGER NOT NULL DEFAULT 0,
                        llm_calls INTEGER NOT NULL DEFAULT 0,
                        tool_calls INTEGER NOT NULL DEFAULT 0,
                        duration_ms INTEGER NOT NULL DEFAULT 0,
                        created_at REAL NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions(id)
                    )
                """)
                # Index for listing sessions by creation time
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_created_at
                    ON sessions(created_at DESC)
                """)
                await db.commit()
            logger.info("session_store_initialized", db_path=self.db_path)
        except Exception as e:
            logger.error(
                "session_store_init_failed",
                db_path=self.db_path,
                error=str(e),
            )
            raise

    # -----------------------------------------------------------------
    # Session CRUD
    # -----------------------------------------------------------------

    async def save_session(
        self,
        session_id: str,
        mode: str,
        task: str,
        status: str,
        llm_config: dict[str, Any] | None = None,
        created_at: float | None = None,
        updated_at: float | None = None,
    ) -> None:
        """Insert a new session record.

        Args:
            session_id: Unique session identifier (e.g. "sess_abc123").
            mode: Agent execution mode (react, decomposition, hypothesis).
            task: The user's coding task/prompt.
            status: Initial status string.
            llm_config: Optional model configuration dict (stored as JSON).
            created_at: Unix timestamp of creation (defaults to now).
            updated_at: Unix timestamp of last update (defaults to now).
        """
        now = time.time()
        created_at = created_at or now
        updated_at = updated_at or now
        llm_config_json = json.dumps(llm_config) if llm_config else None

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO sessions
                        (id, mode, task, status, llm_config, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, mode, task, status, llm_config_json, created_at, updated_at),
                )
                await db.commit()
            logger.debug(
                "session_saved",
                session_id=session_id,
                mode=mode,
                status=status,
            )
        except Exception as e:
            logger.error(
                "session_save_failed",
                session_id=session_id,
                error=str(e),
            )

    async def update_status(
        self,
        session_id: str,
        status: str,
        result_summary: str | None = None,
    ) -> None:
        """Update the status (and optionally a result summary) for a session.

        Args:
            session_id: The session to update.
            status: New status string (e.g. "complete", "error", "cancelled").
            result_summary: Optional human-readable summary of the outcome.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    UPDATE sessions
                    SET status = ?, result_summary = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, result_summary, time.time(), session_id),
                )
                await db.commit()
            logger.debug(
                "session_status_updated",
                session_id=session_id,
                status=status,
            )
        except Exception as e:
            logger.error(
                "session_status_update_failed",
                session_id=session_id,
                error=str(e),
            )

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve a single session by its ID.

        Args:
            session_id: The session to look up.

        Returns:
            A dict with session fields, or None if not found.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM sessions WHERE id = ?",
                    (session_id,),
                )
                row = await cursor.fetchone()
                if row is None:
                    return None

                result = dict(row)
                # Parse llm_config JSON back to dict
                if result.get("llm_config"):
                    try:
                        result["llm_config"] = json.loads(result["llm_config"])
                    except json.JSONDecodeError:
                        result["llm_config"] = None
                return result
        except Exception as e:
            logger.error(
                "session_get_failed",
                session_id=session_id,
                error=str(e),
            )
            return None

    async def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List recent sessions ordered by creation time (newest first).

        Args:
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip.

        Returns:
            List of session dicts.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    """
                    SELECT * FROM sessions
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )
                rows = await cursor.fetchall()
                sessions: list[dict[str, Any]] = []
                for row in rows:
                    session = dict(row)
                    if session.get("llm_config"):
                        try:
                            session["llm_config"] = json.loads(session["llm_config"])
                        except json.JSONDecodeError:
                            session["llm_config"] = None
                    sessions.append(session)
                return sessions
        except Exception as e:
            logger.error(
                "session_list_failed",
                error=str(e),
            )
            return []

    async def clear_all(self) -> int:
        """Delete all persisted sessions and metrics.

        Returns:
            Number of session rows removed from the ``sessions`` table.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM sessions")
                count_row = await cursor.fetchone()
                deleted_count = int(count_row[0]) if count_row else 0

                await db.execute("DELETE FROM session_metrics")
                await db.execute("DELETE FROM sessions")
                await db.commit()

            logger.info("session_store_cleared", deleted_count=deleted_count)
            return deleted_count
        except Exception as e:
            logger.error(
                "session_store_clear_failed",
                error=str(e),
            )
            return 0

    # -----------------------------------------------------------------
    # Metrics CRUD
    # -----------------------------------------------------------------

    async def save_metrics(
        self,
        session_id: str,
        metrics_data: dict[str, Any],
    ) -> None:
        """Save or replace aggregate metrics for a session.

        Uses INSERT OR REPLACE so callers can safely call this multiple
        times (e.g. periodic snapshots or final save).

        Args:
            session_id: The session the metrics belong to.
            metrics_data: Dict with keys: total_tokens, prompt_tokens,
                          completion_tokens, llm_calls, tool_calls,
                          duration_ms.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO session_metrics
                        (session_id, total_tokens, prompt_tokens, completion_tokens,
                         llm_calls, tool_calls, duration_ms, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        metrics_data.get("total_tokens", 0),
                        metrics_data.get("prompt_tokens", 0),
                        metrics_data.get("completion_tokens", 0),
                        metrics_data.get("llm_calls", 0),
                        metrics_data.get("tool_calls", 0),
                        metrics_data.get("duration_ms", 0),
                        time.time(),
                    ),
                )
                await db.commit()
            logger.debug(
                "session_metrics_saved",
                session_id=session_id,
            )
        except Exception as e:
            logger.error(
                "session_metrics_save_failed",
                session_id=session_id,
                error=str(e),
            )

    async def get_metrics(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve aggregate metrics for a session.

        Args:
            session_id: The session to look up.

        Returns:
            A dict with metric fields, or None if not found.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM session_metrics WHERE session_id = ?",
                    (session_id,),
                )
                row = await cursor.fetchone()
                if row is None:
                    return None
                return dict(row)
        except Exception as e:
            logger.error(
                "session_metrics_get_failed",
                session_id=session_id,
                error=str(e),
            )
            return None
