"""Token bucket rate limiter for LLM API calls.

This module provides rate limiting to prevent exceeding LLM provider quotas.
It implements a sliding window approach for both requests-per-minute (RPM)
and tokens-per-minute (TPM) limits.

Usage:
    >>> from rate_limiter import get_rate_limiter
    >>> limiter = get_rate_limiter()
    >>> await limiter.acquire(estimated_tokens=1500)
    >>> # ... make LLM call ...
    >>> limiter.record_usage(actual_tokens=1234)
"""

import asyncio
import time
from collections import deque

import structlog

from config import settings

logger = structlog.get_logger(__name__)


class RateLimitExceededError(Exception):
    """Raised when the rate limiter wait deadline is exceeded."""


class RateLimiter:
    """Token bucket rate limiter for LLM API calls.

    Enforces two independent limits:
    - Requests per minute (RPM): Maximum number of API calls within a 60s window.
    - Tokens per minute (TPM): Maximum estimated/actual tokens within a 60s window.

    The limiter uses a sliding window approach: it tracks timestamps and token
    counts for recent calls, and only considers entries within the last 60 seconds.

    When a limit would be exceeded, ``acquire()`` waits until enough capacity
    is available.

    Attributes:
        max_calls_per_minute: Maximum API calls allowed per 60-second window.
        max_tokens_per_minute: Maximum tokens allowed per 60-second window.
    """

    def __init__(
        self,
        max_calls_per_minute: int = 30,
        max_tokens_per_minute: int = 100_000,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            max_calls_per_minute: Maximum API calls per minute (RPM).
            max_tokens_per_minute: Maximum tokens per minute (TPM).
        """
        self.max_calls_per_minute = max_calls_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute

        # Sliding window tracking: deque of (timestamp, token_count)
        self._call_log: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

        logger.info(
            "rate_limiter_initialized",
            max_rpm=max_calls_per_minute,
            max_tpm=max_tokens_per_minute,
        )

    def _prune_old_entries(self, now: float) -> None:
        """Remove entries older than 60 seconds from the sliding window.

        Args:
            now: The current time as a Unix timestamp.
        """
        cutoff = now - 60.0
        while self._call_log and self._call_log[0][0] < cutoff:
            self._call_log.popleft()

    def _current_rpm(self) -> int:
        """Return the number of calls in the current 60s window."""
        return len(self._call_log)

    def _current_tpm(self) -> int:
        """Return the total tokens consumed in the current 60s window."""
        return sum(tokens for _, tokens in self._call_log)

    async def acquire(
        self,
        estimated_tokens: int = 1000,
        max_wait_seconds: float = 120.0,
    ) -> None:
        """Wait until the rate limit allows a new request, then reserve a slot.

        This method blocks (via ``asyncio.sleep``) if either the RPM or TPM
        limit would be exceeded. Once there is capacity, it records the request
        in the sliding window so subsequent calls see the reservation.

        Args:
            estimated_tokens: Estimated tokens for the upcoming request.
                Used for TPM budgeting before the actual usage is known.
            max_wait_seconds: Maximum time to wait before raising
                ``RateLimitExceededError`` (default: 120s).

        Raises:
            RateLimitExceededError: If the deadline is exceeded.
        """
        deadline = time.monotonic() + max_wait_seconds

        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune_old_entries(now)

                rpm_ok = self._current_rpm() < self.max_calls_per_minute
                tpm_ok = (
                    self._current_tpm() + estimated_tokens
                    <= self.max_tokens_per_minute
                )

                if rpm_ok and tpm_ok:
                    # Reserve the slot
                    self._call_log.append((now, estimated_tokens))
                    logger.debug(
                        "rate_limiter_acquired",
                        current_rpm=self._current_rpm(),
                        current_tpm=self._current_tpm(),
                        estimated_tokens=estimated_tokens,
                    )
                    return

                # Check deadline before waiting
                if now >= deadline:
                    raise RateLimitExceededError(
                        f"Rate limiter wait exceeded {max_wait_seconds}s deadline"
                    )

                # Calculate how long to wait (don't exceed deadline)
                wait_seconds = self._calculate_wait(now, estimated_tokens)
                wait_seconds = min(wait_seconds, deadline - now)

            logger.info(
                "rate_limiter_waiting",
                wait_seconds=round(wait_seconds, 2),
                current_rpm=self._current_rpm(),
                current_tpm=self._current_tpm(),
            )
            await asyncio.sleep(wait_seconds)

    def _calculate_wait(self, now: float, estimated_tokens: int) -> float:
        """Calculate how long to wait before the next request can proceed.

        Args:
            now: Current monotonic time.
            estimated_tokens: Estimated tokens for the next request.

        Returns:
            Seconds to wait (at least 0.1s to avoid busy-spinning).
        """
        if not self._call_log:
            return 0.1

        # Find the earliest entry that, once expired, would free capacity
        oldest_timestamp = self._call_log[0][0]
        wait = (oldest_timestamp + 60.0) - now

        # Ensure we wait at least a small amount to avoid busy loops
        return max(wait, 0.1)

    def record_usage(self, tokens_used: int) -> None:
        """Record the actual token usage after a completed LLM call.

        Adjusts the most recent entry in the sliding window from the
        estimated token count to the actual count. This keeps TPM tracking
        accurate over time.

        Args:
            tokens_used: The actual number of tokens consumed by the call.
        """
        if not self._call_log:
            return

        # Replace the most recent entry's estimated tokens with actual usage
        timestamp, _estimated = self._call_log[-1]
        self._call_log[-1] = (timestamp, tokens_used)

        logger.debug(
            "rate_limiter_usage_recorded",
            tokens_used=tokens_used,
            current_tpm=self._current_tpm(),
        )

    def get_status(self) -> dict[str, object]:
        """Return current rate limiter status for diagnostics.

        Returns:
            Dictionary with current RPM, TPM, and configured limits.
        """
        now = time.monotonic()
        self._prune_old_entries(now)
        return {
            "current_rpm": self._current_rpm(),
            "current_tpm": self._current_tpm(),
            "max_rpm": self.max_calls_per_minute,
            "max_tpm": self.max_tokens_per_minute,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get the global RateLimiter singleton.

    The singleton is created on first call using values from ``config.settings``.

    Returns:
        The global RateLimiter instance.
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            max_calls_per_minute=settings.llm_rate_limit_rpm,
            max_tokens_per_minute=settings.llm_rate_limit_tpm,
        )
    return _rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global RateLimiter singleton.

    Primarily useful for testing.
    """
    global _rate_limiter
    _rate_limiter = None
