"""LLM client utilities and helper functions for agent execution.

This module provides:
- LLMClient: Wrapper around LiteLLM with retry logic, rate limiting, fallback
  model support, and metrics tracking
- rate_limited_completion: Wrapper around litellm.acompletion with rate limiting
- call_llm: Convenience function for simple LLM calls
- parse_plan_tag: Extract plan content from <plan> tags in responses
- parse_tool_calls: Parse tool calls from LLM responses
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import structlog
from litellm import ModelResponse, acompletion
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from config import settings
from events.bus import EventBus
from events.types import AgentEvent, EventType, LLMMetrics
from rate_limiter import RateLimiter, RateLimitExceededError, get_rate_limiter

# Import ToolCall from tools to avoid circular import at runtime
# but allow type checking to see the type
if TYPE_CHECKING:
    from metrics import MetricsCollector

logger = structlog.get_logger()


def _convert_message_to_dict(message: Any) -> dict[str, Any]:
    """Convert a LangChain message object to a plain dict.

    LangGraph's add_messages reducer converts plain dicts to LangChain
    message objects (SystemMessage, HumanMessage, etc.). LiteLLM expects
    plain dicts, so we need to convert them back.

    Args:
        message: Either a dict or a LangChain message object

    Returns:
        A plain dict with 'role' and 'content' keys
    """
    # If already a dict, return as-is
    if isinstance(message, dict):
        return message

    # Handle LangChain message objects
    # They have 'type' or 'role' attribute and 'content' attribute
    msg_dict: dict[str, Any] = {}

    # Get role from type or role attribute
    if hasattr(message, "type"):
        # LangChain uses 'type' for role (e.g., 'system', 'human', 'ai')
        role = message.type
        # Map LangChain types to OpenAI roles
        role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
        msg_dict["role"] = role_map.get(role, role)
    elif hasattr(message, "role"):
        msg_dict["role"] = message.role
    else:
        logger.warning("message_missing_role", message_type=type(message).__name__)
        msg_dict["role"] = "user"

    # Get content
    if hasattr(message, "content"):
        msg_dict["content"] = message.content
    else:
        msg_dict["content"] = str(message)

    # Handle tool calls for assistant messages
    if hasattr(message, "tool_calls") and message.tool_calls:
        msg_dict["tool_calls"] = [
            {
                "id": tc.get("id", tc.get("name", "")),
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": (
                        json.dumps(tc.get("args", {}))
                        if isinstance(tc.get("args"), dict)
                        else tc.get("args", "{}")
                    ),
                },
            }
            for tc in message.tool_calls
        ]

    # Handle tool message specifics
    if hasattr(message, "tool_call_id"):
        msg_dict["tool_call_id"] = message.tool_call_id

    return msg_dict


def _convert_messages_to_dicts(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert a list of messages to plain dicts for LiteLLM.

    Args:
        messages: List of dicts or LangChain message objects

    Returns:
        List of plain dicts
    """
    return [_convert_message_to_dict(msg) for msg in messages]


def normalize_tool_args(raw_args: Any) -> dict[str, Any]:
    """Normalize raw tool-call arguments into a dictionary.

    Models occasionally emit malformed tool arguments (JSON arrays, primitives,
    or partially valid strings). This helper guarantees downstream tool
    execution always receives a dict-like payload.
    """
    if isinstance(raw_args, dict):
        return raw_args

    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {"raw": raw_args}
        return parsed if isinstance(parsed, dict) else {"value": parsed}

    if raw_args is None:
        return {}

    return {"value": raw_args}


@dataclass
class ToolCallData:
    """Parsed tool call from an LLM response.

    This is a local definition used by LLMResponse to avoid circular imports.
    It's structurally identical to tools.ToolCall.

    Attributes:
        id: Unique identifier for this tool call
        name: Name of the tool to call
        args: Arguments to pass to the tool
    """

    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    """Structured response from an LLM call.

    Attributes:
        content: The text content of the response
        tool_calls: List of tool calls if the model requested tools
        finish_reason: Why the model stopped (stop, tool_calls, length, etc.)
        metrics: Token usage and latency metrics
        raw_response: The original ModelResponse from LiteLLM
    """

    content: str
    tool_calls: list[ToolCallData]
    finish_reason: str
    metrics: LLMMetrics
    raw_response: ModelResponse | None = field(default=None, repr=False)


class LLMClient:
    """Wrapper around LiteLLM with retry logic, rate limiting, fallback, and metrics.

    The LLMClient provides:
    - Multi-provider support via LiteLLM
    - Automatic retry on transient failures with exponential backoff
    - Rate limiting (RPM and TPM) via the global RateLimiter
    - Fallback model support when primary model fails after retries
    - Token counting and latency tracking
    - Event emission for observability (including AGENT_ERROR on failure)

    Attributes:
        event_bus: Optional EventBus for emitting LLM call metrics
        default_model: Default model to use if not specified
        fallback_model: Optional fallback model if primary fails after retries
        retry_attempts: Number of retry attempts for failed calls
        retry_delay: Delay between retry attempts in seconds
        rate_limiter: RateLimiter instance for throttling API calls
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        default_model: str | None = None,
        fallback_model: str | None = None,
        retry_attempts: int | None = None,
        retry_delay: float = 1.0,
        rate_limiter: RateLimiter | None = None,
        metrics_collector: Optional["MetricsCollector"] = None,
    ) -> None:
        """Initialize the LLM client.

        Args:
            event_bus: Optional EventBus for metric emission
            default_model: Model to use if not specified in calls
            fallback_model: Model to try if primary fails (defaults to config)
            retry_attempts: Number of retries (defaults to config llm_max_retries)
            retry_delay: Base seconds between retries (exponential backoff applied)
            rate_limiter: RateLimiter instance (defaults to global singleton)
            metrics_collector: Optional MetricsCollector for session-level token tracking
        """
        self.event_bus = event_bus
        self.default_model = default_model or settings.default_model
        self.fallback_model = fallback_model or settings.llm_fallback_model
        self.retry_attempts = (
            retry_attempts if retry_attempts is not None
            else settings.llm_max_retries
        )
        self.retry_delay = retry_delay
        self.rate_limiter = rate_limiter or get_rate_limiter()
        self.metrics_collector = metrics_collector

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
        """Make an LLM call with rate limiting, retry logic, fallback, and metrics.

        Flow:
        1. Acquire rate limit slot (waits if necessary)
        2. Attempt the call with exponential backoff retries
        3. If all retries fail and a fallback model is configured, try once
        4. Record actual token usage in the rate limiter
        5. Emit AGENT_ERROR event on final failure

        Retries on: RateLimitError (429), ServiceUnavailableError (500/502/503),
        Timeout errors.
        Does NOT retry on: AuthenticationError (401/403), BadRequestError (400),
        or other 4xx errors.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            model: Model to use (defaults to self.default_model)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            session_id: Optional session ID for event emission
            agent_id: Optional agent ID for event emission

        Returns:
            LLMResponse with content, tool calls, and metrics

        Raises:
            AuthenticationError: If API key is invalid
            BadRequestError: If request is malformed
            Exception: After all retries and fallback exhausted
        """
        model = model or self.default_model
        start_time = time.time()

        # Convert LangChain messages to plain dicts for LiteLLM
        messages = _convert_messages_to_dicts(messages)

        # Estimate tokens for rate limiting (rough: 4 chars per token)
        estimated_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
        estimated_tokens = max(estimated_tokens, 500)  # minimum estimate

        # Acquire rate limit slot
        try:
            await self.rate_limiter.acquire(estimated_tokens=estimated_tokens)
        except RateLimitExceededError as e:
            logger.error(
                "llm_call_rate_limit_exceeded",
                model=model,
                error=str(e),
            )
            if self.event_bus and session_id:
                await self._emit_error_event(
                    session_id=session_id,
                    agent_id=agent_id,
                    error=e,
                    model=model,
                    retry_count=0,
                    used_fallback=False,
                )
            raise

        # Try primary model with retries
        last_exception: Exception | None = None
        retry_count = 0

        for attempt in range(self.retry_attempts + 1):
            try:
                response = await self._make_request(
                    messages=messages,
                    tools=tools,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                latency_ms = int((time.time() - start_time) * 1000)
                llm_response = self._parse_response(response, model, latency_ms)

                # Record actual token usage in rate limiter
                actual_tokens = (
                    llm_response.metrics.input_tokens
                    + llm_response.metrics.output_tokens
                )
                await self.rate_limiter.record_usage(actual_tokens)

                # Emit metrics event if we have an event bus and session
                if self.event_bus and session_id:
                    await self._emit_metrics_event(
                        llm_response.metrics,
                        session_id,
                        agent_id,
                    )

                # Record in session-level metrics collector
                if self.metrics_collector and session_id:
                    await self.metrics_collector.record_llm_call(
                        session_id,
                        prompt_tokens=llm_response.metrics.input_tokens,
                        completion_tokens=llm_response.metrics.output_tokens,
                    )

                logger.info(
                    "llm_call_complete",
                    model=model,
                    input_tokens=llm_response.metrics.input_tokens,
                    output_tokens=llm_response.metrics.output_tokens,
                    latency_ms=latency_ms,
                    tool_calls=len(llm_response.tool_calls),
                    attempt=attempt + 1,
                )

                return llm_response

            except (RateLimitError, ServiceUnavailableError, Timeout) as e:
                last_exception = e
                retry_count = attempt + 1
                if attempt < self.retry_attempts:
                    delay = min(self.retry_delay * (2 ** attempt), 4.0)  # Cap backoff at 4s
                    logger.warning(
                        "llm_call_retry",
                        model=model,
                        attempt=attempt + 1,
                        max_retries=self.retry_attempts,
                        error_type=type(e).__name__,
                        error=str(e),
                        retry_delay=delay,
                    )

                    # Emit AGENT_ERROR on each retry so frontend shows activity
                    if self.event_bus and session_id:
                        await self._emit_error_event(
                            session_id=session_id,
                            agent_id=agent_id,
                            error=e,
                            model=model,
                            retry_count=retry_count,
                            used_fallback=False,
                        )

                    await self._async_sleep(delay)
                else:
                    logger.error(
                        "llm_call_failed_all_retries",
                        model=model,
                        attempts=self.retry_attempts + 1,
                        error_type=type(e).__name__,
                        error=str(e),
                    )

            except (AuthenticationError, BadRequestError) as e:
                # Don't retry authentication or bad request errors
                logger.error(
                    "llm_call_failed_no_retry",
                    model=model,
                    error_type=type(e).__name__,
                    error=str(e),
                )
                # Emit agent error event
                if self.event_bus and session_id:
                    await self._emit_error_event(
                        session_id=session_id,
                        agent_id=agent_id,
                        error=e,
                        model=model,
                        retry_count=0,
                        used_fallback=False,
                    )
                raise

        # All retries exhausted -- try fallback model if configured
        if self.fallback_model and self.fallback_model != model:
            logger.warning(
                "llm_fallback_attempt",
                primary_model=model,
                fallback_model=self.fallback_model,
                primary_retries=retry_count,
                primary_error=str(last_exception),
            )

            try:
                # Acquire another rate limit slot for the fallback
                await self.rate_limiter.acquire(estimated_tokens=estimated_tokens)

                response = await self._make_request(
                    messages=messages,
                    tools=tools,
                    model=self.fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                latency_ms = int((time.time() - start_time) * 1000)
                llm_response = self._parse_response(
                    response, self.fallback_model, latency_ms
                )

                # Record actual usage
                actual_tokens = (
                    llm_response.metrics.input_tokens
                    + llm_response.metrics.output_tokens
                )
                await self.rate_limiter.record_usage(actual_tokens)

                # Emit metrics
                if self.event_bus and session_id:
                    await self._emit_metrics_event(
                        llm_response.metrics,
                        session_id,
                        agent_id,
                    )

                # Record in session-level metrics collector
                if self.metrics_collector and session_id:
                    await self.metrics_collector.record_llm_call(
                        session_id,
                        prompt_tokens=llm_response.metrics.input_tokens,
                        completion_tokens=llm_response.metrics.output_tokens,
                    )

                logger.info(
                    "llm_fallback_success",
                    fallback_model=self.fallback_model,
                    input_tokens=llm_response.metrics.input_tokens,
                    output_tokens=llm_response.metrics.output_tokens,
                    latency_ms=latency_ms,
                )

                return llm_response

            except Exception as fallback_error:
                logger.error(
                    "llm_fallback_failed",
                    fallback_model=self.fallback_model,
                    error_type=type(fallback_error).__name__,
                    error=str(fallback_error),
                )
                # Keep original exception as the primary cause
                last_exception = last_exception or fallback_error

        # Emit agent error event for final failure
        if self.event_bus and session_id and last_exception:
            await self._emit_error_event(
                session_id=session_id,
                agent_id=agent_id,
                error=last_exception,
                model=model,
                retry_count=retry_count,
                used_fallback=bool(
                    self.fallback_model and self.fallback_model != model
                ),
            )

        raise last_exception or Exception("LLM call failed after all retries")

    async def _make_request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str,
        temperature: float,
        max_tokens: int | None,
    ) -> ModelResponse:
        """Make the actual LiteLLM request.

        Args:
            messages: Message history
            tools: Tool definitions
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Max response tokens

        Returns:
            Raw ModelResponse from LiteLLM
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        kwargs["timeout"] = settings.llm_request_timeout_seconds

        return await acompletion(**kwargs)

    def _parse_response(
        self,
        response: ModelResponse,
        model: str,
        latency_ms: int,
    ) -> LLMResponse:
        """Parse the LiteLLM response into our structured format.

        Args:
            response: Raw ModelResponse
            model: Model that was used
            latency_ms: Request latency

        Returns:
            Structured LLMResponse
        """
        choice = response.choices[0]
        message = choice.message

        # Extract content
        content = message.content or ""

        # Extract tool calls
        tool_calls: list[ToolCallData] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCallData(
                        id=tc.id,
                        name=tc.function.name,
                        args=normalize_tool_args(tc.function.arguments),
                    )
                )

        # Extract token usage
        usage = response.usage
        metrics = LLMMetrics(
            model=model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "unknown",
            metrics=metrics,
            raw_response=response,
        )

    async def _emit_metrics_event(
        self,
        metrics: LLMMetrics,
        session_id: str,
        agent_id: str | None,
    ) -> None:
        """Emit an LLM call complete event with metrics.

        Args:
            metrics: The LLMMetrics to emit
            session_id: Session ID for the event
            agent_id: Optional agent ID
        """
        if self.event_bus:
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.LLM_CALL_COMPLETE,
                    session_id=session_id,
                    agent_id=agent_id,
                    data={
                        "model": metrics.model,
                        "input_tokens": metrics.input_tokens,
                        "output_tokens": metrics.output_tokens,
                        "latency_ms": metrics.latency_ms,
                    },
                )
            )

    async def _emit_error_event(
        self,
        session_id: str,
        agent_id: str | None,
        error: Exception,
        model: str,
        retry_count: int,
        used_fallback: bool,
    ) -> None:
        """Emit an AGENT_ERROR event when an LLM call fails.

        Args:
            session_id: Session ID for the event
            agent_id: Optional agent ID
            error: The exception that caused the failure
            model: The model that was being called
            retry_count: How many retries were attempted
            used_fallback: Whether a fallback model was attempted
        """
        if self.event_bus:
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.AGENT_ERROR,
                    session_id=session_id,
                    agent_id=agent_id,
                    data={
                        "error": str(error),
                        "error_type": type(error).__name__,
                        "model": model,
                        "retry_count": retry_count,
                        "used_fallback": used_fallback,
                        "fallback_model": self.fallback_model,
                        "phase": "llm_call",
                    },
                )
            )

    async def _async_sleep(self, seconds: float) -> None:
        """Async sleep for retry delay.

        Extracted to a method for easier testing/mocking.

        Args:
            seconds: Number of seconds to sleep
        """
        import asyncio
        await asyncio.sleep(seconds)


async def call_llm(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
    temperature: float = 0.7,
) -> LLMResponse:
    """Convenience function for simple LLM calls.

    Creates a temporary LLMClient and makes a call. For repeated calls,
    prefer creating an LLMClient instance directly.

    Args:
        messages: List of message dicts with 'role' and 'content'
        tools: Optional list of tool definitions
        model: Model to use (defaults to settings.default_model)
        temperature: Sampling temperature

    Returns:
        LLMResponse with content, tool calls, and metrics
    """
    client = LLMClient()
    return await client.call(
        messages=messages,
        tools=tools,
        model=model,
        temperature=temperature,
    )


async def rate_limited_completion(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    event_bus: EventBus | None = None,
) -> LLMResponse:
    """Rate-limited wrapper around litellm.acompletion with retry and fallback.

    This is the recommended entry point for making LLM calls from agent code.
    It wraps ``litellm.acompletion`` with:
    - Token bucket rate limiting (RPM and TPM)
    - Exponential backoff retries on transient errors
    - Fallback model support
    - Structured logging and event emission

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model identifier (defaults to settings.default_model)
        tools: Optional list of tool definitions
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
        session_id: Optional session ID for event emission
        agent_id: Optional agent ID for event emission
        event_bus: Optional EventBus for emitting events

    Returns:
        LLMResponse with content, tool calls, and metrics
    """
    client = LLMClient(event_bus=event_bus)
    return await client.call(
        messages=messages,
        tools=tools,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        session_id=session_id,
        agent_id=agent_id,
    )


def sliding_window_prune(
    messages: list[Any],
    max_messages: int | None = None,
    max_tokens: int | None = None,
) -> list[Any]:
    """Prune message history with a sliding window, preserving key context.

    Keeps the system message (index 0), the original user task (index 1),
    and the most recent N messages. Drops middle messages to stay within
    token/count limits.

    Args:
        messages: Full message history
        max_messages: Maximum number of messages to keep (defaults to config)
        max_tokens: Token threshold to trigger pruning (defaults to config)

    Returns:
        Pruned message list
    """
    if max_messages is None:
        max_messages = settings.context_max_messages
    if max_tokens is None:
        max_tokens = settings.context_prune_threshold_tokens

    total_tokens = count_messages_tokens(messages)
    if len(messages) <= max_messages and total_tokens <= max_tokens:
        return messages

    # Preserve the fixed prefix context (system + original task) when present.
    prefix_count = min(2, len(messages))
    preserved_indexes = list(range(prefix_count))
    if len(messages) <= prefix_count:
        return messages

    normalized_messages = [_convert_message_to_dict(msg) for msg in messages]
    message_cap = max(max_messages - 1, prefix_count)  # Reserve 1 slot for prune marker.
    selected_indexes: set[int] = set(preserved_indexes)

    # Keep the prefix even if it exceeds token budget by itself.
    selected_tokens = count_messages_tokens([messages[i] for i in preserved_indexes])

    def is_priority_message(msg: dict[str, Any]) -> bool:
        role = str(msg.get("role", "")).lower()
        content = msg.get("content", "")
        text = content if isinstance(content, str) else json.dumps(content)
        lower = text.lower()

        # Preserve explicit failures and timeout/build diagnostics.
        error_markers = (
            "error",
            "failed",
            "failure",
            "exception",
            "traceback",
            "timeout",
            "timed out",
            "not found",
            "invalid",
        )
        if any(marker in lower for marker in error_markers):
            return True

        # Tool outputs often contain critical diagnostics.
        if role == "tool":
            if "exit code: 0" in lower or "success" in lower:
                return False
            return bool(text.strip())

        # Preserve explicit completion tags to avoid loops.
        return "task_complete" in lower or "needs_revision" in lower

    def try_add_index(index: int) -> bool:
        nonlocal selected_tokens
        if index in selected_indexes:
            return False
        if len(selected_indexes) >= message_cap:
            return False

        msg_tokens = count_messages_tokens([messages[index]])
        # If the preserved prefix already exceeded the token budget, keep selecting by
        # message cap to preserve recency and critical errors.
        over_budget = selected_tokens > max_tokens
        if not over_budget and selected_tokens + msg_tokens > max_tokens:
            return False

        selected_indexes.add(index)
        selected_tokens += msg_tokens
        return True

    # Pass 1: retain high-signal error/failure messages, newest first.
    for index in range(len(messages) - 1, prefix_count - 1, -1):
        if is_priority_message(normalized_messages[index]):
            try_add_index(index)

    # Pass 2: retain recent context (with a floor to avoid starving recency).
    recent_target = max(4, message_cap - prefix_count)
    recent_added = 0
    for index in range(len(messages) - 1, prefix_count - 1, -1):
        added = try_add_index(index)
        if added:
            recent_added += 1
        if len(selected_indexes) >= message_cap:
            break
        if recent_added >= recent_target and selected_tokens <= max_tokens:
            break

    kept_sorted = sorted(selected_indexes)
    dropped_count = len(messages) - len(kept_sorted)
    if dropped_count <= 0:
        return messages

    marker = {
        "role": "user",
        "content": (
            f"[Note: {dropped_count} earlier messages were pruned "
            "for context limits. System/task context, recent turns, "
            "and critical error/tool diagnostics were preserved.]"
        ),
    }
    result = [messages[i] for i in kept_sorted]

    insert_at = prefix_count if len(result) >= prefix_count else len(result)
    result = result[:insert_at] + [marker] + result[insert_at:]

    logger.debug(
        "context_pruned",
        original_messages=len(messages),
        pruned_messages=len(result),
        dropped_count=dropped_count,
        kept_priority_messages=sum(
            1
            for idx in kept_sorted
            if idx >= prefix_count and is_priority_message(normalized_messages[idx])
        ),
    )

    return result


def count_messages_tokens(messages: list[Any]) -> int:
    """Estimate total token count for a list of messages.

    Uses the ~4 characters per token heuristic. Counts content,
    tool call arguments, and tool results.

    Handles both plain dicts and LangChain message objects (which
    LangGraph's ``add_messages`` reducer may produce).

    Args:
        messages: List of message dicts or LangChain message objects

    Returns:
        Estimated token count
    """
    total_chars = 0
    for raw_msg in messages:
        msg = _convert_message_to_dict(raw_msg)
        # Count content
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)

        # Count tool call arguments
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", "")
            if isinstance(args, str):
                total_chars += len(args)
            elif isinstance(args, dict):
                total_chars += len(json.dumps(args))

    return total_chars // 4


def parse_status_tag(response: str) -> str | None:
    """Extract the status from <status> tags in a response.

    Agents are instructed to wrap their completion status in <status> tags.
    Valid values: TASK_COMPLETE, NEEDS_REVISION.

    Args:
        response: The full LLM response text

    Returns:
        The status string if found (e.g. "TASK_COMPLETE"), or None
    """
    pattern = r"<status>\s*(TASK_COMPLETE|NEEDS_REVISION)\s*</status>"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def parse_analysis_tag(response: str) -> str | None:
    """Extract the analysis content from <analysis> tags in a response.

    Evaluators are instructed to wrap their reasoning in <analysis> tags
    before producing the JSON scores.

    Args:
        response: The full LLM response text

    Returns:
        The content inside <analysis> tags, or None if not found
    """
    pattern = r"<analysis>(.*?)</analysis>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def topological_sort(subtasks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Sort subtasks into dependency layers via topological sort.

    Groups subtasks so that all dependencies in layer N are resolved
    before layer N+1 begins. Layer 0 has no dependencies.

    Args:
        subtasks: List of subtask dicts, each with 'id' and 'dependencies' fields

    Returns:
        List of layers, where each layer is a list of subtasks that can
        run in parallel

    Raises:
        ValueError: If a circular dependency is detected
    """
    # Build lookup and dependency graph
    task_map: dict[str, dict[str, Any]] = {}
    for st in subtasks:
        task_map[st["id"]] = st

    # Track which tasks are resolved
    resolved: set[str] = set()
    remaining = {st["id"] for st in subtasks}
    layers: list[list[dict[str, Any]]] = []

    max_iterations = len(subtasks) + 1  # Safety bound
    for _ in range(max_iterations):
        if not remaining:
            break

        # Find tasks whose dependencies are all resolved
        ready = []
        for task_id in remaining:
            deps = task_map[task_id].get("dependencies", [])
            # Filter deps to only those that are actual subtask IDs
            valid_deps = [d for d in deps if d in task_map]
            if all(d in resolved for d in valid_deps):
                ready.append(task_id)

        if not ready:
            # Circular dependency or invalid references - break cycle by
            # adding all remaining tasks to one final layer
            logger.warning(
                "topological_sort_cycle_detected",
                remaining=list(remaining),
            )
            layers.append([task_map[tid] for tid in remaining])
            break

        # Add this layer
        layer = [task_map[tid] for tid in ready]
        layers.append(layer)
        resolved.update(ready)
        remaining -= set(ready)

    return layers


def classify_build_errors(build_output: str) -> dict[str, list[str]]:
    """Classify build errors into categories for targeted fix guidance.

    Categorizes TypeScript/JavaScript build errors by type to help
    the LLM focus on the right kind of fix.

    Args:
        build_output: Raw build output string

    Returns:
        Dict mapping category name to list of error messages
    """
    categories: dict[str, list[str]] = {
        "type_errors": [],
        "import_errors": [],
        "syntax_errors": [],
        "missing_modules": [],
        "other": [],
    }

    lines = build_output.split("\n")
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        lower = line_stripped.lower()

        if re.search(r"TS\(?\d+\)?", line_stripped) or ("type" in lower and "error" in lower):
            categories["type_errors"].append(line_stripped)
        elif "cannot find module" in lower or "module not found" in lower:
            categories["missing_modules"].append(line_stripped)
        elif "import" in lower and ("error" in lower or "not found" in lower):
            categories["import_errors"].append(line_stripped)
        elif "syntaxerror" in lower or "unexpected token" in lower:
            categories["syntax_errors"].append(line_stripped)
        elif "error" in lower:
            categories["other"].append(line_stripped)

    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def parse_plan_tag(response: str) -> str:
    """Extract the plan content from <plan> tags in a response.

    Agents are instructed to wrap their planning output in <plan> tags.
    This function extracts that content for display in the UI.

    Args:
        response: The full LLM response text

    Returns:
        The content inside <plan> tags, or empty string if not found
    """
    pattern = r"<plan>(.*?)</plan>"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        return match.group(1).strip()

    return ""


def parse_tool_calls(response: LLMResponse) -> list[ToolCallData]:
    """Parse tool calls from an LLM response.

    This is a convenience function that returns the tool_calls
    from an LLMResponse. The response already contains parsed
    tool calls, but this provides a consistent interface.

    Args:
        response: The LLMResponse to extract tool calls from

    Returns:
        List of ToolCallData objects, may be empty
    """
    return response.tool_calls


def format_tool_result_for_llm(
    tool_call_id: str,
    result: str,
) -> dict[str, Any]:
    """Format a tool result as a message for the LLM.

    Creates a tool message that can be appended to the message
    history to inform the LLM of the tool execution result.

    Args:
        tool_call_id: The ID of the tool call this result corresponds to
        result: The string result from tool execution

    Returns:
        A message dict in the format expected by LLMs
    """
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": result,
    }


def format_assistant_message_with_tools(
    content: str,
    tool_calls: list[ToolCallData],
) -> dict[str, Any]:
    """Format an assistant message that includes tool calls.

    Creates a message dict representing an assistant response
    that both contains content and requests tool calls.

    Args:
        content: The assistant's text response
        tool_calls: List of ToolCallData the assistant made

    Returns:
        A message dict in the format expected by LLMs
    """
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }

    if tool_calls:
        message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.args),
                },
            }
            for tc in tool_calls
        ]

    return message


def _extract_balanced_json_objects(text: str) -> list[str]:
    """Extract balanced JSON object candidates from arbitrary text."""
    candidates: list[str] = []
    n = len(text)

    for start in range(n):
        if text[start] != "{":
            continue

        depth = 0
        in_string = False
        escaped = False

        for end in range(start, n):
            ch = text[end]

            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : end + 1])
                    break

    return candidates


def extract_json_from_response(response: str) -> dict[str, Any] | None:
    """Extract JSON from an LLM response that may contain extra text.

    Attempts to find and parse JSON within a response that may
    have additional text before or after the JSON content.

    Args:
        response: The full LLM response text

    Returns:
        Parsed JSON dict if found, None otherwise
    """
    def try_parse(candidate: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    # 1) Pure JSON response.
    parsed = try_parse(response.strip())
    if parsed is not None:
        return parsed

    # 2) JSON within fenced blocks.
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    for match in re.finditer(fence_pattern, response, re.IGNORECASE):
        fenced_body = match.group(1).strip()
        parsed = try_parse(fenced_body)
        if parsed is not None:
            return parsed
        for candidate in _extract_balanced_json_objects(fenced_body):
            parsed = try_parse(candidate)
            if parsed is not None:
                return parsed

    # 3) Balanced object extraction from free-form response.
    for candidate in _extract_balanced_json_objects(response):
        parsed = try_parse(candidate)
        if parsed is not None:
            return parsed

    return None


def count_tokens_estimate(text: str) -> int:
    """Estimate token count for a text string.

    This is a rough estimate using the ~4 characters per token
    rule of thumb. For accurate counts, use the LLM's tokenizer.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without API calls.

    Provides predefined responses for testing agent behavior
    without incurring API costs or network latency.

    Usage:
        >>> responses = [
        ...     LLMResponse(content="Hello", tool_calls=[], ...),
        ...     LLMResponse(content="TASK_COMPLETE", tool_calls=[], ...),
        ... ]
        >>> client = MockLLMClient(responses=responses)
        >>> response = await client.call(messages=[...])
    """

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with predefined responses.

        Args:
            responses: List of responses to return in order
            **kwargs: Additional args passed to parent
        """
        super().__init__(**kwargs)
        self.responses = list(responses) if responses else []
        self.call_history: list[dict[str, Any]] = []
        self._response_index = 0

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
        """Return the next predefined response.

        Records the call for later inspection.

        Args:
            All args are recorded but only used for tracking

        Returns:
            Next response from the predefined list

        Raises:
            IndexError: If no more responses available
        """
        self.call_history.append({
            "messages": messages,
            "tools": tools,
            "model": model or self.default_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

        if self._response_index >= len(self.responses):
            raise IndexError("No more mock responses available")

        response = self.responses[self._response_index]
        self._response_index += 1

        logger.debug(
            "mock_llm_call",
            response_index=self._response_index - 1,
            content_preview=response.content[:50] if response.content else "",
            tool_calls=len(response.tool_calls),
        )

        return response

    def reset(self) -> None:
        """Reset the mock to start returning responses from the beginning."""
        self._response_index = 0
        self.call_history.clear()
