"""Event system for agent swarm communication.

This package provides the event infrastructure for communication between
backend agent execution and frontend visualization. The event system is
based on an async pub/sub pattern using asyncio.Queue.

Key Components:
    - EventType: Enum of all event types in the system
    - AgentEvent: Pydantic model for events flowing through the system
    - EventBus: Async pub/sub implementation for event distribution
    - LLMMetrics: Token and latency metrics for individual LLM calls
    - SessionMetrics: Aggregate metrics for entire sessions

Usage:
    >>> from events import EventType, AgentEvent, EventBus, get_event_bus
    >>>
    >>> # Get the global event bus
    >>> bus = get_event_bus()
    >>>
    >>> # Subscribe to a session's events
    >>> queue = bus.subscribe("session_123")
    >>>
    >>> # Publish an event
    >>> await bus.publish(AgentEvent(
    ...     type=EventType.AGENT_SPAWNED,
    ...     session_id="session_123",
    ...     agent_id="agent_1",
    ...     data={"role": "ReAct Agent", "sandbox_id": "sandbox_abc"}
    ... ))
    >>>
    >>> # Receive the event
    >>> event = await queue.get()
    >>> print(f"Received: {event.type.value}")

Event Flow:
    The typical event flow is:
    1. LangGraph node execution emits events via EventBus.publish()
    2. WebSocket handler subscribes to session events
    3. Events are forwarded to frontend via WebSocket
    4. Frontend store dispatches events to UI components
"""

from events.bus import (
    EventBus,
    get_event_bus,
    reset_event_bus,
)
from events.types import (
    AgentEvent,
    EventType,
    LLMMetrics,
    SessionMetrics,
)

__all__ = [
    # Event types
    "EventType",
    "AgentEvent",
    "LLMMetrics",
    "SessionMetrics",
    # Event bus
    "EventBus",
    "get_event_bus",
    "reset_event_bus",
]
