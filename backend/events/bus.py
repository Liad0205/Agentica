"""Async event bus for agent swarm pub/sub communication.

This module provides an EventBus class that enables asynchronous
publish/subscribe communication between backend agents and frontend
consumers (via WebSocket).

The event bus is thread-safe and supports:
- Multiple subscribers per session
- Async event delivery via asyncio.Queue
- Session lifecycle management (close session terminates all subscribers)
"""

import asyncio
import contextlib
import threading
from collections import defaultdict

import structlog

from events.types import AgentEvent, EventType

logger = structlog.get_logger()


class EventBus:
    """Async pub/sub event bus for agent events.

    The EventBus manages subscriptions per session, allowing multiple
    WebSocket connections to receive events for the same session.
    Events are delivered via asyncio.Queue for non-blocking consumption.

    Event Buffering:
        Events published before any subscriber connects are buffered.
        When the first subscriber connects, all buffered events are
        delivered immediately. This handles the race condition where
        the backend starts emitting events before the WebSocket connects.

    Thread Safety:
        All operations use a threading.Lock to ensure thread-safe access
        to the subscription registry, allowing the bus to be safely used
        from multiple threads (e.g., LangGraph execution threads).

    Usage:
        >>> bus = EventBus()
        >>>
        >>> # Subscribe to a session's events
        >>> queue = bus.subscribe("session_123")
        >>>
        >>> # In another coroutine, publish events
        >>> await bus.publish(AgentEvent(
        ...     type=EventType.AGENT_SPAWNED,
        ...     session_id="session_123",
        ...     agent_id="agent_1",
        ...     data={"role": "ReAct Agent"}
        ... ))
        >>>
        >>> # Consumer receives the event
        >>> event = await queue.get()
        >>>
        >>> # Clean up
        >>> bus.unsubscribe("session_123", queue)
        >>> await bus.close_session("session_123")

    Attributes:
        _subscribers: Dict mapping session_id to list of subscriber queues
        _event_buffer: Dict mapping session_id to list of buffered events
        _lock: Threading lock for thread-safe subscriber management
    """

    # Maximum number of events to retain per session for replay on reconnect.
    MAX_HISTORY_PER_SESSION = 5000

    def __init__(self) -> None:
        """Initialize an empty event bus."""
        self._subscribers: dict[str, list[asyncio.Queue[AgentEvent]]] = defaultdict(list)
        self._event_buffer: dict[str, list[AgentEvent]] = defaultdict(list)
        self._event_history: dict[str, list[AgentEvent]] = defaultdict(list)
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        logger.info("event_bus_initialized")

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop, caching it on first use."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.get_running_loop()
        return self._loop

    def subscribe(self, session_id: str) -> asyncio.Queue[AgentEvent]:
        """Subscribe to events for a session.

        Creates a new asyncio.Queue for the subscriber and registers it
        to receive events for the specified session.

        If there are buffered events for this session (events that were
        published before any subscriber connected), they are delivered
        immediately to the new subscriber.

        Args:
            session_id: The session to subscribe to

        Returns:
            An asyncio.Queue that will receive AgentEvent objects
            as they are published for this session
        """
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        buffered_events: list[AgentEvent] = []

        with self._lock:
            self._subscribers[session_id].append(queue)
            subscriber_count = len(self._subscribers[session_id])

            # Get and clear buffered events for this session
            if session_id in self._event_buffer:
                buffered_events = self._event_buffer[session_id]
                del self._event_buffer[session_id]

        # Deliver buffered events to the new subscriber
        for event in buffered_events:
            queue.put_nowait(event)

        logger.info(
            "subscriber_added",
            session_id=session_id,
            subscriber_count=subscriber_count,
            buffered_events_delivered=len(buffered_events),
        )
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue[AgentEvent]) -> None:
        """Unsubscribe a queue from session events.

        Removes the specified queue from the subscriber list for the session.
        If the queue is not registered, this is a no-op.

        Args:
            session_id: The session to unsubscribe from
            queue: The queue to remove
        """
        with self._lock:
            if session_id in self._subscribers:
                try:
                    self._subscribers[session_id].remove(queue)
                    subscriber_count = len(self._subscribers[session_id])
                    logger.info(
                        "subscriber_removed",
                        session_id=session_id,
                        subscriber_count=subscriber_count,
                    )
                    # Clean up empty subscriber lists
                    if not self._subscribers[session_id]:
                        del self._subscribers[session_id]
                        logger.debug(
                            "session_subscribers_cleared",
                            session_id=session_id,
                        )
                except ValueError:
                    # Queue was not in the list
                    logger.warning(
                        "unsubscribe_queue_not_found",
                        session_id=session_id,
                    )

    async def publish(self, event: AgentEvent) -> None:
        """Publish an event to all subscribers for its session.

        The event's session_id determines which subscribers receive it.
        Events are added to all registered queues for that session.

        If there are no subscribers, the event is buffered until a
        subscriber connects. This handles the race condition where
        events are published before the WebSocket connects.

        All published events are also stored in the session's event
        history for replay on reconnect.

        This method is non-blocking; if a subscriber's queue is full,
        the event is still added (queues are unbounded by default).

        Args:
            event: The AgentEvent to publish
        """
        # Cache the event loop on first async publish
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        with self._lock:
            # Store event in history for replay (skip sentinel events)
            if event.type != EventType.SESSION_CLOSED:
                history = self._event_history[event.session_id]
                history.append(event)
                if len(history) > self.MAX_HISTORY_PER_SESSION:
                    self._event_history[event.session_id] = history[-self.MAX_HISTORY_PER_SESSION:]

            subscribers = list(self._subscribers.get(event.session_id, []))

            if not subscribers:
                # Buffer the event for later delivery
                self._event_buffer[event.session_id].append(event)
                logger.debug(
                    "event_buffered",
                    session_id=event.session_id,
                    event_type=event.type.value,
                    buffer_size=len(self._event_buffer[event.session_id]),
                )
                return

        # Publish to all subscribers with timeout to avoid blocking
        # if a consumer stalls (e.g. frozen WebSocket client)
        for queue in subscribers:
            try:
                await asyncio.wait_for(queue.put(event), timeout=5.0)
            except TimeoutError:
                logger.warning(
                    "event_delivery_timeout",
                    session_id=event.session_id,
                    event_type=event.type.value,
                )
            except Exception:
                logger.warning(
                    "event_delivery_failed",
                    session_id=event.session_id,
                    event_type=event.type.value,
                )

        logger.debug(
            "event_published",
            session_id=event.session_id,
            event_type=event.type.value,
            subscriber_count=len(subscribers),
            agent_id=event.agent_id,
        )

    def publish_sync(self, event: AgentEvent) -> None:
        """Synchronously publish an event (for use from non-async contexts).

        This method schedules the put on the event loop thread via
        call_soon_threadsafe, since asyncio.Queue is NOT thread-safe.
        Use this when publishing from synchronous code, such as
        LangGraph callbacks running in executor threads.

        If there are no subscribers, the event is buffered until a
        subscriber connects.

        Args:
            event: The AgentEvent to publish
        """
        with self._lock:
            # Store event in history for replay (skip sentinel events)
            if event.type != EventType.SESSION_CLOSED:
                history = self._event_history[event.session_id]
                history.append(event)
                if len(history) > self.MAX_HISTORY_PER_SESSION:
                    self._event_history[event.session_id] = history[-self.MAX_HISTORY_PER_SESSION:]

            subscribers = list(self._subscribers.get(event.session_id, []))
            loop = self._loop  # Cache inside lock to avoid TOCTOU race

            if not subscribers:
                # Buffer the event for later delivery (dict append is safe
                # here since we hold the lock)
                self._event_buffer[event.session_id].append(event)
                logger.debug(
                    "event_buffered_sync",
                    session_id=event.session_id,
                    event_type=event.type.value,
                    buffer_size=len(self._event_buffer[event.session_id]),
                )
                return

        # Schedule put_nowait on the event loop thread for thread safety.
        # asyncio.Queue.put_nowait is NOT thread-safe and must run on the
        # event loop thread.
        if loop is not None and not loop.is_closed():
            for queue in subscribers:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
        else:
            # Fallback: no loop available yet, try direct put
            for queue in subscribers:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(
                        "queue_full_event_dropped",
                        session_id=event.session_id,
                        event_type=event.type.value,
                    )

        logger.debug(
            "event_published_sync",
            session_id=event.session_id,
            event_type=event.type.value,
            subscriber_count=len(subscribers),
        )

    def get_event_history(self, session_id: str) -> list[AgentEvent]:
        """Get all stored events for a session.

        This is used for replaying events to a newly connected WebSocket
        client (e.g. after a page refresh).

        Args:
            session_id: The session to get history for.

        Returns:
            A list of AgentEvent objects in chronological order.
        """
        with self._lock:
            return list(self._event_history.get(session_id, []))

    async def close_session(self, session_id: str) -> None:
        """Close a session and notify all subscribers.

        Puts a sentinel (None) into each subscriber queue so that consumers
        (e.g. the WebSocket send_events loop) can detect the session has
        ended and break out cleanly. Then removes all subscribers and clears
        buffered events. Event history is preserved for potential reconnect.

        Args:
            session_id: The session to close
        """
        queues_to_signal: list[asyncio.Queue[AgentEvent]] = []

        with self._lock:
            subscriber_count = 0
            buffer_count = 0

            if session_id in self._subscribers:
                subscriber_count = len(self._subscribers[session_id])
                queues_to_signal = list(self._subscribers[session_id])
                del self._subscribers[session_id]

            if session_id in self._event_buffer:
                buffer_count = len(self._event_buffer[session_id])
                del self._event_buffer[session_id]

        # Signal subscribers so they can break out of their read loops.
        # We use a SESSION_CLOSED sentinel event.
        for queue in queues_to_signal:
            try:
                sentinel = AgentEvent(
                    type=EventType.SESSION_CLOSED,
                    session_id=session_id,
                    data={"reason": "session_closed"},
                )
                await queue.put(sentinel)
            except Exception:
                pass

        if subscriber_count > 0 or buffer_count > 0:
            logger.info(
                "session_closed",
                session_id=session_id,
                subscribers_removed=subscriber_count,
                buffered_events_cleared=buffer_count,
            )
        else:
            logger.debug(
                "close_session_not_found",
                session_id=session_id,
            )

    def get_subscriber_count(self, session_id: str) -> int:
        """Get the number of subscribers for a session.

        Args:
            session_id: The session to check

        Returns:
            Number of active subscribers for the session
        """
        with self._lock:
            return len(self._subscribers.get(session_id, []))

    def get_active_sessions(self) -> list[str]:
        """Get list of sessions with active subscribers.

        Returns:
            List of session IDs that have at least one subscriber
        """
        with self._lock:
            return list(self._subscribers.keys())

    def clear_event_history(self, session_id: str) -> None:
        """Clear stored event history for a session.

        Call this when a session is fully destroyed and its data
        should no longer be available for replay.

        Args:
            session_id: The session to clear history for.
        """
        with self._lock:
            self._event_history.pop(session_id, None)


# Global event bus instance
_event_bus: EventBus | None = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Get the global EventBus instance.

    Creates the instance on first call (lazy initialization).
    This function is thread-safe.

    Returns:
        The global EventBus instance
    """
    global _event_bus
    if _event_bus is None:
        with _bus_lock:
            # Double-check locking pattern
            if _event_bus is None:
                _event_bus = EventBus()
    return _event_bus


def reset_event_bus() -> None:
    """Reset the global EventBus instance.

    This is primarily useful for testing to ensure a clean state
    between test runs.
    """
    global _event_bus
    with _bus_lock:
        _event_bus = None
    logger.info("event_bus_reset")
