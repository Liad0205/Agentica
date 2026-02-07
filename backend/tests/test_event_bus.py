"""Tests for events/bus.py -- async pub/sub event bus.

Covers publish/subscribe, buffering, close_session sentinel,
thread-safe publish_sync, error isolation between subscribers,
and the global singleton accessor.
"""

import asyncio
import threading

from events.bus import EventBus, get_event_bus, reset_event_bus
from events.types import AgentEvent, EventType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    session_id: str = "sess_test",
    event_type: EventType = EventType.AGENT_SPAWNED,
) -> AgentEvent:
    return AgentEvent(
        type=event_type,
        session_id=session_id,
        data={"test": True},
    )


# =========================================================================
# Subscribe / Publish basics
# =========================================================================


class TestSubscribePublish:
    """Basic subscribe and async publish."""

    async def test_subscribe_returns_queue(self, event_bus: EventBus) -> None:
        queue = event_bus.subscribe("sess_1")
        assert isinstance(queue, asyncio.Queue)

    async def test_publish_delivers_to_subscriber(self, event_bus: EventBus) -> None:
        queue = event_bus.subscribe("sess_1")
        event = _make_event("sess_1")
        await event_bus.publish(event)
        received = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert received.type == EventType.AGENT_SPAWNED
        assert received.session_id == "sess_1"

    async def test_publish_multiple_subscribers(self, event_bus: EventBus) -> None:
        q1 = event_bus.subscribe("sess_1")
        q2 = event_bus.subscribe("sess_1")
        event = _make_event("sess_1")
        await event_bus.publish(event)
        r1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        r2 = await asyncio.wait_for(q2.get(), timeout=1.0)
        assert r1.type == r2.type == EventType.AGENT_SPAWNED

    async def test_publish_does_not_cross_sessions(self, event_bus: EventBus) -> None:
        q1 = event_bus.subscribe("sess_1")
        q2 = event_bus.subscribe("sess_2")
        await event_bus.publish(_make_event("sess_1"))
        # q1 should have the event
        r1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        assert r1.session_id == "sess_1"
        # q2 should be empty
        assert q2.empty()


# =========================================================================
# Event Buffering
# =========================================================================


class TestEventBuffering:
    """Events published before a subscriber connects are buffered."""

    async def test_buffered_events_delivered_on_subscribe(
        self, event_bus: EventBus
    ) -> None:
        e1 = _make_event("sess_1", EventType.SESSION_STARTED)
        e2 = _make_event("sess_1", EventType.AGENT_SPAWNED)
        await event_bus.publish(e1)
        await event_bus.publish(e2)

        queue = event_bus.subscribe("sess_1")
        r1 = await asyncio.wait_for(queue.get(), timeout=1.0)
        r2 = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert r1.type == EventType.SESSION_STARTED
        assert r2.type == EventType.AGENT_SPAWNED

    async def test_buffer_cleared_after_subscribe(self, event_bus: EventBus) -> None:
        await event_bus.publish(_make_event("sess_1"))
        q1 = event_bus.subscribe("sess_1")
        # The event should be in q1
        assert not q1.empty()
        # A second subscriber should NOT get the already-delivered buffer
        q2 = event_bus.subscribe("sess_1")
        assert q2.empty()


# =========================================================================
# Unsubscribe
# =========================================================================


class TestUnsubscribe:
    """Unsubscribe removes a specific queue from the session."""

    async def test_unsubscribe_removes_queue(self, event_bus: EventBus) -> None:
        queue = event_bus.subscribe("sess_1")
        event_bus.unsubscribe("sess_1", queue)
        assert event_bus.get_subscriber_count("sess_1") == 0

    async def test_unsubscribe_nonexistent_is_noop(self, event_bus: EventBus) -> None:
        dummy: asyncio.Queue[AgentEvent] = asyncio.Queue()
        # Should not raise
        event_bus.unsubscribe("no_such_session", dummy)

    async def test_unsubscribe_wrong_queue_is_noop(self, event_bus: EventBus) -> None:
        event_bus.subscribe("sess_1")
        wrong_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        # Should not raise; the wrong queue is simply not found
        event_bus.unsubscribe("sess_1", wrong_queue)
        assert event_bus.get_subscriber_count("sess_1") == 1

    async def test_after_unsubscribe_events_not_delivered(
        self, event_bus: EventBus
    ) -> None:
        queue = event_bus.subscribe("sess_1")
        event_bus.unsubscribe("sess_1", queue)
        await event_bus.publish(_make_event("sess_1"))
        # The queue should remain empty because we unsubscribed
        assert queue.empty()


# =========================================================================
# close_session -- sentinel
# =========================================================================


class TestCloseSession:
    """close_session sends a SESSION_CLOSED sentinel and cleans up."""

    async def test_close_sends_sentinel(self, event_bus: EventBus) -> None:
        queue = event_bus.subscribe("sess_1")
        await event_bus.close_session("sess_1")
        sentinel = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert sentinel.type == EventType.SESSION_CLOSED
        assert sentinel.session_id == "sess_1"

    async def test_close_removes_subscribers(self, event_bus: EventBus) -> None:
        event_bus.subscribe("sess_1")
        await event_bus.close_session("sess_1")
        assert event_bus.get_subscriber_count("sess_1") == 0

    async def test_close_clears_buffer(self, event_bus: EventBus) -> None:
        await event_bus.publish(_make_event("sess_1"))
        await event_bus.close_session("sess_1")
        # A new subscriber should NOT receive old buffered events
        queue = event_bus.subscribe("sess_1")
        assert queue.empty()

    async def test_close_nonexistent_is_noop(self, event_bus: EventBus) -> None:
        # Should not raise
        await event_bus.close_session("no_such_session")

    async def test_close_multiple_subscribers(self, event_bus: EventBus) -> None:
        q1 = event_bus.subscribe("sess_1")
        q2 = event_bus.subscribe("sess_1")
        await event_bus.close_session("sess_1")
        s1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        s2 = await asyncio.wait_for(q2.get(), timeout=1.0)
        assert s1.type == EventType.SESSION_CLOSED
        assert s2.type == EventType.SESSION_CLOSED


# =========================================================================
# publish_sync -- thread-safe synchronous publish
# =========================================================================


class TestPublishSync:
    """publish_sync uses call_soon_threadsafe for thread safety."""

    async def test_publish_sync_delivers_event(self, event_bus: EventBus) -> None:
        queue = event_bus.subscribe("sess_1")
        # Cache the loop so publish_sync can use it
        event_bus._loop = asyncio.get_running_loop()

        event = _make_event("sess_1")
        event_bus.publish_sync(event)

        # call_soon_threadsafe schedules on the loop; we need to yield
        await asyncio.sleep(0.05)
        assert not queue.empty()
        received = queue.get_nowait()
        assert received.type == EventType.AGENT_SPAWNED

    async def test_publish_sync_from_thread(self, event_bus: EventBus) -> None:
        queue = event_bus.subscribe("sess_1")
        event_bus._loop = asyncio.get_running_loop()

        event = _make_event("sess_1")
        done = threading.Event()

        def bg_publish() -> None:
            event_bus.publish_sync(event)
            done.set()

        thread = threading.Thread(target=bg_publish)
        thread.start()
        done.wait(timeout=2.0)
        thread.join(timeout=2.0)

        await asyncio.sleep(0.05)
        assert not queue.empty()
        received = queue.get_nowait()
        assert received.session_id == "sess_1"

    async def test_publish_sync_buffers_when_no_subscribers(
        self, event_bus: EventBus
    ) -> None:
        event_bus._loop = asyncio.get_running_loop()
        event = _make_event("sess_1")
        event_bus.publish_sync(event)

        # Now subscribe -- buffered events should be delivered
        queue = event_bus.subscribe("sess_1")
        assert not queue.empty()

    async def test_publish_sync_no_loop_fallback(self, event_bus: EventBus) -> None:
        """When no event loop is cached, publish_sync falls back to put_nowait."""
        queue = event_bus.subscribe("sess_1")
        event_bus._loop = None  # No loop available

        event = _make_event("sess_1")
        event_bus.publish_sync(event)

        assert not queue.empty()
        received = queue.get_nowait()
        assert received.type == EventType.AGENT_SPAWNED


# =========================================================================
# Error isolation
# =========================================================================


class TestErrorIsolation:
    """A failing subscriber should not prevent delivery to other subscribers."""

    async def test_error_does_not_block_other_subscribers(
        self, event_bus: EventBus
    ) -> None:
        q1 = event_bus.subscribe("sess_1")
        q2 = event_bus.subscribe("sess_1")

        # Make q1.put raise an exception
        _original_put = q1.put
        call_count = 0

        async def failing_put(item: AgentEvent) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("subscriber error")

        q1.put = failing_put  # type: ignore[assignment]

        event = _make_event("sess_1")
        await event_bus.publish(event)

        # q2 should still have received the event
        assert not q2.empty()
        received = q2.get_nowait()
        assert received.type == EventType.AGENT_SPAWNED
        # The failing put should have been called
        assert call_count == 1


# =========================================================================
# Global singleton
# =========================================================================


class TestGlobalEventBus:
    """get_event_bus / reset_event_bus singleton pattern."""

    def test_get_event_bus_returns_same_instance(self) -> None:
        reset_event_bus()
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_reset_event_bus_creates_new_instance(self) -> None:
        reset_event_bus()
        bus1 = get_event_bus()
        reset_event_bus()
        bus2 = get_event_bus()
        assert bus1 is not bus2


# =========================================================================
# Subscriber count and active sessions
# =========================================================================


class TestSubscriberInfo:
    """Utility methods for inspecting bus state."""

    async def test_subscriber_count(self, event_bus: EventBus) -> None:
        assert event_bus.get_subscriber_count("sess_1") == 0
        event_bus.subscribe("sess_1")
        assert event_bus.get_subscriber_count("sess_1") == 1
        event_bus.subscribe("sess_1")
        assert event_bus.get_subscriber_count("sess_1") == 2

    async def test_active_sessions(self, event_bus: EventBus) -> None:
        assert event_bus.get_active_sessions() == []
        event_bus.subscribe("sess_1")
        event_bus.subscribe("sess_2")
        sessions = event_bus.get_active_sessions()
        assert set(sessions) == {"sess_1", "sess_2"}
