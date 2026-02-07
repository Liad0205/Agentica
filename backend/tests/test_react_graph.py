"""End-to-end execution tests for the ReAct agent graph.

These tests compile the actual LangGraph and invoke ``graph.run()`` with a
``MockLLMClient``, verifying that nodes are traversed correctly, state
transitions happen as expected, and the right events are emitted.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agents.react_graph import ReactGraph, create_initial_state
from events.types import EventType
from tests.conftest import (
    MockLLMClient,
    collect_events,
    make_llm_response,
    make_tool_call,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SESSION_ID = "sess_react_test"
SANDBOX_ID = "sandbox_react_test"


def _build_graph(
    mock_sandbox_manager: AsyncMock,
    event_bus,
    mock_llm: MockLLMClient,
) -> ReactGraph:
    """Compile a real ReactGraph backed by a mock LLM and sandbox."""
    return ReactGraph(
        sandbox_manager=mock_sandbox_manager,
        event_bus=event_bus,
        llm_client=mock_llm,
    )


# ---------------------------------------------------------------------------
# Execution Tests
# ---------------------------------------------------------------------------


class TestReactGraphExecution:
    """Tests that actually compile the graph and run it end-to-end."""

    @pytest.mark.asyncio
    async def test_happy_path_single_iteration(
        self, mock_sandbox_manager, event_bus
    ) -> None:
        """Reason produces a tool call, execute runs it, review completes.

        Expected final state:
        - status == "complete"
        - iteration == 1
        - files_written includes the written file
        """
        # Mock the tool executor to return success
        mock_sandbox_manager.write_file = AsyncMock()
        mock_sandbox_manager.execute_command = AsyncMock(
            return_value=AsyncMock(
                stdout="OK", stderr="", exit_code=0, timed_out=False
            )
        )

        mock_llm = MockLLMClient(responses=[
            # 1) reason: plan + tool calls (write, build, lint)
            make_llm_response(
                content="<plan>Write App.tsx</plan>",
                tool_calls=[
                    make_tool_call(
                        "write_file",
                        {
                            "path": "src/App.tsx",
                            "content": "export default function App() { return <div>Hello</div> }",
                        },
                        call_id="tc_write",
                    ),
                    make_tool_call(
                        "execute_command",
                        {"command": "npm run build"},
                        call_id="tc_build",
                    ),
                    make_tool_call(
                        "execute_command",
                        {"command": "npm run lint"},
                        call_id="tc_lint",
                    ),
                ],
            ),
            # 2) review: TASK_COMPLETE
            make_llm_response(content="<status>TASK_COMPLETE</status>"),
        ])

        graph = _build_graph(mock_sandbox_manager, event_bus, mock_llm)
        initial = create_initial_state(
            task="Build a hello world app",
            sandbox_id=SANDBOX_ID,
            session_id=SESSION_ID,
            max_iterations=5,
        )

        final = await graph.run(initial)

        assert final["status"] == "complete"
        assert final["iteration"] == 1
        assert "src/App.tsx" in final["files_written"]

        # Verify key events were emitted
        events = await collect_events(event_bus, SESSION_ID)
        event_types = [e.type for e in events]
        assert EventType.GRAPH_INITIALIZED in event_types
        assert EventType.AGENT_SPAWNED in event_types
        assert EventType.AGENT_COMPLETE in event_types

    @pytest.mark.asyncio
    async def test_two_iterations_with_revision(
        self, mock_sandbox_manager, event_bus
    ) -> None:
        """Review returns NEEDS_REVISION first, TASK_COMPLETE second.

        Node sequence: reason -> execute -> review -> reason -> execute -> review.
        Final iteration should be 2.
        """
        mock_llm = MockLLMClient(responses=[
            # Iteration 1: reason (tool call)
            make_llm_response(
                content="<plan>Write initial App.tsx</plan>",
                tool_calls=[
                    make_tool_call(
                        "write_file",
                        {"path": "src/App.tsx", "content": "v1"},
                        call_id="tc_1",
                    )
                ],
            ),
            # Iteration 1: review -> NEEDS_REVISION
            make_llm_response(content="<status>NEEDS_REVISION</status>\nNeeds styling."),
            # Iteration 2: reason (tool calls including build and lint)
            make_llm_response(
                content="<plan>Add styling and verify build</plan>",
                tool_calls=[
                    make_tool_call(
                        "write_file",
                        {"path": "src/App.tsx", "content": "v2 with styling"},
                        call_id="tc_2",
                    ),
                    make_tool_call(
                        "execute_command",
                        {"command": "npm run build"},
                        call_id="tc_3",
                    ),
                    make_tool_call(
                        "execute_command",
                        {"command": "npm run lint"},
                        call_id="tc_4",
                    ),
                ],
            ),
            # Iteration 2: review -> TASK_COMPLETE
            make_llm_response(content="<status>TASK_COMPLETE</status>"),
        ])

        graph = _build_graph(mock_sandbox_manager, event_bus, mock_llm)
        initial = create_initial_state(
            task="Build a styled app",
            sandbox_id=SANDBOX_ID,
            session_id=SESSION_ID,
            max_iterations=5,
        )

        final = await graph.run(initial)

        assert final["status"] == "complete"
        assert final["iteration"] == 2
        # LLM was called 4 times: reason, review, reason, review
        assert len(mock_llm.call_history) == 4
        # After NEEDS_REVISION review, graph injects a user handoff so the
        # next reason step has an actionable user turn.
        second_reason_messages = mock_llm.call_history[2]["messages"]
        last_message = second_reason_messages[-1]
        if isinstance(last_message, dict):
            role = str(last_message.get("role", ""))
            content = str(last_message.get("content", ""))
        else:
            role = str(getattr(last_message, "type", getattr(last_message, "role", "")))
            content = str(getattr(last_message, "content", ""))

        assert role in ("user", "human")
        assert "Continue implementing now based on the review above" in content

    @pytest.mark.asyncio
    async def test_completion_blocked_without_build_and_lint_verification(
        self, mock_sandbox_manager, event_bus
    ) -> None:
        """TASK_COMPLETE is rejected when files changed but build/lint wasn't verified."""
        mock_llm = MockLLMClient(responses=[
            make_llm_response(
                content="<plan>Write App.tsx</plan>",
                tool_calls=[
                    make_tool_call(
                        "write_file",
                        {"path": "src/App.tsx", "content": "v1"},
                        call_id="tc_1",
                    )
                ],
            ),
            make_llm_response(content="<status>TASK_COMPLETE</status>"),
            make_llm_response(
                content="<plan>Run build and lint now</plan>",
                tool_calls=[
                    make_tool_call(
                        "execute_command",
                        {"command": "npm run build"},
                        call_id="tc_2",
                    ),
                    make_tool_call(
                        "execute_command",
                        {"command": "npm run lint"},
                        call_id="tc_3",
                    ),
                ],
            ),
            make_llm_response(content="<status>TASK_COMPLETE</status>"),
        ])

        graph = _build_graph(mock_sandbox_manager, event_bus, mock_llm)
        initial = create_initial_state(
            task="Build a simple app",
            sandbox_id=SANDBOX_ID,
            session_id=SESSION_ID,
            max_iterations=5,
        )

        final = await graph.run(initial)

        assert final["status"] == "complete"
        assert final["iteration"] == 2
        assert final["build_verified"] is True
        assert final["lint_verified"] is True

    @pytest.mark.asyncio
    async def test_completion_blocked_after_post_build_shell_mutation(
        self, mock_sandbox_manager, event_bus
    ) -> None:
        """A mutating shell command after build invalidates completion evidence."""
        mock_llm = MockLLMClient(responses=[
            make_llm_response(
                content="<plan>Edit, build, lint, then shell mutate</plan>",
                tool_calls=[
                    make_tool_call(
                        "write_file",
                        {"path": "src/App.tsx", "content": "v1"},
                        call_id="tc_write",
                    ),
                    make_tool_call(
                        "execute_command",
                        {"command": "npm run build"},
                        call_id="tc_build",
                    ),
                    make_tool_call(
                        "execute_command",
                        {"command": "npm run lint"},
                        call_id="tc_lint",
                    ),
                    make_tool_call(
                        "execute_command",
                        {"command": "echo 'x' > src/App.tsx"},
                        call_id="tc_shell_mutate",
                    ),
                ],
            ),
            make_llm_response(content="<status>TASK_COMPLETE</status>"),
            make_llm_response(
                content="<plan>Re-run build and lint for verification</plan>",
                tool_calls=[
                    make_tool_call(
                        "execute_command",
                        {"command": "npm run build"},
                        call_id="tc_build_again",
                    ),
                    make_tool_call(
                        "execute_command",
                        {"command": "npm run lint"},
                        call_id="tc_lint_again",
                    ),
                ],
            ),
            make_llm_response(content="<status>TASK_COMPLETE</status>"),
        ])

        graph = _build_graph(mock_sandbox_manager, event_bus, mock_llm)
        initial = create_initial_state(
            task="Build and verify app",
            sandbox_id=SANDBOX_ID,
            session_id=SESSION_ID,
            max_iterations=5,
        )

        final = await graph.run(initial)

        assert final["status"] == "complete"
        assert final["iteration"] == 2
        assert final["build_verified"] is True

    @pytest.mark.asyncio
    async def test_max_iterations_terminates(
        self, mock_sandbox_manager, event_bus
    ) -> None:
        """Graph ends at max_iterations when review always returns NEEDS_REVISION."""
        max_iter = 2
        mock_llm = MockLLMClient(responses=[
            # Iteration 1: reason
            make_llm_response(
                content="<plan>Try something</plan>",
                tool_calls=[
                    make_tool_call(
                        "write_file", {"path": "src/App.tsx", "content": "v1"}, call_id="tc_1",
                    )
                ],
            ),
            # Iteration 1: review -> NEEDS_REVISION
            make_llm_response(content="<status>NEEDS_REVISION</status>"),
            # Iteration 2: reason
            make_llm_response(
                content="<plan>Try again</plan>",
                tool_calls=[
                    make_tool_call(
                        "write_file", {"path": "src/App.tsx", "content": "v2"}, call_id="tc_2",
                    )
                ],
            ),
            # Iteration 2: review -> NEEDS_REVISION (but max_iterations hit)
            make_llm_response(content="<status>NEEDS_REVISION</status>"),
        ])

        graph = _build_graph(mock_sandbox_manager, event_bus, mock_llm)
        initial = create_initial_state(
            task="Complex task",
            sandbox_id=SANDBOX_ID,
            session_id=SESSION_ID,
            max_iterations=max_iter,
        )

        final = await graph.run(initial)

        # Graph should end without error, iteration == max_iter
        assert final["iteration"] == max_iter
        # Status stays "reasoning" because review set it to "reasoning" and
        # _should_continue stops due to iteration limit
        assert final["status"] in ("reasoning", "complete")

    @pytest.mark.asyncio
    async def test_no_tool_calls_from_reason(
        self, mock_sandbox_manager, event_bus
    ) -> None:
        """Reason returns content only (no tool_calls). Execute handles gracefully."""
        mock_llm = MockLLMClient(responses=[
            # Reason: no tool calls
            make_llm_response(content="I think the task is already done."),
            # Review: TASK_COMPLETE
            make_llm_response(content="<status>TASK_COMPLETE</status>"),
        ])

        graph = _build_graph(mock_sandbox_manager, event_bus, mock_llm)
        initial = create_initial_state(
            task="Simple task",
            sandbox_id=SANDBOX_ID,
            session_id=SESSION_ID,
            max_iterations=5,
        )

        final = await graph.run(initial)

        assert final["status"] == "complete"
        assert final["files_written"] == []

    @pytest.mark.asyncio
    async def test_review_fallback_substring(
        self, mock_sandbox_manager, event_bus
    ) -> None:
        """Review response has no <status> tags but contains 'task_complete' substring.

        The fallback substring matching should still mark status as "complete".
        """
        mock_llm = MockLLMClient(responses=[
            make_llm_response(content="<plan>Write file</plan>"),
            # Review: no tags, but contains task_complete substring
            make_llm_response(
                content="Everything looks good. The result is task_complete."
            ),
        ])

        graph = _build_graph(mock_sandbox_manager, event_bus, mock_llm)
        initial = create_initial_state(
            task="Test fallback",
            sandbox_id=SANDBOX_ID,
            session_id=SESSION_ID,
            max_iterations=5,
        )

        final = await graph.run(initial)

        assert final["status"] == "complete"


# ---------------------------------------------------------------------------
# Unit Tests (using object.__new__() to skip graph compilation)
# ---------------------------------------------------------------------------


class TestShouldContinueRouting:
    """Unit tests for _should_continue routing logic."""

    def _make_graph(self) -> ReactGraph:
        graph = object.__new__(ReactGraph)
        return graph

    def test_returns_end_for_complete(self) -> None:
        graph = self._make_graph()
        result = graph._should_continue(
            {"status": "complete", "iteration": 1, "max_iterations": 5}
        )
        assert result == "end"

    def test_returns_end_for_failed(self) -> None:
        graph = self._make_graph()
        result = graph._should_continue(
            {"status": "failed", "iteration": 1, "max_iterations": 5}
        )
        assert result == "end"

    def test_returns_end_for_max_iterations(self) -> None:
        graph = self._make_graph()
        result = graph._should_continue(
            {"status": "reasoning", "iteration": 5, "max_iterations": 5, "session_id": "sess_test"}
        )
        assert result == "end"

    def test_returns_continue_when_not_done(self) -> None:
        graph = self._make_graph()
        result = graph._should_continue(
            {"status": "reasoning", "iteration": 2, "max_iterations": 5}
        )
        assert result == "continue"


class TestCommandClassification:
    def test_detects_mutating_shell_redirection(self) -> None:
        assert ReactGraph._is_potentially_mutating_command(
            "echo 'x' > src/App.tsx"
        )

    def test_treats_build_and_lint_as_non_mutating(self) -> None:
        assert not ReactGraph._is_potentially_mutating_command("npm run build")
        assert not ReactGraph._is_potentially_mutating_command("npm run lint")

    def test_treats_observational_find_as_non_mutating(self) -> None:
        assert not ReactGraph._is_potentially_mutating_command(
            "find src -type f"
        )
