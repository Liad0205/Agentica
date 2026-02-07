"""Unit + execution tests for the Parallel Hypothesis graph."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.hypothesis_graph import (
    HypothesisGraph,
    create_hypothesis_initial_state,
)
from config import settings
from events.types import EventType
from sandbox.docker_sandbox import CommandResult
from tests.conftest import (
    RoutingMockLLMClient,
    collect_events,
    make_llm_response,
    make_tool_call,
)


def _make_graph_for_unit_tests() -> HypothesisGraph:
    """Create a lightweight graph instance without compiling LangGraph."""
    graph = object.__new__(HypothesisGraph)
    graph.sandbox_manager = AsyncMock()
    graph.event_bus = AsyncMock()
    graph.llm_client = AsyncMock()
    graph.llm_client.metrics_collector = None
    graph.evaluator_model = "mock-evaluator"
    graph._session_start_time = time.monotonic()
    return graph


class TestTimeoutBudgeting:
    def test_allocate_solver_budget_default_range(self) -> None:
        graph = _make_graph_for_unit_tests()
        budget = graph._allocate_solver_budget_seconds()
        assert 20 <= budget <= 180

    def test_allocate_solver_budget_shrinks_near_session_timeout(self) -> None:
        graph = _make_graph_for_unit_tests()
        graph._session_start_time = time.monotonic() - (settings.agent_timeout_seconds - 14)

        budget = graph._allocate_solver_budget_seconds()
        assert budget <= 14

    def test_timeout_before_deadline_clips_default_timeout(self) -> None:
        deadline = time.monotonic() + 6
        timeout = HypothesisGraph._timeout_before_deadline(
            deadline,
            120,
            reserve_seconds=2,
            min_timeout=5,
        )

        assert timeout <= 6
        assert timeout >= 3


class TestEvaluationPromptShaping:
    def test_build_solutions_text_limits_file_content_payload(self) -> None:
        graph = _make_graph_for_unit_tests()
        files = {f"src/Comp{i}.tsx": "x" * 3000 for i in range(20)}

        solutions_text = graph._build_solutions_text(
            [
                {
                    "agent_id": "solver_1",
                    "persona": "clarity",
                    "build_success": True,
                    "build_output": "y" * 3000,
                    "lint_errors": 0,
                    "files": files,
                    "agent_summary": "z" * 1200,
                }
            ]
        )

        # The evaluator context should include content for only a bounded
        # number of files and mark truncation/omission explicitly.
        assert solutions_text.count("\n--- src/Comp") == 4
        assert "... (8 more files omitted)" in solutions_text
        assert "[... 16 additional files omitted from content]" in solutions_text
        assert "[truncated" in solutions_text


class TestEvaluationFallbacks:
    def test_build_fallback_scores_prioritize_build_success(self) -> None:
        graph = _make_graph_for_unit_tests()
        scores = graph._build_fallback_scores(
            [
                {
                    "agent_id": "solver_1",
                    "persona": "clarity",
                    "sandbox_id": "sb1",
                    "files": {"src/App.tsx": "ok"},
                    "build_success": True,
                    "build_output": "OK",
                    "lint_errors": 0,
                    "iterations_used": 3,
                    "agent_summary": "complete",
                },
                {
                    "agent_id": "solver_2",
                    "persona": "creative",
                    "sandbox_id": "sb2",
                    "files": {"src/App.tsx": "broken", "src/ui.tsx": "broken"},
                    "build_success": False,
                    "build_output": "error",
                    "lint_errors": 10,
                    "iterations_used": 3,
                    "agent_summary": "partial",
                },
            ]
        )

        by_agent = {score["agent_id"]: score for score in scores}
        assert by_agent["solver_1"]["total"] > by_agent["solver_2"]["total"]

    def test_build_fallback_scores_rewards_entrypoint_edits(self) -> None:
        graph = _make_graph_for_unit_tests()
        scores = graph._build_fallback_scores(
            [
                {
                    "agent_id": "solver_1",
                    "persona": "clarity",
                    "sandbox_id": "sb1",
                    "files": {"src/App.tsx": "ok", "src/components/A.tsx": "ok"},
                    "edited_files": ["src/components/A.tsx"],
                    "build_success": True,
                    "build_output": "OK",
                    "lint_errors": 0,
                    "iterations_used": 2,
                    "agent_summary": "complete",
                },
                {
                    "agent_id": "solver_2",
                    "persona": "completeness",
                    "sandbox_id": "sb2",
                    "files": {"src/App.tsx": "ok", "src/components/B.tsx": "ok"},
                    "edited_files": ["src/App.tsx", "src/components/B.tsx"],
                    "build_success": True,
                    "build_output": "OK",
                    "lint_errors": 0,
                    "iterations_used": 2,
                    "agent_summary": "complete",
                },
            ]
        )

        by_agent = {score["agent_id"]: score for score in scores}
        assert by_agent["solver_2"]["total"] > by_agent["solver_1"]["total"]

    def test_select_best_agent_from_scores_ignores_unknown_agent(self) -> None:
        graph = _make_graph_for_unit_tests()
        selected_agent, selected_index = graph._select_best_agent_from_scores(
            [
                {
                    "agent_id": "non_existent",
                    "build": 10,
                    "lint": 10,
                    "quality": 10,
                    "completeness": 10,
                    "ux": 10,
                    "total": 10.0,
                    "notes": "invalid id",
                },
                {
                    "agent_id": "solver_2",
                    "build": 8,
                    "lint": 8,
                    "quality": 8,
                    "completeness": 8,
                    "ux": 8,
                    "total": 8.0,
                    "notes": "valid",
                },
            ],
            [
                {
                    "agent_id": "solver_1",
                    "persona": "clarity",
                    "sandbox_id": "sb1",
                    "files": {},
                    "build_success": True,
                    "build_output": "",
                    "lint_errors": 0,
                    "iterations_used": 2,
                    "agent_summary": "",
                },
                {
                    "agent_id": "solver_2",
                    "persona": "creative",
                    "sandbox_id": "sb2",
                    "files": {},
                    "build_success": True,
                    "build_output": "",
                    "lint_errors": 0,
                    "iterations_used": 2,
                    "agent_summary": "",
                },
            ],
        )

        assert selected_agent == "solver_2"
        assert selected_index == 1

    @pytest.mark.asyncio
    async def test_parse_fallback_skips_pairwise_llm(self, monkeypatch) -> None:
        graph = _make_graph_for_unit_tests()
        monkeypatch.setattr(settings, "multi_round_evaluation_threshold", 0.5)

        graph.llm_client.call = AsyncMock(
            return_value=make_llm_response(content="not valid json")
        )
        pairwise_compare = AsyncMock(return_value="solver_2")
        graph._pairwise_compare = pairwise_compare

        state = create_hypothesis_initial_state(
            task="Build something",
            session_id=SESSION_ID,
            num_hypotheses=2,
        )
        state["status"] = "evaluating"
        state["hypothesis_results"] = [
            {
                "agent_id": "solver_1",
                "persona": "clarity",
                "sandbox_id": "sb_1",
                "files": {"src/App.tsx": "export default function App() { return null; }"},
                "build_success": True,
                "build_output": "OK",
                "lint_errors": 0,
                "iterations_used": 2,
                "agent_summary": "done",
            },
            {
                "agent_id": "solver_2",
                "persona": "creativity",
                "sandbox_id": "sb_2",
                "files": {"src/App.tsx": "export default function App() { return <div/>; }"},
                "build_success": True,
                "build_output": "OK",
                "lint_errors": 2,
                "iterations_used": 2,
                "agent_summary": "done",
            },
        ]

        result = await graph._evaluate(state)

        assert "deterministic fallback" in result["evaluation_reasoning"].lower()
        pairwise_compare.assert_not_awaited()

    def test_prefer_entrypoint_wired_winner_switches_within_threshold(self) -> None:
        graph = _make_graph_for_unit_tests()

        selected_agent, selected_index, reason = graph._prefer_entrypoint_wired_winner(
            selected_agent_id="solver_1",
            selected_index=0,
            scores=[
                {
                    "agent_id": "solver_1",
                    "build": 9,
                    "lint": 9,
                    "quality": 9,
                    "completeness": 8,
                    "ux": 7,
                    "total": 8.6,
                    "notes": "",
                },
                {
                    "agent_id": "solver_2",
                    "build": 9,
                    "lint": 8,
                    "quality": 8,
                    "completeness": 8,
                    "ux": 8,
                    "total": 8.3,
                    "notes": "",
                },
            ],
            results=[
                {
                    "agent_id": "solver_1",
                    "persona": "clarity",
                    "sandbox_id": "sb_1",
                    "files": {"src/App.tsx": "default"},
                    "edited_files": ["src/components/Card.tsx"],
                    "build_success": True,
                    "build_output": "OK",
                    "lint_errors": 0,
                    "iterations_used": 2,
                    "agent_summary": "done",
                },
                {
                    "agent_id": "solver_2",
                    "persona": "completeness",
                    "sandbox_id": "sb_2",
                    "files": {"src/App.tsx": "wired"},
                    "edited_files": ["src/App.tsx", "src/components/Card.tsx"],
                    "build_success": True,
                    "build_output": "OK",
                    "lint_errors": 0,
                    "iterations_used": 2,
                    "agent_summary": "done",
                },
            ],
        )

        assert selected_agent == "solver_2"
        assert selected_index == 1
        assert "src/App.tsx" in reason


# ---------------------------------------------------------------------------
# Graph Execution Tests
# ---------------------------------------------------------------------------

SESSION_ID = "sess_hyp_test"


def _evaluator_json(selected: str, scores: list[dict] | None = None) -> str:
    """Build a valid evaluator JSON response string."""
    if scores is None:
        scores = [{
            "agent_id": selected,
            "build": 9, "lint": 8, "quality": 9,
            "completeness": 9, "ux": 8, "total": 8.7,
            "notes": "Good solution",
        }]
    return json.dumps({
        "scores": scores,
        "selected": selected,
        "reasoning": "Selected best solution",
    })


class TestHypothesisGraphExecution:
    """Tests that compile the real HypothesisGraph and run it end-to-end."""

    @pytest.mark.asyncio
    async def test_happy_path_single_solver(
        self, mock_sandbox_manager, event_bus, monkeypatch
    ) -> None:
        """num_hypotheses=1. Solver completes, evaluator scores and selects.

        Final: status == "complete", selected_agent_id set, final_preview_url set.
        """
        # Disable synthesis to simplify
        monkeypatch.setattr(settings, "enable_hypothesis_synthesis", False)

        # Solver responses (agent_id = "solver_1")
        solver_reason = make_llm_response(
            content="<plan>Write App.tsx</plan>",
            tool_calls=[make_tool_call(
                "write_file",
                {"path": "src/App.tsx", "content": "export default function App() {}"},
                call_id="tc_solver",
            )],
        )
        solver_done = make_llm_response(content="TASK_COMPLETE")

        # Evaluator response
        evaluator_response = make_llm_response(
            content=_evaluator_json("solver_1")
        )

        mock_llm = RoutingMockLLMClient(response_map={
            "solver_1": [solver_reason, solver_done],
            "default": [evaluator_response],
        })

        # Sandbox side effects
        async def create_sandbox_side_effect(sandbox_id: str):
            return MagicMock(
                sandbox_id=sandbox_id,
                container_id=f"container_{sandbox_id}",
                port=5173,
                workspace_path="/workspace",
                status="running",
            )

        mock_sandbox_manager.create_sandbox = AsyncMock(side_effect=create_sandbox_side_effect)

        async def execute_command_side_effect(sandbox_id: str, command: str, timeout: int = 120):
            if "find" in command:
                return CommandResult(
                    stdout="./src/App.tsx\n", stderr="", exit_code=0, timed_out=False
                )
            if "eslint" in command:
                return CommandResult(
                    stdout="[]", stderr="", exit_code=0, timed_out=False
                )
            return CommandResult(stdout="OK", stderr="", exit_code=0, timed_out=False)

        mock_sandbox_manager.execute_command = AsyncMock(side_effect=execute_command_side_effect)
        mock_sandbox_manager.read_file = AsyncMock(
            return_value="export default function App() {}"
        )
        mock_sandbox_manager.start_dev_server = AsyncMock(
            return_value="http://localhost:5173"
        )

        graph = HypothesisGraph(
            sandbox_manager=mock_sandbox_manager,
            event_bus=event_bus,
            llm_client=mock_llm,
        )

        initial = create_hypothesis_initial_state(
            task="Build a hello world app",
            session_id=SESSION_ID,
            num_hypotheses=1,
        )

        final = await graph.run(initial)

        assert final["status"] == "complete"
        assert final["selected_agent_id"] == "solver_1"
        assert final["final_preview_url"] == "http://localhost:5173"

        # Verify key events
        events = await collect_events(event_bus, SESSION_ID)
        event_types = [e.type for e in events]
        assert EventType.GRAPH_INITIALIZED in event_types
        assert EventType.EVALUATION_STARTED in event_types
        assert EventType.EVALUATION_RESULT in event_types
        assert EventType.PREVIEW_READY in event_types

    @pytest.mark.asyncio
    async def test_two_solvers_evaluator_picks_winner(
        self, mock_sandbox_manager, event_bus, monkeypatch
    ) -> None:
        """num_hypotheses=2. Both solve, evaluator picks solver_2.

        Loser sandbox should be destroyed. hypothesis_results has 2 entries.
        """
        monkeypatch.setattr(settings, "enable_hypothesis_synthesis", False)
        # Disable multi-round evaluation to avoid pairwise comparison
        monkeypatch.setattr(settings, "multi_round_evaluation_threshold", 0)

        # Solver 1
        solver1_reason = make_llm_response(
            content="<plan>Basic approach</plan>",
            tool_calls=[make_tool_call(
                "write_file",
                {"path": "src/App.tsx", "content": "v1"},
                call_id="tc_s1",
            )],
        )
        solver1_done = make_llm_response(content="TASK_COMPLETE")

        # Solver 2
        solver2_reason = make_llm_response(
            content="<plan>Creative approach</plan>",
            tool_calls=[make_tool_call(
                "write_file",
                {"path": "src/App.tsx", "content": "v2"},
                call_id="tc_s2",
            )],
        )
        solver2_done = make_llm_response(content="TASK_COMPLETE")

        # Evaluator picks solver_2
        evaluator_response = make_llm_response(
            content=_evaluator_json("solver_2", scores=[
                {
                    "agent_id": "solver_1",
                    "build": 8, "lint": 7, "quality": 7,
                    "completeness": 7, "ux": 6, "total": 7.1,
                    "notes": "Basic",
                },
                {
                    "agent_id": "solver_2",
                    "build": 9, "lint": 9, "quality": 9,
                    "completeness": 9, "ux": 8, "total": 8.8,
                    "notes": "Excellent",
                },
            ])
        )

        mock_llm = RoutingMockLLMClient(response_map={
            "solver_1": [solver1_reason, solver1_done],
            "solver_2": [solver2_reason, solver2_done],
            "default": [evaluator_response],
        })

        async def create_sandbox_side_effect(sandbox_id: str):
            return MagicMock(
                sandbox_id=sandbox_id,
                container_id=f"container_{sandbox_id}",
                port=5173,
                workspace_path="/workspace",
                status="running",
            )

        mock_sandbox_manager.create_sandbox = AsyncMock(side_effect=create_sandbox_side_effect)

        async def execute_command_side_effect(sandbox_id: str, command: str, timeout: int = 120):
            if "find" in command:
                return CommandResult(
                    stdout="./src/App.tsx\n", stderr="", exit_code=0, timed_out=False
                )
            if "eslint" in command:
                return CommandResult(stdout="[]", stderr="", exit_code=0, timed_out=False)
            return CommandResult(stdout="OK", stderr="", exit_code=0, timed_out=False)

        mock_sandbox_manager.execute_command = AsyncMock(side_effect=execute_command_side_effect)
        mock_sandbox_manager.read_file = AsyncMock(return_value="code")
        mock_sandbox_manager.start_dev_server = AsyncMock(return_value="http://localhost:5173")

        graph = HypothesisGraph(
            sandbox_manager=mock_sandbox_manager,
            event_bus=event_bus,
            llm_client=mock_llm,
        )

        initial = create_hypothesis_initial_state(
            task="Build an app",
            session_id=SESSION_ID,
            num_hypotheses=2,
        )

        final = await graph.run(initial)

        assert final["status"] == "complete"
        assert final["selected_agent_id"] == "solver_2"
        assert len(final["hypothesis_results"]) == 2

        # Verify loser sandbox was destroyed
        destroy_calls = mock_sandbox_manager.destroy_sandbox.await_args_list
        destroyed_ids = [call.args[0] for call in destroy_calls]
        # The loser (solver_1) sandbox should be destroyed
        assert any("solver1" in sid for sid in destroyed_ids)

    @pytest.mark.asyncio
    async def test_all_solvers_fail_terminates_session(
        self, mock_sandbox_manager, event_bus, monkeypatch
    ) -> None:
        """npm build returns exit_code=1 for all solvers.

        The solver retries (max 2 retries), build still fails each time.
        Evaluator records diagnostics and the graph terminates as failed.
        """
        monkeypatch.setattr(settings, "enable_hypothesis_synthesis", False)

        # Single solver that produces code but build fails.
        # Provide enough responses for initial attempt + 2 retries.
        solver_reason = make_llm_response(
            content="<plan>Write code</plan>",
            tool_calls=[make_tool_call(
                "write_file",
                {"path": "src/App.tsx", "content": "broken code"},
                call_id="tc_s1",
            )],
        )
        solver_done = make_llm_response(content="TASK_COMPLETE")
        # Retry responses: solver tries to fix, then completes
        solver_retry1 = make_llm_response(content="TASK_COMPLETE fixing errors")
        solver_retry2 = make_llm_response(content="TASK_COMPLETE fixing errors")

        mock_llm = RoutingMockLLMClient(response_map={
            "solver_1": [solver_reason, solver_done, solver_retry1, solver_retry2],
            "default": [],  # Evaluator won't be called for all-failed fallback
        })

        async def create_sandbox_side_effect(sandbox_id: str):
            return MagicMock(
                sandbox_id=sandbox_id,
                container_id=f"container_{sandbox_id}",
                port=5173,
                workspace_path="/workspace",
                status="running",
            )

        mock_sandbox_manager.create_sandbox = AsyncMock(side_effect=create_sandbox_side_effect)

        async def execute_command_side_effect(sandbox_id: str, command: str, timeout: int = 120):
            if "find" in command:
                return CommandResult(
                    stdout="./src/App.tsx\n", stderr="", exit_code=0, timed_out=False
                )
            if "eslint" in command:
                return CommandResult(stdout="[]", stderr="", exit_code=0, timed_out=False)
            if "build" in command:
                return CommandResult(
                    stdout="", stderr="Build failed: type errors",
                    exit_code=1, timed_out=False
                )
            return CommandResult(stdout="OK", stderr="", exit_code=0, timed_out=False)

        mock_sandbox_manager.execute_command = AsyncMock(side_effect=execute_command_side_effect)
        mock_sandbox_manager.read_file = AsyncMock(return_value="broken code")
        mock_sandbox_manager.start_dev_server = AsyncMock(return_value="http://localhost:5173")

        graph = HypothesisGraph(
            sandbox_manager=mock_sandbox_manager,
            event_bus=event_bus,
            llm_client=mock_llm,
        )

        initial = create_hypothesis_initial_state(
            task="Build an app",
            session_id=SESSION_ID,
            num_hypotheses=1,
        )

        final = await graph.run(initial)

        # Graph should terminate as failed when all solvers fail build/lint gates.
        assert final["status"] == "failed"
        assert len(final["hypothesis_results"]) == 1
        # Evaluator records best candidate for diagnostics but does not finalize it.
        assert final["selected_agent_id"] == "solver_1"

    def test_should_synthesize_routing(self, monkeypatch) -> None:
        """Unit test: returns 'finalize' when synthesis disabled or no reasoning,
        'synthesize' when both present.
        """
        graph = _make_graph_for_unit_tests()

        # Disabled synthesis -> finalize
        monkeypatch.setattr(settings, "enable_hypothesis_synthesis", False)
        assert graph._should_synthesize(
            {"evaluation_reasoning": "Some reasoning"}
        ) == "finalize"

        # Enabled but no reasoning -> finalize
        monkeypatch.setattr(settings, "enable_hypothesis_synthesis", True)
        assert graph._should_synthesize(
            {"evaluation_reasoning": ""}
        ) == "finalize"

        # Enabled with reasoning -> synthesize
        assert graph._should_synthesize(
            {"evaluation_reasoning": "Improvements: better error handling"}
        ) == "synthesize"

        # Explicit no-improvement language should not trigger synthesis
        assert graph._should_synthesize(
            {"evaluation_reasoning": "No improvements needed. This solution is already good as-is."}
        ) == "finalize"

    def test_route_after_evaluate_ends_on_failed_status(self, monkeypatch) -> None:
        """Failed evaluator status should terminate the graph before finalize."""
        graph = _make_graph_for_unit_tests()
        monkeypatch.setattr(settings, "enable_hypothesis_synthesis", True)
        assert graph._route_after_evaluate({"status": "failed"}) == "end"

    @pytest.mark.asyncio
    async def test_finalize_starts_preview(
        self, mock_sandbox_manager, event_bus, monkeypatch
    ) -> None:
        """After evaluation, finalize calls start_dev_server and emits PREVIEW_READY.

        Uses a single solver for simplicity.
        """
        monkeypatch.setattr(settings, "enable_hypothesis_synthesis", False)

        solver_reason = make_llm_response(
            content="<plan>Write code</plan>",
            tool_calls=[make_tool_call(
                "write_file",
                {"path": "src/App.tsx", "content": "app code"},
                call_id="tc_s1",
            )],
        )
        solver_done = make_llm_response(content="TASK_COMPLETE")

        evaluator_response = make_llm_response(
            content=_evaluator_json("solver_1")
        )

        mock_llm = RoutingMockLLMClient(response_map={
            "solver_1": [solver_reason, solver_done],
            "default": [evaluator_response],
        })

        async def create_sandbox_side_effect(sandbox_id: str):
            return MagicMock(
                sandbox_id=sandbox_id,
                container_id=f"container_{sandbox_id}",
                port=5173,
                workspace_path="/workspace",
                status="running",
            )

        mock_sandbox_manager.create_sandbox = AsyncMock(side_effect=create_sandbox_side_effect)

        async def execute_command_side_effect(sandbox_id: str, command: str, timeout: int = 120):
            if "find" in command:
                return CommandResult(
                    stdout="./src/App.tsx\n", stderr="", exit_code=0, timed_out=False
                )
            if "eslint" in command:
                return CommandResult(stdout="[]", stderr="", exit_code=0, timed_out=False)
            return CommandResult(stdout="OK", stderr="", exit_code=0, timed_out=False)

        mock_sandbox_manager.execute_command = AsyncMock(side_effect=execute_command_side_effect)
        mock_sandbox_manager.read_file = AsyncMock(return_value="app code")
        mock_sandbox_manager.start_dev_server = AsyncMock(return_value="http://localhost:5173")

        graph = HypothesisGraph(
            sandbox_manager=mock_sandbox_manager,
            event_bus=event_bus,
            llm_client=mock_llm,
        )

        initial = create_hypothesis_initial_state(
            task="Build an app",
            session_id=SESSION_ID,
            num_hypotheses=1,
        )

        final = await graph.run(initial)

        assert final["status"] == "complete"
        assert final["final_preview_url"] == "http://localhost:5173"

        # Verify start_dev_server was called
        mock_sandbox_manager.start_dev_server.assert_awaited()

        # Verify PREVIEW_READY event
        events = await collect_events(event_bus, SESSION_ID)
        event_types = [e.type for e in events]
        assert EventType.PREVIEW_READY in event_types
