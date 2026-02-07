"""Unit + execution tests for the Task Decomposition graph."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.decomposition_graph import (
    DecompositionGraph,
    create_decomposition_initial_state,
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


def _make_graph_for_unit_tests() -> DecompositionGraph:
    """Create a lightweight graph instance without compiling LangGraph."""
    graph = object.__new__(DecompositionGraph)
    graph.sandbox_manager = AsyncMock()
    graph.event_bus = AsyncMock()
    graph.llm_client = AsyncMock()
    graph.llm_client.metrics_collector = None
    graph.sub_agent_temperature = 0.7
    graph._session_start_time = 0.0
    return graph


class TestDependencyNormalization:
    def test_normalize_dependency_map_filters_invalid_entries(self) -> None:
        graph = _make_graph_for_unit_tests()

        raw = {
            "react": "^19.0.0",
            "": "^1.0.0",
            "lodash": 123,
            10: "^2.0.0",
            "zod": " ^3.23.0 ",
        }

        normalized = graph._normalize_dependency_map(raw)
        assert normalized == {
            "react": "^19.0.0",
            "zod": "^3.23.0",
        }


class TestSubtaskNormalization:
    def test_validate_injects_app_assembly_subtask_when_missing(self) -> None:
        graph = _make_graph_for_unit_tests()
        subtasks, error = graph._validate_and_normalize_subtasks(
            [
                {
                    "id": "subtask_1",
                    "description": "Build dashboard widgets",
                    "files_responsible": ["src/components/Dashboard.tsx"],
                    "dependencies": [],
                    "complexity": "medium",
                },
                {
                    "id": "subtask_2",
                    "description": "Build data hooks",
                    "files_responsible": ["src/hooks/useDashboard.ts"],
                    "dependencies": [],
                    "complexity": "medium",
                },
            ]
        )

        assert error is None
        assert len(subtasks) == 3
        app_subtask = next(
            subtask for subtask in subtasks if "src/App.tsx" in subtask["files_responsible"]
        )
        assert app_subtask["id"] == "subtask_app_assembly"
        assert app_subtask["dependencies"] == ["subtask_1", "subtask_2"]

    def test_validate_app_subtask_depends_on_peer_subtasks_when_safe(self) -> None:
        graph = _make_graph_for_unit_tests()
        subtasks, error = graph._validate_and_normalize_subtasks(
            [
                {
                    "id": "subtask_1",
                    "description": "Create feature components",
                    "files_responsible": ["src/components/Feature.tsx"],
                    "dependencies": [],
                    "complexity": "low",
                },
                {
                    "id": "subtask_2",
                    "description": "Assemble app shell",
                    "files_responsible": ["src/App.tsx"],
                    "dependencies": [],
                    "complexity": "low",
                },
            ]
        )

        assert error is None
        app_subtask = next(subtask for subtask in subtasks if subtask["id"] == "subtask_2")
        assert app_subtask["dependencies"] == ["subtask_1"]


class TestLayerRouting:
    def test_route_after_orchestrate_ends_on_failure(self) -> None:
        graph = _make_graph_for_unit_tests()
        route = graph._route_after_orchestrate({"status": "failed"})
        assert route == "end"

    def test_route_after_orchestrate_dispatches_when_layers_present(self) -> None:
        graph = _make_graph_for_unit_tests()
        route = graph._route_after_orchestrate(
            {
                "status": "executing",
                "dependency_layers": [[{"id": "subtask_1"}]],
            }
        )
        assert route == "dispatch"

    def test_fan_out_current_layer_dispatches_only_current_layer(self) -> None:
        graph = _make_graph_for_unit_tests()
        sends = graph._fan_out_current_layer(
            {
                "session_id": "sess_123",
                "subtasks": [
                    {"id": "subtask_1"},
                    {"id": "subtask_2"},
                    {"id": "subtask_3"},
                ],
                "dependency_layers": [
                    [{"id": "subtask_1"}, {"id": "subtask_2"}],
                    [{"id": "subtask_3"}],
                ],
                "current_layer_index": 0,
                "additional_dependencies": [],
                "shared_types": "",
                "integration_sandbox_id": "sandbox_integration",
            }
        )

        assert len(sends) == 2
        ids = [send.arg["subtask"]["id"] for send in sends]
        assert ids == ["subtask_1", "subtask_2"]
        assert all(send.arg["dependency_layer"] == 0 for send in sends)
        assert all(send.arg["total_layers"] == 2 for send in sends)

    async def test_advance_layer_increments_index(self) -> None:
        graph = _make_graph_for_unit_tests()
        result = await graph._advance_layer({"current_layer_index": 1})
        assert result == {"current_layer_index": 2}

    async def test_advance_layer_syncs_completed_current_layer_results(self) -> None:
        graph = _make_graph_for_unit_tests()
        graph._merge_successful_results_into_integration = AsyncMock(
            return_value={
                "merged_files": ["src/Foo.tsx"],
                "conflicts_resolved": 0,
                "dependencies_added": 1,
                "dependency_conflicts": [],
                "type_blocks_added": 1,
            }
        )
        graph.sandbox_manager.read_file = AsyncMock(return_value="export type Foo = string;\n")

        result = await graph._advance_layer(
            {
                "current_layer_index": 0,
                "dependency_layers": [
                    [{"id": "subtask_1"}],
                    [{"id": "subtask_2"}],
                ],
                "session_id": "sess_123",
                "integration_sandbox_id": "sandbox_integration",
                "synced_subtask_ids": [],
                "shared_types": "",
                "subtask_results": [
                    {
                        "subtask_id": "subtask_1",
                        "status": "complete",
                        "agent_id": "agent_subtask_1",
                    },
                    {
                        "subtask_id": "subtask_2",
                        "status": "complete",
                        "agent_id": "agent_subtask_2",
                    },
                ],
            }
        )

        assert result["current_layer_index"] == 1
        assert result["synced_subtask_ids"] == ["subtask_1"]
        assert result["shared_types"] == "export type Foo = string;\n"
        graph._merge_successful_results_into_integration.assert_awaited_once()
        merge_call = graph._merge_successful_results_into_integration.await_args
        merged_subtasks = merge_call.kwargs["successful_results"]
        assert len(merged_subtasks) == 1
        assert merged_subtasks[0]["subtask_id"] == "subtask_1"

    async def test_advance_layer_skips_sync_for_already_synced_subtasks(self) -> None:
        graph = _make_graph_for_unit_tests()
        graph._merge_successful_results_into_integration = AsyncMock()
        graph.sandbox_manager.read_file = AsyncMock(return_value="export type Foo = string;\n")

        result = await graph._advance_layer(
            {
                "current_layer_index": 0,
                "dependency_layers": [[{"id": "subtask_1"}]],
                "session_id": "sess_123",
                "integration_sandbox_id": "sandbox_integration",
                "synced_subtask_ids": ["subtask_1"],
                "shared_types": "export type Foo = string;\n",
                "subtask_results": [
                    {
                        "subtask_id": "subtask_1",
                        "status": "complete",
                        "agent_id": "agent_subtask_1",
                    }
                ],
            }
        )

        assert result == {"current_layer_index": 1}
        graph._merge_successful_results_into_integration.assert_not_awaited()

    def test_route_after_layer(self) -> None:
        graph = _make_graph_for_unit_tests()
        assert (
            graph._route_after_layer(
                {
                    "dependency_layers": [[{"id": "subtask_1"}], [{"id": "subtask_2"}]],
                    "current_layer_index": 1,
                }
            )
            == "dispatch"
        )
        assert (
            graph._route_after_layer(
                {
                    "dependency_layers": [[{"id": "subtask_1"}], [{"id": "subtask_2"}]],
                    "current_layer_index": 2,
                }
            )
            == "aggregate"
        )

    def test_route_after_aggregate(self) -> None:
        graph = _make_graph_for_unit_tests()
        assert graph._route_after_aggregate({"status": "failed"}) == "end"
        assert graph._route_after_aggregate({"status": "integrating"}) == "integrate"


class TestDependencyCollection:
    async def test_collect_dependency_requirements_reads_package_json(self) -> None:
        graph = _make_graph_for_unit_tests()
        graph.sandbox_manager.read_file = AsyncMock(
            return_value=json.dumps(
                {
                    "dependencies": {"react-query": "^5.0.0"},
                    "devDependencies": {"@types/node": "^20.0.0"},
                }
            )
        )

        dependencies, dev_dependencies = await graph._collect_dependency_requirements(
            "sandbox_test"
        )

        assert dependencies == {"react-query": "^5.0.0"}
        assert dev_dependencies == {"@types/node": "^20.0.0"}

    async def test_collect_dependency_requirements_handles_missing_package_json(self) -> None:
        graph = _make_graph_for_unit_tests()
        graph.sandbox_manager.read_file = AsyncMock(side_effect=FileNotFoundError())

        dependencies, dev_dependencies = await graph._collect_dependency_requirements(
            "sandbox_test"
        )

        assert dependencies == {}
        assert dev_dependencies == {}


class TestAggregationSync:
    async def test_aggregate_merges_only_unsynced_results(self) -> None:
        graph = _make_graph_for_unit_tests()
        graph._merge_successful_results_into_integration = AsyncMock(
            return_value={
                "merged_files": ["src/New.tsx"],
                "conflicts_resolved": 0,
                "dependencies_added": 0,
                "dependency_conflicts": [],
                "type_blocks_added": 0,
            }
        )

        result = await graph._aggregate(
            {
                "session_id": "sess_123",
                "integration_sandbox_id": "sandbox_integration",
                "subtask_results": [
                    {
                        "subtask_id": "subtask_1",
                        "status": "complete",
                        "agent_id": "agent_subtask_1",
                    },
                    {
                        "subtask_id": "subtask_2",
                        "status": "complete",
                        "agent_id": "agent_subtask_2",
                    },
                ],
                "synced_subtask_ids": ["subtask_1"],
                "shared_types": "",
            }
        )

        assert result["status"] == "integrating"
        graph._merge_successful_results_into_integration.assert_awaited_once()
        merge_call = graph._merge_successful_results_into_integration.await_args
        merged_subtasks = merge_call.kwargs["successful_results"]
        assert len(merged_subtasks) == 1
        assert merged_subtasks[0]["subtask_id"] == "subtask_2"


class TestIntegrationEntrypointGuard:
    async def test_integration_review_retries_when_app_entrypoint_not_edited(self) -> None:
        graph = _make_graph_for_unit_tests()
        graph._attempt_entrypoint_fix = AsyncMock(return_value=False)
        graph.sandbox_manager.execute_command = AsyncMock(
            return_value=CommandResult(
                stdout="ok",
                stderr="",
                exit_code=0,
                timed_out=False,
            )
        )

        result = await graph._integration_review(
            {
                "session_id": "sess_123",
                "integration_sandbox_id": "sandbox_integration",
                "integration_retries": 0,
                "subtasks": [
                    {
                        "id": "subtask_1",
                        "files_responsible": ["src/components/Card.tsx"],
                    },
                    {
                        "id": "subtask_2",
                        "files_responsible": ["src/App.tsx"],
                    },
                ],
                "subtask_results": [
                    {
                        "subtask_id": "subtask_1",
                        "status": "complete",
                        "edited_files": ["src/components/Card.tsx"],
                    },
                    {
                        "subtask_id": "subtask_2",
                        "status": "complete",
                        "edited_files": [],
                    },
                ],
            }
        )

        assert result["status"] == "integrating"
        assert result["integration_retries"] == 1
        assert "src/App.tsx" in (result.get("error_message") or "")
        graph._attempt_entrypoint_fix.assert_awaited_once()


class TestDependencySync:
    async def test_sync_integration_dependencies_unions_new_packages(self) -> None:
        graph = _make_graph_for_unit_tests()
        graph.sandbox_manager.read_file = AsyncMock(
            return_value=json.dumps(
                {
                    "name": "app",
                    "dependencies": {"react": "^19.0.0"},
                    "devDependencies": {"typescript": "^5.0.0"},
                }
            )
        )
        graph.sandbox_manager.write_file = AsyncMock()
        graph.event_bus.publish = AsyncMock()

        successful_results = [
            {
                "agent_id": "agent_subtask_1",
                "dependencies": {"zustand": "^5.0.0"},
                "dev_dependencies": {"vitest": "^2.0.0"},
            }
        ]

        result = await graph._sync_integration_dependencies(
            integration_sandbox_id="sandbox_integration",
            successful_results=successful_results,
            session_id="sess_123",
            aggregator_id="aggregator_123",
        )

        assert result["dependencies_added"] == 2
        assert result["dependency_conflicts"] == []
        graph.sandbox_manager.write_file.assert_awaited_once()

        written_payload = graph.sandbox_manager.write_file.await_args.args[2]
        parsed = json.loads(written_payload)
        assert parsed["dependencies"]["react"] == "^19.0.0"
        assert parsed["dependencies"]["zustand"] == "^5.0.0"
        assert parsed["devDependencies"]["typescript"] == "^5.0.0"
        assert parsed["devDependencies"]["vitest"] == "^2.0.0"

    async def test_sync_integration_dependencies_keeps_existing_on_conflict(self) -> None:
        graph = _make_graph_for_unit_tests()
        graph.sandbox_manager.read_file = AsyncMock(
            return_value=json.dumps(
                {
                    "name": "app",
                    "dependencies": {"zod": "^3.20.0"},
                }
            )
        )
        graph.sandbox_manager.write_file = AsyncMock()
        graph.event_bus.publish = AsyncMock()

        successful_results = [
            {
                "agent_id": "agent_subtask_1",
                "dependencies": {"zod": "^3.23.0"},
                "dev_dependencies": {},
            }
        ]

        result = await graph._sync_integration_dependencies(
            integration_sandbox_id="sandbox_integration",
            successful_results=successful_results,
            session_id="sess_123",
            aggregator_id="aggregator_123",
        )

        assert result["dependencies_added"] == 0
        assert len(result["dependency_conflicts"]) == 1
        graph.sandbox_manager.write_file.assert_not_awaited()


class TestSubtaskBuildStatus:
    async def test_run_subtask_react_loop_marks_failed_when_build_fails(self) -> None:
        graph = _make_graph_for_unit_tests()

        response = MagicMock()
        response.content = "<plan>done</plan>\nTASK_COMPLETE"
        response.tool_calls = []
        graph.llm_client.call = AsyncMock(return_value=response)

        graph.sandbox_manager.execute_command = AsyncMock(
            return_value=CommandResult(
                stdout="",
                stderr="build failed",
                exit_code=1,
                timed_out=False,
            )
        )

        result = await graph._run_subtask_react_loop(
            subtask={
                "description": "Build a component",
                "files_responsible": ["src/App.tsx"],
                "complexity": "low",
            },
            sandbox_id="sandbox_subtask",
            session_id="sess_123",
            agent_id="agent_subtask_1",
            agent_role="Sub-agent 1",
            shared_types="",
        )

        assert result["build_success"] is False
        assert result["status"] == "failed"


class TestSharedTypeSync:
    def test_extract_type_additions_filters_existing_blocks(self) -> None:
        graph = _make_graph_for_unit_tests()

        shared_types = """export interface Item {
  id: string;
}
"""
        subtask_types = """export interface Item {
  id: string;
}

export interface User {
  name: string;
}
"""

        additions = graph._extract_type_additions(shared_types, subtask_types)
        assert "export interface User" in additions
        assert "export interface Item" not in additions

    def test_merge_shared_types_deduplicates_blocks(self) -> None:
        graph = _make_graph_for_unit_tests()

        base_types = """export interface Item {
  id: string;
}
"""
        successful_results = [
            {"agent_id": "agent_b", "type_additions": "export type ItemId = string;"},
            {"agent_id": "agent_a", "type_additions": "export type ItemId = string;"},
            {"agent_id": "agent_c", "type_additions": "export interface User { name: string; }"},
        ]

        merged, added_count = graph._merge_shared_types(base_types, successful_results)
        assert added_count == 2
        assert "export interface Item" in merged
        assert merged.count("export type ItemId = string;") == 1
        assert "export interface User" in merged

    async def test_sync_integration_shared_types_writes_merged_types(self) -> None:
        graph = _make_graph_for_unit_tests()
        graph.sandbox_manager.read_file = AsyncMock(
            return_value="export interface Item { id: string; }\n"
        )
        graph.sandbox_manager.write_file = AsyncMock()
        graph.event_bus.publish = AsyncMock()

        result = await graph._sync_integration_shared_types(
            integration_sandbox_id="sandbox_integration",
            shared_types="export interface Item { id: string; }\n",
            successful_results=[
                {"agent_id": "agent_1", "type_additions": "export type ItemId = string;"},
            ],
            session_id="sess_123",
            aggregator_id="aggregator_123",
        )

        assert result["type_blocks_added"] == 1
        graph.sandbox_manager.write_file.assert_awaited_once()
        write_args = graph.sandbox_manager.write_file.await_args.args
        assert write_args[0] == "sandbox_integration"
        assert write_args[1] == "src/types.ts"
        assert "export type ItemId = string;" in write_args[2]


class TestTimeoutBudgeting:
    def test_allocate_subtask_budget_defaults_from_layer_budget(self) -> None:
        graph = _make_graph_for_unit_tests()

        budget = graph._allocate_subtask_budget_seconds(
            {
                "dependency_layer": 0,
                "total_layers": 3,
            }
        )

        assert 20 <= budget <= 180

    def test_allocate_subtask_budget_shrinks_near_session_timeout(self) -> None:
        graph = _make_graph_for_unit_tests()
        graph._session_start_time = time.monotonic() - (settings.agent_timeout_seconds - 18)

        budget = graph._allocate_subtask_budget_seconds(
            {
                "dependency_layer": 0,
                "total_layers": 1,
            }
        )

        assert budget <= 18

    def test_timeout_before_deadline_clips_default_timeout(self) -> None:
        deadline = time.monotonic() + 7
        timeout = DecompositionGraph._timeout_before_deadline(
            deadline,
            120,
            reserve_seconds=3,
            min_timeout=5,
        )

        assert timeout <= 7
        assert timeout >= 3


# ---------------------------------------------------------------------------
# Graph Execution Tests
# ---------------------------------------------------------------------------

SESSION_ID = "sess_decomp_test"
INTEGRATION_SANDBOX_ID = "sandbox_integration_test"


def _orchestrator_json(subtasks: list[dict], additional_deps: list[str] | None = None) -> str:
    """Build a valid JSON orchestrator response string."""
    return json.dumps({
        "additional_dependencies": additional_deps or [],
        "shared_types": "export interface Item { id: string; }",
        "subtasks": subtasks,
    })


class TestDecompositionGraphExecution:
    """Tests that compile the real DecompositionGraph and run it end-to-end."""

    @pytest.mark.asyncio
    async def test_happy_path_single_subtask(
        self, mock_sandbox_manager, event_bus
    ) -> None:
        """Orchestrator returns 1 subtask, subtask completes, aggregate merges,
        integration build succeeds.

        Final: status == "complete", final_build_success == True,
               subtask_results has 1 entry.
        """
        subtask_def = {
            "id": "subtask_1",
            "description": "Build the App component",
            "files_responsible": ["src/App.tsx"],
            "dependencies": [],
            "complexity": "low",
        }

        # --- Mock LLM responses keyed by agent_id ---
        # Orchestrator calls use agent_id like "orchestrator_decomp_test"
        orchestrator_response = make_llm_response(
            content=_orchestrator_json([subtask_def])
        )
        # Subtask agent calls use agent_id "agent_subtask_1"
        subtask_reason = make_llm_response(
            content="<plan>Write App.tsx</plan>",
            tool_calls=[make_tool_call(
                "write_file",
                {"path": "src/App.tsx", "content": "export default function App() {}"},
                call_id="tc_sub_1",
            )],
        )
        subtask_complete = make_llm_response(
            content="<plan>done</plan>\nTASK_COMPLETE"
        )
        # Integration review calls use agent_id "integration_decomp_test"
        # (no LLM call needed if build succeeds on first try)

        mock_llm = RoutingMockLLMClient(response_map={
            "default": [orchestrator_response],
            "agent_subtask_1": [subtask_reason, subtask_complete],
        })

        # Sandbox manager: create_sandbox should return unique mocks per call
        sandbox_mocks = {}

        async def create_sandbox_side_effect(sandbox_id: str):
            mock = MagicMock(
                sandbox_id=sandbox_id,
                container_id=f"container_{sandbox_id}",
                port=5173,
                workspace_path="/workspace",
                status="running",
            )
            sandbox_mocks[sandbox_id] = mock
            return mock

        mock_sandbox_manager.create_sandbox = AsyncMock(side_effect=create_sandbox_side_effect)

        # execute_command: handle npm install, npm run build, find commands
        async def execute_command_side_effect(sandbox_id: str, command: str, timeout: int = 120):
            if "find" in command:
                return CommandResult(
                    stdout="./src/App.tsx\n", stderr="", exit_code=0, timed_out=False
                )
            # npm install, npm run build => success
            return CommandResult(stdout="OK", stderr="", exit_code=0, timed_out=False)

        mock_sandbox_manager.execute_command = AsyncMock(side_effect=execute_command_side_effect)
        mock_sandbox_manager.read_file = AsyncMock(
            return_value="export default function App() {}"
        )

        graph = DecompositionGraph(
            sandbox_manager=mock_sandbox_manager,
            event_bus=event_bus,
            llm_client=mock_llm,
        )

        initial = create_decomposition_initial_state(
            task="Build a simple app",
            session_id=SESSION_ID,
            integration_sandbox_id=INTEGRATION_SANDBOX_ID,
        )

        final = await graph.run(initial)

        assert final["status"] == "complete"
        assert final["final_build_success"] is True
        assert len(final["subtask_results"]) == 1
        assert final["subtask_results"][0]["subtask_id"] == "subtask_1"

        # Verify key events
        events = await collect_events(event_bus, SESSION_ID)
        event_types = [e.type for e in events]
        assert EventType.GRAPH_INITIALIZED in event_types
        assert EventType.ORCHESTRATOR_PLAN in event_types
        assert EventType.AGGREGATION_STARTED in event_types
        assert EventType.AGGREGATION_COMPLETE in event_types

    @pytest.mark.asyncio
    async def test_orchestrator_invalid_json_uses_fallback_plan(
        self, mock_sandbox_manager, event_bus
    ) -> None:
        """Orchestrator returns non-JSON for all attempts, fallback plan executes."""
        # All 3 attempts (initial + 2 retries) return invalid JSON
        bad_response = make_llm_response(content="I can't produce JSON right now.")
        subtask_reason = make_llm_response(
            content="<plan>Write App.tsx</plan>",
            tool_calls=[
                make_tool_call(
                    "write_file",
                    {"path": "src/App.tsx", "content": "export default function App() {}"},
                    call_id="tc_sub_1",
                )
            ],
        )
        subtask_complete = make_llm_response(content="<status>TASK_COMPLETE</status>")

        mock_llm = RoutingMockLLMClient(response_map={
            "default": [bad_response, bad_response, bad_response],
            "agent_subtask_1": [subtask_reason, subtask_complete],
        })

        graph = DecompositionGraph(
            sandbox_manager=mock_sandbox_manager,
            event_bus=event_bus,
            llm_client=mock_llm,
        )

        initial = create_decomposition_initial_state(
            task="Build something",
            session_id=SESSION_ID,
            integration_sandbox_id=INTEGRATION_SANDBOX_ID,
        )

        final = await graph.run(initial)

        assert final["status"] == "complete"
        assert len(final["subtasks"]) == 1
        assert final["subtasks"][0]["id"] == "subtask_1"
        # Orchestrator was called 3 times (initial + 2 retries)
        orchestrator_calls = [
            entry for entry in mock_llm.call_history
            if entry.get("agent_id", "").startswith("orchestrator_")
        ]
        assert len(orchestrator_calls) == 3

    @pytest.mark.asyncio
    async def test_two_dependency_layers(
        self, mock_sandbox_manager, event_bus
    ) -> None:
        """2 subtasks where subtask_2 depends on subtask_1.

        dependency_layers should have 2 layers. Both complete successfully.
        """
        subtask_1 = {
            "id": "subtask_1",
            "description": "Create types",
            "files_responsible": ["src/types.ts"],
            "dependencies": [],
            "complexity": "low",
        }
        subtask_2 = {
            "id": "subtask_2",
            "description": "Create component using types",
            "files_responsible": ["src/App.tsx"],
            "dependencies": ["subtask_1"],
            "complexity": "low",
        }

        orchestrator_response = make_llm_response(
            content=_orchestrator_json([subtask_1, subtask_2])
        )

        subtask_1_reason = make_llm_response(
            content="<plan>Write shared types</plan>",
            tool_calls=[make_tool_call(
                "write_file",
                {"path": "src/types.ts", "content": "export interface Item { id: string; }"},
                call_id="tc_sub_1",
            )],
        )
        subtask_2_reason = make_llm_response(
            content="<plan>Assemble App</plan>",
            tool_calls=[make_tool_call(
                "write_file",
                {
                    "path": "src/App.tsx",
                    "content": "export default function App() { return <main />; }",
                },
                call_id="tc_sub_2",
            )],
        )
        subtask_done = make_llm_response(content="TASK_COMPLETE")

        mock_llm = RoutingMockLLMClient(response_map={
            "default": [orchestrator_response],
            "agent_subtask_1": [subtask_1_reason, subtask_done],
            "agent_subtask_2": [subtask_2_reason, subtask_done],
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
                    stdout="./src/App.tsx\n./src/types.ts\n",
                    stderr="", exit_code=0, timed_out=False
                )
            return CommandResult(stdout="OK", stderr="", exit_code=0, timed_out=False)

        mock_sandbox_manager.execute_command = AsyncMock(side_effect=execute_command_side_effect)
        mock_sandbox_manager.read_file = AsyncMock(return_value="file code")

        graph = DecompositionGraph(
            sandbox_manager=mock_sandbox_manager,
            event_bus=event_bus,
            llm_client=mock_llm,
        )

        initial = create_decomposition_initial_state(
            task="Build with dependencies",
            session_id=SESSION_ID,
            integration_sandbox_id=INTEGRATION_SANDBOX_ID,
        )

        final = await graph.run(initial)

        assert final["status"] == "complete"
        assert len(final["subtask_results"]) == 2
        # Verify 2 dependency layers were created
        assert len(final["dependency_layers"]) == 2

    @pytest.mark.asyncio
    async def test_subtask_failure_continues_to_aggregation(
        self, mock_sandbox_manager, event_bus
    ) -> None:
        """One subtask's sandbox creation throws. Graph continues,
        failed_agents populated, aggregation still runs.
        """
        subtask_1 = {
            "id": "subtask_1",
            "description": "Subtask that fails",
            "files_responsible": ["src/Bad.tsx"],
            "dependencies": [],
            "complexity": "low",
        }
        subtask_2 = {
            "id": "subtask_2",
            "description": "Subtask that succeeds and assembles App",
            "files_responsible": ["src/App.tsx"],
            "dependencies": [],
            "complexity": "low",
        }

        orchestrator_response = make_llm_response(
            content=_orchestrator_json([subtask_1, subtask_2])
        )

        subtask_reason = make_llm_response(
            content="<plan>Write file</plan>",
            tool_calls=[make_tool_call(
                "write_file",
                {
                    "path": "src/App.tsx",
                    "content": "export default function App() { return <main />; }",
                },
                call_id="tc_good",
            )],
        )
        subtask_done = make_llm_response(content="TASK_COMPLETE")

        mock_llm = RoutingMockLLMClient(response_map={
            "default": [orchestrator_response],
            "agent_subtask_2": [subtask_reason, subtask_done],
        })

        call_count = 0

        async def create_sandbox_side_effect(sandbox_id: str):
            nonlocal call_count
            call_count += 1
            # First subtask sandbox creation fails
            if "subtask_1" in sandbox_id:
                raise RuntimeError("Docker creation failed")
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
            return CommandResult(stdout="OK", stderr="", exit_code=0, timed_out=False)

        mock_sandbox_manager.execute_command = AsyncMock(side_effect=execute_command_side_effect)
        mock_sandbox_manager.read_file = AsyncMock(
            return_value="export default function App() { return <main />; }"
        )

        graph = DecompositionGraph(
            sandbox_manager=mock_sandbox_manager,
            event_bus=event_bus,
            llm_client=mock_llm,
        )

        initial = create_decomposition_initial_state(
            task="Mixed success/failure",
            session_id=SESSION_ID,
            integration_sandbox_id=INTEGRATION_SANDBOX_ID,
        )

        final = await graph.run(initial)

        # Graph should still complete (one subtask succeeded)
        assert final["status"] == "complete"
        assert len(final["subtask_results"]) == 2
        assert "agent_subtask_1" in final["failed_agents"]

    def test_should_retry_integration_routing(self) -> None:
        """Unit test: returns 'end' when final_build_success or retries >= 2."""
        graph = _make_graph_for_unit_tests()

        # Build success -> end
        assert graph._should_retry_integration(
            {"status": "integrating", "final_build_success": True, "integration_retries": 0}
        ) == "end"

        # Max retries -> end
        assert graph._should_retry_integration(
            {"status": "integrating", "final_build_success": False, "integration_retries": 2}
        ) == "end"

        # Complete status -> end
        assert graph._should_retry_integration(
            {"status": "complete", "final_build_success": False, "integration_retries": 0}
        ) == "end"

        # Failed status -> end
        assert graph._should_retry_integration(
            {"status": "failed", "final_build_success": False, "integration_retries": 0}
        ) == "end"

        # Still integrating, build failed, retries < 2 -> retry
        assert graph._should_retry_integration(
            {"status": "integrating", "final_build_success": False, "integration_retries": 1}
        ) == "retry"
