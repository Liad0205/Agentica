"""Task Decomposition Swarm LangGraph implementation.

This module provides the Task Decomposition agent graph - where an orchestrator
decomposes a task into subtasks, parallel sub-agents execute them, and an
aggregator merges the results.

Graph structure:
    START -> orchestrate -> dispatch_layer -> execute_subtask(×layer) -> advance_layer
               ^                                                         |
               |_________________________________________________________|
                   (loop per dependency layer, then aggregate -> integration_review -> END)

Events emitted:
- ORCHESTRATOR_PLAN: When the orchestrator produces the decomposition
- AGENT_SPAWNED: When each sub-agent starts
- AGENT_THINKING, AGENT_TOOL_CALL, AGENT_TOOL_RESULT: During sub-agent execution
- AGGREGATION_STARTED, AGGREGATION_COMPLETE: During file merging
- PREVIEW_READY: When integration sandbox preview is available
"""

import asyncio
import json
import operator
import re
import time
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.prompts import (
    BASE_CODING_AGENT_PROMPT,
    ORCHESTRATOR_PROMPT,
    build_session_contract,
    build_tooling_contract,
    compose_prompt_sections,
    get_subtask_prompt,
)
from agents.tools import (
    ToolCall,
    ToolExecutor,
    get_tool_definitions_for_llm,
)
from agents.utils import (
    LLMClient,
    classify_build_errors,
    extract_json_from_response,
    format_assistant_message_with_tools,
    format_tool_result_for_llm,
    parse_plan_tag,
    parse_status_tag,
    sliding_window_prune,
    topological_sort,
)
from config import settings
from events.bus import EventBus
from events.types import AgentEvent, EventType
from sandbox.docker_sandbox import SandboxManager

logger = structlog.get_logger()

TEMPLATE_SCAFFOLD_FILES = {
    "package.json",
    "vite.config.ts",
    "postcss.config.js",
    "tsconfig.json",
    "tsconfig.app.json",
    "tsconfig.node.json",
    "index.html",
    ".eslintrc.json",
    "src/types.ts",
}

INTEGRATION_SEED_BASE_FILES = {
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "vite.config.ts",
    "postcss.config.js",
    "tsconfig.json",
    "tsconfig.app.json",
    "tsconfig.node.json",
    "src/types.ts",
}

APP_ENTRYPOINT_PATH = "src/App.tsx"


# -----------------------------------------------------------------------------
# State Schema Definitions
# -----------------------------------------------------------------------------


class SubtaskResult(TypedDict):
    """Result from a sub-agent's work on a subtask.

    Attributes:
        subtask_id: The subtask identifier (e.g., "subtask_1")
        agent_id: The agent identifier
        sandbox_id: The sandbox container used
        files_produced: Dict mapping path -> content for all files written
        dependencies: Runtime dependency requirements discovered in package.json
        dev_dependencies: Dev dependency requirements discovered in package.json
        type_additions: Additional exported type blocks to merge into src/types.ts
        edited_files: Paths explicitly edited by the sub-agent via write tools
        build_success: Whether npm run build succeeded
        build_output: Build command output
        lint_output: Lint command output (empty if build failed)
        summary: Agent's summary of what was built
        status: Final status of the sub-agent
        retries: Number of retry attempts made
    """

    subtask_id: str
    agent_id: str
    sandbox_id: str
    files_produced: dict[str, str]
    dependencies: dict[str, str]
    dev_dependencies: dict[str, str]
    type_additions: str
    edited_files: list[str]
    build_success: bool
    build_output: str
    lint_output: str
    summary: str
    status: Literal["complete", "failed", "timeout"]
    retries: int


class DecompositionState(TypedDict):
    """State for the Task Decomposition graph.

    This state flows through the graph and accumulates information
    as the orchestrator plans and sub-agents execute.

    Attributes:
        task: The user's original task/prompt
        session_id: Session identifier for events
        additional_dependencies: Extra npm packages to install beyond pre-installed template
        subtasks: List of subtask definitions from orchestrator
        dependency_layers: Topologically sorted dependency layers for execution
        current_layer_index: Index of the dependency layer currently being dispatched
        shared_types: Content of src/types.ts for sub-agents
        subtask_results: Results from all sub-agents (fan-in via operator.add)
        failed_agents: List of agent IDs that failed (fan-in via operator.add)
        integration_sandbox_id: Sandbox for merged code
        merged_files: List of file paths merged
        integration_retries: Number of integration fix attempts
        status: Current graph status
        final_build_success: Whether final build succeeded
        final_preview_url: Preview URL if available
        error_message: Error message if failed
    """

    task: str
    session_id: str
    run_id: str
    additional_dependencies: list[str]
    subtasks: list[dict[str, Any]]
    dependency_layers: list[list[dict[str, Any]]]
    current_layer_index: int
    shared_types: str
    subtask_results: Annotated[list[SubtaskResult], operator.add]
    failed_agents: Annotated[list[str], operator.add]
    integration_sandbox_id: str
    synced_subtask_ids: list[str]
    merged_files: list[str]
    integration_retries: int
    status: Literal[
        "orchestrating", "executing", "aggregating", "integrating", "complete", "failed"
    ]
    final_build_success: bool
    final_preview_url: str | None
    error_message: str | None


def create_decomposition_initial_state(
    task: str,
    session_id: str,
    integration_sandbox_id: str,
) -> DecompositionState:
    """Create the initial state for a decomposition graph run.

    Args:
        task: The user's task/prompt
        session_id: Session ID for event emission
        integration_sandbox_id: Pre-created integration sandbox ID

    Returns:
        Initial DecompositionState dict
    """
    return DecompositionState(
        task=task,
        session_id=session_id,
        run_id=f"run_{uuid4().hex[:8]}",
        additional_dependencies=[],
        subtasks=[],
        dependency_layers=[],
        current_layer_index=0,
        shared_types="",
        subtask_results=[],
        failed_agents=[],
        integration_sandbox_id=integration_sandbox_id,
        synced_subtask_ids=[],
        merged_files=[],
        integration_retries=0,
        status="orchestrating",
        final_build_success=False,
        final_preview_url=None,
        error_message=None,
    )


# -----------------------------------------------------------------------------
# DecompositionGraph Class
# -----------------------------------------------------------------------------


class DecompositionGraph:
    """The Task Decomposition graph implementation.

    This class encapsulates the LangGraph StateGraph and provides
    the node implementations for orchestration, sub-agent execution,
    aggregation, and integration review.

    Usage:
        >>> graph = DecompositionGraph(sandbox_manager, event_bus, llm_client)
        >>> initial_state = create_decomposition_initial_state(task, session_id, sandbox_id)
        >>> final_state = await graph.run(initial_state)
    """

    def __init__(
        self,
        sandbox_manager: SandboxManager,
        event_bus: EventBus,
        llm_client: LLMClient | None = None,
        orchestrator_model: str | None = None,
        sub_agent_temperature: float = 0.7,
    ) -> None:
        """Initialize the Decomposition graph.

        Args:
            sandbox_manager: Manager for sandbox operations
            event_bus: Event bus for emitting events
            llm_client: LLM client for model calls (creates default if None)
            orchestrator_model: Model used by the orchestration node.
            sub_agent_temperature: Default temperature for sub-agents.
        """
        self.sandbox_manager = sandbox_manager
        self.event_bus = event_bus
        self.llm_client = llm_client or LLMClient(event_bus=event_bus)
        self.orchestrator_model = orchestrator_model or settings.orchestrator_model
        self.sub_agent_temperature = sub_agent_temperature
        self._session_start_time: float = 0.0  # Set when graph starts running
        self._compiled_graph = self._build_graph()

    @staticmethod
    def _build_fallback_subtasks(task: str) -> list[dict[str, Any]]:
        """Build a deterministic fallback decomposition when orchestrator output is invalid."""
        return [
            {
                "id": "subtask_1",
                "role": "Full-Stack Implementer",
                "description": (
                    "Fallback plan: implement the full task end-to-end in this subtask. "
                    f"Task: {task}"
                ),
                "files_responsible": [
                    "src/App.tsx",
                    "src/index.css",
                    "src/types.ts",
                    "src/components/*",
                    "src/hooks/*",
                    "src/lib/*",
                ],
                "dependencies": [],
                "complexity": "high",
            }
        ]

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph StateGraph.

        Returns:
            Compiled StateGraph ready for execution
        """
        graph = StateGraph(DecompositionState)

        # Add nodes
        graph.add_node("orchestrate", self._orchestrate)
        graph.add_node("dispatch_layer", self._dispatch_layer)
        graph.add_node("execute_subtask", self._execute_subtask)
        graph.add_node("advance_layer", self._advance_layer)
        graph.add_node("aggregate", self._aggregate)
        graph.add_node("integration_review", self._integration_review)

        # Add edges
        graph.add_edge(START, "orchestrate")

        # Route after orchestration: either dispatch first layer or end on failure.
        graph.add_conditional_edges(
            "orchestrate",
            self._route_after_orchestrate,
            {
                "dispatch": "dispatch_layer",
                "end": END,
            },
        )

        # Fan-out current dependency layer to execute_subtask nodes.
        graph.add_conditional_edges(
            "dispatch_layer",
            self._fan_out_current_layer,
            ["execute_subtask"],
        )

        # After current layer completes, either dispatch next layer or aggregate.
        graph.add_edge("execute_subtask", "advance_layer")
        graph.add_conditional_edges(
            "advance_layer",
            self._route_after_layer,
            {
                "dispatch": "dispatch_layer",
                "aggregate": "aggregate",
            },
        )

        # Aggregate either proceeds to integration review or terminates on failure.
        graph.add_conditional_edges(
            "aggregate",
            self._route_after_aggregate,
            {
                "integrate": "integration_review",
                "end": END,
            },
        )

        # Integration review can retry or end
        graph.add_conditional_edges(
            "integration_review",
            self._should_retry_integration,
            {
                "retry": "integration_review",
                "end": END,
            },
        )

        return graph.compile()

    # -------------------------------------------------------------------------
    # Event Emission Helpers
    # -------------------------------------------------------------------------

    async def _emit_node_active(
        self, session_id: str, agent_id: str, agent_role: str, node_name: str
    ) -> None:
        """Emit GRAPH_NODE_ACTIVE event."""
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.GRAPH_NODE_ACTIVE,
                session_id=session_id,
                agent_id=agent_id,
                agent_role=agent_role,
                data={"node_id": node_name},
            )
        )

    async def _emit_node_complete(
        self, session_id: str, agent_id: str, agent_role: str, node_name: str
    ) -> None:
        """Emit GRAPH_NODE_COMPLETE event."""
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.GRAPH_NODE_COMPLETE,
                session_id=session_id,
                agent_id=agent_id,
                agent_role=agent_role,
                data={"node_id": node_name},
            )
        )

    async def _emit_thinking(
        self,
        session_id: str,
        agent_id: str,
        agent_role: str,
        content: str,
        streaming: bool = False,
    ) -> None:
        """Emit AGENT_THINKING event."""
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.AGENT_THINKING,
                session_id=session_id,
                agent_id=agent_id,
                agent_role=agent_role,
                data={
                    "content": content,
                    "streaming": streaming,
                },
            )
        )

    def _remaining_session_budget_seconds(self) -> float:
        """Return remaining wall-clock budget for the current session."""
        start_time = getattr(self, "_session_start_time", 0.0)
        if not start_time:
            return float(settings.agent_timeout_seconds)

        elapsed = time.monotonic() - start_time
        return max(0.0, float(settings.agent_timeout_seconds) - elapsed)

    def _allocate_subtask_budget_seconds(self, subtask_state: dict[str, Any]) -> int:
        """Allocate a timeout budget for a subtask based on remaining session time."""
        remaining_budget = self._remaining_session_budget_seconds()
        total_layers = max(int(subtask_state.get("total_layers", 1)), 1)
        dependency_layer = max(int(subtask_state.get("dependency_layer", 0)), 0)
        remaining_layers = max(total_layers - dependency_layer, 1)

        # Keep room for aggregation + integration review after subtasks finish.
        integration_reserve_seconds = 75.0
        usable_budget = max(0.0, remaining_budget - integration_reserve_seconds)
        layer_budget = usable_budget / remaining_layers if usable_budget > 0 else 0.0

        if layer_budget <= 0:
            layer_budget = max(0.0, remaining_budget - 10.0)

        # Subtasks in the same layer run in parallel, so each gets the layer budget.
        dynamic_budget = int(layer_budget)
        dynamic_budget = max(20, min(180, dynamic_budget))

        if remaining_budget > 0:
            dynamic_budget = min(dynamic_budget, max(5, int(remaining_budget)))

        return max(5, dynamic_budget)

    @staticmethod
    def _timeout_before_deadline(
        deadline: float,
        default_timeout: int,
        *,
        reserve_seconds: int = 0,
        min_timeout: int = 15,
    ) -> int:
        """Return a timeout clipped by remaining time until deadline."""
        remaining = int(deadline - time.monotonic()) - reserve_seconds
        if remaining <= 0:
            return 0
        if remaining < min_timeout:
            return remaining
        return min(default_timeout, remaining)

    # Valid npm package name: optional @scope/, then package name, optional @version.
    # Rejects shell metacharacters, spaces, and injection attempts.
    _NPM_DEP_RE = re.compile(
        r"^(@[a-z0-9\-~][a-z0-9\-._~]*/)?[a-z0-9\-~][a-z0-9\-._~]*(@[^\s;|&`$]+)?$"
    )

    @classmethod
    def _normalize_npm_dependency_list(cls, raw_dependencies: Any) -> list[str]:
        """Normalize and validate orchestrator-provided dependency names.

        Rejects dependency strings that don't match a strict npm package name
        pattern to prevent shell injection via crafted names.
        """
        if not isinstance(raw_dependencies, list):
            return []

        normalized: list[str] = []
        seen: set[str] = set()
        for item in raw_dependencies:
            if not isinstance(item, str):
                continue
            dep = item.strip()
            if not dep or dep in seen:
                continue
            if not cls._NPM_DEP_RE.match(dep):
                logger.warning(
                    "skipping_invalid_npm_dependency",
                    dependency=dep[:100],
                )
                continue
            seen.add(dep)
            normalized.append(dep)
        return normalized

    @staticmethod
    def _normalize_repo_path(raw_path: Any) -> str:
        """Normalize a repository-relative path for internal comparisons."""
        if not isinstance(raw_path, str):
            return ""
        return raw_path.strip().lstrip("./")

    @staticmethod
    def _next_unique_subtask_id(existing_ids: set[str], base_id: str) -> str:
        """Return a stable unique subtask ID using the provided base."""
        if base_id not in existing_ids:
            return base_id

        suffix = 2
        while True:
            candidate = f"{base_id}_{suffix}"
            if candidate not in existing_ids:
                return candidate
            suffix += 1

    def _ensure_app_entrypoint_subtask(
        self,
        subtasks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Guarantee there is an App.tsx assembly subtask in the plan.

        Rules enforced here:
        - At least one subtask owns src/App.tsx.
        - If missing, inject a dedicated App assembly subtask that depends on
          all existing subtasks.
        - If present, add dependencies on peer subtasks when doing so does not
          create an immediate dependency cycle.
        """
        if not subtasks:
            return subtasks

        app_owner_index = next(
            (
                index
                for index, subtask in enumerate(subtasks)
                if APP_ENTRYPOINT_PATH in subtask.get("files_responsible", [])
            ),
            None,
        )

        if app_owner_index is None:
            existing_ids = {str(subtask.get("id", "")) for subtask in subtasks}
            assembly_id = self._next_unique_subtask_id(
                existing_ids, "subtask_app_assembly"
            )
            subtasks.append(
                {
                    "id": assembly_id,
                    "role": "App Assembly Integrator",
                    "description": (
                        "Assemble all completed subtask outputs into a working "
                        "application entrypoint in src/App.tsx. Import and render "
                        "the produced components so the final UI is reachable."
                    ),
                    "files_responsible": [APP_ENTRYPOINT_PATH],
                    "dependencies": [str(subtask.get("id", "")) for subtask in subtasks],
                    "complexity": "medium",
                }
            )
            return subtasks

        app_subtask = subtasks[app_owner_index]
        app_subtask_id = str(app_subtask.get("id", ""))
        app_dependencies = [
            str(dep)
            for dep in app_subtask.get("dependencies", [])
            if isinstance(dep, str) and dep
        ]
        app_dependency_set = set(app_dependencies)

        blocked_peer_ids = {
            str(subtask.get("id", ""))
            for subtask in subtasks
            if str(subtask.get("id", "")) != app_subtask_id
            and app_subtask_id in subtask.get("dependencies", [])
        }

        for subtask in subtasks:
            peer_id = str(subtask.get("id", ""))
            if not peer_id or peer_id == app_subtask_id or peer_id in blocked_peer_ids:
                continue
            if peer_id not in app_dependency_set:
                app_dependencies.append(peer_id)
                app_dependency_set.add(peer_id)

        app_subtask["dependencies"] = app_dependencies
        return subtasks

    def _validate_and_normalize_subtasks(
        self, raw_subtasks: Any
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Validate orchestrator subtasks and normalize them for execution."""
        if not isinstance(raw_subtasks, list):
            return [], "Orchestrator 'subtasks' must be an array"

        if not raw_subtasks:
            return [], "Orchestrator produced no subtasks"

        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        file_ownership: dict[str, str] = {}

        for index, raw_subtask in enumerate(raw_subtasks):
            if not isinstance(raw_subtask, dict):
                return [], f"Subtask at index {index} must be an object"

            raw_id = raw_subtask.get("id", "")
            subtask_id = (
                str(raw_id).strip()
                if isinstance(raw_id, str) and raw_id.strip()
                else f"subtask_{index + 1}"
            )
            if subtask_id in seen_ids:
                return [], f"Duplicate subtask id: {subtask_id}"
            seen_ids.add(subtask_id)

            role = raw_subtask.get("role", "")
            role_value = (
                str(role).strip()
                if isinstance(role, str) and role.strip()
                else f"Sub-agent {index + 1}"
            )

            description = raw_subtask.get("description", "")
            if not isinstance(description, str) or not description.strip():
                return [], f"Subtask '{subtask_id}' is missing a description"

            files_raw = raw_subtask.get("files_responsible", [])
            if not isinstance(files_raw, list):
                return [], f"Subtask '{subtask_id}' has invalid files_responsible"

            files_responsible: list[str] = []
            seen_files: set[str] = set()
            for raw_path in files_raw:
                if not isinstance(raw_path, str):
                    continue
                normalized_path = self._normalize_repo_path(raw_path)
                if not normalized_path or normalized_path in seen_files:
                    continue
                seen_files.add(normalized_path)
                files_responsible.append(normalized_path)

                existing_owner = file_ownership.get(normalized_path)
                if existing_owner and existing_owner != subtask_id:
                    return [], (
                        f"File ownership overlap detected: '{normalized_path}' "
                        f"assigned to both '{existing_owner}' and '{subtask_id}'"
                    )
                file_ownership[normalized_path] = subtask_id

            dependencies_raw = raw_subtask.get("dependencies", [])
            if dependencies_raw is None:
                dependencies_raw = []
            if not isinstance(dependencies_raw, list):
                return [], f"Subtask '{subtask_id}' has invalid dependencies list"

            dependencies: list[str] = []
            seen_deps: set[str] = set()
            for dep in dependencies_raw:
                if not isinstance(dep, str):
                    continue
                normalized_dep = dep.strip()
                if not normalized_dep or normalized_dep in seen_deps:
                    continue
                if normalized_dep == subtask_id:
                    return [], f"Subtask '{subtask_id}' cannot depend on itself"
                seen_deps.add(normalized_dep)
                dependencies.append(normalized_dep)

            complexity_raw = raw_subtask.get("complexity", "medium")
            complexity = complexity_raw if isinstance(complexity_raw, str) else "medium"
            complexity = complexity.strip().lower()
            if complexity not in {"low", "medium", "high"}:
                complexity = "medium"

            normalized.append(
                {
                    "id": subtask_id,
                    "role": role_value,
                    "description": description.strip(),
                    "files_responsible": files_responsible,
                    "dependencies": dependencies,
                    "complexity": complexity,
                }
            )

        normalized = self._ensure_app_entrypoint_subtask(normalized)
        id_set = {subtask["id"] for subtask in normalized}
        for subtask in normalized:
            for dep in subtask["dependencies"]:
                if dep not in id_set:
                    return [], (
                        f"Subtask '{subtask['id']}' has unknown dependency '{dep}'"
                    )

        return normalized, None

    @staticmethod
    def _dependency_layers_are_valid(
        subtasks: list[dict[str, Any]],
        dependency_layers: list[list[dict[str, Any]]],
    ) -> bool:
        """Validate that dependency layers obey strict topological ordering."""
        task_map = {
            str(subtask.get("id", "")): subtask
            for subtask in subtasks
            if isinstance(subtask.get("id"), str)
        }
        expected_ids = set(task_map.keys())
        seen: set[str] = set()

        for layer in dependency_layers:
            layer_ids: set[str] = set()
            for subtask in layer:
                subtask_id = str(subtask.get("id", ""))
                if subtask_id not in expected_ids:
                    return False
                if subtask_id in seen or subtask_id in layer_ids:
                    return False
                deps = task_map[subtask_id].get("dependencies", [])
                if any(dep not in seen for dep in deps if dep in expected_ids):
                    return False
                layer_ids.add(subtask_id)
            seen.update(layer_ids)

        return seen == expected_ids

    async def _collect_integration_seed_paths(
        self, integration_sandbox_id: str
    ) -> list[str]:
        """Collect the list of files to seed into a subtask sandbox."""
        seed_paths: set[str] = set(INTEGRATION_SEED_BASE_FILES)
        try:
            entries = await self.sandbox_manager.list_files_recursive(
                integration_sandbox_id,
                path="src",
            )
            for entry in entries:
                if entry.is_directory:
                    continue
                normalized = str(entry.path).strip().lstrip("./")
                if normalized.startswith("src/"):
                    seed_paths.add(normalized)
        except Exception as e:
            logger.warning(
                "collect_integration_seed_paths_failed",
                sandbox_id=integration_sandbox_id,
                error=str(e),
            )

        return sorted(seed_paths)

    async def _seed_subtask_sandbox_from_integration(
        self,
        *,
        integration_sandbox_id: str,
        subtask_sandbox_id: str,
        session_id: str,
        subtask_id: str,
    ) -> int:
        """Seed a subtask sandbox from the canonical integration sandbox."""
        seed_paths = await self._collect_integration_seed_paths(integration_sandbox_id)
        if not seed_paths:
            return 0

        await self.sandbox_manager.copy_files_between(
            source_id=integration_sandbox_id,
            target_id=subtask_sandbox_id,
            paths=seed_paths,
        )
        logger.info(
            "subtask_sandbox_seeded",
            session_id=session_id,
            subtask_id=subtask_id,
            source_sandbox=integration_sandbox_id,
            target_sandbox=subtask_sandbox_id,
            file_count=len(seed_paths),
        )
        return len(seed_paths)

    async def _merge_successful_results_into_integration(
        self,
        *,
        successful_results: list[SubtaskResult],
        integration_sandbox_id: str,
        session_id: str,
        aggregator_id: str,
        shared_types: str,
    ) -> dict[str, Any]:
        """Merge successful subtask outputs into the integration sandbox."""
        all_files: dict[str, str] = {}
        file_sources: dict[str, list[tuple[str, str]]] = {}
        conflicts_resolved = 0

        for result in successful_results:
            files_produced = result.get("files_produced", {})
            agent_id = str(result.get("agent_id", "unknown"))
            if not isinstance(files_produced, dict):
                continue

            for path, content in files_produced.items():
                if not isinstance(path, str) or not isinstance(content, str):
                    continue
                normalized_path = path.strip().lstrip("./")
                if not normalized_path or normalized_path in TEMPLATE_SCAFFOLD_FILES:
                    continue
                file_sources.setdefault(normalized_path, []).append((agent_id, content))

        for path, sources in file_sources.items():
            if len(sources) == 1:
                all_files[path] = sources[0][1]
                continue

            conflicts_resolved += 1
            logger.warning(
                "integration_file_conflict",
                session_id=session_id,
                path=path,
                source_count=len(sources),
            )
            all_files[path] = await self._merge_conflicting_files(
                path, sources, session_id, aggregator_id
            )

        dependency_sync_result = {
            "dependencies_added": 0,
            "dependency_conflicts": [],
        }
        type_sync_result = {"type_blocks_added": 0}
        if successful_results:
            dependency_sync_result = await self._sync_integration_dependencies(
                integration_sandbox_id=integration_sandbox_id,
                successful_results=successful_results,
                session_id=session_id,
                aggregator_id=aggregator_id,
            )
            type_sync_result = await self._sync_integration_shared_types(
                integration_sandbox_id=integration_sandbox_id,
                shared_types=shared_types,
                successful_results=successful_results,
                session_id=session_id,
                aggregator_id=aggregator_id,
            )

        merged_files: list[str] = []
        for path, content in all_files.items():
            try:
                await self.sandbox_manager.write_file(
                    integration_sandbox_id, path, content
                )
                merged_files.append(path)
                await self.event_bus.publish(
                    AgentEvent(
                        type=EventType.FILE_CHANGED,
                        session_id=session_id,
                        agent_id=aggregator_id,
                        agent_role="Aggregator",
                        data={
                            "path": path,
                            "content": content,
                            "sandbox_id": integration_sandbox_id,
                        },
                    )
                )
            except Exception as e:
                logger.error(
                    "integration_write_file_failed",
                    session_id=session_id,
                    path=path,
                    error=str(e),
                )

        return {
            "merged_files": merged_files,
            "conflicts_resolved": conflicts_resolved,
            "dependencies_added": dependency_sync_result["dependencies_added"],
            "dependency_conflicts": dependency_sync_result["dependency_conflicts"],
            "type_blocks_added": type_sync_result["type_blocks_added"],
        }

    # -------------------------------------------------------------------------
    # Node: Orchestrate
    # -------------------------------------------------------------------------

    async def _orchestrate(self, state: DecompositionState) -> dict[str, Any]:
        """Orchestrate node: decompose the task into subtasks.

        This node:
        1. Calls the LLM with ORCHESTRATOR_PROMPT to decompose the task
        2. Parses the JSON response for scaffold and subtasks
        3. Writes scaffold files to the integration sandbox
        4. Emits ORCHESTRATOR_PLAN event

        Args:
            state: Current graph state

        Returns:
            Dict with project_scaffold, subtasks, shared_types, status
        """
        session_id = state["session_id"]
        orchestrator_id = f"orchestrator_{session_id[5:]}"

        await self._emit_node_active(
            session_id, orchestrator_id, "Orchestrator", "orchestrate"
        )

        logger.info(
            "orchestrate_start",
            session_id=session_id,
            task_length=len(state["task"]),
        )

        # Call LLM with orchestrator prompt
        messages = [
            {"role": "system", "content": ORCHESTRATOR_PROMPT},
            {"role": "user", "content": state["task"]},
        ]

        response = await self.llm_client.call(
            messages=messages,
            model=self.orchestrator_model,
            temperature=0.7,
            session_id=session_id,
            agent_id=orchestrator_id,
        )

        # Emit thinking with the response
        await self._emit_thinking(
            session_id, orchestrator_id, "Orchestrator", response.content
        )

        # Parse JSON with retries and self-repair prompts.
        parsed = extract_json_from_response(response.content)
        retry_temperatures = [0.4, 0.2]
        retry_prompts = [
            (
                "Your response was not valid JSON. Respond with ONLY a valid JSON object. "
                "Expected shape: "
                '{"additional_dependencies": ["..."], "shared_types": "...", "subtasks": [...]}'
            ),
            (
                "Repair this into strict valid JSON with no markdown or prose. "
                "Preserve meaning and required keys: "
                "additional_dependencies, shared_types, subtasks."
            ),
        ]

        for retry_index, (retry_temperature, retry_prompt) in enumerate(
            zip(retry_temperatures, retry_prompts, strict=True), start=1
        ):
            if parsed:
                break

            logger.warning(
                "orchestrate_json_parse_failed",
                session_id=session_id,
                attempt=retry_index,
                response_preview=response.content[:200],
            )

            retry_messages = messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": retry_prompt},
            ]

            response = await self.llm_client.call(
                messages=retry_messages,
                model=self.orchestrator_model,
                temperature=retry_temperature,
                session_id=session_id,
                agent_id=orchestrator_id,
            )

            parsed = extract_json_from_response(response.content)

        used_fallback_plan = False
        fallback_reason = ""

        if not parsed:
            logger.warning(
                "orchestrate_json_parse_failed_using_fallback",
                session_id=session_id,
            )
            used_fallback_plan = True
            fallback_reason = "orchestrator_json_parse_failed"
            additional_dependencies: list[str] = []
            shared_types = ""
            subtasks = self._build_fallback_subtasks(state["task"])
        else:
            # Extract and normalize orchestrator fields.
            additional_dependencies = self._normalize_npm_dependency_list(
                parsed.get("additional_dependencies", [])
            )
            shared_types_raw = parsed.get("shared_types", "")
            shared_types = shared_types_raw if isinstance(shared_types_raw, str) else ""
            subtasks_raw = parsed.get("subtasks", [])

            # Backward compat: if orchestrator still returns project_scaffold, extract types
            if not shared_types and "project_scaffold" in parsed:
                shared_types = parsed["project_scaffold"].get("src/types.ts", "")

            subtasks, validation_error = self._validate_and_normalize_subtasks(
                subtasks_raw
            )
            if validation_error:
                logger.warning(
                    "orchestrate_subtasks_invalid_using_fallback",
                    session_id=session_id,
                    error=validation_error,
                )
                used_fallback_plan = True
                fallback_reason = validation_error
                additional_dependencies = []
                subtasks = self._build_fallback_subtasks(state["task"])

        integration_sandbox_id = state["integration_sandbox_id"]

        # Write shared types to integration sandbox if provided
        if shared_types:
            try:
                await self.sandbox_manager.write_file(
                    integration_sandbox_id, "src/types.ts", shared_types
                )
                await self.event_bus.publish(
                    AgentEvent(
                        type=EventType.FILE_CHANGED,
                        session_id=session_id,
                        agent_id=orchestrator_id,
                        agent_role="Orchestrator",
                        data={
                            "path": "src/types.ts",
                            "content": shared_types,
                            "sandbox_id": integration_sandbox_id,
                        },
                    )
                )
            except Exception as e:
                logger.error(
                    "orchestrate_write_types_failed",
                    session_id=session_id,
                    error=str(e),
                )

        # Install additional dependencies if any
        if additional_dependencies:
            deps_str = " ".join(additional_dependencies)
            logger.info(
                "orchestrate_installing_extra_deps",
                session_id=session_id,
                deps=deps_str,
            )
            install_result = await self.sandbox_manager.execute_command(
                integration_sandbox_id, f"npm install {deps_str}", timeout=120
            )
            if install_result.exit_code != 0:
                logger.warning(
                    "orchestrate_extra_deps_install_failed",
                    session_id=session_id,
                    output=install_result.stderr[:500],
                )

        # Emit ORCHESTRATOR_PLAN event
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.ORCHESTRATOR_PLAN,
                session_id=session_id,
                agent_id=orchestrator_id,
                agent_role="Orchestrator",
                data={
                    "subtasks": subtasks,
                    "additional_dependencies": additional_dependencies,
                    "fallback_plan_used": used_fallback_plan,
                    "fallback_reason": fallback_reason if used_fallback_plan else "",
                },
            )
        )

        dependency_layers = topological_sort(subtasks)
        if not dependency_layers:
            logger.error(
                "orchestrate_dependency_layers_empty",
                session_id=session_id,
            )
            await self._emit_node_complete(
                session_id, orchestrator_id, "Orchestrator", "orchestrate"
            )
            return {
                "status": "failed",
                "error_message": "Orchestrator produced no executable dependency layers",
            }

        if not self._dependency_layers_are_valid(subtasks, dependency_layers):
            logger.error(
                "orchestrate_dependency_layers_invalid",
                session_id=session_id,
            )
            await self._emit_node_complete(
                session_id, orchestrator_id, "Orchestrator", "orchestrate"
            )
            return {
                "status": "failed",
                "error_message": "Orchestrator produced invalid dependency ordering",
            }

        logger.info(
            "orchestrate_complete",
            session_id=session_id,
            subtask_count=len(subtasks),
            additional_deps=len(additional_dependencies),
            dependency_layers=len(dependency_layers),
            layer_sizes=[len(layer) for layer in dependency_layers],
        )

        await self._emit_node_complete(
            session_id, orchestrator_id, "Orchestrator", "orchestrate"
        )

        return {
            "additional_dependencies": additional_dependencies,
            "subtasks": subtasks,
            "dependency_layers": dependency_layers,
            "current_layer_index": 0,
            "shared_types": shared_types,
            "status": "executing",
        }

    # -------------------------------------------------------------------------
    # Layer Dispatch & Routing
    # -------------------------------------------------------------------------

    def _route_after_orchestrate(self, state: DecompositionState) -> str:
        """Route to dispatch when orchestration succeeded, else end.

        If orchestration fails (e.g., invalid JSON/no subtasks), we terminate the
        graph early so session status cleanly reflects the orchestrator failure.
        """
        if state.get("status") == "failed":
            return "end"

        layers = state.get("dependency_layers", [])
        if not layers:
            logger.error(
                "orchestrate_no_dependency_layers",
                session_id=state.get("session_id"),
            )
            return "end"

        return "dispatch"

    async def _dispatch_layer(self, state: DecompositionState) -> dict[str, Any]:
        """Dispatch node: announce the current dependency layer.

        Args:
            state: Current graph state with dependency layers

        Returns:
            Optional state updates (none required here).
        """
        current_layer_index = state.get("current_layer_index", 0)
        dependency_layers = state.get("dependency_layers", [])
        session_id = state["session_id"]

        if current_layer_index >= len(dependency_layers):
            logger.info(
                "dispatch_layer_noop_out_of_range",
                session_id=session_id,
                current_layer_index=current_layer_index,
                total_layers=len(dependency_layers),
            )
            return {}

        current_layer = dependency_layers[current_layer_index]
        logger.info(
            "dispatch_layer_start",
            session_id=session_id,
            current_layer_index=current_layer_index,
            total_layers=len(dependency_layers),
            layer_size=len(current_layer),
            subtask_ids=[str(st.get("id", "")) for st in current_layer],
        )
        return {}

    def _fan_out_current_layer(self, state: DecompositionState) -> list[Send]:
        """Create Send() objects for the current dependency layer only."""
        dependency_layers = state.get("dependency_layers", [])
        current_layer_index = state.get("current_layer_index", 0)
        subtasks = state.get("subtasks", [])

        if current_layer_index >= len(dependency_layers):
            return []

        current_layer = dependency_layers[current_layer_index]
        if not current_layer:
            return []

        sends = []
        for subtask in current_layer:
            # Find original index for stable naming and display ordering.
            original_index = next(
                (i for i, st in enumerate(subtasks) if st["id"] == subtask["id"]),
                0,
            )
            subtask_state = {
                "subtask": subtask,
                "subtask_index": original_index,
                "dependency_layer": current_layer_index,
                "total_layers": len(dependency_layers),
                "session_id": state["session_id"],
                "run_id": state.get("run_id", ""),
                "additional_dependencies": state["additional_dependencies"],
                "shared_types": state["shared_types"],
                "integration_sandbox_id": state["integration_sandbox_id"],
            }
            sends.append(Send("execute_subtask", subtask_state))

        logger.info(
            "fan_out_current_layer",
            session_id=state["session_id"],
            current_layer_index=current_layer_index,
            subtask_count=len(sends),
            dependency_layers=len(dependency_layers),
            layer_sizes=[len(layer) for layer in dependency_layers],
        )

        return sends

    async def _advance_layer(self, state: DecompositionState) -> dict[str, Any]:
        """Advance to the next dependency layer after current layer fan-in."""
        current_layer_index = state.get("current_layer_index", 0)
        dependency_layers = state.get("dependency_layers", [])
        session_id = state.get("session_id", "")
        integration_sandbox_id = state.get("integration_sandbox_id", "")
        synced_subtask_ids = list(state.get("synced_subtask_ids", []))
        shared_types = state.get("shared_types", "")
        sync_state_changed = False
        shared_types_changed = False

        if (
            current_layer_index < len(dependency_layers)
            and session_id
            and integration_sandbox_id
        ):
            layer_subtasks = dependency_layers[current_layer_index]
            layer_subtask_ids = {
                str(subtask.get("id", ""))
                for subtask in layer_subtasks
                if isinstance(subtask.get("id"), str)
            }
            pending_sync_results = [
                result
                for result in state.get("subtask_results", [])
                if result.get("status") == "complete"
                and result.get("subtask_id") in layer_subtask_ids
                and result.get("subtask_id") not in synced_subtask_ids
            ]

            if pending_sync_results:
                aggregator_id = f"aggregator_{session_id[5:]}"
                sync_result = await self._merge_successful_results_into_integration(
                    successful_results=pending_sync_results,
                    integration_sandbox_id=integration_sandbox_id,
                    session_id=session_id,
                    aggregator_id=aggregator_id,
                    shared_types=shared_types,
                )
                synced_subtask_ids.extend(
                    str(result.get("subtask_id", ""))
                    for result in pending_sync_results
                    if isinstance(result.get("subtask_id"), str)
                )
                synced_subtask_ids = list(dict.fromkeys(synced_subtask_ids))
                sync_state_changed = True

                logger.info(
                    "layer_sync_complete",
                    session_id=session_id,
                    layer=current_layer_index,
                    synced_subtasks=len(pending_sync_results),
                    files_merged=len(sync_result["merged_files"]),
                    dependencies_added=sync_result["dependencies_added"],
                    type_blocks_added=sync_result["type_blocks_added"],
                )

            try:
                latest_shared_types = await self.sandbox_manager.read_file(
                    integration_sandbox_id, "src/types.ts"
                )
                if latest_shared_types != shared_types:
                    shared_types = latest_shared_types
                    shared_types_changed = True
            except FileNotFoundError:
                shared_types = state.get("shared_types", "")
            except Exception as e:
                logger.warning(
                    "advance_layer_read_shared_types_failed",
                    session_id=session_id,
                    error=str(e),
                )

        next_layer_index = current_layer_index + 1
        updates: dict[str, Any] = {"current_layer_index": next_layer_index}
        if sync_state_changed:
            updates["synced_subtask_ids"] = synced_subtask_ids
        if shared_types_changed:
            updates["shared_types"] = shared_types
        return updates

    def _route_after_layer(self, state: DecompositionState) -> str:
        """Route to next layer dispatch or to aggregate when layers are complete."""
        dependency_layers = state.get("dependency_layers", [])
        current_layer_index = state.get("current_layer_index", 0)

        if current_layer_index >= len(dependency_layers):
            return "aggregate"
        return "dispatch"

    def _route_after_aggregate(self, state: DecompositionState) -> str:
        """Route aggregate output either to integration review or end on failure."""
        if state.get("status") == "failed":
            return "end"
        return "integrate"

    # -------------------------------------------------------------------------
    # Node: Execute Subtask
    # -------------------------------------------------------------------------

    async def _execute_subtask(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute a single subtask with a mini-ReAct loop.

        This node runs in parallel for each subtask. It:
        1. Creates a dedicated sandbox
        2. Writes the project scaffold
        3. Runs a ReAct loop (max 8 iterations)
        4. Collects files produced
        5. Returns a SubtaskResult

        Args:
            state: Subtask-specific state from Send()

        Returns:
            Dict with subtask_results (will be merged via operator.add)
        """
        subtask = state["subtask"]
        subtask_id = subtask["id"]
        session_id = state["session_id"]
        raw_run_id = str(state.get("run_id", "run_default"))
        run_id = "".join(ch for ch in raw_run_id if ch.isalnum() or ch in ("_", "-"))
        if not run_id:
            run_id = "run_default"
        agent_id = f"agent_{subtask_id}"
        agent_role = subtask.get("role", f"Sub-agent {state['subtask_index'] + 1}")
        sandbox_id = f"sandbox_{subtask_id}_{run_id}_{session_id[5:]}"

        logger.info(
            "execute_subtask_start",
            session_id=session_id,
            subtask_id=subtask_id,
            role=agent_role,
        )

        subtask_budget_seconds = self._allocate_subtask_budget_seconds(state)
        subtask_deadline = time.monotonic() + subtask_budget_seconds

        logger.info(
            "execute_subtask_budget_allocated",
            session_id=session_id,
            subtask_id=subtask_id,
            dependency_layer=state.get("dependency_layer"),
            total_layers=state.get("total_layers"),
            budget_seconds=subtask_budget_seconds,
            remaining_session_budget=round(self._remaining_session_budget_seconds(), 1),
        )

        max_retries = 2
        retries = 0
        sandbox_created = False

        try:
            def timeout_result(message: str) -> dict[str, Any]:
                return {
                    "status": "timeout",
                    "iterations": 0,
                    "build_success": False,
                    "build_output": message,
                    "lint_output": "",
                    "summary": "Timed out",
                }

            # Create sandbox for this sub-agent
            # The sandbox image already has the full Vite+React+TS+Tailwind template
            await self.sandbox_manager.create_sandbox(sandbox_id)
            sandbox_created = True

            integration_sandbox_id = state["integration_sandbox_id"]
            try:
                await self._seed_subtask_sandbox_from_integration(
                    integration_sandbox_id=integration_sandbox_id,
                    subtask_sandbox_id=sandbox_id,
                    session_id=session_id,
                    subtask_id=subtask_id,
                )
            except Exception as e:
                logger.warning(
                    "subtask_seed_failed",
                    session_id=session_id,
                    subtask_id=subtask_id,
                    sandbox_id=sandbox_id,
                    error=str(e),
                )

            # Write shared types if provided
            if state.get("shared_types"):
                await self.sandbox_manager.write_file(
                    sandbox_id, "src/types.ts", state["shared_types"]
                )

            # Install dependencies required for this layer context.
            extra_deps = state.get("additional_dependencies", [])
            should_sync_environment = state.get("dependency_layer", 0) > 0
            if extra_deps or should_sync_environment:
                deps_str = " ".join(extra_deps)
                install_command = (
                    "npm install"
                    if should_sync_environment or not deps_str
                    else f"npm install {deps_str}"
                )
                install_timeout = self._timeout_before_deadline(
                    subtask_deadline,
                    120,
                    reserve_seconds=45,
                    min_timeout=10,
                )
                if install_timeout <= 0:
                    result = timeout_result(
                        "Subtask timed out before dependency install"
                        f" (budget {subtask_budget_seconds}s)"
                    )
                else:
                    install_result = await self.sandbox_manager.execute_command(
                        sandbox_id,
                        install_command,
                        timeout=install_timeout,
                    )
                    if install_result.timed_out:
                        result = timeout_result(
                            f"Dependency install timed out after {install_timeout}s"
                        )
                    else:
                        if install_result.exit_code != 0:
                            logger.warning(
                                "subtask_dependency_install_failed",
                                session_id=session_id,
                                subtask_id=subtask_id,
                                command=install_command,
                                exit_code=install_result.exit_code,
                                stderr_preview=install_result.stderr[:500],
                            )
                        result = {}
            else:
                result = {}

            # Emit AGENT_SPAWNED
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.AGENT_SPAWNED,
                    session_id=session_id,
                    agent_id=agent_id,
                    agent_role=agent_role,
                    data={
                        "role": agent_role,
                        "sandbox_id": sandbox_id,
                        "parent_id": "orchestrator",
                        "subtask_id": subtask_id,
                    },
                )
            )

            # Run mini-ReAct loop with retry on build failure
            # Timeout is dynamically clipped by remaining session budget.
            if result.get("status") != "timeout":
                subtask_loop_timeout = self._timeout_before_deadline(
                    subtask_deadline,
                    180,
                    reserve_seconds=15,
                    min_timeout=20,
                )
                if subtask_loop_timeout <= 0:
                    result = timeout_result(
                        f"Subtask timed out before execution (budget {subtask_budget_seconds}s)"
                    )
                else:
                    try:
                        result = await asyncio.wait_for(
                            self._run_subtask_react_loop(
                                subtask=subtask,
                                sandbox_id=sandbox_id,
                                session_id=session_id,
                                agent_id=agent_id,
                                agent_role=agent_role,
                                shared_types=state["shared_types"],
                            ),
                            timeout=subtask_loop_timeout,
                        )
                    except TimeoutError:
                        logger.warning(
                            "subtask_react_loop_timeout",
                            session_id=session_id,
                            subtask_id=subtask_id,
                            timeout=subtask_loop_timeout,
                            budget_seconds=subtask_budget_seconds,
                        )
                        result = timeout_result(
                            f"Subtask timed out after {subtask_loop_timeout}s"
                        )

            # Retry loop: if build failed, re-invoke with error context
            while (
                not result["build_success"]
                and result.get("status") != "timeout"
                and retries < max_retries
            ):
                retries += 1
                logger.info(
                    "subtask_retry",
                    session_id=session_id,
                    subtask_id=subtask_id,
                    retry=retries,
                    build_output=result["build_output"][:200],
                )

                # Classify errors for targeted guidance
                error_categories = classify_build_errors(result["build_output"])
                error_summary = "\n".join(
                    f"- {cat}: {len(errs)} errors"
                    for cat, errs in error_categories.items()
                )

                await self._emit_thinking(
                    session_id, agent_id, agent_role,
                    f"Build failed (retry {retries}/{max_retries}). Errors:\n{error_summary}"
                )

                retry_timeout = self._timeout_before_deadline(
                    subtask_deadline,
                    180,
                    reserve_seconds=20,
                    min_timeout=15,
                )
                if retry_timeout <= 0:
                    result = timeout_result(
                        "Subtask timed out before retry"
                        f" {retries} (budget {subtask_budget_seconds}s)"
                    )
                    break

                # Re-invoke react loop with build errors prepended (also with timeout)
                try:
                    result = await asyncio.wait_for(
                        self._run_subtask_react_loop(
                            subtask=subtask,
                            sandbox_id=sandbox_id,
                            session_id=session_id,
                            agent_id=agent_id,
                            agent_role=agent_role,
                            shared_types=state["shared_types"],
                            build_errors=result["build_output"],
                        ),
                        timeout=retry_timeout,
                    )
                except TimeoutError:
                    logger.warning(
                        "subtask_retry_loop_timeout",
                        session_id=session_id,
                        subtask_id=subtask_id,
                        retry=retries,
                        timeout=retry_timeout,
                        budget_seconds=subtask_budget_seconds,
                    )
                    result = timeout_result(
                        f"Subtask retry timed out after {retry_timeout}s"
                    )
                    break

            # Collect files/dependencies only if we still have budget and the subtask
            # didn't already terminate due to timeout.
            find_timeout = self._timeout_before_deadline(
                subtask_deadline,
                30,
                reserve_seconds=5,
                min_timeout=5,
            )
            if find_timeout <= 0:
                files_produced = {}
                dependencies, dev_dependencies = {}, {}
                type_additions = ""
            elif result.get("status") == "timeout":
                # Attempt to collect partial files even on timeout
                try:
                    files_produced = await asyncio.wait_for(
                        self._collect_sandbox_files(
                            sandbox_id, find_timeout=min(find_timeout, 10)
                        ),
                        timeout=min(find_timeout, 10),
                    )
                except (TimeoutError, Exception):
                    files_produced = {}
                dependencies, dev_dependencies = {}, {}
                type_additions = ""
            else:
                files_produced = await self._collect_sandbox_files(
                    sandbox_id, find_timeout=find_timeout
                )
                dependencies, dev_dependencies = await self._collect_dependency_requirements(
                    sandbox_id
                )
                type_additions = await self._collect_type_additions(
                    sandbox_id=sandbox_id,
                    shared_types=state.get("shared_types", ""),
                )

            # Emit AGENT_COMPLETE
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.AGENT_COMPLETE,
                    session_id=session_id,
                    agent_id=agent_id,
                    agent_role=agent_role,
                    data={
                        "status": result["status"],
                        "iterations": result["iterations"],
                        "files_written": list(files_produced.keys()),
                        "dependencies": sorted(dependencies.keys()),
                        "type_additions": bool(type_additions),
                        "retries": retries,
                    },
                )
            )

            logger.info(
                "execute_subtask_complete",
                session_id=session_id,
                subtask_id=subtask_id,
                status=result["status"],
                files_produced=len(files_produced),
                retries=retries,
            )

            return {
                "subtask_results": [
                    SubtaskResult(
                        subtask_id=subtask_id,
                        agent_id=agent_id,
                        sandbox_id=sandbox_id,
                        files_produced=files_produced,
                        dependencies=dependencies,
                        dev_dependencies=dev_dependencies,
                        type_additions=type_additions,
                        edited_files=result.get("edited_files", []),
                        build_success=result["build_success"],
                        build_output=result["build_output"],
                        lint_output=result.get("lint_output", ""),
                        summary=result["summary"],
                        status=result["status"],
                        retries=retries,
                    )
                ]
            }

        except Exception as e:
            logger.error(
                "execute_subtask_error",
                session_id=session_id,
                subtask_id=subtask_id,
                error=str(e),
            )

            # Emit AGENT_ERROR
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.AGENT_ERROR,
                    session_id=session_id,
                    agent_id=agent_id,
                    agent_role=agent_role,
                    data={
                        "error": str(e),
                        "subtask_id": subtask_id,
                    },
                )
            )

            return {
                "subtask_results": [
                    SubtaskResult(
                        subtask_id=subtask_id,
                        agent_id=agent_id,
                        sandbox_id=sandbox_id,
                        files_produced={},
                        dependencies={},
                        dev_dependencies={},
                        type_additions="",
                        edited_files=[],
                        build_success=False,
                        build_output=str(e),
                        lint_output="",
                        summary=f"Failed with error: {e}",
                        status="failed",
                        retries=retries,
                    )
                ],
                "failed_agents": [agent_id],
            }
        finally:
            if sandbox_created:
                try:
                    await self.sandbox_manager.destroy_sandbox(sandbox_id)
                except KeyError:
                    pass
                except Exception as cleanup_error:
                    logger.warning(
                        "subtask_sandbox_cleanup_failed",
                        session_id=session_id,
                        subtask_id=subtask_id,
                        sandbox_id=sandbox_id,
                        error=str(cleanup_error),
                    )

    async def _run_subtask_react_loop(
        self,
        subtask: dict[str, Any],
        sandbox_id: str,
        session_id: str,
        agent_id: str,
        agent_role: str,
        shared_types: str,
        build_errors: str | None = None,
    ) -> dict[str, Any]:
        """Run a mini ReAct loop for a subtask.

        This is a simplified version of the full ReAct agent that:
        1. Uses subtask-specific prompt
        2. Has max 8 iterations
        3. Runs build at the end

        Args:
            subtask: The subtask definition
            sandbox_id: Sandbox to work in
            session_id: Session ID for events
            agent_id: Agent ID for events
            agent_role: Agent role name
            shared_types: Content of src/types.ts
            build_errors: Optional build errors from a previous attempt to prepend

        Returns:
            Dict with status, iterations, build_success, build_output, summary
        """
        # Use complexity-based iteration limits
        complexity = subtask.get("complexity", "medium")
        complexity_limits = {"low": 4, "medium": 8, "high": 12}
        max_iterations = complexity_limits.get(complexity, settings.max_subtask_iterations)

        description = subtask.get("description", "")
        files_responsible = subtask.get("files_responsible", [])

        # Create subtask-specific prompt
        system_prompt = get_subtask_prompt(description, files_responsible, shared_types)

        task_content = f"Complete your assigned subtask:\n\n{description}"

        # If retrying after build failure, prepend the errors
        if build_errors:
            error_categories = classify_build_errors(build_errors)
            category_summary = "\n".join(
                f"  {cat}: {len(errs)} errors" for cat, errs in error_categories.items()
            )
            task_content = (
                f"RETRY: The previous build failed. Fix these errors:\n\n"
                f"Error categories:\n{category_summary}\n\n"
                f"Build output:\n```\n{build_errors[:2000]}\n```\n\n"
                f"Original task: {description}"
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_content},
        ]

        tool_executor = ToolExecutor(
            sandbox_manager=self.sandbox_manager,
            event_bus=self.event_bus,
            metrics_collector=self.llm_client.metrics_collector,
            auto_preview=False,
        )

        iteration = 0
        summary = ""
        edited_files: set[str] = set()

        while iteration < max_iterations:
            iteration += 1

            # Prune context before LLM call
            pruned_messages = sliding_window_prune(messages)

            # Call LLM
            response = await self.llm_client.call(
                messages=pruned_messages,
                tools=get_tool_definitions_for_llm(),
                session_id=session_id,
                agent_id=agent_id,
                temperature=self.sub_agent_temperature,
            )

            # Emit thinking
            plan = parse_plan_tag(response.content) or response.content
            await self._emit_thinking(session_id, agent_id, agent_role, plan)

            # Check for explicit completion status first, then legacy fallback.
            status_tag = parse_status_tag(response.content)
            is_complete_signal = status_tag == "TASK_COMPLETE"
            if status_tag is None:
                is_complete_signal = "task_complete" in response.content.lower()

            if is_complete_signal and not response.tool_calls:
                summary = response.content
                break

            # Add assistant message to history
            assistant_message = format_assistant_message_with_tools(
                response.content, response.tool_calls
            )
            messages.append(assistant_message)

            # Execute tool calls if any
            if response.tool_calls:
                for tc_data in response.tool_calls:
                    tc = ToolCall(id=tc_data.id, name=tc_data.name, args=tc_data.args)
                    if tc.name == "write_file":
                        edited_path = self._normalize_repo_path(tc.args.get("path"))
                        if edited_path:
                            edited_files.add(edited_path)

                    result = await tool_executor.execute_tool_call(
                        sandbox_id=sandbox_id,
                        tool_call=tc,
                        session_id=session_id,
                        agent_id=agent_id,
                        agent_role=agent_role,
                    )

                    messages.append(format_tool_result_for_llm(tc.id, result.content))
                    if not result.success:
                        messages.append({
                            "role": "user",
                            "content": (
                                "The previous tool call failed. "
                                "Analyze the error, identify the "
                                "root cause, and adjust your "
                                "approach. Do NOT repeat the "
                                "same action."
                            ),
                        })
            else:
                # No tool calls and not complete, ask to continue
                messages.append({
                    "role": "user",
                    "content": (
                        "Please continue with your "
                        "implementation. Use tools to "
                        "write files."
                    ),
                })

            # Periodic build/lint check every 4 iterations to catch issues early
            if iteration > 0 and iteration % 4 == 0 and edited_files:
                build_check = await self.sandbox_manager.execute_command(
                    sandbox_id, "npm run build", timeout=60
                )
                if build_check.exit_code != 0 and not build_check.timed_out:
                    mid_loop_errors = build_check.stdout + "\n" + build_check.stderr
                    error_summary = "\n".join(
                        f"  {cat}: {len(errs)} errors"
                        for cat, errs in classify_build_errors(mid_loop_errors).items()
                    ) or mid_loop_errors[:500]
                    messages.append({
                        "role": "user",
                        "content": (
                            f"BUILD CHECK FAILED at iteration {iteration}. Fix these errors:\n"
                            f"{error_summary}\n\nBuild output:\n{mid_loop_errors[:1500]}"
                        ),
                    })

                # Only run lint if build passed or timed out (no point linting broken code)
                if build_check.exit_code == 0 or build_check.timed_out:
                    lint_check = await self.sandbox_manager.execute_command(
                        sandbox_id, "npm run lint || true", timeout=30
                    )
                    if lint_check.exit_code != 0 and not lint_check.timed_out:
                        lint_output = lint_check.stdout + "\n" + lint_check.stderr
                        if "error" in lint_output.lower():
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"LINT CHECK at iteration {iteration} found errors. "
                                    f"Fix lint errors before continuing:\n{lint_output[:1500]}"
                                ),
                            })

        # Run final build
        build_result = await self.sandbox_manager.execute_command(
            sandbox_id, "npm run build", timeout=60
        )
        build_success = build_result.exit_code == 0
        build_output = build_result.stdout + "\n" + build_result.stderr

        # Run final lint check (only if build succeeded)
        lint_output = ""
        if build_success:
            lint_result = await self.sandbox_manager.execute_command(
                sandbox_id, "npm run lint || true", timeout=30
            )
            lint_output = lint_result.stdout + "\n" + lint_result.stderr

        status = "complete" if build_success else "failed"

        return {
            "status": status,
            "iterations": iteration,
            "build_success": build_success,
            "build_output": build_output[:2000],  # Truncate for storage
            "lint_output": lint_output[:2000],  # Truncate for storage
            "summary": summary[:500] if summary else "Subtask completed",
            "edited_files": sorted(edited_files),
        }

    async def _collect_sandbox_files(
        self, sandbox_id: str, find_timeout: int = 30
    ) -> dict[str, str]:
        """Collect all files from a sandbox's src directory.

        Args:
            sandbox_id: The sandbox to collect from

        Returns:
            Dict mapping path -> content
        """
        files = {}

        # List files recursively in src directory
        try:
            result = await self.sandbox_manager.execute_command(
                sandbox_id,
                r"find src -type f \( -name '*.tsx' -o -name '*.ts' -o -name '*.css' \)",
                timeout=find_timeout,
            )

            if result.exit_code == 0 and result.stdout.strip():
                file_paths = result.stdout.strip().split("\n")

                for path in file_paths:
                    path = path.strip()
                    if path:
                        try:
                            content = await self.sandbox_manager.read_file(
                                sandbox_id, path
                            )
                            files[path] = content
                        except Exception as e:
                            logger.warning(
                                "collect_file_failed",
                                sandbox_id=sandbox_id,
                                path=path,
                                error=str(e),
                            )
        except Exception as e:
            logger.error(
                "collect_sandbox_files_error",
                sandbox_id=sandbox_id,
                error=str(e),
            )

        return files

    async def _collect_type_additions(
        self,
        *,
        sandbox_id: str,
        shared_types: str,
    ) -> str:
        """Collect additive type blocks from a subtask sandbox's src/types.ts."""
        try:
            types_content = await self.sandbox_manager.read_file(sandbox_id, "src/types.ts")
        except FileNotFoundError:
            return ""
        except Exception as e:
            logger.warning(
                "collect_type_additions_failed",
                sandbox_id=sandbox_id,
                error=str(e),
            )
            return ""

        return self._extract_type_additions(shared_types, types_content)

    @staticmethod
    def _split_type_blocks(content: str) -> list[str]:
        """Split TypeScript content into declaration blocks separated by blank lines."""
        if not isinstance(content, str) or not content.strip():
            return []

        blocks: list[str] = []
        current: list[str] = []
        for raw_line in content.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                if current:
                    block = "\n".join(current).strip()
                    if block:
                        blocks.append(block)
                    current = []
                continue
            current.append(line)

        if current:
            block = "\n".join(current).strip()
            if block:
                blocks.append(block)

        return blocks

    @staticmethod
    def _normalize_type_block(block: str) -> str:
        """Normalize a type block for stable deduplication."""
        return "\n".join(line.strip() for line in block.splitlines() if line.strip())

    def _extract_type_additions(
        self,
        shared_types: str,
        subtask_types: str,
    ) -> str:
        """Extract additive type blocks that are not already in shared types."""
        shared_blocks = self._split_type_blocks(shared_types)
        subtask_blocks = self._split_type_blocks(subtask_types)
        if not subtask_blocks:
            return ""

        seen = {
            self._normalize_type_block(block)
            for block in shared_blocks
            if self._normalize_type_block(block)
        }
        additions: list[str] = []

        for block in subtask_blocks:
            normalized = self._normalize_type_block(block)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            additions.append(block)

        return "\n\n".join(additions).strip()

    def _merge_shared_types(
        self,
        base_types: str,
        successful_results: list[SubtaskResult],
    ) -> tuple[str, int]:
        """Merge additive type blocks from successful subtasks into shared types."""
        merged_blocks = self._split_type_blocks(base_types)
        seen = {
            self._normalize_type_block(block)
            for block in merged_blocks
            if self._normalize_type_block(block)
        }
        added_count = 0

        ordered_results = sorted(
            successful_results,
            key=lambda result: str(result.get("agent_id", "")),
        )
        for result in ordered_results:
            additions = str(result.get("type_additions", ""))
            for block in self._split_type_blocks(additions):
                normalized = self._normalize_type_block(block)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                merged_blocks.append(block)
                added_count += 1

        merged_content = "\n\n".join(merged_blocks).strip()
        if merged_content:
            merged_content += "\n"

        return merged_content, added_count

    async def _sync_integration_shared_types(
        self,
        *,
        integration_sandbox_id: str,
        shared_types: str,
        successful_results: list[SubtaskResult],
        session_id: str,
        aggregator_id: str,
    ) -> dict[str, Any]:
        """Merge and persist additive shared types into the integration sandbox."""
        integration_types = ""
        try:
            integration_types = await self.sandbox_manager.read_file(
                integration_sandbox_id, "src/types.ts"
            )
        except FileNotFoundError:
            integration_types = ""
        except Exception as e:
            logger.warning(
                "integration_types_read_failed",
                session_id=session_id,
                sandbox_id=integration_sandbox_id,
                error=str(e),
            )

        base_types = integration_types or shared_types
        merged_types, type_blocks_added = self._merge_shared_types(
            base_types=base_types,
            successful_results=successful_results,
        )

        if type_blocks_added == 0 or not merged_types:
            return {"type_blocks_added": 0}

        await self.sandbox_manager.write_file(
            integration_sandbox_id, "src/types.ts", merged_types
        )
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.FILE_CHANGED,
                session_id=session_id,
                agent_id=aggregator_id,
                agent_role="Aggregator",
                data={
                    "path": "src/types.ts",
                    "content": merged_types,
                    "sandbox_id": integration_sandbox_id,
                },
            )
        )

        logger.info(
            "integration_shared_types_synced",
            session_id=session_id,
            type_blocks_added=type_blocks_added,
        )
        return {"type_blocks_added": type_blocks_added}

    async def _collect_dependency_requirements(
        self, sandbox_id: str
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Collect dependency requirements declared in a sandbox package.json.

        Returns:
            Tuple of (dependencies, dev_dependencies). Both maps are sanitized
            to string->string entries only.
        """
        try:
            package_json = await self.sandbox_manager.read_file(sandbox_id, "package.json")
        except FileNotFoundError:
            return {}, {}
        except Exception as e:
            logger.warning(
                "collect_dependencies_read_failed",
                sandbox_id=sandbox_id,
                error=str(e),
            )
            return {}, {}

        try:
            parsed = json.loads(package_json)
        except json.JSONDecodeError:
            logger.warning(
                "collect_dependencies_invalid_json",
                sandbox_id=sandbox_id,
            )
            return {}, {}

        if not isinstance(parsed, dict):
            return {}, {}

        return (
            self._normalize_dependency_map(parsed.get("dependencies")),
            self._normalize_dependency_map(parsed.get("devDependencies")),
        )

    @staticmethod
    def _normalize_dependency_map(raw: Any) -> dict[str, str]:
        """Return a sanitized dependency map from untyped package.json data."""
        if not isinstance(raw, dict):
            return {}

        normalized: dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            pkg = key.strip()
            version = value.strip()
            if pkg and version:
                normalized[pkg] = version
        return normalized

    async def _sync_integration_dependencies(
        self,
        *,
        integration_sandbox_id: str,
        successful_results: list[SubtaskResult],
        session_id: str,
        aggregator_id: str,
    ) -> dict[str, Any]:
        """Merge sub-agent dependency requirements into integration package.json.

        Uses deterministic, stable conflict handling:
        - Existing integration package.json versions are preserved.
        - New packages from successful sub-agents are unioned in.
        - Conflicts are logged and reported but do not fail aggregation.
        """
        try:
            package_raw = await self.sandbox_manager.read_file(
                integration_sandbox_id, "package.json"
            )
        except FileNotFoundError:
            logger.warning(
                "integration_package_json_missing",
                session_id=session_id,
                sandbox_id=integration_sandbox_id,
            )
            return {"dependencies_added": 0, "dependency_conflicts": []}
        except Exception as e:
            logger.warning(
                "integration_package_json_read_failed",
                session_id=session_id,
                sandbox_id=integration_sandbox_id,
                error=str(e),
            )
            return {"dependencies_added": 0, "dependency_conflicts": []}

        try:
            parsed = json.loads(package_raw)
        except json.JSONDecodeError:
            logger.warning(
                "integration_package_json_invalid_json",
                session_id=session_id,
                sandbox_id=integration_sandbox_id,
            )
            return {"dependencies_added": 0, "dependency_conflicts": []}

        if not isinstance(parsed, dict):
            return {"dependencies_added": 0, "dependency_conflicts": []}

        base_dependencies = self._normalize_dependency_map(parsed.get("dependencies"))
        base_dev_dependencies = self._normalize_dependency_map(parsed.get("devDependencies"))
        merged_dependencies = dict(base_dependencies)
        merged_dev_dependencies = dict(base_dev_dependencies)

        added_count = 0
        conflict_messages: list[str] = []

        ordered_results = sorted(
            successful_results,
            key=lambda result: str(result.get("agent_id", "")),
        )

        for result in ordered_results:
            agent_id = str(result.get("agent_id", "unknown"))

            dep_map = self._normalize_dependency_map(result.get("dependencies"))
            for pkg in sorted(dep_map.keys()):
                version = dep_map[pkg]
                existing = merged_dependencies.get(pkg)
                if existing is None:
                    merged_dependencies[pkg] = version
                    added_count += 1
                elif existing != version:
                    conflict_messages.append(
                        f"dependencies:{pkg} keep={existing} ignore={version} source={agent_id}"
                    )

            dev_dep_map = self._normalize_dependency_map(result.get("dev_dependencies"))
            for pkg in sorted(dev_dep_map.keys()):
                version = dev_dep_map[pkg]
                existing = merged_dev_dependencies.get(pkg)
                if existing is None:
                    merged_dev_dependencies[pkg] = version
                    added_count += 1
                elif existing != version:
                    conflict_messages.append(
                        f"devDependencies:{pkg} keep={existing} ignore={version} source={agent_id}"
                    )

        if (
            merged_dependencies == base_dependencies
            and merged_dev_dependencies == base_dev_dependencies
        ):
            return {
                "dependencies_added": 0,
                "dependency_conflicts": conflict_messages,
            }

        parsed["dependencies"] = dict(sorted(merged_dependencies.items()))
        parsed["devDependencies"] = dict(sorted(merged_dev_dependencies.items()))

        updated_package = json.dumps(parsed, indent=2) + "\n"
        await self.sandbox_manager.write_file(
            integration_sandbox_id, "package.json", updated_package
        )

        await self.event_bus.publish(
            AgentEvent(
                type=EventType.FILE_CHANGED,
                session_id=session_id,
                agent_id=aggregator_id,
                agent_role="Aggregator",
                data={
                    "path": "package.json",
                    "content": updated_package,
                    "sandbox_id": integration_sandbox_id,
                },
            )
        )

        if conflict_messages:
            logger.warning(
                "integration_dependency_conflicts",
                session_id=session_id,
                conflict_count=len(conflict_messages),
            )

        logger.info(
            "integration_dependencies_synced",
            session_id=session_id,
            added_count=added_count,
        )

        return {
            "dependencies_added": added_count,
            "dependency_conflicts": conflict_messages,
        }

    async def _merge_conflicting_files(
        self,
        path: str,
        sources: list[tuple[str, str]],
        session_id: str,
        aggregator_id: str,
    ) -> str:
        """Merge conflicting file versions using LLM.

        When multiple sub-agents produce different versions of the same file,
        this method uses the LLM to intelligently merge them. Falls back to
        last-writer-wins if the LLM merge fails.

        Args:
            path: The conflicting file path
            sources: List of (agent_id, content) tuples
            session_id: Session ID for LLM call
            aggregator_id: Aggregator agent ID

        Returns:
            Merged file content
        """
        try:
            versions_text = ""
            for agent_id, content in sources:
                versions_text += f"\n--- Version from {agent_id} ---\n{content[:3000]}\n"

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a code merger. Given multiple versions of the same file "
                        "produced by different agents, merge them into a single coherent file. "
                        "Combine imports, merge exports, preserve all unique functionality. "
                        "Output ONLY the merged file content, no explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Merge these versions of {path}:\n{versions_text}",
                },
            ]

            response = await self.llm_client.call(
                messages=messages,
                temperature=0.3,
                session_id=session_id,
                agent_id=aggregator_id,
            )

            if response.content.strip():
                logger.info(
                    "merge_conflict_resolved_by_llm",
                    path=path,
                    session_id=session_id,
                )
                return response.content.strip()

        except Exception as e:
            logger.warning(
                "merge_conflict_llm_failed",
                path=path,
                error=str(e),
                session_id=session_id,
            )

        # Fallback: use last version (last-writer-wins)
        return sources[-1][1]

    # -------------------------------------------------------------------------
    # Node: Aggregate
    # -------------------------------------------------------------------------

    async def _aggregate(self, state: DecompositionState) -> dict[str, Any]:
        """Aggregate node: merge all sub-agent files into integration sandbox.

        This node:
        1. Collects results from all sub-agents
        2. Copies files to integration sandbox
        3. Handles any conflicts (last-writer-wins for non-scaffold files)

        Args:
            state: Current graph state with subtask_results

        Returns:
            Dict with merged_files, failed_agents, status
        """
        session_id = state["session_id"]
        aggregator_id = f"aggregator_{session_id[5:]}"
        integration_sandbox_id = state["integration_sandbox_id"]

        await self._emit_node_active(
            session_id, aggregator_id, "Aggregator", "aggregate"
        )

        subtask_results = state.get("subtask_results", [])

        # Guard: no results at all (all subtask executions crashed)
        if not subtask_results:
            logger.error(
                "aggregate_no_results",
                session_id=session_id,
            )
            await self._emit_node_complete(
                session_id, aggregator_id, "Aggregator", "aggregate"
            )
            return {
                "failed_agents": [],
                "status": "failed",
                "error_message": "No subtask produced results.",
            }

        # Emit AGGREGATION_STARTED
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.AGGREGATION_STARTED,
                session_id=session_id,
                agent_id=aggregator_id,
                agent_role="Aggregator",
                data={
                    "subtask_count": len(subtask_results),
                },
            )
        )

        logger.info(
            "aggregate_start",
            session_id=session_id,
            subtask_count=len(subtask_results),
        )

        merged_files: list[str] = []
        failed_agents: list[str] = []
        conflicts_resolved = 0
        dependencies_added = 0
        type_blocks_added = 0
        dependency_conflicts: list[str] = []

        # Collect successful results
        successful_results = [r for r in subtask_results if r.get("status") == "complete"]
        failed_results = [r for r in subtask_results if r.get("status") != "complete"]

        for result in failed_results:
            failed_agents.append(result.get("agent_id", "unknown"))

        # Check if all subtasks failed — only reject if none succeeded
        if not successful_results:
            logger.error(
                "aggregate_all_failed",
                session_id=session_id,
                failed_count=len(failed_agents),
                total_count=len(subtask_results),
            )
            await self._emit_node_complete(
                session_id, aggregator_id, "Aggregator", "aggregate"
            )
            return {
                "failed_agents": failed_agents,
                "status": "failed",
                "error_message": (
                    f"All sub-agents failed: "
                    f"{len(failed_agents)}/{len(subtask_results)}"
                ),
            }

        # Log partial success if some subtasks failed
        if failed_agents:
            logger.warning(
                "aggregate_partial_success",
                session_id=session_id,
                failed_count=len(failed_agents),
                total_count=len(subtask_results),
                successful_count=len(successful_results),
            )

        already_synced_ids = {
            subtask_id
            for subtask_id in state.get("synced_subtask_ids", [])
            if isinstance(subtask_id, str)
        }
        pending_successful_results = [
            result
            for result in successful_results
            if result.get("subtask_id") not in already_synced_ids
        ]

        if pending_successful_results:
            merge_result = await self._merge_successful_results_into_integration(
                successful_results=pending_successful_results,
                integration_sandbox_id=integration_sandbox_id,
                session_id=session_id,
                aggregator_id=aggregator_id,
                shared_types=state.get("shared_types", ""),
            )
            merged_files = merge_result["merged_files"]
            conflicts_resolved = merge_result["conflicts_resolved"]
            dependencies_added = merge_result["dependencies_added"]
            type_blocks_added = merge_result["type_blocks_added"]
            dependency_conflicts = merge_result["dependency_conflicts"]

        # Emit AGGREGATION_COMPLETE
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.AGGREGATION_COMPLETE,
                session_id=session_id,
                agent_id=aggregator_id,
                agent_role="Aggregator",
                data={
                    "files_merged": len(merged_files),
                    "conflicts_resolved": conflicts_resolved,
                    "dependencies_added": dependencies_added,
                    "type_blocks_added": type_blocks_added,
                    # Keep the count for dashboards while exposing a bounded sample
                    # for debuggability in event logs.
                    "dependency_conflict_count": len(dependency_conflicts),
                    "dependency_conflicts": dependency_conflicts[:20],
                },
            )
        )

        logger.info(
            "aggregate_complete",
            session_id=session_id,
            files_merged=len(merged_files),
            conflicts_resolved=conflicts_resolved,
            dependencies_added=dependencies_added,
            type_blocks_added=type_blocks_added,
            dependency_conflict_count=len(dependency_conflicts),
        )

        await self._emit_node_complete(
            session_id, aggregator_id, "Aggregator", "aggregate"
        )

        return {
            "merged_files": merged_files,
            "failed_agents": failed_agents,
            "status": "integrating",
        }

    def _find_entrypoint_wiring_gap(self, state: DecompositionState) -> str | None:
        """Return a human-readable issue when App.tsx integration wiring is missing."""
        subtasks = state.get("subtasks", [])
        if len(subtasks) <= 1:
            return None

        app_subtask_ids = {
            str(subtask.get("id", ""))
            for subtask in subtasks
            if APP_ENTRYPOINT_PATH in subtask.get("files_responsible", [])
        }
        if not app_subtask_ids:
            return (
                f"No subtask owns {APP_ENTRYPOINT_PATH}; "
                "final UI assembly is not guaranteed."
            )

        successful_results = [
            result
            for result in state.get("subtask_results", [])
            if result.get("status") == "complete"
        ]
        successful_ids = {
            str(result.get("subtask_id", ""))
            for result in successful_results
            if isinstance(result.get("subtask_id"), str)
        }
        non_app_successful = successful_ids - app_subtask_ids
        if not non_app_successful:
            return None

        missing_app_results = sorted(app_subtask_ids - successful_ids)
        if missing_app_results:
            missing = ", ".join(missing_app_results)
            return (
                "App assembly subtask did not complete while peer subtasks did. "
                f"Missing completion for: {missing}"
            )

        edited_files = {
            self._normalize_repo_path(path)
            for result in successful_results
            for path in result.get("edited_files", [])
            if isinstance(path, str)
        }
        if APP_ENTRYPOINT_PATH not in edited_files:
            return (
                "Subtasks completed, but none explicitly edited src/App.tsx. "
                "This often leaves generated UI disconnected from the entrypoint."
            )

        return None

    async def _attempt_entrypoint_fix(
        self,
        state: DecompositionState,
        issue: str,
        integration_id: str,
    ) -> bool:
        """Attempt to repair missing App.tsx integration wiring via targeted edits."""
        session_id = state["session_id"]
        integration_sandbox_id = state["integration_sandbox_id"]
        before_app = ""

        try:
            before_app = await self.sandbox_manager.read_file(
                integration_sandbox_id, APP_ENTRYPOINT_PATH
            )
        except Exception:
            before_app = ""

        successful_subtask_summaries = []
        for result in state.get("subtask_results", []):
            if result.get("status") != "complete":
                continue
            subtask_id = str(result.get("subtask_id", ""))
            edited_files = [
                self._normalize_repo_path(path)
                for path in result.get("edited_files", [])
                if isinstance(path, str)
            ]
            cleaned_files = [path for path in edited_files if path]
            successful_subtask_summaries.append(
                f"- {subtask_id}: "
                f"{', '.join(cleaned_files) if cleaned_files else '(no tracked edits)'}"
            )
        subtask_summary_block = "\n".join(successful_subtask_summaries) or "- (none)"

        system_prompt = compose_prompt_sections(
            BASE_CODING_AGENT_PROMPT,
            build_session_contract(
                mode="decomposition/entrypoint-fix",
                objective=(
                    "Repair missing App.tsx integration so the generated UI is actually reachable."
                ),
            ),
            build_tooling_contract(),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Integration validation detected an entrypoint wiring issue:
{issue}

Successful subtask edits:
{subtask_summary_block}

Update src/App.tsx so it imports/renders the generated UI components and the app is usable.

Rules:
- You must edit src/App.tsx directly.
- Keep edits focused and minimal.
- After edits, run `npm run build`.
- Return TASK_COMPLETE only if the build succeeds.
""",
            },
        ]

        response = await self.llm_client.call(
            messages=messages,
            tools=get_tool_definitions_for_llm(),
            session_id=session_id,
            agent_id=integration_id,
            temperature=0.4,
        )

        await self._emit_thinking(
            session_id, integration_id, "Integration", response.content
        )

        if not response.tool_calls:
            return False

        tool_executor = ToolExecutor(
            sandbox_manager=self.sandbox_manager,
            event_bus=self.event_bus,
            metrics_collector=self.llm_client.metrics_collector,
        )
        wrote_entrypoint = False

        for tc_data in response.tool_calls:
            tc = ToolCall(id=tc_data.id, name=tc_data.name, args=tc_data.args)
            if tc.name == "write_file":
                edited_path = self._normalize_repo_path(tc.args.get("path"))
                if edited_path == APP_ENTRYPOINT_PATH:
                    wrote_entrypoint = True
            await tool_executor.execute_tool_call(
                sandbox_id=integration_sandbox_id,
                tool_call=tc,
                session_id=session_id,
                agent_id=integration_id,
                agent_role="Integration",
            )

        if not wrote_entrypoint:
            logger.warning(
                "entrypoint_fix_skipped_app_write",
                session_id=session_id,
                agent_id=integration_id,
            )
            return False

        try:
            after_app = await self.sandbox_manager.read_file(
                integration_sandbox_id, APP_ENTRYPOINT_PATH
            )
        except Exception:
            return False

        if after_app.strip() == before_app.strip():
            logger.warning(
                "entrypoint_fix_no_app_change",
                session_id=session_id,
                agent_id=integration_id,
            )
            return False

        build_result = await self.sandbox_manager.execute_command(
            integration_sandbox_id, "npm run build", timeout=120
        )
        return build_result.exit_code == 0

    # -------------------------------------------------------------------------
    # Node: Integration Review
    # -------------------------------------------------------------------------

    async def _integration_review(self, state: DecompositionState) -> dict[str, Any]:
        """Integration review node: build and test the merged code.

        This node:
        1. Runs npm install
        2. Runs npm run build
        3. If errors, attempts to fix (up to 2 retries)
        4. Starts preview server on success

        Args:
            state: Current graph state

        Returns:
            Dict with final_build_success, final_preview_url, integration_retries, status
        """
        session_id = state["session_id"]
        integration_id = f"integration_{session_id[5:]}"
        integration_sandbox_id = state["integration_sandbox_id"]
        retries = state.get("integration_retries", 0)

        await self._emit_node_active(
            session_id, integration_id, "Integration", "integration_review"
        )

        logger.info(
            "integration_review_start",
            session_id=session_id,
            retry_attempt=retries,
        )

        # Run npm install
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.COMMAND_STARTED,
                session_id=session_id,
                agent_id=integration_id,
                agent_role="Integration",
                data={"command": "npm install", "sandbox_id": integration_sandbox_id},
            )
        )

        install_result = await self.sandbox_manager.execute_command(
            integration_sandbox_id, "npm install", timeout=120
        )

        await self.event_bus.publish(
            AgentEvent(
                type=EventType.COMMAND_COMPLETE,
                session_id=session_id,
                agent_id=integration_id,
                agent_role="Integration",
                data={
                    "command": "npm install",
                    "exit_code": install_result.exit_code,
                    "sandbox_id": integration_sandbox_id,
                },
            )
        )

        if install_result.exit_code != 0:
            logger.error(
                "integration_npm_install_failed",
                session_id=session_id,
                exit_code=install_result.exit_code,
            )

        retries_used = retries
        entrypoint_gap = self._find_entrypoint_wiring_gap(state)
        if entrypoint_gap:
            logger.warning(
                "integration_entrypoint_wiring_gap",
                session_id=session_id,
                issue=entrypoint_gap,
                retry_attempt=retries + 1,
            )
            await self._emit_thinking(
                session_id,
                integration_id,
                "Integration",
                f"Entrypoint validation issue: {entrypoint_gap}",
            )

            if retries >= 2:
                await self._emit_node_complete(
                    session_id, integration_id, "Integration", "integration_review"
                )
                return {
                    "final_build_success": False,
                    "final_preview_url": None,
                    "integration_retries": retries,
                    "status": "failed",
                    "error_message": entrypoint_gap,
                }

            try:
                fix_result = await asyncio.wait_for(
                    self._attempt_entrypoint_fix(state, entrypoint_gap, integration_id),
                    timeout=90,
                )
            except TimeoutError:
                logger.warning(
                    "integration_entrypoint_fix_timeout",
                    session_id=session_id,
                    retry_attempt=retries + 1,
                )
                fix_result = False

            retries_used = retries + 1
            retries = retries_used
            if not fix_result:
                await self._emit_node_complete(
                    session_id, integration_id, "Integration", "integration_review"
                )
                return {
                    "integration_retries": retries_used,
                    "final_build_success": False,
                    "status": "failed" if retries_used >= 2 else "integrating",
                    "error_message": entrypoint_gap,
                }

        # Run npm run build
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.COMMAND_STARTED,
                session_id=session_id,
                agent_id=integration_id,
                agent_role="Integration",
                data={"command": "npm run build", "sandbox_id": integration_sandbox_id},
            )
        )

        build_result = await self.sandbox_manager.execute_command(
            integration_sandbox_id, "npm run build", timeout=120
        )

        build_output = build_result.stdout + "\n" + build_result.stderr

        await self.event_bus.publish(
            AgentEvent(
                type=EventType.COMMAND_OUTPUT,
                session_id=session_id,
                agent_id=integration_id,
                agent_role="Integration",
                data={"output": build_output[:2000], "stream": "stdout"},
            )
        )

        await self.event_bus.publish(
            AgentEvent(
                type=EventType.COMMAND_COMPLETE,
                session_id=session_id,
                agent_id=integration_id,
                agent_role="Integration",
                data={
                    "command": "npm run build",
                    "exit_code": build_result.exit_code,
                    "sandbox_id": integration_sandbox_id,
                },
            )
        )

        build_success = build_result.exit_code == 0

        if not build_success and retries < 2:
            # Attempt to fix errors with LLM (with timeout)
            logger.info(
                "integration_review_attempting_fix",
                session_id=session_id,
                retry_attempt=retries + 1,
            )

            try:
                fix_result = await asyncio.wait_for(
                    self._attempt_build_fix(state, build_output, integration_id),
                    timeout=90,
                )
            except TimeoutError:
                logger.warning(
                    "integration_build_fix_timeout",
                    session_id=session_id,
                    retry_attempt=retries + 1,
                )
                fix_result = False

            retries_used = retries + 1
            if fix_result:
                logger.info(
                    "integration_build_fix_succeeded",
                    session_id=session_id,
                    retry_attempt=retries_used,
                )
                build_success = True
            else:
                await self._emit_node_complete(
                    session_id, integration_id, "Integration", "integration_review"
                )

                will_exhaust_retries = retries_used >= 2
                return {
                    "integration_retries": retries_used,
                    "final_build_success": False,
                    "status": "failed" if will_exhaust_retries else "integrating",
                }

        # Build succeeded or max retries reached
        if build_success:
            # Start preview server
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.PREVIEW_STARTING,
                    session_id=session_id,
                    agent_id=integration_id,
                    agent_role="Integration",
                    data={"sandbox_id": integration_sandbox_id},
                )
            )

            try:
                preview_url = await self.sandbox_manager.start_dev_server(
                    integration_sandbox_id
                )

                await self.event_bus.publish(
                    AgentEvent(
                        type=EventType.PREVIEW_READY,
                        session_id=session_id,
                        agent_id=integration_id,
                        agent_role="Integration",
                        data={
                            "url": preview_url,
                            "sandbox_id": integration_sandbox_id,
                        },
                    )
                )

                logger.info(
                    "integration_review_preview_ready",
                    session_id=session_id,
                    preview_url=preview_url,
                )

                await self._emit_node_complete(
                    session_id, integration_id, "Integration", "integration_review"
                )

                return {
                    "final_build_success": True,
                    "final_preview_url": preview_url,
                    "integration_retries": retries_used,
                    "status": "complete",
                }

            except Exception as e:
                logger.error(
                    "integration_review_preview_failed",
                    session_id=session_id,
                    error=str(e),
                )

                await self.event_bus.publish(
                    AgentEvent(
                        type=EventType.PREVIEW_ERROR,
                        session_id=session_id,
                        agent_id=integration_id,
                        agent_role="Integration",
                        data={"error": str(e)},
                    )
                )

        await self._emit_node_complete(
            session_id, integration_id, "Integration", "integration_review"
        )

        return {
            "final_build_success": build_success,
            "final_preview_url": None,
            "integration_retries": retries_used,
            "status": "complete" if build_success else "failed",
            "error_message": None if build_success else "Build failed after all retries",
        }

    async def _attempt_build_fix(
        self,
        state: DecompositionState,
        build_output: str,
        integration_id: str,
    ) -> bool:
        """Attempt to fix build errors using LLM.

        Args:
            state: Current graph state
            build_output: The error output from build
            integration_id: Integration agent ID

        Returns:
            True if fix was successful
        """
        session_id = state["session_id"]
        integration_sandbox_id = state["integration_sandbox_id"]

        # Create fix prompt
        system_prompt = compose_prompt_sections(
            BASE_CODING_AGENT_PROMPT,
            build_session_contract(
                mode="decomposition/integration-fix",
                objective="Fix integration build failures with minimal, targeted edits.",
            ),
            build_tooling_contract(),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""The build failed with these errors:

```
{build_output[:3000]}
```

Please analyze the errors and fix them. Use the write_file tool to update the necessary files.""",
            },
        ]

        response = await self.llm_client.call(
            messages=messages,
            tools=get_tool_definitions_for_llm(),
            session_id=session_id,
            agent_id=integration_id,
            temperature=0.5,
        )

        await self._emit_thinking(
            session_id, integration_id, "Integration", response.content
        )

        # Execute any tool calls
        if response.tool_calls:
            tool_executor = ToolExecutor(
                sandbox_manager=self.sandbox_manager,
                event_bus=self.event_bus,
                metrics_collector=self.llm_client.metrics_collector,
            )

            for tc_data in response.tool_calls:
                tc = ToolCall(id=tc_data.id, name=tc_data.name, args=tc_data.args)
                await tool_executor.execute_tool_call(
                    sandbox_id=integration_sandbox_id,
                    tool_call=tc,
                    session_id=session_id,
                    agent_id=integration_id,
                    agent_role="Integration",
                )

            # Re-run build to check
            build_result = await self.sandbox_manager.execute_command(
                integration_sandbox_id, "npm run build", timeout=120
            )

            return build_result.exit_code == 0

        return False

    # -------------------------------------------------------------------------
    # Conditional Edge: Should Retry Integration
    # -------------------------------------------------------------------------

    def _should_retry_integration(self, state: DecompositionState) -> str:
        """Determine whether to retry integration or end.

        Args:
            state: Current graph state

        Returns:
            "retry" to loop back, "end" to finish
        """
        if state.get("status") in ("complete", "failed"):
            return "end"

        if state.get("final_build_success"):
            return "end"

        retries = state.get("integration_retries", 0)
        if retries >= 2:
            return "end"

        return "retry"

    # -------------------------------------------------------------------------
    # Public Methods: run() and stream()
    # -------------------------------------------------------------------------

    async def run(self, initial_state: DecompositionState) -> DecompositionState:
        """Run the graph to completion.

        Args:
            initial_state: The starting state

        Returns:
            The final state after completion
        """
        self._session_start_time = time.monotonic()

        # Emit graph initialized event
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.GRAPH_INITIALIZED,
                session_id=initial_state["session_id"],
                agent_id="decomposition_graph",
                agent_role="Decomposition Graph",
                data={
                    "nodes": [
                        "orchestrate",
                        "dispatch_layer",
                        "execute_subtask",
                        "advance_layer",
                        "aggregate",
                        "integration_review",
                    ],
                    "edges": [
                        {"source": "START", "target": "orchestrate"},
                        {"source": "orchestrate", "target": "dispatch_layer"},
                        {"source": "dispatch_layer", "target": "execute_subtask", "parallel": True},
                        {"source": "execute_subtask", "target": "advance_layer"},
                        {
                            "source": "advance_layer",
                            "target": "dispatch_layer",
                            "condition": "more_layers",
                        },
                        {
                            "source": "advance_layer",
                            "target": "aggregate",
                            "condition": "all_layers_complete",
                        },
                        {
                            "source": "aggregate",
                            "target": "integration_review",
                            "condition": "integrate",
                        },
                        {"source": "aggregate", "target": "END", "condition": "end"},
                        {
                            "source": "integration_review",
                            "target": "END",
                            "condition": "end",
                        },
                        {
                            "source": "integration_review",
                            "target": "integration_review",
                            "condition": "retry",
                        },
                    ],
                },
            )
        )

        # Emit agent spawned for orchestrator
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.AGENT_SPAWNED,
                session_id=initial_state["session_id"],
                agent_id=f"orchestrator_{initial_state['session_id'][5:]}",
                agent_role="Orchestrator",
                data={
                    "role": "Orchestrator",
                    "sandbox_id": initial_state["integration_sandbox_id"],
                },
            )
        )

        # Run the graph
        final_state = await self._compiled_graph.ainvoke(initial_state)

        return final_state


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------


def create_decomposition_graph(
    sandbox_manager: SandboxManager,
    event_bus: EventBus,
    llm_client: LLMClient | None = None,
    orchestrator_model: str | None = None,
    sub_agent_temperature: float = 0.7,
) -> DecompositionGraph:
    """Factory function to create a Decomposition graph.

    This is the main entry point for creating a task decomposition agent.

    Args:
        sandbox_manager: Manager for sandbox operations
        event_bus: Event bus for emitting events
        llm_client: Optional LLM client (creates default if None)

    Returns:
        Configured DecompositionGraph instance
    """
    return DecompositionGraph(
        sandbox_manager=sandbox_manager,
        event_bus=event_bus,
        llm_client=llm_client,
        orchestrator_model=orchestrator_model,
        sub_agent_temperature=sub_agent_temperature,
    )
