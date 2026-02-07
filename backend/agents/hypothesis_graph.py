"""Parallel Hypothesis Testing LangGraph implementation.

This module provides the Parallel Hypothesis agent graph - where N agents
independently solve the same task, and an evaluator selects the best solution.

Graph structure:
    START -> broadcast -> [fan_out] -> solve(×N) -> evaluate -> finalize -> END

Events emitted:
- AGENT_SPAWNED: When each solver starts
- AGENT_THINKING, AGENT_TOOL_CALL, AGENT_TOOL_RESULT: During solver execution
- EVALUATION_STARTED, EVALUATION_RESULT: During evaluation
- PREVIEW_READY: When the winning solution preview is available
"""

import asyncio
import json
import operator
import time
from typing import Annotated, Any, Literal, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.prompts import (
    EVALUATOR_PROMPT,
    REVIEW_PROMPT,
    SOLVER_PERSONAS,
    SYNTHESIS_PROMPT,
    get_solver_prompt,
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
    parse_analysis_tag,
    parse_plan_tag,
    parse_status_tag,
    sliding_window_prune,
)
from config import settings
from events.bus import EventBus
from events.types import AgentEvent, EventType
from sandbox.docker_sandbox import SandboxManager

logger = structlog.get_logger()

EVALUATION_MAX_LISTED_FILES = 12
EVALUATION_MAX_FILES_WITH_CONTENT = 4
EVALUATION_MAX_FILE_CONTENT_CHARS = 1000
EVALUATION_MAX_BUILD_OUTPUT_CHARS = 500
EVALUATION_MAX_SUMMARY_CHARS = 400
APP_ENTRYPOINT_PATH = "src/App.tsx"


# -----------------------------------------------------------------------------
# State Schema Definitions
# -----------------------------------------------------------------------------


class HypothesisResult(TypedDict):
    """Result from a solver's attempt at the task.

    Attributes:
        agent_id: The solver agent identifier
        persona: The solver's persona (clarity, completeness, creativity)
        sandbox_id: The sandbox container used
        files: Dict mapping path -> content for all files written
        edited_files: Paths explicitly edited by the solver via write tools
        build_success: Whether npm run build succeeded
        build_output: Build command output
        lint_errors: Count of lint errors
        iterations_used: Number of ReAct iterations
        agent_summary: Solver's summary of what was built
    """

    agent_id: str
    persona: str
    sandbox_id: str
    files: dict[str, str]
    edited_files: list[str]
    build_success: bool
    build_output: str
    lint_errors: int
    iterations_used: int
    agent_summary: str


class EvaluationScore(TypedDict):
    """Score breakdown for a single solution.

    Attributes:
        agent_id: The solver agent identifier
        build: Build success score (0-10, 30% weight)
        lint: Lint cleanliness score (0-10, 15% weight)
        quality: Code quality score (0-10, 25% weight)
        completeness: Requirements completeness score (0-10, 20% weight)
        ux: UX/visual quality score (0-10, 10% weight)
        total: Weighted total score
        notes: Evaluator notes on this solution
    """

    agent_id: str
    build: int
    lint: int
    quality: int
    completeness: int
    ux: int
    total: float
    notes: str


class HypothesisState(TypedDict):
    """State for the Parallel Hypothesis graph.

    This state flows through the graph and accumulates information
    as solvers work and the evaluator assesses.

    Attributes:
        task: The user's original task/prompt
        session_id: Session identifier for events
        num_hypotheses: Number of parallel solvers (default 3)
        solver_personas: List of persona names for solvers
        hypothesis_results: Results from all solvers (fan-in via operator.add)
        scores: Evaluation scores for each solution
        selected_index: Index of selected winner
        selected_agent_id: Agent ID of selected winner
        evaluation_reasoning: Evaluator's reasoning
        final_sandbox_id: Sandbox with winning solution
        final_preview_url: Preview URL for winning solution
        status: Current graph status
        error_message: Error message if failed
    """

    task: str
    session_id: str
    num_hypotheses: int
    solver_personas: list[str]
    hypothesis_results: Annotated[list[HypothesisResult], operator.add]
    scores: list[EvaluationScore]
    selected_index: int
    selected_agent_id: str
    evaluation_reasoning: str
    final_sandbox_id: str
    final_preview_url: str | None
    status: Literal[
        "broadcasting", "solving", "evaluating", "finalizing", "complete", "failed"
    ]
    error_message: str | None


def create_hypothesis_initial_state(
    task: str,
    session_id: str,
    num_hypotheses: int = 3,
) -> HypothesisState:
    """Create the initial state for a hypothesis graph run.

    Args:
        task: The user's task/prompt
        session_id: Session ID for event emission
        num_hypotheses: Number of parallel solvers (default 3)

    Returns:
        Initial HypothesisState dict
    """
    # Assign personas based on num_hypotheses
    persona_names = list(SOLVER_PERSONAS.keys())[:num_hypotheses]

    return HypothesisState(
        task=task,
        session_id=session_id,
        num_hypotheses=num_hypotheses,
        solver_personas=persona_names,
        hypothesis_results=[],
        scores=[],
        selected_index=-1,
        selected_agent_id="",
        evaluation_reasoning="",
        final_sandbox_id="",
        final_preview_url=None,
        status="broadcasting",
        error_message=None,
    )


# -----------------------------------------------------------------------------
# HypothesisGraph Class
# -----------------------------------------------------------------------------


class HypothesisGraph:
    """The Parallel Hypothesis graph implementation.

    This class encapsulates the LangGraph StateGraph and provides
    the node implementations for broadcasting, solving, evaluating,
    and finalizing.

    Usage:
        >>> graph = HypothesisGraph(sandbox_manager, event_bus, llm_client)
        >>> initial_state = create_hypothesis_initial_state(task, session_id)
        >>> final_state = await graph.run(initial_state)
    """

    def __init__(
        self,
        sandbox_manager: SandboxManager,
        event_bus: EventBus,
        llm_client: LLMClient | None = None,
        evaluator_model: str | None = None,
    ) -> None:
        """Initialize the Hypothesis graph.

        Args:
            sandbox_manager: Manager for sandbox operations
            event_bus: Event bus for emitting events
            llm_client: LLM client for model calls (creates default if None)
            evaluator_model: Model used by the evaluator node.
        """
        self.sandbox_manager = sandbox_manager
        self.event_bus = event_bus
        self.llm_client = llm_client or LLMClient(event_bus=event_bus)
        self.evaluator_model = evaluator_model or settings.evaluator_model
        self._session_start_time: float = 0.0  # Set when graph starts running
        self._compiled_graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph StateGraph.

        Returns:
            Compiled StateGraph ready for execution
        """
        graph = StateGraph(HypothesisState)

        # Add nodes
        graph.add_node("broadcast", self._broadcast)
        graph.add_node("solve", self._solve)
        graph.add_node("evaluate", self._evaluate)
        graph.add_node("synthesize", self._synthesize)
        graph.add_node("finalize", self._finalize)

        # Add edges
        graph.add_edge(START, "broadcast")

        # Fan-out from broadcast to solve nodes
        graph.add_conditional_edges(
            "broadcast",
            self._fan_out_solvers,
            ["solve"],
        )

        # All solves fan-in to evaluate
        graph.add_edge("solve", "evaluate")

        # Evaluate conditionally routes to synthesis/finalize or ends on failure.
        graph.add_conditional_edges(
            "evaluate",
            self._route_after_evaluate,
            {
                "synthesize": "synthesize",
                "finalize": "finalize",
                "end": END,
            },
        )

        # Synthesize always goes to finalize
        graph.add_edge("synthesize", "finalize")

        # Finalize to end
        graph.add_edge("finalize", END)

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
        if not self._session_start_time:
            return float(settings.agent_timeout_seconds)

        elapsed = time.monotonic() - self._session_start_time
        return max(0.0, float(settings.agent_timeout_seconds) - elapsed)

    def _allocate_solver_budget_seconds(self) -> int:
        """Allocate budget for a solver within the overall session timeout."""
        remaining_budget = self._remaining_session_budget_seconds()

        # Keep room for evaluator/finalizer nodes after solver fan-in.
        post_solve_reserve_seconds = 75.0
        usable_budget = max(0.0, remaining_budget - post_solve_reserve_seconds)
        if usable_budget <= 0:
            usable_budget = max(0.0, remaining_budget - 10.0)

        budget = int(usable_budget)
        budget = max(20, min(180, budget))
        if remaining_budget > 0:
            budget = min(budget, max(5, int(remaining_budget)))
        return max(5, budget)

    @staticmethod
    def _timeout_before_deadline(
        deadline: float,
        default_timeout: int,
        *,
        reserve_seconds: int = 0,
        min_timeout: int = 10,
    ) -> int:
        """Return a timeout clipped by remaining time until deadline."""
        remaining = int(deadline - time.monotonic()) - reserve_seconds
        if remaining <= 0:
            return 0
        if remaining < min_timeout:
            return remaining
        return min(default_timeout, remaining)

    @staticmethod
    def _summarize_build_errors(build_output: str) -> str:
        """Classify build errors and return a one-line-per-category summary."""
        categories = classify_build_errors(build_output)
        if categories:
            return "\n".join(
                f"- {cat}: {len(errs)} errors"
                for cat, errs in categories.items()
            )
        # Fallback: return truncated raw output when no patterns match
        return build_output[:500] if build_output else "Unknown build error"

    @staticmethod
    def _clip_with_ellipsis(content: str, max_chars: int) -> str:
        """Clip long strings and append a compact truncation marker."""
        if not isinstance(content, str) or max_chars <= 0:
            return ""

        if len(content) <= max_chars:
            return content

        omitted_chars = len(content) - max_chars
        return f"{content[:max_chars]}\n...[truncated {omitted_chars} chars]"

    @staticmethod
    def _reasoning_requests_synthesis(reasoning: str) -> bool:
        """Return whether evaluator reasoning explicitly requests improvements."""
        if not isinstance(reasoning, str):
            return False

        normalized = " ".join(reasoning.lower().split())
        if not normalized:
            return False

        negative_phrases = [
            "no improvement",
            "no improvements",
            "no further improvement",
            "no further improvements",
            "nothing to improve",
            "no changes needed",
            "no change needed",
            "already optimal",
            "already good",
            "looks good as is",
            "looks good as-is",
        ]
        if any(phrase in normalized for phrase in negative_phrases):
            return False

        improvement_keywords = [
            "improv",
            "enhance",
            "fix",
            "missing",
            "should",
            "needs",
            "need to",
            "could add",
            "should add",
        ]
        return any(keyword in normalized for keyword in improvement_keywords)

    # -------------------------------------------------------------------------
    # Node: Broadcast
    # -------------------------------------------------------------------------

    async def _broadcast(self, state: HypothesisState) -> dict[str, Any]:
        """Broadcast node: prepare solver configurations.

        This node initializes the solver personas and prepares
        configurations for parallel execution.

        Args:
            state: Current graph state

        Returns:
            Dict with solver_personas, status
        """
        session_id = state["session_id"]
        broadcast_id = f"broadcast_{session_id[5:]}"

        await self._emit_node_active(
            session_id, broadcast_id, "Broadcast", "broadcast"
        )

        persona_names = state["solver_personas"]

        logger.info(
            "broadcast_start",
            session_id=session_id,
            num_hypotheses=len(persona_names),
            personas=persona_names,
        )

        await self._emit_node_complete(
            session_id, broadcast_id, "Broadcast", "broadcast"
        )

        return {
            "solver_personas": persona_names,
            "status": "solving",
        }

    # -------------------------------------------------------------------------
    # Fan-out Router
    # -------------------------------------------------------------------------

    def _fan_out_solvers(self, state: HypothesisState) -> list[Send]:
        """Create Send() objects for each solver.

        This is the routing function that creates parallel branches
        using LangGraph's Send() API.

        Args:
            state: Current graph state with solver_personas defined

        Returns:
            List of Send objects, one per solver
        """
        sends = []
        personas = state.get("solver_personas", ["clarity", "completeness", "creativity"])

        # Strategic temperature spread for diversity
        # clarity=0.3 (deterministic), completeness=0.7 (balanced),
        # creativity=1.0 (max exploration), test_driven=0.5, performance=0.9
        temperatures = [0.3, 0.7, 1.0, 0.5, 0.9]

        for i, persona in enumerate(personas):
            solver_state = {
                "solver_index": i,
                "persona": persona,
                "temperature": temperatures[i] if i < len(temperatures) else 0.7,
                "task": state["task"],
                "session_id": state["session_id"],
            }
            sends.append(Send("solve", solver_state))

        logger.debug(
            "fan_out_solvers",
            session_id=state["session_id"],
            solver_count=len(sends),
        )

        return sends

    # -------------------------------------------------------------------------
    # Node: Solve
    # -------------------------------------------------------------------------

    async def _solve(self, state: dict[str, Any]) -> dict[str, Any]:
        """Solve node: run full ReAct loop for a single solver.

        This node runs in parallel for each solver. It:
        1. Creates a dedicated sandbox
        2. Runs a full ReAct loop (max 15 iterations) with persona prompt
        3. Collects files produced and runs build/lint
        4. Returns a HypothesisResult

        Args:
            state: Solver-specific state from Send()

        Returns:
            Dict with hypothesis_results (will be merged via operator.add)
        """
        solver_index = state["solver_index"]
        persona = state["persona"]
        temperature = state["temperature"]
        task = state["task"]
        session_id = state["session_id"]

        agent_id = f"solver_{solver_index + 1}"
        agent_role = f"Solver {solver_index + 1} ({persona})"
        sandbox_id = f"sandbox_solver{solver_index + 1}_{session_id[5:]}"

        logger.info(
            "solve_start",
            session_id=session_id,
            solver_index=solver_index,
            persona=persona,
            temperature=temperature,
        )

        solver_budget_seconds = self._allocate_solver_budget_seconds()
        solver_deadline = time.monotonic() + solver_budget_seconds
        solver_timed_out = False

        logger.info(
            "solve_budget_allocated",
            session_id=session_id,
            solver_index=solver_index,
            persona=persona,
            budget_seconds=solver_budget_seconds,
            remaining_session_budget=round(self._remaining_session_budget_seconds(), 1),
        )

        try:
            # Create sandbox for this solver
            await self.sandbox_manager.create_sandbox(sandbox_id)

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
                        "persona": persona,
                    },
                )
            )

            # Run full ReAct loop with timeout clipped by remaining session budget.
            max_retries = 2
            retries = 0

            def timeout_result(msg: str) -> dict[str, Any]:
                return {
                    "status": "timeout",
                    "iterations": 0,
                    "build_success": False,
                    "build_output": msg,
                    "summary": msg,
                }

            solver_loop_timeout = self._timeout_before_deadline(
                solver_deadline,
                180,
                reserve_seconds=20,
                min_timeout=20,
            )
            if solver_loop_timeout <= 0:
                solver_timed_out = True
                result = timeout_result(
                    f"Solver timed out before execution (budget {solver_budget_seconds}s)"
                )
            else:
                try:
                    result = await asyncio.wait_for(
                        self._run_solver_react_loop(
                            task=task,
                            persona=persona,
                            temperature=temperature,
                            sandbox_id=sandbox_id,
                            session_id=session_id,
                            agent_id=agent_id,
                            agent_role=agent_role,
                            solver_deadline=solver_deadline,
                        ),
                        timeout=solver_loop_timeout,
                    )
                except TimeoutError:
                    solver_timed_out = True
                    logger.warning(
                        "solver_react_loop_timeout",
                        session_id=session_id,
                        solver_index=solver_index,
                        timeout=solver_loop_timeout,
                        budget_seconds=solver_budget_seconds,
                    )
                    result = timeout_result(
                        f"Solver timed out after {solver_loop_timeout}s"
                    )

            # Retry loop: if build failed, re-invoke with error context
            while (
                not result.get("build_success")
                and result.get("status") != "timeout"
                and retries < max_retries
            ):
                retries += 1
                logger.info(
                    "solver_retry",
                    session_id=session_id,
                    solver_index=solver_index,
                    retry=retries,
                    build_output=result.get("build_output", "")[:200],
                )

                error_summary = self._summarize_build_errors(
                    result.get("build_output", "")
                )
                await self._emit_thinking(
                    session_id, agent_id, agent_role,
                    f"Build failed (retry {retries}/{max_retries}). Errors:\n{error_summary}",
                )

                retry_timeout = self._timeout_before_deadline(
                    solver_deadline,
                    180,
                    reserve_seconds=20,
                    min_timeout=15,
                )
                if retry_timeout <= 0:
                    result = timeout_result(
                        f"Solver timed out before retry {retries} "
                        f"(budget {solver_budget_seconds}s)"
                    )
                    solver_timed_out = True
                    break

                # Re-invoke react loop with build errors prepended
                try:
                    result = await asyncio.wait_for(
                        self._run_solver_react_loop(
                            task=task,
                            persona=persona,
                            temperature=temperature,
                            sandbox_id=sandbox_id,
                            session_id=session_id,
                            agent_id=agent_id,
                            agent_role=agent_role,
                            solver_deadline=solver_deadline,
                            build_errors=result.get("build_output", ""),
                            max_iterations=5,
                        ),
                        timeout=retry_timeout,
                    )
                except TimeoutError:
                    logger.warning(
                        "solver_retry_loop_timeout",
                        session_id=session_id,
                        solver_index=solver_index,
                        retry=retries,
                        timeout=retry_timeout,
                        budget_seconds=solver_budget_seconds,
                    )
                    result = timeout_result(
                        f"Solver retry timed out after {retry_timeout}s"
                    )
                    solver_timed_out = True
                    break

            build_success = result.get("build_success", False)
            build_output = result.get("build_output", "")

            # Collect files only if we still have budget.
            find_timeout = self._timeout_before_deadline(
                solver_deadline,
                30,
                reserve_seconds=15,
                min_timeout=5,
            )
            if find_timeout <= 0:
                files = {}
            elif solver_timed_out:
                # Attempt to collect partial files even on timeout
                try:
                    files = await asyncio.wait_for(
                        self._collect_sandbox_files(
                            sandbox_id, find_timeout=min(find_timeout, 10)
                        ),
                        timeout=min(find_timeout, 10),
                    )
                except (TimeoutError, Exception):
                    files = {}
            else:
                files = await self._collect_sandbox_files(
                    sandbox_id, find_timeout=find_timeout
                )

            # Run lint only if time remains; penalize skipped/timed-out lint.
            lint_timeout = self._timeout_before_deadline(
                solver_deadline,
                60,
                reserve_seconds=0,
                min_timeout=5,
            )
            if lint_timeout <= 0:
                lint_errors = 999
            else:
                lint_result = await self.sandbox_manager.execute_command(
                    sandbox_id, "npx eslint src --format json || true", timeout=lint_timeout
                )
                lint_errors = self._count_lint_errors(lint_result.stdout)
                if lint_result.timed_out:
                    lint_errors = max(lint_errors, 999)

            # Emit AGENT_COMPLETE
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.AGENT_COMPLETE,
                    session_id=session_id,
                    agent_id=agent_id,
                    agent_role=agent_role,
                    data={
                        "status": "timeout" if solver_timed_out else "complete",
                        "iterations": result.get("iterations", 0),
                        "files_written": list(files.keys()),
                        "build_success": build_success,
                        "retries": retries,
                    },
                )
            )

            logger.info(
                "solve_complete",
                session_id=session_id,
                solver_index=solver_index,
                build_success=build_success,
                lint_errors=lint_errors,
                files_produced=len(files),
                retries=retries,
            )

            return {
                "hypothesis_results": [
                    HypothesisResult(
                        agent_id=agent_id,
                        persona=persona,
                        sandbox_id=sandbox_id,
                        files=files,
                        edited_files=result.get("edited_files", []),
                        build_success=build_success,
                        build_output=build_output[:2000],
                        lint_errors=lint_errors,
                        iterations_used=result.get("iterations", 0),
                        agent_summary=result.get("summary", ""),
                    )
                ]
            }

        except Exception as e:
            logger.error(
                "solve_error",
                session_id=session_id,
                solver_index=solver_index,
                error=str(e),
                exc_info=True,
            )

            # Best-effort AGENT_ERROR event — must not prevent returning a result
            try:
                await self.event_bus.publish(
                    AgentEvent(
                        type=EventType.AGENT_ERROR,
                        session_id=session_id,
                        agent_id=agent_id,
                        agent_role=agent_role,
                        data={
                            "error": str(e),
                        },
                    )
                )
            except Exception:
                logger.warning(
                    "solve_error_event_publish_failed",
                    session_id=session_id,
                    agent_id=agent_id,
                )

            return {
                "hypothesis_results": [
                    HypothesisResult(
                        agent_id=agent_id,
                        persona=persona,
                        sandbox_id=sandbox_id,
                        files={},
                        edited_files=[],
                        build_success=False,
                        build_output=str(e),
                        lint_errors=999,
                        iterations_used=0,
                        agent_summary=f"Failed with error: {e}",
                    )
                ]
            }

    async def _run_solver_react_loop(
        self,
        task: str,
        persona: str,
        temperature: float,
        sandbox_id: str,
        session_id: str,
        agent_id: str,
        agent_role: str,
        solver_deadline: float,
        build_errors: str = "",
        max_iterations: int | None = None,
    ) -> dict[str, Any]:
        """Run a full ReAct loop for a solver.

        Args:
            task: The coding task
            persona: The solver's persona
            temperature: LLM temperature for this solver
            sandbox_id: Sandbox to work in
            session_id: Session ID for events
            agent_id: Agent ID for events
            agent_role: Agent role name
            solver_deadline: Deadline timestamp for this solver
            build_errors: Build error output from a previous attempt (for retries)
            max_iterations: Maximum iterations (defaults to settings.max_react_iterations)

        Returns:
            Dict with status, iterations, build_success, build_output, summary
        """
        if max_iterations is None:
            max_iterations = settings.max_react_iterations

        # Get persona-specific prompt
        system_prompt = get_solver_prompt(persona)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        # If retrying after build failure, prepend error context
        if build_errors:
            error_summary = self._summarize_build_errors(build_errors)
            messages.append({
                "role": "user",
                "content": (
                    "IMPORTANT: Your previous attempt failed to build. "
                    "Fix these build errors before continuing:\n"
                    f"{error_summary}\n\n"
                    f"Build output:\n{build_errors[:1500]}"
                ),
            })

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

            # Call LLM with persona temperature
            response = await self.llm_client.call(
                messages=pruned_messages,
                tools=get_tool_definitions_for_llm(),
                session_id=session_id,
                agent_id=agent_id,
                temperature=temperature,
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
                                "The previous tool call failed. Analyze the error, "
                                "identify the root cause, and adjust your approach. "
                                "Do NOT repeat the same action."
                            ),
                        })
            else:
                # No tool calls and not complete, ask to continue
                messages.append({
                    "role": "user",
                    "content": (
                        "Please continue with your implementation. "
                        "Use tools to write files."
                    ),
                })

            # Add review prompt and periodic build check every 5 iterations
            if iteration % 5 == 0:
                messages.append({
                    "role": "user",
                    "content": REVIEW_PROMPT,
                })

                # Run periodic build check to catch issues mid-loop
                build_check_timeout = self._timeout_before_deadline(
                    solver_deadline, 60, reserve_seconds=5, min_timeout=10,
                )
                if build_check_timeout <= 0:
                    break  # No time left for build checks

                build_check = await self.sandbox_manager.execute_command(
                    sandbox_id, "npm run build", timeout=build_check_timeout
                )
                if build_check.exit_code != 0 and not build_check.timed_out:
                    mid_loop_errors = build_check.stdout + "\n" + build_check.stderr
                    error_summary = self._summarize_build_errors(mid_loop_errors)
                    messages.append({
                        "role": "user",
                        "content": (
                            f"BUILD CHECK FAILED at iteration {iteration}. Fix these errors:\n"
                            f"{error_summary}\n\nBuild output:\n{mid_loop_errors[:1500]}"
                        ),
                    })

                # Run periodic lint check
                lint_check_timeout = self._timeout_before_deadline(
                    solver_deadline, 30, reserve_seconds=5, min_timeout=10,
                )
                if lint_check_timeout > 0:
                    lint_check = await self.sandbox_manager.execute_command(
                        sandbox_id, "npm run lint || true", timeout=lint_check_timeout
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
        final_build_timeout = self._timeout_before_deadline(
            solver_deadline, 60, reserve_seconds=5, min_timeout=10,
        )
        if final_build_timeout > 0:
            build_result = await self.sandbox_manager.execute_command(
                sandbox_id, "npm run build", timeout=final_build_timeout
            )
            build_success = build_result.exit_code == 0 and not build_result.timed_out
            build_output = build_result.stdout + "\n" + build_result.stderr
        else:
            build_success = False
            build_output = "Skipped: no time remaining"

        status = "complete" if build_success else "failed"

        return {
            "status": status,
            "iterations": iteration,
            "build_success": build_success,
            "build_output": build_output[:2000],
            "summary": summary[:500] if summary else "Task completed",
            "edited_files": sorted(edited_files),
        }

    async def _collect_sandbox_files(
        self, sandbox_id: str, find_timeout: int = 30
    ) -> dict[str, str]:
        """Collect all relevant files from a sandbox.

        Args:
            sandbox_id: The sandbox to collect from

        Returns:
            Dict mapping path -> content
        """
        files = {}

        # List files in src directory only (excludes template noise like
        # package.json, tsconfig, etc.)
        try:
            result = await self.sandbox_manager.execute_command(
                sandbox_id,
                r"find src -type f \( -name '*.tsx' -o -name '*.ts'"
                r" -o -name '*.css' \) 2>/dev/null",
                timeout=find_timeout,
            )

            # Fallback: if src/ doesn't exist, search top-level with depth limit
            if result.exit_code != 0 or not result.stdout.strip():
                result = await self.sandbox_manager.execute_command(
                    sandbox_id,
                    r"find . -maxdepth 3 -type f"
                    r" \( -name '*.tsx' -o -name '*.ts' -o -name '*.css' \)"
                    r" ! -path './node_modules/*' ! -path './dist/*'",
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

    def _count_lint_errors(self, lint_output: str) -> int:
        """Count lint errors from ESLint JSON output.

        Args:
            lint_output: ESLint output (ideally JSON format)

        Returns:
            Count of lint errors
        """
        try:
            # Try to parse JSON output
            data = json.loads(lint_output)
            total_errors = 0
            for file_result in data:
                total_errors += file_result.get("errorCount", 0)
            return total_errors
        except (json.JSONDecodeError, TypeError):
            # Fallback: count "error" occurrences
            return lint_output.lower().count("error")

    @staticmethod
    def _normalize_repo_path(raw_path: Any) -> str:
        """Normalize a repository-relative path for internal comparisons."""
        if not isinstance(raw_path, str):
            return ""
        return raw_path.strip().lstrip("./")

    @staticmethod
    def _result_has_entrypoint_edit(result: HypothesisResult) -> bool:
        """Whether a solver explicitly edited src/App.tsx."""
        for raw_path in result.get("edited_files", []):
            normalized = HypothesisGraph._normalize_repo_path(raw_path)
            if normalized == APP_ENTRYPOINT_PATH:
                return True
        return False

    def _prefer_entrypoint_wired_winner(
        self,
        *,
        selected_agent_id: str,
        selected_index: int,
        scores: list[EvaluationScore],
        results: list[HypothesisResult],
    ) -> tuple[str, int, str]:
        """Prefer a winner that explicitly wires the app entrypoint when close in score."""
        if not results:
            return selected_agent_id, selected_index, ""

        score_by_agent = {
            str(score.get("agent_id", "")): float(score.get("total", 0.0) or 0.0)
            for score in scores
        }

        selected_result = next(
            (result for result in results if result.get("agent_id") == selected_agent_id),
            None,
        )
        if selected_result and self._result_has_entrypoint_edit(selected_result):
            return selected_agent_id, selected_index, ""

        candidate_results = [
            result
            for result in results
            if result.get("build_success") and self._result_has_entrypoint_edit(result)
        ]
        if not candidate_results:
            return selected_agent_id, selected_index, ""

        best_candidate = max(
            candidate_results,
            key=lambda result: score_by_agent.get(str(result.get("agent_id", "")), 0.0),
        )
        candidate_id = str(best_candidate.get("agent_id", ""))
        if not candidate_id:
            return selected_agent_id, selected_index, ""

        selected_score = score_by_agent.get(selected_agent_id, 0.0)
        candidate_score = score_by_agent.get(candidate_id, 0.0)

        # Keep evaluator's explicit winner when the entrypoint-aware candidate is
        # materially worse; otherwise prefer the safer integrative result.
        if selected_agent_id and (candidate_score + 0.01) < (selected_score - 0.5):
            return selected_agent_id, selected_index, ""

        candidate_index = next(
            (i for i, result in enumerate(results) if result.get("agent_id") == candidate_id),
            selected_index,
        )
        if candidate_id == selected_agent_id:
            return selected_agent_id, selected_index, ""

        reason = (
            f"Adjusted winner to {candidate_id} because it explicitly edits {APP_ENTRYPOINT_PATH} "
            f"and remains within the tie-break score window."
        )
        return candidate_id, candidate_index, reason

    @staticmethod
    def _build_fallback_scores(
        results: list[HypothesisResult],
    ) -> list[EvaluationScore]:
        """Create deterministic evaluator scores when evaluator JSON parsing fails."""
        scores: list[EvaluationScore] = []

        for result in results:
            files = result.get("files", {})
            file_count = len(files) if isinstance(files, dict) else 0
            lint_errors = int(result.get("lint_errors", 0) or 0)
            build_success = bool(result.get("build_success", False))
            entrypoint_edited = HypothesisGraph._result_has_entrypoint_edit(result)

            build = 10 if build_success else 0
            lint = max(0, 10 - min(lint_errors, 10))
            quality = min(10, 4 + min(file_count, 6))
            completeness = min(10, 3 + min(file_count, 7))
            ux = 7 if build_success else 2

            if build_success and not entrypoint_edited:
                completeness = max(0, completeness - 2)
                ux = max(0, ux - 2)
            elif entrypoint_edited:
                completeness = min(10, completeness + 1)
                ux = min(10, ux + 1)

            total = (
                (build * 0.30) + (lint * 0.15) + (quality * 0.25)
                + (completeness * 0.20) + (ux * 0.10)
            )

            scores.append(
                EvaluationScore(
                    agent_id=str(result.get("agent_id", "")),
                    build=build,
                    lint=lint,
                    quality=quality,
                    completeness=completeness,
                    ux=ux,
                    total=round(total, 2),
                    notes=(
                        "Deterministic fallback score based on build success, lint errors, "
                        "and output footprint."
                    ),
                )
            )

        return scores

    @staticmethod
    def _select_best_agent_from_scores(
        scores: list[EvaluationScore],
        results: list[HypothesisResult],
    ) -> tuple[str, int]:
        """Select the best agent_id/index from score output constrained to known results."""
        result_agent_ids = [str(r.get("agent_id", "")) for r in results]
        best_agent_id = ""
        best_total = -1.0

        for score in scores:
            agent_id = str(score.get("agent_id", ""))
            if agent_id not in result_agent_ids:
                continue
            total = float(score.get("total", 0.0) or 0.0)
            if total > best_total:
                best_total = total
                best_agent_id = agent_id

        if not best_agent_id:
            # Conservative fallback: first available result.
            best_agent_id = result_agent_ids[0] if result_agent_ids else ""

        selected_index = 0
        for i, result in enumerate(results):
            if result.get("agent_id") == best_agent_id:
                selected_index = i
                break

        return best_agent_id, selected_index

    # -------------------------------------------------------------------------
    # Node: Evaluate
    # -------------------------------------------------------------------------

    async def _evaluate(self, state: HypothesisState) -> dict[str, Any]:
        """Evaluate node: score all solutions and select the winner.

        This node:
        1. Collects all hypothesis results
        2. Builds a prompt with all solutions for the evaluator
        3. Parses scores and selection from evaluator response
        4. Emits EVALUATION_RESULT event

        Args:
            state: Current graph state with hypothesis_results

        Returns:
            Dict with scores, selected_index, selected_agent_id, evaluation_reasoning
        """
        session_id = state["session_id"]
        evaluator_id = f"evaluator_{session_id[5:]}"

        await self._emit_node_active(
            session_id, evaluator_id, "Evaluator", "evaluate"
        )

        hypothesis_results = state.get("hypothesis_results", [])

        # Guard: no results at all (all solvers crashed)
        if not hypothesis_results:
            logger.error(
                "evaluate_no_results",
                session_id=session_id,
            )
            await self._emit_node_complete(
                session_id, evaluator_id, "Evaluator", "evaluate"
            )
            return {
                "scores": [],
                "selected_index": 0,
                "selected_agent_id": "",
                "evaluation_reasoning": "No solver produced results.",
                "status": "failed",
            }

        # Emit EVALUATION_STARTED
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.EVALUATION_STARTED,
                session_id=session_id,
                agent_id=evaluator_id,
                agent_role="Evaluator",
                data={
                    "solver_count": len(hypothesis_results),
                },
            )
        )

        logger.info(
            "evaluate_start",
            session_id=session_id,
            hypothesis_count=len(hypothesis_results),
        )

        # Check if all solvers failed
        successful_results = [r for r in hypothesis_results if r.get("build_success")]
        if not successful_results and hypothesis_results:
            # All failed - terminate with a clear failure signal.
            best_result = max(
                hypothesis_results, key=lambda r: len(r.get("files", {}))
            )
            logger.warning(
                "evaluate_all_failed",
                session_id=session_id,
                candidate_agent=best_result.get("agent_id"),
            )

            await self._emit_node_complete(
                session_id, evaluator_id, "Evaluator", "evaluate"
            )

            return {
                "scores": [],
                "selected_index": hypothesis_results.index(best_result),
                "selected_agent_id": best_result.get("agent_id", ""),
                "evaluation_reasoning": (
                    "All solvers failed build/lint gates. "
                    "Session terminated without selecting a winner."
                ),
                "status": "failed",
            }

        # Build evaluation prompt
        solutions_text = self._build_solutions_text(hypothesis_results)

        messages = [
            {"role": "system", "content": EVALUATOR_PROMPT},
            {
                "role": "user",
                "content": f"""Original task: {state['task']}

Here are {len(hypothesis_results)} solutions to evaluate:

{solutions_text}

Please evaluate each solution according to the rubric and select the best one.""",
            },
        ]

        # Call evaluator LLM
        response = await self.llm_client.call(
            messages=messages,
            model=self.evaluator_model,
            temperature=0.3,  # Lower temperature for evaluation
            session_id=session_id,
            agent_id=evaluator_id,
        )

        # Extract and emit chain-of-thought analysis if present
        analysis = parse_analysis_tag(response.content)
        if analysis:
            await self._emit_thinking(
                session_id, evaluator_id, "Evaluator", f"Analysis:\n{analysis}"
            )

        # Emit full response as thinking
        await self._emit_thinking(
            session_id, evaluator_id, "Evaluator", response.content
        )

        # Parse evaluation JSON
        parsed = extract_json_from_response(response.content)
        retry_temperatures = [0.2, 0.1]
        retry_prompts = [
            (
                "Your response was not valid JSON. Respond with ONLY valid JSON using this shape: "
                '{"scores":[{"agent_id":"","build":0,"lint":0,"quality":0,"completeness":0,"ux":0,"total":0.0,"notes":""}],'
                '"selected":"solver_1","reasoning":"..."}'
            ),
            (
                "Repair your previous output into strict valid JSON only. "
                "No markdown, no prose, no trailing commentary."
            ),
        ]

        for retry_index, (retry_temperature, retry_prompt) in enumerate(
            zip(retry_temperatures, retry_prompts, strict=True), start=1
        ):
            if parsed:
                break

            logger.warning(
                "evaluate_json_parse_failed",
                session_id=session_id,
                attempt=retry_index,
            )

            retry_messages = messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": retry_prompt},
            ]

            response = await self.llm_client.call(
                messages=retry_messages,
                model=self.evaluator_model,
                temperature=retry_temperature,
                session_id=session_id,
                agent_id=evaluator_id,
            )

            parsed = extract_json_from_response(response.content)

        # Extract evaluation results
        scores = []
        selected_agent_id = ""
        selected_index = 0
        reasoning = ""
        used_deterministic_fallback = False

        if parsed:
            raw_scores = parsed.get("scores", [])
            selected_agent_id = parsed.get("selected", "")
            reasoning = parsed.get("reasoning", "")

            for score_data in raw_scores:
                scores.append(
                    EvaluationScore(
                        agent_id=score_data.get("agent_id", ""),
                        build=score_data.get("build", 0),
                        lint=score_data.get("lint", 0),
                        quality=score_data.get("quality", 0),
                        completeness=score_data.get("completeness", 0),
                        ux=score_data.get("ux", 0),
                        total=score_data.get("total", 0.0),
                        notes=score_data.get("notes", ""),
                    )
                )

            # Find selected index
            for i, result in enumerate(hypothesis_results):
                if result.get("agent_id") == selected_agent_id:
                    selected_index = i
                    break

            # Guard against evaluator selecting an unknown agent id.
            if not any(
                result.get("agent_id") == selected_agent_id
                for result in hypothesis_results
            ):
                selected_agent_id, selected_index = self._select_best_agent_from_scores(
                    scores, hypothesis_results
                )
                reasoning = (
                    "Evaluator selected an unknown agent_id; "
                    f"fell back to highest valid score ({selected_agent_id})."
                )
        else:
            # Fallback: deterministic score synthesis from observable signals.
            scores = self._build_fallback_scores(hypothesis_results)
            selected_agent_id, selected_index = self._select_best_agent_from_scores(
                scores, hypothesis_results
            )
            used_deterministic_fallback = True
            reasoning = (
                "Evaluation parse failed. Selected winner via deterministic fallback "
                "(build success, lint cleanliness, and produced output)."
            )

        # Multi-round evaluation: if top 2 scores are within threshold,
        # run pairwise comparison for a definitive winner.
        # Skip if < 60s remaining in session budget to avoid timeout.
        threshold = settings.multi_round_evaluation_threshold
        remaining_budget = self._remaining_session_budget_seconds()

        if (
            len(scores) >= 2
            and threshold > 0
            and remaining_budget >= 60
            and not used_deterministic_fallback
        ):
            sorted_scores = sorted(scores, key=lambda s: s.get("total", 0), reverse=True)
            top1_total = sorted_scores[0].get("total", 0)
            top2_total = sorted_scores[1].get("total", 0)

            if abs(top1_total - top2_total) <= threshold:
                logger.info(
                    "evaluate_pairwise_triggered",
                    session_id=session_id,
                    top1=sorted_scores[0].get("agent_id"),
                    top2=sorted_scores[1].get("agent_id"),
                    score_diff=abs(top1_total - top2_total),
                    remaining_budget=round(remaining_budget, 1),
                )

                pairwise_result = await self._pairwise_compare(
                    sorted_scores[0], sorted_scores[1],
                    hypothesis_results, session_id, evaluator_id,
                )
                if pairwise_result:
                    selected_agent_id = pairwise_result
                    for i, result in enumerate(hypothesis_results):
                        if result.get("agent_id") == selected_agent_id:
                            selected_index = i
                            break
                    reasoning += f" (Pairwise comparison selected {selected_agent_id})"
        elif len(scores) >= 2 and threshold > 0 and remaining_budget < 60:
            logger.info(
                "evaluate_pairwise_skipped_time_budget",
                session_id=session_id,
                remaining_budget=round(remaining_budget, 1),
            )

        selected_agent_id, selected_index, entrypoint_reason = (
            self._prefer_entrypoint_wired_winner(
                selected_agent_id=selected_agent_id,
                selected_index=selected_index,
                scores=scores,
                results=hypothesis_results,
            )
        )
        if entrypoint_reason:
            reasoning = f"{reasoning} {entrypoint_reason}".strip()

        # Emit EVALUATION_RESULT
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.EVALUATION_RESULT,
                session_id=session_id,
                agent_id=evaluator_id,
                agent_role="Evaluator",
                data={
                    "selected": selected_agent_id,
                    "scores": [dict(s) for s in scores],
                    "reasoning": reasoning,
                },
            )
        )

        logger.info(
            "evaluate_complete",
            session_id=session_id,
            selected_agent=selected_agent_id,
            selected_index=selected_index,
        )

        await self._emit_node_complete(
            session_id, evaluator_id, "Evaluator", "evaluate"
        )

        return {
            "scores": scores,
            "selected_index": selected_index,
            "selected_agent_id": selected_agent_id,
            "evaluation_reasoning": reasoning,
            "status": "finalizing",
        }

    def _build_solutions_text(self, results: list[HypothesisResult]) -> str:
        """Build a text summary of all solutions for the evaluator.

        Args:
            results: List of HypothesisResult objects

        Returns:
            Formatted text describing all solutions
        """
        parts = []

        def _file_priority(item: tuple[str, str]) -> int:
            """Sort key: App.tsx first, then components, then .tsx, .ts, rest."""
            path = item[0].lower()
            if "app.tsx" in path:
                return 0
            if "/components/" in path:
                return 1
            if path.endswith(".tsx"):
                return 2
            if path.endswith(".ts"):
                return 3
            return 4

        for result in results:
            agent_id = result.get("agent_id", "unknown")
            persona = result.get("persona", "unknown")
            build_success = result.get("build_success", False)
            build_output = self._clip_with_ellipsis(
                str(result.get("build_output", "")),
                EVALUATION_MAX_BUILD_OUTPUT_CHARS,
            )
            lint_errors = result.get("lint_errors", 0)
            files_raw = result.get("files", {})
            files = files_raw if isinstance(files_raw, dict) else {}
            summary = self._clip_with_ellipsis(
                str(result.get("agent_summary", "")),
                EVALUATION_MAX_SUMMARY_CHARS,
            )

            file_paths = [path for path in files if isinstance(path, str)]
            listed_paths = file_paths[:EVALUATION_MAX_LISTED_FILES]
            file_listing_lines = [f"  - {path}" for path in listed_paths]
            omitted_listing_count = max(0, len(file_paths) - len(listed_paths))
            if omitted_listing_count:
                file_listing_lines.append(
                    f"  - ... ({omitted_listing_count} more files omitted)"
                )
            file_listing = "\n".join(file_listing_lines) if file_listing_lines else "  - (none)"

            edited_paths = [
                self._normalize_repo_path(path)
                for path in result.get("edited_files", [])
                if isinstance(path, str)
            ]
            cleaned_edited_paths = [path for path in edited_paths if path]
            edited_listing = (
                ", ".join(cleaned_edited_paths)
                if cleaned_edited_paths
                else "(none tracked)"
            )

            sorted_files = sorted(
                [
                    (path, content)
                    for path, content in files.items()
                    if isinstance(path, str) and isinstance(content, str)
                ],
                key=_file_priority,
            )

            # Include key file contents (truncated)
            file_contents = ""
            top_files = sorted_files[:EVALUATION_MAX_FILES_WITH_CONTENT]
            for path, content in top_files:
                clipped_content = self._clip_with_ellipsis(
                    content, EVALUATION_MAX_FILE_CONTENT_CHARS
                )
                file_contents += f"\n--- {path} ---\n{clipped_content}\n"

            omitted_content_count = max(0, len(sorted_files) - len(top_files))
            if omitted_content_count:
                file_contents += (
                    f"\n[... {omitted_content_count} additional files omitted from content]\n"
                )
            if not file_contents:
                file_contents = "(none)"

            solution_text = f"""
=== {agent_id} ({persona}) ===
Build Success: {"Yes" if build_success else "No"}
Lint Errors: {lint_errors}
Agent Summary: {summary}
Edited files: {edited_listing}

Files produced:
{file_listing}

Build output:
{build_output}

Key file contents:
{file_contents}
"""
            parts.append(solution_text)

        return "\n\n".join(parts)

    async def _pairwise_compare(
        self,
        score_a: EvaluationScore,
        score_b: EvaluationScore,
        hypothesis_results: list[HypothesisResult],
        session_id: str,
        evaluator_id: str,
    ) -> str | None:
        """Run pairwise comparison between two close-scoring solutions.

        When the top 2 solutions are within the threshold, this provides
        a focused head-to-head comparison for a more definitive winner.

        Args:
            score_a: First (highest scoring) solution
            score_b: Second solution
            hypothesis_results: Full results for file access
            session_id: Session ID for LLM call
            evaluator_id: Evaluator agent ID

        Returns:
            The agent_id of the winner, or None if comparison fails
        """
        agent_a = score_a.get("agent_id", "")
        agent_b = score_b.get("agent_id", "")

        # Find the full results for these two
        result_a = next((r for r in hypothesis_results if r.get("agent_id") == agent_a), None)
        result_b = next((r for r in hypothesis_results if r.get("agent_id") == agent_b), None)

        if not result_a or not result_b:
            return None

        # Build focused comparison
        solutions_text = self._build_solutions_text([result_a, result_b])

        messages = [
            {
                "role": "system",
                "content": (
                    "You are comparing two very close solutions. Pick the better one. "
                    "Respond with ONLY the agent_id of the winner (e.g., 'solver_1'). "
                    "No explanation needed."
                ),
            },
            {
                "role": "user",
                "content": (
                    "These two solutions scored within 0.5 points."
                    f" Which is better?\n\n{solutions_text}"
                ),
            },
        ]

        try:
            pairwise_timeout = self._timeout_before_deadline(
                time.monotonic() + self._remaining_session_budget_seconds(),
                60,
                reserve_seconds=10,
                min_timeout=15,
            )
            if pairwise_timeout <= 0:
                return None

            response = await asyncio.wait_for(
                self.llm_client.call(
                    messages=messages,
                    model=self.evaluator_model,
                    temperature=0.2,
                    session_id=session_id,
                    agent_id=evaluator_id,
                ),
                timeout=pairwise_timeout,
            )

            winner = response.content.strip()
            if winner in (agent_a, agent_b):
                logger.info(
                    "pairwise_compare_result",
                    session_id=session_id,
                    winner=winner,
                )
                return winner

        except Exception as e:
            logger.warning(
                "pairwise_compare_failed",
                session_id=session_id,
                error=str(e),
            )

        return None

    # -------------------------------------------------------------------------
    # Conditional Edge: Should Synthesize
    # -------------------------------------------------------------------------

    def _should_synthesize(self, state: HypothesisState) -> str:
        """Determine whether to run synthesis or skip to finalize.

        Synthesis is enabled via config and only runs when the evaluator
        provided improvement suggestions.

        Args:
            state: Current graph state

        Returns:
            "synthesize" to run synthesis, "finalize" to skip
        """
        if not settings.enable_hypothesis_synthesis:
            return "finalize"

        reasoning = state.get("evaluation_reasoning", "")
        if self._reasoning_requests_synthesis(reasoning):
            return "synthesize"
        return "finalize"

    def _route_after_evaluate(self, state: HypothesisState) -> str:
        """Route after evaluate, terminating immediately on evaluator failure."""
        if state.get("status") == "failed":
            return "end"
        return self._should_synthesize(state)

    # -------------------------------------------------------------------------
    # Node: Synthesize
    # -------------------------------------------------------------------------

    async def _synthesize(self, state: HypothesisState) -> dict[str, Any]:
        """Synthesize node: apply evaluator improvements to winning solution.

        This optional node takes the winning solution and applies targeted
        improvements suggested by the evaluator, potentially incorporating
        best patterns from losing solutions.

        Args:
            state: Current graph state with selected winner

        Returns:
            Dict with updated hypothesis_results
        """
        session_id = state["session_id"]
        synthesis_id = f"synthesis_{session_id[5:]}"

        await self._emit_node_active(
            session_id, synthesis_id, "Synthesizer", "synthesize"
        )

        hypothesis_results = state.get("hypothesis_results", [])
        selected_index = state.get("selected_index", 0)

        if not hypothesis_results or selected_index >= len(hypothesis_results):
            await self._emit_node_complete(
                session_id, synthesis_id, "Synthesizer", "synthesize"
            )
            return {}

        winning_result = hypothesis_results[selected_index]
        sandbox_id = winning_result.get("sandbox_id", "")
        improvements = state.get("evaluation_reasoning", "")

        logger.info(
            "synthesize_start",
            session_id=session_id,
            sandbox_id=sandbox_id,
        )

        # Build synthesis prompt
        system_prompt = SYNTHESIS_PROMPT.format(
            improvements=improvements,
            task=state["task"],
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Apply the suggested improvements to the winning solution.",
            },
        ]

        tool_executor = ToolExecutor(
            sandbox_manager=self.sandbox_manager,
            event_bus=self.event_bus,
            metrics_collector=self.llm_client.metrics_collector,
            auto_preview=False,
        )

        # Mini react loop (5 iterations max)
        for _iteration in range(5):
            response = await self.llm_client.call(
                messages=messages,
                tools=get_tool_definitions_for_llm(),
                session_id=session_id,
                agent_id=synthesis_id,
                temperature=0.5,
            )

            plan = parse_plan_tag(response.content) or response.content
            await self._emit_thinking(session_id, synthesis_id, "Synthesizer", plan)

            status_tag = parse_status_tag(response.content)
            is_complete_signal = status_tag == "TASK_COMPLETE"
            if status_tag is None:
                is_complete_signal = "task_complete" in response.content.lower()

            if is_complete_signal and not response.tool_calls:
                break

            assistant_message = format_assistant_message_with_tools(
                response.content, response.tool_calls
            )
            messages.append(assistant_message)

            if response.tool_calls:
                for tc_data in response.tool_calls:
                    tc = ToolCall(id=tc_data.id, name=tc_data.name, args=tc_data.args)
                    result = await tool_executor.execute_tool_call(
                        sandbox_id=sandbox_id,
                        tool_call=tc,
                        session_id=session_id,
                        agent_id=synthesis_id,
                        agent_role="Synthesizer",
                    )
                    messages.append(format_tool_result_for_llm(tc.id, result.content))
            else:
                break

        logger.info("synthesize_complete", session_id=session_id)

        await self._emit_node_complete(
            session_id, synthesis_id, "Synthesizer", "synthesize"
        )

        return {}

    # -------------------------------------------------------------------------
    # Node: Finalize
    # -------------------------------------------------------------------------

    async def _finalize(self, state: HypothesisState) -> dict[str, Any]:
        """Finalize node: prepare the winning solution for preview.

        This node:
        1. Identifies the winning solver's sandbox
        2. Starts the preview server
        3. Cleans up losing sandboxes (optional)

        Args:
            state: Current graph state with selected_index

        Returns:
            Dict with final_sandbox_id, final_preview_url, status
        """
        session_id = state["session_id"]
        finalize_id = f"finalize_{session_id[5:]}"

        await self._emit_node_active(
            session_id, finalize_id, "Finalizer", "finalize"
        )

        hypothesis_results = state.get("hypothesis_results", [])
        selected_index = state.get("selected_index", 0)

        if not hypothesis_results:
            logger.error(
                "finalize_no_results",
                session_id=session_id,
            )
            await self._emit_node_complete(
                session_id, finalize_id, "Finalizer", "finalize"
            )
            return {
                "status": "failed",
                "error_message": "No hypothesis results to finalize",
            }

        # Get winning result
        if selected_index < 0 or selected_index >= len(hypothesis_results):
            selected_index = 0

        winning_result = hypothesis_results[selected_index]
        final_sandbox_id = winning_result.get("sandbox_id", "")

        logger.info(
            "finalize_start",
            session_id=session_id,
            selected_agent=winning_result.get("agent_id"),
            sandbox_id=final_sandbox_id,
        )

        # Cleanup losing sandboxes before starting preview — this runs
        # regardless of whether the preview server succeeds.
        for result in hypothesis_results:
            if result.get("sandbox_id") != final_sandbox_id:
                try:
                    await self.sandbox_manager.destroy_sandbox(
                        result.get("sandbox_id", "")
                    )
                except Exception as e:
                    logger.warning(
                        "finalize_cleanup_sandbox_failed",
                        sandbox_id=result.get("sandbox_id"),
                        error=str(e),
                    )

        # Start preview server on winning sandbox
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.PREVIEW_STARTING,
                session_id=session_id,
                agent_id=finalize_id,
                agent_role="Finalizer",
                data={"sandbox_id": final_sandbox_id},
            )
        )

        try:
            preview_url = await self.sandbox_manager.start_dev_server(final_sandbox_id)

            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.PREVIEW_READY,
                    session_id=session_id,
                    agent_id=finalize_id,
                    agent_role="Finalizer",
                    data={
                        "url": preview_url,
                        "sandbox_id": final_sandbox_id,
                    },
                )
            )

            logger.info(
                "finalize_preview_ready",
                session_id=session_id,
                preview_url=preview_url,
            )

            await self._emit_node_complete(
                session_id, finalize_id, "Finalizer", "finalize"
            )

            return {
                "final_sandbox_id": final_sandbox_id,
                "final_preview_url": preview_url,
                "status": "complete",
            }

        except Exception as e:
            logger.error(
                "finalize_preview_failed",
                session_id=session_id,
                error=str(e),
            )

            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.PREVIEW_ERROR,
                    session_id=session_id,
                    agent_id=finalize_id,
                    agent_role="Finalizer",
                    data={"error": str(e)},
                )
            )

            await self._emit_node_complete(
                session_id, finalize_id, "Finalizer", "finalize"
            )

            return {
                "final_sandbox_id": final_sandbox_id,
                "final_preview_url": None,
                "status": "complete",  # Still complete, just no preview
            }

    # -------------------------------------------------------------------------
    # Public Methods: run() and stream()
    # -------------------------------------------------------------------------

    async def run(self, initial_state: HypothesisState) -> HypothesisState:
        """Run the graph to completion.

        Args:
            initial_state: The starting state

        Returns:
            The final state after completion
        """
        # Emit graph initialized event
        num_solvers = initial_state.get("num_hypotheses", 3)
        solver_nodes = [f"solver_{i+1}" for i in range(num_solvers)]

        # Build graph structure for UI - include synthesize node if enabled
        nodes = ["broadcast"] + solver_nodes + ["evaluate"]
        edges = [
            {"source": "START", "target": "broadcast"},
            *[{"source": "broadcast", "target": s, "parallel": True} for s in solver_nodes],
            *[{"source": s, "target": "evaluate"} for s in solver_nodes],
        ]

        if settings.enable_hypothesis_synthesis:
            nodes.append("synthesize")
            edges.append({"source": "evaluate", "target": "END", "condition": "failed"})
            edges.append({"source": "evaluate", "target": "synthesize", "condition": "synthesize"})
            edges.append({"source": "evaluate", "target": "finalize", "condition": "finalize"})
            edges.append({"source": "synthesize", "target": "finalize"})
        else:
            edges.append({"source": "evaluate", "target": "END", "condition": "failed"})
            edges.append({"source": "evaluate", "target": "finalize"})

        nodes.append("finalize")
        edges.append({"source": "finalize", "target": "END"})

        self._session_start_time = time.monotonic()

        await self.event_bus.publish(
            AgentEvent(
                type=EventType.GRAPH_INITIALIZED,
                session_id=initial_state["session_id"],
                agent_id="hypothesis_graph",
                agent_role="Hypothesis Graph",
                data={
                    "nodes": nodes,
                    "edges": edges,
                },
            )
        )

        # Run the graph
        final_state = await self._compiled_graph.ainvoke(initial_state)

        return final_state


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------


def create_hypothesis_graph(
    sandbox_manager: SandboxManager,
    event_bus: EventBus,
    llm_client: LLMClient | None = None,
    evaluator_model: str | None = None,
) -> HypothesisGraph:
    """Factory function to create a Hypothesis graph.

    This is the main entry point for creating a parallel hypothesis agent.

    Args:
        sandbox_manager: Manager for sandbox operations
        event_bus: Event bus for emitting events
        llm_client: Optional LLM client (creates default if None)

    Returns:
        Configured HypothesisGraph instance
    """
    return HypothesisGraph(
        sandbox_manager=sandbox_manager,
        event_bus=event_bus,
        llm_client=llm_client,
        evaluator_model=evaluator_model,
    )
