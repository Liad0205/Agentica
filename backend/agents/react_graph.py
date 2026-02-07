"""Single ReAct Agent LangGraph implementation.

This module provides the ReAct (Reason, Act, Review) agent graph - the simplest
of the three agent paradigms. It implements a cyclic graph with:

    START -> reason -> execute -> review -> [continue -> reason | end -> END]

The agent follows a loop of:
1. REASON: Plan the next steps based on task and history
2. EXECUTE: Execute tool calls against the sandbox
3. REVIEW: Self-assess if the task is complete or needs more work

Events emitted:
- AGENT_THINKING: When the agent is reasoning/planning
- AGENT_TOOL_CALL: Before each tool execution
- AGENT_TOOL_RESULT: After each tool execution
- FILE_CHANGED: When files are written
- GRAPH_NODE_ACTIVE: When entering a graph node
- GRAPH_NODE_COMPLETE: When exiting a graph node
"""

import re
from typing import Annotated, Any, Literal, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agents.prompts import REFLECTION_PROMPT, REVIEW_PROMPT, get_react_system_prompt
from agents.tools import (
    ToolCall,
    ToolExecutor,
    get_tool_definitions_for_llm,
)
from agents.utils import (
    LLMClient,
    _convert_message_to_dict,
    format_assistant_message_with_tools,
    format_tool_result_for_llm,
    normalize_tool_args,
    parse_plan_tag,
    parse_status_tag,
    sliding_window_prune,
)
from events.bus import EventBus
from events.types import AgentEvent, EventType
from sandbox.docker_sandbox import SandboxManager

logger = structlog.get_logger()


class ReactState(TypedDict):
    """State for the ReAct agent graph.

    This state flows through the graph and accumulates information
    as the agent works through the task.

    Attributes:
        task: The user's original task/prompt
        messages: Full conversation history with add_messages reducer
        files_written: List of file paths created or modified
        sandbox_id: The sandbox container identifier
        iteration: Current iteration count
        max_iterations: Hard limit on iterations
        status: Current agent status
        current_plan: Latest reasoning/planning text
        session_id: Session identifier for events
        agent_id: Agent identifier for events
        build_verified: Whether a successful build command has run in-session
        lint_verified: Whether a successful lint command has run in-session
    """

    task: str
    messages: Annotated[list[dict[str, Any]], add_messages]
    files_written: list[str]
    sandbox_id: str
    iteration: int
    max_iterations: int
    status: Literal["reasoning", "executing", "reviewing", "complete", "failed"]
    current_plan: str
    session_id: str
    agent_id: str
    tool_error_counts: dict[str, int]
    build_verified: bool
    lint_verified: bool


def create_initial_state(
    task: str,
    sandbox_id: str,
    session_id: str,
    agent_id: str = "react_agent",
    max_iterations: int = 15,
) -> ReactState:
    """Create the initial state for a ReAct agent run.

    Args:
        task: The user's task/prompt
        sandbox_id: The sandbox to work in
        session_id: Session ID for event emission
        agent_id: Agent ID for event emission
        max_iterations: Maximum iterations before stopping

    Returns:
        Initial ReactState dict
    """
    return ReactState(
        task=task,
        messages=[
            {"role": "system", "content": get_react_system_prompt(task)},
            {"role": "user", "content": task},
        ],
        files_written=[],
        sandbox_id=sandbox_id,
        iteration=0,
        max_iterations=max_iterations,
        status="reasoning",
        current_plan="",
        session_id=session_id,
        agent_id=agent_id,
        tool_error_counts={},
        build_verified=False,
        lint_verified=False,
    )


class ReactGraph:
    """The ReAct agent graph implementation.

    This class encapsulates the LangGraph StateGraph and provides
    the node implementations that access external dependencies
    (sandbox manager, event bus, LLM client).

    Usage:
        >>> graph = ReactGraph(sandbox_manager, event_bus, llm_client)
        >>> initial_state = create_initial_state(task, sandbox_id, session_id)
        >>> async for event in graph.stream(initial_state):
        ...     process_event(event)
    """

    _BUILD_COMMAND_RE = re.compile(
        r"\b(?:npm|pnpm|yarn)\s+(?:run\s+)?build\b|\bvite\s+build\b",
        re.IGNORECASE,
    )
    _LINT_COMMAND_RE = re.compile(
        r"\b(?:npm|pnpm|yarn)\s+(?:run\s+)?lint\b|\beslint\b",
        re.IGNORECASE,
    )
    _OBSERVATIONAL_COMMAND_RE = re.compile(
        r"^\s*(?:ls|pwd|cat|head|tail|grep|rg|find|wc|which|readlink)\b",
        re.IGNORECASE,
    )
    _MUTATING_COMMAND_RE = re.compile(
        r"(?:\>\s*\S|\>\>\s*\S|\bsed\s+-i\b|\bnpm\s+install\b|\bpnpm\s+add\b|\byarn\s+add\b|\bmkdir\b|\btouch\b|\bmv\b|\bcp\b|\brm\b)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        sandbox_manager: SandboxManager,
        event_bus: EventBus,
        llm_client: LLMClient | None = None,
        temperature: float = 0.7,
    ) -> None:
        """Initialize the ReAct graph.

        Args:
            sandbox_manager: Manager for sandbox operations
            event_bus: Event bus for emitting events
            llm_client: LLM client for model calls (creates default if None)
            temperature: Default temperature for planning/review calls.
        """
        self.sandbox_manager = sandbox_manager
        self.event_bus = event_bus
        self.llm_client = llm_client or LLMClient(event_bus=event_bus)
        self.temperature = temperature
        self._compiled_graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph StateGraph.

        Returns:
            Compiled StateGraph ready for execution
        """
        graph = StateGraph(ReactState)

        # Add nodes
        graph.add_node("reason", self._reason_and_plan)
        graph.add_node("execute", self._execute_tools)
        graph.add_node("review", self._self_review)

        # Add edges
        graph.add_edge(START, "reason")
        graph.add_edge("reason", "execute")
        graph.add_edge("execute", "review")

        # Add conditional edge from review
        graph.add_conditional_edges(
            "review",
            self._should_continue,
            {
                "continue": "reason",
                "end": END,
            },
        )

        return graph.compile()

    async def _emit_node_active(self, state: ReactState, node_name: str) -> None:
        """Emit GRAPH_NODE_ACTIVE event."""
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.GRAPH_NODE_ACTIVE,
                session_id=state["session_id"],
                agent_id=state["agent_id"],
                agent_role="ReAct Agent",
                data={"node_id": node_name},
            )
        )

    async def _emit_node_complete(self, state: ReactState, node_name: str) -> None:
        """Emit GRAPH_NODE_COMPLETE event."""
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.GRAPH_NODE_COMPLETE,
                session_id=state["session_id"],
                agent_id=state["agent_id"],
                agent_role="ReAct Agent",
                data={"node_id": node_name},
            )
        )

    async def _emit_thinking(
        self, state: ReactState, content: str, streaming: bool = False
    ) -> None:
        """Emit AGENT_THINKING event."""
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.AGENT_THINKING,
                session_id=state["session_id"],
                agent_id=state["agent_id"],
                agent_role="ReAct Agent",
                data={
                    "content": content,
                    "streaming": streaming,
                },
            )
        )

    @classmethod
    def _is_build_command(cls, command: str) -> bool:
        return bool(command and cls._BUILD_COMMAND_RE.search(command))

    @classmethod
    def _is_lint_command(cls, command: str) -> bool:
        return bool(command and cls._LINT_COMMAND_RE.search(command))

    @classmethod
    def _is_observational_command(cls, command: str) -> bool:
        return bool(command and cls._OBSERVATIONAL_COMMAND_RE.search(command))

    @classmethod
    def _is_potentially_mutating_command(cls, command: str) -> bool:
        """Conservatively classify shell commands that may change workspace state."""
        if not command:
            return False
        if cls._is_build_command(command) or cls._is_lint_command(command):
            return False
        if cls._is_observational_command(command):
            return False
        if cls._MUTATING_COMMAND_RE.search(command):
            return True
        # Default to mutating for unknown command shapes to avoid stale verification.
        return True

    async def _reason_and_plan(self, state: ReactState) -> dict[str, Any]:
        """Reason about the current state and plan next actions.

        This node:
        1. Calls the LLM with the current conversation history
        2. Extracts the plan from <plan> tags
        3. Returns updated messages with the assistant's response

        Args:
            state: Current graph state

        Returns:
            Dict with updated messages, iteration, and current_plan
        """
        await self._emit_node_active(state, "reason")

        iteration = state["iteration"]

        logger.info(
            "reason_and_plan_start",
            session_id=state["session_id"],
            iteration=iteration,
        )

        # Prune context window to stay within token limits
        messages = sliding_window_prune(list(state["messages"]))

        # Inject reflection checkpoint every 5 iterations
        if iteration > 0 and iteration % 5 == 0:
            reflection = REFLECTION_PROMPT.format(iteration=iteration)
            messages.append({"role": "user", "content": reflection})
            await self._emit_thinking(state, f"[Reflection at iteration {iteration}]")

        # Call the LLM
        response = await self.llm_client.call(
            messages=messages,
            tools=get_tool_definitions_for_llm(),
            session_id=state["session_id"],
            agent_id=state["agent_id"],
            temperature=self.temperature,
        )

        # Extract plan from response
        plan = parse_plan_tag(response.content)
        if plan:
            await self._emit_thinking(state, plan)
        elif response.content:
            # If no plan tag, emit the full content as thinking
            await self._emit_thinking(state, response.content)

        # Format the assistant message for history
        assistant_message = format_assistant_message_with_tools(
            response.content,
            response.tool_calls,
        )

        logger.info(
            "reason_and_plan_complete",
            session_id=state["session_id"],
            iteration=state["iteration"] + 1,
            tool_calls=len(response.tool_calls),
            plan_length=len(plan) if plan else 0,
        )

        await self._emit_node_complete(state, "reason")

        return {
            "messages": [assistant_message],
            "iteration": state["iteration"] + 1,
            "current_plan": plan or response.content,
            "status": "executing",
        }

    async def _execute_tools(self, state: ReactState) -> dict[str, Any]:
        """Execute tool calls from the last assistant message.

        This node:
        1. Extracts tool calls from the last message
        2. Executes each tool call against the sandbox
        3. Returns tool result messages

        Args:
            state: Current graph state

        Returns:
            Dict with updated messages and files_written
        """
        await self._emit_node_active(state, "execute")

        # Get the last message to find tool calls (convert from LangChain if needed)
        last_message = _convert_message_to_dict(state["messages"][-1])
        tool_calls_raw = last_message.get("tool_calls", [])

        if not tool_calls_raw:
            logger.debug(
                "execute_tools_no_tools",
                session_id=state["session_id"],
            )
            await self._emit_node_complete(state, "execute")
            return {"status": "reviewing"}

        # Create tool executor
        tool_executor = ToolExecutor(
            sandbox_manager=self.sandbox_manager,
            event_bus=self.event_bus,
            metrics_collector=self.llm_client.metrics_collector,
        )

        # Execute each tool call, tracking errors for recovery guidance
        tool_messages: list[dict[str, Any]] = []
        new_files: list[str] = []
        error_counts = dict(state.get("tool_error_counts", {}))
        build_verified = bool(state.get("build_verified", False))
        lint_verified = bool(state.get("lint_verified", False))

        for tc_raw in tool_calls_raw:
            function_data = tc_raw.get("function", {}) if isinstance(tc_raw, dict) else {}
            raw_args = function_data.get("arguments", {})

            # Parse the tool call
            tool_call = ToolCall(
                id=tc_raw.get("id", ""),
                name=function_data.get("name", ""),
                args=normalize_tool_args(raw_args),
            )

            logger.debug(
                "executing_tool",
                tool_name=tool_call.name,
                session_id=state["session_id"],
            )

            # Execute the tool
            result = await tool_executor.execute_tool_call(
                sandbox_id=state["sandbox_id"],
                tool_call=tool_call,
                session_id=state["session_id"],
                agent_id=state["agent_id"],
                agent_role="ReAct Agent",
            )

            if tool_call.name == "execute_command":
                command = str(tool_call.args.get("command", ""))
                if self._is_potentially_mutating_command(command):
                    build_verified = False
                    lint_verified = False
                if self._is_build_command(command):
                    build_verified = result.success
                if self._is_lint_command(command):
                    lint_verified = result.success

            # Track tool errors for recovery guidance
            if not result.success:
                error_counts[tool_call.name] = error_counts.get(tool_call.name, 0) + 1
                consecutive_failures = error_counts[tool_call.name]

                if consecutive_failures >= 3:
                    # Inject guidance message after the tool result
                    guidance = (
                        f"Tool '{tool_call.name}' has failed {consecutive_failures} "
                        f"consecutive times. Consider trying a different approach or tool."
                    )
                    tool_messages.append(
                        format_tool_result_for_llm(tool_call.id, result.content)
                    )
                    tool_messages.append({
                        "role": "user",
                        "content": guidance,
                    })
                    logger.warning(
                        "tool_repeated_failure",
                        tool_name=tool_call.name,
                        consecutive_failures=consecutive_failures,
                        session_id=state["session_id"],
                    )
                    continue
            else:
                # Reset counter on success
                error_counts[tool_call.name] = 0

            # Track written files
            if tool_call.name == "write_file" and result.success:
                path = tool_call.args.get("path", "")
                if path and path not in state["files_written"]:
                    new_files.append(path)
                # Any successful source edit invalidates previous verification.
                build_verified = False
                lint_verified = False

            # Add tool result message
            tool_messages.append(
                format_tool_result_for_llm(tool_call.id, result.content)
            )
            if not result.success:
                tool_messages.append({
                    "role": "user",
                    "content": (
                        "The previous tool call failed. Analyze the error, "
                        "identify the root cause, and adjust your approach. "
                        "Do NOT repeat the same action."
                    ),
                })

        logger.info(
            "execute_tools_complete",
            session_id=state["session_id"],
            tools_executed=len(tool_messages),
            files_written=len(new_files),
        )

        await self._emit_node_complete(state, "execute")

        return {
            "messages": tool_messages,
            "files_written": state["files_written"] + new_files,
            "tool_error_counts": error_counts,
            "build_verified": build_verified,
            "lint_verified": lint_verified,
            "status": "reviewing",
        }

    async def _self_review(self, state: ReactState) -> dict[str, Any]:
        """Review current work and decide if complete or needs revision.

        This node is assessment-only - it does NOT execute tools. If fixes
        are needed, it returns NEEDS_REVISION and the reason node will
        handle the fixes in the next iteration.

        This node:
        1. Calls the LLM with the review prompt (no tools)
        2. Checks for structured <status> tags (TASK_COMPLETE / NEEDS_REVISION)
        3. Falls back to substring matching for model compatibility
        4. Updates status accordingly

        Args:
            state: Current graph state

        Returns:
            Dict with updated messages and status
        """
        await self._emit_node_active(state, "review")

        # Build review messages - prune context, then add review instruction
        review_messages = sliding_window_prune(list(state["messages"]))
        review_messages.append({
            "role": "user",
            "content": REVIEW_PROMPT,
        })

        # Call the LLM for review - no tools, lower temperature for determinism
        response = await self.llm_client.call(
            messages=review_messages,
            tools=None,
            session_id=state["session_id"],
            agent_id=state["agent_id"],
            temperature=0.3,
        )

        # Emit the review content
        await self._emit_thinking(state, response.content)

        # Check for structured completion signals first
        status_tag = parse_status_tag(response.content)

        if status_tag == "TASK_COMPLETE":
            is_complete = True
        elif status_tag == "NEEDS_REVISION":
            is_complete = False
        else:
            # Fallback to substring matching for model compatibility
            content_lower = response.content.lower()
            is_complete = "task_complete" in content_lower

        # Format the assistant message (no tool calls in review)
        assistant_message = format_assistant_message_with_tools(
            response.content,
            [],
        )

        # Prevent premature completion when code changed but no successful
        # build or lint evidence exists in-session.
        completion_block_reason = ""
        if is_complete and state.get("files_written"):
            missing_checks: list[str] = []
            if not state.get("build_verified", False):
                missing_checks.append("`npm run build`")
            if not state.get("lint_verified", False):
                missing_checks.append("`npm run lint`")

            if missing_checks:
                is_complete = False
                checks_str = " and ".join(missing_checks)
                completion_block_reason = (
                    f"Completion blocked: no successful {checks_str} result is recorded. "
                    f"Run {checks_str}, fix errors, then review again."
                )
                logger.info(
                    "review_completion_blocked_missing_verification",
                    session_id=state["session_id"],
                    files_written=len(state.get("files_written", [])),
                    missing_checks=missing_checks,
                )

        # Determine new status
        new_status = "complete" if is_complete else "reasoning"

        logger.info(
            "self_review_complete",
            session_id=state["session_id"],
            is_complete=is_complete,
            status_tag=status_tag,
            new_status=new_status,
        )

        await self._emit_node_complete(state, "review")

        if completion_block_reason:
            return {
                "messages": [
                    assistant_message,
                    {"role": "user", "content": completion_block_reason},
                ],
                "status": new_status,
            }

        if not is_complete:
            # Ensure the next reason step receives an explicit user turn.
            # Some models stall or emit minimal output when the prior turn is
            # assistant-only review feedback without a follow-up user message.
            revision_handoff = (
                "Continue implementing now based on the review above. "
                "Use tools to inspect files, make targeted edits, and run "
                "`npm run build` / `npm run lint` before reviewing again."
            )
            return {
                "messages": [
                    assistant_message,
                    {"role": "user", "content": revision_handoff},
                ],
                "status": new_status,
            }

        return {"messages": [assistant_message], "status": new_status}

    def _should_continue(self, state: ReactState) -> str:
        """Determine whether to continue or end the graph.

        Args:
            state: Current graph state

        Returns:
            "continue" to loop back to reason, "end" to finish
        """
        # Check iteration limit
        if state["iteration"] >= state["max_iterations"]:
            logger.warning(
                "max_iterations_reached",
                session_id=state["session_id"],
                iterations=state["iteration"],
                max_iterations=state["max_iterations"],
            )
            return "end"

        # Check status
        if state["status"] in ("complete", "failed"):
            return "end"

        return "continue"

    async def run(self, initial_state: ReactState) -> ReactState:
        """Run the graph to completion.

        Args:
            initial_state: The starting state

        Returns:
            The final state after completion
        """
        # Emit graph initialized event
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.GRAPH_INITIALIZED,
                session_id=initial_state["session_id"],
                agent_id=initial_state["agent_id"],
                agent_role="ReAct Agent",
                data={
                    "nodes": ["reason", "execute", "review"],
                    "edges": [
                        {"source": "START", "target": "reason"},
                        {"source": "reason", "target": "execute"},
                        {"source": "execute", "target": "review"},
                        {"source": "review", "target": "reason", "condition": "continue"},
                        {"source": "review", "target": "END", "condition": "end"},
                    ],
                },
            )
        )

        # Emit agent spawned event
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.AGENT_SPAWNED,
                session_id=initial_state["session_id"],
                agent_id=initial_state["agent_id"],
                agent_role="ReAct Agent",
                data={
                    "role": "ReAct Agent",
                    "sandbox_id": initial_state["sandbox_id"],
                },
            )
        )

        # Run the graph
        final_state = await self._compiled_graph.ainvoke(initial_state)

        # Emit agent complete event
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.AGENT_COMPLETE,
                session_id=initial_state["session_id"],
                agent_id=initial_state["agent_id"],
                agent_role="ReAct Agent",
                data={
                    "status": final_state.get("status", "unknown"),
                    "iterations": final_state.get("iteration", 0),
                    "files_written": final_state.get("files_written", []),
                },
            )
        )

        return final_state

    async def stream(self, initial_state: ReactState):
        """Stream the graph execution, yielding state updates.

        This is useful for real-time UI updates.

        Args:
            initial_state: The starting state

        Yields:
            State dicts as the graph progresses
        """
        # Emit graph initialized event
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.GRAPH_INITIALIZED,
                session_id=initial_state["session_id"],
                agent_id=initial_state["agent_id"],
                agent_role="ReAct Agent",
                data={
                    "nodes": ["reason", "execute", "review"],
                    "edges": [
                        {"source": "START", "target": "reason"},
                        {"source": "reason", "target": "execute"},
                        {"source": "execute", "target": "review"},
                        {"source": "review", "target": "reason", "condition": "continue"},
                        {"source": "review", "target": "END", "condition": "end"},
                    ],
                },
            )
        )

        # Emit agent spawned event
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.AGENT_SPAWNED,
                session_id=initial_state["session_id"],
                agent_id=initial_state["agent_id"],
                agent_role="ReAct Agent",
                data={
                    "role": "ReAct Agent",
                    "sandbox_id": initial_state["sandbox_id"],
                },
            )
        )

        # Stream the graph execution
        final_state = None
        async for state in self._compiled_graph.astream(initial_state):
            final_state = state
            yield state

        # Emit agent complete event
        if final_state:
            # Extract the actual state from the last event
            # LangGraph stream returns {node_name: state_update} dicts
            actual_state = initial_state.copy()
            if isinstance(final_state, dict):
                for node_update in final_state.values():
                    if isinstance(node_update, dict):
                        actual_state.update(node_update)

            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.AGENT_COMPLETE,
                    session_id=initial_state["session_id"],
                    agent_id=initial_state["agent_id"],
                    agent_role="ReAct Agent",
                    data={
                        "status": actual_state.get("status", "unknown"),
                        "iterations": actual_state.get("iteration", 0),
                        "files_written": actual_state.get("files_written", []),
                    },
                )
            )


def create_react_graph(
    sandbox_manager: SandboxManager,
    event_bus: EventBus,
    llm_client: LLMClient | None = None,
    temperature: float = 0.7,
) -> ReactGraph:
    """Factory function to create a ReAct graph.

    This is the main entry point for creating a ReAct agent.

    Args:
        sandbox_manager: Manager for sandbox operations
        event_bus: Event bus for emitting events
        llm_client: Optional LLM client (creates default if None)

    Returns:
        Configured ReactGraph instance

    Example:
        >>> from backend.sandbox.docker_sandbox import SandboxManager
        >>> from backend.events.bus import get_event_bus
        >>>
        >>> sandbox_manager = SandboxManager()
        >>> event_bus = get_event_bus()
        >>> graph = create_react_graph(sandbox_manager, event_bus)
        >>>
        >>> # Create initial state
        >>> state = create_initial_state(
        ...     task="Build a todo app with React",
        ...     sandbox_id="sandbox_123",
        ...     session_id="session_456",
        ... )
        >>>
        >>> # Run the graph
        >>> final_state = await graph.run(state)
    """
    return ReactGraph(
        sandbox_manager=sandbox_manager,
        event_bus=event_bus,
        llm_client=llm_client,
        temperature=temperature,
    )
