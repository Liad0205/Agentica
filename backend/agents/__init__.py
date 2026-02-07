"""Agent tools, prompts, LLM integration, and graph implementations.

This module exports the key components needed for agent execution:
- Tool definitions and executor for sandbox operations
- System prompts for different agent roles
- LLM client utilities with retry logic and metrics tracking
- ReAct graph for single-agent coding tasks
"""

from agents.prompts import (
    BASE_CODING_AGENT_PROMPT,
    EVALUATOR_PROMPT,
    ORCHESTRATOR_PROMPT,
    REVIEW_PROMPT,
    SOLVER_PERSONAS,
    get_solver_prompt,
    get_subtask_prompt,
)
from agents.react_graph import (
    ReactGraph,
    ReactState,
    create_initial_state,
    create_react_graph,
)
from agents.tools import (
    TOOL_DEFINITIONS,
    ToolCall,
    ToolExecutor,
    ToolResult,
    get_tool_definitions_for_llm,
)
from agents.utils import (
    LLMClient,
    LLMResponse,
    MockLLMClient,
    ToolCallData,
    call_llm,
    extract_json_from_response,
    format_assistant_message_with_tools,
    format_tool_result_for_llm,
    parse_plan_tag,
    parse_tool_calls,
)

__all__ = [
    # Tools
    "TOOL_DEFINITIONS",
    "ToolCall",
    "ToolExecutor",
    "ToolResult",
    "get_tool_definitions_for_llm",
    # Prompts
    "BASE_CODING_AGENT_PROMPT",
    "ORCHESTRATOR_PROMPT",
    "EVALUATOR_PROMPT",
    "REVIEW_PROMPT",
    "SOLVER_PERSONAS",
    "get_solver_prompt",
    "get_subtask_prompt",
    # Utils
    "LLMClient",
    "LLMResponse",
    "MockLLMClient",
    "ToolCallData",
    "call_llm",
    "extract_json_from_response",
    "format_assistant_message_with_tools",
    "format_tool_result_for_llm",
    "parse_plan_tag",
    "parse_tool_calls",
    # ReAct Graph
    "ReactState",
    "ReactGraph",
    "create_react_graph",
    "create_initial_state",
]
