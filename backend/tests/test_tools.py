"""Tests for agents/tools.py -- tool definitions and ToolExecutor.

Covers tool dispatch, event emission, error handling, and the
search_files command construction with shlex.quote escaping.
"""

import shlex
from unittest.mock import AsyncMock

from agents.tools import (
    TOOL_DEFINITIONS,
    ToolCall,
    ToolExecutor,
    get_tool_definitions_for_llm,
)
from config import settings
from events.bus import EventBus
from events.types import AgentEvent, EventType
from sandbox.docker_sandbox import CommandResult, FileInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_executor(
    sandbox_manager: AsyncMock | None = None,
    event_bus: EventBus | None = None,
) -> ToolExecutor:
    """Create a ToolExecutor with mocked dependencies."""
    if sandbox_manager is None:
        sandbox_manager = AsyncMock()
    if event_bus is None:
        event_bus = EventBus()
    return ToolExecutor(sandbox_manager=sandbox_manager, event_bus=event_bus)


# =========================================================================
# Tool Definitions
# =========================================================================


class TestToolDefinitions:
    """Ensure tool definitions are well-formed."""

    def test_tool_count(self) -> None:
        assert len(TOOL_DEFINITIONS) == 5

    def test_all_have_name_and_description(self) -> None:
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool

    def test_tool_names(self) -> None:
        names = {t["name"] for t in TOOL_DEFINITIONS}
        assert names == {"write_file", "read_file", "list_files", "execute_command", "search_files"}

    def test_llm_format(self) -> None:
        formatted = get_tool_definitions_for_llm()
        assert len(formatted) == 5
        for tool in formatted:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]


# =========================================================================
# ToolExecutor -- write_file
# =========================================================================


class TestToolExecutorWriteFile:
    """write_file tool dispatch."""

    async def test_write_file_success(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.write_file = AsyncMock()
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="write_file",
            args={"path": "src/App.tsx", "content": "export default function App() {}"},
            session_id="sess_1",
        )

        assert result.success is True
        assert "Successfully wrote" in result.content
        sandbox_mgr.write_file.assert_awaited_once_with(
            "sb_1", "src/App.tsx", "export default function App() {}"
        )

    async def test_write_file_missing_path(self) -> None:
        executor = _make_executor()
        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="write_file",
            args={"content": "hello"},
            session_id="sess_1",
        )
        assert result.success is False
        assert "missing required arguments" in result.content.lower()

    async def test_write_file_emits_file_changed_event(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.write_file = AsyncMock()
        bus = EventBus()
        queue = bus.subscribe("sess_1")
        executor = _make_executor(sandbox_manager=sandbox_mgr, event_bus=bus)

        await executor.execute(
            sandbox_id="sb_1",
            tool_name="write_file",
            args={"path": "index.js", "content": "console.log('hi')"},
            session_id="sess_1",
        )

        # Should have AGENT_TOOL_CALL, FILE_CHANGED, AGENT_TOOL_RESULT
        events: list[AgentEvent] = []
        while not queue.empty():
            events.append(queue.get_nowait())

        event_types = [e.type for e in events]
        assert EventType.AGENT_TOOL_CALL in event_types
        assert EventType.FILE_CHANGED in event_types
        assert EventType.AGENT_TOOL_RESULT in event_types


# =========================================================================
# ToolExecutor -- read_file
# =========================================================================


class TestToolExecutorReadFile:
    """read_file tool dispatch."""

    async def test_read_file_success(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.read_file = AsyncMock(return_value="file content here")
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="read_file",
            args={"path": "src/App.tsx"},
            session_id="sess_1",
        )

        assert result.success is True
        assert result.content == "file content here"

    async def test_read_file_not_found(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.read_file = AsyncMock(side_effect=FileNotFoundError("Not found"))
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="read_file",
            args={"path": "nonexistent.ts"},
            session_id="sess_1",
        )

        assert result.success is True  # Returns error string, not exception
        assert "not found" in result.content.lower()

    async def test_read_file_missing_path(self) -> None:
        executor = _make_executor()
        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="read_file",
            args={},
            session_id="sess_1",
        )
        assert result.success is False
        assert "missing required arguments" in result.content.lower()


# =========================================================================
# ToolExecutor -- list_files
# =========================================================================


class TestToolExecutorListFiles:
    """list_files tool dispatch."""

    async def test_list_files_success(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.list_files = AsyncMock(return_value=[
            FileInfo(name="App.tsx", path="src/App.tsx", is_directory=False, size=100),
            FileInfo(name="components", path="src/components", is_directory=True, size=0),
        ])
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="list_files",
            args={"path": "src"},
            session_id="sess_1",
        )

        assert result.success is True
        assert "App.tsx" in result.content
        assert "components/" in result.content  # directories have / suffix

    async def test_list_files_empty(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.list_files = AsyncMock(return_value=[])
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="list_files",
            args={},
            session_id="sess_1",
        )

        assert result.success is True
        assert "No files" in result.content


# =========================================================================
# ToolExecutor -- execute_command
# =========================================================================


class TestToolExecutorExecuteCommand:
    """execute_command tool dispatch."""

    async def test_execute_success(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.execute_command = AsyncMock(return_value=CommandResult(
            stdout="installed 42 packages",
            stderr="",
            exit_code=0,
            timed_out=False,
        ))
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="execute_command",
            args={"command": "npm install"},
            session_id="sess_1",
        )

        assert result.success is True
        assert "42 packages" in result.content
        sandbox_mgr.execute_command.assert_awaited_once_with(
            "sb_1",
            "npm install",
            timeout=settings.tool_timeout_seconds,
        )

    async def test_execute_failure(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.execute_command = AsyncMock(return_value=CommandResult(
            stdout="",
            stderr="Error: module not found",
            exit_code=1,
            timed_out=False,
        ))
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="execute_command",
            args={"command": "npm test"},
            session_id="sess_1",
        )

        assert result.success is False
        assert "command failed" in result.content.lower()

    async def test_execute_failure_npm_missing_package_shows_hint(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.execute_command = AsyncMock(return_value=CommandResult(
            stdout="npm error code ENOENT\nnpm error Could not read package.json",
            stderr="",
            exit_code=254,
            timed_out=False,
        ))
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="execute_command",
            args={"command": "npm install"},
            session_id="sess_1",
        )

        assert result.success is False
        assert "hints:" in result.content.lower()
        assert "package.json" in result.content
        assert "npm init -y" in result.content

    async def test_execute_failure_tailwind_init_shows_hint(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.execute_command = AsyncMock(return_value=CommandResult(
            stdout="npm error could not determine executable to run",
            stderr="",
            exit_code=1,
            timed_out=False,
        ))
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="execute_command",
            args={"command": "npx tailwindcss init -p"},
            session_id="sess_1",
        )

        assert result.success is False
        assert "tailwind css v4" in result.content.lower()
        assert "@tailwindcss/postcss" in result.content

    async def test_execute_failure_missing_autoprefixer_shows_hint(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.execute_command = AsyncMock(return_value=CommandResult(
            stdout="Failed to load PostCSS config: Cannot find module 'autoprefixer'",
            stderr="",
            exit_code=1,
            timed_out=False,
        ))
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="execute_command",
            args={"command": "npm run build"},
            session_id="sess_1",
        )

        assert result.success is False
        assert "autoprefixer" in result.content.lower()
        assert "@tailwindcss/postcss" in result.content

    async def test_execute_failure_eslint_flat_config_mismatch_shows_hint(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.execute_command = AsyncMock(return_value=CommandResult(
            stdout=(
                "You're using eslint.config.js, some command line flags are no longer available.\n"
                "Invalid option '--ext'"
            ),
            stderr="",
            exit_code=2,
            timed_out=False,
        ))
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="execute_command",
            args={"command": "npx eslint . --ext .ts,.tsx"},
            session_id="sess_1",
        )

        assert result.success is False
        assert "eslint config/version mismatch" in result.content.lower()
        assert "eslint v9" in result.content

    async def test_execute_timeout(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.execute_command = AsyncMock(return_value=CommandResult(
            stdout="partial output",
            stderr="",
            exit_code=124,
            timed_out=True,
        ))
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="execute_command",
            args={"command": "npm run build"},
            session_id="sess_1",
        )

        assert result.success is False
        assert "timed out" in result.content.lower()

    async def test_execute_emits_command_events(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.execute_command = AsyncMock(return_value=CommandResult(
            stdout="OK", stderr="", exit_code=0, timed_out=False,
        ))
        bus = EventBus()
        queue = bus.subscribe("sess_1")
        executor = _make_executor(sandbox_manager=sandbox_mgr, event_bus=bus)

        await executor.execute(
            sandbox_id="sb_1",
            tool_name="execute_command",
            args={"command": "npm install"},
            session_id="sess_1",
        )

        events: list[AgentEvent] = []
        while not queue.empty():
            events.append(queue.get_nowait())

        event_types = [e.type for e in events]
        assert EventType.COMMAND_STARTED in event_types
        assert EventType.COMMAND_COMPLETE in event_types

    async def test_execute_missing_command(self) -> None:
        executor = _make_executor()
        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="execute_command",
            args={},
            session_id="sess_1",
        )
        assert result.success is False
        assert "missing required arguments" in result.content.lower()

    async def test_execute_no_output(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.execute_command = AsyncMock(return_value=CommandResult(
            stdout="",
            stderr="",
            exit_code=0,
            timed_out=False,
        ))
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="execute_command",
            args={"command": "mkdir src"},
            session_id="sess_1",
        )

        assert result.success is True
        assert result.content == "(no output)"

    async def test_execute_blocks_dev_server_command(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.execute_command = AsyncMock()
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="execute_command",
            args={"command": "npm run dev"},
            session_id="sess_1",
        )

        assert result.success is False
        assert "long-running dev/watch processes" in result.content
        sandbox_mgr.execute_command.assert_not_called()

    async def test_execute_combines_stdout_and_stderr(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.execute_command = AsyncMock(return_value=CommandResult(
            stdout="build ok",
            stderr="warning: deprecation",
            exit_code=0,
            timed_out=False,
        ))
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="execute_command",
            args={"command": "npm run build"},
            session_id="sess_1",
        )

        assert result.success is True
        assert "STDOUT:" in result.content
        assert "STDERR:" in result.content


# =========================================================================
# ToolExecutor -- search_files (grep with shlex.quote)
# =========================================================================


class TestToolExecutorSearchFiles:
    """search_files tool with proper escaping via shlex.quote."""

    async def test_search_basic(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.execute_command = AsyncMock(return_value=CommandResult(
            stdout="src/App.tsx:5:import React",
            stderr="",
            exit_code=0,
            timed_out=False,
        ))
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="search_files",
            args={"pattern": "import React", "path": "src"},
            session_id="sess_1",
        )

        assert result.success is True
        assert "App.tsx" in result.content

        # Verify the command was constructed with shlex.quote
        cmd_arg = sandbox_mgr.execute_command.call_args[0][1]
        assert shlex.quote("import React") in cmd_arg
        assert shlex.quote("src") in cmd_arg

    async def test_search_with_glob(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.execute_command = AsyncMock(return_value=CommandResult(
            stdout="match", stderr="", exit_code=0, timed_out=False,
        ))
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        await executor.execute(
            sandbox_id="sb_1",
            tool_name="search_files",
            args={"pattern": "TODO", "file_glob": "*.tsx"},
            session_id="sess_1",
        )

        cmd_arg = sandbox_mgr.execute_command.call_args[0][1]
        assert "--include=" in cmd_arg
        assert shlex.quote("*.tsx") in cmd_arg

    async def test_search_no_matches(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.execute_command = AsyncMock(return_value=CommandResult(
            stdout="", stderr="", exit_code=1, timed_out=False,
        ))
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="search_files",
            args={"pattern": "nonexistent_string"},
            session_id="sess_1",
        )

        assert "No matches" in result.content

    async def test_search_missing_pattern(self) -> None:
        executor = _make_executor()
        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="search_files",
            args={},
            session_id="sess_1",
        )
        assert result.success is False
        assert "missing required arguments" in result.content.lower()

    async def test_search_pattern_with_special_chars(self) -> None:
        """Ensure special characters in patterns are escaped by shlex.quote."""
        sandbox_mgr = AsyncMock()
        sandbox_mgr.execute_command = AsyncMock(return_value=CommandResult(
            stdout="", stderr="", exit_code=1, timed_out=False,
        ))
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        await executor.execute(
            sandbox_id="sb_1",
            tool_name="search_files",
            args={"pattern": "'; rm -rf /; echo '"},
            session_id="sess_1",
        )

        cmd_arg = sandbox_mgr.execute_command.call_args[0][1]
        # The pattern should be properly quoted -- shlex.quote wraps in single
        # quotes and escapes internal single quotes
        escaped = shlex.quote("'; rm -rf /; echo '")
        assert escaped in cmd_arg


# =========================================================================
# ToolExecutor -- unknown tool
# =========================================================================


class TestToolExecutorUnknownTool:
    """Unknown tool name raises ValueError which is caught as error result."""

    async def test_unknown_tool(self) -> None:
        executor = _make_executor()
        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="delete_everything",
            args={},
            session_id="sess_1",
        )
        assert result.success is False
        assert "Unknown tool" in result.content

    async def test_invalid_tool_args_type(self) -> None:
        executor = _make_executor()
        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="read_file",
            args=["src/App.tsx"],
            session_id="sess_1",
        )
        assert result.success is False
        assert "expected an object" in result.content.lower()


# =========================================================================
# ToolExecutor -- execute_tool_call convenience method
# =========================================================================


class TestExecuteToolCall:
    """Convenience method that accepts a ToolCall dataclass."""

    async def test_execute_tool_call(self) -> None:
        sandbox_mgr = AsyncMock()
        sandbox_mgr.read_file = AsyncMock(return_value="content")
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        tc = ToolCall(id="call_1", name="read_file", args={"path": "package.json"})
        result = await executor.execute_tool_call(
            sandbox_id="sb_1",
            tool_call=tc,
            session_id="sess_1",
            agent_id="agent_1",
            agent_role="ReAct Agent",
        )

        assert result.success is True
        assert result.tool_call_id == "call_1"


# =========================================================================
# ToolExecutor -- result truncation
# =========================================================================


class TestToolResultTruncation:
    """Tool results longer than 2000 chars are truncated in events."""

    async def test_long_result_truncated_in_event(self) -> None:
        long_content = "x" * 5000
        sandbox_mgr = AsyncMock()
        sandbox_mgr.read_file = AsyncMock(return_value=long_content)
        bus = EventBus()
        queue = bus.subscribe("sess_1")
        executor = _make_executor(sandbox_manager=sandbox_mgr, event_bus=bus)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="read_file",
            args={"path": "big.txt"},
            session_id="sess_1",
        )

        # The full result is available in the ToolResult
        assert len(result.content) == 5000

        # But the event should have a truncated version
        events: list[AgentEvent] = []
        while not queue.empty():
            events.append(queue.get_nowait())

        tool_result_events = [e for e in events if e.type == EventType.AGENT_TOOL_RESULT]
        assert len(tool_result_events) == 1
        assert len(tool_result_events[0].data["result"]) == 2000
