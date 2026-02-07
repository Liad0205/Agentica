"""Tool definitions and sandbox dispatch for AI agents.

This module defines the tools available to all agents and provides the
ToolExecutor class that routes tool calls to the appropriate sandbox operations.
"""

import asyncio
import contextlib
import re
import shlex
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import structlog

from config import settings
from events.bus import EventBus
from events.types import AgentEvent, EventType
from sandbox.docker_sandbox import SandboxManager

if TYPE_CHECKING:
    from metrics import MetricsCollector

logger = structlog.get_logger()


# Tool definitions following the design specification
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "write_file",
        "description": (
            "Write or create a file at the given path relative to /workspace. "
            "Creates parent directories automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative file path, e.g. 'src/App.tsx'",
                },
                "content": {
                    "type": "string",
                    "description": "Complete file content",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file at the given path relative to /workspace."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative file path",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": (
            "List files and directories at the given path. "
            "Returns names with '/' suffix for directories."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative directory path, default '.'",
                },
            },
            "required": [],
        },
    },
    {
        "name": "execute_command",
        "description": (
            "Execute a shell command in the sandbox terminal. "
            "Use for: npm install, npm run build, npx eslint, npm test, etc. "
            "Working directory is /workspace. Timeout: 60 seconds."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Shell command to execute directly inside the isolated sandbox."
                    ),
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "search_files",
        "description": (
            "Search for a text pattern across files using grep. "
            "Returns matching lines with file paths."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex supported)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in, default '.'",
                },
                "file_glob": {
                    "type": "string",
                    "description": "File glob pattern, e.g. '*.tsx'",
                },
            },
            "required": ["pattern"],
        },
    },
]

_TOOL_DEFINITION_MAP: dict[str, dict[str, Any]] = {
    tool["name"]: tool for tool in TOOL_DEFINITIONS
}

# Keep tool payloads bounded so a single call cannot flood model context.
MAX_READ_FILE_CHARS = 60_000
MAX_SEARCH_OUTPUT_CHARS = 15_000
MAX_COMMAND_OUTPUT_CHARS = 20_000

# Guard against commands that either never terminate or are clearly harmful.
_DEV_SERVER_COMMAND_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:npm|pnpm|yarn)\s+(?:run\s+)?dev\b", re.IGNORECASE),
    re.compile(r"^\s*(?:npm|pnpm|yarn)\s+start\b", re.IGNORECASE),
    re.compile(r"^\s*(?:vite|next\s+dev|webpack\s+serve)\b", re.IGNORECASE),
)
_BLOCKED_COMMAND_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brm\s+-rf\s+/(?:\s|$)"),
    re.compile(r"\brm\s+-rf\s+/\*"),
    re.compile(r"\b(?:shutdown|reboot|halt|poweroff)\b", re.IGNORECASE),
    re.compile(r"\bmkfs\.", re.IGNORECASE),
    re.compile(r"\bdd\s+if=.*\s+of=/dev/", re.IGNORECASE),
    re.compile(r":\(\)\{:\|:&\};:"),
)


class ToolArgumentError(ValueError):
    """Raised when a tool call has invalid or unsupported arguments."""


def get_tool_definitions_for_llm() -> list[dict[str, Any]]:
    """Get tool definitions formatted for LLM function calling.

    Returns:
        List of tool definitions in the format expected by LiteLLM.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in TOOL_DEFINITIONS
    ]


@dataclass
class ToolCall:
    """Represents a parsed tool call from an LLM response.

    Attributes:
        id: Unique identifier for this tool call (from LLM)
        name: Name of the tool to execute
        args: Arguments to pass to the tool
    """

    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool.

    Attributes:
        tool_call_id: ID of the tool call this result corresponds to
        content: The result content as a string
        success: Whether the tool execution succeeded
        error: Error message if execution failed
    """

    tool_call_id: str
    content: str
    success: bool
    error: str | None = None


class ToolExecutor:
    """Executes tool calls against a sandbox and emits events.

    The ToolExecutor routes tool calls to the appropriate SandboxManager
    methods and emits events for each tool call and result.

    Attributes:
        sandbox_manager: The SandboxManager instance for sandbox operations.
        event_bus: The EventBus for emitting tool events.
    """

    def __init__(
        self,
        sandbox_manager: SandboxManager,
        event_bus: EventBus,
        metrics_collector: Optional["MetricsCollector"] = None,
        auto_preview: bool = True,
    ) -> None:
        """Initialize the ToolExecutor.

        Args:
            sandbox_manager: Manager for sandbox operations.
            event_bus: Event bus for emitting tool events.
            metrics_collector: Optional collector for tool-call metrics.
            auto_preview: Whether to auto-start dev server after npm install.
                Set to False for subtask/solver sandboxes to avoid wasteful
                dev server starts.
        """
        self.sandbox_manager = sandbox_manager
        self.event_bus = event_bus
        self.metrics_collector = metrics_collector
        self._auto_preview = auto_preview
        self._background_tasks: set[asyncio.Task[None]] = set()

    def _truncate_text(self, text: str, *, max_chars: int) -> str:
        """Trim large text payloads while preserving a clear truncation marker."""
        if len(text) <= max_chars:
            return text
        omitted = len(text) - max_chars
        return (
            f"{text[:max_chars]}\n"
            f"... [truncated {omitted} characters to protect context window]"
        )

    def _summarize_args_for_event(self, args: Any) -> dict[str, Any]:
        """Create a lightweight args payload for event emission."""
        if not isinstance(args, dict):
            return {"raw": str(args)[:500]}

        summarized: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str) and len(value) > 500:
                summarized[key] = f"{value[:500]}... [truncated]"
            else:
                summarized[key] = value
        return summarized

    def _normalize_tool_args(
        self,
        tool_name: str,
        args: Any,
    ) -> dict[str, Any]:
        """Validate and normalize tool arguments against schema metadata."""
        tool_def = _TOOL_DEFINITION_MAP.get(tool_name)
        if tool_def is None:
            raise ToolArgumentError(f"Unknown tool: {tool_name}")

        if not isinstance(args, dict):
            raise ToolArgumentError(
                f"Invalid arguments for {tool_name}: expected an object"
            )

        params = tool_def.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        normalized: dict[str, Any] = {}
        for key, value in args.items():
            if key not in properties:
                # Ignore unknown fields to keep calls resilient to model drift.
                continue

            if key in {"path", "command", "pattern", "file_glob"}:
                if not isinstance(value, str):
                    raise ToolArgumentError(
                        f"Invalid type for '{key}': expected string"
                    )
                normalized[key] = value.strip()
            elif key == "content":
                if not isinstance(value, str):
                    raise ToolArgumentError(
                        "Invalid type for 'content': expected string"
                    )
                normalized[key] = value
            else:
                normalized[key] = value

        missing = []
        for req in required:
            value = normalized.get(req)
            if value is None or isinstance(value, str) and not value:
                missing.append(req)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ToolArgumentError(f"Missing required arguments: {joined}")

        return normalized

    def _preflight_command(self, command: str) -> None:
        """Reject commands that are harmful or unsuitable for agent loops."""
        for pattern in _BLOCKED_COMMAND_PATTERNS:
            if pattern.search(command):
                raise ToolArgumentError(
                    "Command blocked by safety policy: potentially destructive operation"
                )

        for pattern in _DEV_SERVER_COMMAND_PATTERNS:
            if pattern.search(command):
                raise ToolArgumentError(
                    "Command blocked: long-running dev/watch processes are not supported. "
                    "Run short verification commands like `npm run build`/`npm run lint`."
                )

    def _compose_command_output(
        self,
        *,
        stdout: str,
        stderr: str,
        max_chars: int = MAX_COMMAND_OUTPUT_CHARS,
    ) -> tuple[str, str]:
        """Combine stdout/stderr into one report and a stream label."""
        clean_stdout = stdout.strip()
        clean_stderr = stderr.strip()

        if clean_stdout and clean_stderr:
            combined = f"STDOUT:\n{clean_stdout}\n\nSTDERR:\n{clean_stderr}"
            return self._truncate_text(combined, max_chars=max_chars), "combined"
        if clean_stdout:
            return self._truncate_text(clean_stdout, max_chars=max_chars), "stdout"
        if clean_stderr:
            return self._truncate_text(clean_stderr, max_chars=max_chars), "stderr"
        return "", "none"

    async def execute(
        self,
        sandbox_id: str,
        tool_name: str,
        args: Any,
        session_id: str,
        agent_id: str | None = None,
        agent_role: str | None = None,
        tool_call_id: str | None = None,
    ) -> ToolResult:
        """Execute a tool call against the sandbox.

        Args:
            sandbox_id: The sandbox to execute the tool in.
            tool_name: Name of the tool to execute.
            args: Arguments for the tool.
            session_id: Session ID for event emission.
            agent_id: Optional agent ID for event emission.
            agent_role: Optional agent role for event emission.
            tool_call_id: Optional tool call ID from LLM.

        Returns:
            ToolResult with the execution outcome.
        """
        start_time = time.time()
        call_id = tool_call_id or f"tool_{int(start_time * 1000)}"
        event_args = self._summarize_args_for_event(args)

        # Emit tool call event
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.AGENT_TOOL_CALL,
                session_id=session_id,
                agent_id=agent_id,
                agent_role=agent_role,
                data={
                    "tool": tool_name,
                    "args": event_args,
                    "tool_call_id": call_id,
                },
            )
        )

        if self.metrics_collector is not None:
            await self.metrics_collector.record_tool_call(session_id)

        try:
            normalized_args = self._normalize_tool_args(tool_name, args)
            result = await self._dispatch_tool(
                sandbox_id, tool_name, normalized_args, session_id,
                agent_id=agent_id, agent_role=agent_role,
            )
            success = True
            error = None
        except Exception as e:
            logger.error(
                "tool_execution_failed",
                tool_name=tool_name,
                sandbox_id=sandbox_id,
                error=str(e),
            )
            result = f"Error: {e}"
            success = False
            error = str(e)

        duration_ms = int((time.time() - start_time) * 1000)

        # Emit tool result event
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.AGENT_TOOL_RESULT,
                session_id=session_id,
                agent_id=agent_id,
                agent_role=agent_role,
                data={
                    "tool": tool_name,
                    "result": result[:2000] if len(result) > 2000 else result,
                    "success": success,
                    "tool_call_id": call_id,
                    "duration_ms": duration_ms,
                },
            )
        )

        logger.debug(
            "tool_executed",
            tool_name=tool_name,
            sandbox_id=sandbox_id,
            success=success,
            duration_ms=duration_ms,
        )

        return ToolResult(
            tool_call_id=call_id,
            content=result,
            success=success,
            error=error,
        )

    async def execute_tool_call(
        self,
        sandbox_id: str,
        tool_call: ToolCall,
        session_id: str,
        agent_id: str | None = None,
        agent_role: str | None = None,
    ) -> ToolResult:
        """Execute a ToolCall object against the sandbox.

        Convenience method that extracts fields from a ToolCall.

        Args:
            sandbox_id: The sandbox to execute the tool in.
            tool_call: The ToolCall to execute.
            session_id: Session ID for event emission.
            agent_id: Optional agent ID for event emission.
            agent_role: Optional agent role for event emission.

        Returns:
            ToolResult with the execution outcome.
        """
        return await self.execute(
            sandbox_id=sandbox_id,
            tool_name=tool_call.name,
            args=tool_call.args,
            session_id=session_id,
            agent_id=agent_id,
            agent_role=agent_role,
            tool_call_id=tool_call.id,
        )

    async def _dispatch_tool(
        self,
        sandbox_id: str,
        tool_name: str,
        args: dict[str, Any],
        session_id: str,
        agent_id: str | None = None,
        agent_role: str | None = None,
    ) -> str:
        """Route tool call to appropriate sandbox method.

        Args:
            sandbox_id: The sandbox to execute in.
            tool_name: Name of the tool.
            args: Tool arguments.
            session_id: Session ID for file change events.
            agent_id: Optional agent ID for command events.
            agent_role: Optional agent role for command events.

        Returns:
            String result from the tool execution.

        Raises:
            ValueError: If tool_name is unknown.
        """
        if tool_name == "write_file":
            return await self._execute_write_file(sandbox_id, args, session_id)
        elif tool_name == "read_file":
            return await self._execute_read_file(sandbox_id, args)
        elif tool_name == "list_files":
            return await self._execute_list_files(sandbox_id, args)
        elif tool_name == "execute_command":
            return await self._execute_command(
                sandbox_id, args, session_id,
                agent_id=agent_id, agent_role=agent_role,
            )
        elif tool_name == "search_files":
            return await self._execute_search_files(sandbox_id, args)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    async def _execute_write_file(
        self,
        sandbox_id: str,
        args: dict[str, Any],
        session_id: str,
    ) -> str:
        """Execute write_file tool.

        Args:
            sandbox_id: The sandbox to write to.
            args: Must contain 'path' and 'content'.
            session_id: Session ID for file change event.

        Returns:
            Success message.
        """
        path = args["path"]
        content = args["content"]

        await self.sandbox_manager.write_file(sandbox_id, path, content)

        # Emit file changed event
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.FILE_CHANGED,
                session_id=session_id,
                data={
                    "path": path,
                    "content": content,
                    "sandbox_id": sandbox_id,
                },
            )
        )

        return f"Successfully wrote {len(content)} bytes to {path}"

    async def _execute_read_file(
        self,
        sandbox_id: str,
        args: dict[str, Any],
    ) -> str:
        """Execute read_file tool.

        Args:
            sandbox_id: The sandbox to read from.
            args: Must contain 'path'.

        Returns:
            File content or error message.
        """
        path = args["path"]

        try:
            content = await self.sandbox_manager.read_file(sandbox_id, path)
            return self._truncate_text(content, max_chars=MAX_READ_FILE_CHARS)
        except FileNotFoundError:
            return f"Error: File not found: {path}"

    async def _execute_list_files(
        self,
        sandbox_id: str,
        args: dict[str, Any],
    ) -> str:
        """Execute list_files tool.

        Args:
            sandbox_id: The sandbox to list files in.
            args: May contain 'path' (defaults to '.').

        Returns:
            Formatted file listing.
        """
        path = args.get("path", ".")

        files = await self.sandbox_manager.list_files(sandbox_id, path)

        if not files:
            return f"No files found in {path}"

        # Format output similar to ls
        lines = []
        for file_info in files:
            suffix = "/" if file_info.is_directory else ""
            lines.append(f"{file_info.name}{suffix}")

        return "\n".join(sorted(lines))

    async def _execute_command(
        self,
        sandbox_id: str,
        args: dict[str, Any],
        session_id: str,
        agent_id: str | None = None,
        agent_role: str | None = None,
    ) -> str:
        """Execute execute_command tool.

        Args:
            sandbox_id: The sandbox to execute in.
            args: Must contain 'command'.
            session_id: Session ID for command events.
            agent_id: Optional agent ID for command events.
            agent_role: Optional agent role for command events.

        Returns:
            Command output or error message.
        """
        command = args["command"]
        self._preflight_command(command)

        # Emit command started event
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.COMMAND_STARTED,
                session_id=session_id,
                agent_id=agent_id,
                agent_role=agent_role,
                data={
                    "command": command,
                    "sandbox_id": sandbox_id,
                },
            )
        )

        result = await self.sandbox_manager.execute_command(
            sandbox_id,
            command,
            timeout=settings.tool_timeout_seconds,
        )
        output, output_stream = self._compose_command_output(
            stdout=result.stdout,
            stderr=result.stderr,
        )

        # Emit command output if any
        if output:
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.COMMAND_OUTPUT,
                    session_id=session_id,
                    agent_id=agent_id,
                    agent_role=agent_role,
                    data={
                        "output": output,
                        "stream": output_stream,
                        "sandbox_id": sandbox_id,
                    },
                )
            )

        # Emit command complete event
        await self.event_bus.publish(
            AgentEvent(
                type=EventType.COMMAND_COMPLETE,
                session_id=session_id,
                agent_id=agent_id,
                agent_role=agent_role,
                data={
                    "command": command,
                    "exit_code": result.exit_code,
                    "sandbox_id": sandbox_id,
                    "timed_out": result.timed_out,
                },
            )
        )

        # Treat command timeout/non-zero exit as tool failures so the agent can
        # react with proper retry/backoff behavior.
        if result.timed_out:
            raise RuntimeError(
                f"Command timed out: {command}\nPartial output:\n{output}"
            )

        if result.exit_code != 0:
            raise RuntimeError(
                self._format_command_failure(
                    command=command,
                    exit_code=result.exit_code,
                    output=output,
                )
            )

        # Auto-start dev server preview after successful npm install
        if self._auto_preview and self._is_npm_install(command):
            task = asyncio.create_task(
                self._auto_start_preview(
                    sandbox_id, session_id, agent_id, agent_role,
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        return output or "(no output)"

    def _format_command_failure(
        self,
        *,
        command: str,
        exit_code: int,
        output: str,
    ) -> str:
        """Format command failure output and attach targeted recovery hints."""
        base = f"Command failed (exit code {exit_code}):\n{output}"
        hints: list[str] = []

        command_lower = command.lower()
        output_lower = output.lower()

        # Common bootstrap mistake: npm commands before project initialization.
        if command_lower.startswith("npm ") and (
            "could not read package.json" in output_lower
            or "open '/workspace/package.json'" in output_lower
        ):
            hints.append(
                "No package.json found in /workspace. Initialize first with "
                "`npm create vite@latest . -- --template react-ts` or `npm init -y`."
            )

        # Tailwind CSS v4 no longer uses `tailwindcss init`.
        if "tailwindcss init" in command_lower and (
            "could not determine executable to run" in output_lower
            or 'unknown command "init"' in output_lower
            or "invalid command: init" in output_lower
        ):
            hints.append(
                "Tailwind CSS v4 does not use `tailwindcss init -p`. "
                "Install `@tailwindcss/postcss` and configure PostCSS instead."
            )

        # Common PostCSS dependency mismatch for scaffolded projects.
        if (
            "autoprefixer" in output_lower
            and "cannot find module" in output_lower
        ) or (
            "failed to load postcss config" in output_lower
            and "autoprefixer" in output_lower
        ):
            hints.append(
                "PostCSS config references `autoprefixer` but it is missing. "
                "Either install it (`npm install -D autoprefixer`) or remove it and use "
                "`@tailwindcss/postcss` for Tailwind v4."
            )

        # ESLint config mode/version mismatch (flat config vs legacy .eslintrc).
        if (
            "you're using eslint.config" in output_lower
            and (
                "invalid option '--ext'" in output_lower
                or "some command line flags are no longer available" in output_lower
            )
        ) or (
            "eslint.config" in output_lower
            and ("eslint v8" in output_lower or "eslint: 8." in output_lower)
        ):
            hints.append(
                "ESLint config/version mismatch detected. "
                "Use eslint v9 with `eslint.config.*` and `eslint .`, "
                "or use eslint v8 with `.eslintrc.*`."
            )

        if not hints:
            return base

        hints_text = "\n".join(f"- {hint}" for hint in hints)
        return f"{base}\n\nHints:\n{hints_text}"

    # Pattern matches: npm install, npm i, npm ci, pnpm install
    _NPM_INSTALL_RE = re.compile(
        r"^\s*(?:npm\s+(?:install|i|ci)|pnpm\s+install)\b"
    )

    def _is_npm_install(self, command: str) -> bool:
        """Check if a command is an npm/pnpm install variant.

        Args:
            command: The shell command string.

        Returns:
            True if the command is an npm install variant.
        """
        return bool(self._NPM_INSTALL_RE.match(command))

    async def cancel_background_tasks(self) -> None:
        """Cancel all tracked background tasks (e.g. auto-preview).

        Safe to call multiple times; already-finished tasks are skipped.
        """
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
        for task in list(self._background_tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._background_tasks.clear()

    async def _auto_start_preview(
        self,
        sandbox_id: str,
        session_id: str,
        agent_id: str | None = None,
        agent_role: str | None = None,
    ) -> None:
        """Background task: start dev server and emit preview events.

        Called after a successful npm install. The underlying
        ``start_dev_server`` is idempotent (checks if already running),
        so duplicate triggers are safe.

        Args:
            sandbox_id: The sandbox to start the dev server in.
            session_id: Session ID for event emission.
            agent_id: Optional agent ID for event attribution.
            agent_role: Optional agent role for event attribution.
        """
        try:
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.PREVIEW_STARTING,
                    session_id=session_id,
                    agent_id=agent_id,
                    agent_role=agent_role,
                    data={"sandbox_id": sandbox_id},
                )
            )

            url = await self.sandbox_manager.start_dev_server(sandbox_id)

            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.PREVIEW_READY,
                    session_id=session_id,
                    agent_id=agent_id,
                    agent_role=agent_role,
                    data={"url": url, "sandbox_id": sandbox_id},
                )
            )
            logger.info(
                "auto_preview_started",
                sandbox_id=sandbox_id,
                url=url,
            )
        except Exception as exc:
            logger.warning(
                "auto_preview_failed",
                sandbox_id=sandbox_id,
                error=str(exc),
            )
            await self.event_bus.publish(
                AgentEvent(
                    type=EventType.PREVIEW_ERROR,
                    session_id=session_id,
                    agent_id=agent_id,
                    agent_role=agent_role,
                    data={
                        "error": str(exc),
                        "sandbox_id": sandbox_id,
                    },
                )
            )

    async def _execute_search_files(
        self,
        sandbox_id: str,
        args: dict[str, Any],
    ) -> str:
        """Execute search_files tool using grep.

        Args:
            sandbox_id: The sandbox to search in.
            args: Must contain 'pattern', may contain 'path' and 'file_glob'.

        Returns:
            Matching lines or message if no matches.
        """
        pattern = args["pattern"]
        path = args.get("path", ".")
        file_glob = args.get("file_glob", "")

        # Build grep command with proper escaping to prevent injection.
        # Using grep with -r (recursive), -n (line numbers), -H (filenames)
        escaped_pattern = shlex.quote(pattern)
        escaped_path = shlex.quote(path)

        if file_glob:
            escaped_glob = shlex.quote(file_glob)
            cmd = f"grep -rnH --include={escaped_glob} {escaped_pattern} {escaped_path}"
        else:
            cmd = f"grep -rnH {escaped_pattern} {escaped_path}"

        result = await self.sandbox_manager.execute_command(sandbox_id, cmd)

        if result.exit_code == 1:
            # grep returns 1 when no matches found
            return f"No matches found for pattern: {pattern}"

        if result.exit_code != 0:
            return f"Search failed: {result.stderr or result.stdout}"

        return self._truncate_text(
            result.stdout or "No matches found",
            max_chars=MAX_SEARCH_OUTPUT_CHARS,
        )
