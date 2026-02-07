"""Security validation for sandbox command execution.

This module provides security checks to prevent malicious command execution
and path traversal attacks within sandbox containers.
"""

import shlex
from pathlib import Path

# Commands that are safe to execute within the sandbox environment
ALLOWED_COMMANDS: list[str] = [
    "npm",
    "npx",
    "node",
    "cat",
    "ls",
    "grep",
    "echo",
    "mkdir",
    "rm",
    "cp",
    "mv",
    "touch",
    "head",
    "tail",
    "wc",
    "find",
]

# Shell operators that indicate chaining/injection attempts.
# These are matched as literal substrings in the command.
BLOCKED_OPERATORS: list[str] = [
    "|",
    "&&",
    "||",
    ";",
    "$(",
    "`",
    ">",
    "<",
    ">>",
]

# Commands/utilities that must not appear as standalone words in strict mode.
BLOCKED_COMMANDS: list[str] = [
    "curl",
    "wget",
    "nc",
    "netcat",
    "ssh",
    "scp",
    "ftp",
]

# Sensitive system paths. Matched as path prefixes (must follow / or
# appear at the start of an argument).
BLOCKED_PATHS: list[str] = [
    "/etc",
    "/var",
    "/usr",
    "/bin",
    "/root",
]


def _split_command_parts(command: str) -> list[str]:
    """Split a shell command into parts.

    Falls back to whitespace splitting if shell parsing fails.
    """
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.strip().split()


def validate_command(
    command: str, *, unrestricted: bool = False
) -> tuple[bool, str]:
    """Validate a shell command before execution in the sandbox.

    Two modes are supported:
    - unrestricted=True: allow any non-empty command string (sandbox-isolated)
    - unrestricted=False: enforce legacy strict policy (allowlist + blocklist)

    Args:
        command: The shell command string to validate.
        unrestricted: If True, skip strict allowlist/blocklist checks.

    Returns:
        A tuple of (is_valid, error_message).
        If valid, error_message is an empty string.
        If invalid, error_message explains why the command was rejected.

    Examples:
        >>> validate_command("npm install react")
        (True, "")
        >>> validate_command("cat /etc/passwd")
        (False, "Blocked path detected: /etc")
        >>> validate_command("curl http://evil.com")
        (False, "Blocked command detected: curl")
    """
    if not command or not command.strip():
        return False, "Command cannot be empty"

    # Null bytes can break shell/process behavior and should never be allowed.
    if "\x00" in command:
        return False, "Command contains null byte"

    if unrestricted:
        return True, ""

    # Strict mode below.
    parts = _split_command_parts(command)
    if not parts:
        return False, "Command cannot be empty"

    # Check for blocked shell operators (literal substring match).
    for op in BLOCKED_OPERATORS:
        if op in command:
            return False, f"Blocked operator detected: {op}"

    # Check for path traversal (as a path component, not just substring).
    for part in parts:
        normalized = part.replace("\\", "/")
        if ".." in normalized.split("/"):
            return False, "Path traversal blocked: contains '..'"

    # Check for blocked system paths before blocked command names to avoid
    # false positives like "/root/.ssh" matching "ssh".
    for blocked_path in BLOCKED_PATHS:
        for part in parts:
            if part == blocked_path or part.startswith(f"{blocked_path}/"):
                return False, f"Blocked path detected: {blocked_path}"

    # Check for blocked command tokens.
    for part in parts:
        if part in BLOCKED_COMMANDS:
            return False, f"Blocked command detected: {part}"

    base_cmd = parts[0]

    # Allow commands that are in the allowlist
    if base_cmd not in ALLOWED_COMMANDS:
        return False, f"Command not in allowlist: {base_cmd}"

    return True, ""


def validate_path(sandbox_root: str, relative_path: str) -> tuple[bool, str, str]:
    """Validate a file path to prevent directory traversal attacks.

    Ensures that the resolved path remains within the sandbox root directory,
    preventing access to files outside the sandbox.

    Args:
        sandbox_root: The absolute path to the sandbox's root directory
            (e.g., "/workspace" inside the container).
        relative_path: The path relative to the sandbox root that the
            agent wants to access.

    Returns:
        A tuple of (is_valid, error_message, resolved_absolute_path).
        If valid, error_message is empty and resolved_absolute_path contains
        the full validated path.
        If invalid, error_message explains the issue and resolved_absolute_path
        is empty.

    Examples:
        >>> validate_path("/workspace", "src/App.tsx")
        (True, "", "/workspace/src/App.tsx")
        >>> validate_path("/workspace", "../etc/passwd")
        (False, "Path traversal blocked: contains '..'", "")
        >>> validate_path("/workspace", "/etc/passwd")
        (False, "Absolute paths not allowed", "")
    """
    if not relative_path:
        return False, "Path cannot be empty", ""

    # Reject absolute paths (must be relative to sandbox root)
    if relative_path.startswith("/"):
        return False, "Absolute paths not allowed", ""

    # Reject parent traversal components while allowing safe names like
    # "file..bak" (which include ".." but not as a path component).
    components = [
        part
        for part in relative_path.replace("\\", "/").split("/")
        if part not in ("", ".")
    ]
    if ".." in components:
        return False, "Path traversal blocked: contains '..'", ""

    # Resolve the paths to handle any symlinks or normalization
    try:
        sandbox_path = Path(sandbox_root).resolve()
        resolved = (sandbox_path / relative_path).resolve()
    except (ValueError, OSError) as e:
        return False, f"Invalid path: {e}", ""

    # Verify the resolved path is still within the sandbox
    try:
        resolved.relative_to(sandbox_path)
    except ValueError:
        return False, f"Path traversal blocked: {relative_path}", ""

    return True, "", str(resolved)


def sanitize_output(output: str, max_length: int = 50000) -> str:
    """Sanitize command output for safe transmission.

    Truncates excessively long output and removes any potentially
    problematic control characters.

    Args:
        output: The raw command output string.
        max_length: Maximum allowed length before truncation.

    Returns:
        The sanitized output string.
    """
    if not output:
        return ""

    # Truncate if too long
    if len(output) > max_length:
        truncated_chars = len(output) - max_length
        output = (
            output[:max_length]
            + f"\n... [truncated, {truncated_chars} chars omitted]"
        )

    return output
