"""Tests for file integrity: empty content guards, merge validation, and collection edge cases.

Covers the file write lifecycle from tool argument validation through sandbox
collection and merge, verifying that empty/corrupt content is properly handled
at each stage.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from agents.tools import ToolExecutor
from events.bus import EventBus
from sandbox.docker_sandbox import CommandResult

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


def _make_sandbox_manager() -> AsyncMock:
    """Create a mock sandbox manager with standard return values."""
    mgr = AsyncMock()
    mgr.create_sandbox = AsyncMock()
    mgr.destroy_sandbox = AsyncMock()
    mgr.write_file = AsyncMock()
    mgr.read_file = AsyncMock(return_value="file content")
    mgr.execute_command = AsyncMock(
        return_value=CommandResult(stdout="", stderr="", exit_code=0)
    )
    return mgr


# =========================================================================
# Tool Argument Validation - Empty Content
# =========================================================================


class TestWriteFileEmptyContent:
    """Verify write_file tool handles empty content correctly."""

    async def test_write_file_with_empty_string_content_rejected(self) -> None:
        """write_file with content='' should be rejected as missing required arg."""
        executor = _make_executor()
        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="write_file",
            args={"path": "src/App.tsx", "content": ""},
            session_id="sess_1",
        )
        assert result.success is False
        assert "missing required" in result.content.lower()

    async def test_write_file_with_whitespace_only_content_succeeds(self) -> None:
        """write_file with whitespace-only content should succeed (whitespace is valid content)."""
        sandbox_mgr = AsyncMock()
        sandbox_mgr.write_file = AsyncMock()
        executor = _make_executor(sandbox_manager=sandbox_mgr)

        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="write_file",
            args={"path": "src/empty.ts", "content": "  \n  "},
            session_id="sess_1",
        )
        assert result.success is True

    async def test_write_file_with_none_content_rejected(self) -> None:
        """write_file with content=None should be rejected."""
        executor = _make_executor()
        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="write_file",
            args={"path": "src/App.tsx", "content": None},
            session_id="sess_1",
        )
        assert result.success is False

    async def test_write_file_missing_content_field_rejected(self) -> None:
        """write_file without content field should be rejected."""
        executor = _make_executor()
        result = await executor.execute(
            sandbox_id="sb_1",
            tool_name="write_file",
            args={"path": "src/App.tsx"},
            session_id="sess_1",
        )
        assert result.success is False
        assert "missing required" in result.content.lower()


# =========================================================================
# File Collection - Empty Content Filtering
# =========================================================================


class TestCollectSandboxFilesEmptyGuard:
    """Verify _collect_sandbox_files skips empty file content."""

    async def test_hypothesis_collect_skips_empty_files(self) -> None:
        """Files with empty content should be excluded from collection."""
        from agents.hypothesis_graph import HypothesisGraph

        sandbox_mgr = _make_sandbox_manager()
        bus = EventBus()

        # find returns two files
        sandbox_mgr.execute_command = AsyncMock(
            return_value=CommandResult(
                stdout="src/App.tsx\nsrc/Empty.tsx", stderr="", exit_code=0
            )
        )

        # First file has content, second is empty
        sandbox_mgr.read_file = AsyncMock(
            side_effect=["export default function App() {}", ""]
        )

        graph = HypothesisGraph(sandbox_mgr, bus)
        files = await graph._collect_sandbox_files("sandbox_1")

        assert "src/App.tsx" in files
        assert "src/Empty.tsx" not in files
        assert len(files) == 1

    async def test_decomposition_collect_skips_empty_files(self) -> None:
        """Files with empty content should be excluded from decomposition collection."""
        from agents.decomposition_graph import DecompositionGraph

        sandbox_mgr = _make_sandbox_manager()
        bus = EventBus()

        sandbox_mgr.execute_command = AsyncMock(
            return_value=CommandResult(
                stdout="src/App.tsx\nsrc/Empty.tsx", stderr="", exit_code=0
            )
        )

        sandbox_mgr.read_file = AsyncMock(
            side_effect=["export default function App() {}", ""]
        )

        graph = DecompositionGraph(sandbox_mgr, bus)
        files = await graph._collect_sandbox_files("sandbox_1")

        assert "src/App.tsx" in files
        assert "src/Empty.tsx" not in files
        # Only 1 file with actual content should be collected
        assert len(files) == 1

    async def test_hypothesis_collect_skips_empty_config_files(self) -> None:
        """Root config files with empty content should not be collected."""
        from agents.hypothesis_graph import HypothesisGraph

        sandbox_mgr = _make_sandbox_manager()
        bus = EventBus()

        # No src/ files found
        sandbox_mgr.execute_command = AsyncMock(
            return_value=CommandResult(stdout="", stderr="", exit_code=1)
        )

        # Config file reads: package.json has content, rest are empty or not found
        async def mock_read_file(sandbox_id: str, path: str) -> str:
            if path == "package.json":
                return '{"name": "test"}'
            if path == "index.html":
                return ""  # Empty content
            raise FileNotFoundError(f"Not found: {path}")

        sandbox_mgr.read_file = AsyncMock(side_effect=mock_read_file)

        graph = HypothesisGraph(sandbox_mgr, bus)
        files = await graph._collect_sandbox_files("sandbox_1")

        assert "package.json" in files
        assert "index.html" not in files  # Empty content should be skipped


# =========================================================================
# Merge - Empty Content Protection
# =========================================================================


class TestMergeEmptyContentGuard:
    """Verify merge operations skip empty file content."""

    async def test_merge_skips_empty_files_from_subtask(self) -> None:
        """_merge_successful_results_into_integration should skip files with empty content."""
        from agents.decomposition_graph import DecompositionGraph, SubtaskResult

        sandbox_mgr = _make_sandbox_manager()
        bus = EventBus()
        graph = DecompositionGraph(sandbox_mgr, bus)

        result = SubtaskResult(
            subtask_id="st_1",
            agent_id="agent_st_1",
            sandbox_id="sandbox_st_1",
            files_produced={
                "src/App.tsx": "export default function App() {}",
                "src/Empty.tsx": "",  # Empty content - should be skipped
            },
            dependencies={},
            dev_dependencies={},
            type_additions="",
            edited_files=["src/App.tsx"],
            build_success=True,
            build_output="",
            lint_output="",
            summary="Done",
            status="complete",
            retries=0,
        )

        merge_result = await graph._merge_successful_results_into_integration(
            successful_results=[result],
            integration_sandbox_id="sandbox_integration",
            session_id="sess_test",
            aggregator_id="agg_1",
            shared_types="",
        )

        # Only non-empty file should be merged
        assert "src/App.tsx" in merge_result["merged_files"]
        assert "src/Empty.tsx" not in merge_result["merged_files"]


# =========================================================================
# Synthesis - Empty Collection Guard
# =========================================================================


class TestSynthesizeEmptyFileGuard:
    """Verify synthesis does not replace files with an empty collection result."""

    async def test_synthesize_preserves_original_files_when_collection_empty(
        self,
    ) -> None:
        """If _collect_sandbox_files returns empty, original files should be preserved."""
        from agents.hypothesis_graph import HypothesisGraph, HypothesisResult

        sandbox_mgr = _make_sandbox_manager()
        bus = EventBus()

        # Mock LLM client
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "<status>TASK_COMPLETE</status>"
        mock_response.tool_calls = []
        mock_llm.call = AsyncMock(return_value=mock_response)
        mock_llm.metrics_collector = None

        graph = HypothesisGraph(sandbox_mgr, bus, llm_client=mock_llm)

        original_files = {"src/App.tsx": "original content"}
        winning_result = HypothesisResult(
            agent_id="solver_1",
            persona="clarity",
            sandbox_id="sandbox_solver1",
            files=dict(original_files),
            edited_files=["src/App.tsx"],
            build_success=True,
            build_output="",
            lint_errors=0,
            iterations_used=5,
            agent_summary="Done",
        )

        # Make _collect_sandbox_files return empty (simulating sandbox destruction)
        sandbox_mgr.execute_command = AsyncMock(
            return_value=CommandResult(stdout="", stderr="", exit_code=1)
        )
        # Config file reads all fail
        sandbox_mgr.read_file = AsyncMock(side_effect=FileNotFoundError("gone"))

        state = {
            "session_id": "sess_test",
            "hypothesis_results": [winning_result],
            "selected_index": 0,
            "evaluation_reasoning": "Could improve styling",
            "task": "Build a todo app",
        }

        await graph._synthesize(state)

        # Original files should be preserved since collection returned empty
        assert winning_result["files"] == original_files


# =========================================================================
# Merge Conflict - LLM Output Validation
# =========================================================================


class TestMergeConflictLLMValidation:
    """Verify _merge_conflicting_files validates LLM merge output."""

    async def test_merge_conflict_falls_back_on_empty_llm_output(self) -> None:
        """If LLM returns empty content, fallback to last-writer-wins."""
        from agents.decomposition_graph import DecompositionGraph

        sandbox_mgr = _make_sandbox_manager()
        bus = EventBus()

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = ""  # Empty response
        mock_llm.call = AsyncMock(return_value=mock_response)
        mock_llm.metrics_collector = None

        graph = DecompositionGraph(sandbox_mgr, bus, llm_client=mock_llm)

        sources = [
            ("agent_1", "export function A() {}"),
            ("agent_2", "export function B() {}"),
        ]

        result = await graph._merge_conflicting_files(
            "src/utils.ts", sources, "sess_test", "agg_1"
        )

        # Should fall back to last writer
        assert result == "export function B() {}"

    async def test_merge_conflict_strips_markdown_fencing(self) -> None:
        """LLM output wrapped in markdown fences should have fences stripped."""
        from agents.decomposition_graph import DecompositionGraph

        sandbox_mgr = _make_sandbox_manager()
        bus = EventBus()

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = (
            "```typescript\n"
            "export function Merged() { return 'combined'; }\n"
            "```"
        )
        mock_llm.call = AsyncMock(return_value=mock_response)
        mock_llm.metrics_collector = None

        graph = DecompositionGraph(sandbox_mgr, bus, llm_client=mock_llm)

        sources = [
            ("agent_1", "export function A() {}"),
            ("agent_2", "export function B() {}"),
        ]

        result = await graph._merge_conflicting_files(
            "src/utils.ts", sources, "sess_test", "agg_1"
        )

        # Should have fences stripped
        assert "```" not in result
        assert "Merged" in result

    async def test_merge_conflict_accepts_short_non_empty_llm_output(self) -> None:
        """Non-empty LLM merge content should be accepted even if short."""
        from agents.decomposition_graph import DecompositionGraph

        sandbox_mgr = _make_sandbox_manager()
        bus = EventBus()

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "// merged"  # Too short (< 10 chars after strip)
        mock_llm.call = AsyncMock(return_value=mock_response)
        mock_llm.metrics_collector = None

        graph = DecompositionGraph(sandbox_mgr, bus, llm_client=mock_llm)

        sources = [
            ("agent_1", "export function A() { return 'hello'; }"),
            ("agent_2", "export function B() { return 'world'; }"),
        ]

        result = await graph._merge_conflicting_files(
            "src/utils.ts", sources, "sess_test", "agg_1"
        )

        assert result == "// merged"


# =========================================================================
# copy_files_between - Empty Content Logging
# =========================================================================


class TestCopyFilesBetweenEmptyContent:
    """Verify copy_files_between handles empty source content."""

    async def test_copy_empty_file_still_writes_but_logs_warning(self) -> None:
        """Empty files should still be copied but a warning should be logged."""
        from sandbox.docker_sandbox import SandboxManager

        mgr = SandboxManager.__new__(SandboxManager)
        mgr._sandboxes = {
            "src_sb": MagicMock(workspace_path="/workspace"),
            "tgt_sb": MagicMock(workspace_path="/workspace"),
        }

        mgr.read_file = AsyncMock(return_value="")
        mgr.write_file = AsyncMock()

        with (
            patch(
                "sandbox.docker_sandbox.validate_path",
                return_value=(True, None, "/workspace/test.ts"),
            ),
            patch.object(
                mgr,
                "_get_sandbox",
                side_effect=lambda sid: mgr._sandboxes[sid],
            ),
        ):
            await mgr.copy_files_between("src_sb", "tgt_sb", ["test.ts"])

        # Should still write the empty file
        mgr.write_file.assert_awaited_once_with("tgt_sb", "test.ts", "")

    async def test_copy_nonexistent_file_skipped(self) -> None:
        """FileNotFoundError during copy should be silently skipped."""
        from sandbox.docker_sandbox import SandboxManager

        mgr = SandboxManager.__new__(SandboxManager)
        mgr._sandboxes = {
            "src_sb": MagicMock(workspace_path="/workspace"),
            "tgt_sb": MagicMock(workspace_path="/workspace"),
        }

        mgr.read_file = AsyncMock(side_effect=FileNotFoundError("not found"))
        mgr.write_file = AsyncMock()

        with (
            patch(
                "sandbox.docker_sandbox.validate_path",
                return_value=(True, None, "/workspace/test.ts"),
            ),
            patch.object(
                mgr,
                "_get_sandbox",
                side_effect=lambda sid: mgr._sandboxes[sid],
            ),
        ):
            await mgr.copy_files_between("src_sb", "tgt_sb", ["test.ts"])

        # write_file should NOT be called for missing files
        mgr.write_file.assert_not_awaited()
