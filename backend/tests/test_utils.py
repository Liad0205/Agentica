"""Tests for agents/utils.py -- LLM client utilities and helpers.

Covers:
- _convert_message_to_dict: LangChain message object to plain dict
- extract_json_from_response: balanced-brace JSON extraction
- parse_plan_tag: <plan> tag extraction
- count_tokens_estimate: rough token estimation
- format_tool_result_for_llm and format_assistant_message_with_tools
"""

from unittest.mock import MagicMock

from agents.utils import (
    ToolCallData,
    _convert_message_to_dict,
    _convert_messages_to_dicts,
    _enforce_tool_call_pairing,
    count_tokens_estimate,
    extract_json_from_response,
    format_assistant_message_with_tools,
    format_tool_result_for_llm,
    normalize_tool_args,
    parse_plan_tag,
    sliding_window_prune,
)

# =========================================================================
# _convert_message_to_dict
# =========================================================================


class TestConvertMessageToDict:
    """Convert various message formats to plain dicts."""

    def test_dict_passthrough(self) -> None:
        msg = {"role": "user", "content": "hello"}
        result = _convert_message_to_dict(msg)
        assert result is msg

    def test_langchain_human_message(self) -> None:
        msg = MagicMock()
        msg.type = "human"
        msg.content = "hello"
        msg.tool_calls = None
        # Ensure hasattr checks pass for type
        del msg.role
        result = _convert_message_to_dict(msg)
        assert result["role"] == "user"
        assert result["content"] == "hello"

    def test_langchain_ai_message(self) -> None:
        msg = MagicMock()
        msg.type = "ai"
        msg.content = "I'll help you."
        msg.tool_calls = None
        del msg.role
        result = _convert_message_to_dict(msg)
        assert result["role"] == "assistant"

    def test_langchain_system_message(self) -> None:
        msg = MagicMock()
        msg.type = "system"
        msg.content = "You are helpful."
        msg.tool_calls = None
        del msg.role
        result = _convert_message_to_dict(msg)
        assert result["role"] == "system"

    def test_message_with_role_attribute(self) -> None:
        """Object with 'role' attribute instead of 'type'."""
        msg = MagicMock(spec=[])
        msg.role = "assistant"
        msg.content = "Response"
        result = _convert_message_to_dict(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "Response"

    def test_message_missing_role_falls_back(self) -> None:
        """Object with neither 'type' nor 'role' gets 'user' default."""
        msg = MagicMock(spec=[])
        msg.content = "no role"
        result = _convert_message_to_dict(msg)
        assert result["role"] == "user"

    def test_message_with_tool_calls(self) -> None:
        msg = MagicMock()
        msg.type = "ai"
        msg.content = ""
        msg.tool_calls = [
            {"id": "call_1", "name": "write_file", "args": {"path": "a.txt", "content": "hi"}},
        ]
        del msg.role
        del msg.tool_call_id
        result = _convert_message_to_dict(msg)
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "write_file"

    def test_message_with_tool_call_id(self) -> None:
        msg = MagicMock()
        msg.type = "tool"
        msg.content = "result data"
        msg.tool_call_id = "call_123"
        msg.tool_calls = None
        del msg.role
        result = _convert_message_to_dict(msg)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"


class TestConvertMessagesToDicts:
    """Batch conversion of messages."""

    def test_list_of_dicts(self) -> None:
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _convert_messages_to_dicts(msgs)
        assert len(result) == 2
        assert all(isinstance(m, dict) for m in result)

    def test_empty_list(self) -> None:
        assert _convert_messages_to_dicts([]) == []


class TestNormalizeToolArgs:
    """Normalize malformed tool-call argument payloads."""

    def test_dict_passthrough(self) -> None:
        raw = {"path": "src/App.tsx"}
        assert normalize_tool_args(raw) == raw

    def test_json_string_dict(self) -> None:
        result = normalize_tool_args('{"command":"npm run build"}')
        assert result["command"] == "npm run build"

    def test_json_string_non_dict_wrapped(self) -> None:
        result = normalize_tool_args('["a", "b"]')
        assert result == {"value": ["a", "b"]}

    def test_invalid_json_string_wrapped_as_raw(self) -> None:
        result = normalize_tool_args("{bad json")
        assert result == {"raw": "{bad json"}

    def test_none_returns_empty_dict(self) -> None:
        assert normalize_tool_args(None) == {}


# =========================================================================
# extract_json_from_response -- balanced-brace parser
# =========================================================================


class TestExtractJsonFromResponse:
    """JSON extraction from free-form LLM responses."""

    def test_pure_json(self) -> None:
        result = extract_json_from_response('{"key": "value"}')
        assert result is not None
        assert result["key"] == "value"

    def test_json_in_code_fence(self) -> None:
        response = """Here's the result:
```json
{"status": "complete", "files": ["a.tsx", "b.tsx"]}
```
Done!"""
        result = extract_json_from_response(response)
        assert result is not None
        assert result["status"] == "complete"
        assert len(result["files"]) == 2

    def test_json_in_bare_code_fence(self) -> None:
        response = """```
{"action": "write"}
```"""
        result = extract_json_from_response(response)
        assert result is not None
        assert result["action"] == "write"

    def test_json_with_surrounding_text(self) -> None:
        response = (
            'The plan is: {"subtasks": [{"name": "setup"}, {"name": "implement"}]}'
            ' and more text'
        )
        result = extract_json_from_response(response)
        assert result is not None
        assert len(result["subtasks"]) == 2

    def test_nested_json(self) -> None:
        response = '{"outer": {"inner": {"deep": 42}}}'
        result = extract_json_from_response(response)
        assert result is not None
        assert result["outer"]["inner"]["deep"] == 42

    def test_deeply_nested_balanced_braces(self) -> None:
        """The balanced-brace matcher handles arbitrary nesting depth."""
        response = 'Sure! {"a": {"b": {"c": {"d": "found"}}}}'
        result = extract_json_from_response(response)
        assert result is not None
        assert result["a"]["b"]["c"]["d"] == "found"

    def test_no_json(self) -> None:
        result = extract_json_from_response("No JSON here at all.")
        assert result is None

    def test_empty_string(self) -> None:
        result = extract_json_from_response("")
        assert result is None

    def test_malformed_json(self) -> None:
        result = extract_json_from_response("{bad json: without quotes}")
        assert result is None

    def test_first_valid_json_returned(self) -> None:
        """When multiple JSON objects exist, the first valid one is returned."""
        response = 'Ignore {invalid and {"valid": true}'
        result = extract_json_from_response(response)
        assert result is not None
        assert result["valid"] is True

    def test_json_with_array_values(self) -> None:
        response = '{"items": [1, 2, 3], "nested": {"a": [4, 5]}}'
        result = extract_json_from_response(response)
        assert result is not None
        assert result["items"] == [1, 2, 3]

    def test_json_with_string_containing_braces(self) -> None:
        """Braces inside JSON string values should not confuse the parser."""
        response = '{"code": "function() { return {}; }"}'
        result = extract_json_from_response(response)
        # This is valid JSON, the balanced-brace matcher should find it
        assert result is not None
        assert "function" in result["code"]

    def test_json_code_fence_with_extra_text_inside(self) -> None:
        response = """```json
Here is the final payload:
{"selected":"solver_2","reasoning":"best"}
```"""
        result = extract_json_from_response(response)
        assert result is not None
        assert result["selected"] == "solver_2"


# =========================================================================
# parse_plan_tag
# =========================================================================


class TestParsePlanTag:
    """Extract content from <plan> tags."""

    def test_basic_plan(self) -> None:
        response = "<plan>Build the app in three steps.</plan>"
        assert parse_plan_tag(response) == "Build the app in three steps."

    def test_plan_with_surrounding_text(self) -> None:
        response = "Let me think... <plan>Step 1: Setup\nStep 2: Code</plan> Done."
        result = parse_plan_tag(response)
        assert "Step 1" in result
        assert "Step 2" in result

    def test_plan_with_whitespace(self) -> None:
        response = "<plan>\n  Setup project\n  Write code\n</plan>"
        result = parse_plan_tag(response)
        assert "Setup project" in result

    def test_no_plan_tag(self) -> None:
        result = parse_plan_tag("No plan here.")
        assert result == ""

    def test_empty_plan(self) -> None:
        result = parse_plan_tag("<plan></plan>")
        assert result == ""

    def test_multiline_plan(self) -> None:
        response = """<plan>
1. Initialize project with Vite
2. Create components
3. Add styling
4. Test
</plan>"""
        result = parse_plan_tag(response)
        assert "Initialize" in result
        assert "Test" in result


# =========================================================================
# sliding_window_prune
# =========================================================================


class TestSlidingWindowPrune:
    """Context pruning should preserve key diagnostic context."""

    def test_preserves_error_messages_under_pruning(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Build a dashboard."},
            {"role": "assistant", "content": "Planning"},
            {"role": "tool", "content": "Command output: exit code: 0"},
            {"role": "assistant", "content": "Continuing"},
            {"role": "tool", "content": "Build failed: Cannot find module 'zod'"},
            {"role": "assistant", "content": "Trying fix"},
            {"role": "tool", "content": "Wrote src/App.tsx"},
            {"role": "assistant", "content": "Almost done"},
            {"role": "user", "content": "Please continue."},
        ]

        pruned = sliding_window_prune(messages, max_messages=6, max_tokens=10_000)
        contents = [str(msg.get("content", "")) for msg in pruned if isinstance(msg, dict)]

        assert any("Build failed" in content for content in contents)
        assert any("pruned for context limits" in content for content in contents)
        assert "Please continue." in contents[-1]

    def test_keeps_prefix_when_pruning(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            *[
                {"role": "assistant", "content": f"msg {i}"}
                for i in range(15)
            ],
        ]

        pruned = sliding_window_prune(messages, max_messages=5, max_tokens=10_000)
        assert isinstance(pruned[0], dict)
        assert isinstance(pruned[1], dict)
        assert pruned[0]["content"] == "system"
        assert pruned[1]["content"] == "task"

    def test_strips_tool_calls_when_matching_result_is_missing(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
        ]

        selected_indexes = {0, 1, 2}
        normalized = [_convert_message_to_dict(msg) for msg in messages]
        _enforce_tool_call_pairing(selected_indexes, messages, normalized)

        assert selected_indexes == {0, 1, 2}
        assert "tool_calls" not in normalized[2]


# =========================================================================
# count_tokens_estimate
# =========================================================================


class TestCountTokensEstimate:
    """Rough token count estimation."""

    def test_empty_string(self) -> None:
        assert count_tokens_estimate("") == 0

    def test_short_string(self) -> None:
        # "hello" = 5 chars, 5 // 4 = 1
        assert count_tokens_estimate("hello") == 1

    def test_longer_string(self) -> None:
        # 100 chars -> ~25 tokens
        text = "a" * 100
        assert count_tokens_estimate(text) == 25


# =========================================================================
# format_tool_result_for_llm
# =========================================================================


class TestFormatToolResult:
    """Format tool results for LLM message history."""

    def test_basic_result(self) -> None:
        result = format_tool_result_for_llm("call_123", "File written successfully")
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert result["content"] == "File written successfully"


# =========================================================================
# format_assistant_message_with_tools
# =========================================================================


class TestFormatAssistantMessage:
    """Format assistant messages that include tool calls."""

    def test_with_tool_calls(self) -> None:
        tc = ToolCallData(id="tc_1", name="write_file", args={"path": "a.txt", "content": "hi"})
        msg = format_assistant_message_with_tools("I'll write the file.", [tc])
        assert msg["role"] == "assistant"
        assert msg["content"] == "I'll write the file."
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "write_file"

    def test_without_tool_calls(self) -> None:
        msg = format_assistant_message_with_tools("Just text.", [])
        assert msg["role"] == "assistant"
        assert "tool_calls" not in msg

    def test_multiple_tool_calls(self) -> None:
        tcs = [
            ToolCallData(id="tc_1", name="write_file", args={"path": "a.txt", "content": "1"}),
            ToolCallData(id="tc_2", name="read_file", args={"path": "b.txt"}),
        ]
        msg = format_assistant_message_with_tools("", tcs)
        assert len(msg["tool_calls"]) == 2
