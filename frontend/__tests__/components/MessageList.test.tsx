/**
 * Tests for the MessageList component (components/messages/MessageList.tsx).
 * Covers event grouping, system event handling, filtering, and empty state.
 */

import * as React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MessageList } from "@/components/messages/MessageList";
import type { AgentEvent } from "@/lib/types";

/**
 * Helper to create a minimal AgentEvent for testing.
 */
function makeEvent(
  type: AgentEvent["type"],
  data: Record<string, unknown> = {},
  overrides: Partial<AgentEvent> = {}
): AgentEvent {
  return {
    type,
    timestamp: Date.now() / 1000,
    session_id: "test-session",
    data,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------
describe("MessageList empty state", () => {
  it("shows 'No messages yet' when events array is empty", () => {
    render(<MessageList events={[]} />);

    expect(screen.getByText("No messages yet")).toBeInTheDocument();
  });

  it("shows empty state when all events are filtered out", () => {
    const events = [
      makeEvent("agent_thinking", { content: "hello" }, { agent_id: "agent-1" }),
    ];

    render(<MessageList events={events} filterAgentId="agent-999" />);

    // agent_thinking is NOT a system event, so it gets filtered out when
    // filtering by a different agent ID
    expect(screen.getByText("No messages yet")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Rendering events
// ---------------------------------------------------------------------------
describe("MessageList rendering", () => {
  it("renders session_started as a system message", () => {
    const events = [
      makeEvent("session_started", { mode: "react" }),
    ];

    render(<MessageList events={events} />);

    expect(screen.getByText(/Session started in react mode/)).toBeInTheDocument();
  });

  it("renders session_complete as a system message", () => {
    const events = [
      makeEvent("session_complete"),
    ];

    render(<MessageList events={events} />);

    expect(screen.getByText("Session completed successfully")).toBeInTheDocument();
  });

  it("renders session_error with error message", () => {
    const events = [
      makeEvent("session_error", { error: "Sandbox crashed" }),
    ];

    render(<MessageList events={events} />);

    expect(screen.getByText("Sandbox crashed")).toBeInTheDocument();
  });

  it("renders session_cancelled as a warning", () => {
    const events = [
      makeEvent("session_cancelled"),
    ];

    render(<MessageList events={events} />);

    expect(screen.getByText("Session was cancelled")).toBeInTheDocument();
  });

  it("renders agent_thinking with agent role and content", () => {
    const events = [
      makeEvent(
        "agent_thinking",
        { content: "Analyzing the requirements..." },
        { agent_id: "a1", agent_role: "Code Agent" }
      ),
    ];

    render(<MessageList events={events} />);

    expect(screen.getByText("Code Agent")).toBeInTheDocument();
    expect(screen.getByText("Analyzing the requirements...")).toBeInTheDocument();
  });

  it("renders agent_message with content", () => {
    const events = [
      makeEvent(
        "agent_message",
        { content: "I have completed the task" },
        { agent_id: "a1", agent_role: "Agent" }
      ),
    ];

    render(<MessageList events={events} />);

    expect(screen.getByText("I have completed the task")).toBeInTheDocument();
  });

  it("renders agent_complete", () => {
    const events = [
      makeEvent(
        "agent_complete",
        { summary: "Task completed successfully" },
        { agent_id: "a1", agent_role: "Agent" }
      ),
    ];

    render(<MessageList events={events} />);

    expect(screen.getByText("Task completed successfully")).toBeInTheDocument();
  });

  it("renders agent_error", () => {
    const events = [
      makeEvent(
        "agent_error",
        { error: "Build failed" },
        { agent_id: "a1", agent_role: "Agent" }
      ),
    ];

    render(<MessageList events={events} />);

    expect(screen.getByText("Build failed")).toBeInTheDocument();
  });

  it("returns null for non-display events", () => {
    const events = [
      makeEvent("file_changed", { path: "src/index.ts" }),
      makeEvent("command_output", { output: "hello" }),
      makeEvent("preview_ready", { url: "http://localhost:3000" }),
    ];

    render(<MessageList events={events} />);

    // These events should not render any messages -- but they still
    // count as events so the empty state isn't shown. However, since
    // they all render null in EventMessage, the actual rendered content
    // will just be the scroll area with empty groups.
    // The important thing is that the list doesn't crash.
    expect(screen.queryByText("No messages yet")).not.toBeInTheDocument();
  });

  it("hides resolved tool call pending rows once a matching result arrives", () => {
    const events = [
      makeEvent(
        "agent_tool_call",
        {
          tool: "write_file",
          args: { path: "src/App.tsx" },
          tool_call_id: "tc_1",
        },
        { agent_id: "a1", agent_role: "Agent" }
      ),
      makeEvent(
        "agent_tool_result",
        {
          tool: "write_file",
          success: true,
          result: "ok",
          tool_call_id: "tc_1",
        },
        { agent_id: "a1", agent_role: "Agent" }
      ),
    ];

    render(<MessageList events={events} />);

    // Without reconciliation, both call and result rows render badges.
    // We only expect the finalized result row.
    expect(screen.getAllByText("Write File")).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Event grouping
// ---------------------------------------------------------------------------
describe("MessageList event grouping", () => {
  it("groups consecutive events from the same agent", () => {
    const events = [
      makeEvent(
        "agent_thinking",
        { content: "Step 1" },
        { agent_id: "a1", agent_role: "Agent" }
      ),
      makeEvent(
        "agent_thinking",
        { content: "Step 2" },
        { agent_id: "a1", agent_role: "Agent" }
      ),
    ];

    render(<MessageList events={events} />);

    // Both messages from agent a1 should be rendered
    expect(screen.getByText("Step 1")).toBeInTheDocument();
    expect(screen.getByText("Step 2")).toBeInTheDocument();
  });

  it("separates events from different agents into different groups", () => {
    const events = [
      makeEvent(
        "agent_thinking",
        { content: "Agent 1 thinking" },
        { agent_id: "a1", agent_role: "Solver 1" }
      ),
      makeEvent(
        "agent_thinking",
        { content: "Agent 2 thinking" },
        { agent_id: "a2", agent_role: "Solver 2" }
      ),
    ];

    render(<MessageList events={events} />);

    expect(screen.getByText("Solver 1")).toBeInTheDocument();
    expect(screen.getByText("Solver 2")).toBeInTheDocument();
    expect(screen.getByText("Agent 1 thinking")).toBeInTheDocument();
    expect(screen.getByText("Agent 2 thinking")).toBeInTheDocument();
  });

  it("system events always start a new group", () => {
    const events = [
      makeEvent(
        "agent_thinking",
        { content: "thinking..." },
        { agent_id: "a1", agent_role: "Agent" }
      ),
      makeEvent("session_complete"),
      makeEvent(
        "agent_thinking",
        { content: "more thinking" },
        { agent_id: "a1", agent_role: "Agent" }
      ),
    ];

    render(<MessageList events={events} />);

    // All three messages should be rendered (system events break groups)
    expect(screen.getByText("thinking...")).toBeInTheDocument();
    expect(screen.getByText("Session completed successfully")).toBeInTheDocument();
    expect(screen.getByText("more thinking")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Filtering by agent ID
// ---------------------------------------------------------------------------
describe("MessageList filtering", () => {
  it("filters events to only show the specified agent", () => {
    const events = [
      makeEvent(
        "agent_thinking",
        { content: "Agent 1 output" },
        { agent_id: "a1", agent_role: "Agent 1" }
      ),
      makeEvent(
        "agent_thinking",
        { content: "Agent 2 output" },
        { agent_id: "a2", agent_role: "Agent 2" }
      ),
    ];

    render(<MessageList events={events} filterAgentId="a1" />);

    expect(screen.getByText("Agent 1 output")).toBeInTheDocument();
    expect(screen.queryByText("Agent 2 output")).not.toBeInTheDocument();
  });

  it("always shows system events when filtering", () => {
    const events = [
      makeEvent("session_started", { mode: "react" }),
      makeEvent(
        "agent_thinking",
        { content: "agent 1" },
        { agent_id: "a1", agent_role: "Agent" }
      ),
      makeEvent(
        "agent_thinking",
        { content: "agent 2" },
        { agent_id: "a2", agent_role: "Agent" }
      ),
      makeEvent("session_complete"),
    ];

    render(<MessageList events={events} filterAgentId="a1" />);

    // System events should still be visible
    expect(screen.getByText(/Session started/)).toBeInTheDocument();
    expect(screen.getByText("Session completed successfully")).toBeInTheDocument();

    // Agent 1 visible, Agent 2 filtered out
    expect(screen.getByText("agent 1")).toBeInTheDocument();
    expect(screen.queryByText("agent 2")).not.toBeInTheDocument();
  });

  it("shows all events when filterAgentId is null", () => {
    const events = [
      makeEvent(
        "agent_thinking",
        { content: "agent 1" },
        { agent_id: "a1", agent_role: "Agent" }
      ),
      makeEvent(
        "agent_thinking",
        { content: "agent 2" },
        { agent_id: "a2", agent_role: "Agent" }
      ),
    ];

    render(<MessageList events={events} filterAgentId={null} />);

    expect(screen.getByText("agent 1")).toBeInTheDocument();
    expect(screen.getByText("agent 2")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Multiple event types in sequence
// ---------------------------------------------------------------------------
describe("MessageList mixed event types", () => {
  it("handles a realistic sequence of events", () => {
    const events: AgentEvent[] = [
      makeEvent("session_started", { mode: "react" }),
      makeEvent(
        "agent_spawned",
        { sandbox_id: "sb-1" },
        { agent_id: "a1", agent_role: "React Agent" }
      ),
      makeEvent(
        "agent_thinking",
        { content: "Let me analyze the task..." },
        { agent_id: "a1", agent_role: "React Agent" }
      ),
      makeEvent(
        "agent_tool_call",
        { tool: "write_file", args: { path: "index.ts" } },
        { agent_id: "a1", agent_role: "React Agent" }
      ),
      makeEvent(
        "agent_complete",
        { summary: "All done!" },
        { agent_id: "a1", agent_role: "React Agent" }
      ),
      makeEvent("session_complete"),
    ];

    render(<MessageList events={events} />);

    // Session system messages
    expect(screen.getByText(/Session started in react mode/)).toBeInTheDocument();
    expect(screen.getByText("Session completed successfully")).toBeInTheDocument();

    // Agent messages (agent_spawned returns null, which is fine)
    expect(screen.getByText("Let me analyze the task...")).toBeInTheDocument();
    expect(screen.getByText("All done!")).toBeInTheDocument();
  });
});
