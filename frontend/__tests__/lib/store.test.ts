/**
 * Tests for the Zustand store (lib/store.ts).
 * Covers processEvent dispatcher, state transitions, and bounded array caps.
 */

import { describe, it, expect, beforeEach } from "vitest";
import { useStore } from "@/lib/store";
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

/**
 * Reset the store to its initial state before every test.
 */
beforeEach(() => {
  useStore.getState().resetSession();
});

// ---------------------------------------------------------------------------
// processEvent dispatcher
// ---------------------------------------------------------------------------
describe("processEvent", () => {
  it("session_started sets sessionStatus to running", () => {
    const event = makeEvent("session_started", { sandbox_id: "sandbox-1" });

    useStore.getState().processEvent(event);

    expect(useStore.getState().sessionStatus).toBe("running");
  });

  it("session_started adds sandbox if sandbox_id is provided", () => {
    const event = makeEvent("session_started", { sandbox_id: "sandbox-1" });

    useStore.getState().processEvent(event);

    const state = useStore.getState();
    expect(state.sandboxes).toHaveLength(1);
    expect(state.sandboxes[0]?.id).toBe("sandbox-1");
    expect(state.selectedSandboxId).toBe("sandbox-1");
  });

  it("session_started does not replace existing selectedSandboxId", () => {
    useStore.setState({ selectedSandboxId: "existing-sandbox" });

    const event = makeEvent("session_started", { sandbox_id: "sandbox-2" });
    useStore.getState().processEvent(event);

    expect(useStore.getState().selectedSandboxId).toBe("existing-sandbox");
  });

  it("session_complete sets sessionStatus to complete", () => {
    useStore.getState().processEvent(makeEvent("session_complete"));

    expect(useStore.getState().sessionStatus).toBe("complete");
    expect(useStore.getState().previewLoading).toBe(false);
  });

  it("session_complete settles active agents to complete", () => {
    useStore.getState().processEvent(
      makeEvent(
        "agent_spawned",
        { sandbox_id: "sb-1" },
        { agent_id: "a1", agent_role: "coder" }
      )
    );
    useStore.getState().processEvent(
      makeEvent("agent_tool_call", { tool: "write_file" }, { agent_id: "a1" })
    );
    expect(useStore.getState().agents.get("a1")?.status).toBe("executing");

    useStore.getState().processEvent(makeEvent("session_complete"));
    expect(useStore.getState().agents.get("a1")?.status).toBe("complete");
  });

  it("session_error sets sessionStatus to error", () => {
    useStore.getState().processEvent(makeEvent("session_error"));

    expect(useStore.getState().sessionStatus).toBe("error");
  });

  it("session_error settles active agents to failed", () => {
    useStore.getState().processEvent(
      makeEvent(
        "agent_spawned",
        { sandbox_id: "sb-1" },
        { agent_id: "a1", agent_role: "coder" }
      )
    );
    useStore.getState().processEvent(
      makeEvent("agent_tool_call", { tool: "write_file" }, { agent_id: "a1" })
    );
    expect(useStore.getState().agents.get("a1")?.status).toBe("executing");

    useStore.getState().processEvent(makeEvent("session_error"));
    expect(useStore.getState().agents.get("a1")?.status).toBe("failed");
  });

  it("session_cancelled sets sessionStatus to cancelled", () => {
    useStore.getState().processEvent(makeEvent("session_cancelled"));

    expect(useStore.getState().sessionStatus).toBe("cancelled");
  });

  it("agent_spawned adds agent to the agents map", () => {
    const event = makeEvent(
      "agent_spawned",
      { sandbox_id: "sb-1", max_iterations: 15 },
      { agent_id: "agent-1", agent_role: "coder" }
    );

    useStore.getState().processEvent(event);

    const state = useStore.getState();
    expect(state.agents.size).toBe(1);
    const agent = state.agents.get("agent-1");
    expect(agent).toBeDefined();
    expect(agent?.role).toBe("coder");
    expect(agent?.status).toBe("idle");
    expect(agent?.sandboxId).toBe("sb-1");
    expect(agent?.maxIterations).toBe(15);
  });

  it("agent_spawned in react mode sets reactAgentInfo", () => {
    useStore.setState({ mode: "react" });

    const event = makeEvent(
      "agent_spawned",
      { sandbox_id: "sb-1" },
      { agent_id: "agent-react", agent_role: "react-agent" }
    );

    useStore.getState().processEvent(event);

    const state = useStore.getState();
    expect(state.reactAgentInfo).not.toBeNull();
    expect(state.reactAgentInfo?.id).toBe("agent-react");
  });

  it("agent_thinking updates agent status to reasoning", () => {
    // First spawn the agent
    useStore.getState().processEvent(
      makeEvent("agent_spawned", { sandbox_id: "sb-1" }, { agent_id: "a1", agent_role: "coder" })
    );

    // Then send thinking event
    useStore.getState().processEvent(
      makeEvent("agent_thinking", { content: "analyzing..." }, { agent_id: "a1" })
    );

    const agent = useStore.getState().agents.get("a1");
    expect(agent?.status).toBe("reasoning");
    expect(agent?.iterations).toBe(1);
  });

  it("agent_tool_call updates agent status to executing", () => {
    useStore.getState().processEvent(
      makeEvent("agent_spawned", { sandbox_id: "sb-1" }, { agent_id: "a1", agent_role: "coder" })
    );

    useStore.getState().processEvent(
      makeEvent("agent_tool_call", { tool: "write_file" }, { agent_id: "a1" })
    );

    expect(useStore.getState().agents.get("a1")?.status).toBe("executing");
  });

  it("agent_complete updates agent status to complete", () => {
    useStore.getState().processEvent(
      makeEvent("agent_spawned", { sandbox_id: "sb-1" }, { agent_id: "a1", agent_role: "coder" })
    );

    useStore.getState().processEvent(
      makeEvent("agent_complete", { summary: "done" }, { agent_id: "a1" })
    );

    expect(useStore.getState().agents.get("a1")?.status).toBe("complete");
  });

  it("agent_error updates agent status to failed", () => {
    useStore.getState().processEvent(
      makeEvent("agent_spawned", { sandbox_id: "sb-1" }, { agent_id: "a1", agent_role: "coder" })
    );

    useStore.getState().processEvent(
      makeEvent("agent_error", { error: "crash" }, { agent_id: "a1" })
    );

    expect(useStore.getState().agents.get("a1")?.status).toBe("failed");
  });

  it("agent_timeout updates agent status to timeout", () => {
    useStore.getState().processEvent(
      makeEvent("agent_spawned", { sandbox_id: "sb-1" }, { agent_id: "a1", agent_role: "coder" })
    );

    useStore.getState().processEvent(
      makeEvent("agent_timeout", {}, { agent_id: "a1" })
    );

    expect(useStore.getState().agents.get("a1")?.status).toBe("timeout");
  });

  it("file_changed adds file to fileTree and sets activeFilePath", () => {
    const event = makeEvent("file_changed", { path: "src/index.ts" });

    useStore.getState().processEvent(event);

    const state = useStore.getState();
    expect(state.activeFilePath).toBe("src/index.ts");
    expect(state.fileTree.length).toBeGreaterThan(0);
  });

  it("file_changed creates nested directory structure", () => {
    const event = makeEvent("file_changed", { path: "src/components/Button.tsx" });

    useStore.getState().processEvent(event);

    const state = useStore.getState();
    // Should have src directory at root
    const srcDir = state.fileTree.find((n) => n.name === "src");
    expect(srcDir).toBeDefined();
    expect(srcDir?.type).toBe("directory");
    // Should have components directory inside src
    const componentsDir = srcDir?.children?.find((n) => n.name === "components");
    expect(componentsDir).toBeDefined();
    expect(componentsDir?.type).toBe("directory");
    // Should have file inside components
    const file = componentsDir?.children?.find((n) => n.name === "Button.tsx");
    expect(file).toBeDefined();
    expect(file?.type).toBe("file");
  });

  it("file_deleted removes file from fileTree", () => {
    // Add a file first
    useStore.getState().processEvent(makeEvent("file_changed", { path: "src/temp.ts" }));
    expect(useStore.getState().fileTree.length).toBeGreaterThan(0);

    // Delete it
    useStore.getState().processEvent(makeEvent("file_deleted", { path: "src/temp.ts" }));

    const srcDir = useStore.getState().fileTree.find((n) => n.name === "src");
    const tempFile = srcDir?.children?.find((n) => n.name === "temp.ts");
    expect(tempFile).toBeUndefined();
  });

  it("file_deleted clears activeFilePath if the deleted file was active", () => {
    useStore.getState().processEvent(makeEvent("file_changed", { path: "src/active.ts" }));
    expect(useStore.getState().activeFilePath).toBe("src/active.ts");

    useStore.getState().processEvent(makeEvent("file_deleted", { path: "src/active.ts" }));

    expect(useStore.getState().activeFilePath).toBeNull();
  });

  it("command_output appends to terminalOutput", () => {
    useStore.getState().processEvent(
      makeEvent("command_output", { output: "hello world" })
    );

    expect(useStore.getState().terminalOutputs.all).toContain("hello world");
  });

  it("command_started prepends $ to command in terminal", () => {
    useStore.getState().processEvent(
      makeEvent("command_started", { command: "npm install" })
    );

    expect(useStore.getState().terminalOutputs.all).toContain("$ npm install\n");
  });

  it("command_complete appends exit code to terminal", () => {
    useStore.getState().processEvent(
      makeEvent("command_complete", { exit_code: 0 })
    );

    const output = useStore.getState().terminalOutputs.all ?? [];
    expect(output.some((line) => line.includes("[Exit code: 0]"))).toBe(true);
  });

  // Preview URL validation tests (security fix regression tests)
  describe("preview_ready URL validation", () => {
    it("accepts http URLs", () => {
      useStore.getState().processEvent(
        makeEvent("preview_ready", { url: "http://localhost:5173" })
      );

      expect(useStore.getState().previewUrl).toBe("http://localhost:5173/");
      expect(useStore.getState().previewError).toBeNull();
    });

    it("accepts https URLs", () => {
      useStore.getState().processEvent(
        makeEvent("preview_ready", { url: "https://example.com" })
      );

      expect(useStore.getState().previewUrl).toBe("https://example.com/");
      expect(useStore.getState().previewError).toBeNull();
    });

    it("rejects javascript: URLs", () => {
      useStore.getState().processEvent(
        makeEvent("preview_ready", { url: "javascript:alert(1)" })
      );

      expect(useStore.getState().previewUrl).toBeNull();
      expect(useStore.getState().previewError).toBe("Invalid preview URL received");
    });

    it("rejects data: URLs", () => {
      useStore.getState().processEvent(
        makeEvent("preview_ready", { url: "data:text/html,<h1>xss</h1>" })
      );

      expect(useStore.getState().previewUrl).toBeNull();
      expect(useStore.getState().previewError).toBe("Invalid preview URL received");
    });

    it("rejects malformed URLs", () => {
      useStore.getState().processEvent(
        makeEvent("preview_ready", { url: "not-a-valid-url" })
      );

      expect(useStore.getState().previewUrl).toBeNull();
      expect(useStore.getState().previewError).toBe("Invalid preview URL received");
    });

    it("rejects ftp: URLs", () => {
      useStore.getState().processEvent(
        makeEvent("preview_ready", { url: "ftp://files.example.com/file.txt" })
      );

      expect(useStore.getState().previewUrl).toBeNull();
      expect(useStore.getState().previewError).toBe("Invalid preview URL received");
    });

    it("sets previewLoading to false after preview_ready", () => {
      useStore.setState({ previewLoading: true });

      useStore.getState().processEvent(
        makeEvent("preview_ready", { url: "http://localhost:3000" })
      );

      expect(useStore.getState().previewLoading).toBe(false);
    });

    it("updates sandbox previewUrl for valid URLs", () => {
      useStore.getState().processEvent(
        makeEvent("preview_ready", {
          url: "http://localhost:5173",
          sandbox_id: "sb-1",
        })
      );

      const sandbox = useStore.getState().sandboxes.find((s) => s.id === "sb-1");
      expect(sandbox?.previewUrl).toBe("http://localhost:5173/");
    });
  });

  it("preview_starting sets previewLoading true", () => {
    useStore.getState().processEvent(
      makeEvent("preview_starting", { sandbox_id: "sb-1" })
    );

    expect(useStore.getState().previewLoading).toBe(true);
    expect(useStore.getState().previewError).toBeNull();
  });

  it("preview_error sets previewError and stops loading", () => {
    useStore.setState({ previewLoading: true });

    useStore.getState().processEvent(
      makeEvent("preview_error", { error: "Port not available" })
    );

    expect(useStore.getState().previewError).toBe("Port not available");
    expect(useStore.getState().previewLoading).toBe(false);
  });

  // Decomposition mode events
  it("orchestrator_plan sets subtasks and orchestratorStatus", () => {
    const subtasks = [
      {
        id: "st-1",
        role: "frontend",
        description: "build UI",
        filesResponsible: ["app.tsx"],
        dependencies: [],
      },
    ];

    useStore.getState().processEvent(
      makeEvent("orchestrator_plan", { subtasks })
    );

    const state = useStore.getState();
    expect(state.subtasks).toHaveLength(1);
    expect(state.subtasks[0]?.id).toBe("st-1");
    expect(state.orchestratorStatus).toBe("complete");
  });

  it("orchestrator_plan normalizes snake_case subtasks from backend", () => {
    useStore.getState().processEvent(
      makeEvent("orchestrator_plan", {
        subtasks: [
          {
            id: "subtask_1",
            role: "UI",
            description: "build shell",
            files_responsible: ["src/app.tsx"],
            dependencies: ["subtask_0"],
          },
        ],
      })
    );

    const subtask = useStore.getState().subtasks[0];
    expect(subtask?.filesResponsible).toEqual(["src/app.tsx"]);
    expect(subtask?.dependencies).toEqual(["subtask_0"]);
  });

  it("aggregation_started sets aggregatorStatus to active", () => {
    useStore.getState().processEvent(makeEvent("aggregation_started"));
    expect(useStore.getState().aggregatorStatus).toBe("active");
  });

  it("aggregation_complete sets aggregatorStatus and integrationStatus", () => {
    useStore.getState().processEvent(makeEvent("aggregation_complete"));
    expect(useStore.getState().aggregatorStatus).toBe("complete");
    expect(useStore.getState().integrationStatus).toBe("active");
  });

  // Hypothesis mode events
  it("evaluation_started sets evaluatorStatus to active", () => {
    useStore.getState().processEvent(makeEvent("evaluation_started"));
    expect(useStore.getState().evaluatorStatus).toBe("active");
  });

  it("evaluation_result sets scores, selected solver, and reasoning", () => {
    const scores = [
      { agentId: "s1", build: 1, lint: 1, quality: 0.8, completeness: 0.9, ux: 0.7, total: 4.4 },
      { agentId: "s2", build: 1, lint: 0.5, quality: 0.6, completeness: 0.7, ux: 0.5, total: 3.3 },
    ];

    useStore.getState().processEvent(
      makeEvent("evaluation_result", {
        scores,
        selected: "s1",
        reasoning: "Agent s1 had better quality",
      })
    );

    const state = useStore.getState();
    expect(state.evaluationScores).toHaveLength(2);
    expect(state.selectedSolverId).toBe("s1");
    expect(state.evaluationReasoning).toBe("Agent s1 had better quality");
    expect(state.evaluatorStatus).toBe("complete");
  });

  it("evaluation_result normalizes snake_case score entries", () => {
    useStore.getState().processEvent(
      makeEvent("evaluation_result", {
        scores: [
          {
            agent_id: "solver_1",
            build: 1,
            lint: 1,
            quality: 1,
            completeness: 1,
            ux: 1,
            total: 5,
          },
        ],
        selected: "solver_1",
        reasoning: "normalized",
      })
    );

    const firstScore = useStore.getState().evaluationScores[0];
    expect(firstScore?.agentId).toBe("solver_1");
    expect(firstScore?.total).toBe(5);
  });

  it("agent_spawned in non-react modes does not overwrite reactAgentInfo", () => {
    useStore.setState({ mode: "hypothesis", reactAgentInfo: null });

    useStore.getState().processEvent(
      makeEvent(
        "agent_spawned",
        { sandbox_id: "sb-1" },
        { agent_id: "solver_1", agent_role: "Solver 1" }
      )
    );

    expect(useStore.getState().reactAgentInfo).toBeNull();
  });

  it("graph_node lifecycle events update workflow statuses", () => {
    useStore.getState().processEvent(
      makeEvent("graph_node_active", { node_id: "orchestrate" })
    );
    expect(useStore.getState().orchestratorStatus).toBe("active");

    useStore.getState().processEvent(
      makeEvent("graph_node_complete", { node_id: "orchestrate" })
    );
    expect(useStore.getState().orchestratorStatus).toBe("complete");

    useStore.getState().processEvent(
      makeEvent("graph_node_active", { node_id: "evaluate" })
    );
    expect(useStore.getState().evaluatorStatus).toBe("active");

    useStore.getState().processEvent(
      makeEvent("graph_node_complete", { node_id: "evaluate" })
    );
    expect(useStore.getState().evaluatorStatus).toBe("complete");
  });
});

// ---------------------------------------------------------------------------
// events array cap (2000 max)
// ---------------------------------------------------------------------------
describe("events array cap", () => {
  it("caps events at 2000", () => {
    const { addEvent } = useStore.getState();

    for (let i = 0; i < 2050; i++) {
      addEvent(makeEvent("agent_thinking", { content: `msg ${i}` }));
    }

    const events = useStore.getState().events;
    expect(events.length).toBe(2000);
  });

  it("keeps the most recent events when capped", () => {
    const { addEvent } = useStore.getState();

    for (let i = 0; i < 2050; i++) {
      addEvent(
        makeEvent("agent_thinking", { content: `msg-${i}` })
      );
    }

    const events = useStore.getState().events;
    // The first event should be msg-50 (the oldest 50 were sliced off)
    expect((events[0]?.data.content as string)).toBe("msg-50");
    // The last event should be msg-2049
    expect((events[events.length - 1]?.data.content as string)).toBe("msg-2049");
  });
});

// ---------------------------------------------------------------------------
// terminalOutput cap (5000 max)
// ---------------------------------------------------------------------------
describe("terminalOutput cap", () => {
  it("caps terminalOutput at 5000 lines", () => {
    const { appendTerminalOutput } = useStore.getState();

    for (let i = 0; i < 5050; i++) {
      appendTerminalOutput(`line ${i}`);
    }

    const output = useStore.getState().terminalOutputs.all ?? [];
    expect(output.length).toBe(5000);
  });

  it("keeps the most recent terminal lines when capped", () => {
    const { appendTerminalOutput } = useStore.getState();

    for (let i = 0; i < 5050; i++) {
      appendTerminalOutput(`line-${i}`);
    }

    const output = useStore.getState().terminalOutputs.all ?? [];
    expect(output[0]).toBe("line-50");
    expect(output[output.length - 1]).toBe("line-5049");
  });
});

// ---------------------------------------------------------------------------
// Session lifecycle actions
// ---------------------------------------------------------------------------
describe("startSession", () => {
  it("resets state but preserves mode, numSolvers, promptValue, and task", () => {
    // Set some state first
    useStore.setState({
      mode: "hypothesis",
      numSolvers: 5,
      promptValue: "build a todo app",
      task: "build a todo app",
      events: [makeEvent("agent_thinking")],
      terminalOutputs: { all: ["some output"] },
    });

    useStore.getState().startSession("session-123");

    const state = useStore.getState();
    expect(state.sessionId).toBe("session-123");
    expect(state.sessionStatus).toBe("started");
    expect(state.connectionStatus).toBe("connecting");
    // Preserved fields
    expect(state.mode).toBe("hypothesis");
    expect(state.numSolvers).toBe(5);
    expect(state.promptValue).toBe("build a todo app");
    expect(state.task).toBe("build a todo app");
    // Reset fields
    expect(state.events).toEqual([]);
    expect(state.terminalOutputs).toEqual({ all: [] });
  });

  it("preserves workspace state when continuing the same session", () => {
    useStore.setState({
      sessionId: "session-keep",
      sessionStatus: "complete",
      mode: "react",
      task: "existing task",
      events: [makeEvent("agent_message", { text: "history" })],
      sandboxes: [
        {
          id: "sandbox_keep",
          containerId: "sandbox_keep",
          port: 5173,
          workspacePath: "/workspace",
          previewUrl: "http://localhost:5173",
        },
      ],
      selectedSandboxId: "sandbox_keep",
      fileTree: [{ name: "src", path: "src", type: "directory", children: [] }],
      terminalOutputs: { all: ["previous output"], sandbox_keep: ["sb output"] },
    });

    useStore.getState().startSession("session-keep", {
      preserveWorkspace: true,
    });

    const state = useStore.getState();
    expect(state.sessionId).toBe("session-keep");
    expect(state.sessionStatus).toBe("started");
    expect(state.sandboxes).toHaveLength(1);
    expect(state.selectedSandboxId).toBe("sandbox_keep");
    expect(state.fileTree).toEqual([
      { name: "src", path: "src", type: "directory", children: [] },
    ]);
    expect(state.events.length).toBe(1);
    expect(state.terminalOutputs).toEqual({
      all: ["previous output"],
      sandbox_keep: ["sb output"],
    });
  });
});

describe("endSession", () => {
  it("sets sessionStatus to the provided status", () => {
    useStore.getState().endSession("complete");
    expect(useStore.getState().sessionStatus).toBe("complete");

    useStore.getState().endSession("error");
    expect(useStore.getState().sessionStatus).toBe("error");

    useStore.getState().endSession("cancelled");
    expect(useStore.getState().sessionStatus).toBe("cancelled");
  });
});

describe("resetSession", () => {
  it("resets state to initial but preserves mode and numSolvers", () => {
    useStore.setState({
      mode: "decomposition",
      numSolvers: 7,
      sessionId: "session-456",
      sessionStatus: "running",
      events: [makeEvent("agent_thinking")],
      terminalOutputs: { all: ["output"] },
      fileTree: [{ name: "file.ts", path: "file.ts", type: "file" }],
    });

    useStore.getState().resetSession();

    const state = useStore.getState();
    expect(state.sessionId).toBeNull();
    expect(state.sessionStatus).toBe("idle");
    expect(state.events).toEqual([]);
    expect(state.terminalOutputs).toEqual({ all: [] });
    expect(state.fileTree).toEqual([]);
    // Preserved
    expect(state.mode).toBe("decomposition");
    expect(state.numSolvers).toBe(7);
  });
});

// ---------------------------------------------------------------------------
// File actions
// ---------------------------------------------------------------------------
describe("selectFile", () => {
  it("sets openFilePath and clears fileContent", () => {
    useStore.setState({ fileContent: "old content" });

    useStore.getState().selectFile("src/index.ts");

    const state = useStore.getState();
    expect(state.openFilePath).toBe("src/index.ts");
    expect(state.fileContent).toBeNull();
  });

  it("clears openFilePath when called with null", () => {
    useStore.setState({ openFilePath: "src/index.ts" });

    useStore.getState().selectFile(null);

    expect(useStore.getState().openFilePath).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Sandbox actions
// ---------------------------------------------------------------------------
describe("selectSandbox", () => {
  it("sets selectedSandboxId", () => {
    useStore.getState().selectSandbox("sandbox-42");
    expect(useStore.getState().selectedSandboxId).toBe("sandbox-42");
  });

  it("updates previewUrl to the selected sandbox preview", () => {
    useStore.setState({
      previewUrl: "http://localhost:3000/",
      sandboxes: [
        {
          id: "sandbox-1",
          containerId: "container-1",
          port: 3000,
          workspacePath: "/workspace",
          previewUrl: "http://localhost:5173/",
        },
      ],
    });

    useStore.getState().selectSandbox("sandbox-1");

    expect(useStore.getState().previewUrl).toBe("http://localhost:5173/");
  });

  it("clears previewUrl when selected sandbox has no preview", () => {
    useStore.setState({
      previewUrl: "http://localhost:3000/",
      sandboxes: [
        {
          id: "sandbox-2",
          containerId: "container-2",
          port: 5174,
          workspacePath: "/workspace",
        },
      ],
    });

    useStore.getState().selectSandbox("sandbox-2");

    expect(useStore.getState().previewUrl).toBeNull();
  });

  it("clears selectedSandboxId when called with null", () => {
    useStore.setState({
      selectedSandboxId: "sandbox-42",
      previewUrl: "http://localhost:5173/",
    });

    useStore.getState().selectSandbox(null);

    expect(useStore.getState().selectedSandboxId).toBeNull();
    expect(useStore.getState().previewUrl).toBeNull();
  });
});

describe("addSandbox", () => {
  it("adds a new sandbox and auto-selects it if none selected", () => {
    const sandbox = {
      id: "sb-new",
      containerId: "container-1",
      port: 5173,
      workspacePath: "/workspace",
    };

    useStore.getState().addSandbox(sandbox);

    const state = useStore.getState();
    expect(state.sandboxes).toHaveLength(1);
    expect(state.sandboxes[0]?.id).toBe("sb-new");
    expect(state.selectedSandboxId).toBe("sb-new");
  });

  it("does not change selectedSandboxId if one is already selected", () => {
    useStore.setState({ selectedSandboxId: "sb-existing" });

    const sandbox = {
      id: "sb-new",
      containerId: "container-2",
      port: 5174,
      workspacePath: "/workspace",
    };

    useStore.getState().addSandbox(sandbox);

    expect(useStore.getState().selectedSandboxId).toBe("sb-existing");
  });
});

// ---------------------------------------------------------------------------
// Connection actions
// ---------------------------------------------------------------------------
describe("setConnectionStatus", () => {
  it("updates connectionStatus", () => {
    useStore.getState().setConnectionStatus("connected");
    expect(useStore.getState().connectionStatus).toBe("connected");
  });

  it("clears connectionError when status becomes connected", () => {
    useStore.setState({ connectionError: "previous error" });

    useStore.getState().setConnectionStatus("connected");

    expect(useStore.getState().connectionError).toBeNull();
  });
});

describe("setConnectionError", () => {
  it("sets error and changes connectionStatus to error", () => {
    useStore.getState().setConnectionError("Connection failed");

    const state = useStore.getState();
    expect(state.connectionError).toBe("Connection failed");
    expect(state.connectionStatus).toBe("error");
  });

  it("clears error without changing status when called with null", () => {
    useStore.setState({ connectionStatus: "connected" });

    useStore.getState().setConnectionError(null);

    const state = useStore.getState();
    expect(state.connectionError).toBeNull();
    expect(state.connectionStatus).toBe("connected");
  });
});

// ---------------------------------------------------------------------------
// updateAgentStatus
// ---------------------------------------------------------------------------
describe("updateAgentStatus", () => {
  it("increments iterations when status becomes reasoning", () => {
    // Spawn agent
    useStore.getState().processEvent(
      makeEvent("agent_spawned", { sandbox_id: "sb-1" }, { agent_id: "a1", agent_role: "coder" })
    );

    const agent1 = useStore.getState().agents.get("a1");
    expect(agent1?.iterations).toBe(0);

    useStore.getState().updateAgentStatus("a1", "reasoning");
    expect(useStore.getState().agents.get("a1")?.iterations).toBe(1);

    useStore.getState().updateAgentStatus("a1", "executing");
    expect(useStore.getState().agents.get("a1")?.iterations).toBe(1); // unchanged

    useStore.getState().updateAgentStatus("a1", "reasoning");
    expect(useStore.getState().agents.get("a1")?.iterations).toBe(2);
  });

  it("updates reactAgentInfo if the agent is the react agent", () => {
    useStore.setState({ mode: "react" });

    useStore.getState().processEvent(
      makeEvent("agent_spawned", { sandbox_id: "sb-1" }, { agent_id: "react-a", agent_role: "agent" })
    );

    useStore.getState().updateAgentStatus("react-a", "reasoning");

    expect(useStore.getState().reactAgentInfo?.status).toBe("reasoning");
  });

  it("does nothing for unknown agent IDs", () => {
    const before = useStore.getState();
    useStore.getState().updateAgentStatus("nonexistent-agent", "reasoning");
    const after = useStore.getState();

    expect(before.agents.size).toBe(after.agents.size);
  });
});

// ---------------------------------------------------------------------------
// Every processEvent also adds to the events array
// ---------------------------------------------------------------------------
describe("processEvent always adds to events", () => {
  it("adds event to the events array regardless of type", () => {
    useStore.getState().processEvent(makeEvent("session_started"));
    useStore.getState().processEvent(makeEvent("agent_thinking"));
    useStore.getState().processEvent(makeEvent("llm_call_complete"));

    expect(useStore.getState().events).toHaveLength(3);
  });

  it("adds unknown event types to events array", () => {
    // Force an unknown event type
    const unknownEvent = makeEvent("session_started");
    (unknownEvent as Record<string, unknown>).type = "future_event_type";

    useStore.getState().processEvent(unknownEvent as AgentEvent);

    expect(useStore.getState().events).toHaveLength(1);
  });
});
