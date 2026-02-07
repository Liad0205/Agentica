/**
 * Graph layout functions for different agent modes.
 * Each layout returns positioned nodes and edges for React Flow.
 */

import type {
  GraphNode,
  GraphEdge,
  Mode,
  SubTask,
  AgentInfo,
  AgentStatus,
} from "@/lib/types";

// Layout configuration
const NODE_WIDTH = 200; // Increased to match new AgentNode min-width
const NODE_HEIGHT = 80;
const HORIZONTAL_GAP = 150; // Increased for breadth
const VERTICAL_GAP = 160; // Increased for hierarchy clarity

// Compact (horizontal) layout uses tighter spacing and flows left-to-right
const COMPACT_STEP_X = 260; // step along primary axis (left to right)
const COMPACT_FAN_GAP = 100; // proportional gap — fitView() handles scaling to panel size

interface LayoutResult {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

function mapAgentStatusToNodeStatus(status?: AgentStatus): GraphNode["status"] {
  if (status === "complete") {
    return "complete";
  }
  if (status === "failed" || status === "timeout") {
    return "error";
  }
  if (status === "reasoning" || status === "executing" || status === "reviewing") {
    return "active";
  }
  return "idle";
}

function isAgentActivelyWorking(status?: AgentStatus): boolean {
  return (
    status === "reasoning" ||
    status === "executing" ||
    status === "reviewing"
  );
}

/**
 * Generate layout for ReAct mode: cyclic graph with reason -> act -> review loop.
 * When compact=true, nodes flow left-to-right for the bottom panel.
 */
export function getReactLayout(agentInfo?: AgentInfo, compact = false): LayoutResult {
  // Determine per-phase status: when agent is complete/failed/timeout, all nodes reflect that
  const agentStatus = agentInfo?.status;
  const isFinal = agentStatus === "complete" || agentStatus === "failed" || agentStatus === "timeout";

  function phaseStatus(activeWhen: AgentStatus): GraphNode["status"] {
    if (isFinal) return agentStatus === "complete" ? "complete" : "error";
    if (agentStatus === activeWhen) return "active";
    return "idle";
  }

  // All nodes share the same iteration progress (they represent phases of one agent)
  const sharedProgress = {
    iterations: agentInfo?.iterations ?? 0,
    maxIterations: agentInfo?.maxIterations ?? 15,
  };

  // Position helper: in compact mode, primary axis is X; otherwise Y
  function pos(step: number): { x: number; y: number } {
    if (compact) {
      const startX = 40;
      const centerY = 60;
      return { x: startX + step * COMPACT_STEP_X, y: centerY };
    }
    const centerX = 800 / 2 - NODE_WIDTH / 2;
    const startY = 20;
    return { x: centerX, y: startY + step * (NODE_HEIGHT + VERTICAL_GAP) };
  }

  const nodes: GraphNode[] = [
    {
      id: "reason",
      type: "agent",
      label: "Reason & Plan",
      status: phaseStatus("reasoning"),
      position: pos(0),
      data: {
        icon: "brain",
        role: "Reasoning",
        agentId: agentInfo?.id,
        ...sharedProgress,
      },
    },
    {
      id: "execute",
      type: "agent",
      label: "Execute Tools",
      status: phaseStatus("executing"),
      position: pos(1),
      data: {
        icon: "zap",
        role: "Execution",
        agentId: agentInfo?.id,
        ...sharedProgress,
      },
    },
    {
      id: "review",
      type: "agent",
      label: "Review",
      status: phaseStatus("reviewing"),
      position: pos(2),
      data: {
        icon: "search",
        role: "Review",
        agentId: agentInfo?.id,
        ...sharedProgress,
      },
    },
  ];

  const edges: GraphEdge[] = [
    {
      id: "reason-execute",
      source: "reason",
      target: "execute",
      animated: agentInfo?.status === "reasoning",
    },
    {
      id: "execute-review",
      source: "execute",
      target: "review",
      animated: agentInfo?.status === "executing",
    },
    {
      id: "review-reason",
      source: "review",
      target: "reason",
      animated: agentInfo?.status === "reviewing",
      label: "loop",
    },
  ];

  return { nodes, edges };
}

/**
 * Generate layout for Task Decomposition mode: fan-out/fan-in pattern
 * Orchestrator -> N sub-agents -> Aggregator -> Integration
 * When compact=true, flows left-to-right with fan-out on the Y axis.
 */
export function getDecompositionLayout(
  subtasks: SubTask[] = [],
  agentStatuses: Map<string, AgentInfo> = new Map(),
  orchestratorStatus: "idle" | "active" | "complete" = "idle",
  aggregatorStatus: "idle" | "active" | "complete" = "idle",
  integrationStatus: "idle" | "active" | "complete" = "idle",
  compact = false
): LayoutResult {
  const hasSubtasks = subtasks.length > 0;
  const numAgents = Math.max(subtasks.length, 1);

  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];

  if (compact) {
    // --- Horizontal layout: left-to-right ---
    const startX = 40;
    const fanGap = numAgents <= 4 ? COMPACT_FAN_GAP : Math.max(4, COMPACT_FAN_GAP - (numAgents - 4) * 3);
    const totalFanHeight = numAgents * NODE_HEIGHT + (numAgents - 1) * fanGap;
    const fanStartY = Math.max(20, (totalFanHeight > 300 ? 20 : (300 - totalFanHeight) / 2));

    // Column positions along X
    const col0 = startX;                              // Orchestrator
    const col1 = startX + COMPACT_STEP_X;             // Sub-agents
    const col2 = startX + 2 * COMPACT_STEP_X;         // Aggregator
    const col3 = startX + 3 * COMPACT_STEP_X;         // Integration
    const centerY = fanStartY + totalFanHeight / 2 - NODE_HEIGHT / 2;

    nodes.push({
      id: "orchestrator",
      type: "orchestrator",
      label: "Orchestrator",
      status: orchestratorStatus,
      position: { x: col0, y: centerY },
      data: { subtaskCount: subtasks.length },
    });

    if (hasSubtasks) {
      subtasks.forEach((subtask, index) => {
        const subtaskAgentId = `agent_${subtask.id}`;
        const agentInfo = agentStatuses.get(subtask.id) ?? agentStatuses.get(subtaskAgentId);
        const y = fanStartY + index * (NODE_HEIGHT + fanGap);

        nodes.push({
          id: subtask.id,
          type: "agent",
          label: subtask.role,
          status: mapAgentStatusToNodeStatus(agentInfo?.status),
          position: { x: col1, y },
          data: {
            role: subtask.role,
            description: subtask.description,
            filesResponsible: subtask.filesResponsible,
            iterations: agentInfo?.iterations ?? 0,
            maxIterations: agentInfo?.maxIterations ?? 8,
            agentId: agentInfo?.id ?? subtaskAgentId,
          },
        });

        edges.push({
          id: `orchestrator-${subtask.id}`,
          source: "orchestrator",
          target: subtask.id,
          animated: orchestratorStatus === "complete" && isAgentActivelyWorking(agentInfo?.status),
        });
      });
    }

    nodes.push({
      id: "aggregator",
      type: "aggregator",
      label: "Aggregator",
      status: aggregatorStatus,
      position: { x: hasSubtasks ? col2 : col1, y: centerY },
      data: {
        completedCount: subtasks.filter((subtask) => {
          const subtaskAgentId = `agent_${subtask.id}`;
          const agentInfo = agentStatuses.get(subtask.id) ?? agentStatuses.get(subtaskAgentId);
          return agentInfo?.status === "complete";
        }).length,
        totalCount: subtasks.length,
      },
    });

    if (hasSubtasks) {
      subtasks.forEach((subtask) => {
        const subtaskAgentId = `agent_${subtask.id}`;
        const agentInfo = agentStatuses.get(subtask.id) ?? agentStatuses.get(subtaskAgentId);
        edges.push({
          id: `${subtask.id}-aggregator`,
          source: subtask.id,
          target: "aggregator",
          animated: agentInfo?.status === "complete" && aggregatorStatus !== "complete",
        });
      });
    } else {
      edges.push({
        id: "orchestrator-aggregator",
        source: "orchestrator",
        target: "aggregator",
        animated: orchestratorStatus === "active",
      });
    }

    nodes.push({
      id: "integration",
      type: "agent",
      label: "Integration Review",
      status: integrationStatus,
      position: { x: hasSubtasks ? col3 : col2, y: centerY },
      data: { icon: "check", role: "Integration" },
    });

    edges.push({
      id: "aggregator-integration",
      source: "aggregator",
      target: "integration",
      animated: aggregatorStatus === "complete" && integrationStatus !== "complete",
    });

    return { nodes, edges };
  }

  // --- Vertical layout (original) ---
  // Scale gap down when many agents to prevent excessive spread
  const agentGap = numAgents <= 3 ? HORIZONTAL_GAP : Math.max(40, HORIZONTAL_GAP - (numAgents - 3) * 25);
  const totalAgentWidth = numAgents * NODE_WIDTH + (numAgents - 1) * agentGap;

  // Canvas width expands to fit agents, minimum 800
  const canvasWidth = Math.max(800, totalAgentWidth + 200);
  const centerX = canvasWidth / 2 - NODE_WIDTH / 2;
  const agentStartX = (canvasWidth - totalAgentWidth) / 2;

  // Orchestrator node at top
  nodes.push({
    id: "orchestrator",
    type: "orchestrator",
    label: "Orchestrator",
    status: orchestratorStatus,
    position: { x: centerX, y: 20 },
    data: {
      subtaskCount: subtasks.length,
    },
  });

  // Sub-agent nodes in the middle row (only when subtasks exist)
  const subAgentRowY = 20 + NODE_HEIGHT + VERTICAL_GAP;

  if (hasSubtasks) {
    subtasks.forEach((subtask, index) => {
      const subtaskAgentId = `agent_${subtask.id}`;
      const agentInfo =
        agentStatuses.get(subtask.id) ?? agentStatuses.get(subtaskAgentId);
      const x = agentStartX + index * (NODE_WIDTH + agentGap);

      nodes.push({
        id: subtask.id,
        type: "agent",
        label: subtask.role,
        status: mapAgentStatusToNodeStatus(agentInfo?.status),
        position: { x, y: subAgentRowY },
        data: {
          role: subtask.role,
          description: subtask.description,
          filesResponsible: subtask.filesResponsible,
          iterations: agentInfo?.iterations ?? 0,
          maxIterations: agentInfo?.maxIterations ?? 8,
          agentId: agentInfo?.id ?? subtaskAgentId,
        },
      });

      // Edge from orchestrator to sub-agent
      edges.push({
        id: `orchestrator-${subtask.id}`,
        source: "orchestrator",
        target: subtask.id,
        animated:
          orchestratorStatus === "complete" &&
          isAgentActivelyWorking(agentInfo?.status),
      });
    });
  }

  // Aggregator node: directly below sub-agents if present, otherwise below orchestrator
  const aggregatorY = hasSubtasks
    ? subAgentRowY + NODE_HEIGHT + VERTICAL_GAP
    : 20 + NODE_HEIGHT + VERTICAL_GAP;
  nodes.push({
    id: "aggregator",
    type: "aggregator",
    label: "Aggregator",
    status: aggregatorStatus,
    position: { x: centerX, y: aggregatorY },
    data: {
      completedCount: subtasks.filter((subtask) => {
        const subtaskAgentId = `agent_${subtask.id}`;
        const agentInfo =
          agentStatuses.get(subtask.id) ?? agentStatuses.get(subtaskAgentId);
        return agentInfo?.status === "complete";
      }).length,
      totalCount: subtasks.length,
    },
  });

  if (hasSubtasks) {
    // Edges from sub-agents to aggregator
    subtasks.forEach((subtask) => {
      const subtaskAgentId = `agent_${subtask.id}`;
      const agentInfo =
        agentStatuses.get(subtask.id) ?? agentStatuses.get(subtaskAgentId);
      edges.push({
        id: `${subtask.id}-aggregator`,
        source: subtask.id,
        target: "aggregator",
        animated: agentInfo?.status === "complete" && aggregatorStatus !== "complete",
      });
    });
  } else {
    // Direct edge from orchestrator to aggregator when no subtasks yet
    edges.push({
      id: "orchestrator-aggregator",
      source: "orchestrator",
      target: "aggregator",
      animated: orchestratorStatus === "active",
    });
  }

  // Integration node at bottom
  const integrationY = aggregatorY + NODE_HEIGHT + VERTICAL_GAP;
  nodes.push({
    id: "integration",
    type: "agent",
    label: "Integration Review",
    status: integrationStatus,
    position: { x: centerX, y: integrationY },
    data: {
      icon: "check",
      role: "Integration",
    },
  });

  // Edge from aggregator to integration
  edges.push({
    id: "aggregator-integration",
    source: "aggregator",
    target: "integration",
    animated: aggregatorStatus === "complete" && integrationStatus !== "complete",
  });

  return { nodes, edges };
}

/**
 * Generate layout for Hypothesis mode: parallel solvers to evaluator
 * Broadcast -> N solvers -> Evaluator -> [Synthesizer] -> Selected
 * When compact=true, flows left-to-right with fan-out on the Y axis.
 */
export function getHypothesisLayout(
  numSolvers: number = 3,
  solverStatuses: Map<string, AgentInfo> = new Map(),
  evaluatorStatus: "idle" | "active" | "complete" = "idle",
  selectedSolverId?: string,
  scores?: Map<string, number>,
  synthesizerStatus: "idle" | "active" | "complete" = "idle",
  compact = false
): LayoutResult {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];

  // Persona labels for variation
  const personas = [
    { label: "Clarity", icon: "target" },
    { label: "Completeness", icon: "package" },
    { label: "Creative", icon: "palette" },
    { label: "Performance", icon: "zap" },
    { label: "Simplicity", icon: "minimize" },
  ];

  if (compact) {
    // --- Horizontal layout: left-to-right ---
    const startX = 40;
    const fanGap = numSolvers <= 4 ? COMPACT_FAN_GAP : Math.max(4, COMPACT_FAN_GAP - (numSolvers - 4) * 3);
    const totalFanHeight = numSolvers * NODE_HEIGHT + (numSolvers - 1) * fanGap;
    const fanStartY = Math.max(20, (totalFanHeight > 300 ? 20 : (300 - totalFanHeight) / 2));
    const centerY = fanStartY + totalFanHeight / 2 - NODE_HEIGHT / 2;

    let colIdx = 0;

    // Broadcast
    nodes.push({
      id: "broadcast",
      type: "broadcast",
      label: "Broadcast",
      status: "complete",
      position: { x: startX + colIdx * COMPACT_STEP_X, y: centerY },
      data: { solverCount: numSolvers },
    });
    colIdx++;

    // Solvers (fanned out on Y)
    const solverCol = startX + colIdx * COMPACT_STEP_X;
    for (let i = 0; i < numSolvers; i++) {
      const solverId = `solver_${i + 1}`;
      const agentInfo = solverStatuses.get(solverId);
      const persona = personas[i % personas.length];
      const y = fanStartY + i * (NODE_HEIGHT + fanGap);
      const score = scores?.get(solverId);
      const isSelected = selectedSolverId === solverId;

      nodes.push({
        id: solverId,
        type: "agent",
        label: `Solver ${i + 1}`,
        status: isSelected && evaluatorStatus === "complete"
          ? "complete"
          : mapAgentStatusToNodeStatus(agentInfo?.status),
        position: { x: solverCol, y },
        data: {
          role: persona?.label ?? `Solver ${i + 1}`,
          icon: persona?.icon ?? "code",
          iterations: agentInfo?.iterations ?? 0,
          maxIterations: agentInfo?.maxIterations ?? 15,
          agentId: solverId,
          score,
          isSelected,
          isWinner: isSelected && evaluatorStatus === "complete",
        },
      });

      edges.push({
        id: `broadcast-${solverId}`,
        source: "broadcast",
        target: solverId,
        animated: isAgentActivelyWorking(agentInfo?.status),
      });
    }
    colIdx++;

    // Evaluator
    nodes.push({
      id: "evaluator",
      type: "evaluator",
      label: "Evaluator",
      status: evaluatorStatus,
      position: { x: startX + colIdx * COMPACT_STEP_X, y: centerY },
      data: {
        completedCount: Array.from(solverStatuses.values()).filter((a) => a.status === "complete").length,
        totalCount: numSolvers,
        selectedSolverId,
      },
    });

    for (let i = 0; i < numSolvers; i++) {
      const solverId = `solver_${i + 1}`;
      const agentInfo = solverStatuses.get(solverId);
      const isSelected = selectedSolverId === solverId;
      edges.push({
        id: `${solverId}-evaluator`,
        source: solverId,
        target: "evaluator",
        animated: agentInfo?.status === "complete" && evaluatorStatus !== "complete",
        label: isSelected && evaluatorStatus === "complete" ? "winner" : undefined,
      });
    }
    colIdx++;

    // Synthesizer (if active)
    const showSynthesizer = synthesizerStatus !== "idle";
    if (showSynthesizer) {
      nodes.push({
        id: "synthesizer",
        type: "synthesizer",
        label: "Synthesizer",
        status: synthesizerStatus,
        position: { x: startX + colIdx * COMPACT_STEP_X, y: centerY },
        data: { icon: "sparkles", role: "Synthesizer" },
      });
      edges.push({
        id: "evaluator-synthesizer",
        source: "evaluator",
        target: "synthesizer",
        animated: evaluatorStatus === "complete" && synthesizerStatus !== "complete",
      });
      colIdx++;
    }

    // Selected solution
    if (evaluatorStatus === "complete" && selectedSolverId) {
      nodes.push({
        id: "selected",
        type: "agent",
        label: "Selected Solution",
        status: "complete",
        position: { x: startX + colIdx * COMPACT_STEP_X, y: centerY },
        data: { icon: "trophy", role: "Winner", isWinner: true, selectedFrom: selectedSolverId },
      });
      const edgeSource = showSynthesizer ? "synthesizer" : "evaluator";
      edges.push({
        id: `${edgeSource}-selected`,
        source: edgeSource,
        target: "selected",
        animated: false,
      });
    }

    return { nodes, edges };
  }

  // --- Vertical layout (original) ---
  // Scale gap down when many solvers to prevent excessive spread
  const solverGap = numSolvers <= 3 ? HORIZONTAL_GAP : Math.max(40, HORIZONTAL_GAP - (numSolvers - 3) * 25);
  const totalWidth = numSolvers * NODE_WIDTH + (numSolvers - 1) * solverGap;

  // Canvas width expands to fit solvers, minimum 800
  const canvasWidth = Math.max(800, totalWidth + 200);
  const centerX = canvasWidth / 2 - NODE_WIDTH / 2;
  const solverStartX = (canvasWidth - totalWidth) / 2;

  // Broadcast node at top
  nodes.push({
    id: "broadcast",
    type: "broadcast",
    label: "Broadcast",
    status: "complete", // Broadcast is always complete once started
    position: { x: centerX, y: 20 },
    data: {
      solverCount: numSolvers,
    },
  });

  // Solver nodes in the middle row
  for (let i = 0; i < numSolvers; i++) {
    const solverId = `solver_${i + 1}`;
    const agentInfo = solverStatuses.get(solverId);
    const persona = personas[i % personas.length];
    const x = solverStartX + i * (NODE_WIDTH + solverGap);
    const y = 20 + NODE_HEIGHT + VERTICAL_GAP;
    const score = scores?.get(solverId);
    const isSelected = selectedSolverId === solverId;

    nodes.push({
      id: solverId,
      type: "agent",
      label: `Solver ${i + 1}`,
      status:
        isSelected && evaluatorStatus === "complete"
          ? "complete"
          : mapAgentStatusToNodeStatus(agentInfo?.status),
      position: { x, y },
      data: {
        role: persona?.label ?? `Solver ${i + 1}`,
        icon: persona?.icon ?? "code",
        iterations: agentInfo?.iterations ?? 0,
        maxIterations: agentInfo?.maxIterations ?? 15,
        agentId: solverId,
        score,
        isSelected,
        isWinner: isSelected && evaluatorStatus === "complete",
      },
    });

    // Edge from broadcast to solver
    edges.push({
      id: `broadcast-${solverId}`,
      source: "broadcast",
      target: solverId,
      animated: isAgentActivelyWorking(agentInfo?.status),
    });
  }

  // Evaluator node below solvers
  const evaluatorY = 20 + 2 * (NODE_HEIGHT + VERTICAL_GAP);
  nodes.push({
    id: "evaluator",
    type: "evaluator",
    label: "Evaluator",
    status: evaluatorStatus,
    position: { x: centerX, y: evaluatorY },
    data: {
      completedCount: Array.from(solverStatuses.values())
        .filter((a) => a.status === "complete").length,
      totalCount: numSolvers,
      selectedSolverId,
    },
  });

  // Edges from solvers to evaluator
  for (let i = 0; i < numSolvers; i++) {
    const solverId = `solver_${i + 1}`;
    const agentInfo = solverStatuses.get(solverId);
    const isSelected = selectedSolverId === solverId;

    edges.push({
      id: `${solverId}-evaluator`,
      source: solverId,
      target: "evaluator",
      animated: agentInfo?.status === "complete" && evaluatorStatus !== "complete",
      label: isSelected && evaluatorStatus === "complete" ? "winner" : undefined,
    });
  }

  // Synthesizer node (shown when synthesis is active or complete)
  let nextY = evaluatorY + NODE_HEIGHT + VERTICAL_GAP;
  const showSynthesizer = synthesizerStatus !== "idle";

  if (showSynthesizer) {
    nodes.push({
      id: "synthesizer",
      type: "synthesizer",
      label: "Synthesizer",
      status: synthesizerStatus,
      position: { x: centerX, y: nextY },
      data: {
        icon: "sparkles",
        role: "Synthesizer",
      },
    });

    edges.push({
      id: "evaluator-synthesizer",
      source: "evaluator",
      target: "synthesizer",
      animated: evaluatorStatus === "complete" && synthesizerStatus !== "complete",
    });

    nextY += NODE_HEIGHT + VERTICAL_GAP;
  }

  // Selected/Final node at bottom (only shown after evaluation)
  if (evaluatorStatus === "complete" && selectedSolverId) {
    nodes.push({
      id: "selected",
      type: "agent",
      label: "Selected Solution",
      status: "complete",
      position: { x: centerX, y: nextY },
      data: {
        icon: "trophy",
        role: "Winner",
        isWinner: true,
        selectedFrom: selectedSolverId,
      },
    });

    const edgeSource = showSynthesizer ? "synthesizer" : "evaluator";
    edges.push({
      id: `${edgeSource}-selected`,
      source: edgeSource,
      target: "selected",
      animated: false,
    });
  }

  return { nodes, edges };
}

/**
 * Get the appropriate layout based on mode.
 * When compact=true, layouts flow left-to-right for the bottom panel.
 */
export function getLayoutForMode(
  mode: Mode,
  options: {
    agentInfo?: AgentInfo;
    subtasks?: SubTask[];
    agentStatuses?: Map<string, AgentInfo>;
    orchestratorStatus?: "idle" | "active" | "complete";
    aggregatorStatus?: "idle" | "active" | "complete";
    integrationStatus?: "idle" | "active" | "complete";
    numSolvers?: number;
    solverStatuses?: Map<string, AgentInfo>;
    evaluatorStatus?: "idle" | "active" | "complete";
    selectedSolverId?: string;
    scores?: Map<string, number>;
    synthesizerStatus?: "idle" | "active" | "complete";
    compact?: boolean;
  } = {}
): LayoutResult {
  const compact = options.compact ?? false;

  switch (mode) {
    case "react":
      return getReactLayout(options.agentInfo, compact);

    case "decomposition":
      return getDecompositionLayout(
        options.subtasks,
        options.agentStatuses,
        options.orchestratorStatus,
        options.aggregatorStatus,
        options.integrationStatus,
        compact
      );

    case "hypothesis":
      return getHypothesisLayout(
        options.numSolvers,
        options.solverStatuses,
        options.evaluatorStatus,
        options.selectedSolverId,
        options.scores,
        options.synthesizerStatus,
        compact
      );

    default:
      return getReactLayout(options.agentInfo, compact);
  }
}
