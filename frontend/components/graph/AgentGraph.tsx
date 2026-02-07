"use client";

/**
 * AgentGraph - React Flow wrapper component for agent topology visualization.
 * Features:
 * - Handles different layouts per mode (ReAct, Decomposition, Hypothesis)
 * - Minimap, controls, and background
 * - Fit view on load
 * - Dynamic node/edge updates
 * - Dark theme styling
 */

import { useCallback, useEffect, useMemo, useRef } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type FitViewOptions,
  type ReactFlowInstance,
  BackgroundVariant,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import type {
  Mode,
  GraphNode,
  GraphEdge,
  SubTask,
  AgentInfo,
  EvaluationScore,
} from "@/lib/types";

// Import custom nodes
import { AgentNode } from "./nodes/AgentNode";
import { OrchestratorNode } from "./nodes/OrchestratorNode";
import { AggregatorNode } from "./nodes/AggregatorNode";
import { EvaluatorNode } from "./nodes/EvaluatorNode";

// Import custom edges
import { AnimatedEdge } from "./edges/AnimatedEdge";

// Import layout functions
import { getLayoutForMode } from "./layouts";

// Register custom node types
const nodeTypes = {
  agent: AgentNode,
  orchestrator: OrchestratorNode,
  aggregator: AggregatorNode,
  evaluator: EvaluatorNode,
  broadcast: AgentNode, // Use agent node with broadcast styling
  synthesizer: AgentNode, // Use agent node with synthesizer styling
};

// Register custom edge types
const edgeTypes = {
  animated: AnimatedEdge,
};

// fitViewOptions are computed per-instance based on compact prop (see component body)

// Default edge options
const defaultEdgeOptions = {
  type: "animated",
  markerEnd: {
    type: MarkerType.ArrowClosed,
    color: "hsl(var(--accent) / 0.4)",
    width: 20,
    height: 20,
  },
  style: {
    strokeWidth: 2,
    stroke: "hsl(var(--border) / 0.85)",
  },
};

// Props interface
interface AgentGraphProps {
  mode: Mode;
  // ReAct mode
  agentInfo?: AgentInfo;
  // Decomposition mode
  subtasks?: SubTask[];
  agentStatuses?: Map<string, AgentInfo>;
  orchestratorStatus?: "idle" | "active" | "complete";
  aggregatorStatus?: "idle" | "active" | "complete";
  integrationStatus?: "idle" | "active" | "complete";
  // Hypothesis mode
  numSolvers?: number;
  solverStatuses?: Map<string, AgentInfo>;
  evaluatorStatus?: "idle" | "active" | "complete";
  synthesizerStatus?: "idle" | "active" | "complete";
  selectedSolverId?: string;
  scores?: Map<string, number>;
  evaluationScores?: EvaluationScore[];
  evaluationReasoning?: string;
  // Common
  compact?: boolean;
  className?: string;
}

// Convert GraphNode to React Flow Node
function toReactFlowNode(graphNode: GraphNode): Node {
  return {
    id: graphNode.id,
    type: graphNode.type,
    position: graphNode.position,
    data: {
      ...graphNode.data,
      label: graphNode.label,
      status: graphNode.status,
    },
    draggable: true,
  };
}

// Convert GraphEdge to React Flow Edge
function toReactFlowEdge(graphEdge: GraphEdge): Edge {
  return {
    id: graphEdge.id,
    source: graphEdge.source,
    target: graphEdge.target,
    type: "animated",
    animated: graphEdge.animated ?? false,
    label: graphEdge.label,
    data: {
      label: graphEdge.label,
      isWinner: graphEdge.label === "winner",
    },
  };
}

export function AgentGraph({
  mode,
  agentInfo,
  subtasks,
  agentStatuses,
  orchestratorStatus,
  aggregatorStatus,
  integrationStatus,
  numSolvers,
  solverStatuses,
  evaluatorStatus,
  synthesizerStatus,
  selectedSolverId,
  scores,
  evaluationScores,
  evaluationReasoning,
  compact = false,
  className = "",
}: AgentGraphProps): React.ReactElement {
  // Store React Flow instance for imperative fitView calls
  const reactFlowRef = useRef<ReactFlowInstance | null>(null);

  const onInit = useCallback((instance: ReactFlowInstance) => {
    reactFlowRef.current = instance;
  }, []);

  // Compute fitView options based on compact mode
  const activeFitViewOptions = useMemo((): FitViewOptions => ({
    padding: compact ? 0.15 : 0.12,
    maxZoom: compact ? 1.2 : 1.8,
    minZoom: compact ? 0.3 : 0.6,
  }), [compact]);

  // Get layout based on mode and current state
  const layout = useMemo(() => {
    return getLayoutForMode(mode, {
      agentInfo,
      subtasks,
      agentStatuses,
      orchestratorStatus,
      aggregatorStatus,
      integrationStatus,
      numSolvers,
      solverStatuses,
      evaluatorStatus,
      synthesizerStatus,
      selectedSolverId,
      scores,
      compact,
    });
  }, [
    mode,
    agentInfo,
    subtasks,
    agentStatuses,
    orchestratorStatus,
    aggregatorStatus,
    integrationStatus,
    numSolvers,
    solverStatuses,
    evaluatorStatus,
    synthesizerStatus,
    selectedSolverId,
    scores,
    compact,
  ]);

  // Convert to React Flow format
  const initialNodes = useMemo(
    () => layout.nodes.map(toReactFlowNode),
    [layout.nodes],
  );
  const initialEdges = useMemo(
    () => layout.edges.map(toReactFlowEdge),
    [layout.edges],
  );

  // Use React Flow state hooks
  const [nodes, setNodes, onNodesChange] =
    useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] =
    useEdgesState(initialEdges);

  // Track the previous layout structure to avoid spurious fitView calls
  // when only node data (not structure/position) changes.
  const prevLayoutRef = useRef(layout);
  const prevCompactRef = useRef(compact);

  // Update nodes and edges when layout changes
  useEffect(() => {
    const updatedNodes = layout.nodes.map(toReactFlowNode);

    // Enrich evaluator node with scores if available
    const enrichedNodes = updatedNodes.map((node) => {
      if (node.type === "evaluator" && evaluationScores) {
        return {
          ...node,
          data: {
            ...node.data,
            scores: evaluationScores,
            reasoning: evaluationReasoning,
          },
        };
      }
      return node;
    });

    setNodes(enrichedNodes);
    setEdges(layout.edges.map(toReactFlowEdge));

    // Re-fit when the layout structure changes (node count, IDs, or compact
    // toggle).  Compact toggle changes all positions dramatically even though
    // IDs stay the same, so it must be treated as a structural change.
    // Data-only changes (e.g. agent status) do NOT trigger re-fit so the
    // graph doesn't jump around during normal execution.
    const compactChanged = prevCompactRef.current !== compact;
    const structureChanged =
      compactChanged ||
      (prevLayoutRef.current !== layout &&
        (prevLayoutRef.current.nodes.length !== layout.nodes.length ||
          prevLayoutRef.current.nodes.some(
            (n, i) => n.id !== layout.nodes[i]?.id
          )));
    prevLayoutRef.current = layout;
    prevCompactRef.current = compact;

    // Imperatively re-fit after nodes update so the graph adapts to
    // panel open/close, resize, and layout changes (not just mount).
    // Use a retry loop: React Flow may not have measured nodes yet on
    // first animation frame after mount.
    if (structureChanged || !reactFlowRef.current) {
      let retries = 0;
      const tryFitView = (): void => {
        if (reactFlowRef.current) {
          reactFlowRef.current.fitView(activeFitViewOptions);
        } else if (retries < 5) {
          retries += 1;
          requestAnimationFrame(tryFitView);
        }
      };
      requestAnimationFrame(tryFitView);
    }
  }, [
    layout,
    compact,
    evaluationScores,
    evaluationReasoning,
    setNodes,
    setEdges,
    activeFitViewOptions,
  ]);

  // Re-fit when the container is resized (e.g. the graph panel becomes
  // visible for the first time, or the user resizes the window).  Without
  // this, React Flow may render into a zero-size container and the nodes
  // appear to vanish.
  const containerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = containerRef.current;
    if (!el) {
      return;
    }
    let debounceTimeoutId: ReturnType<typeof setTimeout> | null = null;
    const observer = new ResizeObserver(() => {
      // Debounce the fitView call to avoid excessive re-fits during
      // panel drag operations (which fire on every pixel change).
      if (debounceTimeoutId !== null) {
        clearTimeout(debounceTimeoutId);
      }
      debounceTimeoutId = setTimeout(() => {
        // Only re-fit if the container has non-zero dimensions
        if (el.offsetWidth > 0 && el.offsetHeight > 0) {
          requestAnimationFrame(() => {
            reactFlowRef.current?.fitView(activeFitViewOptions);
          });
        }
        debounceTimeoutId = null;
      }, 150);
    });
    observer.observe(el);
    return () => {
      if (debounceTimeoutId !== null) {
        clearTimeout(debounceTimeoutId);
      }
      observer.disconnect();
    };
  }, [activeFitViewOptions]);

  return (
    <div ref={containerRef} className={`relative h-full w-full ${className}`}>
      <div className="mission-canvas-glow pointer-events-none absolute inset-0" />
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onInit={onInit}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        defaultEdgeOptions={defaultEdgeOptions}
        nodesDraggable={false}
        fitView
        fitViewOptions={activeFitViewOptions}
        minZoom={0.3}
        maxZoom={2}
        attributionPosition="bottom-left"
        proOptions={{ hideAttribution: true }}
        className="agent-graph relative z-10"
      >
        {/* Grid background */}
        <Background
          variant={BackgroundVariant.Dots}
          gap={22}
          size={1.1}
          color="hsl(var(--border) / 0.75)"
          className="bg-background"
        />

        {/* Controls - hidden in compact mode */}
        {!compact && (
          <Controls
            showZoom
            showFitView
            showInteractive={false}
            position="bottom-right"
            className="agent-graph-controls"
          />
        )}
      </ReactFlow>
    </div>
  );
}

export default AgentGraph;
