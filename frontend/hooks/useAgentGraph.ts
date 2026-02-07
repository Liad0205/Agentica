/**
 * Custom hook for agent graph visualization state.
 * Provides mode-specific data for rendering agent graphs in React Flow.
 */

import { useMemo } from "react";
import { useStore } from "@/lib/store";
import type {
  AgentInfo,
  EvaluationScore,
  Mode,
  SubTask,
} from "@/lib/types";

/**
 * Workflow status type for decomposition and hypothesis modes.
 */
type WorkflowStatus = "idle" | "active" | "complete";

/**
 * Return type for the useAgentGraph hook.
 */
export interface UseAgentGraphReturn {
  mode: Mode;
  // ReAct mode
  agentInfo: AgentInfo | null;
  // Decomposition mode
  subtasks: SubTask[];
  agentStatuses: Map<string, AgentInfo>;
  orchestratorStatus: WorkflowStatus;
  aggregatorStatus: WorkflowStatus;
  integrationStatus: WorkflowStatus;
  // Hypothesis mode
  numSolvers: number;
  solverStatuses: Map<string, AgentInfo>;
  evaluatorStatus: WorkflowStatus;
  synthesizerStatus: WorkflowStatus;
  selectedSolverId: string | null;
  scores: Map<string, number>;
  evaluationScores: EvaluationScore[];
  evaluationReasoning: string | null;
}

/**
 * Hook for accessing agent graph visualization state.
 * Provides all necessary data for rendering mode-specific agent graphs.
 */
export function useAgentGraph(): UseAgentGraphReturn {
  const mode = useStore((state) => state.mode);

  // ReAct mode state
  const agentInfo = useStore((state) => state.reactAgentInfo);

  // Decomposition mode state
  const subtasks = useStore((state) => state.subtasks);
  const agents = useStore((state) => state.agents);
  const orchestratorStatus = useStore((state) => state.orchestratorStatus);
  const aggregatorStatus = useStore((state) => state.aggregatorStatus);
  const integrationStatus = useStore((state) => state.integrationStatus);

  // Hypothesis mode state
  const numSolvers = useStore((state) => state.numSolvers);
  const evaluatorStatus = useStore((state) => state.evaluatorStatus);
  const synthesizerStatus = useStore((state) => state.synthesizerStatus);
  const selectedSolverId = useStore((state) => state.selectedSolverId);
  const evaluationScores = useStore((state) => state.evaluationScores);
  const evaluationReasoning = useStore((state) => state.evaluationReasoning);

  // For decomposition mode, agentStatuses is the same as agents
  // (subtask agents are tracked in the agents map)
  const agentStatuses = agents;

  // For hypothesis mode, derive solver statuses from agents
  // Solvers are identified by their role containing "solver"
  const solverStatuses = useMemo((): Map<string, AgentInfo> => {
    const solvers = new Map<string, AgentInfo>();
    agents.forEach((agent, id) => {
      if (agent.role.toLowerCase().includes("solver")) {
        solvers.set(id, agent);
      }
    });
    return solvers;
  }, [agents]);

  // Derive scores map from evaluationScores array
  const scores = useMemo((): Map<string, number> => {
    const scoresMap = new Map<string, number>();
    for (const score of evaluationScores) {
      scoresMap.set(score.agentId, score.total);
    }
    return scoresMap;
  }, [evaluationScores]);

  return {
    mode,
    // ReAct
    agentInfo,
    // Decomposition
    subtasks,
    agentStatuses,
    orchestratorStatus,
    aggregatorStatus,
    integrationStatus,
    // Hypothesis
    numSolvers,
    solverStatuses,
    evaluatorStatus,
    synthesizerStatus,
    selectedSolverId,
    scores,
    evaluationScores,
    evaluationReasoning,
  };
}
