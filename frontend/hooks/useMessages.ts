/**
 * Custom hook for accessing events and agent messages from the store.
 * Provides both the raw Map and a convenience array for iteration.
 */

import { useMemo } from "react";
import { useStore } from "@/lib/store";
import type { AgentEvent, AgentInfo } from "@/lib/types";

/**
 * Return type for the useMessages hook.
 */
export interface UseMessagesReturn {
  events: AgentEvent[];
  agents: Map<string, AgentInfo>;
  agentList: AgentInfo[];
}

/**
 * Hook for accessing events and agent information from the store.
 * Converts the agents Map to an array for convenient iteration.
 */
export function useMessages(): UseMessagesReturn {
  const events = useStore((state) => state.events);
  const agents = useStore((state) => state.agents);

  // Derive an array from the agents Map for easy iteration
  const agentList = useMemo((): AgentInfo[] => {
    return Array.from(agents.values());
  }, [agents]);

  return {
    events,
    agents,
    agentList,
  };
}
