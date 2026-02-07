"use client";

import * as React from "react";
import { Select } from "@/components/ui/select";
import { getAgentColor } from "@/lib/constants";
import type { AgentInfo } from "@/lib/types";

interface MessageFilterProps {
  agents: AgentInfo[];
  selectedAgentId: string | null;
  onAgentSelect: (agentId: string | null) => void;
}

/**
 * Dropdown filter for selecting which agent's messages to display.
 * Shows "All Agents" by default, then lists all active agents with their colors.
 */
export function MessageFilter({
  agents,
  selectedAgentId,
  onAgentSelect,
}: MessageFilterProps): React.ReactElement {
  const options = React.useMemo(() => {
    const allOption = {
      value: "all",
      label: "All Agents",
    };

    const agentOptions = agents.map((agent) => ({
      value: agent.id,
      label: agent.role,
      color: getAgentColor(agent.id),
    }));

    return [allOption, ...agentOptions];
  }, [agents]);

  function handleValueChange(value: string): void {
    onAgentSelect(value === "all" ? null : value);
  }

  return (
    <Select
      value={selectedAgentId ?? "all"}
      onValueChange={handleValueChange}
      options={options}
      placeholder="Filter by agent"
      className="w-40"
    />
  );
}
