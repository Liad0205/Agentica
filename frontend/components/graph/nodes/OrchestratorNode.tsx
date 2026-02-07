"use client";

/**
 * OrchestratorNode - Special node for the orchestrator in decomposition mode.
 * Features:
 * - Distinctive styling to stand out as the coordinator
 * - Task decomposition plan preview
 * - Subtask count display
 * - Status indicator with pulse animation
 */

import * as React from "react";
import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { COLORS, STATUS_COLORS } from "@/lib/constants";
import type { NodeStatus } from "@/lib/types";

// Orchestrator icon
function OrchestrationIcon({ className }: { className?: string }): React.ReactElement {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v4" />
      <path d="M12 18v4" />
      <path d="m4.93 4.93 2.83 2.83" />
      <path d="m16.24 16.24 2.83 2.83" />
      <path d="M2 12h4" />
      <path d="M18 12h4" />
      <path d="m4.93 19.07 2.83-2.83" />
      <path d="m16.24 7.76 2.83-2.83" />
    </svg>
  );
}

function GitBranchIcon({ className }: { className?: string }): React.ReactElement {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="6" y1="3" x2="6" y2="15" />
      <circle cx="18" cy="6" r="3" />
      <circle cx="6" cy="18" r="3" />
      <path d="M18 9a9 9 0 0 1-9 9" />
    </svg>
  );
}

interface OrchestratorNodeData extends Record<string, unknown> {
  status?: NodeStatus;
  subtaskCount?: number;
  subtasks?: Array<{ id: string; role: string }>;
  plan?: string;
}

interface OrchestratorNodeProps extends NodeProps {
  data: OrchestratorNodeData;
}

function OrchestratorNodeComponent({ data, selected }: OrchestratorNodeProps): React.ReactElement {
  const {
    status = "idle",
    subtaskCount = 0,
    subtasks = [],
    plan,
  } = data;

  const statusColor = STATUS_COLORS[status] ?? COLORS.muted;
  const isActive = status === "active";
  const isComplete = status === "complete";

  return (
    <div
      className={`
        relative flex flex-col
        min-w-[200px] max-w-[280px] rounded-lg
        bg-gradient-to-b from-card to-card/80
        border-2 transition-all duration-300
        ${selected ? "ring-2 ring-accent ring-offset-2 ring-offset-background" : ""}
        ${isActive ? "border-accent shadow-lg shadow-accent/20" : "border-border"}
        ${isComplete ? "border-success" : ""}
      `}
    >
      {/* Source handle only (orchestrator fans out) */}
      <Handle
        type="source"
        position={Position.Bottom}
        className="!w-3 !h-3 !bg-accent !border-2 !border-card"
      />

      {/* Header */}
      <div className="flex items-center gap-3 p-3 border-b border-border">
        {/* Icon with status ring */}
        <div className="relative">
          <div
            className={`
              flex items-center justify-center
              w-10 h-10 rounded-full
              bg-accent/10
              ${isActive ? "animate-pulse" : ""}
            `}
          >
            <OrchestrationIcon className="w-5 h-5 text-accent" />
          </div>

          {/* Status dot */}
          <div
            className={`
              absolute -bottom-0.5 -right-0.5
              w-3 h-3 rounded-full border-2 border-card
              ${isActive ? "animate-pulse" : ""}
            `}
            style={{ backgroundColor: statusColor }}
          />
        </div>

        {/* Title and status */}
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-foreground">
            Orchestrator
          </div>
          <div className="text-xs text-foreground-muted">
            {isActive ? "Decomposing task..." : isComplete ? "Plan ready" : "Waiting"}
          </div>
        </div>
      </div>

      {/* Subtask preview section */}
      {(subtaskCount > 0 || subtasks.length > 0) && (
        <div className="p-3">
          <div className="flex items-center gap-2 mb-2 text-xs text-foreground-muted">
            <GitBranchIcon className="w-3.5 h-3.5" />
            <span>
              {subtaskCount > 0 ? `${subtaskCount} subtasks` : `${subtasks.length} subtasks`}
            </span>
          </div>

          {/* Subtask list preview */}
          {subtasks.length > 0 && (
            <div className="space-y-1">
              {subtasks.slice(0, 4).map((subtask, index) => (
                <div
                  key={subtask.id}
                  className="flex items-center gap-2 text-xs py-1 px-2 rounded bg-background/50"
                >
                  <span className="w-4 h-4 flex items-center justify-center rounded-full bg-accent/20 text-accent text-[10px] font-bold">
                    {index + 1}
                  </span>
                  <span className="truncate text-foreground-muted">
                    {subtask.role}
                  </span>
                </div>
              ))}
              {subtasks.length > 4 && (
                <div className="text-xs text-foreground-muted text-center py-1">
                  +{subtasks.length - 4} more
                </div>
              )}
            </div>
          )}

          {/* Plan preview tooltip */}
          {plan && (
            <div className="mt-2 text-xs text-foreground-muted line-clamp-2 italic">
              {plan}
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {subtaskCount === 0 && subtasks.length === 0 && !isActive && (
        <div className="p-3 text-center text-xs text-foreground-muted">
          Task decomposition pending
        </div>
      )}

      {/* Active state indicator */}
      {isActive && subtaskCount === 0 && subtasks.length === 0 && (
        <div className="p-3">
          <div className="flex items-center justify-center gap-2">
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-accent rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
              <div className="w-2 h-2 bg-accent rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
              <div className="w-2 h-2 bg-accent rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
            </div>
            <span className="text-xs text-accent">Analyzing task...</span>
          </div>
        </div>
      )}

      {/* Glow effect for active state */}
      {isActive && (
        <div
          className="absolute inset-0 -z-10 rounded-lg blur-xl opacity-20"
          style={{ backgroundColor: COLORS.accent }}
        />
      )}

      {/* Completion checkmark */}
      {isComplete && (
        <div className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-success flex items-center justify-center">
          <svg className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        </div>
      )}
    </div>
  );
}

// Memoize for performance
export const OrchestratorNode = memo(OrchestratorNodeComponent);

// Export for React Flow node types registration
export const orchestratorNodeTypes = {
  orchestrator: OrchestratorNode,
};

export default OrchestratorNode;
