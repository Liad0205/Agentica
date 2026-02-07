"use client";

/**
 * AggregatorNode - Node for the aggregator in decomposition mode.
 * Features:
 * - File merge progress indicator
 * - Integration status display
 * - Conflict resolution indicator
 * - Animated merge visualization
 */

import * as React from "react";
import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { COLORS, STATUS_COLORS } from "@/lib/constants";
import type { NodeStatus } from "@/lib/types";

// Aggregator icon - merge symbol
function MergeIcon({ className }: { className?: string }): React.ReactElement {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="18" cy="18" r="3" />
      <circle cx="6" cy="6" r="3" />
      <path d="M6 21V9a9 9 0 0 0 9 9" />
    </svg>
  );
}

function FilesIcon({ className }: { className?: string }): React.ReactElement {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  );
}

function CheckCircleIcon({ className }: { className?: string }): React.ReactElement {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <polyline points="9 12 12 15 16 10" />
    </svg>
  );
}

function AlertCircleIcon({ className }: { className?: string }): React.ReactElement {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}

interface AggregatorNodeData extends Record<string, unknown> {
  status?: NodeStatus;
  completedCount?: number;
  totalCount?: number;
  mergedFiles?: string[];
  conflicts?: number;
  buildStatus?: "pending" | "building" | "passed" | "failed";
}

interface AggregatorNodeProps extends NodeProps {
  data: AggregatorNodeData;
}

function AggregatorNodeComponent({ data, selected }: AggregatorNodeProps): React.ReactElement {
  const {
    status = "idle",
    completedCount = 0,
    totalCount = 0,
    mergedFiles = [],
    conflicts = 0,
    buildStatus = "pending",
  } = data;

  const statusColor = STATUS_COLORS[status] ?? COLORS.muted;
  const isActive = status === "active";
  const isComplete = status === "complete";
  const hasConflicts = conflicts > 0;
  const allAgentsComplete = completedCount === totalCount && totalCount > 0;
  const progress = totalCount > 0 ? (completedCount / totalCount) * 100 : 0;

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
        ${hasConflicts && isActive ? "border-warning" : ""}
      `}
    >
      {/* Handles */}
      <Handle
        type="target"
        position={Position.Top}
        className="!w-3 !h-3 !bg-border !border-2 !border-card"
      />
      <Handle
        type="source"
        position={Position.Bottom}
        className="!w-3 !h-3 !bg-border !border-2 !border-card"
      />

      {/* Header */}
      <div className="flex items-center gap-3 p-3 border-b border-border">
        {/* Icon with animation */}
        <div className="relative">
          <div
            className={`
              flex items-center justify-center
              w-10 h-10 rounded-full
              ${hasConflicts ? "bg-warning/10" : "bg-accent/10"}
              ${isActive ? "animate-pulse" : ""}
            `}
          >
            <MergeIcon
              className={`w-5 h-5 ${hasConflicts ? "text-warning" : "text-accent"}`}
            />
          </div>

          {/* Status indicator */}
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
            Aggregator
          </div>
          <div className="text-xs text-foreground-muted">
            {isActive
              ? hasConflicts
                ? "Resolving conflicts..."
                : "Merging files..."
              : isComplete
              ? "Merge complete"
              : "Waiting for agents"}
          </div>
        </div>
      </div>

      {/* Progress section */}
      <div className="p-3 space-y-3">
        {/* Agent completion progress */}
        <div>
          <div className="flex justify-between text-xs text-foreground-muted mb-1">
            <span>Agents completed</span>
            <span>{completedCount}/{totalCount}</span>
          </div>
          <div className="h-1.5 bg-background rounded-full overflow-hidden">
            <div
              className="h-full transition-all duration-500 rounded-full"
              style={{
                width: `${progress}%`,
                backgroundColor: allAgentsComplete ? COLORS.success : COLORS.accent,
              }}
            />
          </div>
        </div>

        {/* Merge status indicators */}
        {isActive && (
          <div className="flex items-center gap-4">
            {/* Files merged */}
            <div className="flex items-center gap-1.5 text-xs">
              <FilesIcon className="w-3.5 h-3.5 text-accent" />
              <span className="text-foreground-muted">
                {mergedFiles.length} files
              </span>
            </div>

            {/* Conflicts */}
            {hasConflicts && (
              <div className="flex items-center gap-1.5 text-xs text-warning">
                <AlertCircleIcon className="w-3.5 h-3.5" />
                <span>{conflicts} conflicts</span>
              </div>
            )}
          </div>
        )}

        {/* Build status */}
        {(isActive || isComplete) && (
          <div className="flex items-center gap-2 pt-2 border-t border-border">
            <span className="text-xs text-foreground-muted">Integration:</span>
            {buildStatus === "building" && (
              <div className="flex items-center gap-1.5">
                <div className="flex space-x-0.5">
                  <div className="w-1.5 h-1.5 bg-accent rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <div className="w-1.5 h-1.5 bg-accent rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <div className="w-1.5 h-1.5 bg-accent rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
                <span className="text-xs text-accent">Building...</span>
              </div>
            )}
            {buildStatus === "passed" && (
              <div className="flex items-center gap-1 text-success">
                <CheckCircleIcon className="w-3.5 h-3.5" />
                <span className="text-xs font-medium">Passed</span>
              </div>
            )}
            {buildStatus === "failed" && (
              <div className="flex items-center gap-1 text-error">
                <AlertCircleIcon className="w-3.5 h-3.5" />
                <span className="text-xs font-medium">Failed</span>
              </div>
            )}
            {buildStatus === "pending" && (
              <span className="text-xs text-foreground-muted">Pending</span>
            )}
          </div>
        )}

        {/* Waiting state */}
        {!isActive && !isComplete && totalCount > 0 && (
          <div className="text-center text-xs text-foreground-muted py-1">
            Waiting for {totalCount - completedCount} agent(s)...
          </div>
        )}
      </div>

      {/* Active merge animation overlay */}
      {isActive && (
        <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-lg">
          <div className="absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-transparent via-accent to-transparent animate-merge-sweep" />
        </div>
      )}

      {/* Glow effect */}
      {isActive && (
        <div
          className="absolute inset-0 -z-10 rounded-lg blur-xl opacity-20"
          style={{ backgroundColor: hasConflicts ? COLORS.warning : COLORS.accent }}
        />
      )}

      {/* Completion indicator */}
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
export const AggregatorNode = memo(AggregatorNodeComponent);

// Export for React Flow node types registration
export const aggregatorNodeTypes = {
  aggregator: AggregatorNode,
};

export default AggregatorNode;
