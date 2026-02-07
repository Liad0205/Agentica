"use client";

/**
 * EvaluatorNode - Node for the evaluator in hypothesis mode.
 * Features:
 * - Evaluation progress display
 * - Score comparison visualization
 * - Winner highlight and announcement
 * - Animated evaluation state
 */

import * as React from "react";
import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { COLORS, STATUS_COLORS } from "@/lib/constants";
import type { NodeStatus, EvaluationScore } from "@/lib/types";

// Evaluator icon - scale/balance
function ScaleIcon({ className }: { className?: string }): React.ReactElement {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z" />
      <path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z" />
      <path d="M7 21h10" />
      <path d="M12 3v18" />
      <path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2" />
    </svg>
  );
}

function TrophyIcon({ className }: { className?: string }): React.ReactElement {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6" />
      <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18" />
      <path d="M4 22h16" />
      <path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22" />
      <path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22" />
      <path d="M18 2H6v7a6 6 0 0 0 12 0V2Z" />
    </svg>
  );
}

function StarIcon({ className }: { className?: string }): React.ReactElement {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="2">
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
    </svg>
  );
}

interface EvaluatorNodeData extends Record<string, unknown> {
  status?: NodeStatus;
  completedCount?: number;
  totalCount?: number;
  selectedSolverId?: string;
  scores?: EvaluationScore[];
  reasoning?: string;
}

interface EvaluatorNodeProps extends NodeProps {
  data: EvaluatorNodeData;
}

function EvaluatorNodeComponent({ data, selected }: EvaluatorNodeProps): React.ReactElement {
  const {
    status = "idle",
    completedCount = 0,
    totalCount = 0,
    selectedSolverId,
    scores = [],
    reasoning,
  } = data;

  const statusColor = STATUS_COLORS[status] ?? COLORS.muted;
  const isActive = status === "active";
  const isComplete = status === "complete";
  const hasWinner = isComplete && selectedSolverId;
  const progress = totalCount > 0 ? (completedCount / totalCount) * 100 : 0;

  // Sort scores descending
  const sortedScores = [...scores].sort((a, b) => b.total - a.total);
  const maxScore = sortedScores.length > 0 ? (sortedScores[0]?.total ?? 0) : 0;

  return (
    <div
      className={`
        relative flex flex-col
        min-w-[220px] max-w-[300px] rounded-lg
        bg-gradient-to-b from-card to-card/80
        border-2 transition-all duration-300
        ${selected ? "ring-2 ring-accent ring-offset-2 ring-offset-background" : ""}
        ${isActive ? "border-accent shadow-lg shadow-accent/20" : "border-border"}
        ${hasWinner ? "border-success shadow-lg shadow-success/30" : ""}
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
        {/* Icon */}
        <div className="relative">
          <div
            className={`
              flex items-center justify-center
              w-10 h-10 rounded-full
              ${hasWinner ? "bg-success/10" : "bg-accent/10"}
              ${isActive ? "animate-pulse" : ""}
            `}
          >
            {hasWinner ? (
              <TrophyIcon className="w-5 h-5 text-success" />
            ) : (
              <ScaleIcon className="w-5 h-5 text-accent" />
            )}
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
            Evaluator
          </div>
          <div className="text-xs text-foreground-muted">
            {isActive
              ? "Evaluating solutions..."
              : hasWinner
              ? "Winner selected!"
              : "Waiting for solvers"}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-3 space-y-3">
        {/* Progress bar (waiting for solvers) */}
        {!isComplete && (
          <div>
            <div className="flex justify-between text-xs text-foreground-muted mb-1">
              <span>Solvers ready</span>
              <span>{completedCount}/{totalCount}</span>
            </div>
            <div className="h-1.5 bg-background rounded-full overflow-hidden">
              <div
                className="h-full transition-all duration-500 rounded-full"
                style={{
                  width: `${progress}%`,
                  backgroundColor: progress >= 100 ? COLORS.success : COLORS.accent,
                }}
              />
            </div>
          </div>
        )}

        {/* Evaluation in progress */}
        {isActive && (
          <div className="flex items-center justify-center gap-2 py-2">
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-accent rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
              <div className="w-2 h-2 bg-accent rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
              <div className="w-2 h-2 bg-accent rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
            </div>
            <span className="text-xs text-accent">Analyzing code quality...</span>
          </div>
        )}

        {/* Score results */}
        {isComplete && scores.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs text-foreground-muted font-medium">Results:</div>
            {sortedScores.map((score, index) => {
              const isWinner = score.agentId === selectedSolverId;
              const barWidth = maxScore > 0 ? (score.total / maxScore) * 100 : 0;

              return (
                <div
                  key={score.agentId}
                  className={`
                    relative rounded p-2
                    ${isWinner
                      ? "bg-success/10 border border-success/30"
                      : "bg-background/50 border border-transparent"
                    }
                  `}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5">
                      {isWinner && (
                        <StarIcon className="w-3 h-3 text-success" />
                      )}
                      <span
                        className={`text-xs font-medium ${
                          isWinner ? "text-success" : "text-foreground"
                        }`}
                      >
                        {score.agentId.replace("_", " ")}
                      </span>
                    </div>
                    <span
                      className={`text-sm font-bold ${
                        isWinner ? "text-success" : "text-foreground"
                      }`}
                    >
                      {score.total.toFixed(2)}
                    </span>
                  </div>

                  {/* Score bar */}
                  <div className="h-1 bg-background/50 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${barWidth}%`,
                        backgroundColor: isWinner ? COLORS.success : COLORS.accent,
                      }}
                    />
                  </div>

                  {/* Winner badge */}
                  {isWinner && index === 0 && (
                    <div className="absolute -top-1 -right-1 px-1.5 py-0.5 bg-success rounded text-[10px] font-bold text-white">
                      BEST
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Winner announcement */}
        {hasWinner && (
          <div className="flex items-center gap-2 p-2 bg-success/10 rounded-lg border border-success/30">
            <TrophyIcon className="w-5 h-5 text-success shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="text-xs text-success font-medium">Winner</div>
              <div className="text-sm font-bold text-foreground truncate">
                {selectedSolverId?.replace("_", " ")}
              </div>
            </div>
          </div>
        )}

        {/* Reasoning preview */}
        {reasoning && isComplete && (
          <div className="pt-2 border-t border-border">
            <div className="text-xs text-foreground-muted italic line-clamp-2">
              {reasoning}
            </div>
          </div>
        )}
      </div>

      {/* Glow effect */}
      {(isActive || hasWinner) && (
        <div
          className="absolute inset-0 -z-10 rounded-lg blur-xl opacity-20"
          style={{ backgroundColor: hasWinner ? COLORS.success : COLORS.accent }}
        />
      )}

      {/* Celebration particles for winner state */}
      {hasWinner && (
        <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-lg">
          <div className="absolute w-2 h-2 bg-success rounded-full animate-float-up" style={{ left: "20%", animationDelay: "0s" }} />
          <div className="absolute w-1.5 h-1.5 bg-accent rounded-full animate-float-up" style={{ left: "40%", animationDelay: "0.3s" }} />
          <div className="absolute w-2 h-2 bg-warning rounded-full animate-float-up" style={{ left: "60%", animationDelay: "0.6s" }} />
          <div className="absolute w-1.5 h-1.5 bg-success rounded-full animate-float-up" style={{ left: "80%", animationDelay: "0.9s" }} />
        </div>
      )}

    </div>
  );
}

// Memoize for performance
export const EvaluatorNode = memo(EvaluatorNodeComponent);

// Export for React Flow node types registration
export const evaluatorNodeTypes = {
  evaluator: EvaluatorNode,
};

export default EvaluatorNode;
