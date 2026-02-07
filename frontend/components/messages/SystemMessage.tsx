"use client";

import * as React from "react";
import {
  Info,
  CheckCircle2,
  AlertCircle,
  Layers,
  Trophy,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { formatTime } from "@/lib/utils";
import { COLORS } from "@/lib/constants";
import type { EventType } from "@/lib/types";
import { MessageContent } from "./MessageContent";

type SystemMessageVariant =
  | "info"
  | "success"
  | "warning"
  | "error"
  | "plan"
  | "evaluation";

interface SystemMessageProps {
  content: string;
  timestamp: number;
  variant?: SystemMessageVariant;
  eventType?: EventType;
  className?: string;
}

const variantConfig: Record<
  SystemMessageVariant,
  {
    icon: React.ElementType;
    color: string;
    bgColor: string;
  }
> = {
  info: {
    icon: Info,
    color: COLORS.accent,
    bgColor: "rgba(0, 212, 255, 0.1)",
  },
  success: {
    icon: CheckCircle2,
    color: COLORS.success,
    bgColor: "rgba(16, 185, 129, 0.1)",
  },
  warning: {
    icon: AlertCircle,
    color: COLORS.warning,
    bgColor: "rgba(245, 158, 11, 0.1)",
  },
  error: {
    icon: AlertCircle,
    color: COLORS.error,
    bgColor: "rgba(239, 68, 68, 0.1)",
  },
  plan: {
    icon: Layers,
    color: COLORS.accent,
    bgColor: "rgba(0, 212, 255, 0.1)",
  },
  evaluation: {
    icon: Trophy,
    color: COLORS.warning,
    bgColor: "rgba(245, 158, 11, 0.1)",
  },
};

/**
 * Derives the variant from an event type if not explicitly provided.
 */
function getVariantFromEventType(
  eventType?: EventType,
): SystemMessageVariant {
  if (!eventType) return "info";

  switch (eventType) {
    case "session_started":
      return "info";
    case "session_complete":
      return "success";
    case "session_error":
      return "error";
    case "session_cancelled":
      return "warning";
    case "orchestrator_plan":
      return "plan";
    case "evaluation_started":
    case "evaluation_result":
      return "evaluation";
    case "aggregation_started":
    case "aggregation_complete":
      return "info";
    default:
      return "info";
  }
}

/**
 * System message component for displaying session-level events.
 * Used for: session started, orchestrator plans, evaluation results, etc.
 *
 * Features distinct styling from agent messages with icon indicators.
 */
export function SystemMessage({
  content,
  timestamp,
  variant,
  eventType,
  className,
}: SystemMessageProps): React.ReactElement {
  const resolvedVariant =
    variant ?? getVariantFromEventType(eventType);
  const config = variantConfig[resolvedVariant];
  const Icon = config.icon;

  return (
    <div
      className={cn(
        "flex items-start gap-3 px-4 py-3 rounded-lg border border-border/50",
        "animate-slide-up",
        className,
      )}
      style={{ backgroundColor: config.bgColor }}
    >
      <div
        className="flex-shrink-0 mt-0.5"
        style={{ color: config.color }}
      >
        <Icon className="h-4 w-4" />
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span
            className="text-xs font-medium uppercase tracking-wide"
            style={{ color: config.color }}
          >
            System
          </span>
          <span className="text-xs text-muted-foreground">
            {formatTime(timestamp)}
          </span>
        </div>

        <div className="text-sm text-foreground/90 whitespace-pre-wrap break-words">
          <MessageContent content={content} />
        </div>
      </div>
    </div>
  );
}

/**
 * Specialized component for displaying orchestrator decomposition plans.
 */
interface OrchestratorPlanMessageProps {
  subtasks: Array<{
    id: string;
    role: string;
    description: string;
    filesResponsible: string[];
  }>;
  timestamp: number;
  className?: string;
}

export function OrchestratorPlanMessage({
  subtasks,
  timestamp,
  className,
}: OrchestratorPlanMessageProps): React.ReactElement {
  const config = variantConfig.plan;

  return (
    <div
      className={cn(
        "flex items-start gap-3 px-4 py-3 rounded-lg border border-border/50",
        "animate-slide-up",
        className,
      )}
      style={{ backgroundColor: config.bgColor }}
    >
      <div
        className="flex-shrink-0 mt-0.5"
        style={{ color: config.color }}
      >
        <Layers className="h-4 w-4" />
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-2">
          <span
            className="text-xs font-medium uppercase tracking-wide"
            style={{ color: config.color }}
          >
            Orchestrator Plan
          </span>
          <span className="text-xs text-muted-foreground">
            {formatTime(timestamp)}
          </span>
        </div>

        <div className="space-y-2">
          {subtasks.map((task, index) => (
            <div
              key={task.id}
              className="pl-3 border-l-2 border-border"
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-medium text-accent">
                  Task {index + 1}
                </span>
                <span className="text-sm font-medium text-foreground">
                  {task.role}
                </span>
              </div>
              <p className="text-xs text-foreground-muted mb-1">
                {task.description}
              </p>
              {task.filesResponsible.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {task.filesResponsible.map((file) => (
                    <code
                      key={file}
                      className="text-xs px-1.5 py-0.5 rounded bg-card text-accent font-mono"
                    >
                      {file}
                    </code>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/**
 * Specialized component for displaying evaluation results.
 */
interface EvaluationResultMessageProps {
  scores: Array<{
    agentId: string;
    total: number;
  }>;
  selectedAgentId: string;
  reasoning: string;
  timestamp: number;
  className?: string;
}

export function EvaluationResultMessage({
  scores,
  selectedAgentId,
  reasoning,
  timestamp,
  className,
}: EvaluationResultMessageProps): React.ReactElement {
  const config = variantConfig.evaluation;

  return (
    <div
      className={cn(
        "flex items-start gap-3 px-4 py-3 rounded-lg border border-border/50",
        "animate-slide-up",
        className,
      )}
      style={{ backgroundColor: config.bgColor }}
    >
      <div
        className="flex-shrink-0 mt-0.5"
        style={{ color: config.color }}
      >
        <Trophy className="h-4 w-4" />
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-2">
          <span
            className="text-xs font-medium uppercase tracking-wide"
            style={{ color: config.color }}
          >
            Evaluation Result
          </span>
          <span className="text-xs text-muted-foreground">
            {formatTime(timestamp)}
          </span>
        </div>

        <div className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {scores.map((score) => (
              <div
                key={score.agentId}
                className={cn(
                  "px-2 py-1 rounded text-xs font-medium",
                  score.agentId === selectedAgentId
                    ? "bg-success/20 text-success border border-success/30"
                    : "bg-card text-muted-foreground",
                )}
              >
                {score.agentId}: {score.total.toFixed(2)}
                {score.agentId === selectedAgentId &&
                  " (Winner)"}
              </div>
            ))}
          </div>

          <p className="text-sm text-foreground/90">
            {reasoning}
          </p>
        </div>
      </div>
    </div>
  );
}
