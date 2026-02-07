"use client";

import * as React from "react";
import {
  CheckCircle2,
  AlertCircle,
  Brain,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { formatTime, getInitials } from "@/lib/utils";
import {
  getAgentColor,
  STATUS_COLORS,
} from "@/lib/constants";
import { MessageContent } from "./MessageContent";

type MessageStatus = "thinking" | "complete" | "error";

interface AgentMessageProps {
  agentId: string;
  agentRole: string;
  content: string;
  timestamp: number;
  status?: MessageStatus;
  isStreaming?: boolean;
  className?: string;
}

const statusConfig: Record<
  MessageStatus,
  {
    icon: React.ElementType;
    label: string;
    animate?: boolean;
  }
> = {
  thinking: {
    icon: Brain,
    label: "Thinking",
    animate: true,
  },
  complete: {
    icon: CheckCircle2,
    label: "Complete",
    animate: false,
  },
  error: {
    icon: AlertCircle,
    label: "Error",
    animate: false,
  },
};

/**
 * Displays a message from an agent with avatar, role name, timestamp,
 * and status indicator.
 *
 * Features:
 * - Colored avatar circle with agent initials
 * - Role name and timestamp in header
 * - Message content with markdown support (basic)
 * - Status indicator (thinking, complete, error)
 * - Streaming animation for thinking state
 */
export function AgentMessage({
  agentId,
  agentRole,
  content,
  timestamp,
  status = "complete",
  isStreaming = false,
  className,
}: AgentMessageProps): React.ReactElement {
  const agentColor = getAgentColor(agentId);
  const initials = getInitials(agentRole);
  const config = statusConfig[status];
  const StatusIcon = config.icon;

  return (
    <div
      className={cn(
        "flex items-start gap-3 px-4 py-3 rounded-lg",
        "bg-card/50 border border-border/30",
        "animate-slide-up",
        className,
      )}
    >
      {/* Agent Avatar */}
      <div
        className={cn(
          "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center",
          "text-xs font-semibold text-background",
          status === "thinking" && "animate-pulse-glow",
        )}
        style={{ backgroundColor: agentColor }}
      >
        {initials}
      </div>

      {/* Message Content */}
      <div className="flex-1 min-w-0">
        {/* Header: Role + Timestamp + Status */}
        <div className="flex items-center gap-2 mb-1.5">
          <span
            className="text-sm font-medium"
            style={{ color: agentColor }}
          >
            {agentRole}
          </span>
          <span className="text-xs text-muted-foreground">
            {formatTime(timestamp)}
          </span>
          {/* Status Indicator */}
          <div className="flex items-center gap-1 ml-auto">
            <StatusIcon
              className={cn(
                "h-3.5 w-3.5",
                config.animate && "animate-pulse",
              )}
              style={{
                color:
                  status === "thinking"
                    ? STATUS_COLORS.reasoning
                    : status === "error"
                      ? STATUS_COLORS.error
                      : STATUS_COLORS.complete,
              }}
            />
            <span
              className="text-xs"
              style={{
                color:
                  status === "thinking"
                    ? STATUS_COLORS.reasoning
                    : status === "error"
                      ? STATUS_COLORS.error
                      : STATUS_COLORS.complete,
              }}
            >
              {config.label}
            </span>
          </div>
        </div>

        {/* Message Body */}
        <div className="text-sm text-foreground/90 whitespace-pre-wrap break-words">
          <MessageContent content={content} />
          {isStreaming && (
            <span className="inline-block w-2 h-4 ml-1 bg-accent animate-pulse" />
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Compact version of AgentMessage for use in dense lists.
 */
interface CompactAgentMessageProps {
  agentId: string;
  agentRole: string;
  content: string;
  timestamp: number;
  className?: string;
}

export function CompactAgentMessage({
  agentId,
  agentRole,
  content,
  timestamp,
  className,
}: CompactAgentMessageProps): React.ReactElement {
  const agentColor = getAgentColor(agentId);

  return (
    <div
      className={cn(
        "flex items-start gap-2 px-3 py-2",
        "hover:bg-card/30 transition-colors",
        className,
      )}
    >
      <div
        className="flex-shrink-0 w-1.5 h-1.5 rounded-full mt-2"
        style={{ backgroundColor: agentColor }}
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span
            className="text-xs font-medium"
            style={{ color: agentColor }}
          >
            {agentRole}
          </span>
          <span className="text-xs text-muted-foreground">
            {formatTime(timestamp)}
          </span>
        </div>
        <p className="text-xs text-foreground/80 truncate">
          {content}
        </p>
      </div>
    </div>
  );
}
