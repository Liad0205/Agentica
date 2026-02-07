"use client";

import * as React from "react";
import {
  ChevronDown,
  ChevronRight,
  FileCode,
  FileText,
  FolderTree,
  Terminal,
  Search,
  CheckCircle2,
  XCircle,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { formatTime, truncate } from "@/lib/utils";
import { getAgentColor, TOOL_DISPLAY_NAMES, COLORS } from "@/lib/constants";
import { Badge } from "@/components/ui/badge";

type ToolStatus = "pending" | "success" | "error";

interface ToolCallMessageProps {
  agentId: string;
  agentRole: string;
  toolName: string;
  args: Record<string, unknown>;
  result?: string;
  status: ToolStatus;
  timestamp: number;
  className?: string;
}

const toolIcons: Record<string, React.ElementType> = {
  write_file: FileCode,
  read_file: FileText,
  list_files: FolderTree,
  execute_command: Terminal,
  search_files: Search,
};

const statusConfig: Record<
  ToolStatus,
  { icon: React.ElementType; color: string; animate?: boolean }
> = {
  pending: {
    icon: Loader2,
    color: COLORS.active,
    animate: true,
  },
  success: {
    icon: CheckCircle2,
    color: COLORS.success,
    animate: false,
  },
  error: {
    icon: XCircle,
    color: COLORS.error,
    animate: false,
  },
};

/**
 * Displays a tool call with expandable/collapsible view.
 *
 * Features:
 * - Tool name badge with icon
 * - Args display (JSON formatted, truncated by default)
 * - Result display (truncated, expandable)
 * - Success/error indicator
 */
export function ToolCallMessage({
  agentId,
  agentRole,
  toolName,
  args,
  result,
  status,
  timestamp,
  className,
}: ToolCallMessageProps): React.ReactElement {
  const [isExpanded, setIsExpanded] = React.useState(false);
  const [showFullResult, setShowFullResult] = React.useState(false);

  const agentColor = getAgentColor(agentId);
  const ToolIcon = toolIcons[toolName] ?? Terminal;
  const displayName = TOOL_DISPLAY_NAMES[toolName] ?? toolName;
  const config = statusConfig[status];
  const StatusIcon = config.icon;

  const argsString = formatArgs(args);
  const truncatedResult = result ? truncate(result, 200) : null;
  const hasLongResult = result && result.length > 200;

  return (
    <div
      className={cn(
        "rounded-lg border border-border/30 overflow-hidden",
        "animate-slide-up",
        className
      )}
    >
      {/* Header - Always visible */}
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className={cn(
          "w-full flex items-center gap-3 px-3 py-2",
          "bg-card/30 hover:bg-card/50 transition-colors",
          "text-left"
        )}
      >
        {/* Expand/Collapse Indicator */}
        {isExpanded ? (
          <ChevronDown className="h-4 w-4 text-muted-foreground flex-shrink-0" />
        ) : (
          <ChevronRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
        )}

        {/* Tool Icon */}
        <ToolIcon className="h-4 w-4 text-muted-foreground flex-shrink-0" />

        {/* Tool Name Badge */}
        <Badge variant="secondary" className="font-mono text-xs">
          {displayName}
        </Badge>

        {/* Brief Args Preview */}
        <span className="flex-1 truncate text-xs text-muted-foreground">
          {getArgsPreview(args)}
        </span>

        {/* Agent Color Dot */}
        <div
          className="w-2 h-2 rounded-full flex-shrink-0"
          style={{ backgroundColor: agentColor }}
          title={agentRole}
        />

        {/* Status Indicator */}
        <StatusIcon
          className={cn("h-4 w-4 flex-shrink-0", config.animate && "animate-spin")}
          style={{ color: config.color }}
        />
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="border-t border-border/30 bg-background/50">
          {/* Arguments Section */}
          <div className="px-4 py-3 border-b border-border/20">
            <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Arguments
            </div>
            <pre className="text-xs font-mono text-foreground/80 overflow-x-auto whitespace-pre-wrap">
              {argsString}
            </pre>
          </div>

          {/* Result Section */}
          {(result || status === "pending") && (
            <div className="px-4 py-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Result
                </span>
                {hasLongResult && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setShowFullResult(!showFullResult);
                    }}
                    className="text-xs text-accent hover:underline"
                  >
                    {showFullResult ? "Show Less" : "Show Full"}
                  </button>
                )}
              </div>

              {status === "pending" ? (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Executing...
                </div>
              ) : (
                <pre
                  className={cn(
                    "text-xs font-mono overflow-x-auto whitespace-pre-wrap",
                    status === "error" ? "text-error" : "text-foreground/80"
                  )}
                >
                  {showFullResult ? result : truncatedResult}
                </pre>
              )}
            </div>
          )}

          {/* Timestamp */}
          <div className="border-t border-border/20 px-4 py-2 text-xs text-muted-foreground">
            {formatTime(timestamp)}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Format arguments for display, handling special cases.
 */
function formatArgs(args: Record<string, unknown>): string {
  // Special handling for write_file - don't show full content
  if ("content" in args && typeof args.content === "string") {
    const content = args.content as string;
    const truncatedContent =
      content.length > 500
        ? content.slice(0, 500) + "\n... (truncated)"
        : content;
    return JSON.stringify({ ...args, content: truncatedContent }, null, 2);
  }

  return JSON.stringify(args, null, 2);
}

/**
 * Get a brief preview of args for the collapsed view.
 */
function getArgsPreview(args: Record<string, unknown>): string {
  // For file operations, show the path
  if ("path" in args && typeof args.path === "string") {
    return args.path;
  }

  // For commands, show the command
  if ("command" in args && typeof args.command === "string") {
    return truncate(args.command as string, 40);
  }

  // For search, show the pattern
  if ("pattern" in args && typeof args.pattern === "string") {
    return `"${args.pattern}"`;
  }

  // Default: show keys
  const keys = Object.keys(args);
  return keys.length > 0 ? keys.join(", ") : "(no args)";
}

/**
 * Compact tool call indicator for use inline with messages.
 */
interface CompactToolCallProps {
  toolName: string;
  status: ToolStatus;
  onClick?: () => void;
  className?: string;
}

export function CompactToolCall({
  toolName,
  status,
  onClick,
  className,
}: CompactToolCallProps): React.ReactElement {
  const ToolIcon = toolIcons[toolName] ?? Terminal;
  const config = statusConfig[status];
  const StatusIcon = config.icon;

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-1.5 px-2 py-1 rounded",
        "bg-card/50 hover:bg-card transition-colors",
        "text-xs font-mono",
        className
      )}
    >
      <ToolIcon className="h-3 w-3 text-muted-foreground" />
      <span className="text-foreground/80">{toolName}</span>
      <StatusIcon
        className={cn("h-3 w-3", config.animate && "animate-spin")}
        style={{ color: config.color }}
      />
    </button>
  );
}
