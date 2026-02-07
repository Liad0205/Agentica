"use client";

import * as React from "react";
import { MessageSquare, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";
import { MessageFilter } from "./MessageFilter";
import { MessageList } from "./MessageList";
import type { AgentEvent, AgentInfo } from "@/lib/types";

interface MessagePanelProps {
  events: AgentEvent[];
  agents: AgentInfo[];
  className?: string;
}

/**
 * Main container for the session log / message panel.
 *
 * Features:
 * - Header with "Session Log" title
 * - Agent filter dropdown
 * - Scrollable message list
 * - Auto-scroll to bottom functionality
 */
export function MessagePanel({
  events,
  agents,
  className,
}: MessagePanelProps): React.ReactElement {
  const [filterAgentId, setFilterAgentId] = React.useState<string | null>(null);

  // Count messages for display
  const messageCount = events.filter((e) => isDisplayableEvent(e.type)).length;
  const filteredCount = filterAgentId
    ? events.filter(
        (e) =>
          e.agent_id === filterAgentId ||
          isSystemEvent(e.type)
      ).filter((e) => isDisplayableEvent(e.type)).length
    : messageCount;

  return (
    <div className={cn("flex flex-col h-full bg-card rounded-lg border border-border", className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-4 w-4 text-accent" />
          <h2 className="text-sm font-medium text-foreground">Session Log</h2>
          {messageCount > 0 && (
            <span className="text-xs text-muted-foreground">
              ({filterAgentId ? `${filteredCount}/` : ""}{messageCount})
            </span>
          )}
        </div>

        <MessageFilter
          agents={agents}
          selectedAgentId={filterAgentId}
          onAgentSelect={setFilterAgentId}
        />
      </div>

      {/* Message List */}
      <div className="flex-1 overflow-hidden">
        <MessageList
          events={events}
          filterAgentId={filterAgentId}
          className="h-full"
        />
      </div>

      {/* Footer - Connection status */}
      {events.length === 0 && (
        <div className="border-t border-border px-4 py-3">
          <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
            <RefreshCw className="h-3 w-3 animate-spin" />
            Waiting for session to start...
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Checks if an event type should be displayed in the message list.
 */
function isDisplayableEvent(type: string): boolean {
  const displayableEvents = [
    "session_started",
    "session_complete",
    "session_error",
    "session_cancelled",
    "agent_thinking",
    "agent_message",
    "agent_complete",
    "agent_error",
    "agent_tool_call",
    "agent_tool_result",
    "orchestrator_plan",
    "evaluation_started",
    "evaluation_result",
    "aggregation_started",
    "aggregation_complete",
  ];
  return displayableEvents.includes(type);
}

/**
 * Checks if an event type is a system-level event.
 */
function isSystemEvent(type: string): boolean {
  const systemEvents = [
    "session_started",
    "session_complete",
    "session_error",
    "session_cancelled",
    "orchestrator_plan",
    "evaluation_started",
    "evaluation_result",
    "aggregation_started",
    "aggregation_complete",
  ];
  return systemEvents.includes(type);
}

/**
 * Compact version of MessagePanel for smaller viewports.
 */
interface CompactMessagePanelProps {
  events: AgentEvent[];
  maxMessages?: number;
  className?: string;
}

export function CompactMessagePanel({
  events,
  maxMessages = 5,
  className,
}: CompactMessagePanelProps): React.ReactElement {
  const recentEvents = events.slice(-maxMessages);

  return (
    <div className={cn("bg-card rounded-lg border border-border p-3", className)}>
      <div className="flex items-center gap-2 mb-2">
        <MessageSquare className="h-3.5 w-3.5 text-accent" />
        <span className="text-xs font-medium text-foreground">Recent Activity</span>
      </div>

      <div className="space-y-1">
        {recentEvents.length === 0 ? (
          <p className="text-xs text-muted-foreground">No activity yet</p>
        ) : (
          recentEvents.map((event, index) => (
            <CompactEventLine
              key={`${event.type}-${event.timestamp}-${index}`}
              event={event}
            />
          ))
        )}
      </div>
    </div>
  );
}

interface CompactEventLineProps {
  event: AgentEvent;
}

function CompactEventLine({ event }: CompactEventLineProps): React.ReactElement | null {
  let content: string;

  switch (event.type) {
    case "agent_thinking":
      content = `${event.agent_role ?? "Agent"} is thinking...`;
      break;
    case "agent_tool_call":
      content = `${event.agent_role ?? "Agent"}: ${event.data.tool}`;
      break;
    case "agent_complete":
      content = `${event.agent_role ?? "Agent"} completed`;
      break;
    case "session_started":
      content = "Session started";
      break;
    case "session_complete":
      content = "Session complete";
      break;
    default:
      return null;
  }

  return (
    <div className="flex items-center gap-2 truncate text-xs text-muted-foreground">
      <span className="w-1.5 h-1.5 rounded-full bg-accent flex-shrink-0" />
      <span className="truncate">{content}</span>
    </div>
  );
}
