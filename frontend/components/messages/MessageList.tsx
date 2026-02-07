"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AgentMessage } from "./AgentMessage";
import { ToolCallMessage } from "./ToolCallMessage";
import { SystemMessage, OrchestratorPlanMessage, EvaluationResultMessage } from "./SystemMessage";
import type { AgentEvent, EventType, SubTask, EvaluationScore } from "@/lib/types";

interface MessageListProps {
  events: AgentEvent[];
  filterAgentId?: string | null;
  className?: string;
}

/**
 * Renders a scrollable list of messages grouped by agent.
 *
 * Features:
 * - Auto-scroll to bottom on new messages
 * - Groups consecutive messages from the same agent
 * - Filter by agent ID
 * - Virtualized rendering for performance (via native scroll)
 */
export function MessageList({
  events,
  filterAgentId,
  className,
}: MessageListProps): React.ReactElement {
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = React.useState(true);
  const [showScrollButton, setShowScrollButton] = React.useState(false);
  const lastEventCountRef = React.useRef(0);

  // Filter events if filterAgentId is set
  const filteredEvents = React.useMemo(() => {
    const scopedEvents = !filterAgentId
      ? events
      : events.filter(
          (event) =>
            event.agent_id === filterAgentId ||
            isSystemEvent(event.type)
        );
    return reconcileToolLifecycleEvents(scopedEvents);
  }, [events, filterAgentId]);

  // Auto-scroll to bottom when new events arrive
  React.useEffect(() => {
    if (autoScroll && filteredEvents.length > lastEventCountRef.current) {
      if (scrollRef.current) {
        if (typeof scrollRef.current.scrollTo === "function") {
          scrollRef.current.scrollTo({
            top: scrollRef.current.scrollHeight,
            behavior: "smooth",
          });
        } else {
          scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
      }
      setShowScrollButton(false);
    }
    lastEventCountRef.current = filteredEvents.length;
  }, [filteredEvents.length, autoScroll]);

  // Detect manual scroll to disable auto-scroll
  function handleScroll(e: React.UIEvent<HTMLDivElement>): void {
    const target = e.currentTarget;
    const isAtBottom =
      Math.abs(target.scrollHeight - target.scrollTop - target.clientHeight) < 50;
    setAutoScroll(isAtBottom);
    setShowScrollButton(!isAtBottom);
  }

  function scrollToBottom(): void {
    if (scrollRef.current) {
      if (typeof scrollRef.current.scrollTo === "function") {
        scrollRef.current.scrollTo({
          top: scrollRef.current.scrollHeight,
          behavior: "smooth",
        });
      } else {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    }
    setAutoScroll(true);
    setShowScrollButton(false);
  }

  // Group consecutive events from the same agent
  const groupedEvents = React.useMemo(() => {
    return groupEventsByAgent(filteredEvents);
  }, [filteredEvents]);

  if (filteredEvents.length === 0) {
    return (
      <div className={cn("flex items-center justify-center h-full", className)}>
        <p className="text-sm text-muted-foreground">No messages yet</p>
      </div>
    );
  }

  return (
    <div className={cn("relative h-full", className)}>
      <ScrollArea
        ref={scrollRef}
        className="h-full"
        onScroll={handleScroll}
      >
        <div className="p-4 space-y-3">
          {groupedEvents.map((group, groupIndex) => (
            <MessageGroup key={`group-${groupIndex}-${group.events[0]?.timestamp}`} group={group} />
          ))}

          {/* Scroll anchor for auto-scroll */}
          <div className="h-px" />
        </div>
      </ScrollArea>
      <ScrollToBottomButton
        onClick={scrollToBottom}
        visible={showScrollButton}
      />
    </div>
  );
}

interface EventGroup {
  agentId: string | null;
  agentRole: string | null;
  events: AgentEvent[];
}

/**
 * Groups consecutive events from the same agent together.
 */
function groupEventsByAgent(events: AgentEvent[]): EventGroup[] {
  const groups: EventGroup[] = [];
  let currentGroup: EventGroup | null = null;

  for (const event of events) {
    const eventAgentId = event.agent_id ?? null;
    const eventAgentRole = event.agent_role ?? null;

    // Start a new group if:
    // - No current group
    // - Different agent
    // - System event (always its own group)
    if (
      !currentGroup ||
      currentGroup.agentId !== eventAgentId ||
      isSystemEvent(event.type)
    ) {
      currentGroup = {
        agentId: eventAgentId,
        agentRole: eventAgentRole,
        events: [event],
      };
      groups.push(currentGroup);
    } else {
      currentGroup.events.push(event);
    }
  }

  return groups;
}

/**
 * Checks if an event type is a system-level event.
 */
function isSystemEvent(type: EventType): boolean {
  const systemEvents: EventType[] = [
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

function reconcileToolLifecycleEvents(events: AgentEvent[]): AgentEvent[] {
  const completedToolCalls = new Set<string>();

  for (const event of events) {
    if (event.type !== "agent_tool_result") {
      continue;
    }
    const toolCallId = event.data.tool_call_id;
    if (typeof toolCallId === "string" && toolCallId.length > 0) {
      completedToolCalls.add(toolCallId);
    }
  }

  return events.filter((event) => {
    if (event.type !== "agent_tool_call") {
      return true;
    }
    const toolCallId = event.data.tool_call_id;
    if (typeof toolCallId !== "string" || toolCallId.length === 0) {
      return true;
    }
    return !completedToolCalls.has(toolCallId);
  });
}

function toRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

function toStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

interface MessageGroupProps {
  group: EventGroup;
}

/**
 * Renders a group of events from the same agent.
 */
function MessageGroup({ group }: MessageGroupProps): React.ReactElement {
  return (
    <div className="space-y-2">
      {group.events.map((event, index) => (
        <EventMessage
          key={`${event.type}-${event.timestamp}-${index}`}
          event={event}
        />
      ))}
    </div>
  );
}

interface EventMessageProps {
  event: AgentEvent;
}

/**
 * Renders a single event as the appropriate message component.
 */
function EventMessage({ event }: EventMessageProps): React.ReactElement | null {
  switch (event.type) {
    case "session_started":
      return (
        <SystemMessage
          content={`Session started in ${(event.data.mode as string) ?? "unknown"} mode`}
          timestamp={event.timestamp}
          eventType={event.type}
        />
      );

    case "session_complete":
      return (
        <SystemMessage
          content="Session completed successfully"
          timestamp={event.timestamp}
          eventType={event.type}
        />
      );

    case "session_error":
      return (
        <SystemMessage
          content={(event.data.error as string) ?? "An error occurred"}
          timestamp={event.timestamp}
          eventType={event.type}
        />
      );

    case "session_cancelled":
      return (
        <SystemMessage
          content="Session was cancelled"
          timestamp={event.timestamp}
          eventType={event.type}
          variant="warning"
        />
      );

    case "agent_thinking":
      return (
        <AgentMessage
          agentId={event.agent_id ?? "unknown"}
          agentRole={event.agent_role ?? "Agent"}
          content={(event.data.content as string) ?? ""}
          timestamp={event.timestamp}
          status="thinking"
          isStreaming={(event.data.streaming as boolean) ?? false}
        />
      );

    case "agent_message":
      return (
        <AgentMessage
          agentId={event.agent_id ?? "unknown"}
          agentRole={event.agent_role ?? "Agent"}
          content={(event.data.content as string) ?? ""}
          timestamp={event.timestamp}
          status="complete"
        />
      );

    case "agent_complete":
      return (
        <AgentMessage
          agentId={event.agent_id ?? "unknown"}
          agentRole={event.agent_role ?? "Agent"}
          content={(event.data.summary as string) ?? "Task completed"}
          timestamp={event.timestamp}
          status="complete"
        />
      );

    case "agent_error":
      return (
        <AgentMessage
          agentId={event.agent_id ?? "unknown"}
          agentRole={event.agent_role ?? "Agent"}
          content={(event.data.error as string) ?? "An error occurred"}
          timestamp={event.timestamp}
          status="error"
        />
      );

    case "agent_tool_call":
      return (
        <ToolCallMessage
          agentId={event.agent_id ?? "unknown"}
          agentRole={event.agent_role ?? "Agent"}
          toolName={(event.data.tool as string) ?? "unknown"}
          args={(event.data.args as Record<string, unknown>) ?? {}}
          status="pending"
          timestamp={event.timestamp}
        />
      );

    case "agent_tool_result":
      return (
        <ToolCallMessage
          agentId={event.agent_id ?? "unknown"}
          agentRole={event.agent_role ?? "Agent"}
          toolName={(event.data.tool as string) ?? "unknown"}
          args={(event.data.args as Record<string, unknown>) ?? {}}
          result={(event.data.result as string) ?? ""}
          status={(event.data.success as boolean) ? "success" : "error"}
          timestamp={event.timestamp}
        />
      );

    case "orchestrator_plan": {
      const subtasks = Array.isArray(event.data.subtasks)
        ? (event.data.subtasks as SubTask[])
        : [];
      return (
        <OrchestratorPlanMessage
          subtasks={subtasks.map((st, index) => {
            const source = toRecord(st);
            return {
              id:
                typeof source.id === "string" && source.id.length > 0
                  ? source.id
                  : `subtask_${index + 1}`,
              role:
                typeof source.role === "string" && source.role.length > 0
                  ? source.role
                  : `Subtask ${index + 1}`,
              description:
                typeof source.description === "string"
                  ? source.description
                  : "",
              filesResponsible: toStringArray(
                source.filesResponsible ?? source.files_responsible
              ),
            };
          })}
          timestamp={event.timestamp}
        />
      );
    }

    case "evaluation_started":
      return (
        <SystemMessage
          content="Evaluation of solutions started..."
          timestamp={event.timestamp}
          eventType={event.type}
        />
      );

    case "evaluation_result": {
      const scores = Array.isArray(event.data.scores)
        ? (event.data.scores as EvaluationScore[])
        : [];
      const selected = (event.data.selected as string) ?? "";
      const reasoning = (event.data.reasoning as string) ?? "";
      return (
        <EvaluationResultMessage
          scores={scores.map((score) => {
            const source = toRecord(score);
            const agentId =
              typeof source.agentId === "string"
                ? source.agentId
                : typeof source.agent_id === "string"
                  ? source.agent_id
                  : "";

            const total =
              typeof source.total === "number" && Number.isFinite(source.total)
                ? source.total
                : 0;

            return { agentId, total };
          })}
          selectedAgentId={selected}
          reasoning={reasoning}
          timestamp={event.timestamp}
        />
      );
    }

    case "aggregation_started":
      return (
        <SystemMessage
          content="Aggregating solutions from sub-agents..."
          timestamp={event.timestamp}
          eventType={event.type}
        />
      );

    case "aggregation_complete":
      return (
        <SystemMessage
          content="Solution aggregation complete"
          timestamp={event.timestamp}
          variant="success"
        />
      );

    // Events we don't display as messages
    case "agent_spawned":
    case "graph_initialized":
    case "graph_node_active":
    case "graph_node_complete":
    case "graph_edge_traversed":
    case "file_changed":
    case "file_deleted":
    case "command_started":
    case "command_output":
    case "command_complete":
    case "preview_starting":
    case "preview_ready":
    case "preview_error":
    case "agent_timeout":
    case "llm_call_complete":
      return null;

    default:
      // Unknown event type - render as system message
      return (
        <SystemMessage
          content={`Unknown event: ${event.type}`}
          timestamp={event.timestamp}
          variant="warning"
        />
      );
  }
}

/**
 * Scroll-to-bottom button component.
 */
interface ScrollToBottomButtonProps {
  onClick: () => void;
  visible: boolean;
}

export function ScrollToBottomButton({
  onClick,
  visible,
}: ScrollToBottomButtonProps): React.ReactElement | null {
  if (!visible) return null;

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "absolute bottom-4 right-4 px-3 py-1.5 rounded-full",
        "bg-accent text-background text-xs font-medium",
        "hover:bg-accent/90 transition-colors",
        "shadow-lg animate-slide-up"
      )}
    >
      Scroll to bottom
    </button>
  );
}
