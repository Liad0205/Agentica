"use client";

import * as React from "react";
import { MessageSquare, Network } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { SessionStatus } from "@/lib/types";

interface SidebarRailProps {
  eventCount: number;
  sessionStatus: SessionStatus;
  onExpandSidebar: () => void;
  onToggleGraph: () => void;
  isGraphVisible: boolean;
  className?: string;
}

const statusColorMap: Record<SessionStatus, string> = {
  idle: "bg-muted-foreground",
  started: "bg-accent",
  running: "bg-accent animate-pulse",
  complete: "bg-green-500",
  error: "bg-red-500",
  cancelled: "bg-yellow-500",
};

export function SidebarRail({
  eventCount,
  sessionStatus,
  onExpandSidebar,
  onToggleGraph,
  isGraphVisible,
  className,
}: SidebarRailProps): React.ReactElement {
  return (
    <TooltipProvider delayDuration={200}>
      <div
        className={cn(
          "flex h-full w-12 flex-col items-center gap-1 border-r border-border/80 bg-card/30 py-3",
          className,
        )}
      >
        {/* Session Log */}
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              onClick={onExpandSidebar}
              className="relative flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              aria-label="Expand session log"
            >
              <MessageSquare className="h-4 w-4" />
              {eventCount > 0 && (
                <span className="absolute -right-0.5 -top-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-accent px-1 text-[9px] font-bold text-accent-foreground">
                  {eventCount > 99 ? "99+" : eventCount}
                </span>
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent side="right">
            <p>Session Log</p>
          </TooltipContent>
        </Tooltip>

        {/* Graph Toggle */}
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              onClick={onToggleGraph}
              className={cn(
                "flex h-9 w-9 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
                isGraphVisible && "bg-muted/50 text-foreground",
              )}
              aria-label={isGraphVisible ? "Hide agent graph" : "Show agent graph"}
              aria-pressed={isGraphVisible}
            >
              <Network className="h-4 w-4" />
            </button>
          </TooltipTrigger>
          <TooltipContent side="right">
            <p>Agent Graph</p>
          </TooltipContent>
        </Tooltip>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Session Status */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex h-9 w-9 items-center justify-center">
              <div
                className={cn(
                  "h-2.5 w-2.5 rounded-full",
                  statusColorMap[sessionStatus],
                )}
              />
            </div>
          </TooltipTrigger>
          <TooltipContent side="right">
            <p>Session: {sessionStatus}</p>
          </TooltipContent>
        </Tooltip>
      </div>
    </TooltipProvider>
  );
}
