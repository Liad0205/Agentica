"use client";

import * as React from "react";
import {
  BarChart3,
  Settings,
  Command,
  PanelLeft,
  Network,
  ChevronDown,
  Plus,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ModeSelector } from "@/components/mode/ModeSelector";
import type { Mode, SessionSummary } from "@/lib/types";

interface HeaderProps {
  selectedMode: Mode;
  onModeChange: (mode: Mode) => void;
  onCompareClick?: () => void;
  onSettingsClick?: () => void;
  sessions?: SessionSummary[];
  activeSessionId?: string | null;
  onSessionSelect?: (sessionId: string) => void;
  onNewSession?: () => void;
  onClearSessions?: () => void;
  disabled?: boolean;
  onToggleSidebar?: () => void;
  onToggleGraph?: () => void;
  isSidebarCollapsed?: boolean;
  isGraphVisible?: boolean;
}

function formatSessionOption(session: SessionSummary): string {
  const compactTask = session.task.trim().replace(/\s+/g, " ");
  const taskLabel = compactTask.length > 44 ? `${compactTask.slice(0, 41)}...` : compactTask;
  return `${session.mode} | ${session.status} | ${taskLabel || "untitled task"}`;
}

/**
 * Streamlined header with sidebar/graph toggle buttons, dropdown session selector.
 */
export function Header({
  selectedMode,
  onModeChange,
  onCompareClick,
  onSettingsClick,
  sessions = [],
  activeSessionId = null,
  onSessionSelect,
  onNewSession,
  onClearSessions,
  disabled = false,
  onToggleSidebar,
  onToggleGraph,
  isSidebarCollapsed,
  isGraphVisible,
}: HeaderProps): React.ReactElement {
  const activeSession = activeSessionId
    ? sessions.find((s) => s.session_id === activeSessionId)
    : undefined;

  return (
    <header className="sticky top-0 z-50 flex h-12 w-full items-center border-b bg-background/80 px-4 backdrop-blur-md supports-[backdrop-filter]:bg-background/60">
      {/* Left: Sidebar toggle + Brand */}
      <div className="flex items-center gap-2">
        {onToggleSidebar && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggleSidebar}
            className="h-8 w-8 text-muted-foreground hover:text-foreground hover:bg-muted"
            aria-label={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-pressed={!isSidebarCollapsed}
          >
            <PanelLeft className="h-4 w-4" />
          </Button>
        )}
        <div className="flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-md bg-foreground text-background">
            <Command className="h-4 w-4" />
          </div>
          <span className="text-sm font-semibold tracking-tight">Agentica</span>
        </div>
      </div>

      {/* Center - Mode Selector */}
      <div className="flex-1 flex justify-center">
        <ModeSelector
          selectedMode={selectedMode}
          onModeChange={onModeChange}
          disabled={disabled}
        />
      </div>

      {/* Right Actions */}
      <div className="flex items-center gap-1">
        {/* Session dropdown */}
        {onSessionSelect && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                disabled={disabled}
                className="h-8 max-w-[280px] gap-1 text-xs text-muted-foreground hover:text-foreground hover:bg-muted"
              >
                <span className="truncate">
                  {activeSession
                    ? formatSessionOption(activeSession)
                    : "Current session"}
                </span>
                <ChevronDown className="h-3 w-3 shrink-0" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-[360px]">
              <DropdownMenuLabel className="text-xs">Sessions</DropdownMenuLabel>
              <DropdownMenuSeparator />
              {sessions.map((session) => (
                <DropdownMenuItem
                  key={session.session_id}
                  onClick={() => onSessionSelect(session.session_id)}
                  className="text-xs"
                >
                  <span className="truncate">{formatSessionOption(session)}</span>
                </DropdownMenuItem>
              ))}
              {sessions.length === 0 && (
                <DropdownMenuItem disabled className="text-xs text-muted-foreground">
                  No sessions yet
                </DropdownMenuItem>
              )}
              <DropdownMenuSeparator />
              {onNewSession && (
                <DropdownMenuItem onClick={onNewSession} className="text-xs">
                  <Plus className="mr-2 h-3.5 w-3.5" />
                  New Session
                </DropdownMenuItem>
              )}
              {onClearSessions && (
                <DropdownMenuItem
                  onClick={onClearSessions}
                  className="text-xs text-destructive focus:text-destructive"
                >
                  <Trash2 className="mr-2 h-3.5 w-3.5" />
                  Clear Sessions
                </DropdownMenuItem>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        )}

        {/* Graph toggle */}
        {onToggleGraph && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggleGraph}
            className={`h-8 w-8 text-muted-foreground hover:text-foreground hover:bg-muted ${
              isGraphVisible ? "bg-muted/50 text-foreground" : ""
            }`}
            aria-label={isGraphVisible ? "Hide graph panel" : "Show graph panel"}
            aria-pressed={isGraphVisible}
          >
            <Network className="h-4 w-4" />
          </Button>
        )}

        {/* Compare (icon-only) */}
        {onCompareClick && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onCompareClick}
            className="h-8 w-8 text-muted-foreground hover:text-foreground hover:bg-muted"
            aria-label="Compare modes"
          >
            <BarChart3 className="h-4 w-4" />
          </Button>
        )}

        {/* Settings */}
        {onSettingsClick && (
          <Button
            variant="ghost"
            size="icon"
            onClick={onSettingsClick}
            className="h-8 w-8 text-muted-foreground hover:text-foreground hover:bg-muted"
          >
            <Settings className="h-4 w-4" />
          </Button>
        )}
      </div>
    </header>
  );
}
