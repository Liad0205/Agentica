"use client";

import * as React from "react";
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
  type ImperativePanelHandle,
} from "react-resizable-panels";
import {
  MessageSquare,
  Network,
  Code2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { SidebarRail } from "./SidebarRail";
import { GraphToggleBar } from "./GraphToggleBar";
import type { SessionStatus, Mode } from "@/lib/types";

interface WorkspaceLayoutProps {
  sidebar: React.ReactNode;
  mainContent: React.ReactNode;
  bottomPanel: React.ReactNode;
  promptBar: React.ReactNode;
  isGraphVisible: boolean;
  onToggleGraph: () => void;
  isSidebarCollapsed: boolean;
  onToggleSidebar: () => void;
  eventCount: number;
  sessionStatus: SessionStatus;
  mode: Mode;
  className?: string;
}

/**
 * Adaptive WorkspaceLayout with collapsible sidebar rail and bottom graph panel.
 * Code Panel is promoted to primary content.
 */
export function WorkspaceLayout({
  sidebar,
  mainContent,
  bottomPanel,
  promptBar,
  isGraphVisible,
  onToggleGraph,
  isSidebarCollapsed,
  onToggleSidebar,
  eventCount,
  sessionStatus,
  mode,
  className,
}: WorkspaceLayoutProps): React.ReactElement {
  const [isMobile, setIsMobile] = React.useState(false);
  const [activeMobilePanel, setActiveMobilePanel] = React.useState<
    "sidebar" | "graph" | "code"
  >("code");

  const sidebarPanelRef = React.useRef<ImperativePanelHandle>(null);
  const graphPanelRef = React.useRef<ImperativePanelHandle>(null);

  // Responsive breakpoint detection
  React.useEffect(() => {
    const mobileQuery = window.matchMedia("(max-width: 1023px)");
    const updateMobile = (): void => setIsMobile(mobileQuery.matches);
    updateMobile();
    mobileQuery.addEventListener("change", updateMobile);
    return () => mobileQuery.removeEventListener("change", updateMobile);
  }, []);

  // Auto-collapse sidebar on narrow screens (1024-1280px)
  React.useEffect(() => {
    if (isMobile) return;
    const narrowQuery = window.matchMedia("(max-width: 1279px)");
    if (narrowQuery.matches && !isSidebarCollapsed) {
      onToggleSidebar();
    }
    // Only run on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isMobile]);

  // Sync sidebar panel with collapsed state
  React.useEffect(() => {
    const panel = sidebarPanelRef.current;
    if (!panel) return;
    if (isSidebarCollapsed) {
      panel.collapse();
    } else {
      panel.expand();
    }
  }, [isSidebarCollapsed]);

  // Sync graph panel with visibility state
  React.useEffect(() => {
    const panel = graphPanelRef.current;
    if (!panel) return;
    if (isGraphVisible) {
      panel.expand();
      panel.resize(60);
    } else {
      panel.collapse();
    }
  }, [isGraphVisible]);

  // Handle sidebar panel collapse/expand events from drag
  const handleSidebarCollapse = React.useCallback((): void => {
    if (!isSidebarCollapsed) {
      onToggleSidebar();
    }
  }, [isSidebarCollapsed, onToggleSidebar]);

  const handleSidebarExpand = React.useCallback((): void => {
    if (isSidebarCollapsed) {
      onToggleSidebar();
    }
  }, [isSidebarCollapsed, onToggleSidebar]);

  // Handle graph panel collapse/expand from drag
  const handleGraphCollapse = React.useCallback((): void => {
    if (isGraphVisible) {
      onToggleGraph();
    }
  }, [isGraphVisible, onToggleGraph]);

  const handleGraphExpand = React.useCallback((): void => {
    if (!isGraphVisible) {
      onToggleGraph();
    }
  }, [isGraphVisible, onToggleGraph]);

  // Mobile layout
  if (isMobile) {
    const mobilePanels = [
      { id: "sidebar" as const, label: "Log", icon: MessageSquare },
      { id: "graph" as const, label: "Graph", icon: Network },
      { id: "code" as const, label: "Code", icon: Code2 },
    ];

    return (
      <div className={cn("flex flex-1 flex-col overflow-hidden bg-background", className)}>
        <div className="flex h-12 items-center border-b border-border/80 bg-card/30 px-4">
          <div className="flex w-full items-center rounded-lg bg-muted/40 p-1">
            {mobilePanels.map((panel) => {
              const Icon = panel.icon;
              const isActive = activeMobilePanel === panel.id;
              return (
                <button
                  key={panel.id}
                  onClick={() => setActiveMobilePanel(panel.id)}
                  className={cn(
                    "flex flex-1 items-center justify-center gap-2 rounded-md px-2 py-1.5 text-xs font-medium transition-all",
                    isActive
                      ? "bg-card text-foreground shadow-sm ring-1 ring-border/80"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {panel.label}
                </button>
              );
            })}
          </div>
        </div>
        <div className="flex-1 overflow-hidden">
          {activeMobilePanel === "sidebar" && <div className="h-full">{sidebar}</div>}
          {activeMobilePanel === "graph" && <div className="h-full">{bottomPanel}</div>}
          {activeMobilePanel === "code" && <div className="h-full">{mainContent}</div>}
        </div>
        <div className="border-t border-border/80 bg-background px-3 py-2">
          {promptBar}
        </div>
      </div>
    );
  }

  // Desktop layout
  return (
    <div className={cn("flex flex-1 flex-col overflow-hidden bg-background", className)}>
      <PanelGroup direction="horizontal" className="flex-1" autoSaveId="workspace-v2-main">
        {/* Sidebar panel - collapsible to rail */}
        <Panel
          ref={sidebarPanelRef}
          defaultSize={22}
          minSize={18}
          maxSize={30}
          collapsible
          collapsedSize={3}
          onCollapse={handleSidebarCollapse}
          onExpand={handleSidebarExpand}
          className={cn(
            "border-r border-border/80",
            isSidebarCollapsed ? "bg-card/30" : "bg-card/25",
          )}
        >
          {isSidebarCollapsed ? (
            <SidebarRail
              eventCount={eventCount}
              sessionStatus={sessionStatus}
              onExpandSidebar={onToggleSidebar}
              onToggleGraph={onToggleGraph}
              isGraphVisible={isGraphVisible}
            />
          ) : (
            sidebar
          )}
        </Panel>

        <PanelResizeHandle className="w-[2px] bg-border/40 transition-colors hover:bg-accent/70" />

        {/* Main content area with vertical split */}
        <Panel defaultSize={78} minSize={50}>
          <div className="flex h-full flex-col">
            <PanelGroup direction="vertical" className="flex-1" autoSaveId="workspace-v2-vertical">
              {/* Code panel (primary) */}
              <Panel defaultSize={100} minSize={40}>
                {mainContent}
              </Panel>

              {/* Graph toggle bar + resize handle */}
              <PanelResizeHandle
                disabled={!isGraphVisible}
                className={cn(
                  "relative flex flex-col items-stretch",
                  isGraphVisible && "hover:bg-accent/10",
                )}
              >
                <GraphToggleBar
                  isGraphVisible={isGraphVisible}
                  onToggle={onToggleGraph}
                  mode={mode}
                />
                {isGraphVisible && (
                  <div className="mx-auto h-[2px] w-12 rounded-full bg-border/60 transition-colors group-hover:bg-accent/70" />
                )}
              </PanelResizeHandle>

              {/* Graph panel - collapsible bottom, drag to resize */}
              <Panel
                ref={graphPanelRef}
                defaultSize={0}
                minSize={20}
                collapsible
                collapsedSize={0}
                onCollapse={handleGraphCollapse}
                onExpand={handleGraphExpand}
              >
                {bottomPanel}
              </Panel>
            </PanelGroup>

            {/* Prompt bar */}
            <div className="border-t border-border/80 bg-background px-3 py-2">
              {promptBar}
            </div>
          </div>
        </Panel>
      </PanelGroup>
    </div>
  );
}

interface PanelContentProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
  actions?: React.ReactNode;
  padding?: boolean;
}

export function PanelContent({
  children,
  className,
  title,
  actions,
  padding = true,
}: PanelContentProps): React.ReactElement {
  return (
    <div className={cn("flex flex-col h-full bg-background", className)}>
      {(title || actions) && (
        <div className="flex h-10 items-center justify-between border-b px-4">
          {title && (
            <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              {title}
            </h2>
          )}
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      )}
      <div className={cn("flex-1 overflow-auto", padding && "p-4")}>
        {children}
      </div>
    </div>
  );
}
