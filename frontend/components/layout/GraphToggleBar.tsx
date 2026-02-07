"use client";

import * as React from "react";
import { Network, ChevronUp, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import type { Mode } from "@/lib/types";
import { MODE_INFO } from "@/lib/constants";

interface GraphToggleBarProps {
  isGraphVisible: boolean;
  onToggle: () => void;
  mode: Mode;
  className?: string;
}

const DRAG_THRESHOLD = 5; // px — movement beyond this is a drag, not a click

export function GraphToggleBar({
  isGraphVisible,
  onToggle,
  mode,
  className,
}: GraphToggleBarProps): React.ReactElement {
  const modeLabel = MODE_INFO[mode].name;
  const pointerStart = React.useRef<{ x: number; y: number } | null>(null);

  const handlePointerDown = React.useCallback((e: React.PointerEvent) => {
    pointerStart.current = { x: e.clientX, y: e.clientY };
  }, []);

  const handlePointerUp = React.useCallback(
    (e: React.PointerEvent) => {
      if (!pointerStart.current) return;
      const dx = Math.abs(e.clientX - pointerStart.current.x);
      const dy = Math.abs(e.clientY - pointerStart.current.y);
      pointerStart.current = null;

      // Only toggle if the pointer barely moved (i.e. a click, not a drag)
      if (dx < DRAG_THRESHOLD && dy < DRAG_THRESHOLD) {
        onToggle();
      }
    },
    [onToggle],
  );

  return (
    <div
      role="button"
      tabIndex={0}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onToggle();
        }
      }}
      className={cn(
        "flex h-8 w-full items-center justify-between border-t border-border/80 bg-card/50 px-3 transition-colors hover:bg-card/70",
        "cursor-pointer select-none",
        className,
      )}
      aria-expanded={isGraphVisible}
      aria-label={isGraphVisible ? "Collapse agent graph" : "Expand agent graph"}
    >
      <div className="flex items-center gap-2">
        <Network className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-xs font-medium text-muted-foreground">Agent Graph</span>
        <Badge variant="outline" className="h-4 px-1.5 text-[10px] font-normal">
          {modeLabel}
        </Badge>
      </div>
      <div className="flex items-center">
        {isGraphVisible ? (
          <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
        ) : (
          <ChevronUp className="h-3.5 w-3.5 text-muted-foreground" />
        )}
      </div>
    </div>
  );
}
