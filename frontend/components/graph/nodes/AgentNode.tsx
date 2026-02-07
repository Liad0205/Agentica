"use client";

import * as React from "react";
import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { NodeStatus } from "@/lib/types";
import { cn } from "@/lib/utils";
import { 
    Brain, 
    Zap, 
    Search, 
    Code, 
    Crosshair, 
    Package, 
    Palette, 
    Trophy, 
    Check 
} from "lucide-react";

const ICONS = {
  brain: Brain,
  zap: Zap,
  search: Search,
  code: Code,
  target: Crosshair,
  package: Package,
  palette: Palette,
  trophy: Trophy,
  check: Check,
};

interface AgentNodeData extends Record<string, unknown> {
  role?: string;
  icon?: string;
  agentId?: string;
  iterations?: number;
  maxIterations?: number;
  score?: number;
  isSelected?: boolean;
  isWinner?: boolean;
  description?: string;
}

interface AgentNodeProps extends NodeProps {
  data: AgentNodeData;
}

function AgentNodeComponent({ data, selected }: AgentNodeProps): React.ReactElement {
  const {
    role = "Agent",
    icon = "code",
    iterations = 0,
    maxIterations = 15,
    score,
    isWinner = false,
    status: dataStatus,
  } = data;

  const status = (dataStatus as NodeStatus | undefined) ?? (isWinner ? "complete" : "idle");
  const isActive = status === "active" || status === "reasoning" || status === "executing";
  const IconComponent = ICONS[icon as keyof typeof ICONS] ?? Code;
  const progress = maxIterations > 0 ? (iterations / maxIterations) * 100 : 0;

  return (
    <div
      className={cn(
        "relative flex w-[240px] flex-col rounded-lg border bg-card p-3 shadow-sm transition-all",
        selected && "ring-1 ring-ring",
        isActive && "border-foreground/50 ring-1 ring-ring/20",
        isWinner && "border-success/50 bg-success/5"
      )}
    >
      <Handle type="target" position={Position.Top} className="!h-1.5 !w-1.5 !bg-muted-foreground/50 transition-colors hover:!bg-foreground" />
      <Handle type="source" position={Position.Bottom} className="!h-1.5 !w-1.5 !bg-muted-foreground/50 transition-colors hover:!bg-foreground" />

      <div className="flex items-start justify-between gap-3">
        <div className="flex gap-3">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border bg-muted/30">
             <IconComponent className="h-4 w-4 text-foreground" />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-semibold text-foreground leading-tight">{role}</span>
          </div>
        </div>
        
        {/* Status Indicator */}
        <div className={cn(
            "h-2 w-2 rounded-full",
            isActive && "bg-blue-500 animate-pulse",
            status === "complete" && "bg-green-500",
            status === "error" && "bg-red-500",
            status === "idle" && "bg-muted"
        )} />
      </div>

      {maxIterations > 0 && (
        <div className="mt-3 space-y-1.5">
          <div className="flex justify-between text-[10px] uppercase text-muted-foreground">
             <span>Progress</span>
             <span>{Math.min(100, Math.round(progress))}%</span>
          </div>
          <div className="h-1 w-full overflow-hidden rounded-full bg-muted">
            <div 
                className="h-full bg-foreground transition-all duration-300" 
                style={{ width: `${progress}%` }} 
            />
          </div>
        </div>
      )}

      {(score !== undefined || isWinner) && (
        <div className="mt-3 flex items-center justify-between border-t border-border pt-2 text-xs">
           {score !== undefined && (
               <div className="flex gap-1 text-muted-foreground">
                   <span>Confidence:</span>
                   <span className="font-medium text-foreground">{(score * 100).toFixed(0)}%</span>
               </div>
           )}
           {isWinner && (
               <div className="ml-auto rounded-full bg-success/10 px-2 py-0.5 text-[10px] font-medium text-success">
                   Winner
               </div>
           )}
        </div>
      )}
    </div>
  );
}

export const AgentNode = memo(AgentNodeComponent);
export const agentNodeTypes = {
  agent: AgentNode,
};
export default AgentNode;
