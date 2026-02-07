"use client";

/**
 * AnimatedEdge - Custom edge component with animated flow and direction indicators.
 * Shows data flow direction with animated dashes when active.
 */

import * as React from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from "@xyflow/react";
import { COLORS } from "@/lib/constants";

interface AnimatedEdgeData {
  label?: string;
  isWinner?: boolean;
}

export function AnimatedEdge(props: EdgeProps): React.ReactElement {
  const {
    id,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    style,
    markerEnd,
    animated,
    label,
  } = props;
  const data = props.data as AnimatedEdgeData | undefined;
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const isWinner = data?.isWinner ?? false;
  const edgeLabel = label ?? data?.label;

  // Determine edge color based on state
  const edgeColor = isWinner
    ? COLORS.success
    : animated
    ? COLORS.accent
    : COLORS.border;

  return (
    <>
      {/* Main edge path */}
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          ...(style ?? {}),
          stroke: edgeColor,
          strokeWidth: isWinner ? 3 : 2,
          strokeDasharray: animated ? "5,5" : undefined,
          transition: "stroke 0.3s ease, stroke-width 0.3s ease",
        }}
        className={animated ? "animated-edge" : undefined}
      />

      {/* Animated overlay for active edges */}
      {animated && (
        <path
          d={edgePath}
          fill="none"
          stroke={COLORS.accent}
          strokeWidth={2}
          strokeDasharray="5,5"
          className="edge-flow-animation"
          style={{
            animation: "dashFlow 1s linear infinite",
          }}
        />
      )}

      {/* Direction indicator arrow at midpoint */}
      {animated && (
        <circle
          r={4}
          fill={COLORS.accent}
          className="edge-flow-dot"
        >
          <animateMotion
            dur="1.5s"
            repeatCount="indefinite"
            path={edgePath}
          />
        </circle>
      )}

      {/* Edge label */}
      {edgeLabel && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              pointerEvents: "all",
            }}
            className="edge-label"
          >
            <span
              className={`
                px-2 py-0.5 text-xs font-medium rounded-full
                ${isWinner
                  ? "bg-success/20 text-success border border-success/30"
                  : "bg-card text-foreground-muted border border-border"
                }
              `}
            >
              {edgeLabel}
            </span>
          </div>
        </EdgeLabelRenderer>
      )}

    </>
  );
}

// Edge type registration for React Flow
export const animatedEdgeTypes = {
  animated: AnimatedEdge,
};

export default AnimatedEdge;
