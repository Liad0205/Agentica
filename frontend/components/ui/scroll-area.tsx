"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

interface ScrollAreaProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

/**
 * A scrollable container with custom scrollbar styling.
 * Uses the CSS scrollbar styles defined in globals.css.
 * Supports ref forwarding for scroll control.
 */
export const ScrollArea = React.forwardRef<HTMLDivElement, ScrollAreaProps>(
  function ScrollArea({ children, className, ...props }, ref): React.ReactElement {
    return (
      <div
        ref={ref}
        className={cn("overflow-auto", className)}
        {...props}
      >
        {children}
      </div>
    );
  }
);

ScrollArea.displayName = "ScrollArea";

interface ScrollAreaViewportProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

/**
 * Inner viewport for ScrollArea, useful when you need more control.
 */
export const ScrollAreaViewport = React.forwardRef<HTMLDivElement, ScrollAreaViewportProps>(
  function ScrollAreaViewport({ children, className, ...props }, ref): React.ReactElement {
    return (
      <div ref={ref} className={cn("h-full w-full", className)} {...props}>
        {children}
      </div>
    );
  }
);

ScrollAreaViewport.displayName = "ScrollAreaViewport";
