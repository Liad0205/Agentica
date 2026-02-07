"use client";

/**
 * FileTree - Recursive tree component for displaying file structure.
 * Features:
 * - Folder expand/collapse
 * - File icons by extension
 * - Click to open in editor
 * - New file indicator (flash animation when created)
 */

import * as React from "react";
import { cn } from "@/lib/utils";
import { getFileIcon } from "@/lib/constants";
import type { FileNode } from "@/lib/types";
import { sortFileNodes } from "@/lib/utils";

interface FileTreeProps {
  /** File tree data structure */
  files: FileNode[];
  /** Currently selected file path */
  selectedPath: string | null;
  /** Callback when a file is selected */
  onSelect: (path: string) => void;
  /** Path of recently changed file (triggers flash animation) */
  activeFilePath?: string | null;
  /** Optional callback to refresh the file tree */
  onRefresh?: () => void | Promise<void>;
  /** Additional className */
  className?: string;
}

export function FileTree({
  files,
  selectedPath,
  onSelect,
  activeFilePath,
  onRefresh,
  className,
}: FileTreeProps): React.ReactElement {
  // Track expanded folders
  const [expandedPaths, setExpandedPaths] = React.useState<Set<string>>(
    new Set()
  );

  // Track paths that should flash (recently created/modified)
  const [flashingPaths, setFlashingPaths] = React.useState<Set<string>>(
    new Set()
  );

  // Handle flash animation for new/changed files
  React.useEffect(() => {
    if (!activeFilePath) {
      return;
    }

    setFlashingPaths((prev) => {
      if (prev.has(activeFilePath)) {
        return prev;
      }
      const next = new Set(prev);
      next.add(activeFilePath);
      return next;
    });

    // Auto-expand parent directories for the active file path.
    setExpandedPaths((prev) => {
      const parts = activeFilePath.split("/");
      const next = new Set(prev);
      let currentPath = "";
      let changed = false;

      for (let i = 0; i < parts.length - 1; i++) {
        const segment = parts[i];
        if (!segment) {
          continue;
        }
        currentPath = currentPath ? `${currentPath}/${segment}` : segment;
        if (!next.has(currentPath)) {
          next.add(currentPath);
          changed = true;
        }
      }

      return changed ? next : prev;
    });

    // Remove flash after animation completes
    const timer = setTimeout(() => {
      setFlashingPaths((prev) => {
        if (!prev.has(activeFilePath)) {
          return prev;
        }
        const next = new Set(prev);
        next.delete(activeFilePath);
        return next;
      });
    }, 1000);

    return () => clearTimeout(timer);
  }, [activeFilePath]);

  const toggleExpanded = (path: string): void => {
    setExpandedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  const sortedFiles = sortFileNodes(files);

  if (sortedFiles.length === 0) {
    return (
      <div className={cn("p-4 text-center text-sm text-foreground-muted", className)}>
        No files yet
      </div>
    );
  }

  return (
    <div className={cn("py-2", className)}>
      <div className="flex items-center justify-between px-3 pb-2">
        <div className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
          Files
        </div>
        {onRefresh && (
          <button
            type="button"
            onClick={() => void onRefresh()}
            className={cn(
              "rounded p-1 text-foreground-muted transition-colors",
              "hover:bg-card/50 hover:text-foreground"
            )}
            title="Refresh file tree"
            aria-label="Refresh file tree"
          >
            <RefreshIcon className="h-4 w-4" />
          </button>
        )}
      </div>
      <div className="space-y-0.5">
        {sortedFiles.map((node) => (
          <FileTreeNode
            key={node.path}
            node={node}
            depth={0}
            selectedPath={selectedPath}
            expandedPaths={expandedPaths}
            flashingPaths={flashingPaths}
            onSelect={onSelect}
            onToggle={toggleExpanded}
          />
        ))}
      </div>
    </div>
  );
}

interface FileTreeNodeProps {
  node: FileNode;
  depth: number;
  selectedPath: string | null;
  expandedPaths: Set<string>;
  flashingPaths: Set<string>;
  onSelect: (path: string) => void;
  onToggle: (path: string) => void;
}

function FileTreeNode({
  node,
  depth,
  selectedPath,
  expandedPaths,
  flashingPaths,
  onSelect,
  onToggle,
}: FileTreeNodeProps): React.ReactElement {
  const isDirectory = node.type === "directory";
  const isExpanded = expandedPaths.has(node.path);
  const isSelected = selectedPath === node.path;
  const isFlashing = flashingPaths.has(node.path);

  const handleClick = (): void => {
    if (isDirectory) {
      onToggle(node.path);
    } else {
      onSelect(node.path);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent): void => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      handleClick();
    }
  };

  const iconType = getFileIcon(node.name, isDirectory);
  const sortedChildren = node.children ? sortFileNodes(node.children) : [];

  return (
    <div>
      <button
        type="button"
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        className={cn(
          "flex w-full items-center gap-1.5 px-2 py-1 text-left text-sm transition-colors",
          "hover:bg-card/50",
          isSelected && "bg-card text-accent",
          isFlashing && "animate-flash"
        )}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
        aria-expanded={isDirectory ? isExpanded : undefined}
      >
        {/* Expand/collapse chevron for directories */}
        {isDirectory && (
          <ChevronIcon
            expanded={isExpanded}
            className="h-3 w-3 flex-shrink-0 text-foreground-muted"
          />
        )}

        {/* File/folder icon */}
        <FileIcon
          type={iconType}
          isDirectory={isDirectory}
          className={cn(
            "h-4 w-4 flex-shrink-0",
            isDirectory ? "text-warning" : "text-foreground-muted"
          )}
        />

        {/* File name */}
        <span
          className={cn(
            "truncate",
            isSelected ? "text-accent" : "text-foreground"
          )}
        >
          {node.name}
        </span>
      </button>

      {/* Render children if expanded */}
      {isDirectory && isExpanded && sortedChildren.length > 0 && (
        <div>
          {sortedChildren.map((child) => (
            <FileTreeNode
              key={child.path}
              node={child}
              depth={depth + 1}
              selectedPath={selectedPath}
              expandedPaths={expandedPaths}
              flashingPaths={flashingPaths}
              onSelect={onSelect}
              onToggle={onToggle}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Chevron icon for expand/collapse
 */
function ChevronIcon({
  expanded,
  className,
}: {
  expanded: boolean;
  className?: string;
}): React.ReactElement {
  return (
    <svg
      className={cn(
        "transition-transform duration-150",
        expanded && "rotate-90",
        className
      )}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M9 5l7 7-7 7"
      />
    </svg>
  );
}

function RefreshIcon({ className }: { className?: string }): React.ReactElement {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M4 4v6h6M20 20v-6h-6"
      />
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M20 9a8 8 0 00-14.9-3M4 15a8 8 0 0014.9 3"
      />
    </svg>
  );
}

/**
 * File icon component with type-specific icons
 */
function FileIcon({
  type,
  isDirectory,
  className,
}: {
  type: string;
  isDirectory: boolean;
  className?: string;
}): React.ReactElement {
  if (isDirectory) {
    return (
      <svg className={className} fill="currentColor" viewBox="0 0 24 24">
        <path d="M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z" />
      </svg>
    );
  }

  // TypeScript/React files
  if (type === "typescript" || type === "react") {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="currentColor">
        <path d="M3 3h18v18H3V3zm15.17 10.02c-.47-.22-.98-.35-1.64-.35-.58 0-.99.14-1.24.41-.25.27-.38.59-.38 1 0 .38.1.69.3.93.2.24.55.46 1.07.68.44.18.75.34.94.5.19.16.28.36.28.61 0 .24-.08.43-.24.57-.16.14-.4.21-.71.21-.36 0-.66-.08-.89-.23-.24-.16-.4-.38-.5-.67l-1.32.53c.17.5.47.89.89 1.17.42.28.94.42 1.58.42.68 0 1.2-.16 1.58-.47.38-.31.57-.73.57-1.25 0-.43-.11-.78-.34-1.06-.23-.28-.63-.52-1.21-.72-.41-.15-.7-.3-.87-.45-.17-.15-.25-.33-.25-.55 0-.2.07-.36.22-.49.15-.13.36-.19.63-.19.28 0 .52.07.71.22.2.15.35.36.46.63l1.26-.51c-.21-.47-.52-.83-.93-1.08-.41-.24-.89-.36-1.44-.36-.58 0-1.06.15-1.44.44-.38.29-.57.69-.57 1.2 0 .35.1.66.29.91.19.26.55.49 1.06.69.44.17.76.33.95.5.19.17.29.38.29.64zm-6.03.34V15h2.3v-1.64h-2.3v-1.21h2.53V10.5H10.64v6h4.2v-1.64h-2.7v-1.5zm-3.98-3.36h1.55V15h1.5v-4.64h1.55V9h-4.6v1.36z" />
      </svg>
    );
  }

  // JavaScript files
  if (type === "javascript") {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="currentColor">
        <path d="M3 3h18v18H3V3zm4.73 15.04c.4.85 1.19 1.55 2.54 1.55 1.5 0 2.53-.8 2.53-2.55v-5.78h-1.7v5.74c0 .86-.35 1.08-.9 1.08-.58 0-.82-.4-1.09-.87l-1.38.83zm5.98-.18c.5.98 1.51 1.73 3.09 1.73 1.6 0 2.8-.83 2.8-2.36 0-1.41-.81-2.04-2.25-2.66l-.42-.18c-.73-.31-1.04-.52-1.04-1.02 0-.41.31-.73.81-.73.48 0 .8.21 1.09.73l1.31-.87c-.55-.96-1.33-1.33-2.4-1.33-1.51 0-2.48.96-2.48 2.23 0 1.38.81 2.03 2.03 2.55l.42.18c.78.34 1.24.55 1.24 1.13 0 .48-.45.83-1.15.83-.83 0-1.31-.43-1.67-1.03l-1.38.8z" />
      </svg>
    );
  }

  // JSON files
  if (type === "json") {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="currentColor">
        <path d="M5 3h2v2H5v5a2 2 0 01-2 2 2 2 0 012 2v5h2v2H5c-1.07-.27-2-.9-2-2v-4a2 2 0 00-2-2H0v-2h1a2 2 0 002-2V5a2 2 0 012-2m14 0a2 2 0 012 2v4a2 2 0 002 2h1v2h-1a2 2 0 00-2 2v4a2 2 0 01-2 2h-2v-2h2v-5a2 2 0 012-2 2 2 0 01-2-2V5h-2V3h2m-7 12a1 1 0 011 1 1 1 0 01-1 1 1 1 0 01-1-1 1 1 0 011-1m-4 0a1 1 0 011 1 1 1 0 01-1 1 1 1 0 01-1-1 1 1 0 011-1m8 0a1 1 0 011 1 1 1 0 01-1 1 1 1 0 01-1-1 1 1 0 011-1z" />
      </svg>
    );
  }

  // CSS files
  if (type === "css" || type === "sass" || type === "less") {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="currentColor">
        <path d="M5 3l-.65 3.34h13.59L17.5 8.5H3.92l-.66 3.33h13.59l-.76 3.81-5.48 1.81-4.75-1.81.33-1.64H2.85l-.79 4 7.85 3 9.05-3 1.2-6.03.24-1.21L21.94 3H5z" />
      </svg>
    );
  }

  // HTML files
  if (type === "html") {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 17.56l4.07-1.13.55-6.1H9.38L9.2 8.3h7.6l.2-1.99H7L7.56 12h6.89l-.23 2.56L12 15.12l-2.22-.56-.14-1.61H7.64l.28 3.04L12 17.56M4.07 3h15.86L18.5 19.2 12 21l-6.5-1.8L4.07 3z" />
      </svg>
    );
  }

  // Markdown files
  if (type === "markdown") {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="currentColor">
        <path d="M20.56 18H3.44C2.65 18 2 17.37 2 16.59V7.41C2 6.63 2.65 6 3.44 6h17.12c.79 0 1.44.63 1.44 1.41v9.18c0 .78-.65 1.41-1.44 1.41M6.81 15.19v-3.66l1.92 2.35 1.92-2.35v3.66h1.93V8.81h-1.93l-1.92 2.35-1.92-2.35H4.89v6.38h1.92M19.69 12h-1.92V8.81h-1.92V12h-1.93l2.89 3.28L19.69 12z" />
      </svg>
    );
  }

  // Default file icon
  return (
    <svg className={className} fill="currentColor" viewBox="0 0 24 24">
      <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z" />
    </svg>
  );
}

export default FileTree;
