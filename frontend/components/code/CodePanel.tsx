"use client";

/**
 * CodePanel - Main container for the code workspace area.
 * Contains tabs for Code (file tree + editor), Preview (iframe), and Terminal.
 * File tree sidebar is now resizable via react-resizable-panels.
 */

import * as React from "react";
import {
  Panel,
  PanelGroup,
  PanelResizeHandle,
} from "react-resizable-panels";
import { cn } from "@/lib/utils";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { FileTree } from "./FileTree";
import { CodeEditor } from "./CodeEditor";
import { PreviewPane } from "./PreviewPane";
import { TerminalPane } from "./TerminalPane";
import type { FileNode, SandboxInfo } from "@/lib/types";

type TabType = "code" | "preview" | "terminal";

interface CodePanelProps {
  sandboxes?: SandboxInfo[];
  selectedSandboxId?: string;
  onSandboxChange?: (sandboxId: string) => void;
  fileTree: FileNode[];
  openFilePath: string | null;
  fileContent: string | null;
  onFileSelect: (path: string) => void;
  onFilesRefresh?: () => void;
  onDownloadProject?: () => void | Promise<void>;
  downloadInProgress?: boolean;
  downloadError?: string | null;
  canDownloadProject?: boolean;
  previewUrl: string | null;
  previewLoading?: boolean;
  previewError?: string | null;
  onPreviewRefresh?: () => void;
  terminalOutput: string[];
  terminalSandboxIds?: string[];
  selectedTerminalSandbox?: string;
  onTerminalSandboxChange?: (sandboxId: string) => void;
  onTerminalClear?: () => void;
  onStartDevServer?: () => void;
  onStopDevServer?: () => void;
  onEditorChange?: (value: string | undefined) => void;
  onSaveFile?: () => Promise<void>;
  fileDirty?: boolean;
  fileSaving?: boolean;
  activeFilePath?: string | null;
  className?: string;
}

export function CodePanel({
  sandboxes = [],
  selectedSandboxId,
  onSandboxChange,
  fileTree,
  openFilePath,
  fileContent,
  onFileSelect,
  onFilesRefresh,
  onDownloadProject,
  downloadInProgress = false,
  downloadError = null,
  canDownloadProject = true,
  previewUrl,
  previewLoading = false,
  previewError,
  onPreviewRefresh,
  terminalOutput,
  terminalSandboxIds,
  selectedTerminalSandbox,
  onTerminalSandboxChange,
  onTerminalClear,
  onStartDevServer,
  onStopDevServer,
  onEditorChange,
  onSaveFile,
  fileDirty = false,
  fileSaving = false,
  activeFilePath,
  className,
}: CodePanelProps): React.ReactElement {
  const [activeTab, setActiveTab] = React.useState<TabType>("code");
  const hasFiles = fileTree.length > 0;

  // Auto-start dev server when Preview tab is selected and server isn't running
  React.useEffect(() => {
    if (activeTab === "preview" && !previewUrl && !previewLoading && !previewError) {
      onStartDevServer?.();
    }
  }, [activeTab, previewUrl, previewLoading, previewError, onStartDevServer]);

  const tabs: {
    id: TabType;
    label: string;
    icon: string;
  }[] = [
    { id: "code", label: "Code", icon: "code" },
    { id: "preview", label: "Preview", icon: "eye" },
    { id: "terminal", label: "Terminal", icon: "terminal" },
  ];

  const showSandboxSelector = sandboxes.length > 1;

  return (
    <div
      className={cn(
        "flex h-full flex-col bg-background",
        className,
      )}
    >
      {/* Header with tabs and sandbox selector */}
      <div className="flex items-center justify-between border-b border-border bg-card/70 px-2 py-1.5">
        {/* Tab buttons */}
        <div className="flex items-center gap-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              aria-pressed={activeTab === tab.id}
              className={cn(
                "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                activeTab === tab.id
                  ? "bg-card text-accent"
                  : "text-foreground-muted hover:bg-card/50 hover:text-foreground",
              )}
            >
              <TabIcon type={tab.icon} />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Active file indicator */}
        {activeTab === "code" && openFilePath && (
          <div className="flex items-center gap-2 text-xs text-foreground-muted min-w-0 px-2">
            <span
              className="truncate text-accent"
              title={openFilePath}
            >
              {openFilePath}
            </span>
          </div>
        )}

        <div className="flex items-center gap-2">
          {/* Sandbox selector (for multi-sandbox modes) */}
          {showSandboxSelector && (
            <>
              <label
                htmlFor="sandbox-select"
                className="text-xs text-foreground-muted"
              >
                Sandbox:
              </label>
              <select
                id="sandbox-select"
                value={selectedSandboxId ?? ""}
                onChange={(e) =>
                  onSandboxChange?.(e.target.value)
                }
                className="rounded-md border border-border bg-card px-2 py-1 text-xs text-foreground focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
              >
                {sandboxes.map((sandbox) => (
                  <option
                    key={sandbox.id}
                    value={sandbox.id}
                  >
                    {sandbox.id}
                  </option>
                ))}
              </select>
            </>
          )}

          {/* Action buttons based on active tab */}
          {activeTab === "terminal" && onTerminalClear && (
            <div className="flex items-center gap-1">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={onTerminalClear}
                className="h-7 w-7 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                title="Clear Terminal"
                aria-label="Clear Terminal"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          )}

          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => void onDownloadProject?.()}
            disabled={
              !canDownloadProject || downloadInProgress
            }
            aria-label="Download project archive"
            className="flex-shrink-0"
          >
            {downloadInProgress
              ? "Exporting..."
              : "Download"}
          </Button>
        </div>
      </div>

      {downloadError && (
        <div className="border-b border-border/80 bg-error/10 px-3 py-1 text-xs text-error">
          {downloadError}
        </div>
      )}

      {/* Tab content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === "code" &&
          (hasFiles ? (
            <PanelGroup direction="horizontal" autoSaveId="code-panel-filetree">
              {/* Resizable file tree sidebar */}
              <Panel
                defaultSize={22}
                minSize={12}
                maxSize={40}
                collapsible
                collapsedSize={0}
                className="overflow-y-auto border-r border-border"
              >
                <FileTree
                  files={fileTree}
                  selectedPath={openFilePath}
                  onSelect={onFileSelect}
                  activeFilePath={activeFilePath}
                  onRefresh={onFilesRefresh}
                />
              </Panel>

              <PanelResizeHandle className="w-[2px] bg-border/40 transition-colors hover:bg-accent/70" />

              {/* Code editor */}
              <Panel defaultSize={78} minSize={40}>
                <CodeEditor
                  filePath={openFilePath}
                  content={fileContent}
                  readOnly={false}
                  onChange={onEditorChange}
                  onSave={onSaveFile}
                  dirty={fileDirty}
                  saving={fileSaving}
                />
              </Panel>
            </PanelGroup>
          ) : (
            <EmptyCodeState
              onRefresh={onFilesRefresh}
              onOpenPreview={
                previewUrl
                  ? () => setActiveTab("preview")
                  : undefined
              }
              onOpenTerminal={() =>
                setActiveTab("terminal")
              }
            />
          ))}

        {activeTab === "preview" && (
          <PreviewPane
            url={previewUrl}
            loading={previewLoading}
            error={previewError}
            onRefresh={onPreviewRefresh}
            onStartDevServer={onStartDevServer}
            onStopDevServer={onStopDevServer}
          />
        )}

        {activeTab === "terminal" && (
          <TerminalPane
            output={terminalOutput}
            onClear={onTerminalClear}
            sandboxIds={terminalSandboxIds}
            activeSandboxId={selectedTerminalSandbox}
            onSandboxChange={onTerminalSandboxChange}
          />
        )}
      </div>
    </div>
  );
}

/**
 * Simple icon component for tabs
 */
function TabIcon({
  type,
}: {
  type: string;
}): React.ReactElement {
  switch (type) {
    case "code":
      return (
        <svg
          className="h-4 w-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
          />
        </svg>
      );
    case "eye":
      return (
        <svg
          className="h-4 w-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
          />
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
          />
        </svg>
      );
    case "terminal":
      return (
        <svg
          className="h-4 w-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
      );
    default:
      return <span className="h-4 w-4" />;
  }
}

export default CodePanel;

interface EmptyCodeStateProps {
  onRefresh?: () => void;
  onOpenPreview?: () => void;
  onOpenTerminal: () => void;
}

function EmptyCodeState({
  onRefresh,
  onOpenPreview,
  onOpenTerminal,
}: EmptyCodeStateProps): React.ReactElement {
  return (
    <div className="flex h-full items-center justify-center bg-card/20 p-6">
      <div className="w-full max-w-sm rounded-xl border border-border/80 bg-background/70 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-foreground">
          Workspace is warming up
        </h3>
        <p className="mt-2 text-xs text-muted-foreground">
          Files will appear here once agents create or
          modify the project.
        </p>
        <div className="mt-4 flex flex-wrap gap-2">
          {onRefresh && (
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={() => onRefresh()}
            >
              Refresh Files
            </Button>
          )}
          {onOpenPreview && (
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={onOpenPreview}
            >
              Open Preview
            </Button>
          )}
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={onOpenTerminal}
          >
            Open Terminal
          </Button>
        </div>
      </div>
    </div>
  );
}
