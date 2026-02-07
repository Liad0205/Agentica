"use client";

/**
 * Main workspace page for the Agentica application.
 * Integrates all components with hooks connected to the Zustand store.
 * Uses adaptive layout: collapsible sidebar, primary code panel, bottom graph.
 */

import * as React from "react";
import { Header } from "@/components/layout/Header";
import { WorkspaceLayout } from "@/components/layout/WorkspaceLayout";
import { PromptBar } from "@/components/layout/PromptBar";
import { MessagePanel } from "@/components/messages/MessagePanel";
import { AgentGraph } from "@/components/graph/AgentGraph";
import { CodePanel } from "@/components/code/CodePanel";
import { ModeComparisonPanel } from "@/components/mode/ModeComparisonPanel";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import {
  useSession,
  useMessages,
  useAgentGraph,
  useCodePanel,
  usePrompt,
} from "@/hooks";

export default function HomePage(): React.ReactElement {
  const [isComparisonOpen, setIsComparisonOpen] = React.useState(false);
  const [isGraphVisible, setIsGraphVisible] = React.useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = React.useState(false);

  // Session state and actions
  const {
    mode,
    setMode,
    isRunning,
    status: sessionStatus,
    sessionId,
    recentSessions,
    switchSession,
    resetSession,
    clearSessions,
  } = useSession({ withSessionList: true });

  // Messages and agents
  const { events, agentList } = useMessages();

  // Graph visualization props
  const {
    agentInfo,
    subtasks,
    agentStatuses,
    orchestratorStatus,
    aggregatorStatus,
    integrationStatus,
    numSolvers,
    solverStatuses,
    evaluatorStatus,
    synthesizerStatus,
    selectedSolverId,
    scores,
    evaluationScores,
    evaluationReasoning,
  } = useAgentGraph();

  // Code panel props
  const {
    sandboxes,
    selectedSandboxId,
    fileTree,
    openFilePath,
    fileContent,
    activeFilePath,
    previewUrl,
    previewLoading,
    previewError,
    terminalOutput,
    terminalSandboxIds,
    selectedTerminalSandbox,
    onTerminalSandboxChange,
    downloadInProgress,
    downloadError,
    onSandboxChange,
    onFileSelect,
    onPreviewRefresh,
    onTerminalClear,
    onDownloadProject,
    fileDirty,
    fileSaving,
    onEditorChange,
    saveFile,
    refreshFiles,
    startDevServer,
    stopDevServer,
  } = useCodePanel();

  // Prompt input props
  const { value, onChange, onSubmit, onCancel, canSubmit } = usePrompt();

  // Toggle callbacks
  const toggleGraph = React.useCallback((): void => {
    setIsGraphVisible((prev) => !prev);
  }, []);

  const toggleSidebar = React.useCallback((): void => {
    setIsSidebarCollapsed((prev) => !prev);
  }, []);

  // Keyboard shortcuts
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent): void => {
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key === "b") {
        e.preventDefault();
        toggleSidebar();
      }
      if (mod && e.key === "g") {
        e.preventDefault();
        toggleGraph();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [toggleSidebar, toggleGraph]);

  // Memoize panels to prevent unnecessary re-renders when prompt input changes
  const sidebarContent = React.useMemo(
    () => (
      <ErrorBoundary>
        <MessagePanel
          events={events}
          agents={agentList}
        />
      </ErrorBoundary>
    ),
    [events, agentList],
  );

  const graphContent = React.useMemo(
    () => (
      <ErrorBoundary>
        <AgentGraph
          mode={mode}
          agentInfo={agentInfo ?? undefined}
          subtasks={subtasks}
          agentStatuses={agentStatuses}
          orchestratorStatus={orchestratorStatus}
          aggregatorStatus={aggregatorStatus}
          integrationStatus={integrationStatus}
          numSolvers={numSolvers}
          solverStatuses={solverStatuses}
          evaluatorStatus={evaluatorStatus}
          synthesizerStatus={synthesizerStatus}
          selectedSolverId={selectedSolverId ?? undefined}
          scores={scores}
          evaluationScores={evaluationScores}
          evaluationReasoning={evaluationReasoning ?? undefined}
          compact={isGraphVisible}
        />
      </ErrorBoundary>
    ),
    [
      mode,
      agentInfo,
      subtasks,
      agentStatuses,
      orchestratorStatus,
      aggregatorStatus,
      integrationStatus,
      numSolvers,
      solverStatuses,
      evaluatorStatus,
      synthesizerStatus,
      selectedSolverId,
      scores,
      evaluationScores,
      evaluationReasoning,
      isGraphVisible,
    ],
  );

  const mainContent = React.useMemo(
    () => (
      <ErrorBoundary>
        <CodePanel
          sandboxes={sandboxes}
          selectedSandboxId={selectedSandboxId ?? undefined}
          onSandboxChange={onSandboxChange}
          fileTree={fileTree}
          openFilePath={openFilePath}
          fileContent={fileContent}
          onFileSelect={onFileSelect}
          onFilesRefresh={() => void refreshFiles()}
          onDownloadProject={() => void onDownloadProject()}
          downloadInProgress={downloadInProgress}
          downloadError={downloadError}
          canDownloadProject={Boolean(sessionId)}
          previewUrl={previewUrl}
          previewLoading={previewLoading}
          previewError={previewError ?? undefined}
          onPreviewRefresh={onPreviewRefresh}
          terminalOutput={terminalOutput}
          terminalSandboxIds={terminalSandboxIds}
          selectedTerminalSandbox={selectedTerminalSandbox}
          onTerminalSandboxChange={onTerminalSandboxChange}
          onTerminalClear={onTerminalClear}
          onStartDevServer={startDevServer}
          onStopDevServer={stopDevServer}
          onEditorChange={onEditorChange}
          onSaveFile={saveFile}
          fileDirty={fileDirty}
          fileSaving={fileSaving}
          activeFilePath={activeFilePath ?? undefined}
        />
      </ErrorBoundary>
    ),
    [
      sandboxes,
      selectedSandboxId,
      onSandboxChange,
      fileTree,
      openFilePath,
      fileContent,
      onFileSelect,
      refreshFiles,
      onDownloadProject,
      downloadInProgress,
      downloadError,
      sessionId,
      previewUrl,
      previewLoading,
      previewError,
      onPreviewRefresh,
      terminalOutput,
      terminalSandboxIds,
      selectedTerminalSandbox,
      onTerminalSandboxChange,
      onTerminalClear,
      startDevServer,
      stopDevServer,
      onEditorChange,
      saveFile,
      fileDirty,
      fileSaving,
      activeFilePath,
    ],
  );

  const promptBar = React.useMemo(
    () => (
      <PromptBar
        value={value}
        onChange={onChange}
        onSubmit={onSubmit}
        onCancel={onCancel}
        isRunning={isRunning}
        isMinimized={isRunning}
        disabled={!canSubmit && !isRunning}
        placeholder="Describe what you want to build..."
      />
    ),
    [value, onChange, onSubmit, onCancel, isRunning, canSubmit],
  );

  return (
    <main className="flex h-screen flex-col">
      {/* Header with mode selector and toggles */}
      <Header
        selectedMode={mode}
        onModeChange={setMode}
        sessions={recentSessions}
        activeSessionId={sessionId}
        onSessionSelect={(nextSessionId) => void switchSession(nextSessionId)}
        onNewSession={resetSession}
        onClearSessions={() => void clearSessions(true)}
        onCompareClick={() => setIsComparisonOpen(true)}
        disabled={isRunning}
        onToggleSidebar={toggleSidebar}
        onToggleGraph={toggleGraph}
        isSidebarCollapsed={isSidebarCollapsed}
        isGraphVisible={isGraphVisible}
      />

      {/* Adaptive workspace layout */}
      <WorkspaceLayout
        sidebar={sidebarContent}
        mainContent={mainContent}
        bottomPanel={graphContent}
        promptBar={promptBar}
        isGraphVisible={isGraphVisible}
        onToggleGraph={toggleGraph}
        isSidebarCollapsed={isSidebarCollapsed}
        onToggleSidebar={toggleSidebar}
        eventCount={events.length}
        sessionStatus={sessionStatus}
        mode={mode}
      />

      <ModeComparisonPanel
        open={isComparisonOpen}
        onClose={() => setIsComparisonOpen(false)}
      />
    </main>
  );
}
