/**
 * Custom hook for code panel state management.
 * Handles file tree, file content, sandbox selection, preview, and terminal.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useStore } from "@/lib/store";
import { api, ApiError } from "@/lib/api";
import type { FileNode, SandboxInfo } from "@/lib/types";

/**
 * Return type for the useCodePanel hook.
 */
export interface UseCodePanelReturn {
  sandboxes: SandboxInfo[];
  selectedSandboxId: string | null;
  fileTree: FileNode[];
  openFilePath: string | null;
  fileContent: string | null;
  activeFilePath: string | null;
  previewUrl: string | null;
  previewLoading: boolean;
  previewError: string | null;
  terminalOutput: string[];
  terminalSandboxIds: string[];
  selectedTerminalSandbox: string;
  onTerminalSandboxChange: (sandboxId: string) => void;
  downloadInProgress: boolean;
  downloadError: string | null;
  onSandboxChange: (sandboxId: string) => void;
  onFileSelect: (path: string) => Promise<void>;
  onPreviewRefresh: () => void;
  onTerminalClear: () => void;
  onDownloadProject: () => Promise<void>;
  fileDirty: boolean;
  fileSaving: boolean;
  onEditorChange: (value: string | undefined) => void;
  saveFile: () => Promise<void>;
  refreshFiles: () => Promise<void>;
  startDevServer: () => Promise<void>;
  stopDevServer: () => Promise<void>;
}

/**
 * Hook for managing code panel state including file tree, editor content,
 * sandbox selection, preview, and terminal output.
 */
export function useCodePanel(): UseCodePanelReturn {
  const sessionId = useStore((state) => state.sessionId);
  const sandboxes = useStore((state) => state.sandboxes);
  const selectedSandboxId = useStore(
    (state) => state.selectedSandboxId,
  );
  const fileTree = useStore((state) => state.fileTree);
  const openFilePath = useStore(
    (state) => state.openFilePath,
  );
  const fileContent = useStore(
    (state) => state.fileContent,
  );
  const activeFilePath = useStore(
    (state) => state.activeFilePath,
  );
  const previewUrl = useStore((state) => state.previewUrl);
  const previewLoading = useStore(
    (state) => state.previewLoading,
  );
  const previewError = useStore(
    (state) => state.previewError,
  );
  const terminalOutputs = useStore(
    (state) => state.terminalOutputs,
  );

  const [
    selectedTerminalSandbox,
    setSelectedTerminalSandbox,
  ] = useState("all");

  // Derive sandbox IDs that have terminal output (excluding "all")
  const terminalSandboxIds = useMemo(
    () =>
      Object.keys(terminalOutputs).filter(
        (k) => k !== "all",
      ),
    [terminalOutputs],
  );

  // Compute terminal output for the selected sandbox tab
  const terminalOutput = useMemo(
    () =>
      terminalOutputs[selectedTerminalSandbox] ??
      terminalOutputs.all ??
      [],
    [terminalOutputs, selectedTerminalSandbox],
  );

  const onTerminalSandboxChange = useCallback(
    (sandboxId: string): void => {
      setSelectedTerminalSandbox(sandboxId);
    },
    [],
  );

  const selectSandbox = useStore(
    (state) => state.selectSandbox,
  );
  const selectFile = useStore((state) => state.selectFile);
  const setFileContent = useStore(
    (state) => state.setFileContent,
  );

  // Request ID to prevent stale file content from overwriting newer selections
  const fileRequestIdRef = useRef(0);
  const [downloadInProgress, setDownloadInProgress] =
    useState(false);
  const [downloadError, setDownloadError] = useState<
    string | null
  >(null);
  const setFileTree = useStore(
    (state) => state.setFileTree,
  );
  const setPreviewUrl = useStore(
    (state) => state.setPreviewUrl,
  );
  const setPreviewLoading = useStore(
    (state) => state.setPreviewLoading,
  );
  const setPreviewError = useStore(
    (state) => state.setPreviewError,
  );
  const clearTerminal = useStore(
    (state) => state.clearTerminal,
  );

  // Track whether the editor has unsaved local changes
  const [fileDirty, setFileDirty] = useState(false);
  const [fileSaving, setFileSaving] = useState(false);

  // Reset dirty flag when a different file is selected
  const prevFilePathRef = useRef(openFilePath);
  useEffect(() => {
    if (openFilePath !== prevFilePathRef.current) {
      prevFilePathRef.current = openFilePath;
      setFileDirty(false);
    }
  }, [openFilePath]);

  /**
   * Handle sandbox selection change.
   */
  const onSandboxChange = useCallback(
    (sandboxId: string): void => {
      selectSandbox(sandboxId);
    },
    [selectSandbox],
  );

  /**
   * Handle file selection - loads file content from the API.
   */
  const onFileSelect = useCallback(
    async (path: string): Promise<void> => {
      if (!sessionId) {
        return;
      }

      // Update selected file in store
      selectFile(path);

      // Increment request ID so older in-flight responses are ignored
      const requestId = ++fileRequestIdRef.current;

      // Load file content from API
      try {
        const content = await api.getFileContent(
          sessionId,
          path,
          selectedSandboxId ?? undefined,
        );
        // Only apply if this is still the latest request
        if (requestId === fileRequestIdRef.current) {
          setFileContent(content);
        }
      } catch (error) {
        if (requestId === fileRequestIdRef.current) {
          // Suppress 404s from stale/cleaned-up sessions
          if (
            error instanceof ApiError &&
            error.status === 404
          ) {
            return;
          }
          console.error(
            "Failed to load file content:",
            error,
          );
          setFileContent(null);
        }
      }
    },
    [
      sessionId,
      selectedSandboxId,
      selectFile,
      setFileContent,
    ],
  );

  /**
   * Refresh the preview.
   * The iframe reload itself is handled inside PreviewPane; this clears stale error state.
   */
  const onPreviewRefresh = useCallback((): void => {
    setPreviewError(null);
  }, [setPreviewError]);

  /**
   * Clear the terminal output.
   */
  const onTerminalClear = useCallback((): void => {
    clearTerminal();
  }, [clearTerminal]);

  /**
   * Download the current sandbox workspace as an archive.
   */
  const onDownloadProject =
    useCallback(async (): Promise<void> => {
      if (!sessionId || downloadInProgress) {
        return;
      }

      setDownloadError(null);
      setDownloadInProgress(true);

      try {
        const { blob, filename } =
          await api.downloadProjectArchive(
            sessionId,
            selectedSandboxId ?? undefined,
          );

        const blobUrl = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = blobUrl;
        anchor.download = filename;
        anchor.style.display = "none";
        document.body.append(anchor);
        anchor.click();
        anchor.remove();
        URL.revokeObjectURL(blobUrl);
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : "Failed to export project";
        setDownloadError(message);
        console.error(
          "Failed to export project archive:",
          error,
        );
      } finally {
        setDownloadInProgress(false);
      }
    }, [sessionId, selectedSandboxId, downloadInProgress]);

  /**
   * Refresh the file tree from the API.
   */
  const refreshFiles =
    useCallback(async (): Promise<void> => {
      if (!sessionId) {
        return;
      }

      try {
        const files = await api.getFiles(
          sessionId,
          selectedSandboxId ?? undefined,
        );
        setFileTree(files);
      } catch (error) {
        // Suppress 404s from stale/cleaned-up sessions
        if (
          error instanceof ApiError &&
          error.status === 404
        ) {
          return;
        }
        console.error("Failed to refresh files:", error);
      }
    }, [sessionId, selectedSandboxId, setFileTree]);

  // Auto-refresh file tree when session or sandbox changes
  useEffect(() => {
    void refreshFiles();
  }, [refreshFiles]);

  /**
   * Start the dev server manually.
   * Sets loading state, then updates previewUrl from the API response.
   */
  const startDevServer =
    useCallback(async (): Promise<void> => {
      if (!sessionId) return;
      setPreviewLoading(true);
      setPreviewError(null);
      try {
        const result = await api.manageDevServer(
          sessionId,
          "start",
          selectedSandboxId ?? undefined,
        );
        if (result.url) {
          setPreviewUrl(result.url);
        } else {
          setPreviewLoading(false);
        }
      } catch (error) {
        const message =
          error instanceof Error
            ? error.message
            : "Failed to start dev server";
        setPreviewError(message);
        console.error("Failed to start dev server:", error);
      }
    }, [sessionId, selectedSandboxId, setPreviewLoading, setPreviewError, setPreviewUrl]);

  /**
   * Stop the dev server manually.
   * Clears the preview URL on success so the UI shows the empty state.
   */
  const stopDevServer =
    useCallback(async (): Promise<void> => {
      if (!sessionId) return;
      try {
        await api.manageDevServer(
          sessionId,
          "stop",
          selectedSandboxId ?? undefined,
        );
        setPreviewUrl(null);
      } catch (error) {
        console.error("Failed to stop dev server:", error);
      }
    }, [sessionId, selectedSandboxId, setPreviewUrl]);

  /**
   * Handle editor content changes (user typing).
   * Updates store content and marks the file as dirty.
   */
  const onEditorChange = useCallback(
    (value: string | undefined): void => {
      setFileContent(value ?? null);
      setFileDirty(true);
    },
    [setFileContent],
  );

  /**
   * Save the currently open file to the sandbox.
   */
  const saveFile = useCallback(async (): Promise<void> => {
    if (!sessionId || !openFilePath || fileContent === null) return;
    setFileSaving(true);
    try {
      await api.saveFileContent(
        sessionId,
        openFilePath,
        fileContent,
        selectedSandboxId ?? undefined,
      );
      setFileDirty(false);
    } catch (error) {
      console.error("Failed to save file:", error);
    } finally {
      setFileSaving(false);
    }
  }, [sessionId, openFilePath, fileContent, selectedSandboxId]);

  return {
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
    fileDirty,
    fileSaving,
    downloadInProgress,
    downloadError,
    onSandboxChange,
    onFileSelect,
    onEditorChange,
    saveFile,
    onPreviewRefresh,
    onTerminalClear,
    onDownloadProject,
    refreshFiles,
    startDevServer,
    stopDevServer,
  };
}
