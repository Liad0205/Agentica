/**
 * Tests for the useCodePanel hook (hooks/useCodePanel.ts).
 * Covers file selection race condition protection and sandbox changes.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { useCodePanel } from "@/hooks/useCodePanel";
import { useStore } from "@/lib/store";
import { api } from "@/lib/api";

// Mock the API module
vi.mock("@/lib/api", () => ({
  api: {
    getFileContent: vi.fn(),
    getFiles: vi.fn(),
    downloadProjectArchive: vi.fn(),
  },
  ApiError: class ApiError extends Error {
    status: number;
    constructor(message: string, status: number) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  },
}));

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(console, "error").mockImplementation(() => {});
  useStore.getState().resetSession();
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// onFileSelect - race condition protection
// ---------------------------------------------------------------------------
describe("onFileSelect race condition protection", () => {
  it("only applies the result of the most recent file selection", async () => {
    // Set up a session so file loading is enabled
    useStore.setState({ sessionId: "sess-1" });

    const mockGetFileContent = vi.mocked(api.getFileContent);

    // First call resolves slowly, second call resolves quickly
    let resolveFirst: (value: string) => void;
    const firstPromise = new Promise<string>((resolve) => {
      resolveFirst = resolve;
    });

    mockGetFileContent
      .mockReturnValueOnce(firstPromise)
      .mockResolvedValueOnce("content of file B");

    const { result } = renderHook(() => useCodePanel());

    // Select file A (slow response)
    await act(async () => {
      // Don't await - let it be in-flight
      void result.current.onFileSelect("fileA.ts");
    });

    // Select file B (fast response) before A resolves
    await act(async () => {
      await result.current.onFileSelect("fileB.ts");
    });

    // File B content should be set
    await waitFor(() => {
      expect(useStore.getState().fileContent).toBe("content of file B");
    });

    // Now resolve file A (stale response)
    await act(async () => {
      resolveFirst!("content of file A (stale)");
    });

    // File content should still be file B's content, not overwritten by stale A
    expect(useStore.getState().fileContent).toBe("content of file B");
  });

  it("does nothing when sessionId is null", async () => {
    useStore.setState({ sessionId: null });

    const { result } = renderHook(() => useCodePanel());

    await act(async () => {
      await result.current.onFileSelect("file.ts");
    });

    expect(api.getFileContent).not.toHaveBeenCalled();
  });

  it("sets file content to null on API error (for latest request only)", async () => {
    useStore.setState({ sessionId: "sess-1" });
    vi.mocked(api.getFileContent).mockRejectedValue(new Error("Not found"));

    const { result } = renderHook(() => useCodePanel());

    await act(async () => {
      await result.current.onFileSelect("missing.ts");
    });

    expect(useStore.getState().fileContent).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// refreshFiles
// ---------------------------------------------------------------------------
describe("refreshFiles", () => {
  it("loads file tree from API", async () => {
    useStore.setState({ sessionId: "sess-1" });

    const mockTree = [
      { name: "src", path: "src", type: "directory" as const, children: [] },
    ];
    vi.mocked(api.getFiles).mockResolvedValue(mockTree);

    const { result } = renderHook(() => useCodePanel());

    await act(async () => {
      await result.current.refreshFiles();
    });

    expect(api.getFiles).toHaveBeenCalledWith("sess-1", undefined);
    expect(useStore.getState().fileTree).toEqual(mockTree);
  });

  it("passes selectedSandboxId to API", async () => {
    useStore.setState({
      sessionId: "sess-1",
      selectedSandboxId: "sandbox-99",
    });

    vi.mocked(api.getFiles).mockResolvedValue([]);

    const { result } = renderHook(() => useCodePanel());

    await act(async () => {
      await result.current.refreshFiles();
    });

    expect(api.getFiles).toHaveBeenCalledWith("sess-1", "sandbox-99");
  });

  it("does nothing when sessionId is null", async () => {
    useStore.setState({ sessionId: null });

    const { result } = renderHook(() => useCodePanel());

    await act(async () => {
      await result.current.refreshFiles();
    });

    expect(api.getFiles).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// onSandboxChange
// ---------------------------------------------------------------------------
describe("onSandboxChange", () => {
  it("updates selectedSandboxId in the store", () => {
    const { result } = renderHook(() => useCodePanel());

    act(() => {
      result.current.onSandboxChange("sandbox-new");
    });

    expect(useStore.getState().selectedSandboxId).toBe("sandbox-new");
  });
});

// ---------------------------------------------------------------------------
// onTerminalClear
// ---------------------------------------------------------------------------
describe("onTerminalClear", () => {
  it("clears terminal output", () => {
    useStore.setState({ terminalOutputs: { all: ["line 1", "line 2"] } });

    const { result } = renderHook(() => useCodePanel());

    act(() => {
      result.current.onTerminalClear();
    });

    expect(result.current.terminalOutput).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// onDownloadProject
// ---------------------------------------------------------------------------
describe("onDownloadProject", () => {
  it("downloads archive when session is available", async () => {
    useStore.setState({ sessionId: "sess-1", selectedSandboxId: "sb-1" });

    if (!("createObjectURL" in URL)) {
      Object.defineProperty(URL, "createObjectURL", {
        configurable: true,
        writable: true,
        value: () => "",
      });
    }
    if (!("revokeObjectURL" in URL)) {
      Object.defineProperty(URL, "revokeObjectURL", {
        configurable: true,
        writable: true,
        value: () => {},
      });
    }

    const createObjectURL = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:archive");
    const revokeObjectURL = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {});
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    vi.mocked(api.downloadProjectArchive).mockResolvedValue({
      blob: new Blob(["archive"]),
      filename: "project.tar",
    });

    const { result } = renderHook(() => useCodePanel());

    await act(async () => {
      await result.current.onDownloadProject();
    });

    expect(api.downloadProjectArchive).toHaveBeenCalledWith("sess-1", "sb-1");
    expect(createObjectURL).toHaveBeenCalledOnce();
    expect(clickSpy).toHaveBeenCalledOnce();
    expect(revokeObjectURL).toHaveBeenCalledOnce();
    expect(result.current.downloadError).toBeNull();
    expect(result.current.downloadInProgress).toBe(false);

    createObjectURL.mockRestore();
    revokeObjectURL.mockRestore();
    clickSpy.mockRestore();
  });

  it("does nothing when session is missing", async () => {
    useStore.setState({ sessionId: null });

    const { result } = renderHook(() => useCodePanel());

    await act(async () => {
      await result.current.onDownloadProject();
    });

    expect(api.downloadProjectArchive).not.toHaveBeenCalled();
  });

  it("sets error state when archive export fails", async () => {
    useStore.setState({ sessionId: "sess-1" });
    vi.mocked(api.downloadProjectArchive).mockRejectedValue(
      new Error("Export failed")
    );

    const { result } = renderHook(() => useCodePanel());

    await act(async () => {
      await result.current.onDownloadProject();
    });

    expect(result.current.downloadError).toBe("Export failed");
    expect(result.current.downloadInProgress).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Return values from store
// ---------------------------------------------------------------------------
describe("return values", () => {
  it("returns file-related state from store", () => {
    useStore.setState({
      fileTree: [{ name: "file.ts", path: "file.ts", type: "file" }],
      openFilePath: "file.ts",
      fileContent: "const x = 1;",
      activeFilePath: "file.ts",
      previewUrl: "http://localhost:5173",
      previewLoading: false,
      previewError: null,
      terminalOutputs: { all: ["$ npm start"] },
    });

    const { result } = renderHook(() => useCodePanel());

    expect(result.current.fileTree).toHaveLength(1);
    expect(result.current.openFilePath).toBe("file.ts");
    expect(result.current.fileContent).toBe("const x = 1;");
    expect(result.current.activeFilePath).toBe("file.ts");
    expect(result.current.previewUrl).toBe("http://localhost:5173");
    expect(result.current.previewLoading).toBe(false);
    expect(result.current.previewError).toBeNull();
    expect(result.current.terminalOutput).toEqual(["$ npm start"]);
  });
});
