/**
 * Tests for the HTTP API client (lib/api.ts).
 * Mocks global fetch to verify request construction and error handling.
 */

import { describe, it, expect, beforeEach, vi } from "vitest";
import { api, ApiError } from "@/lib/api";

/**
 * Helper to create a mock Response object.
 */
function mockResponse(
  body: unknown,
  options: { status?: number; statusText?: string; contentType?: string } = {}
): Response {
  const { status = 200, statusText = "OK", contentType = "application/json" } = options;
  const bodyStr = typeof body === "string" ? body : JSON.stringify(body);

  return new Response(bodyStr, {
    status,
    statusText,
    headers: {
      "content-type": contentType,
    },
  });
}

beforeEach(() => {
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// createSession
// ---------------------------------------------------------------------------
describe("api.createSession", () => {
  it("sends POST with mode and task in body", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({ session_id: "sess-1", websocket_url: "/ws/sess-1", status: "started" })
    );

    const result = await api.createSession({
      mode: "react",
      task: "build a todo app",
    });

    expect(fetchSpy).toHaveBeenCalledOnce();
    const [url, options] = fetchSpy.mock.calls[0] as [string, RequestInit];

    expect(url).toContain("/api/sessions");
    expect(options.method).toBe("POST");

    const body = JSON.parse(options.body as string) as Record<string, unknown>;
    expect(body.mode).toBe("react");
    expect(body.task).toBe("build a todo app");

    expect(result.session_id).toBe("sess-1");
  });

  it("sends llm_config (not modelConfig) in request body", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({ session_id: "sess-2", websocket_url: "/ws/sess-2", status: "started" })
    );

    await api.createSession({
      mode: "decomposition",
      task: "build app",
      modelConfig: {
        orchestrator: "gpt-4",
        subAgent: "gpt-3.5",
        evaluator: "gpt-4",
        temperature: 0.7,
      },
    });

    const body = JSON.parse(
      (fetchSpy.mock.calls[0] as [string, RequestInit])[1].body as string
    ) as Record<string, unknown>;

    // Should use llm_config (snake_case) for the backend
    expect(body.llm_config).toBeDefined();
    expect(body.modelConfig).toBeUndefined();

    const llmConfig = body.llm_config as Record<string, unknown>;
    expect(llmConfig.orchestrator).toBe("gpt-4");
    expect(llmConfig.sub_agent).toBe("gpt-3.5");
    expect(llmConfig.evaluator).toBe("gpt-4");
    expect(llmConfig.temperature).toBe(0.7);
  });

  it("omits llm_config when modelConfig is not provided", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({ session_id: "sess-3", websocket_url: "/ws/sess-3", status: "started" })
    );

    await api.createSession({
      mode: "react",
      task: "build something",
    });

    const body = JSON.parse(
      (fetchSpy.mock.calls[0] as [string, RequestInit])[1].body as string
    ) as Record<string, unknown>;

    expect(body.llm_config).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------
describe("API error handling", () => {
  it("throws ApiError on non-OK response with JSON detail", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse(
        { detail: "Session not found" },
        { status: 404, statusText: "Not Found" }
      )
    );

    await expect(api.getSession("nonexistent")).rejects.toThrow(ApiError);

    try {
      await api.getSession("nonexistent");
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      const apiError = error as ApiError;
      expect(apiError.status).toBe(404);
      expect(apiError.detail).toBe("Session not found");
      expect(apiError.message).toBe("Session not found");
    }
  });

  it("throws ApiError with status text when no detail in response", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse(
        {},
        { status: 500, statusText: "Internal Server Error" }
      )
    );

    await expect(api.healthCheck()).rejects.toThrow(ApiError);
  });

  it("ApiError has correct name property", () => {
    const error = new ApiError("test", 400, "detail");
    expect(error.name).toBe("ApiError");
    expect(error.status).toBe(400);
    expect(error.detail).toBe("detail");
  });
});

// ---------------------------------------------------------------------------
// getFiles
// ---------------------------------------------------------------------------
describe("api.getFiles", () => {
  it("sends correct query parameters", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse([])
    );

    await api.getFiles("sess-1", "sandbox-42", "/src");

    const [url] = fetchSpy.mock.calls[0] as [string, RequestInit | undefined];
    expect(url).toContain("/api/sessions/sess-1/files");
    expect(url).toContain("sandbox_id=sandbox-42");
    expect(url).toContain("path=%2Fsrc");
  });

  it("uses default sandbox_id and path", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse([])
    );

    await api.getFiles("sess-1");

    const [url] = fetchSpy.mock.calls[0] as [string, RequestInit | undefined];
    expect(url).toContain("sandbox_id=primary");
    expect(url).toContain("path=.");
  });

  it("builds a nested file tree from flat response", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse([
        { name: "src", path: "src", is_directory: true },
        { name: "index.ts", path: "src/index.ts", is_directory: false },
        { name: "utils.ts", path: "src/utils.ts", is_directory: false },
      ])
    );

    const tree = await api.getFiles("sess-1");

    expect(tree).toHaveLength(1);
    expect(tree[0]?.name).toBe("src");
    expect(tree[0]?.type).toBe("directory");
    expect(tree[0]?.children).toHaveLength(2);
    // Children should be sorted alphabetically
    expect(tree[0]?.children?.[0]?.name).toBe("index.ts");
    expect(tree[0]?.children?.[1]?.name).toBe("utils.ts");
  });
});

// ---------------------------------------------------------------------------
// getFileContent
// ---------------------------------------------------------------------------
describe("api.getFileContent", () => {
  it("sends correct query parameters", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse("const x = 1;", { contentType: "text/plain" })
    );

    await api.getFileContent("sess-1", "src/index.ts", "sb-1");

    const [url] = fetchSpy.mock.calls[0] as [string, RequestInit | undefined];
    expect(url).toContain("/api/sessions/sess-1/files/content");
    expect(url).toContain("sandbox_id=sb-1");
    expect(url).toContain("path=src%2Findex.ts");
  });
});

// ---------------------------------------------------------------------------
// downloadProjectArchive
// ---------------------------------------------------------------------------
describe("api.downloadProjectArchive", () => {
  it("requests archive endpoint and parses filename", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("archive-bytes", {
        status: 200,
        headers: {
          "content-type": "application/x-tar",
          "content-disposition":
            'attachment; filename="sess-1-sb-1-workspace.tar"',
        },
      })
    );

    const result = await api.downloadProjectArchive("sess-1", "sb-1", ".");

    const [url] = fetchSpy.mock.calls[0] as [string, RequestInit | undefined];
    expect(url).toContain("/api/sessions/sess-1/files/archive");
    expect(url).toContain("sandbox_id=sb-1");
    expect(url).toContain("path=.");
    expect(result.filename).toBe("sess-1-sb-1-workspace.tar");
    expect(result.blob).toBeInstanceOf(Blob);
  });

  it("throws ApiError on failed archive download", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse(
        { detail: "Sandbox not found" },
        { status: 404, statusText: "Not Found" }
      )
    );

    await expect(api.downloadProjectArchive("sess-1")).rejects.toThrow(ApiError);
  });
});

// ---------------------------------------------------------------------------
// cancelSession
// ---------------------------------------------------------------------------
describe("api.cancelSession", () => {
  it("sends POST to cancel endpoint", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse(undefined, { status: 204 })
    );

    await api.cancelSession("sess-1");

    const [url, options] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/api/sessions/sess-1/cancel");
    expect(options.method).toBe("POST");
  });
});

// ---------------------------------------------------------------------------
// clearSessions
// ---------------------------------------------------------------------------
describe("api.clearSessions", () => {
  it("sends DELETE to sessions endpoint with force query", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({
        message: "All sessions cleared",
        in_memory_cleared: 3,
        running_cleared: 0,
        persisted_cleared: 3,
        forced: true,
      })
    );

    const result = await api.clearSessions(true);

    const [url, options] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/api/sessions?force=true");
    expect(options.method).toBe("DELETE");
    expect(result.message).toBe("All sessions cleared");
    expect(result.persisted_cleared).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// healthCheck
// ---------------------------------------------------------------------------
describe("api.healthCheck", () => {
  it("returns health check response", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      mockResponse({ status: "ok", timestamp: 1700000000 })
    );

    const result = await api.healthCheck();

    expect(result.status).toBe("ok");
    expect(result.timestamp).toBe(1700000000);
  });
});
