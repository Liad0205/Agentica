/**
 * HTTP API client for communicating with the backend.
 * Provides type-safe wrappers for all REST endpoints.
 */

import { API_ENDPOINTS, BACKEND_URL } from "./constants";
import type {
  CreateSessionResponse,
  FileNode,
  Mode,
  ModelConfig,
  SessionDetailResponse,
  SessionMetrics,
  SessionSummary,
} from "./types";

/**
 * Custom error class for API errors with status code and detail.
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * Backend file info response structure (flat array from API).
 */
interface FileInfo {
  name: string;
  path: string;
  is_directory: boolean;
}

/**
 * Health check response structure.
 */
interface HealthCheckResponse {
  status: string;
  timestamp: number;
}

/**
 * Generic fetch wrapper that handles common API patterns.
 * - Prepends BACKEND_URL to the endpoint
 * - Sets Content-Type to application/json for requests with body
 * - Parses response as JSON
 * - Handles errors by throwing ApiError
 * - Handles 204 No Content responses
 */
async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit,
): Promise<T> {
  const url = `${BACKEND_URL}${endpoint}`;

  const headers: HeadersInit = {
    ...options?.headers,
  };

  // Add Content-Type for requests with a body (unless explicitly set)
  if (options?.body && !(headers as Record<string, string>)["Content-Type"]) {
    (headers as Record<string, string>)["Content-Type"] =
      "application/json";
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T;
  }

  // Parse response body
  let data: unknown;
  const contentType =
    response.headers.get("content-type") ?? "";
  const parseTarget = response.clone();

  if (contentType.includes("application/json")) {
    try {
      data = await parseTarget.json();
    } catch {
      data = {};
    }
  } else {
    // For non-JSON responses (like plain text file content)
    try {
      data = await parseTarget.text();
    } catch {
      data = "";
    }
  }

  // Handle error responses
  if (!response.ok) {
    const errorData = data as
      | { detail?: string; message?: string }
      | undefined;
    const detail = errorData?.detail ?? errorData?.message;
    const message =
      detail ??
      `HTTP ${response.status}: ${response.statusText}`;
    throw new ApiError(message, response.status, detail);
  }

  return data as T;
}

/**
 * Parse a filename from a Content-Disposition header value.
 */
function parseContentDispositionFilename(
  header: string | null,
): string | null {
  if (!header) {
    return null;
  }

  const utf8Match = header.match(
    /filename\*=UTF-8''([^;]+)/i,
  );
  if (utf8Match?.[1]) {
    try {
      return decodeURIComponent(utf8Match[1]);
    } catch {
      return utf8Match[1];
    }
  }

  const quotedMatch = header.match(/filename="([^"]+)"/i);
  if (quotedMatch?.[1]) {
    return quotedMatch[1];
  }

  const unquotedMatch = header.match(/filename=([^;]+)/i);
  return unquotedMatch?.[1]?.trim() ?? null;
}

/**
 * Builds a nested FileNode tree from a flat array of FileInfo objects.
 * Handles proper nesting of directories and files.
 */
function buildFileTree(files: FileInfo[]): FileNode[] {
  // Map to store all nodes by path for quick lookup
  const nodeMap = new Map<string, FileNode>();
  const roots: FileNode[] = [];

  // Sort files to ensure directories are processed before their children
  const sortedFiles = [...files].sort((a, b) => {
    // Sort by path depth (fewer slashes first) then alphabetically
    const depthA = a.path.split("/").length;
    const depthB = b.path.split("/").length;
    if (depthA !== depthB) {
      return depthA - depthB;
    }
    return a.path.localeCompare(b.path);
  });

  for (const file of sortedFiles) {
    const node: FileNode = {
      name: file.name,
      path: file.path,
      type: file.is_directory ? "directory" : "file",
    };

    if (file.is_directory) {
      node.children = [];
    }

    nodeMap.set(file.path, node);

    // Find parent directory path
    const pathParts = file.path.split("/");
    pathParts.pop(); // Remove the file/directory name
    const parentPath = pathParts.join("/");

    if (parentPath && nodeMap.has(parentPath)) {
      // Add to parent's children
      const parent = nodeMap.get(parentPath);
      if (parent?.children) {
        parent.children.push(node);
      }
    } else {
      // No parent found, this is a root node
      roots.push(node);
    }
  }

  // Sort children within each directory: directories first, then files, alphabetically
  function sortChildren(nodes: FileNode[]): void {
    nodes.sort((a, b) => {
      if (a.type !== b.type) {
        return a.type === "directory" ? -1 : 1;
      }
      return a.name.localeCompare(b.name);
    });
    for (const node of nodes) {
      if (node.children) {
        sortChildren(node.children);
      }
    }
  }

  sortChildren(roots);

  return roots;
}

/**
 * API client object with type-safe methods for all backend endpoints.
 */
export const api = {
  /**
   * Create a new session with the specified configuration.
   */
  async createSession(config: {
    mode: Mode;
    task: string;
    modelConfig?: ModelConfig;
  }): Promise<CreateSessionResponse> {
    return fetchApi<CreateSessionResponse>(
      API_ENDPOINTS.sessions,
      {
        method: "POST",
        body: JSON.stringify({
          mode: config.mode,
          task: config.task,
          llm_config: config.modelConfig
            ? {
                orchestrator:
                  config.modelConfig.orchestrator,
                sub_agent: config.modelConfig.subAgent,
                evaluator: config.modelConfig.evaluator,
                temperature: config.modelConfig.temperature,
              }
            : undefined,
        }),
      },
    );
  },

  /**
   * Continue an existing session on the same backend sandbox/session context.
   */
  async continueSession(
    sessionId: string,
    config: {
      task: string;
      modelConfig?: ModelConfig;
    },
  ): Promise<CreateSessionResponse> {
    return fetchApi<CreateSessionResponse>(
      API_ENDPOINTS.sessionContinue(sessionId),
      {
        method: "POST",
        body: JSON.stringify({
          task: config.task,
          llm_config: config.modelConfig
            ? {
                orchestrator:
                  config.modelConfig.orchestrator,
                sub_agent: config.modelConfig.subAgent,
                evaluator: config.modelConfig.evaluator,
                temperature: config.modelConfig.temperature,
              }
            : undefined,
        }),
      },
    );
  },

  /**
   * Get details for an existing session.
   */
  async getSession(
    sessionId: string,
  ): Promise<SessionDetailResponse> {
    return fetchApi<SessionDetailResponse>(
      `${API_ENDPOINTS.sessions}/${sessionId}`,
    );
  },

  /**
   * List recent sessions for comparison.
   */
  async listSessions(
    limit: number = 25,
  ): Promise<SessionSummary[]> {
    const params = new URLSearchParams({
      limit: String(limit),
    });
    return fetchApi<SessionSummary[]>(
      `${API_ENDPOINTS.sessions}?${params.toString()}`,
    );
  },

  /**
   * Clear all sessions from backend memory and persistence.
   */
  async clearSessions(force: boolean = true): Promise<{
    message: string;
    in_memory_cleared: number;
    running_cleared: number;
    persisted_cleared: number;
    forced: boolean;
  }> {
    const params = new URLSearchParams({
      force: String(force),
    });
    return fetchApi<{
      message: string;
      in_memory_cleared: number;
      running_cleared: number;
      persisted_cleared: number;
      forced: boolean;
    }>(`${API_ENDPOINTS.sessions}?${params.toString()}`, {
      method: "DELETE",
    });
  },

  /**
   * Cancel a running session.
   */
  async cancelSession(sessionId: string): Promise<void> {
    await fetchApi<void>(
      API_ENDPOINTS.sessionCancel(sessionId),
      {
        method: "POST",
      },
    );
  },

  /**
   * Get the file tree for a session's sandbox.
   * @param sessionId - The session ID
   * @param sandboxId - The sandbox ID (defaults to 'primary')
   * @param path - The path within the sandbox (defaults to '.')
   */
  async getFiles(
    sessionId: string,
    sandboxId: string = "primary",
    path: string = ".",
  ): Promise<FileNode[]> {
    const params = new URLSearchParams({
      sandbox_id: sandboxId,
      path: path,
    });
    const endpoint = `${API_ENDPOINTS.sessionFiles(sessionId)}?${params.toString()}`;
    const files = await fetchApi<FileInfo[]>(endpoint);
    return buildFileTree(files);
  },

  /**
   * Get the content of a specific file.
   * @param sessionId - The session ID
   * @param filePath - The path to the file within the sandbox
   * @param sandboxId - The sandbox ID (defaults to 'primary')
   */
  async getFileContent(
    sessionId: string,
    filePath: string,
    sandboxId: string = "primary",
  ): Promise<string> {
    const params = new URLSearchParams({
      sandbox_id: sandboxId,
      path: filePath,
    });
    const endpoint = `${API_ENDPOINTS.sessionFiles(sessionId)}/content?${params.toString()}`;
    return fetchApi<string>(endpoint);
  },

  /**
   * Save (write) file content to a sandbox.
   * @param sessionId - The session ID
   * @param filePath - The path to the file within the sandbox
   * @param content - The file content to write
   * @param sandboxId - The sandbox ID (defaults to 'primary')
   */
  async saveFileContent(
    sessionId: string,
    filePath: string,
    content: string,
    sandboxId: string = "primary",
  ): Promise<{ message: string; path: string }> {
    const params = new URLSearchParams({
      sandbox_id: sandboxId,
      path: filePath,
    });
    const endpoint = `${API_ENDPOINTS.sessionFiles(sessionId)}/content?${params.toString()}`;
    return fetchApi<{ message: string; path: string }>(endpoint, {
      method: "PUT",
      headers: { "Content-Type": "text/plain" },
      body: content,
    });
  },

  /**
   * Download a sandbox project archive.
   */
  async downloadProjectArchive(
    sessionId: string,
    sandboxId: string = "primary",
    path: string = ".",
  ): Promise<{ blob: Blob; filename: string }> {
    const params = new URLSearchParams({
      sandbox_id: sandboxId,
      path,
    });
    const endpoint = `${API_ENDPOINTS.sessionFilesArchive(sessionId)}?${params.toString()}`;
    const url = `${BACKEND_URL}${endpoint}`;

    const response = await fetch(url);
    if (!response.ok) {
      const contentType =
        response.headers.get("content-type") ?? "";
      let detail: string | undefined;

      if (contentType.includes("application/json")) {
        try {
          const payload = (await response.json()) as {
            detail?: string;
            message?: string;
          };
          detail = payload.detail ?? payload.message;
        } catch {
          detail = undefined;
        }
      } else {
        try {
          detail = await response.text();
        } catch {
          detail = undefined;
        }
      }

      const message =
        detail?.trim() ||
        `HTTP ${response.status}: ${response.statusText}`;
      throw new ApiError(message, response.status, detail);
    }

    const blob = await response.blob();
    const filename =
      parseContentDispositionFilename(
        response.headers.get("content-disposition"),
      ) ?? `${sessionId}-workspace.tar`;

    return { blob, filename };
  },

  /**
   * Manually control the dev server.
   */
  async manageDevServer(
    sessionId: string,
    action: "start" | "stop",
    sandboxId: string = "primary",
  ): Promise<{
    message: string;
    url?: string;
    status: string;
  }> {
    const params = new URLSearchParams({
      action,
      sandbox_id: sandboxId,
    });
    const endpoint = `${API_ENDPOINTS.sessionSandboxDevServer(
      sessionId,
    )}?${params.toString()}`;
    return fetchApi(endpoint, {
      method: "POST",
    });
  },

  /**
   * Get metrics for a session (token usage, execution time, etc.).
   */
  async getMetrics(
    sessionId: string,
  ): Promise<SessionMetrics> {
    return fetchApi<SessionMetrics>(
      API_ENDPOINTS.sessionMetrics(sessionId),
    );
  },

  /**
   * Check the health of the backend service.
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    return fetchApi<HealthCheckResponse>(
      API_ENDPOINTS.health,
    );
  },
};
