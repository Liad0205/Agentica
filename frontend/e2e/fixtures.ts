/**
 * Shared fixtures and helpers for E2E tests.
 *
 * Provides:
 * - Page object model for the main workspace
 * - Helper to check backend availability
 * - Helper to wait for WebSocket connection
 */

import { type Page, type Locator } from "@playwright/test";

const BACKEND_URL = "http://localhost:8000";

/**
 * Check if the backend API is available by hitting the health endpoint.
 * Returns true if the backend responds with 200 OK.
 */
export async function isBackendAvailable(): Promise<boolean> {
  try {
    const response = await fetch(`${BACKEND_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Page object model for the main Agentica workspace.
 * Encapsulates locators and common interactions.
 */
export class WorkspacePage {
  readonly page: Page;

  // -- Header --
  readonly header: Locator;
  readonly logo: Locator;
  readonly modeSelector: Locator;
  readonly reactModeButton: Locator;
  readonly decompositionModeButton: Locator;
  readonly hypothesisModeButton: Locator;
  readonly settingsButton: Locator;
  readonly compareButton: Locator;

  // -- Panels --
  readonly leftPanel: Locator;
  readonly centerPanel: Locator;
  readonly rightPanel: Locator;
  readonly resizeHandles: Locator;

  // -- Prompt Bar --
  readonly promptBar: Locator;
  readonly promptInput: Locator;
  readonly executeButton: Locator;
  readonly cancelButton: Locator;

  // -- Left Panel (Session Log) --
  readonly sessionLogTitle: Locator;

  // -- Right Panel (Code Panel) --
  readonly codeTab: Locator;
  readonly previewTab: Locator;
  readonly terminalTab: Locator;

  constructor(page: Page) {
    this.page = page;

    // Header
    this.header = page.locator("header");
    this.logo = page.getByText("AGENT ARENA");
    this.modeSelector = page.getByRole("radiogroup", {
      name: "Select agent mode",
    });
    this.reactModeButton = page.getByRole("radio", { name: /ReAct/ });
    this.decompositionModeButton = page.getByRole("radio", {
      name: /Decomposition/,
    });
    this.hypothesisModeButton = page.getByRole("radio", {
      name: /Hypothesis/,
    });
    this.settingsButton = page.getByRole("button", { name: "Settings" });
    this.compareButton = page.getByRole("button", { name: /Compare/ });

    // Panels (react-resizable-panels uses data-panel-id)
    this.leftPanel = page.locator('[data-panel-id="left-panel"]');
    this.centerPanel = page.locator('[data-panel-id="center-panel"]');
    this.rightPanel = page.locator('[data-panel-id="right-panel"]');
    this.resizeHandles = page.locator('[data-panel-resize-handle-id]');

    // Prompt Bar
    this.promptBar = page.locator("textarea[aria-label='Task prompt']").locator("..");
    this.promptInput = page.getByLabel("Task prompt");
    this.executeButton = page.getByRole("button", { name: "Execute task" });
    this.cancelButton = page.getByRole("button", { name: "Cancel task" });

    // Session Log
    this.sessionLogTitle = page.getByText("Session Log");

    // Code Panel tabs
    this.codeTab = page.getByRole("button", { name: "Code" });
    this.previewTab = page.getByRole("button", { name: "Preview" });
    this.terminalTab = page.getByRole("button", { name: "Terminal" });
  }

  /**
   * Navigate to the workspace page and wait for it to load.
   */
  async goto(): Promise<void> {
    await this.page.goto("/");
    // Wait for the header to be visible as a signal the page has loaded
    await this.logo.waitFor({ state: "visible" });
  }

  /**
   * Select a mode using the mode selector radio buttons.
   */
  async selectMode(
    mode: "react" | "decomposition" | "hypothesis"
  ): Promise<void> {
    switch (mode) {
      case "react":
        await this.reactModeButton.click();
        break;
      case "decomposition":
        await this.decompositionModeButton.click();
        break;
      case "hypothesis":
        await this.hypothesisModeButton.click();
        break;
    }
  }

  /**
   * Type a prompt into the prompt bar.
   */
  async typePrompt(text: string): Promise<void> {
    await this.promptInput.fill(text);
  }

  /**
   * Submit the current prompt by clicking the Execute button.
   */
  async submitPrompt(): Promise<void> {
    await this.executeButton.click();
  }

  /**
   * Cancel the currently running session.
   */
  async cancelSession(): Promise<void> {
    await this.cancelButton.click();
  }

  /**
   * Wait for a WebSocket connection to be established.
   * Monitors the page for WebSocket frames.
   * @param timeout Maximum time to wait in milliseconds.
   */
  async waitForWebSocket(timeout = 15_000): Promise<void> {
    await this.page.waitForEvent("websocket", { timeout });
  }
}
