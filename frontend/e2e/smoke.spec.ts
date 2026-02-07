/**
 * Smoke tests for the Agentica workspace.
 *
 * These tests verify the basic UI renders correctly and does not require
 * the backend to be running.
 */

import { test, expect } from "@playwright/test";
import { WorkspacePage } from "./fixtures";

test.describe("Smoke tests", () => {
  let workspace: WorkspacePage;

  test.beforeEach(async ({ page }) => {
    workspace = new WorkspacePage(page);
    await workspace.goto();
  });

  test("page loads successfully", async () => {
    // The page should have a title (Next.js default or custom)
    await expect(workspace.page).toHaveURL("/");

    // The main element should be present
    const main = workspace.page.locator("main");
    await expect(main).toBeVisible();
  });

  test("header renders with logo and mode selector", async () => {
    // Header is visible
    await expect(workspace.header).toBeVisible();

    // Logo text is visible
    await expect(workspace.logo).toBeVisible();
    await expect(workspace.logo).toHaveText("AGENT ARENA");

    // Mode selector is visible with all three modes
    await expect(workspace.modeSelector).toBeVisible();
    await expect(workspace.reactModeButton).toBeVisible();
    await expect(workspace.decompositionModeButton).toBeVisible();
    await expect(workspace.hypothesisModeButton).toBeVisible();
  });

  test("mode selector switches modes", async () => {
    // ReAct should be selected by default
    await expect(workspace.reactModeButton).toHaveAttribute(
      "aria-checked",
      "true"
    );
    await expect(workspace.decompositionModeButton).toHaveAttribute(
      "aria-checked",
      "false"
    );
    await expect(workspace.hypothesisModeButton).toHaveAttribute(
      "aria-checked",
      "false"
    );

    // Click Decomposition
    await workspace.selectMode("decomposition");
    await expect(workspace.decompositionModeButton).toHaveAttribute(
      "aria-checked",
      "true"
    );
    await expect(workspace.reactModeButton).toHaveAttribute(
      "aria-checked",
      "false"
    );

    // Click Hypothesis
    await workspace.selectMode("hypothesis");
    await expect(workspace.hypothesisModeButton).toHaveAttribute(
      "aria-checked",
      "true"
    );
    await expect(workspace.decompositionModeButton).toHaveAttribute(
      "aria-checked",
      "false"
    );

    // Click back to ReAct
    await workspace.selectMode("react");
    await expect(workspace.reactModeButton).toHaveAttribute(
      "aria-checked",
      "true"
    );
  });

  test("three panels are visible", async () => {
    // All three resizable panels should be present and visible
    await expect(workspace.leftPanel).toBeVisible();
    await expect(workspace.centerPanel).toBeVisible();
    await expect(workspace.rightPanel).toBeVisible();

    // The left panel contains the Session Log
    await expect(workspace.sessionLogTitle).toBeVisible();

    // The right panel contains code/preview/terminal tabs
    await expect(workspace.codeTab).toBeVisible();
    await expect(workspace.previewTab).toBeVisible();
    await expect(workspace.terminalTab).toBeVisible();
  });

  test("prompt bar is visible and accepts input", async () => {
    // Prompt input should be visible
    await expect(workspace.promptInput).toBeVisible();

    // Should have placeholder text
    await expect(workspace.promptInput).toHaveAttribute(
      "placeholder",
      "Describe what you want to build..."
    );

    // Type into the prompt
    await workspace.typePrompt("Build a todo app");
    await expect(workspace.promptInput).toHaveValue("Build a todo app");
  });

  test("prompt bar submit is disabled when empty", async () => {
    // Execute button should be visible
    await expect(workspace.executeButton).toBeVisible();

    // Execute button should be disabled when prompt is empty
    await expect(workspace.executeButton).toBeDisabled();

    // Type something to enable it
    await workspace.typePrompt("Hello");
    await expect(workspace.executeButton).toBeEnabled();

    // Clear the prompt to disable it again
    await workspace.promptInput.fill("");
    await expect(workspace.executeButton).toBeDisabled();
  });
});
