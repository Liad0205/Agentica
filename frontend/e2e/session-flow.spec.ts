/**
 * Session lifecycle E2E tests for the Agentica workspace.
 *
 * These tests require the backend to be running at http://localhost:8000.
 * If the backend is not available, all tests in this file will be skipped.
 */

import { test, expect } from "@playwright/test";
import { WorkspacePage, isBackendAvailable } from "./fixtures";

test.describe("Session lifecycle", () => {
  let workspace: WorkspacePage;
  let backendUp: boolean;

  test.beforeAll(async () => {
    backendUp = await isBackendAvailable();
  });

  test.beforeEach(async ({ page }) => {
    test.skip(!backendUp, "Backend is not available at http://localhost:8000");
    workspace = new WorkspacePage(page);
    await workspace.goto();
  });

  test("start a ReAct session with a simple task", async () => {
    // Ensure ReAct mode is selected
    await expect(workspace.reactModeButton).toHaveAttribute(
      "aria-checked",
      "true"
    );

    // Type a task
    await workspace.typePrompt("Create a simple hello world HTML page");

    // Submit
    await workspace.submitPrompt();

    // The Execute button should disappear and the Cancel button should appear
    // (indicates the session transitioned from idle to running)
    await expect(workspace.cancelButton).toBeVisible({ timeout: 15_000 });

    // The prompt input should be disabled while running
    await expect(workspace.promptInput).toBeDisabled();
  });

  test("verify session status transitions (idle to running)", async () => {
    // Initially the Execute button is visible (idle state)
    await expect(workspace.executeButton).toBeVisible();
    await expect(workspace.cancelButton).not.toBeVisible();

    // Type and submit a task
    await workspace.typePrompt("Build a counter component");
    await workspace.submitPrompt();

    // After submission, should transition to running state:
    // - Cancel button appears
    // - Prompt input becomes disabled
    await expect(workspace.cancelButton).toBeVisible({ timeout: 15_000 });
    await expect(workspace.promptInput).toBeDisabled();
  });

  test("cancel a running session", async () => {
    // Start a session
    await workspace.typePrompt("Create a React todo application with full CRUD");
    await workspace.submitPrompt();

    // Wait for the session to start running
    await expect(workspace.cancelButton).toBeVisible({ timeout: 15_000 });

    // Cancel the session
    await workspace.cancelSession();

    // After cancellation, the Execute button should reappear
    await expect(workspace.executeButton).toBeVisible({ timeout: 10_000 });

    // The Cancel button should no longer be visible
    await expect(workspace.cancelButton).not.toBeVisible();
  });

  test("cancel updates UI state correctly", async () => {
    // Start a session
    await workspace.typePrompt("Build a weather dashboard");
    await workspace.submitPrompt();

    // Wait for running state
    await expect(workspace.cancelButton).toBeVisible({ timeout: 15_000 });

    // Cancel
    await workspace.cancelSession();

    // The prompt input should become enabled again
    await expect(workspace.promptInput).toBeEnabled({ timeout: 10_000 });

    // The Execute button should be visible again
    await expect(workspace.executeButton).toBeVisible();

    // The mode selector should become interactive again (not disabled)
    await expect(workspace.reactModeButton).toBeEnabled();
    await expect(workspace.decompositionModeButton).toBeEnabled();
    await expect(workspace.hypothesisModeButton).toBeEnabled();
  });

  test("reset session clears state", async () => {
    // Start and cancel a session to put the app in a post-session state
    await workspace.typePrompt("Build something");
    await workspace.submitPrompt();
    await expect(workspace.cancelButton).toBeVisible({ timeout: 15_000 });
    await workspace.cancelSession();
    await expect(workspace.executeButton).toBeVisible({ timeout: 10_000 });

    // After cancellation, the prompt should still have the previous value
    // The user can type a new prompt and start fresh
    await workspace.promptInput.fill("");
    await workspace.typePrompt("New task");

    // The Execute button should be enabled for the new prompt
    await expect(workspace.executeButton).toBeEnabled();

    // Mode selector should be interactive
    await workspace.selectMode("decomposition");
    await expect(workspace.decompositionModeButton).toHaveAttribute(
      "aria-checked",
      "true"
    );
  });
});
