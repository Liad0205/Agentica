/**
 * Panel interaction E2E tests for the Agentica workspace.
 *
 * Tests the react-resizable-panels layout, including resizing,
 * collapsing, and expanding panels.
 */

import { test, expect } from "@playwright/test";
import { WorkspacePage } from "./fixtures";

test.describe("Panel interactions", () => {
  let workspace: WorkspacePage;

  test.beforeEach(async ({ page }) => {
    // Clear localStorage to reset saved panel sizes
    await page.addInitScript(() => {
      localStorage.clear();
    });
    workspace = new WorkspacePage(page);
    await workspace.goto();
  });

  test("panels can be resized via drag handles", async ({ page }) => {
    // Get initial width of the left panel
    const leftPanelBox = await workspace.leftPanel.boundingBox();
    expect(leftPanelBox).not.toBeNull();
    const initialLeftWidth = leftPanelBox!.width;

    // Find the first resize handle (between left and center panels)
    const resizeHandles = page.locator("[data-panel-resize-handle-id]");
    const firstHandle = resizeHandles.first();
    await expect(firstHandle).toBeVisible();

    // Get the handle's bounding box
    const handleBox = await firstHandle.boundingBox();
    expect(handleBox).not.toBeNull();

    // Drag the handle to the right to increase the left panel size
    const startX = handleBox!.x + handleBox!.width / 2;
    const startY = handleBox!.y + handleBox!.height / 2;
    const dragDistance = 100;

    await page.mouse.move(startX, startY);
    await page.mouse.down();
    await page.mouse.move(startX + dragDistance, startY, { steps: 10 });
    await page.mouse.up();

    // Wait for the resize to take effect
    await page.waitForTimeout(300);

    // Verify the left panel has changed size
    const newLeftPanelBox = await workspace.leftPanel.boundingBox();
    expect(newLeftPanelBox).not.toBeNull();

    // The width should have increased (allowing some tolerance for rounding)
    expect(newLeftPanelBox!.width).toBeGreaterThan(initialLeftWidth + 20);
  });

  test("left panel can be collapsed and re-expanded", async ({ page }) => {
    // Verify left panel is initially visible
    await expect(workspace.leftPanel).toBeVisible();
    await expect(workspace.sessionLogTitle).toBeVisible();

    // Get the first resize handle
    const firstHandle = page.locator("[data-panel-resize-handle-id]").first();
    const handleBox = await firstHandle.boundingBox();
    expect(handleBox).not.toBeNull();

    // Drag the handle all the way to the left to collapse
    const startX = handleBox!.x + handleBox!.width / 2;
    const startY = handleBox!.y + handleBox!.height / 2;

    await page.mouse.move(startX, startY);
    await page.mouse.down();
    // Drag far to the left (past the minimum panel size)
    await page.mouse.move(0, startY, { steps: 20 });
    await page.mouse.up();

    // Wait for animation
    await page.waitForTimeout(300);

    // Left panel should be collapsed (hidden class applied)
    // When collapsed, the panel gets the "hidden" class per WorkspaceLayout
    await expect(workspace.leftPanel).toBeHidden();

    // An expand button should appear near the collapsed panel
    const expandButton = page.getByRole("button", {
      name: /Expand.*panel/i,
    });
    await expect(expandButton).toBeVisible({ timeout: 5_000 });

    // Click the expand button to re-expand
    await expandButton.click();

    // Wait for expansion animation
    await page.waitForTimeout(500);

    // Left panel should be visible again
    await expect(workspace.leftPanel).toBeVisible();
    await expect(workspace.sessionLogTitle).toBeVisible();
  });

  test("center panel can be collapsed and re-expanded", async ({ page }) => {
    // Verify center panel is initially visible
    await expect(workspace.centerPanel).toBeVisible();

    // Get the second resize handle (between center and right panels)
    const resizeHandles = page.locator("[data-panel-resize-handle-id]");
    const secondHandle = resizeHandles.nth(1);
    await expect(secondHandle).toBeVisible();

    const handleBox = await secondHandle.boundingBox();
    expect(handleBox).not.toBeNull();

    // Get the position of the first handle to know where the center panel
    // starts so we can drag past it to collapse the center panel
    const firstHandle = resizeHandles.first();
    const firstHandleBox = await firstHandle.boundingBox();
    expect(firstHandleBox).not.toBeNull();

    const startX = handleBox!.x + handleBox!.width / 2;
    const startY = handleBox!.y + handleBox!.height / 2;

    // Drag the second handle to the left past the first handle
    // to collapse the center panel
    await page.mouse.move(startX, startY);
    await page.mouse.down();
    await page.mouse.move(firstHandleBox!.x, startY, { steps: 20 });
    await page.mouse.up();

    // Wait for animation
    await page.waitForTimeout(300);

    // Center panel should be collapsed (hidden)
    await expect(workspace.centerPanel).toBeHidden();

    // An expand button should appear
    const expandButtons = page.getByRole("button", {
      name: /Expand.*panel/i,
    });
    // There might be one or more expand buttons depending on state
    const expandCount = await expandButtons.count();
    expect(expandCount).toBeGreaterThanOrEqual(1);

    // Click the last expand button (the one for the center panel)
    await expandButtons.last().click();

    // Wait for expansion animation
    await page.waitForTimeout(500);

    // Center panel should be visible again
    await expect(workspace.centerPanel).toBeVisible();
  });

  test("panel sizes persist via autoSaveId", async ({ page, context }) => {
    // Get the first resize handle
    const firstHandle = page.locator("[data-panel-resize-handle-id]").first();
    const handleBox = await firstHandle.boundingBox();
    expect(handleBox).not.toBeNull();

    // Drag the handle to resize
    const startX = handleBox!.x + handleBox!.width / 2;
    const startY = handleBox!.y + handleBox!.height / 2;
    const dragDistance = 80;

    await page.mouse.move(startX, startY);
    await page.mouse.down();
    await page.mouse.move(startX + dragDistance, startY, { steps: 10 });
    await page.mouse.up();

    // Wait for the resize and localStorage save
    await page.waitForTimeout(500);

    // Get the left panel width after resize
    const resizedBox = await workspace.leftPanel.boundingBox();
    expect(resizedBox).not.toBeNull();
    const resizedWidth = resizedBox!.width;

    // Verify that localStorage has the panel sizes saved
    // react-resizable-panels uses a key based on the autoSaveId
    const savedLayout = await page.evaluate(() => {
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.includes("workspace-layout")) {
          return localStorage.getItem(key);
        }
      }
      return null;
    });

    expect(savedLayout).not.toBeNull();

    // Navigate away and come back to verify persistence
    const newPage = await context.newPage();
    await newPage.goto("/");
    await newPage.waitForSelector("text=AGENT ARENA");

    // Wait for panel layout to restore
    await newPage.waitForTimeout(500);

    // The left panel should have the same width as before (within tolerance)
    const newWorkspace = new WorkspacePage(newPage);
    const restoredBox = await newWorkspace.leftPanel.boundingBox();
    expect(restoredBox).not.toBeNull();

    // Allow a small tolerance for rounding differences
    expect(restoredBox!.width).toBeCloseTo(resizedWidth, -1);

    await newPage.close();
  });
});
