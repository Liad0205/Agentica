import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright configuration for Agent Swarm POC E2E tests.
 *
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  testDir: "./e2e",
  /* Run tests in files in parallel */
  fullyParallel: true,
  /* Fail the build on CI if you accidentally left test.only in the source code */
  forbidOnly: !!process.env.CI,
  /* Retry once on failure */
  retries: 1,
  /* Reporter to use */
  reporter: process.env.CI ? "github" : "html",
  /* Shared settings for all the projects below */
  use: {
    /* Base URL to use in actions like `await page.goto('/')` */
    baseURL: "http://localhost:3000",
    /* Collect trace on first retry */
    trace: "on-first-retry",
    /* Screenshot on failure */
    screenshot: "only-on-failure",
  },
  /* Global timeout for each test (agents can be slow) */
  timeout: 60_000,
  /* Expect timeout */
  expect: {
    timeout: 10_000,
  },

  /* Configure projects for chromium only (for speed in dev) */
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  /* Run the Next.js dev server before starting the tests */
  webServer: {
    command: "npm run dev",
    url: "http://localhost:3000",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
