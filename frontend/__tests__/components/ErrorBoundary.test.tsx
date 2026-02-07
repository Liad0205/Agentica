/**
 * Tests for the ErrorBoundary component (components/ErrorBoundary.tsx).
 * Verifies normal rendering, error catching, fallback UI, and reset behavior.
 */

import * as React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ErrorBoundary } from "@/components/ErrorBoundary";

// Suppress React error boundary console output during tests
beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

/**
 * Component that throws on render (for testing error boundaries).
 */
function ThrowingComponent({ shouldThrow }: { shouldThrow: boolean }): React.ReactElement {
  if (shouldThrow) {
    throw new Error("Test error message");
  }
  return <div data-testid="child">Children rendered successfully</div>;
}

// ---------------------------------------------------------------------------
// Normal rendering
// ---------------------------------------------------------------------------
describe("ErrorBoundary", () => {
  it("renders children when no error occurs", () => {
    render(
      <ErrorBoundary>
        <ThrowingComponent shouldThrow={false} />
      </ErrorBoundary>
    );

    expect(screen.getByTestId("child")).toBeInTheDocument();
    expect(screen.getByText("Children rendered successfully")).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Error catching
  // ---------------------------------------------------------------------------
  it("catches errors and shows fallback UI", () => {
    render(
      <ErrorBoundary>
        <ThrowingComponent shouldThrow={true} />
      </ErrorBoundary>
    );

    // Should show error message
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
    expect(screen.getByText("Test error message")).toBeInTheDocument();

    // Children should NOT be rendered
    expect(screen.queryByTestId("child")).not.toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Custom fallback
  // ---------------------------------------------------------------------------
  it("renders custom fallback when provided", () => {
    render(
      <ErrorBoundary fallback={<div data-testid="custom-fallback">Custom error UI</div>}>
        <ThrowingComponent shouldThrow={true} />
      </ErrorBoundary>
    );

    expect(screen.getByTestId("custom-fallback")).toBeInTheDocument();
    expect(screen.getByText("Custom error UI")).toBeInTheDocument();

    // Default fallback should NOT be shown
    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Try again button
  // ---------------------------------------------------------------------------
  it("'Try again' button resets error state", () => {
    // Use a stateful wrapper to control whether the child throws
    function TestWrapper(): React.ReactElement {
      const [shouldThrow, setShouldThrow] = React.useState(true);

      return (
        <div>
          <button
            data-testid="stop-throwing"
            onClick={() => setShouldThrow(false)}
          >
            Stop throwing
          </button>
          <ErrorBoundary>
            <ThrowingComponent shouldThrow={shouldThrow} />
          </ErrorBoundary>
        </div>
      );
    }

    render(<TestWrapper />);

    // Should show error UI initially
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();

    // Stop the child from throwing
    fireEvent.click(screen.getByTestId("stop-throwing"));

    // Click "Try again"
    fireEvent.click(screen.getByText("Try again"));

    // Should re-render children successfully
    expect(screen.getByTestId("child")).toBeInTheDocument();
    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Error without message
  // ---------------------------------------------------------------------------
  it("shows default error text when error has no message", () => {
    function ThrowNoMessage(): React.ReactElement {
      throw new Error();
    }

    render(
      <ErrorBoundary>
        <ThrowNoMessage />
      </ErrorBoundary>
    );

    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
    // The fallback text for empty message
    expect(screen.getByText("An unexpected error occurred")).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // componentDidCatch is called
  // ---------------------------------------------------------------------------
  it("logs error via componentDidCatch", () => {
    const consoleSpy = vi.spyOn(console, "error");

    render(
      <ErrorBoundary>
        <ThrowingComponent shouldThrow={true} />
      </ErrorBoundary>
    );

    // componentDidCatch should have logged the error
    expect(consoleSpy).toHaveBeenCalled();
    const calls = consoleSpy.mock.calls;
    const errorBoundaryCalls = calls.filter(
      (call) => typeof call[0] === "string" && call[0].includes("ErrorBoundary caught:")
    );
    expect(errorBoundaryCalls.length).toBeGreaterThan(0);
  });
});
