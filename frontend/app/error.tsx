"use client";

import * as React from "react";

interface ErrorPageProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function ErrorPage({
  error,
  reset,
}: ErrorPageProps): React.ReactElement {
  return (
    <main className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="w-full max-w-xl rounded-xl border border-border bg-card p-6">
        <p className="text-xs uppercase tracking-wider text-warning">Application Error</p>
        <h1 className="mt-2 text-2xl font-semibold text-foreground">
          Something went wrong while rendering the workspace
        </h1>
        <p className="mt-3 text-sm text-foreground-muted">
          {error.message || "An unexpected error occurred."}
        </p>
        <div className="mt-6 flex items-center gap-3">
          <button
            type="button"
            onClick={reset}
            className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-background transition-opacity hover:opacity-90"
          >
            Retry
          </button>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="rounded-md border border-border px-4 py-2 text-sm text-foreground-muted transition-colors hover:bg-background hover:text-foreground"
          >
            Reload Page
          </button>
        </div>
      </div>
    </main>
  );
}
