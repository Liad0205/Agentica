"use client";

/**
 * PreviewPane - iframe component for previewing the sandbox dev server.
 * Features:
 * - Loading state while sandbox is starting
 * - Error state when sandbox is not ready
 * - Refresh button
 * - URL display
 * - Responsive iframe
 */

import * as React from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface PreviewPaneProps {
  /** URL of the sandbox dev server preview */
  url: string | null;
  /** Whether the preview is loading */
  loading?: boolean;
  /** Error message if preview is unavailable */
  error?: string | null;
  /** Callback to refresh the preview */
  onRefresh?: () => void;
  /** Callback to start the dev server */
  onStartDevServer?: () => void;
  /** Callback to stop the dev server */
  onStopDevServer?: () => void;
  /** Additional className */
  className?: string;
}

export function PreviewPane({
  url,
  loading = false,
  error,
  onRefresh,
  onStartDevServer,
  onStopDevServer,
  className,
}: PreviewPaneProps): React.ReactElement {
  const iframeRef = React.useRef<HTMLIFrameElement>(null);
  const [iframeError, setIframeError] = React.useState(false);
  const [iframeLoading, setIframeLoading] = React.useState(false);

  // Reset iframe error when URL changes
  React.useEffect(() => {
    if (url) {
      setIframeError(false);
      setIframeLoading(true);
    }
  }, [url]);

  const handleRefresh = (): void => {
    onRefresh?.();

    if (iframeRef.current && url) {
      setIframeLoading(true);
      setIframeError(false);
      iframeRef.current.src = buildRefreshedUrl(url);
    }
  };

  const handleIframeLoad = (): void => {
    setIframeLoading(false);
  };

  const handleIframeError = (): void => {
    setIframeLoading(false);
    setIframeError(true);
  };

  // Loading state
  if (loading) {
    return (
      <div
        className={cn(
          "flex h-full flex-col items-center justify-center bg-card",
          className
        )}
      >
        <LoadingSpinner className="mb-4 h-8 w-8 text-accent" />
        <p className="text-sm text-foreground-muted">
          Starting development server...
        </p>
        <p className="mt-2 text-xs text-foreground-muted/70">
          This may take a few seconds
        </p>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div
        className={cn(
          "flex h-full flex-col items-center justify-center bg-card",
          className
        )}
      >
        <ErrorIcon className="mb-4 h-12 w-12 text-error" />
        <p className="text-sm font-medium text-foreground">Preview Error</p>
        <p className="mt-2 max-w-md text-center text-xs text-foreground-muted">
          {error}
        </p>
        {onRefresh && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            className="mt-4"
          >
            <RefreshIcon className="mr-2 h-4 w-4" />
            Try Again
          </Button>
        )}
      </div>
    );
  }

  // No URL state
  if (!url) {
    return (
      <div
        className={cn(
          "flex h-full flex-col items-center justify-center bg-card",
          className
        )}
      >
        <PreviewIcon className="mb-4 h-12 w-12 text-foreground-muted/50" />
        <p className="text-sm font-medium text-foreground-muted">
          No Preview Available
        </p>
        <p className="mt-2 max-w-md text-center text-xs text-foreground-muted/70">
          The development server has not been started yet.
        </p>
        {onStartDevServer && (
          <Button
            variant="outline"
            size="sm"
            onClick={onStartDevServer}
            className="mt-4"
          >
            <PlayIcon className="mr-2 h-4 w-4" />
            Start Dev Server
          </Button>
        )}
      </div>
    );
  }

  return (
    <div className={cn("flex h-full flex-col bg-card", className)}>
      {/* Header with URL and refresh */}
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <GlobeIcon className="h-4 w-4 flex-shrink-0 text-foreground-muted" />
          <span
            className="truncate text-sm text-foreground-muted"
            title={url}
          >
            {url}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {/* Loading indicator */}
          {iframeLoading && (
            <LoadingSpinner className="h-4 w-4 text-accent" />
          )}

          {/* Open in new tab */}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => window.open(url, "_blank", "noopener,noreferrer")}
            title="Open in new tab"
          >
            <ExternalLinkIcon className="h-4 w-4" />
          </Button>

          {/* Refresh button */}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={handleRefresh}
            title="Refresh preview"
            disabled={iframeLoading}
          >
            <RefreshIcon className={cn("h-4 w-4", iframeLoading && "animate-spin")} />
          </Button>

          {/* Stop dev server */}
          {onStopDevServer && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-muted-foreground hover:bg-yellow-500/10 hover:text-yellow-500"
              onClick={onStopDevServer}
              title="Stop Dev Server"
            >
              <StopIcon className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      {/* iframe container */}
      <div className="relative flex-1">
        {/* iframe error state */}
        {iframeError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-card">
            <ErrorIcon className="mb-4 h-8 w-8 text-warning" />
            <p className="text-sm text-foreground-muted">
              Failed to load preview
            </p>
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              className="mt-4"
            >
              <RefreshIcon className="mr-2 h-4 w-4" />
              Retry
            </Button>
          </div>
        )}

        {/* iframe */}
        <iframe
          ref={iframeRef}
          src={url}
          title="Preview"
          className={cn(
            "h-full w-full border-none bg-white",
            iframeError && "invisible"
          )}
          onLoad={handleIframeLoad}
          onError={handleIframeError}
          sandbox="allow-forms allow-modals allow-orientation-lock allow-pointer-lock allow-popups allow-popups-to-escape-sandbox allow-presentation allow-same-origin allow-scripts"
        />
      </div>
    </div>
  );
}

function buildRefreshedUrl(url: string): string {
  try {
    const parsed = new URL(url);
    parsed.searchParams.set("_refresh", Date.now().toString());
    return parsed.toString();
  } catch {
    return url;
  }
}

/**
 * Loading spinner component
 */
function LoadingSpinner({
  className,
}: {
  className?: string;
}): React.ReactElement {
  return (
    <svg
      className={cn("animate-spin", className)}
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}

/**
 * Error icon
 */
function ErrorIcon({ className }: { className?: string }): React.ReactElement {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
      />
    </svg>
  );
}

/**
 * Preview icon for empty state
 */
function PreviewIcon({
  className,
}: {
  className?: string;
}): React.ReactElement {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
      />
    </svg>
  );
}

/**
 * Globe icon for URL display
 */
function GlobeIcon({ className }: { className?: string }): React.ReactElement {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"
      />
    </svg>
  );
}

/**
 * Refresh icon
 */
function RefreshIcon({
  className,
}: {
  className?: string;
}): React.ReactElement {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
      />
    </svg>
  );
}

/**
 * External link icon
 */
function ExternalLinkIcon({
  className,
}: {
  className?: string;
}): React.ReactElement {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
      />
    </svg>
  );
}

/**
 * Play icon for start button
 */
function PlayIcon({ className }: { className?: string }): React.ReactElement {
  return (
    <svg
      className={className}
      fill="currentColor"
      viewBox="0 0 24 24"
    >
      <path d="M8 5v14l11-7z" />
    </svg>
  );
}

/**
 * Stop icon for stop button
 */
function StopIcon({ className }: { className?: string }): React.ReactElement {
  return (
    <svg
      className={className}
      fill="currentColor"
      viewBox="0 0 24 24"
    >
      <rect x="6" y="6" width="12" height="12" />
    </svg>
  );
}

export default PreviewPane;
