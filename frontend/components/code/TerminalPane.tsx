"use client";

/**
 * TerminalPane - Terminal emulator using xterm.js.
 * Features:
 * - Display command output stream
 * - Dark theme terminal
 * - Auto-scroll to bottom
 * - Clear button
 * - Copy output support
 */

import * as React from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

// Types for xterm
interface ITerminalOptions {
  cursorBlink?: boolean;
  fontSize?: number;
  fontFamily?: string;
  theme?: {
    background?: string;
    foreground?: string;
    cursor?: string;
    cursorAccent?: string;
    selection?: string;
    black?: string;
    red?: string;
    green?: string;
    yellow?: string;
    blue?: string;
    magenta?: string;
    cyan?: string;
    white?: string;
    brightBlack?: string;
    brightRed?: string;
    brightGreen?: string;
    brightYellow?: string;
    brightBlue?: string;
    brightMagenta?: string;
    brightCyan?: string;
    brightWhite?: string;
  };
  scrollback?: number;
  convertEol?: boolean;
  disableStdin?: boolean;
}

interface ITerminal {
  open: (parent: HTMLElement) => void;
  write: (data: string) => void;
  writeln: (data: string) => void;
  clear: () => void;
  dispose: () => void;
  scrollToBottom: () => void;
  onData: (callback: (data: string) => void) => { dispose: () => void };
}

interface IFitAddon {
  fit: () => void;
  dispose: () => void;
}

interface TerminalPaneProps {
  /** Terminal output lines to display */
  output: string[];
  /** Callback when user clears the terminal */
  onClear?: () => void;
  /** Sandbox IDs that have terminal output (for tabs) */
  sandboxIds?: string[];
  /** Currently active sandbox tab */
  activeSandboxId?: string;
  /** Callback when user switches sandbox tab */
  onSandboxChange?: (sandboxId: string) => void;
  /** Additional className */
  className?: string;
}

export function TerminalPane({
  output,
  onClear,
  sandboxIds = [],
  activeSandboxId = "all",
  onSandboxChange,
  className,
}: TerminalPaneProps): React.ReactElement {
  const terminalRef = React.useRef<HTMLDivElement>(null);
  const xtermRef = React.useRef<ITerminal | null>(null);
  const fitAddonRef = React.useRef<IFitAddon | null>(null);
  const lastOutputLengthRef = React.useRef<number>(0);
  const [isLoaded, setIsLoaded] = React.useState(false);
  const [loadError, setLoadError] = React.useState<string | null>(null);

  // Initialize xterm.js
  React.useEffect(() => {
    let mounted = true;

    const initTerminal = async (): Promise<void> => {
      if (!terminalRef.current) return;

      try {
        // Dynamic import of xterm to avoid SSR issues
        const { Terminal } = await import("@xterm/xterm");
        const { FitAddon } = await import("@xterm/addon-fit");

        // Import xterm CSS
        await import("@xterm/xterm/css/xterm.css");

        if (!mounted || !terminalRef.current) return;

        // Terminal options with dark theme
        const options: ITerminalOptions = {
          cursorBlink: false,
          fontSize: 13,
          fontFamily: "JetBrains Mono, Menlo, Monaco, monospace",
          theme: {
            background: "#12121a",
            foreground: "#ffffff",
            cursor: "#00d4ff",
            cursorAccent: "#12121a",
            selection: "rgba(0, 212, 255, 0.3)",
            black: "#0a0a0f",
            red: "#ef4444",
            green: "#10b981",
            yellow: "#f59e0b",
            blue: "#3b82f6",
            magenta: "#a855f7",
            cyan: "#00d4ff",
            white: "#ffffff",
            brightBlack: "#6b7280",
            brightRed: "#f87171",
            brightGreen: "#34d399",
            brightYellow: "#fbbf24",
            brightBlue: "#60a5fa",
            brightMagenta: "#c084fc",
            brightCyan: "#22d3ee",
            brightWhite: "#ffffff",
          },
          scrollback: 5000,
          convertEol: true,
          disableStdin: true, // Read-only terminal
        };

        // Create terminal instance
        const terminal = new Terminal(options);
        const fitAddon = new FitAddon();
        terminal.loadAddon(fitAddon);

        // Open terminal in container
        terminal.open(terminalRef.current);
        fitAddon.fit();

        xtermRef.current = terminal as unknown as ITerminal;
        fitAddonRef.current = fitAddon as unknown as IFitAddon;

        // Don't write output here - let the second useEffect handle it
        // once isLoaded is true, to avoid stale closure over `output`.
        setIsLoaded(true);
      } catch (err) {
        console.error("Failed to initialize terminal:", err);
        setLoadError("Failed to load terminal");
      }
    };

    initTerminal();

    return () => {
      mounted = false;
      if (xtermRef.current) {
        xtermRef.current.dispose();
        xtermRef.current = null;
      }
      if (fitAddonRef.current) {
        fitAddonRef.current.dispose();
        fitAddonRef.current = null;
      }
    };
  }, []); // Empty deps - only initialize once

  // Handle new output
  React.useEffect(() => {
    if (!xtermRef.current || !isLoaded) return;

    // If output was reset (e.g. cleared), reset terminal state as well.
    if (output.length < lastOutputLengthRef.current) {
      xtermRef.current.clear();
      lastOutputLengthRef.current = 0;
    }

    // Only write new chunks since last update.
    const newChunks = output.slice(lastOutputLengthRef.current);
    newChunks.forEach((chunk) => {
      xtermRef.current?.write(chunk);
    });
    lastOutputLengthRef.current = output.length;

    // Auto-scroll to bottom
    xtermRef.current.scrollToBottom();
  }, [output, isLoaded]);

  // Handle resize
  React.useEffect(() => {
    const handleResize = (): void => {
      if (fitAddonRef.current) {
        fitAddonRef.current.fit();
      }
    };

    window.addEventListener("resize", handleResize);

    // Also fit on initial render after a short delay
    const timer = setTimeout(handleResize, 100);

    return () => {
      window.removeEventListener("resize", handleResize);
      clearTimeout(timer);
    };
  }, [isLoaded]);

  const handleClear = (): void => {
    if (xtermRef.current) {
      xtermRef.current.clear();
      lastOutputLengthRef.current = 0;
    }
    onClear?.();
  };

  const handleCopyAll = async (): Promise<void> => {
    try {
      const text = output.join("\n");
      await navigator.clipboard.writeText(text);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  // Error state
  if (loadError) {
    return (
      <div
        className={cn(
          "flex h-full flex-col items-center justify-center bg-card",
          className
        )}
      >
        <TerminalIcon className="mb-4 h-12 w-12 text-error" />
        <p className="text-sm text-foreground">{loadError}</p>
        <p className="mt-2 text-xs text-foreground-muted">
          Please refresh the page to try again
        </p>
      </div>
    );
  }

  const showTabs = sandboxIds.length > 1;

  return (
    <div className={cn("flex h-full flex-col bg-[#12121a]", className)}>
      {/* Sandbox tab bar (multi-sandbox modes only) */}
      {showTabs && (
        <div className="flex items-center gap-1 border-b border-border px-3 py-1.5 bg-[#0a0a0f]">
          <button
            type="button"
            onClick={() => onSandboxChange?.("all")}
            className={cn(
              "rounded px-2 py-0.5 text-xs font-medium transition-colors",
              activeSandboxId === "all"
                ? "bg-[#00d4ff]/15 text-[#00d4ff]"
                : "text-[#6b7280] hover:text-[#ffffff]"
            )}
          >
            All
          </button>
          {sandboxIds.map((id) => (
            <button
              key={id}
              type="button"
              onClick={() => onSandboxChange?.(id)}
              className={cn(
                "rounded px-2 py-0.5 text-xs font-medium transition-colors truncate max-w-[120px]",
                activeSandboxId === id
                  ? "bg-[#00d4ff]/15 text-[#00d4ff]"
                  : "text-[#6b7280] hover:text-[#ffffff]"
              )}
              title={id}
            >
              {id.length > 12 ? `${id.slice(0, 12)}...` : id}
            </button>
          ))}
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <div className="flex items-center gap-2">
          <TerminalIcon className="h-4 w-4 text-foreground-muted" />
          <span className="text-sm text-foreground">Terminal Output</span>
          {output.length > 0 && (
            <span className="text-xs text-foreground-muted">
              ({output.length} lines)
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {/* Copy all */}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={handleCopyAll}
            title="Copy all output"
            disabled={output.length === 0}
          >
            <CopyIcon className="h-4 w-4" />
          </Button>

          {/* Clear button */}
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={handleClear}
            title="Clear terminal"
            disabled={output.length === 0}
          >
            <TrashIcon className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Terminal container */}
      <div className="relative flex-1 overflow-hidden">
        {/* Loading state */}
        {!isLoaded && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#12121a]">
            <LoadingSpinner className="h-6 w-6 text-accent" />
          </div>
        )}

        {/* Empty state */}
        {isLoaded && output.length === 0 && (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <TerminalIcon className="mb-3 h-8 w-8 text-foreground-muted/30" />
            <p className="text-sm text-foreground-muted/50">
              Waiting for command output...
            </p>
          </div>
        )}

        {/* xterm container */}
        <div
          ref={terminalRef}
          className={cn(
            "h-full w-full p-2",
            !isLoaded && "invisible"
          )}
        />
      </div>
    </div>
  );
}

/**
 * Terminal icon
 */
function TerminalIcon({
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
        d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
      />
    </svg>
  );
}

/**
 * Copy icon
 */
function CopyIcon({ className }: { className?: string }): React.ReactElement {
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
        d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
      />
    </svg>
  );
}

/**
 * Trash icon
 */
function TrashIcon({ className }: { className?: string }): React.ReactElement {
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
        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
      />
    </svg>
  );
}

/**
 * Loading spinner
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

export default TerminalPane;
