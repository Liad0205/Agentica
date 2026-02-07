"use client";

import * as React from "react";
import { ArrowUp, Square, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { SHORTCUTS } from "@/lib/constants";

interface PromptBarProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onCancel: () => void;
  isRunning: boolean;
  isMinimized?: boolean;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

/**
 * Adaptive PromptBar: full textarea when idle, thin status bar when minimized & running.
 */
export function PromptBar({
  value,
  onChange,
  onSubmit,
  onCancel,
  isRunning,
  isMinimized = false,
  disabled = false,
  placeholder = "Describe your task...",
  className,
}: PromptBarProps): React.ReactElement {
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const [elapsedSeconds, setElapsedSeconds] = React.useState(0);

  // Auto-resize textarea
  const handleInput = React.useCallback((): void => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      const maxHeight = 200;
      const newHeight = Math.min(textarea.scrollHeight, maxHeight);
      textarea.style.height = `${newHeight}px`;
    }
  }, []);

  React.useEffect(() => {
    handleInput();
  }, [value, handleInput]);

  // Track elapsed time when running
  React.useEffect(() => {
    if (!isRunning) {
      setElapsedSeconds(0);
      return;
    }
    setElapsedSeconds(0);
    const interval = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [isRunning]);

  const handleKeyDown = React.useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement>): void => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        if (!isRunning && !disabled && value.trim()) {
          onSubmit();
        }
      }
      if (event.key === "Escape" && isRunning) {
        event.preventDefault();
        onCancel();
      }
    },
    [isRunning, disabled, value, onSubmit, onCancel],
  );

  const canSubmit = !isRunning && !disabled && value.trim().length > 0;

  const formatElapsed = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  };

  // Minimized running status bar
  if (isRunning && isMinimized) {
    return (
      <div className={cn("flex h-7 items-center justify-between px-3", className)}>
        <div className="flex items-center gap-2">
          <Loader2 className="h-3 w-3 animate-spin text-accent" />
          <span className="text-xs text-muted-foreground">
            Running... {formatElapsed(elapsedSeconds)}
          </span>
        </div>
        <Button
          size="sm"
          variant="ghost"
          onClick={onCancel}
          className="h-5 px-2 text-[10px] text-muted-foreground hover:text-foreground"
        >
          <Square className="mr-1 h-2.5 w-2.5 fill-current" />
          Stop
        </Button>
      </div>
    );
  }

  return (
    <div className={cn("relative w-full", className)}>
      <div
        className={cn(
          "relative flex items-end gap-2 rounded-xl border bg-background p-2 ring-1 ring-border shadow-sm transition-all focus-within:ring-2 focus-within:ring-foreground/20",
          isRunning && "opacity-80",
        )}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder={placeholder}
          disabled={isRunning}
          rows={1}
          className="flex-1 min-h-[40px] max-h-[200px] w-full resize-none bg-transparent px-3 py-2.5 text-sm placeholder:text-muted-foreground focus:outline-none disabled:cursor-not-allowed break-words overflow-y-auto"
        />

        <div className="flex items-center pb-1">
          {isRunning ? (
            <Button
              size="icon"
              variant="default"
              className="h-8 w-8 rounded-lg bg-foreground text-background hover:bg-foreground/90"
              onClick={onCancel}
            >
              <Square className="h-4 w-4 fill-current" />
            </Button>
          ) : (
            <Button
              size="icon"
              variant="default"
              disabled={!canSubmit}
              className={cn(
                "h-8 w-8 rounded-lg transition-all",
                canSubmit
                  ? "bg-foreground text-background hover:bg-foreground/90"
                  : "bg-muted text-muted-foreground hover:bg-muted",
              )}
              onClick={onSubmit}
            >
              <ArrowUp className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
      <div className="mt-2 text-center">
        <span className="text-[10px] text-muted-foreground font-mono opacity-50">
          {SHORTCUTS.submitPrompt} to run
        </span>
      </div>
    </div>
  );
}

export function usePromptBar(): {
  value: string;
  setValue: React.Dispatch<React.SetStateAction<string>>;
  clear: () => void;
} {
  const [value, setValue] = React.useState("");
  const clear = (): void => setValue("");
  return { value, setValue, clear };
}
