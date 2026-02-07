/**
 * Custom hook for prompt input state management.
 * Handles prompt value, submission, and cancellation.
 */

import { useCallback, useMemo } from "react";
import { useStore } from "@/lib/store";
import { useSession } from "./useSession";
import type { ModelConfig } from "@/lib/types";

/**
 * Return type for the usePrompt hook.
 */
export interface UsePromptReturn {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (modelConfig?: ModelConfig) => Promise<void>;
  onCancel: () => Promise<void>;
  isRunning: boolean;
  canSubmit: boolean;
}

/**
 * Hook for managing prompt input state and submission.
 * Integrates with useSession for session operations.
 */
export function usePrompt(): UsePromptReturn {
  const promptValue = useStore(
    (state) => state.promptValue,
  );
  const setPromptValue = useStore(
    (state) => state.setPromptValue,
  );

  const { isRunning, startSession, cancelSession } =
    useSession();

  /**
   * Handle prompt value change.
   */
  const onChange = useCallback(
    (value: string): void => {
      setPromptValue(value);
    },
    [setPromptValue],
  );

  /**
   * Submit the current prompt to start or continue a session run.
   */
  const onSubmit = useCallback(
    async (modelConfig?: ModelConfig): Promise<void> => {
      const trimmedValue = promptValue.trim();
      if (!trimmedValue || isRunning) {
        return;
      }

      try {
        await startSession(trimmedValue, modelConfig);
        setPromptValue("");
      } catch (error) {
        console.error("Failed to start session:", error);
      }
    },
    [promptValue, isRunning, startSession, setPromptValue],
  );

  /**
   * Cancel the current running session.
   */
  const onCancel = useCallback(async (): Promise<void> => {
    await cancelSession();
  }, [cancelSession]);

  /**
   * Determine if the submit button should be enabled.
   * Can submit when there's a non-empty prompt and no session is running.
   */
  const canSubmit = useMemo((): boolean => {
    return promptValue.trim().length > 0 && !isRunning;
  }, [promptValue, isRunning]);

  return {
    value: promptValue,
    onChange,
    onSubmit,
    onCancel,
    isRunning,
    canSubmit,
  };
}
