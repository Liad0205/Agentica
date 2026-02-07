"use client";

import * as React from "react";
import { Brain, GitBranch, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { MODE_INFO } from "@/lib/constants";
import type { Mode } from "@/lib/types";

interface ModeSelectorProps {
  selectedMode: Mode;
  onModeChange: (mode: Mode) => void;
  disabled?: boolean;
}

interface ModeOption {
  mode: Mode;
  icon: React.ReactNode;
  name: string;
  description: string;
}

const modeOptions: ModeOption[] = [
  {
    mode: "react",
    icon: <Brain className="w-4 h-4" />,
    name: MODE_INFO.react.name,
    description: MODE_INFO.react.description,
  },
  {
    mode: "decomposition",
    icon: <GitBranch className="w-4 h-4" />,
    name: MODE_INFO.decomposition.name,
    description: MODE_INFO.decomposition.description,
  },
  {
    mode: "hypothesis",
    icon: <Sparkles className="w-4 h-4" />,
    name: MODE_INFO.hypothesis.name,
    description: MODE_INFO.hypothesis.description,
  },
];

/**
 * ModeSelector component for switching between agent operation modes.
 * Displays as a radio group with visual indicators for each mode.
 */
export function ModeSelector({
  selectedMode,
  onModeChange,
  disabled = false,
}: ModeSelectorProps): React.ReactElement {
  return (
    <div
      className="flex items-center p-1 rounded-md bg-muted/50 border border-border"
      role="radiogroup"
      aria-label="Select agent mode"
    >
      {modeOptions.map((option) => (
        <ModeButton
          key={option.mode}
          option={option}
          isSelected={selectedMode === option.mode}
          onClick={() => onModeChange(option.mode)}
          disabled={disabled}
        />
      ))}
    </div>
  );
}

interface ModeButtonProps {
  option: ModeOption;
  isSelected: boolean;
  onClick: () => void;
  disabled: boolean;
}

function ModeButton({
  option,
  isSelected,
  onClick,
  disabled,
}: ModeButtonProps): React.ReactElement {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={isSelected}
      aria-label={`${option.name}: ${option.description}`}
      title={option.description}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 px-3 py-1.5 rounded-sm text-sm font-medium transition-all",
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        isSelected
          ? "bg-background text-foreground shadow-sm ring-1 ring-border"
          : "text-muted-foreground hover:text-foreground hover:bg-muted/50",
        disabled && "opacity-50 cursor-not-allowed"
      )}
    >
      <span
        className={cn(
          "transition-colors",
          isSelected ? "text-foreground" : "text-muted-foreground"
        )}
      >
        {option.icon}
      </span>
      <span>{option.name}</span>
      <span
        className={cn(
          "w-2 h-2 rounded-full transition-all",
          isSelected ? "bg-accent" : "bg-border"
        )}
        aria-hidden="true"
      />
    </button>
  );
}
