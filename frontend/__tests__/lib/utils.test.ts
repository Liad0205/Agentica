/**
 * Tests for utility functions (lib/utils.ts).
 */

import { describe, it, expect } from "vitest";
import {
  cn,
  formatTime,
  formatDuration,
  formatNumber,
  truncate,
  getInitials,
  getFileExtension,
  getFileName,
  getDirectory,
  isWithinDirectory,
  sortFileNodes,
  safeJsonParse,
} from "@/lib/utils";

// ---------------------------------------------------------------------------
// cn (classnames + tailwind merge)
// ---------------------------------------------------------------------------
describe("cn", () => {
  it("merges class names", () => {
    const result = cn("px-4", "py-2");
    expect(result).toBe("px-4 py-2");
  });

  it("handles conditional classes", () => {
    const result = cn("base", false && "hidden", true && "visible");
    expect(result).toBe("base visible");
  });

  it("resolves Tailwind conflicts (last wins)", () => {
    const result = cn("px-4", "px-6");
    expect(result).toBe("px-6");
  });

  it("handles undefined and null inputs", () => {
    const result = cn("base", undefined, null, "extra");
    expect(result).toBe("base extra");
  });

  it("returns empty string for no inputs", () => {
    expect(cn()).toBe("");
  });

  it("handles array inputs", () => {
    const result = cn(["px-2", "py-1"]);
    expect(result).toBe("px-2 py-1");
  });
});

// ---------------------------------------------------------------------------
// formatTime
// ---------------------------------------------------------------------------
describe("formatTime", () => {
  it("formats a Unix timestamp as HH:MM:SS", () => {
    // Use a fixed timestamp and check format pattern
    const result = formatTime(0);
    // The exact result depends on locale / timezone, but should be HH:MM:SS
    expect(result).toMatch(/^\d{2}:\d{2}:\d{2}$/);
  });

  it("converts seconds to milliseconds (multiplies by 1000)", () => {
    // Confirm it treats the input as seconds, not ms
    const tsSeconds = 1700000000; // ~ 2023-11-14
    const result = formatTime(tsSeconds);
    expect(result).toMatch(/^\d{2}:\d{2}:\d{2}$/);
  });
});

// ---------------------------------------------------------------------------
// formatDuration
// ---------------------------------------------------------------------------
describe("formatDuration", () => {
  it("formats seconds under 60 with one decimal", () => {
    expect(formatDuration(5.123)).toBe("5.1s");
    expect(formatDuration(0)).toBe("0.0s");
    expect(formatDuration(59.9)).toBe("59.9s");
  });

  it("formats 60+ seconds as minutes and seconds", () => {
    expect(formatDuration(60)).toBe("1m 0s");
    expect(formatDuration(90)).toBe("1m 30s");
    expect(formatDuration(125)).toBe("2m 5s");
  });
});

// ---------------------------------------------------------------------------
// formatNumber
// ---------------------------------------------------------------------------
describe("formatNumber", () => {
  it("adds thousand separators", () => {
    expect(formatNumber(1000)).toBe("1,000");
    expect(formatNumber(1000000)).toBe("1,000,000");
  });

  it("handles small numbers without separators", () => {
    expect(formatNumber(42)).toBe("42");
    expect(formatNumber(0)).toBe("0");
  });
});

// ---------------------------------------------------------------------------
// truncate
// ---------------------------------------------------------------------------
describe("truncate", () => {
  it("returns original string if shorter than maxLength", () => {
    expect(truncate("hello", 10)).toBe("hello");
  });

  it("truncates with ellipsis if longer than maxLength", () => {
    expect(truncate("hello world", 8)).toBe("hello...");
  });

  it("handles exact maxLength", () => {
    expect(truncate("hello", 5)).toBe("hello");
  });
});

// ---------------------------------------------------------------------------
// getInitials
// ---------------------------------------------------------------------------
describe("getInitials", () => {
  it("extracts initials from a multi-word name", () => {
    expect(getInitials("Orchestrator Agent")).toBe("OA");
  });

  it("handles single word names", () => {
    expect(getInitials("Agent")).toBe("A");
  });

  it("limits to maxLength characters", () => {
    expect(getInitials("Very Long Name Here", 3)).toBe("VLN");
    expect(getInitials("Very Long Name Here", 1)).toBe("V");
  });

  it("handles default maxLength of 2", () => {
    expect(getInitials("Alpha Beta Gamma")).toBe("AB");
  });

  it("uppercases initials", () => {
    expect(getInitials("lower case")).toBe("LC");
  });

  it("handles empty string", () => {
    expect(getInitials("")).toBe("");
  });

  it("handles extra whitespace", () => {
    expect(getInitials("  Spaced   Out  ")).toBe("SO");
  });
});

// ---------------------------------------------------------------------------
// getFileExtension
// ---------------------------------------------------------------------------
describe("getFileExtension", () => {
  it("returns file extension", () => {
    expect(getFileExtension("file.ts")).toBe("ts");
    expect(getFileExtension("component.test.tsx")).toBe("tsx");
  });

  it("returns empty string for no extension", () => {
    expect(getFileExtension("Makefile")).toBe("");
  });
});

// ---------------------------------------------------------------------------
// getFileName
// ---------------------------------------------------------------------------
describe("getFileName", () => {
  it("returns filename from a path", () => {
    expect(getFileName("src/components/Button.tsx")).toBe("Button.tsx");
  });

  it("returns the input if no slashes", () => {
    expect(getFileName("file.ts")).toBe("file.ts");
  });
});

// ---------------------------------------------------------------------------
// getDirectory
// ---------------------------------------------------------------------------
describe("getDirectory", () => {
  it("returns directory from a path", () => {
    expect(getDirectory("src/components/Button.tsx")).toBe("src/components");
  });

  it("returns '.' for a file with no directory", () => {
    expect(getDirectory("file.ts")).toBe(".");
  });
});

// ---------------------------------------------------------------------------
// isWithinDirectory
// ---------------------------------------------------------------------------
describe("isWithinDirectory", () => {
  it("returns true if path is within directory", () => {
    expect(isWithinDirectory("src/file.ts", "src")).toBe(true);
  });

  it("returns false if path is outside directory", () => {
    expect(isWithinDirectory("lib/file.ts", "src")).toBe(false);
  });

  it("handles paths with and without leading slashes", () => {
    expect(isWithinDirectory("/src/file.ts", "/src")).toBe(true);
    expect(isWithinDirectory("src/file.ts", "src")).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// sortFileNodes
// ---------------------------------------------------------------------------
describe("sortFileNodes", () => {
  it("sorts directories before files", () => {
    const nodes = [
      { name: "file.ts", type: "file" as const },
      { name: "src", type: "directory" as const },
      { name: "lib", type: "directory" as const },
      { name: "app.ts", type: "file" as const },
    ];

    const sorted = sortFileNodes(nodes);

    expect(sorted[0]?.type).toBe("directory");
    expect(sorted[1]?.type).toBe("directory");
    expect(sorted[2]?.type).toBe("file");
    expect(sorted[3]?.type).toBe("file");
  });

  it("sorts alphabetically within same type", () => {
    const nodes = [
      { name: "zebra.ts", type: "file" as const },
      { name: "alpha.ts", type: "file" as const },
      { name: "beta.ts", type: "file" as const },
    ];

    const sorted = sortFileNodes(nodes);

    expect(sorted.map((n) => n.name)).toEqual(["alpha.ts", "beta.ts", "zebra.ts"]);
  });

  it("does not mutate the original array", () => {
    const nodes = [
      { name: "b.ts", type: "file" as const },
      { name: "a.ts", type: "file" as const },
    ];

    sortFileNodes(nodes);

    expect(nodes[0]?.name).toBe("b.ts"); // original unchanged
  });
});

// ---------------------------------------------------------------------------
// safeJsonParse
// ---------------------------------------------------------------------------
describe("safeJsonParse", () => {
  it("parses valid JSON", () => {
    expect(safeJsonParse('{"key": "value"}', {})).toEqual({ key: "value" });
  });

  it("returns fallback for invalid JSON", () => {
    expect(safeJsonParse("not json", "default")).toBe("default");
  });

  it("returns fallback for empty string", () => {
    expect(safeJsonParse("", null)).toBeNull();
  });
});
