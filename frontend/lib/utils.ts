/**
 * Utility functions for the frontend application.
 */

import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Combines class names with Tailwind CSS merge support.
 * Uses clsx for conditional classes and twMerge to handle Tailwind conflicts.
 *
 * @example
 * cn("px-4 py-2", isActive && "bg-blue-500", className)
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Format a timestamp as a human-readable time string.
 */
export function formatTime(timestamp: number): string {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

/**
 * Format a duration in seconds as a human-readable string.
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
}

/**
 * Format a number with thousand separators.
 */
export function formatNumber(num: number): string {
  return num.toLocaleString("en-US");
}

/**
 * Truncate a string to a maximum length with ellipsis.
 */
export function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) {
    return str;
  }
  return `${str.slice(0, maxLength - 3)}...`;
}

/**
 * Generate a unique ID for frontend use.
 */
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * Debounce a function call.
 */
export function debounce<T extends (...args: Parameters<T>) => void>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: NodeJS.Timeout | null = null;
  return (...args: Parameters<T>) => {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

/**
 * Extract file extension from a path.
 */
export function getFileExtension(path: string): string {
  const parts = path.split(".");
  return parts.length > 1 ? (parts.pop() ?? "") : "";
}

/**
 * Get the filename from a path.
 */
export function getFileName(path: string): string {
  return path.split("/").pop() ?? path;
}

/**
 * Get the directory from a path.
 */
export function getDirectory(path: string): string {
  const parts = path.split("/");
  parts.pop();
  return parts.join("/") || ".";
}

/**
 * Check if a path is within a directory.
 */
export function isWithinDirectory(path: string, directory: string): boolean {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const normalizedDir = directory.startsWith("/") ? directory : `/${directory}`;
  return normalizedPath.startsWith(normalizedDir);
}

/**
 * Sort file nodes with directories first, then alphabetically.
 */
export function sortFileNodes<T extends { name: string; type: "file" | "directory" }>(
  nodes: T[]
): T[] {
  return [...nodes].sort((a, b) => {
    // Directories first
    if (a.type === "directory" && b.type !== "directory") return -1;
    if (a.type !== "directory" && b.type === "directory") return 1;
    // Then alphabetically
    return a.name.localeCompare(b.name);
  });
}

/**
 * Parse JSON safely with a fallback value.
 */
export function safeJsonParse<T>(json: string, fallback: T): T {
  try {
    return JSON.parse(json) as T;
  } catch {
    return fallback;
  }
}

/**
 * Sleep for a given number of milliseconds.
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Create initials from a name (e.g., "Orchestrator Agent" -> "OA").
 */
export function getInitials(name: string, maxLength: number = 2): string {
  const words = name
    .trim()
    .split(/\s+/)
    .filter((word) => word.length > 0);

  return words
    .map((word) => word.charAt(0).toUpperCase())
    .slice(0, maxLength)
    .join("");
}
