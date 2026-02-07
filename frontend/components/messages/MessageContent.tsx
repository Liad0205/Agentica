"use client";

import * as React from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

// Import styles for syntax highlighting
// You might need to add a css import in your global css or layout
// import "highlight.js/styles/github-dark.css";

interface MessageContentProps {
  content: string;
  className?: string;
}

/**
 * Robustly renders message content:
 * 1. Checks if content is valid JSON -> renders as highlighted code
 * 2. Renders as Markdown with GFM and syntax highlighting
 * 3. Handles <plan> tags specially
 */
export function MessageContent({
  content,
  className,
}: MessageContentProps): React.ReactElement {
  // 1. Try to parse as JSON
  const jsonContent = React.useMemo(() => {
    try {
      const parsed = JSON.parse(content);
      // Only treat as JSON if it's an object or array, not a primitive
      if (typeof parsed === "object" && parsed !== null) {
        return JSON.stringify(parsed, null, 2);
      }
    } catch {
      // Not JSON
    }
    return null;
  }, [content]);

  // 2. Handle specific tags like <plan>
  // We'll replace <plan>...</plan> with a styled div for markdown to render inside
  // or use a custom component. For simplicity/robustness, let's pre-process the string
  // to turn <plan> into a blockquote or similar if react-markdown doesn't handle custom HTML tags well by default.
  // Actually, react-markdown handles HTML if rehype-raw is used, but for security we might want to avoid that.
  // Let's us a simple regex replacement to turn <plan> into a visually distinct block.

  const processedContent = React.useMemo(() => {
    if (jsonContent) return null;

    let processed = content;

    // Generic tag handling: <tagName>content</tagName>
    // We replace them with a blockquote structure with a header
    // The regex captures: 1. Tag name, 2. Content inside
    processed = processed.replace(
      // Match <tag>...</tag> across multiple lines
      // \s\S includes newlines
      /<(\w+)>([\s\S]*?)<\/\1>/g,
      (match, tagName, content) => {
        const title = tagName.toUpperCase();
        // Add newlines to ensure markdown parses the blockquote correctly
        return `\n> **${title}**\n> ${content.trim().replace(/\n/g, "\n> ")}\n\n`;
      },
    );

    return processed;
  }, [content, jsonContent]);

  if (jsonContent) {
    return (
      <div className={cn("text-xs font-mono", className)}>
        <pre className="p-3 rounded bg-zinc-950/50 border border-border/50 overflow-x-auto">
          <code className="language-json">
            {jsonContent}
          </code>
        </pre>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "markdown-content space-y-2",
        className,
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          // Custom renderers for specific elements
          pre({ children, ...props }) {
            return (
              <pre
                {...props}
                className="p-3 my-2 rounded bg-zinc-950/50 border border-border/50 overflow-x-auto"
              >
                {children}
              </pre>
            );
          },
          code({ className, children, ...props }) {
            const match = /language-(\w+)/.exec(
              className || "",
            );
            const isInline =
              !match && !String(children).includes("\n");

            if (isInline) {
              return (
                <code
                  className={cn(
                    "px-1.5 py-0.5 rounded bg-zinc-800/50 text-accent/90 font-mono text-xs",
                    className,
                  )}
                  {...props}
                >
                  {children}
                </code>
              );
            }

            return (
              <code
                className={cn(
                  "text-xs font-mono text-foreground/90",
                  className,
                )}
                {...props}
              >
                {children}
              </code>
            );
          },
          // Style <plan> if we decide to treat it as a custom element (requires rehype-raw which we avoid)
          // Instead, we rely on the pre-processing or just Markdown parsing.

          // Style formatting
          h1: ({ children }) => (
            <h1 className="text-lg font-bold mt-4 mb-2">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-base font-bold mt-3 mb-2">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-sm font-bold mt-2 mb-1">
              {children}
            </h3>
          ),
          ul: ({ children }) => (
            <ul className="list-disc pl-4 space-y-1">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal pl-4 space-y-1">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="pl-1">{children}</li>
          ),
          p: ({ children }) => (
            <p className="mb-2 last:mb-0 leading-relaxed">
              {children}
            </p>
          ),
          a: ({ href, children }) => (
            <a
              href={href}
              className="text-accent hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              {children}
            </a>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-accent/50 pl-3 py-1 my-2 bg-accent/5 text-accent/90 italic">
              {children}
            </blockquote>
          ),
        }}
      >
        {processedContent || ""}
      </ReactMarkdown>
    </div>
  );
}
