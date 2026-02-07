"use client";

/**
 * CodeEditor - Monaco editor wrapper for displaying code.
 * Features:
 * - Read-only mode for observing agent work
 * - Syntax highlighting for TypeScript, TSX, JavaScript, JSON, CSS, HTML
 * - Dark theme (vs-dark)
 * - Line numbers
 * - Current file path in header
 */

import * as React from "react";
import { cn } from "@/lib/utils";
import { getFileLanguage } from "@/lib/constants";

// Dynamic import for Monaco to avoid SSR issues
import dynamic from "next/dynamic";
import type { OnMount, Monaco } from "@monaco-editor/react";
import type { editor } from "monaco-editor";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center bg-card">
      <div className="flex items-center gap-2 text-foreground-muted">
        <LoadingSpinner />
        <span>Loading editor...</span>
      </div>
    </div>
  ),
});

interface CodeEditorProps {
  /** Path of the file being edited */
  filePath: string | null;
  /** Content of the file */
  content: string | null;
  /** Whether the editor is read-only (default: true) */
  readOnly?: boolean;
  /** Callback when content changes (only when readOnly is false) */
  onChange?: (value: string | undefined) => void;
  /** Callback to save the current file (Cmd+S / Ctrl+S) */
  onSave?: () => void;
  /** Whether the file has unsaved changes */
  dirty?: boolean;
  /** Whether a save is currently in progress */
  saving?: boolean;
  /** Additional className */
  className?: string;
}

export function CodeEditor({
  filePath,
  content,
  readOnly = true,
  onChange,
  onSave,
  dirty = false,
  saving = false,
  className,
}: CodeEditorProps): React.ReactElement {
  const editorRef = React.useRef<editor.IStandaloneCodeEditor | null>(null);
  const monacoRef = React.useRef<Monaco | null>(null);
  const onSaveRef = React.useRef(onSave);
  React.useEffect(() => {
    onSaveRef.current = onSave;
  }, [onSave]);

  // Determine language from file extension
  const language = filePath ? getFileLanguage(filePath) : "plaintext";

  // Handle Monaco editor mount
  const handleEditorMount: OnMount = React.useCallback(
    (editor, monaco) => {
      editorRef.current = editor;
      monacoRef.current = monaco;

      // Configure TypeScript/JavaScript settings
      monaco.languages.typescript.typescriptDefaults.setCompilerOptions({
        target: monaco.languages.typescript.ScriptTarget.ESNext,
        allowNonTsExtensions: true,
        moduleResolution:
          monaco.languages.typescript.ModuleResolutionKind.NodeJs,
        module: monaco.languages.typescript.ModuleKind.ESNext,
        noEmit: true,
        esModuleInterop: true,
        jsx: monaco.languages.typescript.JsxEmit.React,
        reactNamespace: "React",
        allowJs: true,
        strict: true,
        skipLibCheck: true,
      });

      monaco.languages.typescript.javascriptDefaults.setCompilerOptions({
        target: monaco.languages.typescript.ScriptTarget.ESNext,
        allowNonTsExtensions: true,
        moduleResolution:
          monaco.languages.typescript.ModuleResolutionKind.NodeJs,
        module: monaco.languages.typescript.ModuleKind.ESNext,
        noEmit: true,
        esModuleInterop: true,
        jsx: monaco.languages.typescript.JsxEmit.React,
        allowJs: true,
      });

      // Disable diagnostics for read-only mode to avoid error squiggles
      if (readOnly) {
        monaco.languages.typescript.typescriptDefaults.setDiagnosticsOptions({
          noSemanticValidation: true,
          noSyntaxValidation: false, // Keep syntax validation
        });
        monaco.languages.typescript.javascriptDefaults.setDiagnosticsOptions({
          noSemanticValidation: true,
          noSyntaxValidation: false,
        });
      }

      // Set editor focus only when editable to avoid stealing focus
      if (!readOnly) {
        editor.focus();
      }

      // Register Cmd+S / Ctrl+S keybinding for save
      editor.addAction({
        id: "save-file",
        label: "Save File",
        keybindings: [
          monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS,
        ],
        run: () => {
          onSaveRef.current?.();
        },
      });
    },
    [readOnly]
  );

  // When file changes, scroll to top
  React.useEffect(() => {
    if (editorRef.current) {
      editorRef.current.setScrollPosition({ scrollTop: 0 });
    }
  }, [filePath]);

  // Empty state
  if (!filePath || content === null) {
    return (
      <div
        className={cn(
          "flex h-full flex-col items-center justify-center bg-card text-foreground-muted",
          className
        )}
      >
        <FileIcon className="mb-3 h-12 w-12 opacity-50" />
        <p className="text-sm">Select a file to view its contents</p>
      </div>
    );
  }

  return (
    <div className={cn("flex h-full flex-col", className)}>
      {/* File path header */}
      <div className="flex items-center justify-between border-b border-border bg-card px-3 py-1.5">
        <div className="flex items-center gap-2">
          <FilePathIcon className="h-4 w-4 text-foreground-muted" />
          <span className="text-sm text-foreground">{filePath}</span>
        </div>
        <div className="flex items-center gap-2">
          <LanguageBadge language={language} />
          {readOnly ? (
            <span className="text-xs text-foreground-muted">(Read-only)</span>
          ) : saving ? (
            <span className="text-xs text-accent">Saving...</span>
          ) : dirty ? (
            <span className="text-xs text-warning">(Unsaved)</span>
          ) : null}
        </div>
      </div>

      {/* Monaco Editor */}
      <div className="flex-1">
        <MonacoEditor
          height="100%"
          language={language}
          value={content}
          theme="vs-dark"
          options={{
            readOnly,
            minimap: { enabled: false },
            lineNumbers: "on",
            scrollBeyondLastLine: false,
            fontSize: 13,
            fontFamily: "JetBrains Mono, Menlo, Monaco, monospace",
            fontLigatures: true,
            tabSize: 2,
            wordWrap: "on",
            automaticLayout: true,
            renderLineHighlight: "line",
            cursorBlinking: readOnly ? "solid" : "blink",
            folding: true,
            foldingHighlight: true,
            showFoldingControls: "mouseover",
            bracketPairColorization: { enabled: true },
            guides: {
              indentation: true,
              bracketPairs: true,
            },
            padding: { top: 8, bottom: 8 },
            smoothScrolling: true,
            contextmenu: !readOnly,
            quickSuggestions: !readOnly,
            suggestOnTriggerCharacters: !readOnly,
            parameterHints: { enabled: !readOnly },
            scrollbar: {
              vertical: "auto",
              horizontal: "auto",
              verticalScrollbarSize: 10,
              horizontalScrollbarSize: 10,
            },
          }}
          onChange={onChange}
          onMount={handleEditorMount}
        />
      </div>
    </div>
  );
}

/**
 * Language badge component
 */
function LanguageBadge({
  language,
}: {
  language: string;
}): React.ReactElement {
  const displayNames: Record<string, string> = {
    typescript: "TypeScript",
    javascript: "JavaScript",
    json: "JSON",
    css: "CSS",
    scss: "SCSS",
    html: "HTML",
    markdown: "Markdown",
    plaintext: "Text",
  };

  const colorClasses: Record<string, string> = {
    typescript: "bg-blue-500/20 text-blue-400",
    javascript: "bg-yellow-500/20 text-yellow-400",
    json: "bg-orange-500/20 text-orange-400",
    css: "bg-purple-500/20 text-purple-400",
    scss: "bg-pink-500/20 text-pink-400",
    html: "bg-red-500/20 text-red-400",
    markdown: "bg-green-500/20 text-green-400",
    plaintext: "bg-gray-500/20 text-gray-400",
  };

  return (
    <span
      className={cn(
        "rounded px-1.5 py-0.5 text-xs font-medium",
        colorClasses[language] ?? colorClasses.plaintext
      )}
    >
      {displayNames[language] ?? language}
    </span>
  );
}

/**
 * Loading spinner component
 */
function LoadingSpinner(): React.ReactElement {
  return (
    <svg
      className="h-5 w-5 animate-spin"
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
 * File icon for empty state
 */
function FileIcon({ className }: { className?: string }): React.ReactElement {
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
        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
      />
    </svg>
  );
}

/**
 * File path icon for header
 */
function FilePathIcon({
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
        d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
      />
    </svg>
  );
}

export default CodeEditor;
