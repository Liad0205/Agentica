"use client";

import * as React from "react";
import {
  BarChart3,
  Brain,
  CheckCircle2,
  Clock,
  GitBranch,
  RefreshCw,
  Sparkles,
  X,
  XCircle,
  Zap,
} from "lucide-react";
import { api } from "@/lib/api";
import { MODE_INFO } from "@/lib/constants";
import { cn, formatDuration, formatNumber, truncate } from "@/lib/utils";
import type { Mode, SessionStatus, SessionSummary } from "@/lib/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ModeComparisonPanelProps {
  open: boolean;
  onClose: () => void;
}

interface ModeAggregate {
  runs: number;
  complete: number;
  errored: number;
  totalDuration: number;
  totalTokens: number;
  totalLlmCalls: number;
  totalToolCalls: number;
}

type PanelTab = "overview" | "history";

interface ModeDetail {
  mode: Mode;
  icon: React.ReactNode;
  tagline: string;
  strengths: string[];
  weaknesses: string[];
  bestFor: string[];
  architecture: string[];
}

// ---------------------------------------------------------------------------
// Static mode detail data
// ---------------------------------------------------------------------------

const MODE_DETAILS: Record<Mode, ModeDetail> = {
  react: {
    mode: "react",
    icon: <Brain className="h-5 w-5" />,
    tagline: "Single agent with a reason-act-review loop",
    strengths: [
      "Simple and predictable execution flow",
      "Low overhead -- one sandbox, one LLM thread",
      "Easiest to debug; linear chain of thought",
      "Best token efficiency for small tasks",
    ],
    weaknesses: [
      "Scales poorly to complex multi-file tasks",
      "No parallelism; wall-clock time grows linearly",
      "Single point of failure -- one error can derail the run",
    ],
    bestFor: [
      "Quick prototypes and single-file changes",
      "Bug fixes with a clear root cause",
      "Small utilities and scripts",
      "Learning and experimentation",
    ],
    architecture: [
      "User",
      "  |",
      "  v",
      "[ ReAct Agent ]  <-- reason / act / review loop",
      "  |",
      "  v",
      "Result",
    ],
  },
  decomposition: {
    mode: "decomposition",
    icon: <GitBranch className="h-5 w-5" />,
    tagline: "Orchestrator decomposes task into parallel sub-agents",
    strengths: [
      "Parallelism across independent sub-tasks",
      "Orchestrator can allocate the right model per sub-task",
      "Aggregator merges results for a coherent output",
      "Scales well for multi-component projects",
    ],
    weaknesses: [
      "Higher token cost due to orchestration overhead",
      "Plan quality depends heavily on the orchestrator LLM",
      "Merge conflicts possible when sub-agents touch shared files",
      "Longer cold-start while the plan is generated",
    ],
    bestFor: [
      "Full-stack features (frontend + backend + tests)",
      "Multi-file refactors with clear ownership boundaries",
      "Projects with separable components",
      "Teams that need structured task breakdowns",
    ],
    architecture: [
      "        User",
      "          |",
      "          v",
      "   [ Orchestrator ]  <-- generates plan",
      "    /     |      \\",
      "   v      v       v",
      "[Sub-1] [Sub-2] [Sub-N]  <-- parallel execution",
      "   \\      |      /",
      "    v     v     v",
      "   [ Aggregator ]  <-- merges results",
      "         |",
      "         v",
      "   [ Integration ]  <-- final review",
      "         |",
      "         v",
      "       Result",
    ],
  },
  hypothesis: {
    mode: "hypothesis",
    icon: <Sparkles className="h-5 w-5" />,
    tagline: "N agents compete; evaluator picks the best solution",
    strengths: [
      "Explores multiple approaches simultaneously",
      "Evaluator selects the highest-quality output",
      "Resilient to individual agent failures",
      "Great for subjective or open-ended tasks",
    ],
    weaknesses: [
      "Highest token cost -- N full solutions generated",
      "Redundant work across solvers",
      "Evaluator quality bottleneck",
      "Needs enough diversity in solver approaches to be valuable",
    ],
    bestFor: [
      "UI/UX tasks where aesthetics matter",
      "Algorithm design with multiple valid approaches",
      "Creative coding challenges",
      "High-stakes tasks where quality outweighs cost",
    ],
    architecture: [
      "          User",
      "            |",
      "            v",
      "       [ Broadcast ]",
      "      /    |    \\",
      "     v     v     v",
      "[Solver 1][Solver 2][Solver 3]  <-- independent",
      "     \\     |     /",
      "      v    v    v",
      "     [ Evaluator ]  <-- scores & selects",
      "          |",
      "          v",
      "    Winner highlighted",
    ],
  },
};

const MODES: Mode[] = ["react", "decomposition", "hypothesis"];

const TERMINAL_STATUSES: Set<SessionStatus> = new Set([
  "complete",
  "error",
  "cancelled",
]);

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function ModeComparisonPanel({
  open,
  onClose,
}: ModeComparisonPanelProps): React.ReactElement | null {
  const [sessions, setSessions] = React.useState<SessionSummary[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [activeTab, setActiveTab] = React.useState<PanelTab>("overview");

  const fetchSessions = React.useCallback(async (): Promise<void> => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listSessions(40);
      setSessions(data);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to load sessions";
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    if (open) {
      void fetchSessions();
    }
  }, [open, fetchSessions]);

  React.useEffect(() => {
    if (!open) {
      return;
    }

    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open, onClose]);

  const aggregates = React.useMemo((): Record<Mode, ModeAggregate> => {
    const result: Record<Mode, ModeAggregate> = {
      react: { runs: 0, complete: 0, errored: 0, totalDuration: 0, totalTokens: 0, totalLlmCalls: 0, totalToolCalls: 0 },
      decomposition: { runs: 0, complete: 0, errored: 0, totalDuration: 0, totalTokens: 0, totalLlmCalls: 0, totalToolCalls: 0 },
      hypothesis: { runs: 0, complete: 0, errored: 0, totalDuration: 0, totalTokens: 0, totalLlmCalls: 0, totalToolCalls: 0 },
    };

    for (const session of sessions) {
      const target = result[session.mode];
      target.runs += 1;
      if (session.status === "complete") {
        target.complete += 1;
      }
      if (session.status === "error") {
        target.errored += 1;
      }
      if (TERMINAL_STATUSES.has(session.status)) {
        target.totalDuration += session.metrics.execution_time_seconds;
      }
      target.totalTokens +=
        session.metrics.total_input_tokens +
        session.metrics.total_output_tokens;
      target.totalLlmCalls += session.metrics.total_llm_calls;
      target.totalToolCalls += session.metrics.total_tool_calls;
    }

    return result;
  }, [sessions]);

  const sessionsByMode = React.useMemo((): Record<Mode, SessionSummary[]> => {
    const result: Record<Mode, SessionSummary[]> = {
      react: [],
      decomposition: [],
      hypothesis: [],
    };
    for (const session of sessions) {
      result[session.mode].push(session);
    }
    return result;
  }, [sessions]);

  if (!open) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
      onClick={onClose}
      role="presentation"
    >
      <aside
        className="absolute inset-y-0 right-0 flex w-full max-w-6xl flex-col border-l border-border bg-card shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-5 py-4">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-accent" />
            <h2 className="text-base font-semibold text-foreground">
              Mode Comparison
            </h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => void fetchSessions()}
              className={cn(
                "inline-flex items-center gap-2 rounded-md border border-border px-3 py-1.5 text-sm",
                "text-foreground-muted transition-colors hover:bg-background hover:text-foreground"
              )}
            >
              <RefreshCw
                className={cn("h-4 w-4", loading && "animate-spin")}
              />
              Refresh
            </button>
            <button
              type="button"
              onClick={onClose}
              className={cn(
                "rounded-md p-1.5 text-foreground-muted transition-colors",
                "hover:bg-background hover:text-foreground"
              )}
              aria-label="Close comparison panel"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Tab bar */}
        <div className="flex border-b border-border px-5">
          <TabButton
            label="Overview"
            isActive={activeTab === "overview"}
            onClick={() => setActiveTab("overview")}
          />
          <TabButton
            label="Session History"
            isActive={activeTab === "history"}
            onClick={() => setActiveTab("history")}
          />
        </div>

        {/* Content */}
        <div className="min-h-0 flex-1 overflow-y-auto">
          {activeTab === "overview" ? (
            <OverviewTab aggregates={aggregates} />
          ) : (
            <HistoryTab
              sessions={sessions}
              sessionsByMode={sessionsByMode}
              aggregates={aggregates}
              loading={loading}
              error={error}
            />
          )}
        </div>
      </aside>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab button
// ---------------------------------------------------------------------------

function TabButton({
  label,
  isActive,
  onClick,
}: {
  label: string;
  isActive: boolean;
  onClick: () => void;
}): React.ReactElement {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "relative px-4 py-3 text-sm font-medium transition-colors",
        isActive
          ? "text-accent"
          : "text-foreground-muted hover:text-foreground"
      )}
    >
      {label}
      {isActive && (
        <span className="absolute inset-x-0 bottom-0 h-0.5 bg-accent" />
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Overview tab -- static mode info + aggregate metrics in three columns
// ---------------------------------------------------------------------------

function OverviewTab({
  aggregates,
}: {
  aggregates: Record<Mode, ModeAggregate>;
}): React.ReactElement {
  return (
    <div className="grid grid-cols-1 gap-4 p-5 md:grid-cols-3">
      {MODES.map((mode) => (
        <ModeColumn
          key={mode}
          detail={MODE_DETAILS[mode]}
          aggregate={aggregates[mode]}
        />
      ))}
    </div>
  );
}

function ModeColumn({
  detail,
  aggregate,
}: {
  detail: ModeDetail;
  aggregate: ModeAggregate;
}): React.ReactElement {
  const successRate =
    aggregate.runs === 0
      ? 0
      : Math.round((aggregate.complete / aggregate.runs) * 100);
  const avgDuration =
    aggregate.runs === 0 ? 0 : aggregate.totalDuration / aggregate.runs;
  const avgTokens =
    aggregate.runs === 0
      ? 0
      : Math.round(aggregate.totalTokens / aggregate.runs);

  return (
    <div className="flex flex-col gap-4 rounded-xl border border-border bg-background/40 p-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent/10 text-accent">
          {detail.icon}
        </div>
        <div>
          <h3 className="text-sm font-semibold text-foreground">
            {MODE_INFO[detail.mode].name}
          </h3>
          <p className="text-xs text-foreground-muted">{detail.tagline}</p>
        </div>
      </div>

      {/* Aggregate metrics */}
      {aggregate.runs > 0 && (
        <div className="grid grid-cols-2 gap-2">
          <MetricTile label="Runs" value={String(aggregate.runs)} />
          <MetricTile
            label="Success"
            value={`${successRate}%`}
            highlight={successRate >= 75 ? "success" : successRate >= 40 ? "warning" : "error"}
          />
          <MetricTile label="Avg time" value={formatDuration(avgDuration)} />
          <MetricTile label="Avg tokens" value={formatNumber(avgTokens)} />
          <MetricTile
            label="LLM calls"
            value={formatNumber(aggregate.totalLlmCalls)}
          />
          <MetricTile
            label="Tool calls"
            value={formatNumber(aggregate.totalToolCalls)}
          />
        </div>
      )}

      {aggregate.runs === 0 && (
        <div className="rounded-lg border border-border/60 bg-card/60 px-3 py-2 text-xs text-foreground-muted">
          No sessions recorded yet
        </div>
      )}

      {/* Strengths */}
      <Section title="Strengths" variant="success">
        {detail.strengths.map((s) => (
          <BulletItem key={s} text={s} variant="success" />
        ))}
      </Section>

      {/* Weaknesses */}
      <Section title="Weaknesses" variant="error">
        {detail.weaknesses.map((w) => (
          <BulletItem key={w} text={w} variant="error" />
        ))}
      </Section>

      {/* Best for */}
      <Section title="When to use" variant="accent">
        {detail.bestFor.map((b) => (
          <BulletItem key={b} text={b} variant="accent" />
        ))}
      </Section>

      {/* Architecture */}
      <div>
        <SectionTitle text="Architecture" />
        <pre className="mt-1.5 overflow-x-auto rounded-lg border border-border/60 bg-[#0a0a0f] px-3 py-2 font-mono text-[11px] leading-relaxed text-foreground-muted">
          {detail.architecture.join("\n")}
        </pre>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// History tab -- session list grouped by mode + aggregates
// ---------------------------------------------------------------------------

function HistoryTab({
  sessions,
  sessionsByMode,
  aggregates,
  loading,
  error,
}: {
  sessions: SessionSummary[];
  sessionsByMode: Record<Mode, SessionSummary[]>;
  aggregates: Record<Mode, ModeAggregate>;
  loading: boolean;
  error: string | null;
}): React.ReactElement {
  return (
    <div className="p-5">
      {/* Aggregate summary cards */}
      <div className="mb-5 grid grid-cols-1 gap-3 md:grid-cols-3">
        {MODES.map((mode) => (
          <AggregateSummaryCard
            key={mode}
            mode={mode}
            aggregate={aggregates[mode]}
          />
        ))}
      </div>

      {/* Session list */}
      {loading && sessions.length === 0 ? (
        <PanelMessage text="Loading session history..." />
      ) : error ? (
        <PanelMessage text={error} variant="error" />
      ) : sessions.length === 0 ? (
        <PanelMessage text="No sessions found yet. Run a task to start comparing modes." />
      ) : (
        <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
          {MODES.map((mode) => (
            <div key={mode} className="flex flex-col gap-2">
              <h3 className="flex items-center gap-2 text-sm font-semibold text-foreground">
                <span className="text-accent">{MODE_DETAILS[mode].icon}</span>
                {MODE_INFO[mode].name}
                <span className="text-xs font-normal text-foreground-muted">
                  ({sessionsByMode[mode].length})
                </span>
              </h3>
              {sessionsByMode[mode].length === 0 ? (
                <div className="rounded-lg border border-border/60 bg-background/40 px-3 py-2 text-xs text-foreground-muted">
                  No sessions
                </div>
              ) : (
                sessionsByMode[mode].map((session) => (
                  <SessionRow key={session.session_id} session={session} />
                ))
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Aggregate summary card (used in history tab)
// ---------------------------------------------------------------------------

function AggregateSummaryCard({
  mode,
  aggregate,
}: {
  mode: Mode;
  aggregate: ModeAggregate;
}): React.ReactElement {
  const successRate =
    aggregate.runs === 0
      ? 0
      : Math.round((aggregate.complete / aggregate.runs) * 100);
  const avgDuration =
    aggregate.runs === 0 ? 0 : aggregate.totalDuration / aggregate.runs;
  const avgTokens =
    aggregate.runs === 0
      ? 0
      : Math.round(aggregate.totalTokens / aggregate.runs);

  return (
    <div className="rounded-lg border border-border bg-background/60 p-3">
      <div className="flex items-center gap-2">
        <span className="text-accent">{MODE_DETAILS[mode].icon}</span>
        <span className="text-sm font-medium text-foreground">
          {MODE_INFO[mode].name}
        </span>
      </div>
      <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
        <MetricTile label="Runs" value={String(aggregate.runs)} compact />
        <MetricTile
          label="Success"
          value={`${successRate}%`}
          compact
          highlight={
            aggregate.runs === 0
              ? undefined
              : successRate >= 75
                ? "success"
                : successRate >= 40
                  ? "warning"
                  : "error"
          }
        />
        <MetricTile label="Avg time" value={formatDuration(avgDuration)} compact />
      </div>
      <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-foreground-muted">
        <span>Avg tokens: {formatNumber(avgTokens)}</span>
        <span>LLM calls: {formatNumber(aggregate.totalLlmCalls)}</span>
        <span>Tool calls: {formatNumber(aggregate.totalToolCalls)}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Session row
// ---------------------------------------------------------------------------

function SessionRow({
  session,
}: {
  session: SessionSummary;
}): React.ReactElement {
  const totalTokens =
    session.metrics.total_input_tokens + session.metrics.total_output_tokens;

  return (
    <div className="rounded-lg border border-border bg-background/50 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge status={session.status} />
        <span className="text-[11px] text-foreground-muted">
          {formatDate(session.created_at)}
        </span>
      </div>

      <div className="mt-1.5 text-xs text-foreground">
        {truncate(session.task, 100)}
      </div>

      <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-foreground-muted">
        <span className="inline-flex items-center gap-1">
          <Clock className="h-3 w-3" />
          {formatDuration(session.metrics.execution_time_seconds)}
        </span>
        <span className="inline-flex items-center gap-1">
          <Zap className="h-3 w-3" />
          {session.metrics.total_llm_calls} LLM
        </span>
        <span>{formatNumber(totalTokens)} tok</span>
      </div>

      {session.error_message && (
        <div className="mt-1.5 rounded border border-error/40 bg-error/10 px-2 py-1 text-[11px] text-error">
          {truncate(session.error_message, 160)}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared sub-components
// ---------------------------------------------------------------------------

function MetricTile({
  label,
  value,
  highlight,
  compact = false,
}: {
  label: string;
  value: string;
  highlight?: "success" | "warning" | "error";
  compact?: boolean;
}): React.ReactElement {
  const highlightClass: Record<string, string> = {
    success: "text-success",
    warning: "text-warning",
    error: "text-error",
  };

  return (
    <div
      className={cn(
        "rounded border border-border/70 bg-card",
        compact ? "px-2 py-1" : "px-2.5 py-1.5"
      )}
    >
      <div className="text-[10px] uppercase tracking-wide text-foreground-muted">
        {label}
      </div>
      <div
        className={cn(
          "font-medium text-foreground",
          compact ? "text-xs" : "text-sm",
          highlight ? highlightClass[highlight] : undefined
        )}
      >
        {value}
      </div>
    </div>
  );
}

function SectionTitle({ text }: { text: string }): React.ReactElement {
  return (
    <h4 className="text-[11px] font-semibold uppercase tracking-wider text-foreground-muted">
      {text}
    </h4>
  );
}

function Section({
  title,
  variant,
  children,
}: {
  title: string;
  variant: "success" | "error" | "accent";
  children: React.ReactNode;
}): React.ReactElement {
  const borderColor: Record<string, string> = {
    success: "border-success/20",
    error: "border-error/20",
    accent: "border-accent/20",
  };

  return (
    <div>
      <SectionTitle text={title} />
      <ul
        className={cn(
          "mt-1.5 space-y-1 rounded-lg border bg-card/40 px-3 py-2",
          borderColor[variant]
        )}
      >
        {children}
      </ul>
    </div>
  );
}

function BulletItem({
  text,
  variant,
}: {
  text: string;
  variant: "success" | "error" | "accent";
}): React.ReactElement {
  const iconMap: Record<string, React.ReactNode> = {
    success: <CheckCircle2 className="mt-0.5 h-3 w-3 shrink-0 text-success" />,
    error: <XCircle className="mt-0.5 h-3 w-3 shrink-0 text-error" />,
    accent: <Zap className="mt-0.5 h-3 w-3 shrink-0 text-accent" />,
  };

  return (
    <li className="flex items-start gap-2 text-xs text-foreground-muted">
      {iconMap[variant]}
      <span>{text}</span>
    </li>
  );
}

function StatusBadge({
  status,
}: {
  status: SessionStatus;
}): React.ReactElement {
  const config: Record<SessionStatus, string> = {
    idle: "border-border text-foreground-muted",
    started: "border-accent/40 text-accent",
    running: "border-blue-400/40 text-blue-300",
    complete: "border-success/40 text-success",
    error: "border-error/40 text-error",
    cancelled: "border-warning/40 text-warning",
  };

  return (
    <span className={cn("rounded border px-2 py-0.5 text-[11px]", config[status])}>
      {status}
    </span>
  );
}

function PanelMessage({
  text,
  variant = "info",
}: {
  text: string;
  variant?: "info" | "error";
}): React.ReactElement {
  return (
    <div
      className={cn(
        "rounded-lg border px-4 py-3 text-sm",
        variant === "error"
          ? "border-error/40 bg-error/10 text-error"
          : "border-border bg-background text-foreground-muted"
      )}
    >
      {text}
    </div>
  );
}

function formatDate(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString();
}
