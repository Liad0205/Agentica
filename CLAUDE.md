# Agent Swarm POC

## Project Overview
A proof-of-concept comparing three AI-assisted coding paradigms: Single ReAct Agent, Task Decomposition Swarm, and Parallel Hypothesis Testing. The application provides a visual "mission control" interface to observe agents working in real-time.

## Tech Stack
- **Frontend:** Next.js 15, React 19, TypeScript 5.x, Tailwind CSS 4, shadcn/ui, React Flow 12, Monaco Editor, xterm.js
- **Backend:** Python 3.12, FastAPI 0.115+, LangGraph 0.2+, LiteLLM, docker-py, Pydantic v2
- **Package Management:** UV for Python, npm for Node.js
- **Infrastructure:** Docker containers for sandboxed code execution

## Project Structure
```
agent-swarm-poc/
├── frontend/                 # Next.js application
│   ├── app/                  # App router pages
│   │   ├── layout.tsx        # Root layout with fonts and metadata
│   │   └── page.tsx          # Main workspace page (integrates all components)
│   ├── components/
│   │   ├── ErrorBoundary.tsx  # React error boundary for graceful UI error handling
│   │   ├── layout/           # Header, WorkspaceLayout, PromptBar
│   │   ├── messages/         # MessagePanel, MessageItem, EventCard
│   │   ├── graph/            # AgentGraph, custom nodes (AgentNode, OrchestratorNode, etc.)
│   │   ├── code/             # CodePanel, FileTree, EditorPane, PreviewPane, TerminalPane
│   │   ├── mode/             # ModeSelector, ModeComparisonPanel
│   │   └── ui/               # shadcn/ui components (button, card, tabs, etc.)
│   ├── hooks/                # Custom React hooks
│   │   ├── useSession.ts     # Session lifecycle, WebSocket connection
│   │   ├── useMessages.ts    # Events and agent messages from store
│   │   ├── useAgentGraph.ts  # Graph visualization state by mode
│   │   ├── useCodePanel.ts   # File tree, editor, preview, terminal state
│   │   ├── usePrompt.ts      # Prompt input with submit/cancel
│   │   └── index.ts          # Re-exports all hooks
│   ├── lib/
│   │   ├── store.ts          # Zustand global state with processEvent dispatcher
│   │   ├── websocket.ts      # WebSocket client singleton with reconnect
│   │   ├── api.ts            # HTTP API client wrapper
│   │   ├── types.ts          # All TypeScript types and interfaces
│   │   ├── constants.ts      # Colors, endpoints, config defaults
│   │   └── utils.ts          # Utility functions (cn for classnames)
│   └── types/
│       └── css.d.ts          # CSS module type declarations
├── backend/
│   ├── main.py               # FastAPI app entry point
│   ├── api/
│   │   ├── routes.py         # HTTP endpoints
│   │   └── websocket.py      # WebSocket handler
│   ├── config.py              # Centralized configuration (env vars, defaults)
│   ├── agents/
│   │   ├── react_graph.py    # ReAct agent LangGraph implementation
│   │   ├── decomposition_graph.py  # Task Decomposition LangGraph (orchestrate → execute → aggregate → review)
│   │   ├── hypothesis_graph.py     # Parallel Hypothesis LangGraph (broadcast → solve → evaluate → finalize)
│   │   ├── tools.py          # Agent tools (file ops, commands)
│   │   ├── prompts.py        # System prompts for all modes
│   │   └── utils.py          # Agent utilities
│   ├── sandbox/
│   │   └── manager.py        # Docker sandbox management
│   ├── events/
│   │   ├── bus.py            # Event bus for pub/sub
│   │   └── types.py          # Event type definitions
│   └── models/
│       └── schemas.py        # Pydantic request/response models
└── sandbox-image/            # Docker context for sandbox containers
```

## Development Commands
```bash
# Backend (uses UV)
cd backend && uv run uvicorn main:app --reload

# Frontend
cd frontend && npm run dev

# Build frontend
cd frontend && npm run build

# Full stack
make dev

# Tests
cd backend && uv run pytest
cd frontend && npm test

# Build sandbox image
make sandbox-image
```

## Code Standards

### Python
- Use UV for all package management (`uv add`, `uv run`)
- Strict type hints on all functions
- Async/await for all I/O operations
- Use structlog for structured JSON logging
- Pydantic v2 for all data models

### TypeScript
- Strict mode enabled (`"strict": true`, `"noUncheckedIndexedAccess": true`)
- No `any` types - use `unknown` for truly unknown types
- Explicit return types on all functions: `React.ReactElement` (not `JSX.Element` - React 19 change)
- Use discriminated unions for type safety
- Interface data types for React Flow must extend `Record<string, unknown>`

### React
- Functional components only (no class components)
- Hooks for all state and side effects
- Zustand for global state management
- React Flow 12 for graph visualization
- All components use `"use client"` directive for client-side rendering

### Styling
- Tailwind CSS 4 with `@tailwindcss/postcss` plugin (not direct `tailwindcss`)
- "Mission Control" design aesthetic:
  - Background: #0a0a0f (near-black)
  - Card surfaces: #12121a
  - Accent: #00d4ff (electric cyan)
  - Warning: #f59e0b (amber)
  - Success: #10b981 (green)
  - Error: #ef4444 (red)
- JetBrains Mono for code, Geist Sans for UI

## Architecture

### Frontend State Management
```
Zustand Store (lib/store.ts)
├── Session state (sessionId, status, mode, task)
├── Connection state (WebSocket status, errors)
├── Events array (all AgentEvents)
├── Agents map (id -> AgentInfo)
├── Mode-specific state (ReAct, Decomposition, Hypothesis)
├── File state (tree, content, active path)
├── Sandbox state (list, selected, preview URL)
├── Terminal output
└── Prompt value

Key action: processEvent(event) - dispatches all WebSocket events to update state
```

### Event Flow
```
Backend LangGraph Node
    ↓
EventBus.publish(event)
    ↓
WebSocket broadcast
    ↓
Frontend wsClient.onEvent()
    ↓
store.processEvent(event)
    ↓
UI re-renders via hooks
```

### Error Boundaries
The frontend uses React error boundaries (`components/ErrorBoundary.tsx`) to catch and display runtime errors gracefully, preventing full-page crashes during agent execution or rendering failures.

### Custom Hooks Pattern
Each hook selects specific slices from the store:
- `useSession()` - session lifecycle + WebSocket management
- `useMessages()` - events array + agents list
- `useAgentGraph()` - mode-specific graph state
- `useCodePanel()` - files, editor, preview, terminal
- `usePrompt()` - input value + submit/cancel actions

### Sandbox Communication
All agent-sandbox communication goes through `SandboxManager`:
```python
sandbox_manager.write_file(sandbox_id, path, content)
sandbox_manager.read_file(sandbox_id, path)
sandbox_manager.execute_command(sandbox_id, command)
sandbox_manager.start_dev_server(sandbox_id)
```

### Security
- Command execution policy is configurable (`SANDBOX_ALLOW_UNRESTRICTED_COMMANDS`)
- Prevent path traversal (no `..` in paths)
- Containers run as non-root with dropped capabilities
- Sandbox containers are isolated and resource-limited

## Environment Variables
Create `.env` in project root:
```
XAI_API_KEY=xai-...           # For Grok models
GEMINI_API_KEY=AIzaSy...      # For Gemini models
```

## Three Agent Modes

### 1. ReAct (Single Agent) - IMPLEMENTED
Simple reason→act→review loop. One agent, one sandbox.
- Graph: User → Agent (with status indicator) → Result
- Agent cycles through: idle → reasoning → executing → complete

### 2. Task Decomposition - IMPLEMENTED
Orchestrator decomposes task → parallel sub-agents execute → aggregator merges → integration review.
- Graph: User → Orchestrator → [Sub-agents] → Aggregator → Result
- LangGraph: orchestrate → execute_subtask → aggregate → integration_review
- Implementation: `backend/agents/decomposition_graph.py`

### 3. Parallel Hypothesis - IMPLEMENTED
N agents independently solve the full task → evaluator scores and picks winner.
- Graph: User → [Solver 1, 2, 3...] → Evaluator → Winner highlighted
- LangGraph: broadcast → solve → evaluate → finalize
- Implementation: `backend/agents/hypothesis_graph.py`

## API Endpoints
- `POST /api/sessions` - Create new session with mode and task
- `GET /api/sessions/{id}` - Get session details
- `POST /api/sessions/{id}/cancel` - Cancel running session
- `GET /api/sessions/{id}/files` - Get file tree for sandbox
- `GET /api/sessions/{id}/files/content` - Get file content
- `GET /api/sessions/{id}/metrics` - Get token usage
- `GET /health` - Health check
- `WS /ws/{session_id}` - Real-time event stream

## Event Types
Key events the frontend handles:
- `session_started`, `session_complete`, `session_error`
- `agent_spawned`, `agent_thinking`, `agent_tool_call`, `agent_tool_result`, `agent_complete`
- `file_changed`, `file_deleted`
- `command_started`, `command_output`, `command_complete`
- `preview_starting`, `preview_ready`, `preview_error`
- `orchestrator_plan`, `aggregation_started`, `aggregation_complete`
- `evaluation_started`, `evaluation_result`

## Common Issues & Fixes

### React 19 + TypeScript
- Use `React.ReactElement` instead of `JSX.Element` for return types
- React Flow node/edge data interfaces must extend `Record<string, unknown>`
- Import React namespace: `import * as React from "react"`

### Tailwind CSS 4
- Use `@tailwindcss/postcss` in postcss.config.js, not `tailwindcss` directly
- PostCSS config: `{ "@tailwindcss/postcss": {} }`

### Pydantic v2 `model_config` Conflict
- Pydantic v2 reserves `model_config` as a class-level attribute (`ConfigDict`). The API uses `llm_config` as the field alias for LLM configuration in session creation requests. Always use `llm_config` in API payloads, not `model_config`.

### useEffect Return Values
- All useEffect callbacks must return consistently (cleanup function or undefined)
- Early returns should use `return;` not `return () => {}`

## Testing
- Backend: pytest + pytest-asyncio, 80%+ coverage on critical paths
- Frontend: Vitest + React Testing Library
- E2E: Playwright
- Use `USE_MOCK_LLM=true` for testing without API costs

## Running E2E Test
1. Build sandbox image: `make sandbox-image`
2. Start backend: `cd backend && uv run uvicorn main:app --reload`
3. Start frontend: `cd frontend && npm run dev`
4. Open http://localhost:3000
5. Select mode, enter task, observe agents working
