# Contributing to Agent Swarm POC

Thank you for your interest in contributing! This guide will help you get started.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (for sandboxed code execution)
- [Node.js](https://nodejs.org/) 20+
- [Python](https://www.python.org/) 3.12+
- [UV](https://docs.astral.sh/uv/) (Python package manager)

## Development Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/agent-swarm-poc.git
   cd agent-swarm-poc
   ```

2. **Build the sandbox Docker image**

   ```bash
   make sandbox-image
   ```

3. **Install backend dependencies**

   ```bash
   make backend-sync
   ```

4. **Install frontend dependencies**

   ```bash
   cd frontend && npm install
   ```

5. **Configure environment variables**

   ```bash
   cp env.example .env
   ```

   Edit `.env` and add your API keys.

6. **Start the development servers**

   ```bash
   make dev
   ```

   This starts both the backend (FastAPI) and frontend (Next.js) servers.

## Code Standards

### Python

- Use **UV** for all package management (`uv add`, `uv run`)
- Strict **type hints** on all functions
- **Async/await** for all I/O operations
- **structlog** for structured JSON logging
- **Pydantic v2** for all data models

### TypeScript

- **Strict mode** enabled (`"strict": true`)
- No `any` types -- use `unknown` for truly unknown types
- **Explicit return types** on all functions
- Use **discriminated unions** for type safety

## Running Tests

Run all tests:

```bash
make test
```

Or run tests individually:

```bash
# Backend tests
cd backend && uv run pytest

# Frontend tests
cd frontend && npm test
```

## Linting

```bash
# Backend
make lint-backend

# Frontend
cd frontend && npm run lint
```

## Submitting Changes

1. **Fork** the repository
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and ensure they follow the code standards above
4. **Run tests** and linting to verify nothing is broken:
   ```bash
   make test
   make lint-backend
   cd frontend && npm run lint
   ```
5. **Commit** with a clear, concise message describing the change
6. **Push** your branch and open a **Pull Request** against `main`

## Reporting Issues

If you find a bug or have a feature request, please [open an issue](../../issues/new/choose) using the appropriate template.
