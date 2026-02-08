#!/bin/bash
# Development startup script for Agentica
# Usage:
#   ./dev.sh          - Start all services
#   ./dev.sh stop     - Stop all services
#   ./dev.sh clean    - Stop services and clean up Docker resources
#   ./dev.sh logs     - Tail logs from all services

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PID_FILE="$PROJECT_ROOT/.backend.pid"
FRONTEND_PID_FILE="$PROJECT_ROOT/.frontend.pid"
LOG_DIR="$PROJECT_ROOT/.logs"
UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.uv-cache}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Ensure log directory exists
mkdir -p "$LOG_DIR"

stop_services() {
    log_info "Stopping services..."

    # Stop backend
    if [ -f "$BACKEND_PID_FILE" ]; then
        PID=$(cat "$BACKEND_PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
            log_success "Backend stopped (PID: $PID)"
        fi
        rm -f "$BACKEND_PID_FILE"
    fi

    # Stop frontend
    if [ -f "$FRONTEND_PID_FILE" ]; then
        PID=$(cat "$FRONTEND_PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
            log_success "Frontend stopped (PID: $PID)"
        fi
        rm -f "$FRONTEND_PID_FILE"
    fi

    # Kill any remaining processes on our ports
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true

    log_success "All services stopped"
}

clean_docker() {
    log_info "Cleaning up Docker resources..."

    # Stop and remove sandbox containers
    docker ps -a --filter "ancestor=agentica-sandbox:latest" -q | xargs -r docker rm -f 2>/dev/null || true

    # Remove dangling containers with sandbox prefix
    docker ps -a --filter "name=sandbox_" -q | xargs -r docker rm -f 2>/dev/null || true

    log_success "Docker resources cleaned"
}

clean_all() {
    stop_services
    clean_docker

    # Clean log files
    rm -rf "$LOG_DIR"

    log_success "Full cleanup complete"
}

build_sandbox_image() {
    if ! docker image inspect agentica-sandbox:latest >/dev/null 2>&1; then
        log_info "Building sandbox Docker image..."
        docker build -t agentica-sandbox:latest "$PROJECT_ROOT/sandbox-image"
        log_success "Sandbox image built"
    else
        log_info "Sandbox image already exists"
    fi
}

check_dependencies() {
    log_info "Checking dependencies..."

    # Check Docker
    if ! command -v docker &>/dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    if ! docker info &>/dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi

    # Check UV (Python)
    if ! command -v uv &>/dev/null; then
        log_error "UV is not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi

    # Check Node.js
    if ! command -v npm &>/dev/null; then
        log_error "npm is not installed"
        exit 1
    fi

    # Check .env file
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        log_warn ".env file not found, copying from env.example"
        cp "$PROJECT_ROOT/env.example" "$PROJECT_ROOT/.env"
        log_warn "Please edit .env and add your API keys"
    fi

    # Clarify expected Python env location
    if [ -d "$PROJECT_ROOT/.venv" ] && [ ! -d "$PROJECT_ROOT/backend/.venv" ]; then
        log_warn "Detected root .venv but backend/.venv is missing."
        log_warn "Backend Python environment should live in backend/.venv (project is backend/pyproject.toml)."
    fi

    log_success "Dependencies OK"
}

start_backend() {
    log_info "Starting backend..."

    cd "$PROJECT_ROOT/backend"

    # Install dependencies if needed
    if [ ! -d ".venv" ]; then
        log_info "Installing Python dependencies..."
        UV_CACHE_DIR="$UV_CACHE_DIR" uv sync
    fi

    # Start backend in background
    # Unset VIRTUAL_ENV to prevent conflicts with uv's expected environment path
    unset VIRTUAL_ENV
    UV_CACHE_DIR="$UV_CACHE_DIR" uv run python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 > "$LOG_DIR/backend.log" 2>&1 &
    echo $! > "$BACKEND_PID_FILE"

    # Wait for backend to be ready
    log_info "Waiting for backend to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            log_success "Backend started on http://localhost:8000"
            return 0
        fi
        sleep 1
    done

    log_error "Backend failed to start. Check $LOG_DIR/backend.log"
    return 1
}

start_frontend() {
    log_info "Starting frontend..."

    cd "$PROJECT_ROOT/frontend"

    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        log_info "Installing Node.js dependencies..."
        npm install
    fi

    # Start frontend in background
    npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
    echo $! > "$FRONTEND_PID_FILE"

    # Wait for frontend to be ready
    log_info "Waiting for frontend to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:3000 >/dev/null 2>&1; then
            log_success "Frontend started on http://localhost:3000"
            return 0
        fi
        sleep 1
    done

    log_error "Frontend failed to start. Check $LOG_DIR/frontend.log"
    return 1
}

show_logs() {
    if [ ! -d "$LOG_DIR" ]; then
        log_error "No logs found"
        exit 1
    fi

    tail -f "$LOG_DIR"/*.log
}

start_all() {
    echo ""
    echo "======================================"
    echo "   Agentica - Development"
    echo "======================================"
    echo ""

    # Clean up any existing processes first
    stop_services

    check_dependencies
    build_sandbox_image
    start_backend
    start_frontend

    echo ""
    echo "======================================"
    log_success "All services running!"
    echo ""
    echo "  Frontend:  http://localhost:3000"
    echo "  Backend:   http://localhost:8000"
    echo "  API Docs:  http://localhost:8000/docs"
    echo ""
    echo "  Logs:      $LOG_DIR/"
    echo ""
    echo "  Stop:      ./dev.sh stop"
    echo "  Clean:     ./dev.sh clean"
    echo "  Logs:      ./dev.sh logs"
    echo "======================================"
    echo ""

    # Set up trap for cleanup on Ctrl+C
    trap 'echo ""; log_info "Shutting down..."; stop_services; exit 0' INT TERM

    # Keep script running and show logs
    log_info "Press Ctrl+C to stop all services"
    echo ""
    tail -f "$LOG_DIR"/*.log
}

# Main command handling
case "${1:-}" in
    stop)
        stop_services
        ;;
    clean)
        clean_all
        ;;
    logs)
        show_logs
        ;;
    *)
        start_all
        ;;
esac
