.PHONY: help dev dev-backend dev-frontend build test lint-backend lint install-hooks backend-sync backend-env-check clean sandbox-image

UV_CACHE_DIR ?= $(CURDIR)/.uv-cache

help:
	@echo "Available targets:"
	@echo "  dev           - Start full stack in development mode"
	@echo "  dev-backend   - Start backend only"
	@echo "  dev-frontend  - Start frontend only"
	@echo "  backend-sync  - Sync backend dependencies (including dev extras)"
	@echo "  backend-env-check - Verify backend virtualenv + key dev tools"
	@echo "  install-hooks - Install pre-commit hooks into .git/hooks"
	@echo "  lint          - Run all pre-commit checks on all files"
	@echo "  lint-backend  - Run backend Ruff checks"
	@echo "  build         - Build all containers"
	@echo "  test          - Run all tests"
	@echo "  clean         - Remove containers and images"
	@echo "  sandbox-image - Build the sandbox Docker image"

dev:
	docker-compose up

dev-backend:
	cd backend && UV_CACHE_DIR=$(UV_CACHE_DIR) uv run python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev

backend-sync:
	cd backend && UV_CACHE_DIR=$(UV_CACHE_DIR) uv sync --extra dev

backend-env-check:
	cd backend && UV_CACHE_DIR=$(UV_CACHE_DIR) uv run python -c "import sys; print('python:', sys.executable)"
	cd backend && UV_CACHE_DIR=$(UV_CACHE_DIR) uv run python -c "import structlog; print('structlog:', structlog.__version__)"
	cd backend && UV_CACHE_DIR=$(UV_CACHE_DIR) uv run python -c "import pytest; print('pytest:', pytest.__version__)"
	cd backend && UV_CACHE_DIR=$(UV_CACHE_DIR) uv run python -m ruff --version

install-hooks:
	cd backend && UV_CACHE_DIR=$(UV_CACHE_DIR) uv run pre-commit install

lint:
	cd backend && UV_CACHE_DIR=$(UV_CACHE_DIR) uv run pre-commit run --all-files

lint-backend:
	cd backend && UV_CACHE_DIR=$(UV_CACHE_DIR) uv run python -m ruff check .

build:
	docker-compose build

test:
	cd backend && UV_CACHE_DIR=$(UV_CACHE_DIR) uv run python -m pytest
	cd frontend && npm test

clean:
	docker-compose down -v --rmi local
	docker image rm agent-swarm-sandbox:latest || true

sandbox-image:
	docker build -t agent-swarm-sandbox:latest ./sandbox-image
