"""FastAPI application entry point for Agent Swarm backend.

This module initializes the FastAPI application with all middleware,
routers, and event handlers configured.

Usage:
    uv run uvicorn main:app --reload
"""

import contextlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.routes import set_session_manager as set_routes_session_manager
from api.websocket import (
    set_session_manager as set_websocket_session_manager,
)
from api.websocket import (
    websocket_router,
)
from config import configure_logging, settings
from events import get_event_bus
from metrics import MetricsCollector
from models.database import SessionStore
from sandbox import SandboxManager
from session_manager import SessionManager

# Configure structured logging
configure_logging(settings.log_level, settings.log_format)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown events.

    Handles initialization and cleanup of resources like the sandbox manager
    and event bus.

    Args:
        app: The FastAPI application instance.

    Yields:
        None during application runtime.
    """
    # Startup
    logger.info(
        "application_starting",
        backend_port=settings.backend_port,
        log_level=settings.log_level,
        use_mock_llm=settings.use_mock_llm,
    )

    # Initialize resources
    sandbox_manager = SandboxManager()
    event_bus = get_event_bus()
    metrics_collector = MetricsCollector()

    session_store: SessionStore | None = None
    try:
        session_store = SessionStore(settings.database_path)
        await session_store.init()
    except Exception as e:
        # Keep the API available even if persistence initialization fails.
        logger.warning("session_store_init_failed", error=str(e))

    session_manager = SessionManager(
        sandbox_manager,
        event_bus,
        session_store=session_store,
        metrics_collector=metrics_collector,
    )

    # Register session manager with routes
    set_routes_session_manager(session_manager)
    set_websocket_session_manager(session_manager)

    # Store on app.state for access
    app.state.session_manager = session_manager
    app.state.session_store = session_store

    # Start periodic sandbox health check background task
    health_check_task = await session_manager.start_health_check_loop(
        interval_seconds=30.0
    )
    app.state.health_check_task = health_check_task

    logger.info("resources_initialized")
    logger.info("application_started")

    yield

    # Shutdown
    logger.info("application_shutting_down")

    # Cancel the health check background task
    health_check_task = app.state.health_check_task
    if health_check_task and not health_check_task.done():
        health_check_task.cancel()
        with contextlib.suppress(Exception):
            await health_check_task

    # Cleanup resources (session_manager.cleanup_all already calls
    # sandbox_manager.cleanup_all, so we don't call it separately).
    session_manager = app.state.session_manager
    await session_manager.cleanup_all()

    logger.info("application_shutdown_complete")


# Create FastAPI application
app = FastAPI(
    title="Agent Swarm POC",
    description="Backend API for comparing AI-assisted coding paradigms: "
    "Single ReAct Agent, Task Decomposition Swarm, and Parallel Hypothesis Testing.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include HTTP routes
app.include_router(router, tags=["sessions"])

# Include WebSocket routes
app.include_router(websocket_router, tags=["websocket"])


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Root endpoint that redirects to API documentation.

    Returns:
        A welcome message with documentation URL.
    """
    return {
        "message": "Agent Swarm POC API",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.backend_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
