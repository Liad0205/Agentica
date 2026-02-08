"""API module for HTTP routes and WebSocket handlers.

This module exposes the FastAPI routers for the Agentica backend.
"""

from api.routes import router
from api.websocket import websocket_router

__all__ = ["router", "websocket_router"]
