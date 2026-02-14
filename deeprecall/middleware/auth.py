"""API key authentication middleware for the DeepRecall server."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

try:
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError:
    raise ImportError(
        "fastapi is required for middleware. Install with: pip install deeprecall[server]"
    ) from None


class APIKeyAuth(BaseHTTPMiddleware):
    """FastAPI middleware that validates API keys via Bearer tokens.

    Args:
        app: The FastAPI application.
        api_keys: List of valid API keys. If None, auth is disabled.
        validate_fn: Optional custom validation function (key -> bool).
        exempt_paths: Paths that skip auth (default: /health, /docs, /openapi.json).
    """

    def __init__(
        self,
        app: Any,
        api_keys: list[str] | None = None,
        validate_fn: Callable[[str], bool] | Callable[[str], Coroutine[Any, Any, bool]] | None = None,
        exempt_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self.api_keys = set(api_keys) if api_keys else None
        self.validate_fn = validate_fn
        self.exempt_paths = set(exempt_paths or ["/health", "/docs", "/openapi.json"])

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        # Skip auth if no keys configured
        if self.api_keys is None and self.validate_fn is None:
            return await call_next(request)

        # Skip exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Extract key from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Missing API key", "type": "auth_error"}},
            )

        api_key = auth_header[7:]  # Strip "Bearer "

        # Validate (supports both sync and async validate_fn)
        is_valid = False
        if self.validate_fn:
            if asyncio.iscoroutinefunction(self.validate_fn):
                is_valid = await self.validate_fn(api_key)
            else:
                # Run sync validators in a thread to avoid blocking the loop
                is_valid = await asyncio.to_thread(self.validate_fn, api_key)
        elif self.api_keys:
            is_valid = api_key in self.api_keys

        if not is_valid:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid API key", "type": "auth_error"}},
            )

        # Store key on request state for rate limiting
        request.state.api_key = api_key
        return await call_next(request)
