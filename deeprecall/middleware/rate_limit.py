"""Token bucket rate limiter middleware for the DeepRecall server."""

from __future__ import annotations

import threading
import time
from typing import Any

try:
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError:
    raise ImportError(
        "fastapi is required for middleware. Install with: pip install deeprecall[server]"
    ) from None


class _TokenBucket:
    """Simple token bucket for rate limiting."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    @property
    def retry_after(self) -> float:
        """Seconds until at least 1 token is available."""
        if self.tokens >= 1:
            return 0.0
        return (1 - self.tokens) / self.rate


class RateLimiter(BaseHTTPMiddleware):
    """Rate limiter middleware using per-key token buckets.

    Args:
        app: The FastAPI application.
        requests_per_minute: Max requests per minute per key.
        exempt_paths: Paths that skip rate limiting.
    """

    def __init__(
        self,
        app: Any,
        requests_per_minute: int = 60,
        exempt_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.capacity = requests_per_minute
        self.exempt_paths = set(exempt_paths or ["/health", "/docs", "/openapi.json"])
        self._buckets: dict[str, _TokenBucket] = {}
        self._lock = threading.Lock()  # protect bucket dict from concurrent access

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Use API key or IP as the rate limit key
        key = getattr(request.state, "api_key", None) or (request.client.host if request.client else "unknown")

        with self._lock:
            if key not in self._buckets:
                # Evict stale buckets (full capacity = idle) when dict grows large
                if len(self._buckets) > 10000:
                    stale = [k for k, b in self._buckets.items() if b.tokens >= b.capacity]
                    for k in stale:
                        del self._buckets[k]
                self._buckets[key] = _TokenBucket(rate=self.rate, capacity=self.capacity)

            bucket = self._buckets[key]
            consumed = bucket.consume()

        if not consumed:
            retry_after = bucket.retry_after
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "retry_after": round(retry_after, 1),
                    }
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        return await call_next(request)
