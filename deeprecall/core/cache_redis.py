"""Redis-backed distributed cache for DeepRecall.

Provides a production-grade, distributed cache using Redis. Supports
Redis standalone, Redis Cluster, AWS ElastiCache, GCP Memorystore,
Azure Cache for Redis, and any Redis-protocol-compatible service.

Install: pip install deeprecall[redis]
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from deeprecall.core.cache import BaseCache

logger = logging.getLogger(__name__)


class RedisCache(BaseCache):
    """Redis-backed distributed cache for DeepRecall.

    Works with any Redis-compatible backend: standalone Redis, Redis Cluster,
    AWS ElastiCache, GCP Memorystore, Azure Cache for Redis, Upstash, etc.

    Args:
        url: Redis connection URL (e.g. ``redis://localhost:6379/0``).
            Supports ``redis://``, ``rediss://`` (TLS), and ``unix://``.
        host: Redis host (alternative to url).
        port: Redis port (default: 6379).
        db: Redis database number (default: 0).
        password: Redis password.
        default_ttl: Default TTL in seconds (default: 3600, None = no expiry).
        prefix: Key prefix for namespace isolation (default: "deeprecall:").
        ssl: Enable TLS connection (default: False). Use ``rediss://`` URL instead
            for simpler TLS setup.
        **kwargs: Extra keyword arguments passed to ``redis.Redis()`` or
            ``redis.from_url()``. Useful for connection pool size, socket
            timeout, retry settings, etc.

    Examples:
        Local Redis:

        >>> from deeprecall.core.cache_redis import RedisCache
        >>> cache = RedisCache(url="redis://localhost:6379/0")

        AWS ElastiCache (TLS):

        >>> cache = RedisCache(
        ...     url="rediss://my-cluster.abc123.use1.cache.amazonaws.com:6379/0",
        ...     default_ttl=7200,
        ... )

        With connection pool tuning:

        >>> cache = RedisCache(
        ...     host="redis.example.com",
        ...     password="secret",
        ...     max_connections=50,
        ...     socket_timeout=5.0,
        ... )
    """

    def __init__(
        self,
        url: str | None = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        default_ttl: int | None = 3600,
        prefix: str = "deeprecall:",
        ssl: bool = False,
        **kwargs: Any,
    ):
        try:
            import redis
        except ImportError:
            raise ImportError(
                "redis is required for RedisCache. Install it with: pip install deeprecall[redis]"
            ) from None

        self.default_ttl = default_ttl
        self.prefix = prefix
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

        if url:
            self._client = redis.from_url(url, decode_responses=True, **kwargs)
        else:
            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                ssl=ssl,
                decode_responses=True,
                **kwargs,
            )

        # Verify connectivity
        try:
            self._client.ping()
            logger.info("RedisCache connected to %s", url or f"{host}:{port}/{db}")
        except redis.ConnectionError as e:
            raise ConnectionError(
                f"Could not connect to Redis at {url or f'{host}:{port}'}: {e}"
            ) from e

    def _key(self, key: str) -> str:
        """Prefix the key for namespace isolation."""
        return f"{self.prefix}{key}"

    @staticmethod
    def _serialize(value: Any) -> str:
        """Safely serialize a value to JSON string."""
        if hasattr(value, "to_dict"):
            return json.dumps(value.to_dict())
        return json.dumps(value, default=str)

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value. Returns None on miss or Redis error."""
        try:
            raw = self._client.get(self._key(key))
        except Exception:
            logger.warning("RedisCache read error for key %s", key, exc_info=True)
            with self._lock:
                self._misses += 1
            return None

        if raw is None:
            with self._lock:
                self._misses += 1
            return None

        with self._lock:
            self._hits += 1
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("RedisCache corrupted entry for key %s", key)
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value with optional TTL. Silently handles Redis errors."""
        effective_ttl = ttl if ttl is not None else self.default_ttl

        try:
            prefixed = self._key(key)
            serialized = self._serialize(value)

            if effective_ttl:
                self._client.setex(prefixed, effective_ttl, serialized)
            else:
                self._client.set(prefixed, serialized)
        except Exception:
            logger.warning("RedisCache write error for key %s", key, exc_info=True)

    def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache."""
        try:
            self._client.delete(self._key(key))
        except Exception:
            logger.warning("RedisCache delete error for key %s", key, exc_info=True)

    def clear(self) -> None:
        """Remove all entries with this cache's prefix."""
        try:
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor=cursor, match=f"{self.prefix}*", count=500)
                if keys:
                    self._client.delete(*keys)
                if cursor == 0:
                    break
        except Exception:
            logger.warning("RedisCache clear error", exc_info=True)
        with self._lock:
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        try:
            # Count keys with our prefix using SCAN (non-blocking)
            count = 0
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor=cursor, match=f"{self.prefix}*", count=500)
                count += len(keys)
                if cursor == 0:
                    break
        except Exception:
            count = -1

        with self._lock:
            hits, misses = self._hits, self._misses
        total = hits + misses
        return {
            "type": "redis",
            "prefix": self.prefix,
            "size": count,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hits / total, 4) if total > 0 else 0.0,
        }

    def health_check(self) -> dict[str, Any]:
        """Check Redis connectivity and return server info.

        Returns:
            Dict with connection status and Redis server version.
        """
        try:
            start = time.perf_counter()
            self._client.ping()
            latency_ms = (time.perf_counter() - start) * 1000
            info = self._client.info("server")
            return {
                "status": "connected",
                "latency_ms": round(latency_ms, 2),
                "redis_version": info.get("redis_version", "unknown"),
            }
        except Exception as e:
            return {"status": "disconnected", "error": str(e)}
