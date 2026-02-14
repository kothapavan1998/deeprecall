"""Caching layer for DeepRecall queries and search results.

Provides in-memory and disk-based caching to avoid redundant LLM calls
and vector store queries.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class BaseCache(ABC):
    """Abstract cache interface for DeepRecall."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Retrieve a cached value by key. Returns None on miss."""

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value with an optional TTL in seconds."""

    @abstractmethod
    def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from the cache."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return cache statistics (hits, misses, size)."""


class InMemoryCache(BaseCache):
    """In-memory LRU cache with TTL support.

    Args:
        max_size: Maximum number of entries. Oldest evicted when full.
        default_ttl: Default TTL in seconds. None = no expiration.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int | None = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, expires_at = self._cache[key]

            # Check TTL
            if expires_at is not None and time.time() > expires_at:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + effective_ttl if effective_ttl else None

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, expires_at)

            # Evict oldest if over capacity
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "type": "in_memory",
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            }


class DiskCache(BaseCache):
    """SQLite-backed persistent cache for cross-session caching.

    Args:
        db_path: Path to the SQLite database file.
        default_ttl: Default TTL in seconds. None = no expiration.
    """

    def __init__(self, db_path: str = ".deeprecall_cache.db", default_ttl: int | None = 3600):
        self.db_path = db_path
        self.default_ttl = default_ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._local = threading.local()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Get a thread-local SQLite connection (reused across calls)."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, timeout=10)
        return self._local.conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")

    @staticmethod
    def _serialize(value: Any) -> str:
        """Safely serialize a value to JSON string."""
        if hasattr(value, "to_dict"):
            return json.dumps(value.to_dict())
        return json.dumps(value, default=str)

    def get(self, key: str) -> Any | None:
        with self._lock:
            try:
                conn = self._connect()
                row = conn.execute(
                    "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
                ).fetchone()
            except sqlite3.Error:
                logger.warning("DiskCache read error for key %s", key, exc_info=True)
                self._misses += 1
                return None

            if row is None:
                self._misses += 1
                return None

            value_str, expires_at = row

            if expires_at is not None and time.time() > expires_at:
                try:
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
                except sqlite3.Error:
                    pass
                self._misses += 1
                return None

            self._hits += 1
            try:
                return json.loads(value_str)
            except json.JSONDecodeError:
                logger.warning("DiskCache corrupted entry for key %s", key)
                return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + effective_ttl if effective_ttl else None
        now = time.time()

        with self._lock:
            try:
                conn = self._connect()
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value, expires_at, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (key, self._serialize(value), expires_at, now),
                )
                conn.commit()
            except (sqlite3.Error, TypeError):
                logger.warning("DiskCache write error for key %s", key, exc_info=True)

    def invalidate(self, key: str) -> None:
        with self._lock:
            try:
                conn = self._connect()
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
            except sqlite3.Error:
                pass

    def clear(self) -> None:
        with self._lock:
            try:
                conn = self._connect()
                conn.execute("DELETE FROM cache")
                conn.commit()
            except sqlite3.Error:
                pass
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            try:
                conn = self._connect()
                count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            except sqlite3.Error:
                count = -1
            total = self._hits + self._misses
            return {
                "type": "disk",
                "db_path": self.db_path,
                "size": count,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            }

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns number of entries deleted."""
        with self._lock:
            try:
                conn = self._connect()
                cursor = conn.execute(
                    "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (time.time(),),
                )
                conn.commit()
                return cursor.rowcount
            except sqlite3.Error:
                return 0
