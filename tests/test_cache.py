"""Tests for the caching layer."""

from __future__ import annotations

import hashlib
import tempfile
import time

from deeprecall.core.cache import DiskCache, InMemoryCache


class TestInMemoryCache:
    def test_get_and_set(self):
        cache = InMemoryCache()
        cache.set("key1", {"answer": "hello"})
        assert cache.get("key1") == {"answer": "hello"}

    def test_miss_returns_none(self):
        cache = InMemoryCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        cache = InMemoryCache(default_ttl=1)
        cache.set("key1", "value", ttl=1)
        assert cache.get("key1") == "value"  # Still valid
        # Manually expire by tweaking the internal entry
        cache._cache["key1"] = ("value", time.time() - 1)
        assert cache.get("key1") is None

    def test_no_ttl_persists(self):
        cache = InMemoryCache(default_ttl=None)
        cache.set("key1", "value")
        assert cache.get("key1") == "value"

    def test_lru_eviction(self):
        cache = InMemoryCache(max_size=2, default_ttl=None)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # Should evict "a"

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_invalidate(self):
        cache = InMemoryCache()
        cache.set("key1", "value")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_clear(self):
        cache = InMemoryCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_stats(self):
        cache = InMemoryCache(max_size=100)
        cache.set("a", 1)
        cache.get("a")  # hit
        cache.get("b")  # miss

        stats = cache.stats()
        assert stats["type"] == "in_memory"
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_move_to_end_on_access(self):
        cache = InMemoryCache(max_size=2, default_ttl=None)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # Move "a" to end
        cache.set("c", 3)  # Should evict "b" (least recently used)

        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3


class TestDiskCache:
    def test_get_and_set(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache = DiskCache(db_path=f.name)
            cache.set("key1", {"answer": "hello"})
            assert cache.get("key1") == {"answer": "hello"}

    def test_miss_returns_none(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache = DiskCache(db_path=f.name)
            assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache = DiskCache(db_path=f.name, default_ttl=1)
            cache.set("key1", "value", ttl=1)
            assert cache.get("key1") == "value"
            # Manually set expired timestamp in DB
            with cache._connect() as conn:
                conn.execute(
                    "UPDATE cache SET expires_at = ? WHERE key = ?",
                    (time.time() - 1, "key1"),
                )
            assert cache.get("key1") is None

    def test_invalidate(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache = DiskCache(db_path=f.name)
            cache.set("key1", "value")
            cache.invalidate("key1")
            assert cache.get("key1") is None

    def test_clear(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache = DiskCache(db_path=f.name)
            cache.set("a", 1)
            cache.set("b", 2)
            cache.clear()
            assert cache.get("a") is None

    def test_stats(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache = DiskCache(db_path=f.name)
            cache.set("a", 1)
            cache.get("a")  # hit
            cache.get("b")  # miss

            stats = cache.stats()
            assert stats["type"] == "disk"
            assert stats["size"] == 1
            assert stats["hits"] == 1
            assert stats["misses"] == 1

    def test_cleanup_expired(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache = DiskCache(db_path=f.name, default_ttl=None)
            cache.set("fresh", "value", ttl=3600)
            cache.set("expired", "value", ttl=1)
            # Manually expire
            with cache._connect() as conn:
                conn.execute(
                    "UPDATE cache SET expires_at = ? WHERE key = ?",
                    (time.time() - 1, "expired"),
                )

            deleted = cache.cleanup_expired()
            assert deleted == 1
            assert cache.get("fresh") == "value"


class TestCacheKeyDeterminism:
    """Test that cache keys are deterministic and unique."""

    @staticmethod
    def _make_key(*parts: str) -> str:
        combined = "|".join(str(p) for p in parts)
        return hashlib.sha256(combined.encode()).hexdigest()

    def test_deterministic(self):
        key1 = self._make_key("query", "openai", "gpt-4o-mini")
        key2 = self._make_key("query", "openai", "gpt-4o-mini")
        assert key1 == key2

    def test_different_inputs_different_keys(self):
        key1 = self._make_key("query1")
        key2 = self._make_key("query2")
        assert key1 != key2
