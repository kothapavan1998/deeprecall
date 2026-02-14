"""Tests for RedisCache.

Uses a mock Redis client to test without a real Redis server.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Create a fake redis module so RedisCache can import cleanly
# ---------------------------------------------------------------------------


def _make_mock_redis_module():
    """Build a mock redis module that behaves like redis-py."""
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_module.Redis.return_value = mock_client
    mock_module.from_url.return_value = mock_client
    mock_module.ConnectionError = ConnectionError
    return mock_module, mock_client


class TestRedisCache:
    """Test RedisCache with mocked redis client."""

    def _make_cache(self, mock_client=None, **kwargs):
        """Create a RedisCache with a mocked Redis underneath."""
        mock_module, default_client = _make_mock_redis_module()
        client = mock_client or default_client
        mock_module.from_url.return_value = client

        with patch.dict(sys.modules, {"redis": mock_module}):
            # Force fresh import so it picks up our mock
            import importlib

            import deeprecall.core.cache_redis as mod

            importlib.reload(mod)
            cache = mod.RedisCache(url="redis://localhost:6379/0", **kwargs)

        cache._client = client
        return cache, client

    def test_get_hit(self):
        cache, client = self._make_cache()
        client.get.return_value = json.dumps({"answer": "hello"})

        result = cache.get("test_key")

        assert result == {"answer": "hello"}
        assert cache._hits == 1
        assert cache._misses == 0
        client.get.assert_called_once_with("deeprecall:test_key")

    def test_get_miss(self):
        cache, client = self._make_cache()
        client.get.return_value = None

        result = cache.get("missing_key")

        assert result is None
        assert cache._misses == 1

    def test_get_corrupted_json(self):
        cache, client = self._make_cache()
        client.get.return_value = "not-valid-json{{"

        result = cache.get("bad_key")

        assert result is None
        # Still counted as a hit (key existed) but returned None due to corruption
        assert cache._hits == 1

    def test_get_redis_error(self):
        cache, client = self._make_cache()
        client.get.side_effect = Exception("Connection lost")

        result = cache.get("error_key")

        assert result is None
        assert cache._misses == 1

    def test_set_with_ttl(self):
        cache, client = self._make_cache()

        cache.set("key", {"value": 42}, ttl=300)

        client.setex.assert_called_once_with(
            "deeprecall:key", 300, json.dumps({"value": 42})
        )

    def test_set_without_ttl(self):
        cache, client = self._make_cache(default_ttl=3600)

        cache.set("key", "value")

        client.setex.assert_called_once_with(
            "deeprecall:key", 3600, json.dumps("value")
        )

    def test_set_no_expiry(self):
        cache, client = self._make_cache(default_ttl=None)

        cache.set("key", "forever")

        client.set.assert_called_once_with(
            "deeprecall:key", json.dumps("forever")
        )

    def test_set_with_to_dict_object(self):
        cache, client = self._make_cache()

        class FakeResult:
            def to_dict(self):
                return {"answer": "test", "sources": []}

        cache.set("key", FakeResult(), ttl=60)

        expected = json.dumps({"answer": "test", "sources": []})
        client.setex.assert_called_once_with("deeprecall:key", 60, expected)

    def test_invalidate(self):
        cache, client = self._make_cache()

        cache.invalidate("old_key")

        client.delete.assert_called_once_with("deeprecall:old_key")

    def test_clear(self):
        cache, client = self._make_cache()
        client.scan.side_effect = [
            (42, ["deeprecall:key1", "deeprecall:key2"]),
            (0, ["deeprecall:key3"]),
        ]

        cache.clear()

        assert client.delete.call_count == 2
        assert cache._hits == 0
        assert cache._misses == 0

    def test_stats(self):
        cache, client = self._make_cache(prefix="test:")
        client.scan.return_value = (0, ["test:a", "test:b", "test:c"])
        cache._hits = 10
        cache._misses = 3

        stats = cache.stats()

        assert stats["type"] == "redis"
        assert stats["prefix"] == "test:"
        assert stats["size"] == 3
        assert stats["hits"] == 10
        assert stats["misses"] == 3
        assert stats["hit_rate"] == round(10 / 13, 4)

    def test_custom_prefix(self):
        cache, client = self._make_cache(prefix="myapp:")
        client.get.return_value = None  # cache miss

        cache.get("key")

        client.get.assert_called_once_with("myapp:key")

    def test_health_check_connected(self):
        cache, client = self._make_cache()
        client.ping.return_value = True
        client.info.return_value = {"redis_version": "7.2.4"}

        health = cache.health_check()

        assert health["status"] == "connected"
        assert "latency_ms" in health
        assert health["redis_version"] == "7.2.4"

    def test_health_check_disconnected(self):
        cache, client = self._make_cache()
        client.ping.side_effect = Exception("Connection refused")

        health = cache.health_check()

        assert health["status"] == "disconnected"
        assert "error" in health
