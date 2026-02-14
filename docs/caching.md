# Caching

DeepRecall supports three cache backends to reduce LLM costs and latency.

## Query Cache

Caches the full `DeepRecallResult` for identical queries. The second time you ask the same question with the same config, the result is returned instantly with zero LLM cost.

## Search Cache

Caches vector store search results. If the LLM searches for the same query across iterations or across different user queries, the vector DB is not hit again.

## Backends

### InMemoryCache

Fast, ephemeral. Lost when the process exits. Best for single-instance dev/testing.

```python
from deeprecall import InMemoryCache, DeepRecallConfig

config = DeepRecallConfig(
    cache=InMemoryCache(max_size=1000, default_ttl=3600),
)
```

Parameters:
- `max_size`: Max entries before LRU eviction (default: 1000)
- `default_ttl`: Time-to-live in seconds (default: 3600, None = no expiry)

### DiskCache

SQLite-backed. Persists across restarts. Good for single-machine deployments.

```python
from deeprecall.core.cache import DiskCache

config = DeepRecallConfig(
    cache=DiskCache(db_path=".deeprecall_cache.db", default_ttl=86400),
)
```

Parameters:
- `db_path`: Path to SQLite database file
- `default_ttl`: Time-to-live in seconds

### RedisCache

Redis-backed distributed cache. Works with any Redis-compatible service:
**Redis**, **AWS ElastiCache**, **GCP Memorystore**, **Azure Cache for Redis**, **Upstash**, **Redis Cloud**.

```bash
pip install deeprecall[redis]
```

```python
from deeprecall import RedisCache, DeepRecallConfig

# Local Redis
config = DeepRecallConfig(
    cache=RedisCache(url="redis://localhost:6379/0"),
)

# AWS ElastiCache (TLS)
config = DeepRecallConfig(
    cache=RedisCache(
        url="rediss://my-cluster.abc123.use1.cache.amazonaws.com:6379/0",
        default_ttl=7200,
    ),
)

# Host/port style with password
config = DeepRecallConfig(
    cache=RedisCache(
        host="redis.example.com",
        port=6379,
        password="secret",
        ssl=True,
    ),
)
```

Parameters:
- `url`: Redis URL (`redis://`, `rediss://` for TLS, `unix://`)
- `host` / `port` / `db`: Alternative to URL
- `password`: Redis auth password
- `default_ttl`: Default TTL in seconds (default: 3600)
- `prefix`: Key prefix for namespace isolation (default: `"deeprecall:"`)
- `ssl`: Enable TLS (or use `rediss://` URL)
- `**kwargs`: Passed to `redis.Redis()` -- use for `max_connections`, `socket_timeout`, etc.

Health check:

```python
cache = RedisCache(url="redis://localhost:6379/0")
print(cache.health_check())
# {"status": "connected", "latency_ms": 0.34, "redis_version": "7.2.4"}
```

## Cache Management

### Via API

```bash
# Clear all caches
curl -X POST http://localhost:8000/v1/cache/clear
```

### Programmatically

```python
engine.config.cache.clear()           # Clear everything
engine.config.cache.invalidate(key)   # Remove one entry
engine.config.cache.stats()           # Get hit/miss stats
```

### Cache Stats

```python
stats = engine.config.cache.stats()
# InMemoryCache: {"type": "in_memory", "size": 42, "max_size": 1000, "hits": 150, ...}
# DiskCache:     {"type": "disk", "db_path": ".deeprecall_cache.db", "size": 42, ...}
# RedisCache:    {"type": "redis", "prefix": "deeprecall:", "size": 42, ...}
```

## Choosing a Backend

| Backend | Persistence | Distributed | Speed | Use Case |
|---------|-------------|-------------|-------|----------|
| InMemoryCache | No | No | Fastest | Dev, testing, single-process |
| DiskCache | Yes | No | Fast | Single-machine production |
| RedisCache | Yes | Yes | Fast | Multi-instance, team, production |
