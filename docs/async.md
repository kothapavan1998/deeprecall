# Async Support & Concurrency

DeepRecall is designed for high-concurrency production use. This guide covers the async engine, server concurrency model, and thread-safety guarantees.

---

## AsyncDeepRecall

Non-blocking wrapper around `DeepRecallEngine`. All heavy operations run in a thread pool via `asyncio.to_thread()`.

```python
import asyncio
from deeprecall import AsyncDeepRecall
from deeprecall.vectorstores import ChromaStore

async def main():
    store = ChromaStore(collection_name="docs")
    engine = AsyncDeepRecall(
        vectorstore=store,
        backend="openai",
        backend_kwargs={"model_name": "gpt-4o-mini"},
    )

    # Non-blocking queries -- run multiple concurrently
    results = await asyncio.gather(
        engine.query("What are the key themes?"),
        engine.query("Who are the main authors?"),
    )

    for r in results:
        print(r.answer[:100])

    await engine.close()

asyncio.run(main())
```

### Constructor

Same as `DeepRecall` -- accepts `vectorstore`, `config`, or individual kwargs.

### Methods

| Method | Signature | Description |
|---|---|---|
| `query()` | `async (query, root_prompt=None, top_k=None, budget=None) -> DeepRecallResult` | Non-blocking query |
| `add_documents()` | `async (documents, metadatas=None, ids=None) -> list[str]` | Non-blocking document ingestion |
| `close()` | `async () -> None` | Clean up resources |

### Context Manager

```python
async with AsyncDeepRecall(vectorstore=store, backend="openai") as engine:
    result = await engine.query("question")
```

### Properties

| Property | Type | Description |
|---|---|---|
| `config` | `DeepRecallConfig` | Current engine configuration |
| `vectorstore` | `BaseVectorStore` | The underlying vector store |

---

## Thread Safety Guarantees

Every component that holds shared mutable state is protected:

### Callbacks

| Component | Protection | Details |
|---|---|---|
| `UsageTrackingCallback` | `threading.Lock` | All counter mutations (`total_queries`, `total_tokens`, etc.) and `summary()` reads |
| `JSONLCallback` | `threading.Lock` | File writes are serialized -- no interleaved JSONL lines |
| `OpenTelemetryCallback` | `threading.local()` | Each thread gets isolated span state (`current_span`, `step_count`, `ctx_token`) |

### Caching

| Component | Protection | Details |
|---|---|---|
| `InMemoryCache` | `threading.Lock` | LRU dict and hit/miss counters |
| `DiskCache` | `threading.Lock` + `threading.local()` | Per-thread SQLite connections, locked counter access |
| `RedisCache` | `threading.Lock` | Hit/miss counters. Redis client itself is thread-safe (connection pool) |

### Server Middleware

| Component | Protection | Details |
|---|---|---|
| `RateLimiter` | `threading.Lock` | Bucket dict lookup + create + consume is atomic |
| `APIKeyAuth` | `asyncio.to_thread()` | Sync `validate_fn` runs in thread pool, async `validate_fn` awaited directly |

### Server Endpoints

All blocking operations are offloaded from the event loop:

| Endpoint | Blocking Call | Solution |
|---|---|---|
| `POST /v1/chat/completions` | `engine.query()` | `asyncio.to_thread()` |
| `POST /v1/documents` | `engine.add_documents()` | `asyncio.to_thread()` |
| `POST /v1/cache/clear` | `cache.clear()` | `asyncio.to_thread()` |

---

## Concurrent Queries

When multiple queries run simultaneously (via `asyncio.gather` or concurrent HTTP requests):

1. Each `engine.query()` runs in its own thread pool worker
2. Each query gets its own `SearchServer` (HTTP bridge to the vector store)
3. Each query gets its own `DeepRecallTracer` (reasoning step tracker)
4. OTEL spans are isolated per-thread via `threading.local()`
5. Shared callbacks (usage tracker, JSONL logger) are lock-protected

This means 10 concurrent queries will use 10 thread pool workers, with no state bleeding between them.

---

## Performance Tips

1. **Use `InMemoryCache`** to skip repeated queries entirely -- zero LLM cost for cache hits
2. **Set `max_search_calls`** in `QueryBudget` to prevent runaway reasoning loops
3. **Use `AsyncDeepRecall`** in web applications to handle many concurrent users
4. **Use `RedisCache`** in multi-process deployments (uvicorn with multiple workers)
5. **Use `max_time_seconds`** to enforce SLA deadlines on every query
