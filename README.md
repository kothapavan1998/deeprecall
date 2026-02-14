<h1 align="center">DeepRecall</h1>

<p align="center">
  <b>Recursive reasoning over your data. Plug into any vector DB or agent framework.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/deeprecall/"><img src="https://img.shields.io/pypi/v/deeprecall?color=blue&v=1" alt="PyPI"></a>
  <a href="https://pypi.org/project/deeprecall/"><img src="https://img.shields.io/pypi/pyversions/deeprecall?v=1" alt="Python"></a>
  <a href="https://github.com/kothapavan1998/deeprecall/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"></a>
</p>

---

Standard RAG retrieves documents once and stuffs them into a prompt. DeepRecall uses MIT's [Recursive Language Models](https://github.com/alexzhang13/rlm) to let your LLM **search, reason, search again, and repeat** -- until it actually has enough information to answer properly.

The LLM gets a `search_db()` function injected into a sandboxed Python REPL. It decides what to search for, analyzes results with code, refines its queries based on what it found, and synthesizes a final answer. This is not a fixed pipeline -- the LLM drives the retrieval strategy.

## Install

```bash
pip install deeprecall[chroma]    # ChromaDB (local, zero-config)
pip install deeprecall[milvus]    # Milvus
pip install deeprecall[qdrant]    # Qdrant
pip install deeprecall[pinecone]  # Pinecone
pip install deeprecall[redis]     # Redis distributed cache
pip install deeprecall[otel]      # OpenTelemetry tracing
pip install deeprecall[all]       # Everything
```

## Quick Start

```python
from deeprecall import DeepRecall
from deeprecall.vectorstores import ChromaStore

store = ChromaStore(collection_name="my_docs")
store.add_documents(["doc 1 text...", "doc 2 text...", "doc 3 text..."])

engine = DeepRecall(
    vectorstore=store,
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": "sk-..."},
)

result = engine.query("What are the key themes across these documents?")
print(result.answer)
print(f"Sources: {len(result.sources)}")
print(f"Steps: {len(result.reasoning_trace)}")
print(f"Time: {result.execution_time:.1f}s")
```

## What's New in v0.2

### Budget Guardrails

Control exactly how much a query can spend -- tokens, time, searches, or dollars.

```python
from deeprecall import DeepRecall, QueryBudget

engine = DeepRecall(vectorstore=store, backend="openai",
                    backend_kwargs={"model_name": "gpt-4o-mini"})

result = engine.query(
    "Complex multi-hop question?",
    budget=QueryBudget(
        max_search_calls=10,     # Stop after 10 vector DB searches
        max_tokens=50000,        # Total token budget
        max_time_seconds=30.0,   # Wall-clock timeout
    ),
)

# Check what was used
print(result.budget_status)  # {"iterations_used": 5, "search_calls_used": 8, ...}
```

### Reasoning Trace

Full visibility into what the LLM did at every step -- code executed, outputs, searches made.

```python
result = engine.query("What caused the 2008 financial crisis?")

for step in result.reasoning_trace:
    print(f"Step {step.iteration}: {step.action}")
    if step.searches:
        print(f"  Searched: {[s['query'] for s in step.searches]}")
    if step.code:
        print(f"  Code: {step.code[:100]}...")
```

### Callbacks

Hook into the reasoning pipeline for monitoring, logging, or custom integrations.

```python
from deeprecall import DeepRecall, DeepRecallConfig, ConsoleCallback, JSONLCallback

config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini"},
    callbacks=[
        ConsoleCallback(),                     # Live step-by-step output
        JSONLCallback(log_dir="./logs"),        # Structured logging
    ],
)
engine = DeepRecall(vectorstore=store, config=config)
```

### OpenTelemetry Tracing

Emit distributed traces to Jaeger, Datadog, Grafana Tempo, Honeycomb, or any OTLP backend.

```python
from deeprecall import DeepRecall, DeepRecallConfig, OpenTelemetryCallback

otel = OpenTelemetryCallback(
    service_name="my-rag-service",
    # endpoint="https://otlp.datadoghq.com:4317",  # Datadog
    # headers={"DD-API-KEY": "your-key"},
)
config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini"},
    callbacks=[otel],
)
# Every query() call emits a trace with child spans for each reasoning step and search
```

### Caching (In-Memory, Disk, Redis)

Avoid redundant LLM and vector DB calls. Three backends: in-memory (dev), SQLite (single-machine), Redis (distributed/production).

```python
from deeprecall import DeepRecall, DeepRecallConfig, InMemoryCache, RedisCache

# In-memory (fastest, ephemeral)
config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini"},
    cache=InMemoryCache(max_size=500, default_ttl=3600),
)

# Redis (distributed, production -- works with AWS ElastiCache, GCP Memorystore, etc.)
config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini"},
    cache=RedisCache(url="redis://localhost:6379/0"),
    # Or: RedisCache(url="rediss://my-cluster.abc123.cache.amazonaws.com:6379/0")
)
engine = DeepRecall(vectorstore=store, config=config)
# Second identical query hits cache -- zero LLM cost
```

### Reranking

Improve search quality with Cohere or cross-encoder rerankers.

```python
from deeprecall.core.reranker import CohereReranker

config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini"},
    reranker=CohereReranker(api_key="co-..."),
)
```

### Async Support & Thread Safety

DeepRecall is designed for high-concurrency production use. Every blocking operation (LLM calls, vector DB searches, cache I/O, file writes) is offloaded from the async event loop via `asyncio.to_thread()`. All shared state is protected with proper synchronization.

```python
from deeprecall import AsyncDeepRecall

engine = AsyncDeepRecall(vectorstore=store, backend="openai",
                          backend_kwargs={"model_name": "gpt-4o-mini"})

# Non-blocking -- multiple queries can run concurrently
result = await engine.query("question")
await engine.add_documents(["new doc..."])
```

Thread safety highlights:

- **Server endpoints** -- `query`, `add_documents`, `cache/clear` all run in the thread pool, never blocking the event loop
- **Callbacks** -- `UsageTrackingCallback` counters and `JSONLCallback` file writes are lock-protected for concurrent queries
- **OpenTelemetry** -- span state is thread-local, so parallel queries produce isolated traces
- **Rate limiter** -- bucket state is lock-protected against concurrent access
- **Redis cache** -- uses the thread-safe `redis-py` client; hit/miss counters are lock-protected
- **Auth middleware** -- supports both sync and async `validate_fn`; sync validators run in a thread

### Server Auth & Rate Limiting

```bash
deeprecall serve --api-keys "key1,key2" --rate-limit 60 --port 8000
```

## How It Works

1. A lightweight HTTP server wraps your vector store on a random port
2. A `search_db(query, top_k)` function is injected into the RLM's sandboxed REPL
3. The LLM enters a recursive loop -- it can search, write Python, call sub-LLMs, and search again
4. When it has enough info, it returns a `FINAL()` answer
5. You get back the answer, sources, full reasoning trace, budget usage, and confidence score

## Vector Stores

| Store | Install | Needs embedding_fn? |
|-------|---------|---------------------|
| ChromaDB | `deeprecall[chroma]` | No (built-in) |
| Milvus | `deeprecall[milvus]` | Yes |
| Qdrant | `deeprecall[qdrant]` | Yes |
| Pinecone | `deeprecall[pinecone]` | Yes |

All stores implement the same interface: `add_documents()`, `search()`, `delete()`, `count()`.

## Framework Adapters

**LangChain** / **LlamaIndex** / **OpenAI-compatible API** -- see [docs/adapters.md](docs/adapters.md).

```bash
deeprecall serve --vectorstore chroma --collection my_docs --port 8000
```

## CLI

```bash
deeprecall init                        # Generate starter config
deeprecall ingest --path ./docs/       # Ingest documents
deeprecall query "question" --max-searches 10 --max-time 30
deeprecall serve --port 8000 --api-keys "key1,key2"
deeprecall delete doc_id_1 doc_id_2    # Delete documents
```

## Project Structure

```
deeprecall/
├── core/           # Engine, config, guardrails, tracer, cache, callbacks, reranker
│   ├── cache.py          # InMemoryCache, DiskCache (SQLite)
│   ├── cache_redis.py    # RedisCache (distributed)
│   ├── callbacks.py      # ConsoleCallback, JSONLCallback, UsageTrackingCallback
│   ├── callback_otel.py  # OpenTelemetry distributed tracing
│   ├── async_engine.py   # AsyncDeepRecall (non-blocking wrapper)
│   └── ...
├── vectorstores/   # ChromaDB, Milvus, Qdrant, Pinecone adapters
├── adapters/       # LangChain, LlamaIndex, OpenAI-compatible server
├── middleware/      # API key auth (sync + async), rate limiting (thread-safe)
├── prompts/        # System prompts for the RLM
└── cli.py          # CLI entry point

tests/
├── test_concurrency.py   # Thread safety & race condition tests
├── test_cache_redis.py   # Redis cache unit tests
├── test_callback_otel.py # OpenTelemetry callback unit tests
└── ...                   # 114+ tests total
```

## Contributing

```bash
git clone https://github.com/kothapavan1998/deeprecall.git
cd deeprecall
pip install -e ".[all]"
make check
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

Built on [Recursive Language Models](https://arxiv.org/abs/2512.24601) by Zhang, Kraska, and Khattab (MIT).

## License

MIT
