<h1 align="center">DeepRecall</h1>

<p align="center">
  <b>Recursive reasoning over your data. Plug into any vector DB or LLM framework.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/deeprecall/"><img src="https://img.shields.io/pypi/v/deeprecall?color=blue&v=3" alt="PyPI"></a>
  <a href="https://pypi.org/project/deeprecall/"><img src="https://img.shields.io/pypi/pyversions/deeprecall?v=3" alt="Python"></a>
  <a href="https://github.com/kothapavan1998/deeprecall/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"></a>
</p>

---

Standard RAG retrieves documents once and stuffs them into a prompt. DeepRecall uses MIT's [Recursive Language Models](https://github.com/alexzhang13/rlm) to let your LLM **search, reason, search again, and repeat** -- until it actually has enough information to answer properly.

The LLM gets a `search_db()` function injected into a sandboxed Python REPL. It decides what to search for, analyzes results with code, refines its queries based on what it found, and synthesizes a final answer. This is not a fixed pipeline -- the LLM drives the retrieval strategy.

## Install

```bash
pip install deeprecall[chroma]              # ChromaDB (local, zero-config)
pip install deeprecall[milvus]              # Milvus
pip install deeprecall[qdrant]              # Qdrant
pip install deeprecall[pinecone]            # Pinecone
pip install deeprecall[faiss]               # FAISS (local, ML-native)
pip install deeprecall[server]              # API server (FastAPI + uvicorn)
pip install deeprecall[rich]                # Rich console output (verbose mode)
pip install deeprecall[redis]               # Redis distributed cache
pip install deeprecall[otel]                # OpenTelemetry tracing
pip install deeprecall[langchain]           # LangChain adapter
pip install deeprecall[llamaindex]          # LlamaIndex adapter
pip install deeprecall[rerank-cohere]       # Cohere reranker
pip install deeprecall[rerank-cross-encoder] # Cross-encoder reranker
pip install deeprecall[all]                 # Everything
```

> **Note**: DeepRecall depends on `rlms` which transitively installs its own dependencies (OpenAI SDK, etc.). If you see dependency conflicts, check `pip show rlms` for the transitive tree.

## Quick Start

```python
from deeprecall import DeepRecall
from deeprecall.vectorstores import ChromaStore

store = ChromaStore(collection_name="my_docs")
store.add_documents(["doc 1 text...", "doc 2 text...", "doc 3 text..."])

# Context manager ensures cleanup (search server, connections)
with DeepRecall(
    vectorstore=store,
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": "sk-..."},
) as engine:
    result = engine.query("What are the key themes across these documents?")
    print(result.answer)
    print(f"Sources: {len(result.sources)}")
    print(f"Steps: {len(result.reasoning_trace)}")
    print(f"Time: {result.execution_time:.1f}s")
```

> **Tip**: Always use `with` or call `engine.close()` when done to release background resources. Vector stores with persistent connections (Milvus, Qdrant) also support `with store:` for automatic cleanup.

## What's New in v0.4

### RLM v0.1.1a Support

DeepRecall now requires `rlms>=0.1.1,<0.2.0`, unlocking depth>1 recursive subcalls, context compaction, cost tracking, and scaffold protection from the upstream RLM library. RLM-level limit exceptions (`TimeoutExceededError`, `TokenLimitExceededError`, `ErrorThresholdExceededError`) are now caught gracefully with partial answer recovery.

### Cost Tracking

Real USD cost is now extracted automatically when using OpenRouter. Every result includes cost data in `result.usage.total_cost_usd` and per-model breakdown.

```python
result = engine.query("question")
print(f"Cost: ${result.usage.total_cost_usd}")         # e.g. $0.0045
print(result.usage.model_breakdown)                      # per-model cost_usd
print(f"Budget spent: ${result.budget_status['cost_usd']}")
```

### Cost Budget Enforcement

`max_cost_usd` is now actively enforced -- both at the RLM level (stops the reasoning loop) and at the tracer level. Previously this was reserved for future use.

```python
result = engine.query(
    "Complex question?",
    budget=QueryBudget(max_cost_usd=0.10),  # Hard USD cap
)
```

### Context Compaction

For queries that require many reasoning steps, enable compaction to avoid hitting the model's context window limit. When enabled, RLM automatically summarises the conversation history when token usage nears the threshold.

```python
config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini"},
    compaction=True,                  # Enable context summarisation
    compaction_threshold_pct=0.85,    # Trigger at 85% of context window
)
```

### Execution Limits

New `max_timeout`, `max_errors`, and `max_tokens` config params give fine-grained control over RLM execution.

```python
config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini"},
    max_timeout=120.0,   # Kill after 2 minutes wall-clock
    max_errors=5,        # Abort after 5 consecutive REPL errors
    max_tokens=50000,    # Total token limit (input + output)
)
```

### Iteration Lifecycle Callbacks

New `on_iteration_start` and `on_iteration_complete` hooks fire before/after each RLM reasoning iteration, giving more granular observability than the existing `on_reasoning_step`.

```python
from deeprecall.core.callbacks import BaseCallback

class MyCallback(BaseCallback):
    def on_iteration_start(self, iteration):
        print(f"Starting iteration {iteration}...")

    def on_iteration_complete(self, iteration, has_final_answer):
        if has_final_answer:
            print(f"Got final answer at iteration {iteration}")
```

### Metadata Filters & Context Injection

`query()` now accepts `filters` (metadata dict applied to every search) and `context_prefix` (text prepended to the prompt). Useful for per-rule or per-section compliance queries.

```python
result = engine.query(
    "Does this deal comply with anti-money laundering rules?",
    filters={"section": "4.2"},
    context_prefix="Policy Section 4.2: Anti-Money Laundering Requirements",
)
```

### Async Embedding Functions

Vector stores now accept async `embedding_fn` callables. Async functions are auto-detected and wrapped, so you no longer need a manual thread pool executor workaround.

```python
async def my_async_embed(texts: list[str]) -> list[list[float]]:
    return await embedding_service.embed(texts)

store = MilvusStore(collection_name="docs", embedding_fn=my_async_embed)
```

---

## What's New in v0.3

### Exception Handling

All DeepRecall errors inherit from `DeepRecallError` -- catch at the boundary for production use.

```python
from deeprecall import DeepRecall, DeepRecallError, LLMProviderError, VectorStoreError

try:
    result = engine.query("question")
except LLMProviderError:
    # LLM call failed (timeout, rate limit, etc.)
    ...
except VectorStoreError:
    # Vector DB unreachable or query failed
    ...
except DeepRecallError:
    # Catch-all for any DeepRecall error
    ...
```

### Retry with Exponential Backoff

Automatic retries for transient LLM and vector store failures.

```python
from deeprecall import DeepRecall, DeepRecallConfig, RetryConfig

config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini"},
    retry=RetryConfig(max_retries=3, base_delay=1.0, jitter=True),
)
engine = DeepRecall(vectorstore=store, config=config)
```

### Batch Queries

Run multiple queries concurrently with a thread pool.

```python
results = engine.query_batch(
    ["Question 1?", "Question 2?", "Question 3?"],
    max_concurrency=4,
)
for r in results:
    print(r.answer[:100])
```

### FAISS Vector Store

Local vector index used by most ML teams.

```python
from deeprecall.vectorstores import FAISSStore

store = FAISSStore(dimension=384, embedding_fn=my_embed_fn)
store.add_documents(["Hello world", "Foo bar"])
results = store.search("greeting")

# Persistence
store.save("./my_index")
store = FAISSStore.load("./my_index", embedding_fn=my_embed_fn)
```

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
from deeprecall import DeepRecall, DeepRecallConfig, InMemoryCache, DiskCache, RedisCache

# In-memory (fastest, ephemeral -- good for dev)
config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini"},
    cache=InMemoryCache(max_size=500, default_ttl=3600),
)

# Disk / SQLite (persists across restarts, single machine)
config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini"},
    cache=DiskCache(db_path="./deeprecall_cache.db"),
)

# Redis (distributed, production -- works with AWS ElastiCache, GCP Memorystore, etc.)
config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini"},
    cache=RedisCache(url="redis://localhost:6379/0"),
)
engine = DeepRecall(vectorstore=store, config=config)
# Second identical query hits cache -- zero LLM cost
```

### Reranking

Improve search quality with Cohere or cross-encoder rerankers.

```python
from deeprecall.core import CohereReranker  # or: CrossEncoderReranker

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

# Async batch queries
results = await engine.query_batch(["q1?", "q2?"], max_concurrency=4)
```

### Server Auth & Rate Limiting

```bash
deeprecall serve --api-keys "key1,key2" --rate-limit 60 --port 8000
```

## How It Works

1. A lightweight HTTP server wraps your vector store on a random port
2. A `search_db(query, top_k, filters)` function is injected into the RLM's sandboxed REPL
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
| FAISS | `deeprecall[faiss]` | Yes |

All stores implement the same interface: `add_documents()`, `search()`, `delete()`, `count()`, `close()`.

All stores support context managers for automatic cleanup:

```python
with ChromaStore(collection_name="my_docs") as store:
    store.add_documents(["Hello world"])
    results = store.search("greeting")
# connections released automatically
```

### Custom Embedding Functions

Stores that require `embedding_fn` expect a callable with this signature. Both sync and async functions are supported -- async functions are auto-detected and wrapped.

```python
def my_embed_fn(texts: list[str]) -> list[list[float]]:
    """Takes a list of strings, returns a list of embedding vectors."""
    # Example using OpenAI:
    from openai import OpenAI
    client = OpenAI()
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return [e.embedding for e in response.data]

store = MilvusStore(collection_name="docs", embedding_fn=my_embed_fn)

# Async embedding functions also work directly (v0.4+):
async def my_async_embed(texts: list[str]) -> list[list[float]]:
    return await embedding_service.embed(texts)

store = MilvusStore(collection_name="docs", embedding_fn=my_async_embed)
```

## Framework Adapters

**LangChain** / **LlamaIndex** / **OpenAI-compatible API** -- see [adapters docs](https://github.com/kothapavan1998/deeprecall/blob/main/docs/adapters.md).

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
deeprecall status                      # Show version, installed extras
deeprecall benchmark --queries q.json  # Run benchmark
```

The CLI automatically loads environment variables from a `.env` file via `python-dotenv`, so you can set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. without exporting them in your shell.

## Project Structure

```
deeprecall/
├── core/           # Engine, config, guardrails, tracer, cache, callbacks, reranker
│   ├── exceptions.py    # DeepRecallError hierarchy
│   ├── retry.py         # Exponential backoff with jitter
│   ├── deprecations.py  # @deprecated decorator
│   ├── logging_config.py # configure_logging() helper
│   ├── cache.py          # InMemoryCache, DiskCache (SQLite)
│   ├── cache_redis.py    # RedisCache (distributed)
│   ├── callbacks.py      # ConsoleCallback, JSONLCallback, UsageTrackingCallback, ProgressCallback
│   ├── callback_otel.py  # OpenTelemetry distributed tracing
│   ├── async_engine.py   # AsyncDeepRecall (non-blocking wrapper)
│   └── ...
├── vectorstores/   # ChromaDB, Milvus, Qdrant, Pinecone, FAISS adapters
├── adapters/       # LangChain, LlamaIndex, OpenAI-compatible server
├── middleware/      # API key auth (sync + async), rate limiting (thread-safe)
├── prompts/        # System prompts for the RLM
└── cli.py          # CLI entry point

tests/
├── test_exceptions.py    # Exception hierarchy tests
├── test_retry.py         # Retry logic tests
├── test_batch.py         # Batch query tests
├── test_deprecations.py  # Deprecation utility tests
├── test_concurrency.py   # Thread safety & race condition tests
└── ...                   # 460+ tests (unit + integration + live + e2e)
```

## Contributing

```bash
git clone https://github.com/kothapavan1998/deeprecall.git
cd deeprecall
pip install -e ".[all]"
make check
```

See [CONTRIBUTING.md](https://github.com/kothapavan1998/deeprecall/blob/main/CONTRIBUTING.md).

## Citation

Built on [Recursive Language Models](https://arxiv.org/abs/2512.24601) by Zhang, Kraska, and Khattab (MIT).

## License

MIT
