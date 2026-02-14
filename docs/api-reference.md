# API Reference

Complete reference for every class, method, and parameter in DeepRecall.

---

## DeepRecall / DeepRecallEngine

The main entry point. Alias: `DeepRecall = DeepRecallEngine`.

### Constructor

```python
from deeprecall import DeepRecall

engine = DeepRecall(
    vectorstore=store,          # required
    config=None,                # or pass individual kwargs below
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": "sk-..."},
    verbose=False,
    max_iterations=15,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vectorstore` | `BaseVectorStore` | **required** | Any adapter: `ChromaStore`, `MilvusStore`, `QdrantStore`, `PineconeStore` |
| `config` | `DeepRecallConfig \| None` | `None` | Full config object. If provided, all other kwargs are ignored |
| `backend` | `str` | `"openai"` | LLM backend (see [Backends](backends.md)) |
| `backend_kwargs` | `dict` | `{"model_name": "gpt-4o-mini"}` | Backend-specific settings |
| `verbose` | `bool` | `False` | Rich console output during reasoning |
| `max_iterations` | `int` | `15` | Max reasoning iterations |
| `**kwargs` | `Any` | | Forwarded to `DeepRecallConfig` |

You can pass **either** a `config` object or individual kwargs -- not both.

### `.query()` Method

```python
result = engine.query(
    "What are the key themes?",
    root_prompt=None,
    top_k=None,
    budget=QueryBudget(max_search_calls=10),
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | **required** | The question or task |
| `root_prompt` | `str \| None` | `None` | Short prompt visible to the root LM. Defaults to `query` |
| `top_k` | `int \| None` | `None` | Override config `top_k` for this query |
| `budget` | `QueryBudget \| None` | `None` | Per-query resource limits (overrides config `budget`) |

### `.add_documents()` Method

```python
ids = engine.add_documents(
    documents=["doc 1...", "doc 2..."],
    metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
    ids=["id-1", "id-2"],
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `documents` | `list[str]` | **required** | Document texts |
| `metadatas` | `list[dict] \| None` | `None` | Metadata per document |
| `ids` | `list[str] \| None` | `None` | Custom IDs. Auto-generated if not provided |

### `.close()` and Context Manager

```python
# Context manager
with DeepRecall(vectorstore=store, backend="openai") as engine:
    result = engine.query("question")

# Or manual
engine = DeepRecall(vectorstore=store, backend="openai")
engine.close()
```

---

## DeepRecallConfig

Full configuration dataclass. Pass to `DeepRecall(config=...)`.

```python
from deeprecall import DeepRecallConfig, QueryBudget, InMemoryCache, ConsoleCallback

config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini", "api_key": "sk-..."},
    environment="local",
    environment_kwargs={},
    max_iterations=15,
    max_depth=1,
    top_k=5,
    verbose=False,
    log_dir=None,
    other_backends=None,
    other_backend_kwargs=None,
    budget=QueryBudget(max_search_calls=20),
    callbacks=[ConsoleCallback()],
    cache=InMemoryCache(max_size=500),
    cache_ttl=3600,
    reranker=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backend` | `BackendType` | `"openai"` | LLM provider. See [Backends](backends.md) for full list |
| `backend_kwargs` | `dict` | `{"model_name": "gpt-4o-mini"}` | Provider-specific: `model_name`, `api_key`, `base_url`, etc. |
| `environment` | `EnvironmentType` | `"local"` | REPL sandbox. See [Backends](backends.md#repl-environments) |
| `environment_kwargs` | `dict` | `{}` | Environment-specific: `image`, `timeout`, etc. |
| `max_iterations` | `int` | `15` | Max reasoning loop iterations before forcing an answer |
| `max_depth` | `int` | `1` | Max recursion depth for sub-LLM calls (`llm_query()`) |
| `top_k` | `int` | `5` | Default number of vector search results per `search_db()` call |
| `verbose` | `bool` | `False` | Show rich console panels during reasoning |
| `log_dir` | `str \| None` | `None` | Directory for JSONL trajectory logs. `None` disables |
| `other_backends` | `list[str] \| None` | `None` | Additional LLM backends for sub-calls via `llm_query()` |
| `other_backend_kwargs` | `list[dict] \| None` | `None` | Kwargs for each additional backend (parallel to `other_backends`) |
| `budget` | `QueryBudget \| None` | `None` | Default resource limits for all queries |
| `callbacks` | `list[BaseCallback] \| None` | `None` | Observability hooks (see [Callbacks](callbacks.md)) |
| `cache` | `BaseCache \| None` | `None` | Cache backend (see [Caching](caching.md)) |
| `cache_ttl` | `int` | `3600` | Time-to-live for cached results, in seconds |
| `reranker` | `BaseReranker \| None` | `None` | Post-retrieval reranker (see [Reranking](reranking.md)) |

---

## QueryBudget

Resource limits for a single query. All fields are optional -- set to `None` to disable that limit.

```python
from deeprecall import QueryBudget

budget = QueryBudget(
    max_iterations=10,
    max_search_calls=20,
    max_tokens=50000,
    max_time_seconds=30.0,
    max_cost_usd=0.50,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_iterations` | `int \| None` | `None` | Max reasoning loop steps |
| `max_search_calls` | `int \| None` | `None` | Max `search_db()` calls |
| `max_tokens` | `int \| None` | `None` | Total token budget (input + output) |
| `max_time_seconds` | `float \| None` | `None` | Wall-clock timeout in seconds |
| `max_cost_usd` | `float \| None` | `None` | Dollar cost limit (requires pricing data) |

---

## DeepRecallResult

Returned by `engine.query()`. Contains everything about the query execution.

```python
result = engine.query("What caused the 2008 crisis?")

result.answer              # str -- final synthesized answer
result.sources             # list[Source] -- documents retrieved
result.reasoning_trace     # list[ReasoningStep] -- every iteration
result.usage               # UsageInfo -- token counts
result.execution_time      # float -- wall-clock seconds
result.query               # str -- original query
result.budget_status       # dict | None -- budget utilization
result.error               # str | None -- error if budget exceeded
result.confidence          # float | None -- 0-1 score from source relevance
```

| Field | Type | Description |
|---|---|---|
| `answer` | `str` | Final answer. Prefixed with `[Partial - budget exceeded]` if budget hit |
| `sources` | `list[Source]` | Every document retrieved. Each has: `content`, `metadata`, `score`, `id` |
| `reasoning_trace` | `list[ReasoningStep]` | Each step: `iteration`, `action`, `code`, `output`, `searches`, `sub_llm_calls` |
| `usage` | `UsageInfo` | `total_input_tokens`, `total_output_tokens`, `total_calls`, `model_breakdown` |
| `execution_time` | `float` | Total seconds from start to finish |
| `query` | `str` | Original query string |
| `budget_status` | `dict \| None` | Keys: `iterations_used`, `search_calls_used`, `tokens_used`, `time_elapsed`, `budget_exceeded` |
| `error` | `str \| None` | Error message (e.g. "Budget exceeded: Search calls: 10/10") |
| `confidence` | `float \| None` | Average of top-3 source scores, 0-1. `None` if no sources |

### Serialization

```python
result.to_dict()  # Returns a plain dict (JSON-serializable)
```

---

## Source

A document retrieved during reasoning.

| Field | Type | Description |
|---|---|---|
| `content` | `str` | Document text |
| `metadata` | `dict` | Document metadata (source, filename, etc.) |
| `score` | `float` | Relevance score (0-1, higher = more relevant) |
| `id` | `str` | Document ID in the vector store |

## ReasoningStep

A single iteration in the recursive reasoning loop.

| Field | Type | Description |
|---|---|---|
| `iteration` | `int` | Step number (1-based) |
| `action` | `str` | What the LLM did: `"search"`, `"compute"`, `"final_answer"`, etc. |
| `code` | `str \| None` | Python code the LLM executed |
| `output` | `str \| None` | REPL output from the code |
| `searches` | `list[dict]` | Vector searches made: `{"query": ..., "results": ...}` |
| `sub_llm_calls` | `int` | Number of `llm_query()` calls in this step |
| `iteration_time` | `float \| None` | Time spent on this step |

## UsageInfo

Token usage breakdown.

| Field | Type | Description |
|---|---|---|
| `total_input_tokens` | `int` | Total prompt tokens across all LLM calls |
| `total_output_tokens` | `int` | Total completion tokens |
| `total_calls` | `int` | Number of LLM API calls |
| `model_breakdown` | `dict` | Per-model usage: `{"gpt-4o-mini": {"input_tokens": ..., "output_tokens": ..., "calls": ...}}` |
