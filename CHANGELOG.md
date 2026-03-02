# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-02-24

### Added

- **RLM v0.1.1a support** -- Upgraded dependency to `rlms>=0.1.1,<0.2.0` (pinned upper bound for safety).
- **Cost tracking** -- `UsageInfo` now includes `total_cost_usd` and per-model `cost_usd` in the breakdown. Costs are extracted automatically when using OpenRouter.
- **Compaction support** -- New `compaction` and `compaction_threshold_pct` config params. When enabled, RLM summarises conversation history when approaching the context window limit, allowing longer reasoning chains.
- **Execution limit params** -- New `max_timeout` (wall-clock seconds), `max_errors` (consecutive REPL error cap), and `max_tokens` (total token limit) config fields, passed through to RLM.
- **Native cost budget enforcement** -- `QueryBudget.max_cost_usd` is now enforced at the RLM level via its `max_budget` parameter, in addition to the existing tracer-level check.
- **Iteration lifecycle callbacks** -- `on_iteration_start(iteration)` and `on_iteration_complete(iteration, has_final_answer)` hooks on `BaseCallback`, `CallbackManager`, `JSONLCallback`, and `ProgressCallback`.
- **Full RLMLogger protocol** -- `DeepRecallTracer` now implements `clear_iterations()` and `get_trajectory()`, matching the updated RLM v0.1.1 logger interface.
- **Graceful RLM exception handling** -- `TimeoutExceededError`, `TokenLimitExceededError`, `ErrorThresholdExceededError`, `BudgetExceededError`, and `CancellationError` from RLM are now caught and converted to partial `DeepRecallResult` objects with the `partial_answer` preserved, instead of propagating as unhandled errors.
- **Budget exhaustion synthesis** -- When any budget limit is hit (iterations, search calls, time, tokens, cost), the engine now makes a one-shot LLM call to synthesize a best-effort answer from retrieved sources instead of leaking raw REPL state into `.answer`.
- **Async embedding_fn support** -- `BaseVectorStore` now accepts both sync and async `embedding_fn` callables. Async functions are auto-detected via `inspect.iscoroutinefunction()` and wrapped for synchronous execution. Fixes silent failure when async embedding services were passed directly.
- **`filters` param on `query()`** -- New optional `filters: dict` parameter constrains all `search_db()` calls to matching metadata. The LLM can also pass additional filters at call time.
- **`context_prefix` param on `query()`** -- New optional `context_prefix: str` parameter prepends contextual information (e.g., section name, rule ID) to the RLM prompt.
- **`search_db()` now accepts `filters`** -- The REPL-injected `search_db(query, top_k, filters)` function now supports metadata filtering, forwarded to the search server.
- **Expanded test suite** -- 41 new tests: 21 unit tests for partial synthesis, async embedding wrapping, filters, and context_prefix; 20 live integration tests verifying all new features against real OpenAI API with edge cases (budget exhaustion, special characters, empty stores, combined params). Plus 16 earlier tests validating v0.4 features against real Redis on Docker.

### Changed

- **`FINAL_VAR` prompt updated** -- System prompt now documents that `FINAL_VAR` accepts both variable names and direct values (upstream RLM v0.1.1a feature).
- **`history` variable documented** -- System prompt lists the `history` REPL variable available when compaction is enabled.
- **`max_cost_usd` is now active** -- `QueryBudget.max_cost_usd` docstring updated from "reserved for future use" to active, reflecting real cost extraction from OpenRouter.
- **`rlms` dependency pinned** -- Upper bound `<0.2.0` added to prevent silent breakage from future RLM major releases.

### Fixed

- **`compaction_threshold` parameter name mismatch** -- DeepRecall was passing `compaction_threshold` to RLM, but RLM's constructor expects `compaction_threshold_pct`. This caused a `TypeError` whenever compaction was enabled. Renamed everywhere.
- **System prompt format incompatibility** -- RLM v0.1.1a calls `.format(custom_tools_section=...)` on custom system prompts. Literal curly braces in code examples (e.g., `{len(results)}`) were interpreted as format placeholders, causing `KeyError`. All literal braces are now escaped and the `{custom_tools_section}` placeholder is included.
- **Redis integration test port** -- Updated from hardcoded 6380 to 6379 to match standard Docker Redis setup.

## [0.3.0] - 2026-02-15

### Added

- **Custom exception hierarchy** -- `DeepRecallError` base class with `ConfigurationError`, `VectorStoreError`, `VectorStoreConnectionError`, `LLMProviderError`, `LLMTimeoutError`, `LLMRateLimitError`, `CacheError`, and `SearchServerError`. Production teams can now `except DeepRecallError` at the boundary.
- **Retry / resilience layer** -- `RetryConfig` with exponential backoff + jitter for transient LLM and vector store failures. Honors `Retry-After` headers from rate-limited providers.
- **Batch query API** -- `engine.query_batch()` for concurrent multi-query execution via thread pool (sync) and `asyncio.gather` (async). Returns results in input order with per-query error isolation.
- **FAISS vector store** -- `FAISSStore` adapter for the most widely used local vector index. Supports L2 and inner-product metrics, persistence via `save()`/`load()`, and ID-based deletion.
- **Enhanced callbacks** -- `on_sub_llm_call()` and `on_progress()` hooks. `ProgressCallback` for thread-safe event accumulation. Search events now fire `on_search()` from the HTTP search server.
- **Search server reuse** -- `reuse_search_server=True` (default) in config. The background HTTP server persists across queries, eliminating per-query port binding overhead.
- **Deprecation infrastructure** -- `@deprecated` decorator and `warn_deprecated_param()` helper for graceful API evolution with clear warnings.
- **Structured logging helper** -- `configure_logging()` to set up the `deeprecall.*` logger namespace without touching the root logger.
- **CLI `status` command** -- Shows Python version, DeepRecall version, installed extras, and RLM version.
- **CLI `benchmark` command** -- Run queries from a JSON file and collect timing/quality metrics to structured JSON output.
- **PEP 561 `py.typed` marker** -- mypy/pyright/Pylance now type-check against DeepRecall.
- **Version single source of truth** -- `__version__` is read from `importlib.metadata` at runtime; no more version string drift.
- **Lazy imports for optional dependencies** -- `import deeprecall` no longer fails when `redis` or `opentelemetry` are not installed.
- **Comprehensive test suite** -- 358 unit tests + 3 e2e tests covering engine, async engine, API server, CLI, middleware, callbacks, concurrency, caching (in-memory, disk, Redis), rerankers, retry, guardrails, all vector store adapters (ChromaDB, FAISS, Milvus, Qdrant, Pinecone), framework adapters (LangChain, LlamaIndex), and prompt templates.
- **CHANGELOG.md** -- This file.
- **`.pre-commit-config.yaml`** -- Ruff lint + format hooks for contributors.
- **CI enhancements** -- Added `typecheck` (mypy), `pre-commit` validation, and coverage reporting jobs.

### Changed

- `BudgetExceededError` is now defined in `deeprecall.core.exceptions` and re-exported from `deeprecall.core.guardrails` for backward compatibility.
- Vector store adapters (Chroma, Milvus, Qdrant, Pinecone) now wrap provider exceptions in `VectorStoreError` / `VectorStoreConnectionError`.
- Engine wraps RLM failures as `LLMProviderError` and search server failures as `SearchServerError`.
- `DeepRecallConfig` gained `retry` and `reuse_search_server` fields (both default to preserving existing behavior).

### Fixed

- CLI version was hardcoded as "0.2.0" while package was at "0.2.1". Now derived from package metadata.
- OpenAI adapter server version was hardcoded as "0.2.0". Now derived from package metadata.
- **Milvus filter expression injection** -- Filter keys are now validated against an alphanumeric allowlist, and values are properly escaped to prevent operator injection.
- **FAISS path traversal** -- `save()` and `load()` now validate and normalise paths via `os.path.normpath`/`abspath` with null-byte rejection.
- **FAISS atomic metadata writes** -- `save()` now writes metadata to a temp file and atomically renames it, preventing corruption on interrupted writes.
- **FAISS filtered search returning fewer results than `top_k`** -- When filters are provided, FAISS now fetches `top_k * 4` candidates and stops once `top_k` matches are collected.
- **Retry jitter doubling delays** -- Jitter now uses decorrelated randomisation (`random.uniform(0, delay)`) instead of additive jitter that doubled the computed delay.
- **Retry overflow for large attempt numbers** -- Exponential backoff computation now catches `OverflowError` and falls back to `max_delay`.
- **`LLMRateLimitError` accepting negative `retry_after`** -- Negative values are now silently clamped to `None`.
- **`query_batch()` could return `None` entries** -- Any unfulfilled result slots are now filled with error-bearing `DeepRecallResult` objects before returning.
- **Search server leaking internal error details** -- HTTP error responses now return generic messages; full details are logged server-side only.
- **CLI `benchmark` command accepting unbounded JSON files** -- Added 10 MB file size limit and `JSONDecodeError` handling.
- **`query_batch()` accepting unbounded input** -- Added 10,000 query maximum to prevent memory exhaustion.
- Resolved four mypy strict-mode type errors across `deprecations.py`, `engine.py`, and `async_engine.py`.
- **`query_batch()` race condition with shared search server** -- Batch queries now temporarily disable server reuse so each thread gets its own isolated search server with independent source tracking.
- **Cache deserialization type mismatch** -- Persistent cache backends (Disk/Redis) return raw dicts; engine now reconstructs `DeepRecallResult` from dict via new `from_dict()` classmethod on cache hits.
- **Pinecone metadata key collision** -- Document text is now stored under `_deeprecall_content` instead of `content` to prevent user metadata from silently overwriting document text.
- **Off-by-one in budget guardrails** -- Budget checks now use `>=` instead of `>`, so `max_iterations=5` means exactly 5 iterations, not 6.
- **Async `query_batch()` missing batch size limit** -- Now enforces the same 10,000 query maximum as the sync version.
- **`config.to_dict()` missing fields** -- Now includes `environment_kwargs`, `log_dir`, `other_backends`, `other_backend_kwargs`, `callbacks`, `cache`, `reranker`, `retry`, and `reuse_search_server`.
- **CLI `ingest --persist-dir` default** -- Now defaults to `./faiss_index` when `--vectorstore faiss` instead of the confusing `./chroma_db`.
- **`ProgressCallback` and `RetryConfig` not exported** -- Both are now available from the top-level `deeprecall` package.
- Streaming warning documented in `openai_server.py` -- `_stream_response` now clearly states it runs the full query before chunking.
- CI now installs `pytest-asyncio` and `pytest-mock` to avoid silent test skips.
- **Async `query_batch()` race condition with shared search server** -- Now temporarily disables server reuse during batch execution (same fix as sync version).
- **Pinecone `or`-pattern treating falsy values as missing** -- All `getattr(...) or dict.get(...)` patterns replaced with `is None` checks so `score=0.0`, `id=""`, and `metadata={}` are preserved correctly across Pinecone SDK v5+.
- **Pinecone `_deeprecall_content` empty-string corruption** -- Content retrieval now uses `is not None` instead of `or`, so documents with empty-string content are not silently replaced with legacy `content` key.
- **ChromaDB empty collection search crash** -- `search()` now returns `[]` immediately when the collection has 0 documents, preventing ChromaDB from raising on `n_results > 0`.
- **Exception subclasses not exported** -- All 10 exception classes (`BudgetExceededError`, `CacheError`, `LLMRateLimitError`, `LLMTimeoutError`, `SearchServerError`, `VectorStoreConnectionError`) are now importable from the top-level `deeprecall` package.
- **CLI `delete`/`query`/`benchmark` ignoring `--persist-dir` defaults** -- Smart defaults (`./chroma_db` for ChromaDB, `./faiss_index` for FAISS) now apply to all CLI commands, not just `ingest`.
- **Sync `query_batch()` config mutation race condition** -- Save/restore of `reuse_search_server` is now protected by a `threading.Lock`, preventing corruption from overlapping batch calls.
- **`_compute_confidence` exceeding 1.0** -- Confidence is now clamped to `[0.0, 1.0]` so dot-product scores > 1 from Milvus/Pinecone cannot produce invalid values.
- **OpenAI server exposing stack traces on errors** -- `engine.query()` in the chat completions endpoint (both streaming and non-streaming) is now wrapped in try/except; errors are logged server-side and a generic `500 Internal Server Error` is returned to clients.
- **`config.to_dict()` leaking API keys from `other_backend_kwargs`** -- All backend kwargs (primary and secondary) now strip `api_key`, `api_secret`, `secret_key`, and `token` before serialization.
- **Budget `time`/`cost` checks using `>` while other limits use `>=`** -- All budget checks now consistently use `>=` for boundary enforcement.
- **`top_k=0` silently falling through to default** -- Engine now uses `is not None` check instead of falsy `or` pattern, so explicitly passing `top_k=0` is respected.
- **Search server cache key non-deterministic for dict filters** -- Filters are now serialized via `json.dumps(sort_keys=True)` for insertion-order-independent cache keys.
- **No double-start protection on `SearchServer`** -- `start()` now returns immediately if the server is already running, preventing leaked threads.
- **First streaming SSE chunk missing `role: assistant`** -- Now conforms to the OpenAI streaming spec; strict clients (Python openai SDK) no longer reject the response.
- **CLI hardcoding `OPENAI_API_KEY` for all backends** -- Each backend now reads its own environment variable (e.g., `ANTHROPIC_API_KEY` for `--backend anthropic`, `GOOGLE_API_KEY` for `--backend gemini`).
- **`DeepRecallConfig` accepting negative values** -- Added `__post_init__` validation rejecting negative `max_iterations`, `max_depth`, `top_k`, and `cache_ttl`.
- **`QueryBudget` accepting negative limits** -- Added `__post_init__` validation rejecting negative values for all budget fields.
- **No `close()` on vector store base class** -- `BaseVectorStore.close()` now provides a default no-op; `QdrantStore.close()` releases the client connection.
- **CLI `ingest` accepting arbitrarily large files and crashing on binary files** -- Added 50 MB per-file size limit and `UnicodeDecodeError` handling with skip warnings.
- **`vectorstores.__init__.__all__` missing store class names** -- All five store classes are now listed in `__all__`.
- **`JSONLCallback` not logging search, sub-LLM, or progress events** -- Added `on_search()`, `on_sub_llm_call()`, and `on_progress()` hooks.
- **`ConsoleCallback._start_time` not thread-safe in batch mode** -- Changed to `threading.local()` so each query thread gets its own start time.
- **CI missing coverage threshold** -- Added `--cov-fail-under=60` to the test job.
- **`_build_cache_key()` calling `vectorstore.count()` on every query** -- Removed the count call; cache keys are now based on query, backend, model, and top_k only.
- **Qdrant `delete()` failing on non-UUID string IDs** -- Numeric string IDs are now auto-converted to integers before passing to Qdrant.
- **`query_batch()` config mutation not thread-safe** -- Save/restore of `reuse_search_server` now holds a lock for the full batch lifetime, preventing corruption from overlapping batch calls.
- **`__repr__` triggering network call** -- `DeepRecallEngine.__repr__` no longer calls `vectorstore.count()`, which could fail if the database is unreachable.
- **ChromaDB score formula broken for L2 metric** -- Score now uses `1 / (1 + dist)` instead of `1 - dist`, which produced `0.0` for all L2 results. Works correctly for L2, cosine, and inner-product.
- **Milvus `close()` missing** -- `MilvusStore.close()` now releases the gRPC connection, preventing connection leaks in long-running services.
- **Milvus NaN/Inf filter values accepted** -- Float filter values are now validated; `NaN` and `Inf` are rejected with a clear error message.
- **Qdrant invalid distance silently falling back to Cosine** -- Invalid distance strings (e.g., `"L2"`, `"cosine"`) now raise `ValueError` instead of silently defaulting.
- **Pinecone batch upsert error missing batch index** -- Error messages now include the batch start index for easier debugging.
- **`BaseVectorStore.add_documents([])` accepted** -- `_validate_inputs()` now rejects empty document lists.
- **No context manager on vector stores** -- `BaseVectorStore` now implements `__enter__` / `__exit__` for `with store:` usage.
- **OpenAI server discarding multi-turn conversation history** -- Prior user/assistant messages are now included as conversation context, not just the last user message.
- **OpenAI server `/v1/cache/clear` unhandled errors** -- Redis connection failures now return a proper 500 response instead of an unhandled exception.
- **OpenAI server private attribute access** -- `/v1/usage` now uses `getattr()` instead of accessing `engine._callback_manager` directly.
- **`SearchServer` port stale after stop/start** -- `start()` now re-acquires a free port, preventing "address already in use" errors after a restart.
- **Source dedup key collision on 100-char prefix** -- Dedup now uses SHA-256 hash of full content instead of a prefix, so documents sharing the same first 100 characters are no longer conflated.
- **`config.to_dict()` serializing retry as `bool`** -- Now serializes retry as a dict with `max_retries`, `base_delay`, and `max_delay` fields.
- **CLI type annotations incorrect** -- `persist_dir` parameters now correctly typed as `str | None`; `_build_vectorstore` and `_build_engine` return proper types instead of `Any`.
- **`vectorstores.__init__` missing `__dir__`** -- Added `__dir__()` override so IDE autocomplete discovers store classes.
- **`core/__init__.py` missing exports** -- `ProgressCallback`, `RetryConfig`, `CohereReranker`, and `CrossEncoderReranker` now importable from `deeprecall.core`.
- **`async query_batch` Semaphore(0) deadlock** -- `max_concurrency < 1` now raises `ValueError` instead of hanging forever.
- **README missing install extras** -- Added `langchain`, `llamaindex`, `rerank-cohere`, `rerank-cross-encoder` to the install section.
- **README missing `close()` / context manager docs** -- Quick Start now shows `with` usage; added production cleanup guidance.
- **README missing `embedding_fn` signature** -- Documented the required callable contract with an OpenAI example.
- **CONTRIBUTING.md wrong GitHub URL** -- Fixed `kothapavan` to `kothapavan1998`.
- **CONTRIBUTING.md stale roadmap** -- Marked FAISS and async support as shipped; updated remaining items.
- **CI not ignoring `test_v2_e2e.py`** -- Added to pytest ignore list (requires real API key).
- **README test count overstated** -- Changed from "180+" to "170+" to match actual count.
- **Async `query_batch()` lock released too early** -- Lock now held for full batch lifetime via `asyncio.Lock`, preventing concurrent batches from corrupting the `reuse_search_server` flag.
- **`config.log_dir` not wired** -- Setting `log_dir` now auto-creates a `JSONLCallback` if none is present, matching user expectations.
- **Tracer regex recompiled on every call** -- Search-detection patterns promoted to class-level compiled constants.
- **Dead code cleanup** -- Removed unused `get_total_search_calls()` method from tracer. Moved `hashlib` to module-level import in engine.

## [0.2.1] - 2025-01-15

### Fixed

- Minor bug fixes and documentation improvements.

## [0.2.0] - 2025-01-10

### Added

- Budget guardrails (`QueryBudget`, `BudgetStatus`, `BudgetExceededError`).
- Reasoning trace with full step-by-step visibility.
- Callback system (`ConsoleCallback`, `JSONLCallback`, `UsageTrackingCallback`).
- OpenTelemetry distributed tracing (`OpenTelemetryCallback`).
- Caching backends (`InMemoryCache`, `DiskCache`, `RedisCache`).
- Reranking (`CohereReranker`, `CrossEncoderReranker`).
- Async engine (`AsyncDeepRecallEngine`).
- OpenAI-compatible API server.
- LangChain and LlamaIndex adapters.
- API key auth and rate limiting middleware.
- CLI (`serve`, `query`, `ingest`, `delete`, `init`).

## [0.1.0] - 2024-12-01

### Added

- Initial release with core engine and ChromaDB support.
