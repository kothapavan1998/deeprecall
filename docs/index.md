# DeepRecall Documentation

**DeepRecall** is a recursive reasoning engine that bridges AI agents with vector databases using MIT's [Recursive Language Models (RLM)](https://github.com/alexzhang13/rlm) paradigm.

## What Makes DeepRecall Different?

Standard RAG (Retrieval-Augmented Generation) performs a single pass: retrieve relevant documents, stuff them into a prompt, and generate an answer. DeepRecall replaces this with **recursive, multi-hop reasoning**:

1. The LLM decomposes your query into sub-questions
2. It searches the vector database for relevant documents
3. It writes and executes Python code to analyze retrieved results
4. It decides whether it needs more information and searches again
5. It synthesizes a comprehensive answer across all reasoning steps

## Documentation

### Getting Started

- [Quickstart](quickstart.md) -- Installation, first query, how it works
- [Architecture](architecture.md) -- System design and component overview

### API Reference

- [API Reference](api-reference.md) -- Complete parameter reference for every class and method
  - `DeepRecall` / `DeepRecallEngine` constructor, `.query()`, `.add_documents()`
  - `DeepRecallConfig` -- all 16 configuration parameters
  - `QueryBudget` -- resource limits
  - `DeepRecallResult`, `Source`, `ReasoningStep`, `UsageInfo` -- return types

### Configuration

- [LLM Backends & Environments](backends.md) -- 9 LLM providers, 6 REPL sandbox environments
- [Budget Guardrails](guardrails.md) -- Token, time, cost, and search limits
- [Caching](caching.md) -- InMemoryCache, DiskCache, RedisCache
- [Reranking](reranking.md) -- CohereReranker, CrossEncoderReranker

### Observability

- [Callbacks](callbacks.md) -- ConsoleCallback, JSONLCallback, UsageTrackingCallback
- [Observability & OpenTelemetry](observability.md) -- Distributed tracing with Jaeger, Datadog, Grafana

### Integrations

- [Vector Stores](vectorstores.md) -- ChromaDB, Milvus, Qdrant, Pinecone
- [Framework Adapters](adapters.md) -- LangChain, LlamaIndex, OpenAI-compatible API

### Deployment

- [OpenAI-Compatible Server](server.md) -- REST API, authentication, rate limiting
- [Async & Concurrency](async.md) -- AsyncDeepRecall, thread safety, production concurrency
- [CLI Reference](cli.md) -- `init`, `ingest`, `query`, `serve`, `delete`

## Links

- [GitHub Repository](https://github.com/kothapavan1998/deeprecall)
- [PyPI Package](https://pypi.org/project/deeprecall/)
- [RLM Paper](https://arxiv.org/abs/2512.24601)
