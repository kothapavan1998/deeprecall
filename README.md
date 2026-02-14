<h1 align="center">DeepRecall</h1>

<p align="center">
  <b>Recursive reasoning over your data. Plug into any vector DB or agent framework.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/deeprecall/"><img src="https://img.shields.io/pypi/v/deeprecall?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/deeprecall/"><img src="https://img.shields.io/pypi/pyversions/deeprecall" alt="Python"></a>
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
pip install deeprecall[all]       # Everything
```

## Usage

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
print(result.sources)
print(result.execution_time)
```

## What happens when you call `.query()`

1. A lightweight HTTP server wraps your vector store on a random port
2. A `search_db(query, top_k)` function is injected into the RLM's sandboxed REPL
3. The LLM enters a recursive loop -- it can search, write Python, call sub-LLMs, and search again
4. When it has enough info, it returns a `FINAL()` answer
5. You get back the answer, every source document accessed, token usage, and execution time

No modifications to RLM. It's used as a pip dependency. The bridge is `setup_code` + a custom system prompt.

## Vector Stores

| Store | Install | Needs embedding_fn? |
|-------|---------|---------------------|
| ChromaDB | `deeprecall[chroma]` | No (built-in) |
| Milvus | `deeprecall[milvus]` | Yes |
| Qdrant | `deeprecall[qdrant]` | Yes |
| Pinecone | `deeprecall[pinecone]` | Yes |

```python
from deeprecall.vectorstores import ChromaStore, MilvusStore, QdrantStore, PineconeStore

# ChromaDB -- no embedding function needed
store = ChromaStore(collection_name="docs")

# Milvus / Qdrant / Pinecone -- pass your own embedding function
store = MilvusStore(collection_name="docs", uri="http://localhost:19530", embedding_fn=my_fn)
```

All stores implement the same interface: `add_documents()`, `search()`, `delete()`, `count()`.

## Framework Adapters

**LangChain**

```python
from deeprecall.adapters.langchain import DeepRecallRetriever, DeepRecallChatModel

retriever = DeepRecallRetriever(engine=engine)
docs = retriever.invoke("question")

llm = DeepRecallChatModel(engine=engine)
response = llm.invoke("question")
```

**LlamaIndex**

```python
from deeprecall.adapters.llamaindex import DeepRecallQueryEngine

query_engine = DeepRecallQueryEngine(engine=engine)
response = query_engine.query("question")
```

**OpenAI-compatible API** -- works with any client that speaks the OpenAI protocol:

```bash
deeprecall serve --vectorstore chroma --collection my_docs --port 8000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="deeprecall",
    messages=[{"role": "user", "content": "question"}],
)
```

## CLI

```bash
deeprecall ingest --path ./documents/ --vectorstore chroma --collection my_docs
deeprecall query "What is the conclusion?" --vectorstore chroma --collection my_docs
deeprecall serve --vectorstore chroma --collection my_docs --port 8000
```

## Configuration

```python
from deeprecall import DeepRecall, DeepRecallConfig

config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o", "api_key": "sk-..."},
    max_iterations=15,
    max_depth=1,
    top_k=5,
    verbose=True,
)

engine = DeepRecall(vectorstore=store, config=config)
```

Supported backends: `openai`, `anthropic`, `azure_openai`, `gemini`, `vllm`, `litellm`, `portkey`, `openrouter`.

## Project Structure

```
deeprecall/
├── core/           # Engine, config, types, search server
├── vectorstores/   # ChromaDB, Milvus, Qdrant, Pinecone adapters
├── adapters/       # LangChain, LlamaIndex, OpenAI-compatible server
├── prompts/        # System prompts for the RLM
└── cli.py          # CLI entry point
```

## Contributing

```bash
git clone https://github.com/kothapavan/deeprecall.git
cd deeprecall
pip install -e ".[all,dev,test]"
make check
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

Built on [Recursive Language Models](https://arxiv.org/abs/2512.24601) by Zhang, Kraska, and Khattab (MIT).

## License

MIT
