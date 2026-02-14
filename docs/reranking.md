# Reranking

Rerankers re-score search results using a dedicated relevance model, improving what the LLM reasons over. Vector similarity search is fast but approximate -- rerankers are slower but more accurate.

---

## When to Use Reranking

- Your queries are complex or multi-faceted
- Embedding similarity misses semantically relevant documents
- You want higher precision in the top results
- You can afford 50-200ms of additional latency per search

---

## CohereReranker

Uses the [Cohere Rerank API](https://docs.cohere.com/reference/rerank). Cloud-hosted, no GPU needed.

```bash
pip install deeprecall[rerank-cohere]
```

```python
from deeprecall import DeepRecall, DeepRecallConfig
from deeprecall.core.reranker import CohereReranker

config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini"},
    reranker=CohereReranker(
        api_key="co-...",          # or set COHERE_API_KEY env var
        model="rerank-v3.5",       # default
    ),
)
engine = DeepRecall(vectorstore=store, config=config)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str \| None` | `None` | Cohere API key. Falls back to `COHERE_API_KEY` env var |
| `model` | `str` | `"rerank-v3.5"` | Cohere rerank model name |

### How It Works

1. Vector store returns top-K results by embedding similarity
2. CohereReranker sends the query + all result texts to Cohere's API
3. Cohere returns relevance scores (0-1) for each query-document pair
4. Results are re-sorted by relevance score and trimmed to `top_k`

---

## CrossEncoderReranker

Uses a [sentence-transformers](https://www.sbert.net/) cross-encoder model. Runs locally, no API calls needed.

```bash
pip install deeprecall[rerank-cross-encoder]
```

```python
from deeprecall import DeepRecall, DeepRecallConfig
from deeprecall.core.reranker import CrossEncoderReranker

config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini"},
    reranker=CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",  # default
    ),
)
engine = DeepRecall(vectorstore=store, config=config)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | HuggingFace cross-encoder model |

### Popular Models

| Model | Size | Speed | Quality |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 23M | Fast | Good |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M | Medium | Better |
| `BAAI/bge-reranker-base` | 278M | Slow | Best |

### How It Works

1. Vector store returns top-K results
2. CrossEncoderReranker creates (query, document) pairs for each result
3. The cross-encoder model scores each pair locally
4. Results are re-sorted by score and trimmed to `top_k`

---

## Custom Reranker

Implement the `BaseReranker` interface:

```python
from deeprecall.core.reranker import BaseReranker
from deeprecall.core.types import SearchResult

class MyReranker(BaseReranker):
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        # Your reranking logic here
        scored = [(r, my_score(query, r.content)) for r in results]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            SearchResult(content=r.content, metadata=r.metadata, score=s, id=r.id)
            for r, s in scored[:top_k]
        ]
```

---

## Comparison

| | CohereReranker | CrossEncoderReranker |
|---|---|---|
| **Runs** | Cloud API | Local (CPU/GPU) |
| **Latency** | 50-150ms | 100-500ms (CPU) |
| **Cost** | Cohere API pricing | Free (open-source models) |
| **Privacy** | Documents sent to Cohere | Documents stay local |
| **Quality** | Excellent | Good to excellent |
| **Install** | `deeprecall[rerank-cohere]` | `deeprecall[rerank-cross-encoder]` |

### Fallback Behavior

Both rerankers have built-in error handling. If the reranking call fails (API error, model load failure), the original results are returned unchanged with a warning logged. Your queries never break due to a reranker issue.
