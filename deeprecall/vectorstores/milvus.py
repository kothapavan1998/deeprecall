"""Milvus vector store adapter for DeepRecall."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from deeprecall.core.types import SearchResult
from deeprecall.vectorstores.base import BaseVectorStore

# Default embedding dimension when using OpenAI text-embedding-3-small
_DEFAULT_DIMENSION = 1536


class MilvusStore(BaseVectorStore):
    """Vector store adapter for Milvus.

    Connects to a Milvus instance and provides document storage and
    vector similarity search. Requires an embedding function since
    Milvus does not generate embeddings natively.

    Args:
        collection_name: Name of the Milvus collection.
        uri: Milvus server URI (default: "http://localhost:19530").
        dimension: Vector embedding dimension.
        metric_type: Distance metric ("COSINE", "L2", "IP").
        embedding_fn: Function to generate embeddings from text.
            Required for add_documents() and search() unless pre-computed
            embeddings are provided.

    Example:
        ```python
        from deeprecall.vectorstores import MilvusStore

        store = MilvusStore(
            collection_name="my_docs",
            uri="http://localhost:19530",
            embedding_fn=my_embed_fn,
        )
        store.add_documents(["Hello world"])
        results = store.search("greeting")
        ```
    """

    def __init__(
        self,
        collection_name: str = "deeprecall",
        uri: str = "http://localhost:19530",
        dimension: int = _DEFAULT_DIMENSION,
        metric_type: str = "COSINE",
        embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
        **kwargs: Any,
    ):
        super().__init__(embedding_fn=embedding_fn)

        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise ImportError(
                "pymilvus is required for MilvusStore. "
                "Install it with: pip install deeprecall[milvus]"
            ) from None

        self._client = MilvusClient(uri=uri, **kwargs)
        self._collection_name = collection_name
        self._dimension = dimension
        self._metric_type = metric_type

        # Create collection if it doesn't exist
        if not self._client.has_collection(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                dimension=dimension,
                metric_type=metric_type,
                auto_id=False,
            )

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        self._validate_inputs(documents, metadatas, ids, embeddings)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        if embeddings is None:
            embeddings = self._generate_embeddings(documents)
            if embeddings is None:
                raise ValueError(
                    "MilvusStore requires embeddings. Either provide an embedding_fn "
                    "in the constructor or pass pre-computed embeddings."
                )

        data = []
        for i, (doc_id, doc, emb) in enumerate(zip(ids, documents, embeddings, strict=False)):
            record: dict[str, Any] = {
                "id": doc_id,
                "vector": emb,
                "content": doc,
            }
            if metadatas and i < len(metadatas):
                record["metadata"] = metadatas[i]
            data.append(record)

        self._client.insert(collection_name=self._collection_name, data=data)
        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        query_embeddings = self._generate_embeddings([query])
        if query_embeddings is None:
            raise ValueError(
                "MilvusStore requires an embedding_fn for search. Provide one in the constructor."
            )

        search_params: dict[str, Any] = {
            "collection_name": self._collection_name,
            "data": query_embeddings,
            "limit": top_k,
            "output_fields": ["content", "metadata"],
        }

        if filters is not None:
            filter_parts = []
            for key, value in filters.items():
                # Sanitize key and value to prevent filter expression injection
                safe_key = str(key).replace('"', '\\"')
                if isinstance(value, str):
                    safe_val = value.replace('"', '\\"')
                    filter_parts.append(f'metadata["{safe_key}"] == "{safe_val}"')
                elif isinstance(value, bool):
                    filter_parts.append(f'metadata["{safe_key}"] == {"true" if value else "false"}')
                elif isinstance(value, (int, float)):
                    filter_parts.append(f'metadata["{safe_key}"] == {value}')
                else:
                    # Convert other types to string
                    safe_val = str(value).replace('"', '\\"')
                    filter_parts.append(f'metadata["{safe_key}"] == "{safe_val}"')
            if filter_parts:
                search_params["filter"] = " and ".join(filter_parts)

        results = self._client.search(**search_params)

        search_results: list[SearchResult] = []
        if results:
            for hits in results:
                for hit in hits:
                    entity = hit.get("entity", {})
                    search_results.append(
                        SearchResult(
                            content=entity.get("content", ""),
                            metadata=entity.get("metadata", {}),
                            score=hit.get("distance", 0.0),
                            id=str(hit.get("id", "")),
                        )
                    )

        return search_results

    def delete(self, ids: list[str]) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            ids=ids,
        )

    def count(self) -> int:
        stats = self._client.get_collection_stats(self._collection_name)
        row_count = stats.get("row_count", 0)
        return int(row_count)  # some versions return string
