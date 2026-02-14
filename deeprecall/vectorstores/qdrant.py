"""Qdrant vector store adapter for DeepRecall."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from deeprecall.core.types import SearchResult
from deeprecall.vectorstores.base import BaseVectorStore

_DEFAULT_DIMENSION = 1536


class QdrantStore(BaseVectorStore):
    """Vector store adapter for Qdrant.

    Connects to a Qdrant instance for vector similarity search.
    Requires an embedding function since Qdrant stores raw vectors.

    Args:
        collection_name: Name of the Qdrant collection.
        url: Qdrant server URL (default: "http://localhost:6333").
        dimension: Vector embedding dimension.
        distance: Distance metric ("Cosine", "Euclid", "Dot").
        embedding_fn: Function to generate embeddings from text.
        api_key: Optional API key for Qdrant Cloud.

    Example:
        ```python
        from deeprecall.vectorstores import QdrantStore

        store = QdrantStore(
            collection_name="my_docs",
            embedding_fn=my_embed_fn,
        )
        store.add_documents(["Hello world"])
        results = store.search("greeting")
        ```
    """

    def __init__(
        self,
        collection_name: str = "deeprecall",
        url: str = "http://localhost:6333",
        dimension: int = _DEFAULT_DIMENSION,
        distance: str = "Cosine",
        embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(embedding_fn=embedding_fn)

        try:
            from qdrant_client import QdrantClient, models
        except ImportError:
            raise ImportError(
                "qdrant-client is required for QdrantStore. "
                "Install it with: pip install deeprecall[qdrant]"
            ) from None

        self._models = models
        self._client = QdrantClient(url=url, api_key=api_key, **kwargs)
        self._collection_name = collection_name
        self._dimension = dimension

        # Map string distance to Qdrant enum
        distance_map = {
            "Cosine": models.Distance.COSINE,
            "Euclid": models.Distance.EUCLID,
            "Dot": models.Distance.DOT,
        }
        qdrant_distance = distance_map.get(distance, models.Distance.COSINE)

        # Create collection if it doesn't exist
        collections = [c.name for c in self._client.get_collections().collections]
        if collection_name not in collections:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=qdrant_distance,
                ),
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
                    "QdrantStore requires embeddings. Either provide an embedding_fn "
                    "in the constructor or pass pre-computed embeddings."
                )

        points = []
        for i, (doc_id, doc, emb) in enumerate(zip(ids, documents, embeddings, strict=False)):
            payload: dict[str, Any] = {"content": doc}
            if metadatas and i < len(metadatas):
                payload["metadata"] = metadatas[i]

            points.append(
                self._models.PointStruct(
                    id=doc_id,
                    vector=emb,
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=self._collection_name, points=points)
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
                "QdrantStore requires an embedding_fn for search. Provide one in the constructor."
            )

        search_kwargs: dict[str, Any] = {
            "collection_name": self._collection_name,
            "query_vector": query_embeddings[0],
            "limit": top_k,
            "with_payload": True,
        }

        if filters is not None:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    self._models.FieldCondition(
                        key=f"metadata.{key}",
                        match=self._models.MatchValue(value=value),
                    )
                )
            if conditions:
                search_kwargs["query_filter"] = self._models.Filter(must=conditions)

        results = self._client.search(**search_kwargs)

        search_results: list[SearchResult] = []
        for point in results:
            payload = point.payload or {}
            search_results.append(
                SearchResult(
                    content=payload.get("content", ""),
                    metadata=payload.get("metadata", {}),
                    score=point.score,
                    id=str(point.id),
                )
            )

        return search_results

    def delete(self, ids: list[str]) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=self._models.PointIdsList(points=ids),
        )

    def count(self) -> int:
        info = self._client.get_collection(self._collection_name)
        return info.points_count or 0
