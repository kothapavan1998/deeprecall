"""ChromaDB vector store adapter for DeepRecall."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from deeprecall.core.types import SearchResult
from deeprecall.vectorstores.base import BaseVectorStore


class ChromaStore(BaseVectorStore):
    """Vector store adapter for ChromaDB.

    Supports both in-memory and persistent ChromaDB clients. If no embeddings
    or embedding_fn are provided, ChromaDB's default embedding function is used.

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_directory: Path for persistent storage. None uses in-memory.
        host: ChromaDB server host for client/server mode.
        port: ChromaDB server port for client/server mode.
        embedding_fn: Optional custom embedding function.

    Example:
        ```python
        store = ChromaStore(collection_name="my_docs")
        store.add_documents(["Hello world", "Foo bar"])
        results = store.search("greeting", top_k=1)
        ```
    """

    def __init__(
        self,
        collection_name: str = "deeprecall",
        persist_directory: str | None = None,
        host: str | None = None,
        port: int = 8000,
        embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ):
        super().__init__(embedding_fn=embedding_fn)

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaStore. "
                "Install it with: pip install deeprecall[chroma]"
            ) from None

        if host is not None:
            self._client = chromadb.HttpClient(host=host, port=port)
        elif persist_directory is not None:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(name=collection_name)
        self._collection_name = collection_name

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

        kwargs: dict[str, Any] = {
            "documents": documents,
            "ids": ids,
        }
        if metadatas is not None:
            kwargs["metadatas"] = metadatas
        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        self._collection.add(**kwargs)
        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        doc_count = self._collection.count()
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(top_k, doc_count) if doc_count > 0 else top_k,
        }

        if filters is not None:
            kwargs["where"] = filters

        query_embeddings = self._generate_embeddings([query])
        if query_embeddings is not None:
            kwargs["query_embeddings"] = query_embeddings
            kwargs.pop("query_texts", None)

        results = self._collection.query(**kwargs)

        search_results: list[SearchResult] = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)
            result_ids = results["ids"][0] if results["ids"] else [""] * len(docs)

            for doc, meta, dist, rid in zip(docs, metas, distances, result_ids, strict=False):
                search_results.append(
                    SearchResult(
                        content=doc,
                        metadata=meta or {},
                        score=max(0.0, 1.0 - dist),  # ChromaDB distances -> similarity, clamped
                        id=rid,
                    )
                )

        return search_results

    def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)

    def count(self) -> int:
        return self._collection.count()
