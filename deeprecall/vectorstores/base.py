"""Base vector store interface for DeepRecall."""

from __future__ import annotations

import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from deeprecall.core.types import SearchResult

_logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract base class for all vector store adapters.

    Every vector store adapter must implement these four methods to provide
    a consistent interface for the DeepRecall engine.

    Args:
        embedding_fn: Optional custom embedding function. Accepts both sync and async
            callables. Async functions are auto-wrapped for sync execution.
            Signature: (texts: list[str]) -> list[list[float]]
    """

    def __init__(self, embedding_fn: Callable[[list[str]], list[list[float]]] | None = None):
        self._async_embedding_fn: Callable | None = None
        self.embedding_fn: Callable[[list[str]], list[list[float]]] | None = embedding_fn
        if embedding_fn is not None and inspect.iscoroutinefunction(embedding_fn):
            _logger.debug("Wrapping async embedding_fn for synchronous execution")
            self._async_embedding_fn = embedding_fn
            self.embedding_fn = self._wrap_async_embedding_fn(embedding_fn)

    @abstractmethod
    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: List of document texts to add.
            metadatas: Optional metadata dicts for each document.
            ids: Optional unique IDs. Auto-generated if not provided.
            embeddings: Optional pre-computed embeddings. If not provided,
                the store will generate them using its embedding function.

        Returns:
            List of document IDs that were added.

        Raises:
            ValueError: If metadatas/ids/embeddings length doesn't match documents.
        """
        raise NotImplementedError

    def _validate_inputs(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Validate that input list lengths match and documents is non-empty."""
        if not documents:
            raise ValueError("documents must be a non-empty list")
        n = len(documents)
        if metadatas is not None and len(metadatas) != n:
            raise ValueError(
                f"metadatas length ({len(metadatas)}) must match documents length ({n})"
            )
        if ids is not None and len(ids) != n:
            raise ValueError(f"ids length ({len(ids)}) must match documents length ({n})")
        if embeddings is not None and len(embeddings) != n:
            raise ValueError(
                f"embeddings length ({len(embeddings)}) must match documents length ({n})"
            )

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search the vector store for documents similar to the query.

        Args:
            query: The search query text.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Return the total number of documents in the store."""
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027
        """Release any resources held by the vector store.

        Subclasses with persistent connections should override this.
        """

    def __enter__(self) -> BaseVectorStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @staticmethod
    def _wrap_async_embedding_fn(
        fn: Callable,
    ) -> Callable[[list[str]], list[list[float]]]:
        """Wrap an async embedding function so it can be called synchronously."""

        def _sync_wrapper(texts: list[str]) -> list[list[float]]:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(fn(texts))
            # Already inside an event loop -- run in a fresh thread to avoid deadlock.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, fn(texts)).result()

        return _sync_wrapper

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]] | None:
        """Generate embeddings using the custom embedding function if provided."""
        if self.embedding_fn is not None:
            return self.embedding_fn(texts)
        return None
