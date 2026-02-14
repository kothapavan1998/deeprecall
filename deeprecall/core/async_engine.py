"""Async wrapper for the DeepRecall engine.

Provides async versions of query and add_documents methods
using asyncio.to_thread() for non-blocking operation.
"""

from __future__ import annotations

import asyncio
from typing import Any

from deeprecall.core.config import DeepRecallConfig
from deeprecall.core.engine import DeepRecallEngine
from deeprecall.core.guardrails import QueryBudget
from deeprecall.core.types import DeepRecallResult
from deeprecall.vectorstores.base import BaseVectorStore


class AsyncDeepRecallEngine:
    """Async wrapper around DeepRecallEngine.

    All heavy operations run in a thread pool to avoid blocking the event loop.

    Args:
        vectorstore: A vector store adapter instance.
        config: Engine configuration.

    Example:
        ```python
        import asyncio
        from deeprecall import AsyncDeepRecall
        from deeprecall.vectorstores import ChromaStore

        async def main():
            store = ChromaStore(collection_name="my_docs")
            engine = AsyncDeepRecall(vectorstore=store, backend="openai",
                                     backend_kwargs={"model_name": "gpt-4o-mini"})
            result = await engine.query("What are the main themes?")
            print(result.answer)

        asyncio.run(main())
        ```
    """

    def __init__(
        self,
        vectorstore: BaseVectorStore,
        config: DeepRecallConfig | None = None,
        **kwargs: Any,
    ):
        self._engine = DeepRecallEngine(vectorstore=vectorstore, config=config, **kwargs)

    async def query(
        self,
        query: str,
        root_prompt: str | None = None,
        top_k: int | None = None,
        budget: QueryBudget | None = None,
    ) -> DeepRecallResult:
        """Execute a recursive reasoning query asynchronously.

        Args:
            query: The question or task to answer.
            root_prompt: Optional short prompt visible to the root LM.
            top_k: Override the default top_k for this query.
            budget: Per-query budget override.

        Returns:
            DeepRecallResult with answer, sources, reasoning trace, and usage info.
        """
        return await asyncio.to_thread(
            self._engine.query,
            query,
            root_prompt,
            top_k,
            budget,
        )

    async def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add documents to the vector store asynchronously."""
        return await asyncio.to_thread(
            self._engine.add_documents,
            documents,
            metadatas,
            ids,
        )

    @property
    def config(self) -> DeepRecallConfig:
        return self._engine.config

    @property
    def vectorstore(self) -> BaseVectorStore:
        return self._engine.vectorstore

    async def close(self) -> None:
        """Clean up resources."""
        await asyncio.to_thread(self._engine.close)

    async def __aenter__(self) -> AsyncDeepRecallEngine:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        await self.close()
        return False

    def __repr__(self) -> str:
        return f"Async{repr(self._engine)}"
