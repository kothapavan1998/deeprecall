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
        self._batch_lock = asyncio.Lock()

    async def query(
        self,
        query: str,
        root_prompt: str | None = None,
        top_k: int | None = None,
        budget: QueryBudget | None = None,
        filters: dict[str, Any] | None = None,
        context_prefix: str | None = None,
    ) -> DeepRecallResult:
        """Execute a recursive reasoning query asynchronously.

        Args:
            query: The question or task to answer.
            root_prompt: Optional short prompt visible to the root LM.
            top_k: Override the default top_k for this query.
            budget: Per-query budget override.
            filters: Optional metadata filters applied to every search_db call.
            context_prefix: Optional text prepended to the context (e.g., section info).
        """
        return await asyncio.to_thread(
            self._engine.query,
            query,
            root_prompt,
            top_k,
            budget,
            filters,
            context_prefix,
        )

    async def query_batch(
        self,
        queries: list[str],
        *,
        max_concurrency: int = 4,
        root_prompt: str | None = None,
        top_k: int | None = None,
        budget: QueryBudget | None = None,
    ) -> list[DeepRecallResult]:
        """Execute multiple queries concurrently using asyncio.

        Returns a list of DeepRecallResult in the same order as *queries*.

        Raises:
            ValueError: If the batch exceeds 10 000 queries or max_concurrency < 1.
        """
        from deeprecall.core.types import UsageInfo

        if len(queries) > 10_000:
            raise ValueError(
                f"Batch size {len(queries)} exceeds maximum of 10,000. Split into smaller batches."
            )
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")

        # Hold an asyncio.Lock for the full batch lifetime to prevent
        # concurrent batches from corrupting the saved reuse_search_server flag.
        async with self._batch_lock:
            saved_reuse = self._engine.config.reuse_search_server
            self._engine.config.reuse_search_server = False

            semaphore = asyncio.Semaphore(max_concurrency)

            async def _run(q: str) -> DeepRecallResult:
                async with semaphore:
                    return await self.query(q, root_prompt=root_prompt, top_k=top_k, budget=budget)

            try:
                tasks = [_run(q) for q in queries]
                settled = await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                self._engine.config.reuse_search_server = saved_reuse

        results: list[DeepRecallResult] = []
        for i, outcome in enumerate(settled):
            if isinstance(outcome, BaseException):
                results.append(
                    DeepRecallResult(
                        answer="",
                        sources=[],
                        reasoning_trace=[],
                        usage=UsageInfo(),
                        execution_time=0.0,
                        query=queries[i],
                        budget_status={},
                        error=str(outcome),
                        confidence=None,
                    )
                )
            else:
                results.append(outcome)
        return results

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
