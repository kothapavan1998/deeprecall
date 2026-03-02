"""Tests for AsyncDeepRecallEngine."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from deeprecall.core.async_engine import AsyncDeepRecallEngine
from deeprecall.core.types import DeepRecallResult, UsageInfo


@pytest.fixture
def mock_vectorstore():
    """Create a minimal mock vectorstore for async engine tests."""
    store = MagicMock()
    store.count.return_value = 5
    store.add_documents.return_value = ["id-1", "id-2"]
    return store


class TestAsyncDeepRecallEngine:
    @pytest.mark.asyncio
    async def test_init(self, mock_vectorstore):
        engine = AsyncDeepRecallEngine(vectorstore=mock_vectorstore)
        assert engine._engine is not None
        assert isinstance(engine._batch_lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_repr(self, mock_vectorstore):
        engine = AsyncDeepRecallEngine(vectorstore=mock_vectorstore)
        r = repr(engine)
        assert "AsyncDeepRecall" in r or "DeepRecall" in r

    @pytest.mark.asyncio
    async def test_config_property(self, mock_vectorstore):
        engine = AsyncDeepRecallEngine(vectorstore=mock_vectorstore)
        assert engine.config.backend == "openai"

    @pytest.mark.asyncio
    async def test_vectorstore_property(self, mock_vectorstore):
        engine = AsyncDeepRecallEngine(vectorstore=mock_vectorstore)
        assert engine.vectorstore is mock_vectorstore

    @pytest.mark.asyncio
    async def test_add_documents(self, mock_vectorstore):
        engine = AsyncDeepRecallEngine(vectorstore=mock_vectorstore)
        ids = await engine.add_documents(documents=["doc1", "doc2"])
        assert len(ids) == 2
        mock_vectorstore.add_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, mock_vectorstore):
        engine = AsyncDeepRecallEngine(vectorstore=mock_vectorstore)
        with patch.object(engine._engine, "close") as mock_close:
            await engine.close()
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_vectorstore):
        async with AsyncDeepRecallEngine(vectorstore=mock_vectorstore) as engine:
            assert engine is not None

    @pytest.mark.asyncio
    async def test_query_batch_too_large_raises(self, mock_vectorstore):
        engine = AsyncDeepRecallEngine(vectorstore=mock_vectorstore)
        with pytest.raises(ValueError, match="exceeds maximum"):
            await engine.query_batch(["q"] * 10_001)

    @pytest.mark.asyncio
    async def test_query_batch_zero_concurrency_raises(self, mock_vectorstore):
        engine = AsyncDeepRecallEngine(vectorstore=mock_vectorstore)
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            await engine.query_batch(["q1"], max_concurrency=0)

    @pytest.mark.asyncio
    async def test_query_delegates_to_sync_engine(self, mock_vectorstore):
        """Verify async query() delegates to the sync engine via asyncio.to_thread."""
        engine = AsyncDeepRecallEngine(vectorstore=mock_vectorstore)

        expected_result = DeepRecallResult(
            answer="Mocked async answer",
            usage=UsageInfo(total_input_tokens=10, total_output_tokens=5),
            execution_time=0.5,
            query="test question",
        )

        with patch.object(engine._engine, "query", return_value=expected_result) as mock_query:
            result = await engine.query("test question", top_k=3)

        assert result.answer == "Mocked async answer"
        assert result.query == "test question"
        mock_query.assert_called_once_with("test question", None, 3, None, None, None)

    @pytest.mark.asyncio
    async def test_query_with_budget(self, mock_vectorstore):
        """Verify budget is passed through to sync engine."""
        from deeprecall.core.guardrails import QueryBudget

        engine = AsyncDeepRecallEngine(vectorstore=mock_vectorstore)
        budget = QueryBudget(max_search_calls=5)

        expected = DeepRecallResult(answer="ok", usage=UsageInfo(), execution_time=0.1)

        mock_sync_query = MagicMock(return_value=expected)
        with patch.object(engine._engine, "query", mock_sync_query):
            result = await engine.query("q", budget=budget)

        assert result.answer == "ok"
        mock_sync_query.assert_called_once_with("q", None, None, budget, None, None)

    @pytest.mark.asyncio
    async def test_batch_lock_prevents_concurrent_corruption(self, mock_vectorstore):
        """Verify the asyncio.Lock prevents overlapping batches from corrupting state."""
        engine = AsyncDeepRecallEngine(vectorstore=mock_vectorstore)

        # Track reuse_search_server changes
        saved_values: list[bool] = []

        async def mock_query(*args, **kwargs):
            saved_values.append(engine._engine.config.reuse_search_server)
            return DeepRecallResult(answer="ok", usage=UsageInfo(), execution_time=0.1)

        with patch.object(engine, "query", side_effect=mock_query):
            # Run two batches concurrently -- with the lock, they'll serialize
            await asyncio.gather(
                engine.query_batch(["q1", "q2"]),
                engine.query_batch(["q3", "q4"]),
            )

        # During each batch, reuse_search_server should be False
        assert all(v is False for v in saved_values)
        # After both batches, it should be restored to True
        assert engine._engine.config.reuse_search_server is True
