"""Tests for the reranking system."""

from __future__ import annotations

from deeprecall.core.reranker import BaseReranker
from deeprecall.core.types import SearchResult


class SimpleReranker(BaseReranker):
    """Test reranker that reverses the score order."""

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5):
        # Just reverse the order and assign new scores
        reversed_results = list(reversed(results))
        reranked = []
        for i, r in enumerate(reversed_results[:top_k]):
            reranked.append(
                SearchResult(
                    content=r.content,
                    metadata=r.metadata,
                    score=1.0 - (i * 0.1),
                    id=r.id,
                )
            )
        return reranked


class TestBaseReranker:
    def test_rerank_basic(self):
        reranker = SimpleReranker()
        results = [
            SearchResult(content="Doc A", score=0.9, id="1"),
            SearchResult(content="Doc B", score=0.8, id="2"),
            SearchResult(content="Doc C", score=0.7, id="3"),
        ]

        reranked = reranker.rerank("test query", results, top_k=2)

        assert len(reranked) == 2
        assert reranked[0].content == "Doc C"  # Reversed order
        assert reranked[0].score == 1.0

    def test_rerank_empty_results(self):
        reranker = SimpleReranker()
        reranked = reranker.rerank("test", [], top_k=5)
        assert reranked == []

    def test_top_k_limits_results(self):
        reranker = SimpleReranker()
        results = [SearchResult(content=f"Doc {i}", score=0.5, id=str(i)) for i in range(10)]

        reranked = reranker.rerank("test", results, top_k=3)
        assert len(reranked) == 3
