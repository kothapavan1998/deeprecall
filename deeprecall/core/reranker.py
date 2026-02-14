"""Post-retrieval reranking for improving search quality.

Rerankers score query-document pairs more accurately than embedding similarity,
improving what the LLM reasons over.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from deeprecall.core.types import SearchResult

_logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank search results by relevance to the query.

        Args:
            query: The search query.
            results: Initial results from the vector store.
            top_k: Number of top results to return after reranking.

        Returns:
            Reranked list of SearchResult, trimmed to top_k.
        """


class CohereReranker(BaseReranker):
    """Reranker using the Cohere Rerank API.

    Args:
        api_key: Cohere API key. Falls back to COHERE_API_KEY env var.
        model: Rerank model name (default: rerank-v3.5).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "rerank-v3.5",
    ):
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere is required for CohereReranker. "
                "Install with: pip install deeprecall[rerank-cohere]"
            ) from None

        import os

        resolved_key = api_key or os.environ.get("COHERE_API_KEY")
        if not resolved_key:
            raise ValueError("Cohere API key required. Pass api_key or set COHERE_API_KEY.")

        self.client = cohere.Client(resolved_key)
        self.model = model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []

        try:
            documents = [r.content for r in results]
            response = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_n=top_k,
            )

            reranked: list[SearchResult] = []
            for item in response.results:
                original = results[item.index]
                reranked.append(
                    SearchResult(
                        content=original.content,
                        metadata=original.metadata,
                        score=item.relevance_score,
                        id=original.id,
                    )
                )
            return reranked
        except Exception:
            _logger.warning("Cohere reranker failed, returning original results", exc_info=True)
            return results[:top_k]


class CrossEncoderReranker(BaseReranker):
    """Reranker using a sentence-transformers cross-encoder model.

    Runs locally, no API calls needed.

    Args:
        model_name: HuggingFace model name for the cross-encoder.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install with: pip install deeprecall[rerank-cross-encoder]"
            ) from None

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []

        try:
            pairs = [(query, r.content) for r in results]
            scores = self.model.predict(pairs)

            scored = list(zip(results, scores, strict=False))
            scored.sort(key=lambda x: x[1], reverse=True)

            reranked: list[SearchResult] = []
            for result, score in scored[:top_k]:
                reranked.append(
                    SearchResult(
                        content=result.content,
                        metadata=result.metadata,
                        score=float(score),
                        id=result.id,
                    )
                )
            return reranked
        except Exception:
            _logger.warning("CrossEncoder reranker failed, returning original results", exc_info=True)
            return results[:top_k]
