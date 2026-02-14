"""Core type definitions for DeepRecall."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """A single result from a vector store search."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "id": self.id,
        }


@dataclass
class Source:
    """A source document referenced in the final answer."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "id": self.id,
        }

    @classmethod
    def from_search_result(cls, result: SearchResult) -> Source:
        return cls(
            content=result.content,
            metadata=result.metadata,
            score=result.score,
            id=result.id,
        )


@dataclass
class ReasoningStep:
    """A single step in the recursive reasoning trace."""

    iteration: int
    action: str
    code: str | None = None
    output: str | None = None
    searches: list[dict[str, Any]] = field(default_factory=list)
    sub_llm_calls: int = 0
    iteration_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "action": self.action,
            "code": self.code,
            "output": self.output,
            "searches": self.searches,
            "sub_llm_calls": self.sub_llm_calls,
            "iteration_time": self.iteration_time,
        }


@dataclass
class UsageInfo:
    """Token usage information."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    model_breakdown: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_calls": self.total_calls,
            "model_breakdown": self.model_breakdown,
        }


@dataclass
class DeepRecallResult:
    """The result of a DeepRecall query."""

    answer: str
    sources: list[Source] = field(default_factory=list)
    reasoning_trace: list[ReasoningStep] = field(default_factory=list)
    usage: UsageInfo = field(default_factory=UsageInfo)
    execution_time: float = 0.0
    query: str = ""
    budget_status: dict[str, Any] | None = None
    error: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "reasoning_trace": [r.to_dict() for r in self.reasoning_trace],
            "usage": self.usage.to_dict(),
            "execution_time": self.execution_time,
            "query": self.query,
            "budget_status": self.budget_status,
            "error": self.error,
            "confidence": self.confidence,
        }
