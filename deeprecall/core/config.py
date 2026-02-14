"""Configuration for DeepRecall engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from deeprecall.core.cache import BaseCache
    from deeprecall.core.callbacks import BaseCallback
    from deeprecall.core.guardrails import QueryBudget
    from deeprecall.core.reranker import BaseReranker

# Supported LLM backends (mirrors RLM's ClientBackend)
BackendType = Literal[
    "openai",
    "portkey",
    "openrouter",
    "vercel",
    "vllm",
    "litellm",
    "anthropic",
    "azure_openai",
    "gemini",
]

# Supported environment types (mirrors RLM's EnvironmentType)
EnvironmentType = Literal["local", "docker", "modal", "prime", "daytona", "e2b"]


@dataclass
class DeepRecallConfig:
    """Configuration for the DeepRecall engine.

    Args:
        backend: LLM provider to use (e.g., "openai", "anthropic").
        backend_kwargs: Provider-specific kwargs (model_name, api_key, etc.).
        environment: REPL sandbox type for code execution.
        environment_kwargs: Environment-specific kwargs.
        max_iterations: Max recursive reasoning iterations before forcing an answer.
        max_depth: Max recursion depth for sub-LM calls.
        top_k: Default number of results to retrieve from vector store.
        verbose: Enable rich console output for debugging.
        log_dir: Directory for JSONL trajectory logs. None disables logging.
        other_backends: Additional LLM backends for sub-calls.
        other_backend_kwargs: Kwargs for additional backends.
        budget: Resource limits for queries (token, time, cost budgets).
        callbacks: List of callback handlers for observability.
        cache: Cache backend for query and search result caching.
        cache_ttl: Time-to-live for cached results in seconds.
        reranker: Post-retrieval reranker for improving search quality.
    """

    backend: BackendType = "openai"
    backend_kwargs: dict[str, Any] = field(default_factory=lambda: {"model_name": "gpt-4o-mini"})
    environment: EnvironmentType = "local"
    environment_kwargs: dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 15
    max_depth: int = 1
    top_k: int = 5
    verbose: bool = False
    log_dir: str | None = None
    other_backends: list[BackendType] | None = None
    other_backend_kwargs: list[dict[str, Any]] | None = None
    budget: QueryBudget | None = None
    callbacks: list[BaseCallback] | None = None
    cache: BaseCache | None = None
    cache_ttl: int = 3600
    reranker: BaseReranker | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "backend_kwargs": {k: v for k, v in self.backend_kwargs.items() if k != "api_key"},
            "environment": self.environment,
            "max_iterations": self.max_iterations,
            "max_depth": self.max_depth,
            "top_k": self.top_k,
            "verbose": self.verbose,
            "budget": self.budget.to_dict() if self.budget else None,
            "cache_ttl": self.cache_ttl,
        }
