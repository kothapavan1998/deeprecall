"""Core module for DeepRecall."""

from deeprecall.core.async_engine import AsyncDeepRecallEngine
from deeprecall.core.cache import BaseCache, DiskCache, InMemoryCache
from deeprecall.core.cache_redis import RedisCache
from deeprecall.core.callback_otel import OpenTelemetryCallback
from deeprecall.core.callbacks import (
    BaseCallback,
    CallbackManager,
    ConsoleCallback,
    JSONLCallback,
    UsageTrackingCallback,
)
from deeprecall.core.config import DeepRecallConfig
from deeprecall.core.engine import DeepRecallEngine
from deeprecall.core.guardrails import BudgetExceededError, BudgetStatus, QueryBudget
from deeprecall.core.reranker import BaseReranker
from deeprecall.core.tracer import DeepRecallTracer
from deeprecall.core.types import DeepRecallResult, ReasoningStep, SearchResult, Source, UsageInfo

__all__ = [
    "AsyncDeepRecallEngine",
    "BaseCache",
    "BaseCallback",
    "BaseReranker",
    "BudgetExceededError",
    "BudgetStatus",
    "CallbackManager",
    "ConsoleCallback",
    "DeepRecallConfig",
    "DeepRecallEngine",
    "DeepRecallResult",
    "DeepRecallTracer",
    "DiskCache",
    "InMemoryCache",
    "JSONLCallback",
    "OpenTelemetryCallback",
    "QueryBudget",
    "ReasoningStep",
    "RedisCache",
    "SearchResult",
    "Source",
    "UsageInfo",
    "UsageTrackingCallback",
]
