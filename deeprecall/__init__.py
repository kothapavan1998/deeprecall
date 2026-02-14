"""DeepRecall -- Recursive reasoning engine for AI agents and vector databases.

Usage:
    from deeprecall import DeepRecall
    from deeprecall.vectorstores import ChromaStore

    store = ChromaStore(collection_name="my_docs")
    store.add_documents(["Document text..."])

    engine = DeepRecall(vectorstore=store, backend="openai",
                        backend_kwargs={"model_name": "gpt-4o-mini"})
    result = engine.query("Your question here")
    print(result.answer)
"""

from deeprecall.core.async_engine import AsyncDeepRecallEngine
from deeprecall.core.cache import DiskCache, InMemoryCache
from deeprecall.core.cache_redis import RedisCache
from deeprecall.core.callback_otel import OpenTelemetryCallback
from deeprecall.core.callbacks import (
    BaseCallback,
    ConsoleCallback,
    JSONLCallback,
    UsageTrackingCallback,
)
from deeprecall.core.config import DeepRecallConfig
from deeprecall.core.engine import DeepRecallEngine
from deeprecall.core.guardrails import QueryBudget
from deeprecall.core.types import DeepRecallResult, SearchResult, Source

# Convenience aliases
DeepRecall = DeepRecallEngine
AsyncDeepRecall = AsyncDeepRecallEngine

__version__ = "0.2.0"
__all__ = [
    "AsyncDeepRecall",
    "AsyncDeepRecallEngine",
    "BaseCallback",
    "ConsoleCallback",
    "DeepRecall",
    "DeepRecallConfig",
    "DeepRecallEngine",
    "DeepRecallResult",
    "DiskCache",
    "InMemoryCache",
    "JSONLCallback",
    "OpenTelemetryCallback",
    "QueryBudget",
    "RedisCache",
    "SearchResult",
    "Source",
    "UsageTrackingCallback",
]
