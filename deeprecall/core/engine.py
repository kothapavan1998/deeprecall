"""DeepRecall Engine -- the core recursive reasoning engine.

Bridges RLM (Recursive Language Models) with vector databases to enable
recursive, multi-hop retrieval and reasoning.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from deeprecall.core.config import DeepRecallConfig
from deeprecall.core.guardrails import BudgetExceededError, QueryBudget
from deeprecall.core.search_server import SearchServer
from deeprecall.core.tracer import DeepRecallTracer
from deeprecall.core.types import DeepRecallResult, Source, UsageInfo
from deeprecall.prompts.templates import DEEPRECALL_SYSTEM_PROMPT, build_search_setup_code
from deeprecall.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class DeepRecallEngine:
    """Recursive reasoning engine powered by RLM and vector databases.

    Combines RLM's recursive decomposition with vector DB retrieval to
    enable multi-hop reasoning over large document collections.

    Args:
        vectorstore: A vector store adapter instance (ChromaStore, MilvusStore, etc.).
        config: Engine configuration. Uses defaults if not provided.

    Example:
        ```python
        from deeprecall import DeepRecall
        from deeprecall.vectorstores import ChromaStore

        store = ChromaStore(collection_name="my_docs")
        store.add_documents(["Document 1 text...", "Document 2 text..."])

        engine = DeepRecall(vectorstore=store, backend="openai",
                            backend_kwargs={"model_name": "gpt-4o-mini"})
        result = engine.query("What are the main themes?")
        print(result.answer)
        ```
    """

    def __init__(
        self,
        vectorstore: BaseVectorStore,
        config: DeepRecallConfig | None = None,
        *,
        backend: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        verbose: bool = False,
        max_iterations: int | None = None,
        **kwargs: Any,
    ):
        self.vectorstore = vectorstore

        # Allow either config object or individual kwargs
        if config is not None:
            self.config = config
        else:
            config_kwargs: dict[str, Any] = {}
            if backend is not None:
                config_kwargs["backend"] = backend
            if backend_kwargs is not None:
                config_kwargs["backend_kwargs"] = backend_kwargs
            if max_iterations is not None:
                config_kwargs["max_iterations"] = max_iterations
            config_kwargs["verbose"] = verbose
            config_kwargs.update(kwargs)
            self.config = DeepRecallConfig(**config_kwargs)

        self._callback_manager = None
        self._setup_callbacks()
        self._validate_setup()

    def _validate_setup(self) -> None:
        """Validate that the engine is properly configured."""
        if self.vectorstore is None:
            raise ValueError("A vectorstore is required. Pass a BaseVectorStore instance.")

    def _setup_callbacks(self) -> None:
        """Initialize callback manager from config."""
        if self.config.callbacks:
            from deeprecall.core.callbacks import CallbackManager

            self._callback_manager = CallbackManager(self.config.callbacks)

    def query(
        self,
        query: str,
        root_prompt: str | None = None,
        top_k: int | None = None,
        budget: QueryBudget | None = None,
    ) -> DeepRecallResult:
        """Execute a recursive reasoning query over the vector database.

        Args:
            query: The question or task to answer.
            root_prompt: Optional short prompt visible to the root LM.
            top_k: Override the default top_k for this query.
            budget: Per-query budget override. Falls back to config.budget.

        Returns:
            DeepRecallResult with answer, sources, reasoning trace, and usage info.
        """
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        try:
            from rlm import RLM
        except ImportError:
            raise ImportError(
                "rlms is required for DeepRecall. Install it with: pip install rlms"
            ) from None

        time_start = time.perf_counter()
        effective_top_k = top_k or self.config.top_k
        effective_budget = budget or self.config.budget

        # Check query cache
        cache_key: str | None = None
        if self.config.cache:
            cache_key = self._build_cache_key(query, effective_top_k)
            cached = self.config.cache.get(cache_key)
            if cached is not None:
                return cached

        # Fire on_query_start callback
        if self._callback_manager:
            self._callback_manager.on_query_start(query, self.config)

        if self.config.verbose:
            self._print_query_panel(query, effective_budget)

        # Start the search server (with reranker if configured)
        search_server = SearchServer(
            self.vectorstore,
            reranker=self.config.reranker,
            cache=self.config.cache,
        )
        search_server.start()

        # Create tracer for this query
        tracer = DeepRecallTracer(
            budget=effective_budget,
            callback_manager=self._callback_manager,
            start_time=time_start,
        )

        try:
            # Build setup code that injects search_db() into the REPL
            setup_code = build_search_setup_code(
                server_port=search_server.port,
                max_search_calls=(effective_budget.max_search_calls if effective_budget else None),
            )

            # Build environment kwargs
            env_kwargs = {**self.config.environment_kwargs, "setup_code": setup_code}

            # Build context
            context = self._build_context(query, effective_top_k)

            # Create and run the RLM with our tracer as the logger
            rlm = RLM(
                backend=self.config.backend,
                backend_kwargs=self.config.backend_kwargs,
                environment=self.config.environment,
                environment_kwargs=env_kwargs,
                max_iterations=self.config.max_iterations,
                max_depth=self.config.max_depth,
                custom_system_prompt=DEEPRECALL_SYSTEM_PROMPT,
                other_backends=self.config.other_backends,
                other_backend_kwargs=self.config.other_backend_kwargs,
                logger=tracer,
                verbose=self.config.verbose,
            )

            # Run recursive completion
            root = root_prompt or query
            rlm_result = rlm.completion(prompt=context, root_prompt=root)

            # Build final result
            result = self._build_result(
                rlm_result=rlm_result,
                search_server=search_server,
                tracer=tracer,
                query=query,
                time_start=time_start,
            )

            # Store in cache
            if self.config.cache and cache_key and not result.error:
                self.config.cache.set(cache_key, result, ttl=self.config.cache_ttl)

            # Fire on_query_end callback
            if self._callback_manager:
                self._callback_manager.on_query_end(result)

            return result

        except BudgetExceededError as e:
            # Return partial result when budget exceeded
            result = self._build_result(
                rlm_result=None,
                search_server=search_server,
                tracer=tracer,
                query=query,
                time_start=time_start,
                error=str(e),
            )
            if self._callback_manager:
                self._callback_manager.on_budget_warning(e.status)
                self._callback_manager.on_query_end(result)
            return result

        except Exception as e:
            if self._callback_manager:
                self._callback_manager.on_error(e)
            raise

        finally:
            search_server.stop()

    def _print_query_panel(self, query: str, budget: QueryBudget | None) -> None:
        """Print verbose query panel using rich (lazy import)."""
        try:
            from rich.console import Console
            from rich.panel import Panel

            budget_info = ""
            if budget:
                limits = []
                if budget.max_search_calls is not None:
                    limits.append(f"searches\u2264{budget.max_search_calls}")
                if budget.max_tokens is not None:
                    limits.append(f"tokens\u2264{budget.max_tokens}")
                if budget.max_time_seconds is not None:
                    limits.append(f"time\u2264{budget.max_time_seconds}s")
                budget_info = f"\n[bold]Budget:[/bold] {', '.join(limits)}" if limits else ""

            Console().print(
                Panel(
                    f"[bold]Query:[/bold] {query}\n"
                    f"[bold]Vector Store:[/bold] {type(self.vectorstore).__name__} "
                    f"({self.vectorstore.count()} docs)\n"
                    f"[bold]Backend:[/bold] {self.config.backend} / "
                    f"{self.config.backend_kwargs.get('model_name', 'unknown')}"
                    f"{budget_info}",
                    title="[bold blue]DeepRecall[/bold blue]",
                    border_style="blue",
                )
            )
        except Exception:
            pass  # verbose panel is non-critical

    def _build_result(
        self,
        rlm_result: Any | None,
        search_server: SearchServer,
        tracer: DeepRecallTracer,
        query: str,
        time_start: float,
        error: str | None = None,
    ) -> DeepRecallResult:
        """Build a DeepRecallResult from RLM output and tracer data."""
        execution_time = time.perf_counter() - time_start
        sources = search_server.get_accessed_sources()
        usage = self._extract_usage(rlm_result) if rlm_result else UsageInfo()

        # Update tracer budget with token info
        tracer.budget_status.tokens_used = usage.total_input_tokens + usage.total_output_tokens

        # Calculate confidence from source scores
        confidence = self._compute_confidence(sources)

        # Get answer (partial from trace if budget exceeded)
        if rlm_result:
            answer = rlm_result.response
        elif tracer.steps:
            last_output = tracer.steps[-1].output or ""
            answer = f"[Partial - budget exceeded] {last_output[:500]}"
        else:
            answer = "[No answer - budget exceeded before first iteration]"

        return DeepRecallResult(
            answer=answer,
            sources=sources,
            reasoning_trace=tracer.get_trace(),
            usage=usage,
            execution_time=execution_time,
            query=query,
            budget_status=tracer.budget_status.to_dict(),
            error=error,
            confidence=confidence,
        )

    def _compute_confidence(self, sources: list[Source]) -> float | None:
        """Compute a confidence score based on source relevance scores."""
        if not sources:
            return None
        scores = [s.score for s in sources if s.score > 0]
        if not scores:
            return None
        # Average of top-3 source scores, normalized to 0-1
        top_scores = sorted(scores, reverse=True)[:3]
        return round(sum(top_scores) / len(top_scores), 4)

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Convenience method to add documents to the vector store."""
        return self.vectorstore.add_documents(documents=documents, metadatas=metadatas, ids=ids)

    def _build_context(self, query: str, top_k: int) -> str:
        """Build context string for the RLM with query info and DB stats."""
        doc_count = self.vectorstore.count()
        context_parts = [
            f"USER QUERY: {query}",
            "",
            f"VECTOR DATABASE: {type(self.vectorstore).__name__} with {doc_count} documents.",
            f"You have access to search_db(query, top_k={top_k}) to search this database.",
            "",
            "INSTRUCTIONS:",
            "1. Use search_db() to find relevant documents for the query.",
            "2. Analyze retrieved documents and search again if needed.",
            "3. Use llm_query() to reason over large amounts of retrieved text.",
            "4. Provide a comprehensive final answer with FINAL().",
        ]
        return "\n".join(context_parts)

    def _extract_usage(self, rlm_result: Any) -> UsageInfo:
        """Extract token usage from the RLM result."""
        usage = UsageInfo()
        try:
            summary = rlm_result.usage_summary
            if hasattr(summary, "model_usage_summaries"):
                for model_name, model_usage in summary.model_usage_summaries.items():
                    usage.total_input_tokens += model_usage.total_input_tokens or 0
                    usage.total_output_tokens += model_usage.total_output_tokens or 0
                    usage.total_calls += model_usage.total_calls or 0
                    usage.model_breakdown[model_name] = {
                        "input_tokens": model_usage.total_input_tokens or 0,
                        "output_tokens": model_usage.total_output_tokens or 0,
                        "calls": model_usage.total_calls or 0,
                    }
        except Exception:
            logger.debug("Could not extract usage info from RLM result", exc_info=True)
        return usage

    def _build_cache_key(self, query: str, top_k: int) -> str:
        """Build a cache key for a query."""
        import hashlib

        key_data = (
            f"{query}|{self.config.backend}|"
            f"{self.config.backend_kwargs.get('model_name', '')}|"
            f"{top_k}|{self.vectorstore.count()}"
        )
        return hashlib.sha256(key_data.encode()).hexdigest()

    def close(self) -> None:
        """Clean up resources (cache, callbacks)."""
        if self.config.cache:
            logger.debug("Engine closed; cache retained with %s", self.config.cache.stats())

    def __enter__(self) -> DeepRecallEngine:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"DeepRecall(vectorstore={type(self.vectorstore).__name__}, "
            f"backend={self.config.backend!r}, "
            f"docs={self.vectorstore.count()})"
        )
