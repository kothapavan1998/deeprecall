"""DeepRecall Engine -- core recursive reasoning engine."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any

from deeprecall.core.config import DeepRecallConfig
from deeprecall.core.exceptions import (
    BudgetExceededError,
    LLMProviderError,
    SearchServerError,
)
from deeprecall.core.guardrails import QueryBudget
from deeprecall.core.search_server import SearchServer
from deeprecall.core.tracer import DeepRecallTracer
from deeprecall.core.types import DeepRecallResult, Source, UsageInfo
from deeprecall.prompts.templates import DEEPRECALL_SYSTEM_PROMPT, build_search_setup_code
from deeprecall.vectorstores.base import BaseVectorStore

logger = logging.getLogger(__name__)

try:
    from rlm.utils.exceptions import (
        BudgetExceededError as RLMBudgetExceededError,
    )
    from rlm.utils.exceptions import (
        CancellationError as RLMCancellationError,
    )
    from rlm.utils.exceptions import (
        ErrorThresholdExceededError,
        TimeoutExceededError,
        TokenLimitExceededError,
    )

    _RLM_LIMIT_ERRORS: tuple[type[Exception], ...] = (
        RLMBudgetExceededError,
        TimeoutExceededError,
        TokenLimitExceededError,
        ErrorThresholdExceededError,
        RLMCancellationError,
    )
except ImportError:
    _RLM_LIMIT_ERRORS = ()


class DeepRecallEngine:
    """Recursive reasoning engine powered by RLM and vector databases."""

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
        if config is not None:
            self.config = config
        else:
            config_kwargs: dict[str, Any] = {"verbose": verbose}
            if backend is not None:
                config_kwargs["backend"] = backend
            if backend_kwargs is not None:
                config_kwargs["backend_kwargs"] = backend_kwargs
            if max_iterations is not None:
                config_kwargs["max_iterations"] = max_iterations
            config_kwargs.update(kwargs)
            self.config = DeepRecallConfig(**config_kwargs)

        self._callback_manager: Any = None
        self._search_server: SearchServer | None = None
        self._batch_lock = threading.Lock()
        self._setup_callbacks()
        if self.vectorstore is None:
            raise ValueError("A vectorstore is required. Pass a BaseVectorStore instance.")

    def _setup_callbacks(self) -> None:
        """Initialize callback manager; auto-creates JSONLCallback for log_dir."""
        callbacks = list(self.config.callbacks) if self.config.callbacks else []
        if self.config.log_dir:
            from deeprecall.core.callbacks import JSONLCallback

            if not any(isinstance(c, JSONLCallback) for c in callbacks):
                callbacks.append(JSONLCallback(log_dir=self.config.log_dir))
        if callbacks:
            from deeprecall.core.callbacks import CallbackManager

            self._callback_manager = CallbackManager(callbacks)

    def query(
        self,
        query: str,
        root_prompt: str | None = None,
        top_k: int | None = None,
        budget: QueryBudget | None = None,
        filters: dict[str, Any] | None = None,
        context_prefix: str | None = None,
    ) -> DeepRecallResult:
        """Execute a recursive reasoning query over the vector database."""
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        try:
            from rlm import RLM
        except ImportError:
            raise ImportError(
                "rlms is required for DeepRecall. Install it with: pip install rlms"
            ) from None

        time_start = time.perf_counter()
        effective_top_k = top_k if top_k is not None else self.config.top_k
        effective_budget = budget if budget is not None else self.config.budget

        cache_key: str | None = None
        if self.config.cache:
            _kd = f"{query}|{self.config.backend}|{self.config.backend_kwargs.get('model_name', '')}|{effective_top_k}"
            cache_key = hashlib.sha256(_kd.encode()).hexdigest()
            cached = self.config.cache.get(cache_key)
            if cached is not None:
                if isinstance(cached, DeepRecallResult):
                    return cached
                if isinstance(cached, dict):
                    return DeepRecallResult.from_dict(cached)

        if self._callback_manager:
            self._callback_manager.on_query_start(query, self.config)
        if self.config.verbose:
            self._print_query_panel(query, effective_budget)

        owns_server = False
        if self.config.reuse_search_server and self._search_server is not None:
            search_server = self._search_server
            search_server.reset_sources()
        else:
            search_server = SearchServer(
                self.vectorstore,
                reranker=self.config.reranker,
                cache=self.config.cache,
                callback_manager=self._callback_manager,
            )
            try:
                search_server.start()
            except SearchServerError:
                raise
            except Exception as e:
                raise SearchServerError(f"Failed to start search server: {e}") from e

            if self.config.reuse_search_server:
                self._search_server = search_server
            else:
                owns_server = True

        tracer = DeepRecallTracer(
            budget=effective_budget,
            callback_manager=self._callback_manager,
            start_time=time_start,
        )

        try:
            setup_code = build_search_setup_code(
                server_port=search_server.port,
                max_search_calls=(effective_budget.max_search_calls if effective_budget else None),
                default_filters=filters,
            )

            env_kwargs = {**self.config.environment_kwargs, "setup_code": setup_code}
            context = self._build_context(query, effective_top_k, context_prefix)
            rlm_kwargs: dict[str, Any] = {}
            if effective_budget and effective_budget.max_cost_usd is not None:
                rlm_kwargs["max_budget"] = effective_budget.max_cost_usd
            if self.config.max_timeout is not None:
                rlm_kwargs["max_timeout"] = self.config.max_timeout
            if self.config.max_tokens is not None:
                rlm_kwargs["max_tokens"] = self.config.max_tokens
            if self.config.max_errors is not None:
                rlm_kwargs["max_errors"] = self.config.max_errors
            if self.config.compaction:
                rlm_kwargs["compaction"] = True
                rlm_kwargs["compaction_threshold_pct"] = self.config.compaction_threshold_pct

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
                **rlm_kwargs,
            )

            root = root_prompt or query

            def _run_completion() -> Any:
                try:
                    return rlm.completion(prompt=context, root_prompt=root)
                except BudgetExceededError:
                    raise
                except LLMProviderError:
                    raise
                except _RLM_LIMIT_ERRORS:
                    raise
                except Exception as e:
                    raise LLMProviderError(f"LLM completion failed: {e}") from e

            if self.config.retry is not None:
                from deeprecall.core.retry import retry_with_backoff

                rlm_result = retry_with_backoff(_run_completion, self.config.retry)
            else:
                rlm_result = _run_completion()

            result = self._build_result(
                rlm_result=rlm_result,
                search_server=search_server,
                tracer=tracer,
                query=query,
                time_start=time_start,
            )

            if self.config.cache and cache_key and not result.error:
                self.config.cache.set(cache_key, result, ttl=self.config.cache_ttl)
            if self._callback_manager:
                self._callback_manager.on_query_end(result)

            return result

        except BudgetExceededError as e:
            partial = self._synthesize_partial(query, search_server.get_accessed_sources())
            result = self._build_result(
                rlm_result=None,
                search_server=search_server,
                tracer=tracer,
                query=query,
                time_start=time_start,
                error=str(e),
                rlm_partial_answer=partial,
            )
            if self._callback_manager:
                self._callback_manager.on_budget_warning(e.status)
                self._callback_manager.on_query_end(result)
            return result

        except _RLM_LIMIT_ERRORS as e:
            pa = getattr(e, "partial_answer", None) or ""
            if not pa:
                pa = self._synthesize_partial(query, search_server.get_accessed_sources())
            result = self._build_result(
                rlm_result=None,
                search_server=search_server,
                tracer=tracer,
                query=query,
                time_start=time_start,
                error=str(e),
                rlm_partial_answer=pa,
            )
            if self._callback_manager:
                self._callback_manager.on_error(e)
                self._callback_manager.on_query_end(result)
            return result

        except Exception as e:
            if self._callback_manager:
                self._callback_manager.on_error(e)
            raise

        finally:
            if owns_server:
                search_server.stop()

    def _print_query_panel(self, query: str, budget: QueryBudget | None) -> None:
        """Print verbose query panel using rich (best-effort, never raises)."""
        try:
            from rich.console import Console
            from rich.panel import Panel

            limits: list[str] = []
            if budget:
                for attr, lbl in [("max_search_calls", "searches"), ("max_tokens", "tokens")]:
                    if (v := getattr(budget, attr, None)) is not None:
                        limits.append(f"{lbl}\u2264{v}")
                if budget.max_time_seconds is not None:
                    limits.append(f"time\u2264{budget.max_time_seconds}s")
            bi = f"\n[bold]Budget:[/bold] {', '.join(limits)}" if limits else ""
            vs = type(self.vectorstore).__name__
            dc = self.vectorstore.count() if hasattr(self.vectorstore, "count") else "?"
            m = self.config.backend_kwargs.get("model_name", "unknown")
            Console().print(
                Panel(
                    f"[bold]Query:[/bold] {query}\n[bold]Store:[/bold] {vs} ({dc} docs)\n"
                    f"[bold]Backend:[/bold] {self.config.backend} / {m}{bi}",
                    title="[bold blue]DeepRecall[/bold blue]",
                    border_style="blue",
                )
            )
        except Exception:
            pass

    def _synthesize_partial(self, query: str, sources: list[Source]) -> str:
        """Best-effort one-shot synthesis from retrieved sources on budget exhaustion."""
        if not sources:
            return ""
        evidence = "\n---\n".join(s.content[:500] for s in sources[:5])
        prompt = (
            f"Based on these retrieved documents, provide a concise best-effort answer "
            f"to: {query}\n\nDocuments:\n{evidence}"
        )
        try:
            from rlm import RLM

            synth = RLM(
                backend=self.config.backend,
                backend_kwargs=self.config.backend_kwargs,
                max_iterations=1,
            )
            result = synth.completion(prompt=prompt)
            return result.response
        except Exception:
            logger.debug("Partial synthesis failed; falling back to source summary", exc_info=True)
            return f"Retrieved {len(sources)} relevant sources but could not synthesize an answer."

    def _build_result(
        self,
        rlm_result: Any | None,
        search_server: SearchServer,
        tracer: DeepRecallTracer,
        query: str,
        time_start: float,
        error: str | None = None,
        rlm_partial_answer: str | None = None,
    ) -> DeepRecallResult:
        """Build a DeepRecallResult from RLM output and tracer data."""
        execution_time = time.perf_counter() - time_start
        sources = search_server.get_accessed_sources()
        usage = self._extract_usage(rlm_result) if rlm_result else UsageInfo()

        tracer.budget_status.tokens_used = usage.total_input_tokens + usage.total_output_tokens
        if usage.total_cost_usd is not None:
            tracer.budget_status.cost_usd = usage.total_cost_usd

        confidence = self._compute_confidence(sources)

        if rlm_result:
            answer = rlm_result.response
        elif rlm_partial_answer:
            answer = f"[Partial] {rlm_partial_answer}"
        else:
            answer = "[No answer - execution limit reached]"

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
        scores = [s.score for s in sources if s.score > 0]
        if not scores:
            return None
        top = sorted(scores, reverse=True)[:3]
        return round(min(max(sum(top) / len(top), 0.0), 1.0), 4)

    @staticmethod
    def _error_result(q: str, error: str) -> DeepRecallResult:
        return DeepRecallResult(
            answer="",
            sources=[],
            reasoning_trace=[],
            usage=UsageInfo(),
            execution_time=0.0,
            query=q,
            budget_status={},
            error=error,
            confidence=None,
        )

    def query_batch(
        self,
        queries: list[str],
        *,
        max_concurrency: int = 4,
        root_prompt: str | None = None,
        top_k: int | None = None,
        budget: QueryBudget | None = None,
    ) -> list[DeepRecallResult]:
        """Execute multiple queries concurrently. Returns results in input order."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if len(queries) > 10_000:
            raise ValueError(f"Batch size {len(queries)} exceeds 10,000.")
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        results: list[DeepRecallResult | None] = [None] * len(queries)

        def _run(idx: int, q: str) -> tuple[int, DeepRecallResult]:
            return idx, self.query(q, root_prompt=root_prompt, top_k=top_k, budget=budget)

        with self._batch_lock:
            saved = self.config.reuse_search_server
            self.config.reuse_search_server = False
            try:
                with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
                    futs = {pool.submit(_run, i, q): i for i, q in enumerate(queries)}
                    for fut in as_completed(futs):
                        try:
                            idx, res = fut.result()
                            results[idx] = res
                        except Exception as e:
                            results[futs[fut]] = self._error_result(queries[futs[fut]], str(e))
            finally:
                self.config.reuse_search_server = saved
        for i, r in enumerate(results):
            if r is None:
                results[i] = self._error_result(queries[i], "Query failed: no result produced")
        return results  # type: ignore[return-value]

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Convenience method to add documents to the vector store."""
        return self.vectorstore.add_documents(documents=documents, metadatas=metadatas, ids=ids)

    def _build_context(self, query: str, top_k: int, prefix: str | None = None) -> str:
        vs = type(self.vectorstore).__name__
        ctx = (
            f"USER QUERY: {query}\n\n"
            f"VECTOR DATABASE: {vs} with {self.vectorstore.count()} documents.\n"
            f"You have access to search_db(query, top_k={top_k}) to search this database.\n\n"
            "INSTRUCTIONS:\n"
            "1. Use search_db() to find relevant documents for the query.\n"
            "2. Analyze retrieved documents and search again if needed.\n"
            "3. Use llm_query() to reason over large amounts of retrieved text.\n"
            "4. Provide a comprehensive final answer with FINAL()."
        )
        return f"{prefix}\n\n{ctx}" if prefix else ctx

    def _extract_usage(self, rlm_result: Any) -> UsageInfo:
        usage = UsageInfo()
        try:
            summaries = getattr(rlm_result.usage_summary, "model_usage_summaries", None)
            if not summaries:
                return usage
            for name, mu in summaries.items():
                usage.total_input_tokens += mu.total_input_tokens or 0
                usage.total_output_tokens += mu.total_output_tokens or 0
                usage.total_calls += mu.total_calls or 0
                bd: dict[str, int | float | None] = {
                    "input_tokens": mu.total_input_tokens or 0,
                    "output_tokens": mu.total_output_tokens or 0,
                    "calls": mu.total_calls or 0,
                }
                if (c := getattr(mu, "total_cost", None)) is not None:
                    bd["cost_usd"] = c
                usage.model_breakdown[name] = bd
            costs = [v["cost_usd"] for v in usage.model_breakdown.values() if v.get("cost_usd")]
            usage.total_cost_usd = sum(costs) if costs else None  # type: ignore[arg-type]
        except Exception:
            logger.debug("Could not extract usage from RLM result", exc_info=True)
        return usage

    def close(self) -> None:
        """Clean up resources (search server, cache, callbacks)."""
        if self._search_server is not None:
            self._search_server.stop()
            self._search_server = None
        if self.config.cache:
            logger.debug("Engine closed; cache retained with %s", self.config.cache.stats())

    def __enter__(self) -> DeepRecallEngine:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"DeepRecall(vectorstore={type(self.vectorstore).__name__}, "
            f"backend={self.config.backend!r})"
        )
