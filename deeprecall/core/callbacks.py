"""Callback system for DeepRecall observability.

Provides hooks into the reasoning pipeline for monitoring, logging,
and custom integrations.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from deeprecall.core.config import DeepRecallConfig
    from deeprecall.core.guardrails import BudgetStatus
    from deeprecall.core.types import DeepRecallResult, ReasoningStep


class BaseCallback:  # noqa: B024
    """Base class for DeepRecall callbacks.

    Implement any subset of hooks -- unimplemented methods are no-ops.
    Not using @abstractmethod intentionally: all hooks are optional.
    """

    def on_query_start(self, query: str, config: DeepRecallConfig) -> None:  # noqa: B027
        """Called when a query begins."""

    def on_reasoning_step(self, step: ReasoningStep, budget_status: BudgetStatus) -> None:  # noqa: B027
        """Called after each reasoning iteration."""

    def on_search(self, query: str, num_results: int, time_ms: float) -> None:  # noqa: B027
        """Called after each vector store search."""

    def on_query_end(self, result: DeepRecallResult) -> None:  # noqa: B027
        """Called when a query completes (success or partial)."""

    def on_error(self, error: Exception) -> None:  # noqa: B027
        """Called when an unrecoverable error occurs."""

    def on_budget_warning(self, status: BudgetStatus) -> None:  # noqa: B027
        """Called when a budget limit is exceeded."""


class CallbackManager:
    """Manages and dispatches events to multiple callbacks."""

    def __init__(self, callbacks: list[BaseCallback] | None = None):
        self.callbacks = callbacks or []

    def add(self, callback: BaseCallback) -> None:
        self.callbacks.append(callback)

    def _safe_call(self, cb: BaseCallback, method: str, *args: Any, **kwargs: Any) -> None:
        try:
            getattr(cb, method)(*args, **kwargs)
        except Exception:
            _logger.warning(
                "Callback %s.%s failed",
                type(cb).__name__,
                method,
                exc_info=True,
            )

    def on_query_start(self, query: str, config: DeepRecallConfig) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_query_start", query, config)

    def on_reasoning_step(self, step: ReasoningStep, budget_status: BudgetStatus) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_reasoning_step", step, budget_status)

    def on_search(self, query: str, num_results: int, time_ms: float) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_search", query, num_results, time_ms)

    def on_query_end(self, result: DeepRecallResult) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_query_end", result)

    def on_error(self, error: Exception) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_error", error)

    def on_budget_warning(self, status: BudgetStatus) -> None:
        for cb in self.callbacks:
            self._safe_call(cb, "on_budget_warning", status)


# ---------------------------------------------------------------------------
# Built-in callback implementations
# ---------------------------------------------------------------------------


class ConsoleCallback(BaseCallback):
    """Rich console output showing reasoning steps in real time."""

    def __init__(self, show_code: bool = True, show_output: bool = True):
        self.show_code = show_code
        self.show_output = show_output
        self._start_time: float | None = None

    def on_query_start(self, query: str, config: DeepRecallConfig) -> None:
        from rich.console import Console
        from rich.panel import Panel

        self._start_time = time.perf_counter()
        console = Console()
        console.print(Panel(f"[bold]{query}[/bold]", title="Query", border_style="cyan"))

    def on_reasoning_step(self, step: ReasoningStep, budget_status: BudgetStatus) -> None:
        from rich.console import Console

        console = Console()
        elapsed = time.perf_counter() - (self._start_time or time.perf_counter())

        header = (
            f"[bold]Step {step.iteration}[/bold] | "
            f"{step.action} | "
            f"searches={budget_status.search_calls_used} | "
            f"{elapsed:.1f}s"
        )
        console.print(f"\n{'─' * 60}")
        console.print(header)

        if self.show_code and step.code:
            console.print(f"[dim]{step.code[:300]}{'...' if len(step.code) > 300 else ''}[/dim]")
        if self.show_output and step.output:
            console.print(
                f"[green]{step.output[:200]}{'...' if len(step.output) > 200 else ''}[/green]"
            )

    def on_query_end(self, result: DeepRecallResult) -> None:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        summary = (
            f"Time: {result.execution_time:.2f}s | "
            f"Sources: {len(result.sources)} | "
            f"Steps: {len(result.reasoning_trace)} | "
            f"Tokens: {result.usage.total_input_tokens + result.usage.total_output_tokens}"
        )
        if result.confidence is not None:
            summary += f" | Confidence: {result.confidence:.2f}"
        if result.error:
            summary += f"\n[red]Error: {result.error}[/red]"
        console.print(Panel(summary, title="Complete", border_style="green"))

    def on_budget_warning(self, status: BudgetStatus) -> None:
        from rich.console import Console

        Console().print(f"[bold red]Budget exceeded: {status.exceeded_reason}[/bold red]")


class JSONLCallback(BaseCallback):
    """Logs all events to a JSONL file for post-hoc analysis."""

    def __init__(self, log_dir: str, filename_prefix: str = "deeprecall"):
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{filename_prefix}_{ts}.jsonl")
        self._lock = threading.Lock()

    def _write(self, event_type: str, data: dict[str, Any]) -> None:
        entry = {"type": event_type, "timestamp": time.time(), **data}
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                json.dump(entry, f, default=str)
                f.write("\n")

    def on_query_start(self, query: str, config: DeepRecallConfig) -> None:
        self._write("query_start", {"query": query, "config": config.to_dict()})

    def on_reasoning_step(self, step: ReasoningStep, budget_status: BudgetStatus) -> None:
        self._write("reasoning_step", {"step": step.to_dict(), "budget": budget_status.to_dict()})

    def on_query_end(self, result: DeepRecallResult) -> None:
        self._write(
            "query_end",
            {
                "answer_length": len(result.answer),
                "sources": len(result.sources),
                "steps": len(result.reasoning_trace),
                "execution_time": result.execution_time,
                "error": result.error,
            },
        )

    def on_error(self, error: Exception) -> None:
        self._write("error", {"error": str(error), "type": type(error).__name__})

    def on_budget_warning(self, status: BudgetStatus) -> None:
        self._write("budget_warning", {"status": status.to_dict()})


class UsageTrackingCallback(BaseCallback):
    """Tracks cumulative usage across multiple queries for billing.

    Thread-safe: all counter mutations and reads are protected by a lock.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_queries: int = 0
        self.total_tokens: int = 0
        self.total_searches: int = 0
        self.total_time: float = 0.0
        self.errors: int = 0

    def on_query_end(self, result: DeepRecallResult) -> None:
        with self._lock:
            self.total_queries += 1
            self.total_tokens += result.usage.total_input_tokens + result.usage.total_output_tokens
            self.total_time += result.execution_time
            if result.budget_status:
                self.total_searches += result.budget_status.get("search_calls_used", 0)
            if result.error:
                self.errors += 1

    def on_error(self, error: Exception) -> None:
        with self._lock:
            self.errors += 1

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_queries": self.total_queries,
                "total_tokens": self.total_tokens,
                "total_searches": self.total_searches,
                "total_time": round(self.total_time, 2),
                "errors": self.errors,
            }
