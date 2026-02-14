"""Budget controls and guardrails for DeepRecall queries.

Provides cost, time, and resource limits to prevent runaway LLM spending.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


class BudgetExceededError(Exception):
    """Raised when a query exceeds its allocated budget."""

    def __init__(self, reason: str, status: BudgetStatus):
        self.reason = reason
        self.status = status
        super().__init__(f"Budget exceeded: {reason}")


@dataclass
class QueryBudget:
    """Resource limits for a single DeepRecall query.

    Set any field to None to disable that limit.

    Args:
        max_iterations: Max reasoning iterations (RLM loop steps).
        max_search_calls: Max vector DB search calls allowed.
        max_tokens: Total token budget (input + output combined).
        max_time_seconds: Wall-clock time limit for the query.
        max_cost_usd: Dollar cost limit (reserved for future use -- requires pricing info).
    """

    max_iterations: int | None = None
    max_search_calls: int | None = None
    max_tokens: int | None = None
    max_time_seconds: float | None = None
    max_cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "max_search_calls": self.max_search_calls,
            "max_tokens": self.max_tokens,
            "max_time_seconds": self.max_time_seconds,
            "max_cost_usd": self.max_cost_usd,
        }


@dataclass
class BudgetStatus:
    """Live tracking of resource usage against a budget.

    Updated after each reasoning iteration by the tracer.
    """

    iterations_used: int = 0
    search_calls_used: int = 0
    tokens_used: int = 0
    time_elapsed: float = 0.0
    cost_usd: float = 0.0
    budget_exceeded: bool = False
    exceeded_reason: str | None = None
    budget: QueryBudget = field(default_factory=QueryBudget)

    def check(self, start_time: float) -> None:
        """Check all budget limits and raise if any are exceeded.

        Args:
            start_time: The time.perf_counter() value when the query started.

        Raises:
            BudgetExceededError: If any limit is exceeded.
        """
        self.time_elapsed = time.perf_counter() - start_time
        budget = self.budget

        if budget.max_iterations is not None and self.iterations_used > budget.max_iterations:
            self._exceed(f"Iterations: {self.iterations_used}/{budget.max_iterations}")

        if budget.max_search_calls is not None and self.search_calls_used > budget.max_search_calls:
            self._exceed(f"Search calls: {self.search_calls_used}/{budget.max_search_calls}")

        if budget.max_tokens is not None and self.tokens_used > budget.max_tokens:
            self._exceed(f"Tokens: {self.tokens_used}/{budget.max_tokens}")

        if budget.max_time_seconds is not None and self.time_elapsed > budget.max_time_seconds:
            self._exceed(f"Time: {self.time_elapsed:.1f}s/{budget.max_time_seconds}s")

        if budget.max_cost_usd is not None and self.cost_usd > budget.max_cost_usd:
            self._exceed(f"Cost: ${self.cost_usd:.4f}/${budget.max_cost_usd}")

    def _exceed(self, reason: str) -> None:
        self.budget_exceeded = True
        self.exceeded_reason = reason
        raise BudgetExceededError(reason=reason, status=self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations_used": self.iterations_used,
            "search_calls_used": self.search_calls_used,
            "tokens_used": self.tokens_used,
            "time_elapsed": round(self.time_elapsed, 3),
            "cost_usd": round(self.cost_usd, 6),
            "budget_exceeded": self.budget_exceeded,
            "exceeded_reason": self.exceeded_reason,
            "budget": self.budget.to_dict(),
        }

    @property
    def utilization(self) -> dict[str, float | None]:
        """Return usage as a fraction of budget for each limit."""
        b = self.budget
        return {
            "iterations": (
                self.iterations_used / b.max_iterations if b.max_iterations else None
            ),
            "search_calls": (
                self.search_calls_used / b.max_search_calls if b.max_search_calls else None
            ),
            "tokens": self.tokens_used / b.max_tokens if b.max_tokens else None,
            "time": self.time_elapsed / b.max_time_seconds if b.max_time_seconds else None,
            "cost": self.cost_usd / b.max_cost_usd if b.max_cost_usd else None,
        }
