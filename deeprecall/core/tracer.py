"""DeepRecall Tracer -- captures reasoning iterations from RLM.

Implements RLM's RLMLogger interface to intercept every iteration,
build the reasoning trace, enforce budget limits, and fire callbacks.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

from deeprecall.core.guardrails import BudgetStatus, QueryBudget
from deeprecall.core.types import ReasoningStep

if TYPE_CHECKING:
    from deeprecall.core.callbacks import CallbackManager


class DeepRecallTracer:
    """RLMLogger-compatible tracer that captures iterations and enforces budgets.

    Passed to RLM as the ``logger`` argument. RLM calls ``log(iteration)``
    after every reasoning iteration, giving us full visibility.

    Args:
        budget: Optional budget limits to enforce.
        callback_manager: Optional callback manager to notify on each step.
        start_time: The time.perf_counter() value when the query started.
    """

    def __init__(
        self,
        budget: QueryBudget | None = None,
        callback_manager: CallbackManager | None = None,
        start_time: float | None = None,
    ):
        self.steps: list[ReasoningStep] = []
        self.budget_status = BudgetStatus(budget=budget or QueryBudget())
        self.callback_manager = callback_manager
        self.start_time = start_time or time.perf_counter()
        self._search_pattern = re.compile(r"search_db\s*\(")

    def log(self, iteration: Any) -> None:
        """Called by RLM after each reasoning iteration.

        Extracts code, output, search calls, and sub-LLM calls from the
        RLMIteration object and builds a ReasoningStep.

        Args:
            iteration: An rlm.core.types.RLMIteration instance.
        """
        step_num = len(self.steps) + 1

        # Extract data from code blocks
        code_parts: list[str] = []
        output_parts: list[str] = []
        search_calls: list[dict[str, Any]] = []
        sub_llm_count = 0

        for block in getattr(iteration, "code_blocks", []):
            code_parts.append(block.code)

            result = block.result
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(f"[stderr] {result.stderr}")

            # Count sub-LLM calls
            sub_llm_count += len(getattr(result, "rlm_calls", []))

            # Detect search_db() calls in the code
            if self._search_pattern.search(block.code):
                search_calls.extend(self._extract_search_calls(block.code, result.stdout))

        # Determine action type
        action = "reasoning"
        if search_calls:
            action = "search_and_reasoning"
        elif getattr(iteration, "final_answer", None):
            action = "final_answer"

        step = ReasoningStep(
            iteration=step_num,
            action=action,
            code="\n".join(code_parts) if code_parts else None,
            output="\n".join(output_parts) if output_parts else None,
            searches=search_calls,
            sub_llm_calls=sub_llm_count,
            iteration_time=getattr(iteration, "iteration_time", None),
        )
        self.steps.append(step)

        # Update budget status
        self.budget_status.iterations_used = step_num
        self.budget_status.search_calls_used += len(search_calls)

        # Fire callback
        if self.callback_manager:
            self.callback_manager.on_reasoning_step(step, self.budget_status)

        # Check budget limits (may raise BudgetExceededError)
        self.budget_status.check(self.start_time)

    def log_metadata(self, metadata: Any) -> None:
        """Accept metadata calls from RLM (no-op for our use case)."""

    def _extract_search_calls(
        self, code: str, stdout: str | None
    ) -> list[dict[str, Any]]:
        """Parse search_db() calls from code to track queries."""
        calls: list[dict[str, Any]] = []
        # Find all search_db("...") patterns in the code
        pattern = re.compile(r'search_db\s*\(\s*["\']([^"\']+)["\']')
        for match in pattern.finditer(code):
            calls.append({
                "query": match.group(1),
                "has_results": stdout is not None and "content" in (stdout or ""),
            })
        # If we detected search_db calls but couldn't parse the query
        if not calls and self._search_pattern.search(code):
            calls.append({"query": "<dynamic>", "has_results": True})
        return calls

    def get_total_search_calls(self) -> int:
        """Return total search calls across all iterations."""
        return self.budget_status.search_calls_used

    def get_trace(self) -> list[ReasoningStep]:
        """Return the collected reasoning trace."""
        return list(self.steps)
