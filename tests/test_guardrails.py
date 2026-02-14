"""Tests for the guardrails and budget system."""

from __future__ import annotations

import time

import pytest

from deeprecall.core.guardrails import BudgetExceededError, BudgetStatus, QueryBudget


class TestQueryBudget:
    def test_default_budget_no_limits(self):
        budget = QueryBudget()
        assert budget.max_iterations is None
        assert budget.max_search_calls is None
        assert budget.max_tokens is None
        assert budget.max_time_seconds is None
        assert budget.max_cost_usd is None

    def test_custom_budget(self):
        budget = QueryBudget(
            max_iterations=10,
            max_search_calls=20,
            max_tokens=50000,
            max_time_seconds=30.0,
            max_cost_usd=0.50,
        )
        assert budget.max_iterations == 10
        assert budget.max_search_calls == 20
        assert budget.max_tokens == 50000
        assert budget.max_time_seconds == 30.0
        assert budget.max_cost_usd == 0.50

    def test_to_dict(self):
        budget = QueryBudget(max_search_calls=10)
        d = budget.to_dict()
        assert d["max_search_calls"] == 10
        assert d["max_iterations"] is None


class TestBudgetStatus:
    def test_default_status(self):
        status = BudgetStatus()
        assert status.iterations_used == 0
        assert status.budget_exceeded is False

    def test_check_passes_with_no_limits(self):
        status = BudgetStatus()
        status.iterations_used = 100
        status.search_calls_used = 100
        # No budget limits set, so no exception
        status.check(start_time=time.perf_counter())

    def test_check_iteration_limit(self):
        budget = QueryBudget(max_iterations=5)
        status = BudgetStatus(budget=budget)
        status.iterations_used = 6

        with pytest.raises(BudgetExceededError, match="Iterations"):
            status.check(start_time=time.perf_counter())

    def test_check_search_limit(self):
        budget = QueryBudget(max_search_calls=3)
        status = BudgetStatus(budget=budget)
        status.search_calls_used = 4

        with pytest.raises(BudgetExceededError, match="Search calls"):
            status.check(start_time=time.perf_counter())

    def test_check_token_limit(self):
        budget = QueryBudget(max_tokens=1000)
        status = BudgetStatus(budget=budget)
        status.tokens_used = 1500

        with pytest.raises(BudgetExceededError, match="Tokens"):
            status.check(start_time=time.perf_counter())

    def test_check_time_limit(self):
        budget = QueryBudget(max_time_seconds=0.001)
        status = BudgetStatus(budget=budget)
        start = time.perf_counter() - 1.0  # 1 second ago

        with pytest.raises(BudgetExceededError, match="Time"):
            status.check(start_time=start)

    def test_budget_exceeded_error_has_status(self):
        budget = QueryBudget(max_iterations=1)
        status = BudgetStatus(budget=budget)
        status.iterations_used = 2

        with pytest.raises(BudgetExceededError) as exc_info:
            status.check(start_time=time.perf_counter())

        assert exc_info.value.status.budget_exceeded is True
        assert exc_info.value.status.exceeded_reason is not None

    def test_to_dict(self):
        status = BudgetStatus(iterations_used=5, search_calls_used=3)
        d = status.to_dict()
        assert d["iterations_used"] == 5
        assert d["search_calls_used"] == 3

    def test_utilization(self):
        budget = QueryBudget(max_iterations=10, max_search_calls=20)
        status = BudgetStatus(budget=budget)
        status.iterations_used = 5
        status.search_calls_used = 10

        util = status.utilization
        assert util["iterations"] == 0.5
        assert util["search_calls"] == 0.5
        assert util["tokens"] is None  # No limit set
