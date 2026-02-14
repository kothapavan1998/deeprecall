"""Tests for the DeepRecall tracer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deeprecall.core.guardrails import BudgetExceededError, QueryBudget
from deeprecall.core.tracer import DeepRecallTracer


def _make_mock_iteration(code: str = "", stdout: str = "", stderr: str = ""):
    """Create a mock RLMIteration-like object."""
    block = MagicMock()
    block.code = code
    block.result = MagicMock()
    block.result.stdout = stdout
    block.result.stderr = stderr
    block.result.rlm_calls = []

    iteration = MagicMock()
    iteration.code_blocks = [block]
    iteration.final_answer = None
    iteration.iteration_time = 0.5
    return iteration


class TestDeepRecallTracer:
    def test_basic_logging(self):
        tracer = DeepRecallTracer()
        iteration = _make_mock_iteration(
            code='results = search_db("python")\nprint(results)',
            stdout='[{"content": "Python is...", "score": 0.9}]',
        )
        tracer.log(iteration)

        assert len(tracer.steps) == 1
        step = tracer.steps[0]
        assert step.iteration == 1
        assert step.action == "search_and_reasoning"
        assert step.code is not None
        assert "search_db" in step.code
        assert len(step.searches) > 0

    def test_iteration_counting(self):
        tracer = DeepRecallTracer()
        for _ in range(3):
            tracer.log(_make_mock_iteration())

        assert len(tracer.steps) == 3
        assert tracer.budget_status.iterations_used == 3

    def test_search_call_counting(self):
        tracer = DeepRecallTracer()
        tracer.log(_make_mock_iteration(code='r1 = search_db("topic1")\nr2 = search_db("topic2")'))

        assert tracer.budget_status.search_calls_used == 2

    def test_budget_enforcement(self):
        budget = QueryBudget(max_iterations=2)
        tracer = DeepRecallTracer(budget=budget)

        tracer.log(_make_mock_iteration())  # iteration 1 - ok
        tracer.log(_make_mock_iteration())  # iteration 2 - ok

        with pytest.raises(BudgetExceededError):
            tracer.log(_make_mock_iteration())  # iteration 3 - exceeds budget

    def test_final_answer_action(self):
        tracer = DeepRecallTracer()
        iteration = _make_mock_iteration()
        iteration.final_answer = "The answer is 42"
        tracer.log(iteration)

        assert tracer.steps[0].action == "final_answer"

    def test_sub_llm_call_tracking(self):
        tracer = DeepRecallTracer()
        iteration = _make_mock_iteration()
        iteration.code_blocks[0].result.rlm_calls = [MagicMock(), MagicMock()]
        tracer.log(iteration)

        assert tracer.steps[0].sub_llm_calls == 2

    def test_get_trace_returns_copy(self):
        tracer = DeepRecallTracer()
        tracer.log(_make_mock_iteration())

        trace1 = tracer.get_trace()
        trace2 = tracer.get_trace()
        assert trace1 is not trace2
        assert trace1[0].iteration == trace2[0].iteration

    def test_log_metadata_is_noop(self):
        tracer = DeepRecallTracer()
        tracer.log_metadata(MagicMock())  # Should not raise

    def test_callback_integration(self):
        callback_manager = MagicMock()
        tracer = DeepRecallTracer(callback_manager=callback_manager)
        tracer.log(_make_mock_iteration())

        callback_manager.on_reasoning_step.assert_called_once()
