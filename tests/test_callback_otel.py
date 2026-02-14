"""Tests for OpenTelemetryCallback.

OpenTelemetry is installed, so we use the real API but mock the tracer/spans
to test without a real OTLP collector.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from deeprecall.core.guardrails import BudgetStatus, QueryBudget
from deeprecall.core.types import DeepRecallResult, ReasoningStep, Source, UsageInfo


class TestOpenTelemetryCallback:
    """Test OpenTelemetryCallback with mocked tracer."""

    def _make_callback(self):
        """Create an OpenTelemetryCallback with a mocked tracer (no real OTLP export)."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_span.return_value = mock_span

        with patch("deeprecall.core.callback_otel._init_tracer", return_value=mock_tracer):
            from deeprecall.core.callback_otel import OpenTelemetryCallback

            cb = OpenTelemetryCallback(service_name="test-service")
            return cb, mock_tracer, mock_span

    def _make_config(self):
        from deeprecall.core.config import DeepRecallConfig

        return DeepRecallConfig(
            backend="openai",
            backend_kwargs={"model_name": "gpt-4o-mini"},
        )

    def test_on_query_start_creates_root_span(self):
        """Should create a root span with query attributes."""
        cb, mock_tracer, mock_span = self._make_callback()
        config = self._make_config()

        cb.on_query_start("What is AI?", config)

        mock_tracer.start_span.assert_called_once()
        call_args = mock_tracer.start_span.call_args
        assert call_args[0][0] == "deeprecall.query"
        attrs = call_args[1]["attributes"]
        assert attrs["deeprecall.query"] == "What is AI?"
        assert attrs["deeprecall.backend"] == "openai"
        assert attrs["deeprecall.model"] == "gpt-4o-mini"
        assert cb._local.current_span is mock_span

    def test_on_reasoning_step_creates_child_span(self):
        """Should create a child span for each reasoning step."""
        cb, mock_tracer, _ = self._make_callback()

        step = ReasoningStep(
            iteration=1,
            action="search",
            code="search_db('test')",
            output="Found 3 results",
        )
        budget = BudgetStatus(budget=QueryBudget())
        budget.iterations_used = 1
        budget.search_calls_used = 1

        cb.on_reasoning_step(step, budget)

        assert mock_tracer.start_span.call_count == 1
        child_span = mock_tracer.start_span.return_value
        child_span.end.assert_called_once()

        call_args = mock_tracer.start_span.call_args
        assert call_args[0][0] == "deeprecall.step.1"
        attrs = call_args[1]["attributes"]
        assert attrs["deeprecall.step.action"] == "search"
        assert attrs["deeprecall.step.has_code"] is True

    def test_on_search_creates_search_span(self):
        """Should create a span for vector store searches."""
        cb, mock_tracer, _ = self._make_callback()

        cb.on_search("machine learning", num_results=5, time_ms=12.5)

        call_args = mock_tracer.start_span.call_args
        assert call_args[0][0] == "deeprecall.search"
        attrs = call_args[1]["attributes"]
        assert attrs["deeprecall.search.query"] == "machine learning"
        assert attrs["deeprecall.search.num_results"] == 5
        assert attrs["deeprecall.search.latency_ms"] == 12.5

    def test_on_query_end_closes_span_success(self):
        """Should set final attributes and close the root span on success."""
        cb, mock_tracer, mock_span = self._make_callback()
        cb._local.current_span = mock_span
        # Fake the context token so detach doesn't fail
        cb._local.ctx_token = MagicMock()

        result = DeepRecallResult(
            answer="AI is artificial intelligence.",
            sources=[Source(content="doc1", metadata={}, score=0.9, id="1")],
            reasoning_trace=[],
            usage=UsageInfo(total_input_tokens=100, total_output_tokens=50, total_calls=2),
            execution_time=1.5,
            query="What is AI?",
            confidence=0.85,
        )

        cb.on_query_end(result)

        mock_span.set_attributes.assert_called_once()
        set_attrs = mock_span.set_attributes.call_args[0][0]
        assert set_attrs["deeprecall.execution_time_s"] == 1.5
        assert set_attrs["deeprecall.tokens.total"] == 150
        assert set_attrs["deeprecall.sources_count"] == 1
        assert set_attrs["deeprecall.answer_length"] == len("AI is artificial intelligence.")

        mock_span.set_attribute.assert_called_with("deeprecall.confidence", 0.85)
        mock_span.end.assert_called_once()
        assert cb._local.current_span is None

    def test_on_query_end_with_error(self):
        """Should set error status when result has an error."""
        cb, _, mock_span = self._make_callback()
        cb._local.current_span = mock_span
        cb._local.ctx_token = MagicMock()

        result = DeepRecallResult(
            answer="[Partial]",
            sources=[],
            reasoning_trace=[],
            usage=UsageInfo(),
            execution_time=5.0,
            query="test",
            error="Budget exceeded",
        )

        cb.on_query_end(result)

        # Verify error was set (StatusCode.ERROR from real opentelemetry)
        mock_span.set_status.assert_called_once()
        status_args = mock_span.set_status.call_args[0]
        # First arg should be StatusCode.ERROR (value=2 in OTEL)
        assert str(status_args[0]).endswith("ERROR") or status_args[0].value == 2
        assert status_args[1] == "Budget exceeded"

        mock_span.set_attribute.assert_any_call("deeprecall.error", "Budget exceeded")
        mock_span.end.assert_called_once()

    def test_on_error_records_exception(self):
        """Should record exception and close span on error."""
        cb, _, mock_span = self._make_callback()
        cb._local.current_span = mock_span
        error = ValueError("Something broke")

        cb.on_error(error)

        mock_span.record_exception.assert_called_once_with(error)
        mock_span.end.assert_called_once()
        assert cb._local.current_span is None

    def test_on_budget_warning_adds_event(self):
        """Should add a span event for budget warnings."""
        cb, _, mock_span = self._make_callback()
        cb._local.current_span = mock_span

        status = BudgetStatus(budget=QueryBudget(max_search_calls=5))
        status.search_calls_used = 5
        status.iterations_used = 3
        status.exceeded_reason = "search_calls"

        cb.on_budget_warning(status)

        mock_span.add_event.assert_called_once_with(
            "budget_exceeded",
            attributes={
                "reason": "search_calls",
                "iterations_used": 3,
                "search_calls_used": 5,
            },
        )

    def test_step_counter_increments(self):
        """Should increment step counter for each reasoning step."""
        cb, mock_tracer, _ = self._make_callback()
        budget = BudgetStatus(budget=QueryBudget())

        for i in range(3):
            step = ReasoningStep(iteration=i + 1, action="compute")
            cb.on_reasoning_step(step, budget)

        assert cb._local.step_count == 3
        calls = mock_tracer.start_span.call_args_list
        assert calls[0][0][0] == "deeprecall.step.1"
        assert calls[1][0][0] == "deeprecall.step.2"
        assert calls[2][0][0] == "deeprecall.step.3"

    def test_no_span_does_not_crash(self):
        """on_query_end and on_error should be no-ops if no current span."""
        cb, _, _ = self._make_callback()
        cb._local.current_span = None

        result = DeepRecallResult(
            answer="ok", sources=[], reasoning_trace=[], usage=UsageInfo(),
            execution_time=0.1, query="test",
        )
        cb.on_query_end(result)  # Should not raise
        cb.on_error(ValueError("test"))  # Should not raise
        cb.on_budget_warning(BudgetStatus(budget=QueryBudget()))  # Should not raise

    def test_truncation(self):
        """Should truncate long query text in span attributes."""
        from deeprecall.core.callback_otel import _truncate

        short = "Hello"
        assert _truncate(short, 100) == "Hello"

        long_text = "A" * 2000
        result = _truncate(long_text, 1000)
        assert len(result) == 1003  # 1000 + "..."
        assert result.endswith("...")
