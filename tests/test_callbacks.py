"""Tests for the callback system."""

from __future__ import annotations

import tempfile

from deeprecall.core.callbacks import (
    BaseCallback,
    CallbackManager,
    ConsoleCallback,
    JSONLCallback,
    UsageTrackingCallback,
)
from deeprecall.core.config import DeepRecallConfig
from deeprecall.core.guardrails import BudgetStatus
from deeprecall.core.types import DeepRecallResult, ReasoningStep, UsageInfo


class RecordingCallback(BaseCallback):
    """Test callback that records all events."""

    def __init__(self):
        self.events: list[str] = []

    def on_query_start(self, query, config):
        self.events.append(f"start:{query}")

    def on_reasoning_step(self, step, budget_status):
        self.events.append(f"step:{step.iteration}")

    def on_query_end(self, result):
        self.events.append(f"end:{len(result.answer)}")

    def on_error(self, error):
        self.events.append(f"error:{error}")

    def on_budget_warning(self, status):
        self.events.append(f"budget:{status.exceeded_reason}")


class TestCallbackManager:
    def test_dispatches_to_all_callbacks(self):
        cb1 = RecordingCallback()
        cb2 = RecordingCallback()
        manager = CallbackManager([cb1, cb2])

        config = DeepRecallConfig()
        manager.on_query_start("test query", config)

        assert cb1.events == ["start:test query"]
        assert cb2.events == ["start:test query"]

    def test_add_callback(self):
        manager = CallbackManager()
        cb = RecordingCallback()
        manager.add(cb)

        manager.on_query_start("test", DeepRecallConfig())
        assert len(cb.events) == 1

    def test_error_in_callback_does_not_propagate(self):
        class FailingCallback(BaseCallback):
            def on_query_start(self, query, config):
                raise RuntimeError("oops")

        cb_good = RecordingCallback()
        manager = CallbackManager([FailingCallback(), cb_good])

        # Should not raise, and the good callback still fires
        manager.on_query_start("test", DeepRecallConfig())
        assert len(cb_good.events) == 1

    def test_reasoning_step_dispatch(self):
        cb = RecordingCallback()
        manager = CallbackManager([cb])

        step = ReasoningStep(iteration=1, action="reasoning")
        status = BudgetStatus()
        manager.on_reasoning_step(step, status)

        assert cb.events == ["step:1"]


class TestUsageTrackingCallback:
    def test_tracks_multiple_queries(self):
        tracker = UsageTrackingCallback()

        for _i in range(3):
            result = DeepRecallResult(
                answer="answer",
                usage=UsageInfo(total_input_tokens=100, total_output_tokens=50),
                execution_time=1.0,
                budget_status={"search_calls_used": 2},
            )
            tracker.on_query_end(result)

        summary = tracker.summary()
        assert summary["total_queries"] == 3
        assert summary["total_tokens"] == 450  # (100+50) * 3
        assert summary["total_searches"] == 6  # 2 * 3
        assert summary["total_time"] == 3.0

    def test_tracks_errors(self):
        tracker = UsageTrackingCallback()
        result = DeepRecallResult(answer="partial", error="Budget exceeded")
        tracker.on_query_end(result)
        tracker.on_error(RuntimeError("fail"))

        assert tracker.errors == 2


class TestJSONLCallback:
    def test_writes_events_to_file(self):
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            cb = JSONLCallback(log_dir=tmpdir)

            cb.on_query_start("test query", DeepRecallConfig())
            cb.on_query_end(DeepRecallResult(answer="answer", execution_time=1.0))

            # Read the log file
            with open(cb.log_path) as f:
                lines = f.readlines()

            assert len(lines) == 2
            entry1 = json.loads(lines[0])
            assert entry1["type"] == "query_start"
            assert entry1["query"] == "test query"

            entry2 = json.loads(lines[1])
            assert entry2["type"] == "query_end"


class TestConsoleCallback:
    def test_instantiation(self):
        cb = ConsoleCallback(show_code=True, show_output=False)
        assert cb.show_code is True
        assert cb.show_output is False

    def test_on_query_start(self, capsys):
        cb = ConsoleCallback()
        cb.on_query_start("test query", DeepRecallConfig())
        # Just verify it doesn't crash -- rich output is complex to assert
        assert cb._start_time is not None
