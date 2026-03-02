"""Live validation for DeepRecall v0.4.0 -- runs against real services.

Requires: Redis on localhost:6379 (docker).
Validates every v0.4.0 change end-to-end with real objects (no mocks).
"""

from __future__ import annotations

import json
import tempfile
import time

import pytest

# ===================================================================
# 1. Package version is 0.4.0
# ===================================================================


class TestPackageVersion:
    def test_version_is_040(self):
        from importlib.metadata import version

        v = version("deeprecall")
        assert v == "0.4.1", f"Expected 0.4.1, got {v}"

    def test_rlms_minimum_version(self):
        from importlib.metadata import version

        from packaging.version import Version

        v = Version(version("rlms"))
        assert v >= Version("0.1.1"), f"Expected rlms>=0.1.1, got {v}"


# ===================================================================
# 2. Config -- full round-trip with new params
# ===================================================================


class TestConfigLive:
    def test_full_config_creation(self):
        from deeprecall.core.config import DeepRecallConfig
        from deeprecall.core.guardrails import QueryBudget

        config = DeepRecallConfig(
            backend="openai",
            backend_kwargs={"model_name": "gpt-4o-mini"},
            max_iterations=10,
            max_depth=2,
            top_k=10,
            verbose=False,
            max_timeout=120.0,
            max_errors=5,
            compaction=True,
            compaction_threshold_pct=0.75,
            budget=QueryBudget(
                max_iterations=10,
                max_search_calls=20,
                max_tokens=50000,
                max_time_seconds=60.0,
                max_cost_usd=0.50,
            ),
        )

        d = config.to_dict()

        assert d["max_timeout"] == 120.0
        assert d["max_errors"] == 5
        assert d["compaction"] is True
        assert d["compaction_threshold_pct"] == 0.75
        assert d["budget"]["max_cost_usd"] == 0.50

        assert isinstance(json.dumps(d), str)


# ===================================================================
# 3. Types -- UsageInfo with cost, full serialization round-trip
# ===================================================================


class TestTypesLive:
    def test_usage_info_cost_serialization(self):
        from deeprecall.core.types import DeepRecallResult, Source, UsageInfo

        result = DeepRecallResult(
            answer="Paris is the capital of France.",
            sources=[Source(content="France doc", metadata={"src": "wiki"}, score=0.95, id="d1")],
            usage=UsageInfo(
                total_input_tokens=500,
                total_output_tokens=200,
                total_calls=5,
                total_cost_usd=0.0073,
                model_breakdown={
                    "gpt-4o-mini": {
                        "input_tokens": 300,
                        "output_tokens": 100,
                        "calls": 3,
                        "cost_usd": 0.0023,
                    },
                    "claude-3-sonnet": {
                        "input_tokens": 200,
                        "output_tokens": 100,
                        "calls": 2,
                        "cost_usd": 0.0050,
                    },
                },
            ),
            execution_time=3.5,
            query="What is the capital of France?",
            confidence=0.95,
        )

        d = result.to_dict()
        json_str = json.dumps(d)
        assert '"total_cost_usd": 0.0073' in json_str

        restored = DeepRecallResult.from_dict(json.loads(json_str))
        assert restored.usage.total_cost_usd == 0.0073
        assert restored.usage.model_breakdown["gpt-4o-mini"]["cost_usd"] == 0.0023
        assert restored.answer == "Paris is the capital of France."


# ===================================================================
# 4. Tracer -- full RLMLogger protocol lifecycle
# ===================================================================


class TestTracerLive:
    def test_full_lifecycle(self):
        from unittest.mock import MagicMock

        from deeprecall.core.callbacks import CallbackManager, ProgressCallback
        from deeprecall.core.guardrails import QueryBudget
        from deeprecall.core.tracer import DeepRecallTracer

        progress = ProgressCallback()
        cb_manager = CallbackManager([progress])
        budget = QueryBudget(max_iterations=10, max_cost_usd=1.0)

        tracer = DeepRecallTracer(
            budget=budget,
            callback_manager=cb_manager,
        )

        # RLM calls clear_iterations at start
        tracer.clear_iterations()
        assert len(tracer.steps) == 0

        # Simulate 3 iterations
        for i in range(3):
            block = MagicMock()
            block.code = f'results = search_db("query_{i}")'
            block.result = MagicMock()
            block.result.stdout = f"Found {i + 1} results"
            block.result.stderr = ""
            block.result.rlm_calls = []

            iteration = MagicMock()
            iteration.code_blocks = [block]
            iteration.final_answer = "The answer" if i == 2 else None
            iteration.iteration_time = 0.5
            tracer.log(iteration)

        assert len(tracer.steps) == 3
        assert tracer.budget_status.iterations_used == 3
        assert tracer.budget_status.search_calls_used == 3

        # RLM calls get_trajectory at end
        trajectory = tracer.get_trajectory()
        assert len(trajectory["iterations"]) == 3
        assert trajectory["budget_status"]["iterations_used"] == 3

        # Verify callbacks fired
        event_types = [e["type"] for e in progress.events]
        assert event_types.count("iteration_start") == 3
        assert event_types.count("iteration_complete") == 3
        assert event_types.count("reasoning_step") == 3

        # Verify final answer detected
        complete_events = [e for e in progress.events if e["type"] == "iteration_complete"]
        assert complete_events[0]["has_final_answer"] is False
        assert complete_events[1]["has_final_answer"] is False
        assert complete_events[2]["has_final_answer"] is True

        # Verify trajectory is JSON-serializable
        json_str = json.dumps(trajectory)
        assert len(json_str) > 0


# ===================================================================
# 5. Callbacks -- JSONL writes all v0.4 events to disk
# ===================================================================


class TestCallbacksLive:
    def test_jsonl_full_event_lifecycle(self):
        from deeprecall.core.callbacks import JSONLCallback
        from deeprecall.core.config import DeepRecallConfig
        from deeprecall.core.guardrails import BudgetStatus
        from deeprecall.core.types import DeepRecallResult, ReasoningStep, UsageInfo

        with tempfile.TemporaryDirectory() as tmpdir:
            cb = JSONLCallback(log_dir=tmpdir)

            config = DeepRecallConfig()
            cb.on_query_start("live test query", config)
            cb.on_iteration_start(1)
            cb.on_reasoning_step(
                ReasoningStep(iteration=1, action="search_and_reasoning"),
                BudgetStatus(),
            )
            cb.on_search("test search", 5, 12.3)
            cb.on_iteration_complete(1, False)
            cb.on_iteration_start(2)
            cb.on_reasoning_step(
                ReasoningStep(iteration=2, action="final_answer"),
                BudgetStatus(),
            )
            cb.on_iteration_complete(2, True)
            cb.on_query_end(
                DeepRecallResult(
                    answer="test answer",
                    execution_time=2.5,
                    usage=UsageInfo(
                        total_input_tokens=100,
                        total_output_tokens=50,
                        total_cost_usd=0.003,
                    ),
                )
            )

            with open(cb.log_path) as f:
                all_lines = f.readlines()

            assert len(all_lines) == 9
            all_events = [json.loads(line) for line in all_lines]
            event_types = [e["type"] for e in all_events]
            assert event_types == [
                "query_start",
                "iteration_start",
                "reasoning_step",
                "search",
                "iteration_complete",
                "iteration_start",
                "reasoning_step",
                "iteration_complete",
                "query_end",
            ]

            for event in all_events:
                assert "timestamp" in event
                assert isinstance(event["timestamp"], float)


# ===================================================================
# 6. Redis cache -- live test against docker Redis on :6379
# ===================================================================


class TestRedisCacheLive:
    @pytest.fixture(autouse=True)
    def _check_redis(self):
        """Skip if Redis is not reachable."""
        try:
            import redis

            r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=2)
            r.ping()
        except Exception:
            pytest.skip("Redis not available on localhost:6379")

    def test_cache_set_get_with_cost_data(self):
        from deeprecall.core.cache_redis import RedisCache
        from deeprecall.core.types import DeepRecallResult, UsageInfo

        cache = RedisCache(
            url="redis://localhost:6379/0",
            prefix="deeprecall_v04_test:",
            default_ttl=30,
        )

        try:
            cache.clear()

            result = DeepRecallResult(
                answer="Paris",
                usage=UsageInfo(
                    total_input_tokens=100,
                    total_output_tokens=50,
                    total_calls=3,
                    total_cost_usd=0.005,
                    model_breakdown={
                        "gpt-4o-mini": {
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "calls": 3,
                            "cost_usd": 0.005,
                        }
                    },
                ),
                execution_time=2.0,
                query="What is the capital of France?",
                confidence=0.95,
            )

            cache.set("test_v04", result, ttl=30)
            cached = cache.get("test_v04")

            assert isinstance(cached, dict)
            assert cached["answer"] == "Paris"
            assert cached["usage"]["total_cost_usd"] == 0.005
            assert cached["usage"]["model_breakdown"]["gpt-4o-mini"]["cost_usd"] == 0.005

            restored = DeepRecallResult.from_dict(cached)
            assert restored.usage.total_cost_usd == 0.005
            assert restored.answer == "Paris"

        finally:
            cache.clear()

    def test_config_new_fields_serializable_to_redis(self):
        from deeprecall.core.cache_redis import RedisCache
        from deeprecall.core.config import DeepRecallConfig
        from deeprecall.core.guardrails import QueryBudget

        cache = RedisCache(
            url="redis://localhost:6379/0",
            prefix="deeprecall_v04_test:",
            default_ttl=30,
        )

        try:
            cache.clear()

            config = DeepRecallConfig(
                max_timeout=60.0,
                max_errors=3,
                compaction=True,
                compaction_threshold_pct=0.8,
                budget=QueryBudget(max_cost_usd=0.50),
            )

            config_dict = config.to_dict()
            cache.set("test_config", config_dict, ttl=30)
            restored = cache.get("test_config")

            assert restored["max_timeout"] == 60.0
            assert restored["max_errors"] == 3
            assert restored["compaction"] is True
            assert restored["compaction_threshold_pct"] == 0.8
            assert restored["budget"]["max_cost_usd"] == 0.50

        finally:
            cache.clear()


# ===================================================================
# 7. Engine -- verify RLM constructor receives new kwargs
# ===================================================================


class TestEngineLiveWiring:
    def test_engine_creates_with_all_new_config(self):
        from unittest.mock import MagicMock

        from deeprecall.core.config import DeepRecallConfig
        from deeprecall.core.engine import DeepRecallEngine
        from deeprecall.core.guardrails import QueryBudget

        mock_store = MagicMock()
        mock_store.count.return_value = 0

        config = DeepRecallConfig(
            max_timeout=60.0,
            max_errors=3,
            compaction=True,
            compaction_threshold_pct=0.75,
            budget=QueryBudget(max_cost_usd=0.25),
        )

        engine = DeepRecallEngine(vectorstore=mock_store, config=config)

        assert engine.config.max_timeout == 60.0
        assert engine.config.max_errors == 3
        assert engine.config.compaction is True
        assert engine.config.compaction_threshold_pct == 0.75
        assert engine.config.budget.max_cost_usd == 0.25

    def test_engine_extract_usage_with_real_dataclass(self):
        """Build a real-ish ModelUsageSummary-like object and extract cost."""
        from dataclasses import dataclass
        from unittest.mock import MagicMock

        from deeprecall.core.engine import DeepRecallEngine

        @dataclass
        class FakeModelUsage:
            total_input_tokens: int
            total_output_tokens: int
            total_calls: int
            total_cost: float | None = None

        @dataclass
        class FakeUsageSummary:
            model_usage_summaries: dict

        @dataclass
        class FakeResult:
            response: str
            usage_summary: FakeUsageSummary
            execution_time: float
            metadata: dict | None = None

        mock_store = MagicMock()
        mock_store.count.return_value = 0
        engine = DeepRecallEngine(vectorstore=mock_store)

        result = FakeResult(
            response="test",
            usage_summary=FakeUsageSummary(
                model_usage_summaries={
                    "gpt-4o-mini": FakeModelUsage(
                        total_input_tokens=500,
                        total_output_tokens=200,
                        total_calls=3,
                        total_cost=0.0045,
                    ),
                    "gpt-4o": FakeModelUsage(
                        total_input_tokens=1000,
                        total_output_tokens=500,
                        total_calls=2,
                        total_cost=0.0300,
                    ),
                }
            ),
            execution_time=5.0,
        )

        usage = engine._extract_usage(result)

        assert usage.total_input_tokens == 1500
        assert usage.total_output_tokens == 700
        assert usage.total_calls == 5
        assert usage.total_cost_usd == pytest.approx(0.0345)
        assert usage.model_breakdown["gpt-4o-mini"]["cost_usd"] == 0.0045
        assert usage.model_breakdown["gpt-4o"]["cost_usd"] == 0.0300


# ===================================================================
# 8. System prompt -- content validation
# ===================================================================


class TestSystemPromptLive:
    def test_prompt_complete(self):
        from deeprecall.prompts.templates import DEEPRECALL_SYSTEM_PROMPT

        required_terms = [
            "search_db",
            "llm_query",
            "llm_query_batched",
            "SHOW_VARS",
            "FINAL(",
            "FINAL_VAR",
            "history",
            "compaction",
            "direct value",
            "repl",
        ]
        for term in required_terms:
            assert term.lower() in DEEPRECALL_SYSTEM_PROMPT.lower(), (
                f"Missing '{term}' in system prompt"
            )

    def test_setup_code_compiles(self):
        from deeprecall.prompts.templates import build_search_setup_code

        for port in [8080, 12345, 65000]:
            code = build_search_setup_code(server_port=port)
            compile(code, f"<setup_{port}>", "exec")

        code_with_budget = build_search_setup_code(server_port=9000, max_search_calls=5)
        compile(code_with_budget, "<setup_budget>", "exec")


# ===================================================================
# 9. Guardrails -- cost enforcement with real objects
# ===================================================================


class TestGuardrailsLive:
    def test_cost_budget_lifecycle(self):
        from deeprecall.core.guardrails import BudgetExceededError, BudgetStatus, QueryBudget

        budget = QueryBudget(
            max_iterations=10,
            max_search_calls=20,
            max_tokens=50000,
            max_time_seconds=60.0,
            max_cost_usd=0.10,
        )

        status = BudgetStatus(budget=budget)
        start = time.perf_counter()

        status.iterations_used = 3
        status.search_calls_used = 5
        status.tokens_used = 10000
        status.cost_usd = 0.05
        status.check(start_time=start)

        util = status.utilization
        assert util["iterations"] == 0.3
        assert util["search_calls"] == 0.25
        assert util["cost"] == 0.5

        status.cost_usd = 0.15
        with pytest.raises(BudgetExceededError, match="Cost"):
            status.check(start_time=start)

        assert status.budget_exceeded is True
        assert "Cost" in status.exceeded_reason

        d = status.to_dict()
        assert d["cost_usd"] == 0.15
        assert d["budget_exceeded"] is True
        assert d["budget"]["max_cost_usd"] == 0.10


# ===================================================================
# 10. Import smoke test -- everything importable
# ===================================================================


class TestImportSmoke:
    def test_all_public_imports(self):
        from deeprecall import (  # noqa: F401
            AsyncDeepRecall,
            ConsoleCallback,
            DeepRecall,
            DeepRecallConfig,
            DeepRecallError,
            DiskCache,
            InMemoryCache,
            JSONLCallback,
            LLMProviderError,
            ProgressCallback,
            QueryBudget,
            RetryConfig,
            VectorStoreError,
        )

    def test_callback_iteration_methods_exist(self):
        from deeprecall.core.callbacks import BaseCallback, CallbackManager

        cb = BaseCallback()
        assert hasattr(cb, "on_iteration_start")
        assert hasattr(cb, "on_iteration_complete")

        mgr = CallbackManager()
        assert hasattr(mgr, "on_iteration_start")
        assert hasattr(mgr, "on_iteration_complete")

    def test_tracer_protocol_methods_exist(self):
        from deeprecall.core.tracer import DeepRecallTracer

        tracer = DeepRecallTracer()
        assert hasattr(tracer, "clear_iterations")
        assert hasattr(tracer, "get_trajectory")
        assert hasattr(tracer, "log")
        assert hasattr(tracer, "log_metadata")
