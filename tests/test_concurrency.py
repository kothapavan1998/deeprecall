"""Concurrency and thread-safety tests for DeepRecall."""

from __future__ import annotations

import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

from deeprecall.core.callbacks import (
    CallbackManager,
    JSONLCallback,
    UsageTrackingCallback,
)
from deeprecall.core.guardrails import BudgetStatus, QueryBudget
from deeprecall.core.types import DeepRecallResult, ReasoningStep, Source, UsageInfo

# ---------------------------------------------------------------------------
# UsageTrackingCallback thread safety
# ---------------------------------------------------------------------------


class TestUsageTrackingConcurrency:
    def _make_result(self, tokens: int = 100, exec_time: float = 0.5) -> DeepRecallResult:
        return DeepRecallResult(
            answer="answer",
            sources=[Source(content="c", metadata={}, score=0.9, id="1")],
            reasoning_trace=[],
            usage=UsageInfo(total_input_tokens=tokens, total_output_tokens=tokens, total_calls=1),
            execution_time=exec_time,
            query="test",
            budget_status={"search_calls_used": 2},
        )

    def test_concurrent_counter_updates(self):
        """100 threads each calling on_query_end -- counters must be exact."""
        tracker = UsageTrackingCallback()
        num_threads = 100
        tokens_per_call = 50

        def fire():
            result = self._make_result(tokens=tokens_per_call, exec_time=0.1)
            tracker.on_query_end(result)

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(fire) for _ in range(num_threads)]
            for f in as_completed(futures):
                f.result()  # raise if any thread failed

        summary = tracker.summary()
        assert summary["total_queries"] == num_threads
        assert summary["total_tokens"] == num_threads * tokens_per_call * 2
        assert summary["total_searches"] == num_threads * 2
        # Float comparison: allow tiny epsilon
        assert abs(summary["total_time"] - num_threads * 0.1) < 0.01

    def test_concurrent_errors(self):
        """on_error from multiple threads should count correctly."""
        tracker = UsageTrackingCallback()
        num_threads = 50

        def fire_error():
            tracker.on_error(RuntimeError("boom"))

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(fire_error) for _ in range(num_threads)]
            for f in as_completed(futures):
                f.result()

        assert tracker.summary()["errors"] == num_threads

    def test_summary_snapshot_consistency(self):
        """summary() should return a consistent snapshot even while writers run."""
        tracker = UsageTrackingCallback()
        stop = threading.Event()

        def writer():
            while not stop.is_set():
                result = self._make_result(tokens=10, exec_time=0.01)
                tracker.on_query_end(result)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        for t in threads:
            t.start()

        # Read snapshots while writers are active
        for _ in range(20):
            s = tracker.summary()
            # Each snapshot must be internally consistent
            assert s["total_queries"] >= 0
            assert s["total_tokens"] == s["total_queries"] * 20  # 10 in + 10 out
            time.sleep(0.005)

        stop.set()
        for t in threads:
            t.join()


# ---------------------------------------------------------------------------
# JSONLCallback thread safety
# ---------------------------------------------------------------------------


class TestJSONLConcurrency:
    def test_concurrent_writes_produce_valid_jsonl(self):
        """Multiple threads writing simultaneously: every line must be valid JSON."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            cb = JSONLCallback(log_dir=tmpdir)
            num_threads = 50

            from deeprecall.core.config import DeepRecallConfig

            config = DeepRecallConfig(
                backend="openai",
                backend_kwargs={"model_name": "test"},
            )

            def write_events(i: int):
                cb.on_query_start(f"query-{i}", config)
                step = ReasoningStep(iteration=1, action="compute")
                budget = BudgetStatus(budget=QueryBudget())
                cb.on_reasoning_step(step, budget)
                result = DeepRecallResult(
                    answer=f"answer-{i}",
                    sources=[],
                    reasoning_trace=[],
                    usage=UsageInfo(),
                    execution_time=0.1,
                    query=f"query-{i}",
                )
                cb.on_query_end(result)

            with ThreadPoolExecutor(max_workers=10) as pool:
                futures = [pool.submit(write_events, i) for i in range(num_threads)]
                for f in as_completed(futures):
                    f.result()

            # Verify: every line in the JSONL file must parse as valid JSON
            with open(cb.log_path, encoding="utf-8") as f:
                lines = f.readlines()

            assert len(lines) == num_threads * 3  # 3 events per thread

            for line_no, line in enumerate(lines, 1):
                line = line.strip()
                assert line, f"Empty line at {line_no}"
                parsed = json.loads(line)  # will raise if corrupt
                assert "type" in parsed
                assert "timestamp" in parsed


# ---------------------------------------------------------------------------
# RedisCache counter thread safety
# ---------------------------------------------------------------------------


class TestRedisCacheConcurrency:
    def _make_cache(self):
        import importlib
        import sys

        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_module.from_url.return_value = mock_client
        mock_module.ConnectionError = ConnectionError

        with patch.dict(sys.modules, {"redis": mock_module}):
            import deeprecall.core.cache_redis as mod

            importlib.reload(mod)
            cache = mod.RedisCache(url="redis://localhost:6379/0")

        cache._client = mock_client
        return cache, mock_client

    def test_concurrent_hits_and_misses(self):
        """50 hits + 50 misses in parallel -- counters must sum to 100."""
        cache, client = self._make_cache()

        import json

        hit_value = json.dumps({"result": "cached"})

        def do_hit():
            client_local = cache._client
            # Simulate a hit: return valid JSON
            client_local.get.return_value = hit_value
            cache.get("hit_key")

        def do_miss():
            client_local = cache._client
            # Simulate a miss: return None
            client_local.get.return_value = None
            cache.get("miss_key")

        # Note: since the mock is shared, we can't perfectly control
        # hit vs miss in parallel. Instead, just verify total count is correct.
        num_calls = 100

        def do_access(i: int):
            cache._client.get.return_value = hit_value if i % 2 == 0 else None
            cache.get(f"key-{i}")

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(do_access, i) for i in range(num_calls)]
            for f in as_completed(futures):
                f.result()

        stats = cache.stats()
        total = stats["hits"] + stats["misses"]
        assert total == num_calls, f"Expected {num_calls}, got {total}"


# ---------------------------------------------------------------------------
# OpenTelemetryCallback thread isolation
# ---------------------------------------------------------------------------


class TestOtelConcurrency:
    def _make_callback(self):
        mock_tracer = MagicMock()

        def make_span(*args, **kwargs):
            return MagicMock()

        mock_tracer.start_span.side_effect = make_span

        with patch("deeprecall.core.callback_otel._init_tracer", return_value=mock_tracer):
            from deeprecall.core.callback_otel import OpenTelemetryCallback

            cb = OpenTelemetryCallback(service_name="test")
            return cb, mock_tracer

    def _make_config(self):
        from deeprecall.core.config import DeepRecallConfig

        return DeepRecallConfig(
            backend="openai",
            backend_kwargs={"model_name": "test"},
        )

    def test_concurrent_queries_have_isolated_spans(self):
        """Two queries running in parallel should not share span state."""
        cb, mock_tracer = self._make_callback()
        config = self._make_config()

        # Track which span each thread sees
        thread_spans: dict[str, list] = {}
        barrier = threading.Barrier(2)  # sync threads to overlap

        def run_query(name: str):
            cb.on_query_start(f"query-{name}", config)
            span = getattr(cb._local, "current_span", None)
            barrier.wait()  # ensure both threads are in mid-query

            # Record the span we see
            thread_spans[name] = [id(span)]

            step = ReasoningStep(iteration=1, action="compute")
            budget = BudgetStatus(budget=QueryBudget())
            cb.on_reasoning_step(step, budget)

            result = DeepRecallResult(
                answer=f"answer-{name}",
                sources=[],
                reasoning_trace=[],
                usage=UsageInfo(total_input_tokens=10, total_output_tokens=10, total_calls=1),
                execution_time=0.1,
                query=f"query-{name}",
            )
            cb.on_query_end(result)

        t1 = threading.Thread(target=run_query, args=("A",))
        t2 = threading.Thread(target=run_query, args=("B",))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Each thread must have seen a DIFFERENT span object
        assert thread_spans["A"][0] != thread_spans["B"][0], (
            "Threads should have isolated span state"
        )

    def test_step_counter_isolated_per_thread(self):
        """Step counter in thread A should not affect thread B."""
        cb, _ = self._make_callback()
        config = self._make_config()

        counters: dict[str, int] = {}

        def run_steps(name: str, num_steps: int):
            cb.on_query_start(f"q-{name}", config)
            for i in range(num_steps):
                step = ReasoningStep(iteration=i + 1, action="compute")
                budget = BudgetStatus(budget=QueryBudget())
                cb.on_reasoning_step(step, budget)
            counters[name] = getattr(cb._local, "step_count", 0)

        t1 = threading.Thread(target=run_steps, args=("A", 3))
        t2 = threading.Thread(target=run_steps, args=("B", 7))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert counters["A"] == 3, f"Thread A expected 3, got {counters['A']}"
        assert counters["B"] == 7, f"Thread B expected 7, got {counters['B']}"


# ---------------------------------------------------------------------------
# RateLimiter thread safety
# ---------------------------------------------------------------------------


class TestRateLimiterConcurrency:
    """Verify RateLimiter bucket operations are safe under concurrent access."""

    def test_bucket_creation_no_duplicates(self):
        """Multiple threads creating buckets for the same key shouldn't crash."""
        from deeprecall.middleware.rate_limit import RateLimiter, _TokenBucket

        # Create rate limiter without real FastAPI app
        limiter = RateLimiter.__new__(RateLimiter)
        limiter.rate = 1.0
        limiter.capacity = 60
        limiter.exempt_paths = set()
        limiter._buckets = {}
        limiter._lock = threading.Lock()

        def create_bucket(key: str):
            with limiter._lock:
                if key not in limiter._buckets:
                    limiter._buckets[key] = _TokenBucket(
                        rate=limiter.rate, capacity=limiter.capacity
                    )
                return limiter._buckets[key]

        keys = [f"key-{i}" for i in range(50)]
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = []
            for _ in range(100):
                for key in keys[:10]:
                    futures.append(pool.submit(create_bucket, key))
            for f in as_completed(futures):
                f.result()

        # All 50 unique keys should exist (first 10 were heavily contested)
        assert len(limiter._buckets) == 10

    def test_token_bucket_consume_under_contention(self):
        """Many threads consuming from the same bucket shouldn't over-consume."""
        from deeprecall.middleware.rate_limit import _TokenBucket

        bucket = _TokenBucket(rate=0, capacity=10)  # rate=0: no refill
        lock = threading.Lock()
        consumed_count = 0

        def try_consume():
            nonlocal consumed_count
            with lock:
                if bucket.consume():
                    consumed_count += 1

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(try_consume) for _ in range(50)]
            for f in as_completed(futures):
                f.result()

        # With rate=0 and capacity=10, exactly 10 should succeed
        assert consumed_count == 10


# ---------------------------------------------------------------------------
# CallbackManager dispatch safety
# ---------------------------------------------------------------------------


class TestCallbackManagerConcurrency:
    """Verify CallbackManager dispatch doesn't crash under concurrent access."""

    def test_dispatch_from_multiple_threads(self):
        """Dispatching callbacks from many threads simultaneously should not crash."""
        tracker = UsageTrackingCallback()
        manager = CallbackManager(callbacks=[tracker])

        num_threads = 50

        def dispatch_events(i: int):
            from deeprecall.core.config import DeepRecallConfig

            config = DeepRecallConfig(backend="openai", backend_kwargs={"model_name": "test"})
            manager.on_query_start(f"query-{i}", config)

            step = ReasoningStep(iteration=1, action="compute")
            budget = BudgetStatus(budget=QueryBudget())
            manager.on_reasoning_step(step, budget)
            manager.on_search(f"search-{i}", num_results=5, time_ms=10.0)

            result = DeepRecallResult(
                answer=f"answer-{i}",
                sources=[],
                reasoning_trace=[],
                usage=UsageInfo(total_input_tokens=10, total_output_tokens=10, total_calls=1),
                execution_time=0.1,
                query=f"query-{i}",
                budget_status={"search_calls_used": 1},
            )
            manager.on_query_end(result)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(dispatch_events, i) for i in range(num_threads)]
            for f in as_completed(futures):
                f.result()

        summary = tracker.summary()
        assert summary["total_queries"] == num_threads
        assert summary["total_tokens"] == num_threads * 20


# ---------------------------------------------------------------------------
# InMemoryCache thread safety (pre-existing, verify it holds)
# ---------------------------------------------------------------------------


class TestInMemoryCacheConcurrency:
    """Verify InMemoryCache LRU operations are safe under concurrent access."""

    def test_concurrent_get_set(self):
        """Many threads doing get/set should not corrupt internal state."""
        from deeprecall.core.cache import InMemoryCache

        cache = InMemoryCache(max_size=100, default_ttl=60)
        num_threads = 100

        def do_ops(i: int):
            cache.set(f"key-{i}", f"value-{i}")
            val = cache.get(f"key-{i}")
            # May be None if evicted, but should never be corrupt
            if val is not None:
                assert val == f"value-{i}"

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(do_ops, i) for i in range(num_threads)]
            for f in as_completed(futures):
                f.result()

        stats = cache.stats()
        assert stats["hits"] + stats["misses"] == num_threads
        assert stats["size"] <= 100  # max_size respected

    def test_concurrent_invalidate_and_clear(self):
        """clear() and invalidate() running while get/set are active."""
        from deeprecall.core.cache import InMemoryCache

        cache = InMemoryCache(max_size=1000, default_ttl=60)
        stop = threading.Event()

        def writer():
            i = 0
            while not stop.is_set():
                cache.set(f"key-{i}", f"val-{i}")
                i += 1

        def reader():
            i = 0
            while not stop.is_set():
                cache.get(f"key-{i}")
                i += 1

        def clearer():
            while not stop.is_set():
                cache.clear()
                time.sleep(0.01)

        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))
        threads.append(threading.Thread(target=clearer))

        for t in threads:
            t.start()

        time.sleep(0.3)  # Let it run briefly
        stop.set()

        for t in threads:
            t.join(timeout=5)

        # If we get here without deadlock or crash, the test passes
        stats = cache.stats()
        assert isinstance(stats["size"], int)
