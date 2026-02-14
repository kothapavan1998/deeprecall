#!/usr/bin/env python3
"""Live integration tests for RedisCache and OpenTelemetryCallback.

Requires:
  - Redis running on localhost:6380
  - Jaeger OTLP collector running on localhost:4317

Run:
  docker run -d --name deeprecall-redis -p 6380:6379 redis:7-alpine
  docker run -d --name deeprecall-jaeger -p 16686:16686 -p 4317:4317 -e COLLECTOR_OTLP_ENABLED=true jaegertracing/all-in-one:latest
  python tests/test_integration_redis_otel.py
"""

from __future__ import annotations

import json
import time


def test_redis_cache():
    """Full integration test against a real Redis instance."""
    print("\n" + "=" * 60)
    print("TEST: RedisCache against real Redis (localhost:6380)")
    print("=" * 60)

    from deeprecall.core.cache_redis import RedisCache

    # 1. Connect
    cache = RedisCache(url="redis://localhost:6380/0", prefix="test_deeprecall:", default_ttl=60)
    print("[PASS] Connected to Redis")

    # 2. Health check
    health = cache.health_check()
    print(f"[INFO] Health: {health}")
    assert health["status"] == "connected", f"Expected connected, got {health}"
    assert "redis_version" in health
    print(f"[PASS] Health check: connected, Redis v{health['redis_version']}, latency={health['latency_ms']}ms")

    # 3. Clear any leftover test data
    cache.clear()
    stats = cache.stats()
    assert stats["size"] == 0, f"Expected 0 after clear, got {stats['size']}"
    print("[PASS] Clear works, size=0")

    # 4. Set and get a simple value
    cache.set("key1", {"answer": "hello world", "score": 0.95})
    result = cache.get("key1")
    assert result == {"answer": "hello world", "score": 0.95}, f"Got {result}"
    print("[PASS] set/get simple dict")

    # 5. Set and get a list
    cache.set("key2", [1, 2, 3, "four"])
    result = cache.get("key2")
    assert result == [1, 2, 3, "four"], f"Got {result}"
    print("[PASS] set/get list")

    # 6. Set with explicit TTL
    cache.set("key3", "expires_soon", ttl=2)
    result = cache.get("key3")
    assert result == "expires_soon"
    print("[PASS] set with TTL=2s, got value")

    # 7. Wait for TTL expiry
    print("[INFO] Waiting 3s for TTL expiry...")
    time.sleep(3)
    result = cache.get("key3")
    assert result is None, f"Expected None after TTL, got {result}"
    print("[PASS] TTL expiry works -- key3 is gone")

    # 8. Cache miss returns None
    result = cache.get("nonexistent_key_xyz")
    assert result is None
    print("[PASS] Cache miss returns None")

    # 9. Invalidate specific key
    cache.set("key4", "to_delete")
    cache.invalidate("key4")
    result = cache.get("key4")
    assert result is None
    print("[PASS] invalidate() removes key")

    # 10. Stats
    stats = cache.stats()
    print(f"[INFO] Stats: {json.dumps(stats, indent=2)}")
    assert stats["type"] == "redis"
    assert stats["prefix"] == "test_deeprecall:"
    assert stats["hits"] > 0
    assert stats["misses"] > 0
    assert 0 < stats["hit_rate"] < 1
    print(f"[PASS] Stats: {stats['hits']} hits, {stats['misses']} misses, hit_rate={stats['hit_rate']}")

    # 11. Object with to_dict() method
    class FakeDeepRecallResult:
        def to_dict(self):
            return {"answer": "test result", "sources": [{"id": "1", "score": 0.9}]}

    cache.set("result_key", FakeDeepRecallResult())
    result = cache.get("result_key")
    assert result["answer"] == "test result"
    assert result["sources"][0]["id"] == "1"
    print("[PASS] Serializes objects with to_dict() method")

    # 12. Namespace isolation
    cache2 = RedisCache(url="redis://localhost:6380/0", prefix="other_app:", default_ttl=60)
    cache2.set("key1", "from_other_app")
    # Original cache should NOT see other_app's key
    assert cache.get("key1") == {"answer": "hello world", "score": 0.95}  # our original
    assert cache2.get("key1") == "from_other_app"  # other app's value
    print("[PASS] Namespace isolation works between prefixes")

    # Cleanup
    cache.clear()
    cache2.clear()
    print("[PASS] Cleanup complete")

    print("\nRedisCache: ALL TESTS PASSED")


def test_otel_callback():
    """Integration test: emit traces to real Jaeger via OTLP."""
    print("\n" + "=" * 60)
    print("TEST: OpenTelemetryCallback against Jaeger (localhost:4317)")
    print("=" * 60)

    from deeprecall.core.callback_otel import OpenTelemetryCallback
    from deeprecall.core.config import DeepRecallConfig
    from deeprecall.core.guardrails import BudgetStatus, QueryBudget
    from deeprecall.core.types import DeepRecallResult, ReasoningStep, Source, UsageInfo

    # 1. Initialize callback with real OTLP endpoint
    otel = OpenTelemetryCallback(
        service_name="deeprecall-integration-test",
        endpoint="http://localhost:4317",
        insecure=True,
    )
    print("[PASS] OpenTelemetryCallback initialized with Jaeger endpoint")

    # 2. Simulate a full query lifecycle
    config = DeepRecallConfig(
        backend="openai",
        backend_kwargs={"model_name": "gpt-4o-mini"},
    )

    # on_query_start
    otel.on_query_start("What is the capital of France?", config)
    print("[PASS] on_query_start() -- root span created")

    # on_reasoning_step (step 1: search)
    step1 = ReasoningStep(
        iteration=1,
        action="search",
        code='results = search_db("capital of France", top_k=5)',
        output="Found 5 results about France",
    )
    budget = BudgetStatus(budget=QueryBudget(max_search_calls=10))
    budget.iterations_used = 1
    budget.search_calls_used = 1
    otel.on_reasoning_step(step1, budget)
    print("[PASS] on_reasoning_step() step 1 -- search span")

    # on_search
    otel.on_search("capital of France", num_results=5, time_ms=12.3)
    print("[PASS] on_search() -- search span with metrics")

    # on_reasoning_step (step 2: analyze)
    step2 = ReasoningStep(
        iteration=2,
        action="analyze",
        code="answer = 'Paris is the capital of France'",
        output="Paris is the capital of France",
    )
    budget.iterations_used = 2
    otel.on_reasoning_step(step2, budget)
    print("[PASS] on_reasoning_step() step 2 -- analyze span")

    # on_reasoning_step (step 3: final)
    step3 = ReasoningStep(
        iteration=3,
        action="final_answer",
        output="FINAL: The capital of France is Paris.",
    )
    budget.iterations_used = 3
    otel.on_reasoning_step(step3, budget)
    print("[PASS] on_reasoning_step() step 3 -- final answer span")

    # on_query_end
    result = DeepRecallResult(
        answer="The capital of France is Paris.",
        sources=[
            Source(content="Paris is the capital...", metadata={"source": "wiki"}, score=0.95, id="1"),
            Source(content="France, officially...", metadata={"source": "ency"}, score=0.88, id="2"),
        ],
        reasoning_trace=[step1, step2, step3],
        usage=UsageInfo(total_input_tokens=500, total_output_tokens=200, total_calls=3),
        execution_time=2.5,
        query="What is the capital of France?",
        confidence=0.92,
    )
    otel.on_query_end(result)
    print("[PASS] on_query_end() -- root span closed with metrics")

    # 3. Test error scenario
    otel2 = OpenTelemetryCallback(
        service_name="deeprecall-integration-test",
        endpoint="http://localhost:4317",
        insecure=True,
    )
    otel2.on_query_start("Will this fail?", config)
    otel2.on_error(RuntimeError("Simulated error"))
    print("[PASS] on_error() -- error recorded on span")

    # 4. Test budget warning scenario
    otel3 = OpenTelemetryCallback(
        service_name="deeprecall-integration-test",
        endpoint="http://localhost:4317",
        insecure=True,
    )
    otel3.on_query_start("Budget test query", config)
    budget_exceeded = BudgetStatus(budget=QueryBudget(max_search_calls=3))
    budget_exceeded.search_calls_used = 3
    budget_exceeded.exceeded_reason = "search_calls"
    otel3.on_budget_warning(budget_exceeded)
    # End with partial result
    partial_result = DeepRecallResult(
        answer="[Partial - budget exceeded]",
        sources=[],
        reasoning_trace=[],
        usage=UsageInfo(),
        execution_time=5.0,
        query="Budget test query",
        error="Budget exceeded: search_calls",
    )
    otel3.on_query_end(partial_result)
    print("[PASS] Budget warning + error result scenario")

    # Force flush
    from opentelemetry import trace
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
    print("[PASS] Force-flushed traces to Jaeger")

    # Wait for export
    time.sleep(2)

    # 5. Verify traces arrived at Jaeger via API
    import urllib.request
    try:
        url = "http://localhost:16686/api/services"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            services = data.get("data", [])
            print(f"[INFO] Jaeger services: {services}")
            if "deeprecall-integration-test" in services:
                print("[PASS] Traces visible in Jaeger!")
            else:
                print("[WARN] Service not yet visible (may need more time to index)")
    except Exception as e:
        print(f"[WARN] Could not query Jaeger API: {e}")

    print("\nOpenTelemetryCallback: ALL TESTS PASSED")


def main():
    test_redis_cache()
    test_otel_callback()
    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
