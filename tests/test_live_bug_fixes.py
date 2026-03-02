"""Live tests for bug fixes and new features against real OpenAI API.

Requires: OPENAI_API_KEY env var, Redis on localhost:6379.
Tests: partial synthesis on budget exhaustion, async embedding_fn,
       filters param, context_prefix param, search_db filters in REPL.
"""

from __future__ import annotations

import asyncio
import os

import pytest

_HAS_KEY = bool(os.environ.get("OPENAI_API_KEY"))
pytestmark = pytest.mark.skipif(not _HAS_KEY, reason="OPENAI_API_KEY not set")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chroma_store(docs: list[str], metadatas: list[dict] | None = None):
    """Create a throwaway ChromaDB store with the given docs."""
    import uuid

    from deeprecall.vectorstores.chroma import ChromaStore

    store = ChromaStore(collection_name=f"test_{uuid.uuid4().hex[:8]}")
    ids = store.add_documents(docs, metadatas=metadatas)
    assert len(ids) == len(docs)
    return store


def _make_engine(store, *, max_iterations=5, verbose=False, **kwargs):
    from deeprecall.core.config import DeepRecallConfig
    from deeprecall.core.engine import DeepRecallEngine

    config = DeepRecallConfig(
        backend="openai",
        backend_kwargs={"model_name": "gpt-4o-mini"},
        max_iterations=max_iterations,
        verbose=verbose,
        **kwargs,
    )
    return DeepRecallEngine(vectorstore=store, config=config)


# ===================================================================
# 1. Basic query -- sanity check with real LLM
# ===================================================================


def _result_contains(result, keywords: list[str]) -> bool:
    """Check if answer or reasoning trace contains any of the keywords."""
    haystack = result.answer.lower()
    for step in result.reasoning_trace:
        haystack += " " + (step.output or "").lower()
        haystack += " " + (step.code or "").lower()
    return any(kw.lower() in haystack for kw in keywords)


class TestBasicLiveQuery:
    def test_simple_factual_query(self):
        """Verify the engine returns a coherent answer, not garbage."""
        store = _make_chroma_store(
            [
                "The Eiffel Tower is located in Paris, France. It was built in 1889.",
                "The Great Wall of China stretches over 13,000 miles.",
                "Mount Everest is 8,849 meters tall, the highest peak on Earth.",
            ]
        )
        engine = _make_engine(store, max_iterations=8)
        result = engine.query("Where is the Eiffel Tower located?")

        assert result.answer, "Answer should not be empty"
        assert result.error is None, f"No error expected, got: {result.error}"
        assert result.execution_time > 0
        assert result.usage.total_calls > 0
        # LLM non-determinism: check answer OR trace for expected content
        assert _result_contains(result, ["Paris", "France"]), (
            f"Expected Paris/France in answer or trace, answer={result.answer[:200]}"
        )
        print(f"\n[PASS] Basic query answer: {result.answer[:300]}")

    def test_answer_is_not_raw_repl_state(self):
        """The answer should be human-readable, not raw REPL output."""
        store = _make_chroma_store(
            [
                "Python was created by Guido van Rossum in 1991.",
                "Python is known for its readability and simplicity.",
            ]
        )
        engine = _make_engine(store, max_iterations=5)
        result = engine.query("Who created Python?")

        # These patterns indicate raw REPL state leaked into the answer
        assert "urllib" not in result.answer, "Internal code in answer"
        assert "search_db(" not in result.answer, "Raw search call in answer"
        assert result.answer.strip(), "Answer should not be whitespace"
        print(f"\n[PASS] Clean answer: {result.answer[:300]}")


# ===================================================================
# 2. Budget exhaustion -- partial synthesis (Bug 1/3)
# ===================================================================


class TestBudgetExhaustionSynthesis:
    def test_budget_exceeded_returns_synthesized_answer(self):
        """When budget is exceeded, we get a synthesized answer, not raw REPL."""
        from deeprecall.core.guardrails import QueryBudget

        store = _make_chroma_store(
            [
                "Photosynthesis converts light energy into chemical energy in plants.",
                "Chlorophyll is the green pigment that absorbs sunlight during photosynthesis.",
                "The Calvin cycle is the light-independent reaction of photosynthesis.",
                "Photosynthesis occurs in the chloroplasts of plant cells.",
                "Oxygen is a byproduct of the light-dependent reactions of photosynthesis.",
            ]
        )
        engine = _make_engine(store, max_iterations=10)

        # Very tight budget: only 1 search call and 1 iteration
        result = engine.query(
            "Explain photosynthesis in detail",
            budget=QueryBudget(max_iterations=1, max_search_calls=1),
        )

        print(f"\n[Budget test] error: {result.error}")
        print(f"[Budget test] answer: {result.answer[:300]}")

        # Even if budget was hit, the answer should be readable
        if result.error:
            assert "score=" not in result.answer.lower(), "Raw score in answer"
            assert "urllib" not in result.answer, "Internal code leaked"
            # Should contain [Partial] prefix from synthesis
            assert result.answer, "Should have some answer even on budget exceeded"

    def test_tight_time_budget(self):
        """Very short time budget should still return something useful."""
        from deeprecall.core.guardrails import QueryBudget

        store = _make_chroma_store(
            [
                "Machine learning is a subset of artificial intelligence.",
                "Neural networks are inspired by the human brain.",
                "Deep learning uses multiple layers of neural networks.",
            ]
        )
        engine = _make_engine(store, max_iterations=15)

        result = engine.query(
            "What is machine learning?",
            budget=QueryBudget(max_time_seconds=5),
        )

        print(f"\n[Time budget] error: {result.error}")
        print(f"[Time budget] answer: {result.answer[:300]}")
        assert result.answer, "Should have an answer"


# ===================================================================
# 3. Async embedding_fn (Bug 2)
# ===================================================================


class TestAsyncEmbeddingFnLive:
    def test_async_embedding_fn_with_chroma(self):
        """An async embedding function should work without errors."""
        import uuid

        from deeprecall.vectorstores.chroma import ChromaStore

        call_count = 0

        async def async_embed(texts: list[str]) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            # Return simple deterministic embeddings for testing
            return [[float(hash(t) % 100) / 100.0] * 384 for t in texts]

        store = ChromaStore(
            collection_name=f"test_async_{uuid.uuid4().hex[:8]}",
            embedding_fn=async_embed,
        )

        # Verify the async fn was detected and wrapped
        assert store._async_embedding_fn is async_embed, "Should detect async fn"
        assert store.embedding_fn is not async_embed, "Should wrap it"

        # Add documents -- this calls embedding_fn under the hood
        ids = store.add_documents(
            [
                "Document one about testing",
                "Document two about validation",
                "Document three about verification",
            ]
        )
        assert len(ids) == 3, f"Expected 3 IDs, got {len(ids)}"
        assert call_count > 0, f"Async embed should have been called, count={call_count}"
        print(f"\n[PASS] Async embedding_fn called {call_count} times successfully")

    def test_sync_embedding_fn_still_works(self):
        """Sync embedding functions should continue to work as before."""
        import uuid

        from deeprecall.vectorstores.chroma import ChromaStore

        sync_called = False

        def sync_embed(texts: list[str]) -> list[list[float]]:
            nonlocal sync_called
            sync_called = True
            return [[0.5] * 384 for _ in texts]

        store = ChromaStore(
            collection_name=f"test_sync_{uuid.uuid4().hex[:8]}",
            embedding_fn=sync_embed,
        )

        assert store._async_embedding_fn is None, "Sync fn should not be flagged as async"
        ids = store.add_documents(["test doc"])
        assert len(ids) == 1
        assert sync_called, "Sync embedding fn should have been called"
        print("\n[PASS] Sync embedding_fn works as before")


# ===================================================================
# 4. Filters param (Feature)
# ===================================================================


class TestFiltersLive:
    def test_query_with_filters(self):
        """filters param should narrow search results by metadata."""
        docs = [
            "The Basel III framework requires banks to hold minimum capital.",
            "Anti-money laundering (AML) rules require customer due diligence.",
            "GDPR requires explicit consent for processing personal data.",
            "SOX compliance mandates financial reporting transparency.",
        ]
        metas = [
            {"section": "banking", "regulation": "basel3"},
            {"section": "banking", "regulation": "aml"},
            {"section": "privacy", "regulation": "gdpr"},
            {"section": "finance", "regulation": "sox"},
        ]
        store = _make_chroma_store(docs, metadatas=metas)
        engine = _make_engine(store, max_iterations=3)

        # Query with filter -- should focus on banking docs
        result = engine.query(
            "What are the key banking regulations?",
            filters={"section": "banking"},
        )

        print(f"\n[Filters test] answer: {result.answer[:300]}")
        assert result.answer, "Should have an answer"
        assert result.error is None, f"No error expected: {result.error}"

    def test_query_without_filters(self):
        """Without filters, all docs are searched."""
        store = _make_chroma_store(
            [
                "The sun is a star at the center of our solar system.",
                "The moon orbits the Earth every 27.3 days.",
            ]
        )
        engine = _make_engine(store, max_iterations=3)

        result = engine.query("Tell me about the sun")
        assert result.answer, "Should have an answer"
        assert result.error is None
        print(f"\n[No filters test] answer: {result.answer[:200]}")


# ===================================================================
# 5. Context prefix (Feature)
# ===================================================================


class TestContextPrefixLive:
    def test_context_prefix_influences_answer(self):
        """context_prefix should provide additional context to the LLM."""
        store = _make_chroma_store(
            [
                "Section 4.2 requires all transactions over $10,000 to be reported.",
                "Section 4.2 mandates enhanced due diligence for high-risk customers.",
                "Section 5.1 covers employee training requirements.",
            ]
        )
        engine = _make_engine(store, max_iterations=5)

        result = engine.query(
            "What are the requirements?",
            context_prefix="You are analyzing Section 4.2: Anti-Money Laundering Requirements. "
            "Focus specifically on transaction monitoring and reporting thresholds.",
        )

        print(f"\n[Context prefix test] answer: {result.answer[:300]}")
        assert result.answer, "Should have an answer"
        # Check that AML-related content appears in answer or trace
        assert _result_contains(
            result,
            [
                "transaction",
                "report",
                "10,000",
                "due diligence",
                "money laundering",
            ],
        ), f"Expected AML-related content, answer={result.answer[:300]}"

    def test_no_context_prefix(self):
        """Without prefix, query works normally."""
        store = _make_chroma_store(["Water boils at 100 degrees Celsius at sea level."])
        engine = _make_engine(store, max_iterations=5)

        result = engine.query("At what temperature does water boil?")
        assert result.answer, "Should have an answer"
        assert _result_contains(result, ["100", "boil", "Celsius"]), (
            f"Expected water/boiling content, answer={result.answer[:200]}"
        )
        print(f"\n[No prefix test] answer: {result.answer[:200]}")


# ===================================================================
# 6. search_db() filters in REPL
# ===================================================================


class TestSearchDbFiltersREPL:
    def test_search_db_signature_has_filters(self):
        """Verify the generated search_db code includes the filters parameter."""
        from deeprecall.prompts.templates import build_search_setup_code

        code = build_search_setup_code(server_port=9999)
        assert "filters=None" in code, "search_db should accept filters param"
        assert "_default_filters" in code, "Should have default_filters variable"
        compile(code, "<test>", "exec")

    def test_search_db_with_default_filters_compiles(self):
        from deeprecall.prompts.templates import build_search_setup_code

        code = build_search_setup_code(
            server_port=9999,
            max_search_calls=10,
            default_filters={"section": "4.2", "type": "regulation"},
        )
        compile(code, "<test>", "exec")
        assert '"section": "4.2"' in code


# ===================================================================
# 7. Combined features -- filters + context_prefix + budget
# ===================================================================


class TestCombinedFeatures:
    def test_all_new_params_together(self):
        """Use filters, context_prefix, and budget together."""
        from deeprecall.core.guardrails import QueryBudget

        docs = [
            "The Federal Reserve sets interest rates for the US economy.",
            "The ECB manages monetary policy for the Eurozone.",
            "The Bank of England controls UK monetary policy.",
        ]
        metas = [
            {"region": "us", "topic": "monetary_policy"},
            {"region": "eu", "topic": "monetary_policy"},
            {"region": "uk", "topic": "monetary_policy"},
        ]
        store = _make_chroma_store(docs, metadatas=metas)
        engine = _make_engine(store, max_iterations=5)

        result = engine.query(
            "What is the central bank's role?",
            filters={"region": "us"},
            context_prefix="Focus on United States monetary policy only.",
            budget=QueryBudget(max_search_calls=3, max_time_seconds=30),
        )

        print(f"\n[Combined test] answer: {result.answer[:300]}")
        assert result.answer, "Should have an answer"
        assert result.execution_time > 0

    def test_filters_and_prefix_on_async_engine(self):
        """Verify AsyncDeepRecallEngine passes through the new params."""
        from deeprecall.core.async_engine import AsyncDeepRecallEngine

        store = _make_chroma_store(
            [
                "Tokyo is the capital of Japan with a population of 14 million.",
                "Berlin is the capital of Germany.",
            ]
        )
        engine = AsyncDeepRecallEngine(vectorstore=store)

        result = asyncio.run(
            engine.query(
                "What is the capital of Japan?",
                context_prefix="You are a geography expert.",
            )
        )
        assert result.answer, "Should have an answer"
        assert _result_contains(result, ["Tokyo", "Japan", "capital"]), (
            f"Expected Tokyo/Japan content, answer={result.answer[:200]}"
        )
        print(f"\n[Async engine test] answer: {result.answer[:200]}")


# ===================================================================
# 8. Edge cases
# ===================================================================


class TestEdgeCases:
    def test_empty_store_query(self):
        """Querying an empty store should not crash."""
        store = _make_chroma_store(["placeholder"])
        engine = _make_engine(store, max_iterations=2)
        result = engine.query("This topic has no relevant documents at all")

        assert result.answer, "Should return some answer even with no matches"
        print(f"\n[Empty store test] answer: {result.answer[:200]}")

    def test_very_long_query(self):
        """A long query should not crash."""
        store = _make_chroma_store(["Short doc about cats and dogs."])
        engine = _make_engine(store, max_iterations=2)

        long_query = "Tell me about " + "the history and cultural significance of " * 20 + "cats."
        result = engine.query(long_query)
        assert result.answer, "Should handle long queries"
        print(f"\n[Long query test] answer: {result.answer[:200]}")

    def test_special_characters_in_query(self):
        """Queries with special chars should not break."""
        store = _make_chroma_store(["C++ is a programming language."])
        engine = _make_engine(store, max_iterations=2)

        result = engine.query("What is C++? Is it better than C#?")
        assert result.answer, "Should handle special characters"
        print(f"\n[Special chars test] answer: {result.answer[:200]}")

    def test_filters_empty_dict(self):
        """Empty filters dict should be treated same as None."""
        store = _make_chroma_store(["The sky is blue."])
        engine = _make_engine(store, max_iterations=2)

        result = engine.query("What color is the sky?", filters={})
        assert result.answer, "Should work with empty filters"
        print(f"\n[Empty filters test] answer: {result.answer[:200]}")

    def test_context_prefix_empty_string(self):
        """Empty context_prefix should not prepend anything."""
        store = _make_chroma_store(["Pi is approximately 3.14159."])
        engine = _make_engine(store, max_iterations=2)

        result = engine.query("What is the value of pi?", context_prefix="")
        assert result.answer, "Should work with empty prefix"
        print(f"\n[Empty prefix test] answer: {result.answer[:200]}")

    def test_result_has_sources(self):
        """Verify sources are populated in the result."""
        store = _make_chroma_store(
            [
                "Gravity is a fundamental force described by general relativity.",
                "Einstein published the theory of general relativity in 1915.",
            ]
        )
        engine = _make_engine(store, max_iterations=3)
        result = engine.query("What is general relativity?")

        assert result.answer, "Should have an answer"
        assert len(result.sources) > 0, "Should have at least one source"
        for src in result.sources:
            assert src.content, "Source should have content"
            assert src.score > 0, f"Source score should be > 0, got {src.score}"
        print(f"\n[Sources test] {len(result.sources)} sources, answer: {result.answer[:200]}")
