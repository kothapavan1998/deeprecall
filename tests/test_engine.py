"""Tests for the DeepRecall engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from deeprecall.core.config import DeepRecallConfig
from deeprecall.core.engine import DeepRecallEngine
from deeprecall.core.search_server import SearchServer
from deeprecall.core.types import DeepRecallResult, ReasoningStep, Source, UsageInfo

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestDeepRecallConfig:
    def test_default_config(self):
        config = DeepRecallConfig()
        assert config.backend == "openai"
        assert config.max_iterations == 15
        assert config.max_depth == 1
        assert config.top_k == 5
        assert config.verbose is False

    def test_custom_config(self):
        config = DeepRecallConfig(
            backend="anthropic",
            backend_kwargs={"model_name": "claude-3-sonnet"},
            max_iterations=20,
            verbose=True,
        )
        assert config.backend == "anthropic"
        assert config.max_iterations == 20
        assert config.verbose is True

    def test_config_to_dict_excludes_api_key(self):
        config = DeepRecallConfig(backend_kwargs={"model_name": "gpt-4o", "api_key": "sk-secret"})
        d = config.to_dict()
        assert "api_key" not in d["backend_kwargs"]
        assert d["backend_kwargs"]["model_name"] == "gpt-4o"

    # -- __post_init__ validation --

    def test_max_iterations_zero_raises(self):
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            DeepRecallConfig(max_iterations=0)

    def test_max_depth_negative_raises(self):
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            DeepRecallConfig(max_depth=-1)

    def test_top_k_negative_raises(self):
        with pytest.raises(ValueError, match="top_k must be >= 0"):
            DeepRecallConfig(top_k=-1)

    def test_cache_ttl_negative_raises(self):
        with pytest.raises(ValueError, match="cache_ttl must be >= 0"):
            DeepRecallConfig(cache_ttl=-10)

    # -- to_dict with optional fields --

    def test_to_dict_with_budget(self):
        from deeprecall.core.guardrails import QueryBudget

        config = DeepRecallConfig(budget=QueryBudget(max_iterations=10))
        d = config.to_dict()
        assert d["budget"]["max_iterations"] == 10

    def test_to_dict_with_callbacks(self):
        from deeprecall.core.callbacks import UsageTrackingCallback

        config = DeepRecallConfig(callbacks=[UsageTrackingCallback()])
        d = config.to_dict()
        assert d["callbacks"] == ["UsageTrackingCallback"]

    def test_to_dict_strips_all_secret_keys(self):
        config = DeepRecallConfig(
            backend_kwargs={"model_name": "x", "api_key": "s", "token": "t", "api_secret": "a"}
        )
        d = config.to_dict()
        assert "api_key" not in d["backend_kwargs"]
        assert "token" not in d["backend_kwargs"]
        assert "api_secret" not in d["backend_kwargs"]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestDeepRecallEngine:
    def test_init_with_config(self, mock_vectorstore):
        config = DeepRecallConfig(backend="openai")
        engine = DeepRecallEngine(vectorstore=mock_vectorstore, config=config)
        assert engine.config.backend == "openai"

    def test_init_with_kwargs(self, mock_vectorstore):
        engine = DeepRecallEngine(
            vectorstore=mock_vectorstore,
            backend="anthropic",
            backend_kwargs={"model_name": "claude-3"},
            verbose=True,
        )
        assert engine.config.backend == "anthropic"
        assert engine.config.verbose is True

    def test_init_requires_vectorstore(self):
        with pytest.raises(ValueError, match="vectorstore is required"):
            DeepRecallEngine(vectorstore=None)

    def test_add_documents(self, mock_vectorstore):
        engine = DeepRecallEngine(vectorstore=mock_vectorstore)
        ids = engine.add_documents(
            documents=["New doc 1", "New doc 2"],
            metadatas=[{"source": "test"}, {"source": "test"}],
        )
        assert len(ids) == 2
        assert mock_vectorstore.count() == 7  # 5 existing + 2 new

    def test_repr(self, mock_vectorstore):
        engine = DeepRecallEngine(vectorstore=mock_vectorstore)
        r = repr(engine)
        assert "DeepRecall" in r
        assert "MockVectorStore" in r

    def test_context_manager_calls_close(self, mock_vectorstore):
        engine = DeepRecallEngine(vectorstore=mock_vectorstore)
        with patch.object(engine, "close") as mock_close:
            with engine:
                pass
            mock_close.assert_called_once()

    def test_close_stops_search_server(self, mock_vectorstore):
        engine = DeepRecallEngine(vectorstore=mock_vectorstore)
        mock_server = MagicMock()
        engine._search_server = mock_server
        engine.close()
        mock_server.stop.assert_called_once()
        assert engine._search_server is None

    # -- query() with mocked RLM --

    def test_query_empty_string_raises(self, mock_vectorstore):
        engine = DeepRecallEngine(vectorstore=mock_vectorstore)
        with pytest.raises(ValueError, match="non-empty string"):
            engine.query("")

    def test_query_whitespace_only_raises(self, mock_vectorstore):
        engine = DeepRecallEngine(vectorstore=mock_vectorstore)
        with pytest.raises(ValueError, match="non-empty string"):
            engine.query("   ")

    @patch("deeprecall.core.engine.SearchServer")
    @patch("deeprecall.core.engine.RLM", create=True)
    def test_query_returns_result(self, MockRLM, MockServer, mock_vectorstore, mock_rlm_result):
        """Full query flow with mocked RLM and search server."""
        # Arrange: mock the RLM class
        import sys
        from unittest.mock import MagicMock as MM

        # Ensure rlm module is importable
        mock_rlm_module = MM()
        mock_rlm_cls = MM()
        mock_rlm_instance = MM()
        mock_rlm_instance.completion.return_value = mock_rlm_result
        mock_rlm_cls.return_value = mock_rlm_instance
        mock_rlm_module.RLM = mock_rlm_cls

        mock_server_inst = MM()
        mock_server_inst.port = 9999
        mock_server_inst.get_accessed_sources.return_value = [
            Source(content="doc1", score=0.9, id="s1"),
        ]
        MockServer.return_value = mock_server_inst

        engine = DeepRecallEngine(vectorstore=mock_vectorstore, reuse_search_server=False)

        with patch.dict(sys.modules, {"rlm": mock_rlm_module}):
            result = engine.query("What is Python?")

        assert isinstance(result, DeepRecallResult)
        assert "test answer" in result.answer.lower()
        assert result.execution_time > 0
        assert len(result.sources) == 1
        assert result.confidence is not None

    @patch("deeprecall.core.engine.SearchServer")
    @patch("deeprecall.core.engine.RLM", create=True)
    def test_query_uses_cache(self, MockRLM, MockServer, mock_vectorstore, mock_rlm_result):
        """Verify cache hit skips RLM call entirely."""
        from unittest.mock import MagicMock as MM

        mock_cache = MM()
        cached_result = DeepRecallResult(answer="Cached answer", query="What is Python?")
        mock_cache.get.return_value = cached_result

        engine = DeepRecallEngine(vectorstore=mock_vectorstore, cache=mock_cache)

        result = engine.query("What is Python?")

        assert result.answer == "Cached answer"
        # RLM should never be constructed
        MockRLM.assert_not_called()
        MockServer.assert_not_called()

    def test_query_batch_too_large_raises(self, mock_vectorstore):
        engine = DeepRecallEngine(vectorstore=mock_vectorstore)
        with pytest.raises(ValueError, match="exceeds 10,000"):
            engine.query_batch(["q"] * 10_001)

    def test_query_batch_zero_concurrency_raises(self, mock_vectorstore):
        engine = DeepRecallEngine(vectorstore=mock_vectorstore)
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            engine.query_batch(["q1"], max_concurrency=0)

    # -- _compute_confidence --

    def test_compute_confidence_no_sources(self, mock_vectorstore):
        engine = DeepRecallEngine(vectorstore=mock_vectorstore)
        assert engine._compute_confidence([]) is None

    def test_compute_confidence_zero_scores(self, mock_vectorstore):
        engine = DeepRecallEngine(vectorstore=mock_vectorstore)
        sources = [Source(content="a", score=0.0), Source(content="b", score=0.0)]
        assert engine._compute_confidence(sources) is None

    def test_compute_confidence_valid_scores(self, mock_vectorstore):
        engine = DeepRecallEngine(vectorstore=mock_vectorstore)
        sources = [
            Source(content="a", score=0.9),
            Source(content="b", score=0.8),
            Source(content="c", score=0.7),
            Source(content="d", score=0.5),
        ]
        confidence = engine._compute_confidence(sources)
        # Average of top-3 = (0.9 + 0.8 + 0.7) / 3 = 0.8
        assert confidence == 0.8

    # _build_cache_key was inlined into query(); cache key logic tested via cache integration


# ---------------------------------------------------------------------------
# Config -- all parameters coverage
# ---------------------------------------------------------------------------


class TestDeepRecallConfigAllParams:
    """Ensure every DeepRecallConfig field is tested with non-default values."""

    def test_environment_param(self):
        config = DeepRecallConfig(environment="docker")
        assert config.environment == "docker"
        assert config.to_dict()["environment"] == "docker"

    def test_environment_kwargs_param(self):
        config = DeepRecallConfig(environment_kwargs={"timeout": 60})
        assert config.environment_kwargs == {"timeout": 60}
        assert config.to_dict()["environment_kwargs"] == {"timeout": 60}

    def test_log_dir_param(self):
        config = DeepRecallConfig(log_dir="/tmp/logs")
        assert config.log_dir == "/tmp/logs"
        assert config.to_dict()["log_dir"] == "/tmp/logs"

    def test_other_backends_param(self):
        config = DeepRecallConfig(other_backends=["anthropic", "gemini"])
        assert config.other_backends == ["anthropic", "gemini"]
        assert config.to_dict()["other_backends"] == ["anthropic", "gemini"]

    def test_other_backend_kwargs_param(self):
        config = DeepRecallConfig(
            other_backend_kwargs=[
                {"model_name": "claude-3", "api_key": "secret"},
                {"model_name": "gemini-pro"},
            ]
        )
        d = config.to_dict()
        # Secrets must be stripped
        assert "api_key" not in d["other_backend_kwargs"][0]
        assert d["other_backend_kwargs"][0]["model_name"] == "claude-3"
        assert d["other_backend_kwargs"][1]["model_name"] == "gemini-pro"

    def test_reranker_in_to_dict(self):
        mock_reranker = MagicMock()
        type(mock_reranker).__name__ = "CohereReranker"
        config = DeepRecallConfig(reranker=mock_reranker)
        d = config.to_dict()
        assert d["reranker"] == "CohereReranker"

    def test_retry_in_to_dict(self):
        from deeprecall.core.retry import RetryConfig

        config = DeepRecallConfig(retry=RetryConfig(max_retries=5, base_delay=2.0))
        d = config.to_dict()
        assert d["retry"]["max_retries"] == 5
        assert d["retry"]["base_delay"] == 2.0

    def test_reuse_search_server_in_to_dict(self):
        config = DeepRecallConfig(reuse_search_server=False)
        assert config.reuse_search_server is False
        assert config.to_dict()["reuse_search_server"] is False

    def test_cache_in_to_dict(self):
        from deeprecall.core.cache import InMemoryCache

        config = DeepRecallConfig(cache=InMemoryCache(max_size=10))
        d = config.to_dict()
        assert d["cache"] == "InMemoryCache"

    def test_log_dir_creates_jsonl_callback(self):
        """Engine with log_dir should auto-create a JSONLCallback."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = DeepRecallEngine(
                vectorstore=MagicMock(count=MagicMock(return_value=0)),
                log_dir=tmpdir,
            )
            assert engine._callback_manager is not None


# ---------------------------------------------------------------------------
# SearchServer
# ---------------------------------------------------------------------------


class TestSearchServer:
    def test_start_stop(self, mock_vectorstore):
        server = SearchServer(mock_vectorstore)
        server.start()
        assert server.port > 0
        server.stop()

    def test_search_endpoint(self, mock_vectorstore):
        import json
        import urllib.request

        server = SearchServer(mock_vectorstore)
        server.start()

        try:
            data = json.dumps({"query": "python programming", "top_k": 3}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{server.port}/search",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                results = json.loads(resp.read())

            assert isinstance(results, list)
            assert len(results) <= 3
            assert all("content" in r for r in results)
        finally:
            server.stop()

    def test_get_accessed_sources(self, mock_vectorstore):
        import json
        import urllib.request

        server = SearchServer(mock_vectorstore)
        server.start()

        try:
            data = json.dumps({"query": "python", "top_k": 2}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{server.port}/search",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)

            sources = server.get_accessed_sources()
            assert len(sources) > 0
        finally:
            server.stop()

    def test_reset_sources(self, mock_vectorstore):
        server = SearchServer(mock_vectorstore)
        server.reset_sources()
        assert server.get_accessed_sources() == []


# ---------------------------------------------------------------------------
# DeepRecallResult
# ---------------------------------------------------------------------------


class TestDeepRecallResult:
    def test_result_to_dict(self):
        result = DeepRecallResult(
            answer="Test answer",
            query="Test query",
            execution_time=1.5,
        )
        d = result.to_dict()
        assert d["answer"] == "Test answer"
        assert d["query"] == "Test query"
        assert d["execution_time"] == 1.5

    def test_from_dict_round_trip(self):
        """Verify to_dict → from_dict produces an equivalent result."""
        original = DeepRecallResult(
            answer="Paris is the capital.",
            sources=[
                Source(content="Geography doc", score=0.95, id="s1", metadata={"type": "geo"})
            ],
            reasoning_trace=[
                ReasoningStep(
                    iteration=1,
                    action="search_and_reasoning",
                    code="search_db('capital of France')",
                    output="Found: Paris",
                    searches=[{"query": "capital of France", "has_results": True}],
                    sub_llm_calls=1,
                    iteration_time=0.5,
                )
            ],
            usage=UsageInfo(
                total_input_tokens=500,
                total_output_tokens=200,
                total_calls=3,
                model_breakdown={
                    "gpt-4o-mini": {"input_tokens": 500, "output_tokens": 200, "calls": 3}
                },
            ),
            execution_time=2.5,
            query="What is the capital of France?",
            budget_status={"iterations_used": 1, "budget_exceeded": False},
            error=None,
            confidence=0.95,
        )

        data = original.to_dict()
        restored = DeepRecallResult.from_dict(data)

        assert restored.answer == original.answer
        assert restored.query == original.query
        assert restored.execution_time == original.execution_time
        assert restored.confidence == original.confidence
        assert restored.error is None

        assert len(restored.sources) == 1
        assert restored.sources[0].content == "Geography doc"
        assert restored.sources[0].score == 0.95
        assert restored.sources[0].metadata == {"type": "geo"}

        assert len(restored.reasoning_trace) == 1
        assert restored.reasoning_trace[0].action == "search_and_reasoning"
        assert restored.reasoning_trace[0].sub_llm_calls == 1

        assert restored.usage.total_input_tokens == 500
        assert restored.usage.total_calls == 3
        assert "gpt-4o-mini" in restored.usage.model_breakdown

    def test_from_dict_minimal(self):
        """from_dict handles an empty/minimal dict gracefully."""
        result = DeepRecallResult.from_dict({})
        assert result.answer == ""
        assert result.sources == []
        assert result.reasoning_trace == []
        assert result.usage.total_input_tokens == 0
