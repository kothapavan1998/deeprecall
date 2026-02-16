"""Tests for the DeepRecall CLI."""

from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner

from deeprecall.cli import _build_vectorstore, _default_persist_dir, init, main, status


class TestInitCommand:
    def test_creates_config_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(init, ["--path", "test.toml"])
            assert result.exit_code == 0
            assert "Created config file" in result.output
            assert os.path.exists("test.toml")
            with open("test.toml") as f:
                content = f.read()
            assert "[engine]" in content
            assert "[vectorstore]" in content
            assert "[budget]" in content
            assert "[server]" in content
            assert "[cache]" in content

    def test_skips_existing_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("test.toml", "w") as f:
                f.write("existing content")
            result = runner.invoke(init, ["--path", "test.toml"])
            assert result.exit_code == 0
            assert "already exists" in result.output
            with open("test.toml") as f:
                assert f.read() == "existing content"


class TestStatusCommand:
    def test_shows_version_info(self):
        runner = CliRunner()
        result = runner.invoke(status)
        assert result.exit_code == 0
        assert "Python:" in result.output
        assert "DeepRecall:" in result.output
        assert "RLM (rlms):" in result.output
        assert "Extras:" in result.output


class TestDefaultPersistDir:
    def test_user_override_takes_precedence(self):
        assert _default_persist_dir("chroma", "/custom/dir") == "/custom/dir"

    def test_chroma_default(self):
        assert _default_persist_dir("chroma", None) == "./chroma_db"

    def test_faiss_default(self):
        assert _default_persist_dir("faiss", None) == "./faiss_index"

    def test_other_stores_return_none(self):
        assert _default_persist_dir("milvus", None) is None
        assert _default_persist_dir("qdrant", None) is None


class TestBuildVectorstore:
    def test_chroma_store(self):
        store = _build_vectorstore("chroma", "test_collection")
        from deeprecall.vectorstores.chroma import ChromaStore

        assert isinstance(store, ChromaStore)

    def test_unknown_store_raises(self):
        with pytest.raises(ValueError, match="Unknown vectorstore"):
            _build_vectorstore("invalid_store", "col")


class TestIngestCommand:
    def test_ingest_reads_files(self):
        """Test that ingest correctly discovers and reads text files."""
        from unittest.mock import MagicMock, patch

        runner = CliRunner()
        with runner.isolated_filesystem():
            os.makedirs("docs")
            with open("docs/file1.txt", "w") as f:
                f.write("Document one content about Python programming.")
            with open("docs/file2.md", "w") as f:
                f.write("Document two content about machine learning.")
            with open("docs/skip.bin", "w") as f:
                f.write("binary data")

            mock_store = MagicMock()
            mock_store.add_documents.return_value = ["id-1", "id-2"]
            mock_store.count.return_value = 2

            with patch("deeprecall.cli._build_vectorstore", return_value=mock_store):
                result = runner.invoke(
                    main,
                    ["ingest", "--path", "docs", "--vectorstore", "chroma", "--collection", "t"],
                )

            assert result.exit_code == 0
            assert "Ingested 2 documents" in result.output
            # Verify only .txt and .md were passed (not .bin)
            call_args = mock_store.add_documents.call_args
            assert len(call_args.kwargs["documents"]) == 2

    def test_ingest_empty_dir(self):
        """Test that ingest handles empty directories gracefully."""
        from unittest.mock import MagicMock, patch

        runner = CliRunner()
        with runner.isolated_filesystem():
            os.makedirs("empty_docs")

            mock_store = MagicMock()
            with patch("deeprecall.cli._build_vectorstore", return_value=mock_store):
                result = runner.invoke(
                    main,
                    [
                        "ingest",
                        "--path",
                        "empty_docs",
                        "--vectorstore",
                        "chroma",
                        "--collection",
                        "t",
                    ],
                )

            assert result.exit_code == 0
            assert "No documents found" in result.output


class TestDeleteCommand:
    def test_delete_by_ids(self):
        from unittest.mock import MagicMock, patch

        runner = CliRunner()
        mock_store = MagicMock()
        mock_store.count.return_value = 0

        with patch("deeprecall.cli._build_vectorstore", return_value=mock_store):
            result = runner.invoke(
                main,
                ["delete", "--vectorstore", "chroma", "--collection", "t", "id-1"],
            )

        assert result.exit_code == 0
        assert "Deleted 1 document(s)" in result.output
        mock_store.delete.assert_called_once_with(ids=["id-1"])


class TestQueryCommand:
    def test_query_runs_engine_and_prints_result(self):
        from unittest.mock import MagicMock, patch

        from deeprecall.core.types import DeepRecallResult, UsageInfo

        mock_result = DeepRecallResult(
            answer="Paris is the capital of France.",
            sources=[MagicMock()],
            reasoning_trace=[MagicMock()],
            usage=UsageInfo(total_input_tokens=100, total_output_tokens=50, total_calls=2),
            execution_time=1.5,
            query="What is the capital of France?",
            confidence=0.9,
        )

        runner = CliRunner()
        mock_engine = MagicMock()
        mock_engine.query.return_value = mock_result

        with patch("deeprecall.cli._build_engine", return_value=mock_engine):
            result = runner.invoke(
                main,
                ["query", "What is the capital of France?", "--vectorstore", "chroma"],
            )

        assert result.exit_code == 0
        assert "Paris is the capital of France." in result.output
        assert "Sources: 1" in result.output
        assert "Steps: 1" in result.output
        assert "Confidence: 0.90" in result.output
        mock_engine.query.assert_called_once()

    def test_query_with_budget_flags(self):
        from unittest.mock import MagicMock, patch

        from deeprecall.core.types import DeepRecallResult, UsageInfo

        mock_result = DeepRecallResult(
            answer="Answer",
            usage=UsageInfo(),
            execution_time=0.5,
            budget_status={"budget_exceeded": True, "exceeded_reason": "Time limit"},
            error="Budget exceeded",
        )

        runner = CliRunner()
        mock_engine = MagicMock()
        mock_engine.query.return_value = mock_result

        with patch("deeprecall.cli._build_engine", return_value=mock_engine):
            result = runner.invoke(
                main,
                [
                    "query",
                    "test",
                    "--max-searches",
                    "5",
                    "--max-tokens",
                    "1000",
                    "--max-time",
                    "10.0",
                ],
            )

        assert result.exit_code == 0
        assert "Warning: Budget exceeded" in result.output
        assert "Budget: Time limit" in result.output
        assert mock_engine.query.called


class TestServeCommand:
    def test_serve_builds_app_and_runs(self):
        from unittest.mock import MagicMock, patch

        runner = CliRunner()
        mock_engine = MagicMock()
        mock_app = MagicMock()
        mock_uvicorn = MagicMock()

        with (
            patch("deeprecall.cli._build_engine", return_value=mock_engine),
            patch("deeprecall.adapters.openai_server.create_app", return_value=mock_app),
            patch("deeprecall.cli.uvicorn", mock_uvicorn, create=True),
            patch.dict("sys.modules", {"uvicorn": mock_uvicorn}),
        ):
            result = runner.invoke(
                main,
                [
                    "serve",
                    "--vectorstore",
                    "chroma",
                    "--port",
                    "9999",
                    "--api-keys",
                    "key1,key2",
                    "--rate-limit",
                    "30",
                ],
            )

        assert result.exit_code == 0
        assert "9999" in result.output
        assert "2 API key(s)" in result.output
        assert "30 req/min" in result.output


class TestBuildEngine:
    def test_build_engine_sets_api_key_from_env(self):
        from unittest.mock import MagicMock, patch

        mock_store = MagicMock()
        with (
            patch("deeprecall.cli._build_vectorstore", return_value=mock_store),
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}),
        ):
            from deeprecall.cli import _build_engine

            engine = _build_engine("chroma", "test", None, "openai", "gpt-4o-mini")
            assert engine.config.backend_kwargs["api_key"] == "sk-test123"
            assert engine.config.backend_kwargs["model_name"] == "gpt-4o-mini"

    def test_build_engine_anthropic_key(self):
        from unittest.mock import MagicMock, patch

        mock_store = MagicMock()
        with (
            patch("deeprecall.cli._build_vectorstore", return_value=mock_store),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "ant-key"}),
        ):
            from deeprecall.cli import _build_engine

            engine = _build_engine("chroma", "test", None, "anthropic", "claude-3")
            assert engine.config.backend_kwargs["api_key"] == "ant-key"


class TestBenchmarkValidation:
    def test_invalid_json_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("queries.json", "w") as f:
                f.write("not valid json{{{")

            result = runner.invoke(
                main,
                ["benchmark", "--queries", "queries.json", "--vectorstore", "chroma"],
            )
            assert result.exit_code != 0
            assert "invalid JSON" in result.output

    def test_non_array_json(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("queries.json", "w") as f:
                json.dump({"key": "value"}, f)

            result = runner.invoke(
                main,
                ["benchmark", "--queries", "queries.json", "--vectorstore", "chroma"],
            )
            assert result.exit_code != 0
            assert "JSON array" in result.output

    def test_oversized_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create a file larger than 10 MB
            with open("queries.json", "w") as f:
                f.write("[" + ",".join(['"x"'] * 3_000_000) + "]")

            result = runner.invoke(
                main,
                ["benchmark", "--queries", "queries.json", "--vectorstore", "chroma"],
            )
            assert result.exit_code != 0
            assert "too large" in result.output
