"""Tests for deeprecall.prompts.templates."""

from __future__ import annotations

from deeprecall.prompts.templates import DEEPRECALL_SYSTEM_PROMPT, build_search_setup_code


class TestSystemPrompt:
    """Tests for the DEEPRECALL_SYSTEM_PROMPT constant."""

    def test_prompt_is_non_empty_string(self):
        assert isinstance(DEEPRECALL_SYSTEM_PROMPT, str)
        assert len(DEEPRECALL_SYSTEM_PROMPT) > 100

    def test_prompt_mentions_search_db(self):
        assert "search_db" in DEEPRECALL_SYSTEM_PROMPT

    def test_prompt_mentions_final_var(self):
        assert "FINAL_VAR" in DEEPRECALL_SYSTEM_PROMPT

    def test_prompt_mentions_final(self):
        assert "FINAL(" in DEEPRECALL_SYSTEM_PROMPT

    def test_prompt_mentions_repl(self):
        assert "repl" in DEEPRECALL_SYSTEM_PROMPT.lower()

    def test_prompt_mentions_llm_query(self):
        assert "llm_query" in DEEPRECALL_SYSTEM_PROMPT

    def test_prompt_format_compatible_with_rlm(self):
        """RLM v0.1.1a calls .format(custom_tools_section=...) on the prompt."""
        result = DEEPRECALL_SYSTEM_PROMPT.format(custom_tools_section="")
        assert "search_db" in result
        assert "{custom_tools_section}" not in result

    def test_prompt_format_injects_tools(self):
        tools = "\n8. my_custom_tool() - does something"
        result = DEEPRECALL_SYSTEM_PROMPT.format(custom_tools_section=tools)
        assert "my_custom_tool()" in result


class TestBuildSearchSetupCode:
    """Tests for the build_search_setup_code function."""

    def test_injects_port(self):
        code = build_search_setup_code(server_port=12345)
        assert "12345" in code
        assert "http://127.0.0.1:12345/search" in code

    def test_code_is_valid_python(self):
        code = build_search_setup_code(server_port=8080)
        # Should compile without syntax errors
        compile(code, "<test>", "exec")

    def test_defines_search_db_function(self):
        code = build_search_setup_code(server_port=8080)
        assert "def search_db(query, top_k=5, filters=None):" in code

    def test_no_budget_by_default(self):
        code = build_search_setup_code(server_port=8080)
        assert "_max_search_calls" not in code
        assert "_search_call_count" not in code

    def test_budget_code_included_when_max_search_calls_set(self):
        code = build_search_setup_code(server_port=8080, max_search_calls=10)
        assert "_max_search_calls = 10" in code
        assert "_search_call_count = 0" in code
        assert "Search budget exceeded" in code

    def test_budget_code_is_valid_python(self):
        code = build_search_setup_code(server_port=9000, max_search_calls=5)
        compile(code, "<test>", "exec")

    def test_uses_urllib(self):
        code = build_search_setup_code(server_port=8080)
        assert "urllib.request" in code

    def test_uses_json(self):
        code = build_search_setup_code(server_port=8080)
        assert "json" in code

    def test_error_handling_in_generated_code(self):
        code = build_search_setup_code(server_port=8080)
        assert "except Exception" in code
        assert "Search error" in code
