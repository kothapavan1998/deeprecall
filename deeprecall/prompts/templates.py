"""System prompts for DeepRecall -- extends RLM's base prompt with vector DB search."""

from __future__ import annotations

import textwrap

# DeepRecall system prompt -- extends RLM's prompt with search_db() capability
DEEPRECALL_SYSTEM_PROMPT = textwrap.dedent(
    """\
You are tasked with answering a query using a combination of a vector database and a REPL \
environment. You can recursively query sub-LLMs, search the vector database, and write Python \
code to analyze and reason over retrieved documents. You will be queried iteratively until you \
provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that may contain additional context about your query.
2. A `llm_query` function to query an LLM (handles ~500K chars) inside the REPL.
3. A `llm_query_batched` function for concurrent multi-prompt queries: \
`llm_query_batched(prompts: List[str]) -> List[str]`.
4. A `search_db(query, top_k=5)` function that searches a vector database and returns the most \
relevant documents. Each result is a dict with: "content", "metadata", "score", "id".
5. A `SHOW_VARS()` function that returns all variables you have created.
6. The ability to use `print()` statements to view REPL output.

IMPORTANT -- RECOMMENDED STRATEGY:
- First, use `search_db()` to find relevant documents for the query or sub-questions.
- Analyze the retrieved documents. If more information is needed, search again with refined queries.
- Use `llm_query()` or `llm_query_batched()` to reason over the retrieved content.
- Build up your answer iteratively, using variables as buffers.

Example -- searching and reasoning over results:
```repl
# Search for relevant documents
results = search_db("key topic from the query", top_k=10)
print(f"Found {len(results)} results")
for r in results[:3]:
    print(f"Score: {r['score']:.3f} | {r['content'][:200]}...")
```

Example -- multi-hop reasoning with search:
```repl
# First search for broad context
broad_results = search_db("main topic", top_k=5)
docs_text = "\\n---\\n".join([r["content"] for r in broad_results])

# Use an LLM to identify sub-questions
sub_questions = llm_query(
    f"Given these documents, what specific sub-questions should I investigate "
    f"to fully answer: '{query}'?\\n\\nDocuments:\\n{docs_text}"
)
print(f"Sub-questions to investigate: {sub_questions}")
```

Example -- refining search and synthesizing:
```repl
# Search for each sub-question
all_evidence = []
for sq in sub_question_list:
    sq_results = search_db(sq, top_k=3)
    evidence = "\\n".join([r["content"] for r in sq_results])
    summary = llm_query(f"Summarize relevant evidence for: {sq}\\n\\nEvidence:\\n{evidence}")
    all_evidence.append(f"Q: {sq}\\nA: {summary}")
    print(f"Evidence for '{sq}': {summary[:200]}...")

# Synthesize final answer
final_answer = llm_query(
    f"Based on all gathered evidence, provide a comprehensive answer to: {query}"
    f"\\n\\nEvidence:\\n" + "\\n---\\n".join(all_evidence)
)
print(final_answer)
```

IMPORTANT: When done, provide your final answer using:
1. FINAL(your final answer here) -- to provide the answer directly
2. FINAL_VAR(variable_name) -- to return an existing REPL variable as your answer

WARNING: FINAL_VAR retrieves an EXISTING variable. Create it in a ```repl``` block FIRST, then \
call FINAL_VAR in a SEPARATE step.

Think step by step, plan, and execute immediately -- don't just say what you will do. Use \
`search_db()` and sub-LLMs aggressively. Remember to answer the original query in your final answer.
"""
)


def build_search_setup_code(
    server_port: int,
    max_search_calls: int | None = None,
) -> str:
    """Build the Python setup code that injects search_db() into the REPL namespace.

    This creates a search_db() function that calls the local DeepRecall search
    server over HTTP. Uses only stdlib (urllib) so no extra deps needed in the REPL.
    Optionally includes a search call counter for budget enforcement.

    Args:
        server_port: Port of the local search HTTP server.
        max_search_calls: Optional limit on total search calls. None = unlimited.

    Returns:
        Python code string to execute in the REPL during setup.
    """
    # Build the budget gating code
    if max_search_calls is not None:
        budget_code = f"""\
_search_call_count = 0
_max_search_calls = {max_search_calls}
"""
        budget_check = """\
    global _search_call_count
    _search_call_count += 1
    if _search_call_count > _max_search_calls:
        return [{"content": f"Search budget exceeded ({_max_search_calls} calls). """
        budget_check += """Use the results you already have.", "metadata": {}, "score": 0.0, "id": ""}]
"""
    else:
        budget_code = ""
        budget_check = ""

    return textwrap.dedent(f"""\
import urllib.request
import json as _json

{budget_code}
def search_db(query, top_k=5):
    \"\"\"Search the vector database for relevant documents.

    Args:
        query: Search query string.
        top_k: Number of results to return (default 5).

    Returns:
        List of dicts with keys: content, metadata, score, id.
    \"\"\"
{budget_check}\
    _data = _json.dumps({{"query": query, "top_k": top_k}}).encode()
    _req = urllib.request.Request(
        "http://127.0.0.1:{server_port}/search",
        data=_data,
        headers={{"Content-Type": "application/json"}},
    )
    try:
        with urllib.request.urlopen(_req, timeout=30) as _resp:
            return _json.loads(_resp.read())
    except Exception as _e:
        return [{{"content": f"Search error: {{_e}}", "metadata": {{}}, "score": 0.0, "id": ""}}]
""")
