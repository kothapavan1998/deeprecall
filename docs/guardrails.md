# Budget Guardrails

DeepRecall lets you set hard limits on any query to prevent runaway LLM costs.

## QueryBudget

```python
from deeprecall import QueryBudget

budget = QueryBudget(
    max_iterations=10,        # Max reasoning loop iterations
    max_search_calls=20,      # Max vector DB searches
    max_tokens=50000,         # Total token budget (input + output)
    max_time_seconds=60.0,    # Wall-clock timeout
    max_cost_usd=0.50,        # Dollar cost limit (USD)
)
```

Set any field to `None` (default) to disable that limit.

### Cost Budget (v0.4+)

`max_cost_usd` is now actively enforced. When set, it is passed to RLM as `max_budget` which stops the reasoning loop when the cost limit is reached. Cost data is automatically extracted when using the OpenRouter backend. The cost is also tracked at the tracer level and reported in `result.budget_status["cost_usd"]`.

```python
result = engine.query(
    "Expensive multi-hop question?",
    budget=QueryBudget(max_cost_usd=0.10),
)
print(f"Total cost: ${result.usage.total_cost_usd}")
```

### Execution Limits (v0.4+)

In addition to per-query budgets, you can set execution limits on the engine config:

```python
from deeprecall import DeepRecallConfig

config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini"},
    max_timeout=120.0,   # Wall-clock timeout in seconds
    max_errors=5,        # Max consecutive REPL errors before abort
    max_tokens=50000,    # Total token limit (input + output)
)
```

### Compaction (v0.4+)

For very long reasoning chains, enable compaction to prevent hitting the model's context window:

```python
config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini"},
    compaction=True,               # Summarise history when nearing context limit
    compaction_threshold_pct=0.85,  # Trigger at 85% of context window (default)
)
```

## Usage

### Per-query budget

```python
result = engine.query("question", budget=QueryBudget(max_search_calls=5))
```

### Default budget via config

```python
from deeprecall import DeepRecallConfig, QueryBudget

config = DeepRecallConfig(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o-mini"},
    budget=QueryBudget(max_search_calls=20, max_time_seconds=60),
)
```

### CLI

```bash
deeprecall query "question" --max-searches 10 --max-tokens 50000 --max-time 30
```

## What happens when a budget is exceeded

When any budget limit is hit, the engine:

1. Catches the `BudgetExceededError` internally
2. Collects all retrieved sources from the search server
3. Makes a **one-shot synthesis LLM call** to produce a best-effort answer from the sources gathered so far (instead of returning raw REPL state)
4. Returns a `[Partial]` result with the synthesized answer
5. Sets `result.error` with the reason
6. Sets `result.budget_status["budget_exceeded"] = True`

If the synthesis call itself fails, the result falls back to a clean message indicating how many sources were retrieved.

```python
result = engine.query("question", budget=QueryBudget(max_search_calls=3))

if result.error:
    print(f"Budget hit: {result.error}")
    print(f"Partial answer: {result.answer}")  # Synthesized, not raw REPL state
```

## Search gating

When `max_search_calls` is set, the `search_db()` function injected into the REPL includes a counter. Once the limit is reached, `search_db()` returns a message telling the LLM to use its existing results. This prevents the LLM from making more searches even within a single iteration.

## BudgetStatus

Every result includes a `budget_status` dict:

```python
{
    "iterations_used": 5,
    "search_calls_used": 8,
    "tokens_used": 12345,
    "time_elapsed": 15.2,
    "cost_usd": 0.02,
    "budget_exceeded": False,
    "exceeded_reason": None,
    "budget": {"max_iterations": None, "max_search_calls": 20, ...}
}
```
