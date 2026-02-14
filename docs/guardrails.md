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
    max_cost_usd=0.50,        # Dollar cost limit
)
```

Set any field to `None` (default) to disable that limit.

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

- The engine catches the `BudgetExceededError` internally
- Returns a partial result with whatever the LLM produced so far
- Sets `result.error` with the reason
- Sets `result.budget_status["budget_exceeded"] = True`

```python
result = engine.query("question", budget=QueryBudget(max_search_calls=3))

if result.error:
    print(f"Budget hit: {result.error}")
    print(f"Partial answer: {result.answer}")
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
