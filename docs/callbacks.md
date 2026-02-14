# Callbacks

DeepRecall provides a callback system for observability, logging, and monitoring.

## Built-in Callbacks

### ConsoleCallback

Real-time rich console output showing each reasoning step:

```python
from deeprecall import DeepRecallConfig, ConsoleCallback

config = DeepRecallConfig(
    callbacks=[ConsoleCallback(show_code=True, show_output=True)],
)
```

### JSONLCallback

Structured logging to a JSONL file:

```python
from deeprecall import JSONLCallback

config = DeepRecallConfig(
    callbacks=[JSONLCallback(log_dir="./logs")],
)
```

Events logged: `query_start`, `reasoning_step`, `query_end`, `error`, `budget_warning`.

### UsageTrackingCallback

Aggregates usage across queries for billing:

```python
from deeprecall.core.callbacks import UsageTrackingCallback

tracker = UsageTrackingCallback()
config = DeepRecallConfig(callbacks=[tracker])
engine = DeepRecall(vectorstore=store, config=config)

engine.query("question 1")
engine.query("question 2")

print(tracker.summary())
# {"total_queries": 2, "total_tokens": 8500, "total_searches": 12, ...}
```

## Custom Callbacks

Implement `BaseCallback` and override any hooks you need:

```python
from deeprecall.core.callbacks import BaseCallback

class SlackAlertCallback(BaseCallback):
    def on_budget_warning(self, status):
        send_slack_alert(f"Budget exceeded: {status.exceeded_reason}")

    def on_error(self, error):
        send_slack_alert(f"DeepRecall error: {error}")
```

## Available Hooks

| Hook | When it fires |
|------|---------------|
| `on_query_start(query, config)` | Query begins |
| `on_reasoning_step(step, budget_status)` | After each RLM iteration |
| `on_search(query, num_results, time_ms)` | After each vector store search |
| `on_query_end(result)` | Query completes |
| `on_error(error)` | Unrecoverable error |
| `on_budget_warning(status)` | Budget limit exceeded |

All hooks are optional -- unimplemented ones are silently skipped. Exceptions in callbacks never propagate to the caller.
