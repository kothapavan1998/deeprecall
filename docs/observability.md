# Observability

DeepRecall provides a callback system for full observability into the reasoning pipeline. Every query, reasoning step, search call, and error can be monitored, traced, and logged.

## Built-in Callbacks

### ConsoleCallback

Rich terminal output showing reasoning steps in real time.

```python
from deeprecall import ConsoleCallback, DeepRecallConfig

config = DeepRecallConfig(
    callbacks=[ConsoleCallback(show_code=True, show_output=True)],
)
```

### JSONLCallback

Logs all events to a JSONL file for post-hoc analysis.

```python
from deeprecall import JSONLCallback, DeepRecallConfig

config = DeepRecallConfig(
    callbacks=[JSONLCallback(log_dir="./logs")],
)
# Creates: ./logs/deeprecall_20260213_143022.jsonl
```

### UsageTrackingCallback

Tracks cumulative token usage, query count, and errors across multiple queries.

```python
from deeprecall import UsageTrackingCallback, DeepRecallConfig

tracker = UsageTrackingCallback()
config = DeepRecallConfig(callbacks=[tracker])

# After running queries:
print(tracker.summary())
# {"total_queries": 10, "total_tokens": 45000, "total_searches": 32, ...}
```

### OpenTelemetryCallback

Emit distributed traces to **Jaeger**, **Datadog**, **Grafana Tempo**, **Honeycomb**, **AWS X-Ray**, or any OTLP-compatible backend.

```bash
pip install deeprecall[otel]
```

```python
from deeprecall import OpenTelemetryCallback, DeepRecallConfig

# Auto-detects local OTLP collector (localhost:4317)
otel = OpenTelemetryCallback(service_name="my-rag-service")

config = DeepRecallConfig(callbacks=[otel])
```

#### Trace Structure

Each `query()` call produces a trace:

```
deeprecall.query (root span)
  |-- deeprecall.step.1    (reasoning iteration)
  |-- deeprecall.search    (vector store search)
  |-- deeprecall.step.2    (reasoning iteration)
  |-- deeprecall.search    (vector store search)
  |-- deeprecall.step.3    (final answer)
```

#### Span Attributes

| Attribute | Description |
|-----------|-------------|
| `deeprecall.query` | The user query text |
| `deeprecall.backend` | LLM backend name |
| `deeprecall.model` | Model name |
| `deeprecall.execution_time_s` | Total query time |
| `deeprecall.tokens.total` | Total tokens used |
| `deeprecall.sources_count` | Number of retrieved sources |
| `deeprecall.steps_count` | Number of reasoning steps |
| `deeprecall.confidence` | Confidence score (0-1) |
| `deeprecall.search.query` | Search query text |
| `deeprecall.search.latency_ms` | Search latency |
| `deeprecall.budget.*` | Budget utilization |

#### Connecting to Backends

**Jaeger (local):**

```python
otel = OpenTelemetryCallback(
    endpoint="http://localhost:4317",
    insecure=True,
)
```

**Datadog:**

```python
otel = OpenTelemetryCallback(
    endpoint="https://otlp.datadoghq.com:4317",
    headers={"DD-API-KEY": "your-datadog-api-key"},
    service_name="my-app",
)
```

**Grafana Cloud / Tempo:**

```python
otel = OpenTelemetryCallback(
    endpoint="https://otlp-gateway-prod-us-east-0.grafana.net/otlp",
    headers={"Authorization": "Basic <base64-encoded-credentials>"},
    use_http=True,
    service_name="deeprecall-prod",
)
```

**Honeycomb:**

```python
otel = OpenTelemetryCallback(
    endpoint="https://api.honeycomb.io:443",
    headers={"x-honeycomb-team": "your-api-key"},
    service_name="deeprecall",
)
```

**AWS X-Ray (via OTLP collector):**

```python
# Run the AWS OTEL collector sidecar, then point DeepRecall at it
otel = OpenTelemetryCallback(
    endpoint="http://localhost:4317",
    insecure=True,
    service_name="deeprecall-prod",
)
```

## Custom Callbacks

Extend `BaseCallback` to build your own:

```python
from deeprecall import BaseCallback

class SlackAlertCallback(BaseCallback):
    def on_error(self, error: Exception) -> None:
        send_slack_message(f"DeepRecall error: {error}")

    def on_budget_warning(self, status) -> None:
        send_slack_message(f"Budget exceeded: {status.exceeded_reason}")
```

## Combining Callbacks

Use multiple callbacks simultaneously:

```python
config = DeepRecallConfig(
    callbacks=[
        ConsoleCallback(),              # Live terminal output
        JSONLCallback(log_dir="./logs"), # Persistent file logs
        OpenTelemetryCallback(),        # Distributed traces
        UsageTrackingCallback(),        # Usage metrics
    ],
)
```
