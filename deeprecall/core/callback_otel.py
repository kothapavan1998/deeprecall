"""OpenTelemetry callback for DeepRecall distributed tracing.

Emits traces and spans for every query, reasoning step, and search call
so you can monitor DeepRecall in Jaeger, Datadog, Grafana Tempo, Honeycomb,
AWS X-Ray, or any OTLP-compatible backend.

Install: pip install deeprecall[otel]
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from deeprecall.core.callbacks import BaseCallback

if TYPE_CHECKING:
    from deeprecall.core.config import DeepRecallConfig
    from deeprecall.core.guardrails import BudgetStatus
    from deeprecall.core.types import DeepRecallResult, ReasoningStep

logger = logging.getLogger(__name__)


class OpenTelemetryCallback(BaseCallback):
    """Emit OpenTelemetry traces for DeepRecall queries.

    Each ``query()`` call becomes a parent span. Reasoning steps and searches
    become child spans. All span attributes follow OpenTelemetry semantic
    conventions for GenAI where applicable.

    Args:
        service_name: Service name for the OTEL resource (default: "deeprecall").
        tracer_name: Name of the tracer instrument (default: "deeprecall.engine").
        endpoint: OTLP collector endpoint. If None, uses the
            ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var or default (localhost:4317).
        use_http: Use HTTP (proto/http) exporter instead of gRPC (default: False).
        headers: Extra headers for the OTLP exporter (e.g. auth tokens).
        insecure: Disable TLS for gRPC exporter (default: True for local dev).

    Examples:
        Minimal (auto-detects local Jaeger or OTLP collector):

        >>> from deeprecall.core.callback_otel import OpenTelemetryCallback
        >>> otel = OpenTelemetryCallback()

        With Datadog or Grafana Cloud:

        >>> otel = OpenTelemetryCallback(
        ...     endpoint="https://otlp.datadoghq.com:4317",
        ...     headers={"DD-API-KEY": "your-key"},
        ...     service_name="my-rag-service",
        ... )

        Wire it up:

        >>> from deeprecall import DeepRecall, DeepRecallConfig
        >>> config = DeepRecallConfig(callbacks=[otel], ...)
        >>> engine = DeepRecall(vectorstore=store, config=config)
    """

    def __init__(
        self,
        service_name: str = "deeprecall",
        tracer_name: str = "deeprecall.engine",
        endpoint: str | None = None,
        use_http: bool = False,
        headers: dict[str, str] | None = None,
        insecure: bool = True,
    ):
        self._tracer = _init_tracer(
            service_name=service_name,
            tracer_name=tracer_name,
            endpoint=endpoint,
            use_http=use_http,
            headers=headers,
            insecure=insecure,
        )
        # Thread-local storage: each concurrent query (running in its own
        # thread via asyncio.to_thread) gets isolated span state.
        self._local = threading.local()

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_query_start(self, query: str, config: DeepRecallConfig) -> None:
        """Start a root span for the query."""
        from opentelemetry import trace

        span = self._tracer.start_span(
            "deeprecall.query",
            attributes={
                "deeprecall.query": _truncate(query, 1000),
                "deeprecall.backend": config.backend,
                "deeprecall.model": config.backend_kwargs.get("model_name", "unknown"),
                "deeprecall.max_iterations": config.max_iterations,
                "deeprecall.top_k": config.top_k,
            },
        )
        self._local.current_span = span
        self._local.step_count = 0
        # Set the span as current context (so child spans nest correctly)
        self._local.ctx_token = trace.context_api.attach(trace.set_span_in_context(span))

    def on_reasoning_step(self, step: ReasoningStep, budget_status: BudgetStatus) -> None:
        """Create a child span for each reasoning iteration."""
        if self._tracer is None:
            return

        step_count = getattr(self._local, "step_count", 0) + 1
        self._local.step_count = step_count
        span = self._tracer.start_span(
            f"deeprecall.step.{step_count}",
            attributes={
                "deeprecall.step.iteration": step.iteration,
                "deeprecall.step.action": step.action or "unknown",
                "deeprecall.step.has_code": bool(step.code),
                "deeprecall.step.output_length": len(step.output or ""),
                "deeprecall.budget.iterations_used": budget_status.iterations_used,
                "deeprecall.budget.search_calls_used": budget_status.search_calls_used,
            },
        )
        span.end()

    def on_search(self, query: str, num_results: int, time_ms: float) -> None:
        """Create a child span for each vector store search."""
        if self._tracer is None:
            return

        span = self._tracer.start_span(
            "deeprecall.search",
            attributes={
                "deeprecall.search.query": _truncate(query, 500),
                "deeprecall.search.num_results": num_results,
                "deeprecall.search.latency_ms": round(time_ms, 2),
            },
        )
        span.end()

    def on_query_end(self, result: DeepRecallResult) -> None:
        """Close the root span with final metrics."""
        from opentelemetry import trace
        from opentelemetry.trace import StatusCode

        span = getattr(self._local, "current_span", None)
        if span is None:
            return

        span.set_attributes(
            {
                "deeprecall.execution_time_s": round(result.execution_time, 3),
                "deeprecall.answer_length": len(result.answer),
                "deeprecall.sources_count": len(result.sources),
                "deeprecall.steps_count": len(result.reasoning_trace),
                "deeprecall.tokens.input": result.usage.total_input_tokens,
                "deeprecall.tokens.output": result.usage.total_output_tokens,
                "deeprecall.tokens.total": (
                    result.usage.total_input_tokens + result.usage.total_output_tokens
                ),
            }
        )

        if result.confidence is not None:
            span.set_attribute("deeprecall.confidence", result.confidence)

        if result.error:
            span.set_status(StatusCode.ERROR, result.error)
            span.set_attribute("deeprecall.error", result.error)
        else:
            span.set_status(StatusCode.OK)

        span.end()

        # Detach context
        ctx_token = getattr(self._local, "ctx_token", None)
        if ctx_token is not None:
            trace.context_api.detach(ctx_token)
        self._local.current_span = None
        self._local.ctx_token = None

    def on_error(self, error: Exception) -> None:
        """Record the error on the current span."""
        from opentelemetry.trace import StatusCode

        span = getattr(self._local, "current_span", None)
        if span:
            span.set_status(StatusCode.ERROR, str(error))
            span.record_exception(error)
            span.end()
            self._local.current_span = None

    def on_budget_warning(self, status: BudgetStatus) -> None:
        """Add a span event for budget exceeded."""
        span = getattr(self._local, "current_span", None)
        if span:
            span.add_event(
                "budget_exceeded",
                attributes={
                    "reason": status.exceeded_reason or "unknown",
                    "iterations_used": status.iterations_used,
                    "search_calls_used": status.search_calls_used,
                },
            )


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _init_tracer(
    service_name: str,
    tracer_name: str,
    endpoint: str | None,
    use_http: bool,
    headers: dict[str, str] | None,
    insecure: bool,
) -> Any:
    """Initialize an OpenTelemetry tracer with OTLP exporter.

    Lazy-imports all OTEL dependencies so they only need to be installed
    when this callback is actually used.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        raise ImportError(
            "opentelemetry-sdk and opentelemetry-exporter-otlp are required "
            "for OpenTelemetryCallback. Install with: pip install deeprecall[otel]"
        ) from None

    # Build OTLP exporter (gRPC or HTTP)
    exporter_kwargs: dict[str, Any] = {}
    if endpoint:
        exporter_kwargs["endpoint"] = endpoint
    if headers:
        exporter_kwargs["headers"] = headers

    if use_http:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
    else:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter_kwargs["insecure"] = insecure

    exporter = OTLPSpanExporter(**exporter_kwargs)

    # Build provider
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # Register as global tracer provider only if not already set by the user
    existing = trace.get_tracer_provider()
    # The default NoOpTracerProvider has no add_span_processor method
    if not hasattr(existing, "add_span_processor"):
        trace.set_tracer_provider(provider)
    else:
        # User already set a provider; add our processor to it
        existing.add_span_processor(BatchSpanProcessor(exporter))
        provider = existing  # noqa: F841

    return trace.get_tracer(tracer_name)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len to avoid oversized span attributes."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
