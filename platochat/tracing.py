"""
Tracing for the chatbot, through OpenTelemetry rather than one vendor's SDK.

plato_core.py records what each request did -- the route, the two searches,
the rerank, every LLM call -- as nested spans. Where they go is configuration,
not code: any backend that accepts OTLP over HTTP will take them (Langfuse,
Arize Phoenix, Grafana Tempo, Jaeger ...), so changing backend means changing
.env, not this module or plato_core.py.

Where the spans go, first match wins:

    PLATO_TRACING=0                       tracing off
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT    the standard OpenTelemetry variables,
    OTEL_EXPORTER_OTLP_ENDPOINT           read by the exporter itself (with
                                          OTEL_EXPORTER_OTLP_HEADERS for auth)
    LANGFUSE_HOST + LANGFUSE_PUBLIC_KEY   Langfuse's OTLP endpoint, derived
    + LANGFUSE_SECRET_KEY                 here, so the old .env keeps working
    (none of these)                       tracing off

With tracing off, span() still works and costs next to nothing.

Attributes. Backends disagree on which names they read, so each span carries
two small vocabularies that between them cover the ones that matter:

    OpenInference (Arize, Apache-2.0)  openinference.span.kind, input.value,
                                       output.value, llm.model_name,
                                       llm.token_count.*   -- Phoenix reads
                                       these, and Langfuse maps them too
    OpenTelemetry GenAI conventions    gen_ai.operation.name,
                                       gen_ai.request.model,
                                       gen_ai.usage.*      -- the standard,
                                       which Langfuse uses to tell an LLM call
                                       ("generation") from other steps

Traces contain what users typed. Whichever backend receives them, it needs a
retention period (GDPR): the questions come from the public.
"""

import base64
import json
import os
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

_provider = None  # the SDK TracerProvider, when tracing is on


def _endpoint_and_headers() -> tuple[str | None, dict | None]:
    """Where to send spans, or (None, None) for tracing off."""
    if os.environ.get("PLATO_TRACING", "1").strip().lower() in ("0", "false", "no", "off"):
        return None, None
    if os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return "", None  # the exporter reads the OTEL_* variables itself
    host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
    public, secret = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
    if host and public and secret:
        auth = base64.b64encode(f"{public}:{secret}".encode()).decode()
        return host.rstrip("/") + "/api/public/otel/v1/traces", {
            "Authorization": f"Basic {auth}",
            "x-langfuse-ingestion-version": "4",
        }
    return None, None


def configure() -> None:
    """Set up the exporter once. Call after load_dotenv()."""
    global _provider
    if _provider is not None:
        return
    endpoint, headers = _endpoint_and_headers()
    if endpoint is None:
        print("Tracing off (no OTLP endpoint or Langfuse keys configured).")
        return

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers) if endpoint else OTLPSpanExporter()
    _provider = TracerProvider(resource=Resource.create({
        "service.name": os.environ.get("OTEL_SERVICE_NAME", "platochat"),
    }))
    _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)
    print(f"Tracing to {endpoint or os.environ.get('OTEL_EXPORTER_OTLP_TRACES_ENDPOINT') or os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT')}")


def flush() -> None:
    """Send whatever is still queued. For scripts that exit right after."""
    if _provider is not None:
        _provider.force_flush()


def _json(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


class Span:
    """A thin handle on an OpenTelemetry span, with input/output as JSON."""

    def __init__(self, span):
        self._span = span

    def update(self, input=None, output=None, **attributes) -> None:
        if input is not None:
            self._span.set_attribute("input.value", _json(input))
            self._span.set_attribute("input.mime_type",
                                     "text/plain" if isinstance(input, str) else "application/json")
        if output is not None:
            self._span.set_attribute("output.value", _json(output))
            self._span.set_attribute("output.mime_type",
                                     "text/plain" if isinstance(output, str) else "application/json")
        for key, value in attributes.items():
            if value is not None:
                self._span.set_attribute(key, value)

    def usage(self, input_tokens: int | None, output_tokens: int | None) -> None:
        """Token counts of an LLM call, under both vocabularies."""
        self.update(**{
            "gen_ai.usage.input_tokens":   input_tokens,
            "gen_ai.usage.output_tokens":  output_tokens,
            "llm.token_count.prompt":      input_tokens,
            "llm.token_count.completion":  output_tokens,
        })

    def error(self, exc: BaseException) -> None:
        """Mark a failure that the code catches and recovers from."""
        self._span.record_exception(exc)
        self._span.set_status(Status(StatusCode.ERROR, str(exc)))


@contextmanager
def span(name: str, kind: str = "CHAIN", input=None):
    """A step of the pipeline, nested under whatever span is open.

    kind is OpenInference's span kind: CHAIN (a step), RETRIEVER (a search),
    TOOL (a tool call the model made), AGENT (the whole request).
    An exception escaping the block is recorded on the span and re-raised.
    """
    with trace.get_tracer("platochat").start_as_current_span(name) as otel_span:
        handle = Span(otel_span)
        handle.update(input=input, **{"openinference.span.kind": kind})
        yield handle


@contextmanager
def llm_span(name: str, model: str, messages: list[dict], tools: list | None = None):
    """One call to a language model; the caller adds output and usage."""
    with span(name, kind="LLM", input={"messages": messages, "tools": tools} if tools else messages) as handle:
        handle.update(**{
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model":  model,
            "llm.model_name":        model,
        })
        yield handle
