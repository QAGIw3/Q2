from __future__ import annotations

from contextlib import suppress

# Optional OpenTelemetry wiring; no hard dependency.
def setup_tracing() -> None:
    with suppress(Exception):
        from opentelemetry import trace  # type: ignore
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore

        provider = TracerProvider(resource=Resource.create({"service.name": "q2-compute-api"}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)