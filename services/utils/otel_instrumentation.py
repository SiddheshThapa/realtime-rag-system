# services/utils/otel_instrumentation.py
from __future__ import annotations

import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

# 🔌 Auto-instrumentation hooks
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
except Exception:
    # If any optional instrumentation packages are missing, we’ll skip them gracefully.
    FastAPIInstrumentor = RequestsInstrumentor = RedisInstrumentor = None  # type: ignore


def _jaeger_exporter() -> JaegerExporter:
    """
    Build a Jaeger Thrift exporter that talks to the Jaeger agent.
    Uses env vars if present, falls back to docker-compose service name/port.
    """
    host = os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST", "jaeger")
    port = int(os.getenv("OTEL_EXPORTER_JAEGER_AGENT_PORT", "6831"))
    return JaegerExporter(agent_host_name=host, agent_port=port)


def _resource(service_name: str) -> Resource:
    return Resource.create(
        {
            "service.name": service_name,
            "service.version": os.getenv("SERVICE_VERSION", "1.0.0"),
            "deployment.environment": os.getenv("ENV", "local"),
        }
    )


def _provider(service_name: str) -> TracerProvider:
    # Sample everything by default; override with OTEL_TRACES_SAMPLER_ARG (0.0–1.0)
    sample_ratio = float(os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0"))
    sampler = ParentBased(TraceIdRatioBased(sample_ratio))

    provider = TracerProvider(resource=_resource(service_name), sampler=sampler)
    provider.add_span_processor(BatchSpanProcessor(_jaeger_exporter()))
    return provider


def _auto_instrument():
    """
    Enable library auto-instrumentation.
    FastAPIInstrumentor.instrument() patches FastAPI so apps created AFTER this call
    are automatically traced (fits your current import order).
    """
    if FastAPIInstrumentor:
        try:
            FastAPIInstrumentor().instrument()
        except Exception:
            # already instrumented or package missing — ignore
            pass

    if RequestsInstrumentor:
        try:
            RequestsInstrumentor().instrument()
        except Exception:
            pass

    if RedisInstrumentor:
        try:
            RedisInstrumentor().instrument()
        except Exception:
            pass


def init_tracer(service_name: str):
    """
    Initializes tracing + auto-instrumentation and returns a tracer.
    Keep the same call site as you already have:
        tracer = init_tracer("multi-stage-rag-api")

    No other code changes required.
    """
    provider = _provider(service_name)
    trace.set_tracer_provider(provider)
    _auto_instrument()  # make FastAPI/requests/redis emit spans automatically
    return trace.get_tracer(service_name)
