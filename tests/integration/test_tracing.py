"""Create real SDK spans and verify correlation without contacting Langfuse Cloud."""

from unittest.mock import patch
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from weldscope.core.config import RAGConfig
from weldscope.observability.tracer import RAGTracer


def test_real_trace_has_correlated_retrieval_and_generation_spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = Langfuse(
        public_key="pk-local-span-test",
        secret_key="sk-local-span-test",
        tracer_provider=provider,
        span_exporter=exporter,
    )
    config = RAGConfig(
        langfuse_enabled=True,
        langfuse_public_key="pk-local-span-test",
        langfuse_secret_key="sk-local-span-test",
    )
    with patch("weldscope.observability.tracer.Langfuse", return_value=client):
        tracer = RAGTracer(config)

    @tracer.trace_query("local-query")
    def query():
        tracer.log_retrieval("welding question", 3, 2.0)
        tracer.log_generation("evidence prompt", "answer", None, 5.0)
        return client.get_current_trace_id()

    trace_id = query()
    provider.force_flush()
    spans = exporter.get_finished_spans()
    query_span = next(span for span in spans if span.name == "local-query")
    assert trace_id == format(query_span.context.trace_id, "032x")
    children = [span for span in spans if span.name in {"retrieval", "generation"}]
    assert len(children) >= 2
    assert all(span.context.trace_id == query_span.context.trace_id for span in children)
    assert all(span.parent.span_id == query_span.context.span_id for span in children)
    client.shutdown()
