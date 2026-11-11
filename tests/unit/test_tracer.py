"""Test RAG tracer module."""

import pytest
from unittest.mock import Mock, patch
from weldscope.observability.tracer import RAGTracer
from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import RAGException


def test_tracer_disabled_by_default():
    """Tracer is disabled when langfuse_enabled=False."""
    config = RAGConfig(groq_api_key="test", langfuse_enabled=False)
    tracer = RAGTracer(config)

    assert tracer.enabled is False
    assert tracer.client is None


def test_tracer_requires_keys_when_enabled():
    """Tracer raises exception if keys missing when enabled."""
    config = RAGConfig(groq_api_key="test", langfuse_enabled=True)

    with pytest.raises(RAGException, match="Langfuse keys required"):
        RAGTracer(config)


@patch("weldscope.observability.tracer.Langfuse")
def test_tracer_initializes_client_when_enabled(mock_langfuse):
    """Tracer initializes Langfuse client when enabled with keys."""
    config = RAGConfig(
        groq_api_key="test",
        langfuse_enabled=True,
        langfuse_public_key="pk_test",
        langfuse_secret_key="sk_test",
    )

    tracer = RAGTracer(config)

    assert tracer.enabled is True
    mock_langfuse.assert_called_once()


def test_trace_query_decorator_disabled():
    """Trace decorator is no-op when disabled."""
    config = RAGConfig(groq_api_key="test", langfuse_enabled=False)
    tracer = RAGTracer(config)

    @tracer.trace_query()
    def test_func():
        return "result"

    assert test_func() == "result"


def test_trace_retrieval_disabled():
    """Log retrieval is no-op when disabled."""
    config = RAGConfig(groq_api_key="test", langfuse_enabled=False)
    tracer = RAGTracer(config)

    tracer.log_retrieval("query", 5, 100.0)


def test_trace_generation_disabled():
    """Log generation is no-op when disabled."""
    config = RAGConfig(groq_api_key="test", langfuse_enabled=False)
    tracer = RAGTracer(config)

    tracer.log_generation("prompt", "answer", 100, 200.0)


def test_score_feedback_disabled():
    """Score feedback is no-op when disabled."""
    config = RAGConfig(groq_api_key="test", langfuse_enabled=False)
    tracer = RAGTracer(config)

    tracer.score_feedback("trace_id", 1.0, "good")


@patch("weldscope.observability.tracer.Langfuse")
def test_flush_when_enabled(mock_langfuse):
    """Flush calls client flush when enabled."""
    config = RAGConfig(
        groq_api_key="test",
        langfuse_enabled=True,
        langfuse_public_key="pk_test",
        langfuse_secret_key="sk_test",
    )

    tracer = RAGTracer(config)
    tracer.flush()

    tracer.client.flush.assert_called_once()
