"""Test RAG pipeline."""

import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from weldscope.pipeline.pipeline import RAGPipeline
from weldscope.core.config import RAGConfig
from weldscope.core.models import RAGResponse
from weldscope.core.exceptions import RetrievalError


@pytest.fixture
def config():
    """Create test config."""
    return RAGConfig(groq_api_key="test_token")


@pytest.fixture
def mock_retriever():
    """Create mock retriever."""
    return Mock()


@pytest.fixture
def mock_tracer():
    """Create mock tracer."""
    tracer = Mock()
    tracer.trace_query = lambda name, metadata=None: lambda f: f
    return tracer


@pytest.fixture
def pipeline(mock_retriever, config):
    """Create pipeline instance."""
    with (
        patch("weldscope.pipeline.pipeline.Reranker"),
        patch("weldscope.pipeline.pipeline.LLMGenerator"),
    ):
        return RAGPipeline(mock_retriever, config)


def test_pipeline_initialization(pipeline, mock_retriever, config):
    """Pipeline initializes with retriever and config."""
    assert pipeline.retriever is mock_retriever
    assert pipeline.config is config
    assert pipeline.reranker is not None
    assert pipeline.generator is not None


def test_format_context_creates_numbered_list(pipeline):
    """Format context creates numbered list with sources."""
    docs = [
        Document(page_content="Content 1", metadata={"source": "doc1.pdf", "page": 5}),
        Document(page_content="Content 2", metadata={"source": "doc2.pdf"}),
    ]

    context = pipeline._format_context(docs)

    assert "[1] doc1.pdf (Page 6):" in context
    assert "Content 1" in context
    assert "[2] doc2.pdf:" in context
    assert "Content 2" in context


def test_query_raises_on_no_documents(pipeline):
    """Query raises RetrievalError when no documents retrieved."""
    pipeline.retriever.retrieve = Mock(return_value=[])

    with pytest.raises(RetrievalError, match="No documents retrieved"):
        pipeline.query("FSW question")


def test_query_delivers_only_after_generation_and_validation(pipeline):
    docs = [Document(page_content="Evidence", metadata={"source": "fsw.pdf"})]
    pipeline.retriever.retrieve.return_value = docs
    pipeline.reranker.rerank.return_value = docs
    delivered = []

    def generate():
        yield "First "
        assert delivered == []
        yield "second"

    pipeline.generator.stream.return_value = generate()
    response = pipeline.query("FSW question", stream=True, on_chunk=delivered.append)
    assert delivered == ["First second"]
    assert response.answer == "First second"
    assert response.sources[0].doc_name == "fsw.pdf"


def test_query_executes_full_pipeline(pipeline):
    """Query executes retrieval, reranking, and generation."""
    mock_docs = [Document(page_content="Test content", metadata={"source": "test.pdf"})]

    pipeline.retriever.retrieve = Mock(return_value=mock_docs)
    pipeline.reranker.rerank = Mock(return_value=mock_docs)
    pipeline.generator.create_prompt = Mock(return_value="Test prompt")
    pipeline.generator.generate = Mock(return_value="Test answer")

    result = pipeline.query("FSW question")

    assert isinstance(result, RAGResponse)
    assert result.query == "FSW question"
    assert result.answer == "Test answer"
    assert len(result.sources) == 1

    pipeline.retriever.retrieve.assert_called_once_with("FSW question")
    pipeline.reranker.rerank.assert_called_once()
    pipeline.generator.generate.assert_called_once()


def test_query_with_streaming(pipeline):
    """Query with streaming collects chunks."""
    mock_docs = [Document(page_content="Test content", metadata={"source": "test.pdf"})]

    pipeline.retriever.retrieve = Mock(return_value=mock_docs)
    pipeline.reranker.rerank = Mock(return_value=mock_docs)
    pipeline.generator.create_prompt = Mock(return_value="Test prompt")
    pipeline.generator.stream = Mock(return_value=iter(["chunk1", "chunk2"]))

    result = pipeline.query("FSW question", stream=True)

    assert result.answer == "chunk1chunk2"
    pipeline.generator.stream.assert_called_once()


def test_query_with_tracing(mock_retriever, config, mock_tracer):
    """Query with tracer logs retrieval and generation."""
    with (
        patch("weldscope.pipeline.pipeline.Reranker"),
        patch("weldscope.pipeline.pipeline.LLMGenerator"),
    ):
        pipeline = RAGPipeline(mock_retriever, config, tracer=mock_tracer)

    mock_docs = [Document(page_content="Test content", metadata={"source": "test.pdf"})]

    pipeline.retriever.retrieve = Mock(return_value=mock_docs)
    pipeline.reranker.rerank = Mock(return_value=mock_docs)
    pipeline.generator.create_prompt = Mock(return_value="Test prompt")
    pipeline.generator.generate = Mock(return_value="Test answer")

    result = pipeline.query("FSW question")

    assert isinstance(result, RAGResponse)
    mock_tracer.log_retrieval.assert_called_once()
    mock_tracer.log_generation.assert_called_once()


def test_stream_query_yields_chunks(pipeline):
    """Stream query yields response chunks."""
    mock_docs = [Document(page_content="Test content", metadata={"source": "test.pdf"})]

    pipeline.retriever.retrieve = Mock(return_value=mock_docs)
    pipeline.reranker.rerank = Mock(return_value=mock_docs)
    pipeline.generator.create_prompt = Mock(return_value="Test prompt")
    pipeline.generator.stream = Mock(return_value=iter(["chunk1", "chunk2", "chunk3"]))

    chunks = list(pipeline.stream_query("FSW question"))

    assert chunks == ["chunk1chunk2chunk3"]


def test_stream_query_raises_on_no_documents(pipeline):
    """Stream query raises RetrievalError when no documents retrieved."""
    pipeline.retriever.retrieve = Mock(return_value=[])

    with pytest.raises(RetrievalError, match="No documents retrieved"):
        list(pipeline.stream_query("FSW question"))


def test_query_uses_rerank_top_k_from_config(pipeline, config):
    """Query uses rerank_top_k from config."""
    mock_docs = [Document(page_content="Test content", metadata={"source": "test.pdf"})]

    pipeline.retriever.retrieve = Mock(return_value=mock_docs)
    pipeline.reranker.rerank = Mock(return_value=mock_docs)
    pipeline.generator.create_prompt = Mock(return_value="Test prompt")
    pipeline.generator.generate = Mock(return_value="Test answer")

    pipeline.query("FSW question")

    pipeline.reranker.rerank.assert_called_once()
    call_args = pipeline.reranker.rerank.call_args
    assert call_args[1]["top_k"] == config.rerank_top_k


def test_pipeline_records_timings_without_fabricating_token_usage(pipeline):
    docs = [Document(page_content="Evidence")]
    pipeline.retriever.retrieve.return_value = docs
    pipeline.reranker.rerank.return_value = docs
    pipeline.generator.stream.return_value = iter(["answer"])
    response = pipeline.query("FSW defects?", stream=True)
    timings = response.metadata["timings"]
    assert timings["retrieval_ms"] >= 0
    assert timings["reranking_ms"] >= 0
    assert timings["first_token_ms"] is not None
    assert response.metadata["token_usage"] is None
    assert pipeline.monitor.metrics_history[-1].total_latency_ms == response.latency_ms
    assert response.confidence is None


def test_failed_query_is_recorded_with_error_type(pipeline):
    pipeline.retriever.retrieve.side_effect = RetrievalError("unavailable")
    with pytest.raises(RetrievalError):
        pipeline.query("FSW defects?")
    metric = pipeline.monitor.metrics_history[-1]
    assert metric.metadata["error_type"] == "RetrievalError"
    assert metric.metadata["stage_timings_available"] is False
