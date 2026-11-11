"""Integration test for end-to-end RAG pipeline."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from weldscope.core.config import RAGConfig
from weldscope.ingestion.builder import VectorStoreBuilder
from weldscope.retrieval.retriever import HybridRetriever
from weldscope.pipeline.pipeline import RAGPipeline


@pytest.fixture
def config():
    """Create test config."""
    return RAGConfig(hf_token="test_token")


@pytest.fixture
def mock_vectorstore_data():
    """Mock vectorstore collection data."""
    return {
        "ids": ["doc1", "doc2", "doc3"],
        "documents": [
            "Wormhole defects occur due to insufficient heat input.",
            "Tunnel defects are caused by excessive tool rotation speed.",
            "Surface cracks result from improper cooling rates.",
        ],
        "metadatas": [
            {"source": "fsw_doc1.pdf", "page": 5},
            {"source": "fsw_doc2.pdf", "page": 12},
            {"source": "fsw_doc3.pdf", "page": 8},
        ],
    }


@patch("weldscope.ingestion.builder.Chroma")
@patch("weldscope.ingestion.builder.HuggingFaceEmbeddings")
def test_full_pipeline_integration(
    mock_embeddings, mock_chroma, config, mock_vectorstore_data, tmp_path
):
    """Test complete RAG pipeline from vectorstore to response."""
    # Setup mocks
    mock_embeddings.return_value = Mock()

    mock_vectorstore = Mock()
    mock_vectorstore.get.return_value = mock_vectorstore_data
    mock_vectorstore.as_retriever.return_value = Mock()
    mock_chroma.return_value = mock_vectorstore

    persist_dir = tmp_path / "vectorstore"
    persist_dir.mkdir()

    # Initialize builder and load vectorstore
    builder = VectorStoreBuilder(config)
    vectorstore, documents = builder.load_vectorstore(str(persist_dir))

    # Verify documents loaded
    assert len(documents) == 3
    assert documents[0].page_content == "Wormhole defects occur due to insufficient heat input."
    assert documents[0].metadata["source"] == "fsw_doc1.pdf"

    # Initialize retriever
    with patch("weldscope.retrieval.retriever.BM25Retriever"):
        retriever = HybridRetriever(vectorstore, documents, config)
        assert retriever is not None

    # Initialize pipeline
    with (
        patch("weldscope.pipeline.pipeline.Reranker"),
        patch("weldscope.pipeline.pipeline.LLMGenerator"),
    ):
        pipeline = RAGPipeline(retriever, config)

        # Mock retrieval and generation
        pipeline.retriever.retrieve = Mock(return_value=documents[:2])
        pipeline.reranker.rerank = Mock(return_value=documents[:1])
        pipeline.generator.create_prompt = Mock(return_value="Test prompt")
        pipeline.generator.generate = Mock(
            return_value="Wormhole defects are caused by insufficient heat input during the FSW process."
        )

        # Execute query
        response = pipeline.query("What causes wormhole defects?")

        # Verify response
        assert response is not None
        assert response.query == "What causes wormhole defects?"
        assert "Wormhole defects" in response.answer
        assert len(response.sources) == 1
        assert response.sources[0].doc_name == "fsw_doc1.pdf"
        assert response.metadata["retrieved_count"] == 2
        assert response.metadata["reranked_count"] == 1


@patch("weldscope.ingestion.builder.Chroma")
@patch("weldscope.ingestion.builder.HuggingFaceEmbeddings")
def test_pipeline_handles_empty_results(mock_embeddings, mock_chroma, config, tmp_path):
    """Test pipeline handles empty retrieval results gracefully."""
    mock_embeddings.return_value = Mock()

    mock_vectorstore = Mock()
    mock_vectorstore.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    mock_chroma.return_value = mock_vectorstore

    persist_dir = tmp_path / "vectorstore"
    persist_dir.mkdir()

    builder = VectorStoreBuilder(config)
    vectorstore, documents = builder.load_vectorstore(str(persist_dir))

    assert len(documents) == 0


@patch("weldscope.ingestion.builder.Chroma")
@patch("weldscope.ingestion.builder.HuggingFaceEmbeddings")
def test_pipeline_with_streaming(
    mock_embeddings, mock_chroma, config, mock_vectorstore_data, tmp_path
):
    """Test pipeline streaming functionality."""
    mock_embeddings.return_value = Mock()

    mock_vectorstore = Mock()
    mock_vectorstore.get.return_value = mock_vectorstore_data
    mock_vectorstore.as_retriever.return_value = Mock()
    mock_chroma.return_value = mock_vectorstore

    persist_dir = tmp_path / "vectorstore"
    persist_dir.mkdir()

    builder = VectorStoreBuilder(config)
    vectorstore, documents = builder.load_vectorstore(str(persist_dir))

    with patch("weldscope.retrieval.retriever.BM25Retriever"):
        retriever = HybridRetriever(vectorstore, documents, config)

    with (
        patch("weldscope.pipeline.pipeline.Reranker"),
        patch("weldscope.pipeline.pipeline.LLMGenerator"),
    ):
        pipeline = RAGPipeline(retriever, config)

        pipeline.retriever.retrieve = Mock(return_value=documents[:1])
        pipeline.reranker.rerank = Mock(return_value=documents[:1])
        pipeline.generator.create_prompt = Mock(return_value="Test prompt")
        pipeline.generator.stream = Mock(return_value=iter(["Chunk ", "1", " and ", "2"]))

        chunks = list(pipeline.stream_query("Test query"))

        assert chunks == ["Chunk ", "1", " and ", "2"]
