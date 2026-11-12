"""Test reranker."""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from langchain_core.documents import Document
from weldscope.retrieval.reranker import Reranker
from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import RetrievalError


@pytest.fixture
def config():
    """Create test config."""
    return RAGConfig(groq_api_key="test_token")


@pytest.fixture
def reranker(config):
    """Create reranker instance."""
    return Reranker(config)


def test_reranker_initialization(reranker):
    """Reranker initializes with config."""
    assert reranker.config is not None
    assert reranker._model is None


@patch("weldscope.retrieval.reranker.LocalModel")
def test_get_model_initializes_cross_encoder(mock_cross_encoder, reranker):
    """Get model initializes CrossEncoder."""
    mock_cross_encoder.return_value = Mock()

    model = reranker._get_model()

    assert model is not None
    mock_cross_encoder.assert_called_once()
    assert reranker._model is not None


@patch("weldscope.retrieval.reranker.LocalModel")
def test_get_model_caches_instance(mock_cross_encoder, reranker):
    """Get model caches CrossEncoder instance."""
    mock_cross_encoder.return_value = Mock()

    model1 = reranker._get_model()
    model2 = reranker._get_model()

    assert model1 is model2
    mock_cross_encoder.assert_called_once()


@patch("weldscope.retrieval.reranker.LocalModel")
def test_get_model_raises_on_error(mock_cross_encoder, reranker):
    """Get model raises RetrievalError on initialization failure."""
    mock_cross_encoder.side_effect = Exception("Model error")

    with pytest.raises(RetrievalError, match="Failed to initialize reranker"):
        reranker._get_model()


def test_rerank_empty_documents(reranker):
    """Rerank returns empty list for empty input."""
    result = reranker.rerank("test query", [])
    assert result == []


@patch("weldscope.retrieval.reranker.LocalModel")
def test_model_loading_and_prediction_are_separate_per_query(mock_factory, reranker):
    mock_factory.return_value.predict.return_value = [0.5]
    documents = [Document(page_content="FSW evidence")]
    cold, warm = {}, {}
    reranker.rerank("FSW", documents, timings=cold)
    reranker.rerank("FSW", documents, timings=warm)
    assert cold["reranker_load_ms"] > 0
    assert warm["reranker_load_ms"] == 0
    assert cold["reranker_compute_ms"] >= 0
    assert warm["reranker_compute_ms"] >= 0
    mock_factory.assert_called_once_with(reranker.config.reranker_model)


@patch("weldscope.retrieval.reranker.LocalModel")
def test_rerank_sorts_by_score(mock_cross_encoder, reranker):
    """Rerank sorts documents by relevance score."""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([0.3, 0.9, 0.6])
    mock_cross_encoder.return_value = mock_model

    docs = [
        Document(page_content="Low relevance", metadata={"source": "doc1"}),
        Document(page_content="High relevance", metadata={"source": "doc2"}),
        Document(page_content="Medium relevance", metadata={"source": "doc3"}),
    ]

    results = reranker.rerank("test query", docs)

    assert len(results) == 3
    assert results[0].page_content == "High relevance"
    assert results[1].page_content == "Medium relevance"
    assert results[2].page_content == "Low relevance"
    assert results[0].metadata["rerank_score"] == 0.9


@patch("weldscope.retrieval.reranker.LocalModel")
def test_rerank_respects_top_k(mock_cross_encoder, reranker):
    """Rerank returns only top_k documents."""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([0.3, 0.9, 0.6])
    mock_cross_encoder.return_value = mock_model

    docs = [
        Document(page_content="Doc 1"),
        Document(page_content="Doc 2"),
        Document(page_content="Doc 3"),
    ]

    results = reranker.rerank("test query", docs, top_k=2)

    assert len(results) == 2
    assert results[0].metadata["rerank_score"] == 0.9
    assert results[1].metadata["rerank_score"] == 0.6


@patch("weldscope.retrieval.reranker.LocalModel")
def test_rerank_adds_scores_to_metadata(mock_cross_encoder, reranker):
    """Rerank adds rerank_score to document metadata."""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([0.85])
    mock_cross_encoder.return_value = mock_model

    docs = [Document(page_content="Test doc", metadata={"source": "test.pdf"})]

    results = reranker.rerank("test query", docs)

    assert "rerank_score" in results[0].metadata
    assert results[0].metadata["rerank_score"] == 0.85
    assert results[0].metadata["source"] == "test.pdf"


@patch("weldscope.retrieval.reranker.LocalModel")
def test_rerank_raises_on_error(mock_cross_encoder, reranker):
    """Rerank raises RetrievalError on failure."""
    mock_model = Mock()
    mock_model.predict.side_effect = Exception("Prediction error")
    mock_cross_encoder.return_value = mock_model

    docs = [Document(page_content="Test doc")]

    with pytest.raises(RetrievalError, match="Reranking failed"):
        reranker.rerank("test query", docs)


@patch("weldscope.retrieval.reranker.LocalModel")
def test_score_pairs_returns_scores(mock_cross_encoder, reranker):
    """Score pairs returns list of scores."""
    mock_model = Mock()
    mock_model.predict.return_value = np.array([0.8, 0.6, 0.9])
    mock_cross_encoder.return_value = mock_model

    texts = ["Text 1", "Text 2", "Text 3"]
    scores = reranker.score_pairs("test query", texts)

    assert len(scores) == 3
    assert scores == [0.8, 0.6, 0.9]
    assert all(isinstance(s, float) for s in scores)


@patch("weldscope.retrieval.reranker.LocalModel")
def test_score_pairs_raises_on_error(mock_cross_encoder, reranker):
    """Score pairs raises RetrievalError on failure."""
    mock_model = Mock()
    mock_model.predict.side_effect = Exception("Scoring error")
    mock_cross_encoder.return_value = mock_model

    with pytest.raises(RetrievalError, match="Scoring failed"):
        reranker.score_pairs("test query", ["Text 1"])
