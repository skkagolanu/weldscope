"""Test hybrid retriever."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document
from weldscope.retrieval.retriever import HybridRetriever
from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import RetrievalError


@pytest.fixture
def config():
    """Create test config."""
    return RAGConfig(hf_token="test_token")


@pytest.fixture
def mock_vectorstore():
    """Create mock vectorstore."""
    mock = Mock()
    mock_retriever = Mock()
    mock_retriever.search_kwargs = {}
    mock.as_retriever.return_value = mock_retriever
    return mock


@pytest.fixture
def documents():
    """Create test documents."""
    return [
        Document(page_content="Test content 1", metadata={"source": "doc1.pdf"}),
        Document(page_content="Test content 2", metadata={"source": "doc2.pdf"}),
    ]


@pytest.fixture
def retriever(mock_vectorstore, documents, config):
    """Create retriever instance."""
    with patch("weldscope.retrieval.retriever.BM25Retriever"):
        return HybridRetriever(mock_vectorstore, documents, config)


def test_retriever_initialization(retriever):
    """Retriever initializes with vectorstore and documents."""
    assert retriever.vectorstore is not None
    assert retriever.config is not None
    assert retriever.bm25_retriever is not None
    assert retriever.vector_retriever is not None


def test_retriever_initialization_error(mock_vectorstore, documents, config):
    """Retriever raises RetrievalError on initialization failure."""
    with patch("weldscope.retrieval.retriever.BM25Retriever") as mock_bm25:
        mock_bm25.from_documents.side_effect = Exception("Init error")

        with pytest.raises(RetrievalError, match="Failed to initialize"):
            HybridRetriever(mock_vectorstore, documents, config)


def test_retrieve_returns_documents(retriever):
    """Retrieve returns documents from hybrid search."""
    bm25_docs = [Document(page_content="BM25 result", metadata={"source": "doc1.pdf"})]
    vector_docs = [Document(page_content="Vector result", metadata={"source": "doc2.pdf"})]

    retriever.bm25_retriever.invoke = Mock(return_value=bm25_docs)
    retriever.vector_retriever.invoke = Mock(return_value=vector_docs)

    results = retriever.retrieve("test query")

    assert len(results) <= retriever.config.retrieval_k
    assert all(isinstance(doc, Document) for doc in results)
    assert all("score" in doc.metadata for doc in results)


def test_retrieve_adds_scores(retriever):
    """Retrieve adds scores to merged documents."""
    bm25_docs = [Document(page_content="Result 1", metadata={"source": "doc1.pdf"})]
    vector_docs = [Document(page_content="Result 2", metadata={"source": "doc2.pdf"})]

    retriever.bm25_retriever.invoke = Mock(return_value=bm25_docs)
    retriever.vector_retriever.invoke = Mock(return_value=vector_docs)

    results = retriever.retrieve("test query")

    assert all("score" in doc.metadata for doc in results)
    assert all(isinstance(doc.metadata["score"], float) for doc in results)


def test_retrieve_with_custom_k(retriever):
    """Retrieve accepts custom k parameter."""
    retriever.bm25_retriever.invoke = Mock(return_value=[])
    retriever.vector_retriever.invoke = Mock(return_value=[])

    retriever.retrieve("test query", k=10)

    assert retriever.bm25_retriever.k == 10
    assert retriever.vector_retriever.search_kwargs["k"] == 10


def test_retrieve_with_filters(retriever):
    """Retrieve accepts metadata filters."""
    retriever.bm25_retriever.invoke = Mock(return_value=[])
    retriever.vector_retriever.invoke = Mock(return_value=[])
    filters = {"source": "doc1.pdf"}

    retriever.retrieve("test query", filters=filters)

    assert retriever.vector_retriever.search_kwargs["filter"] == filters


def test_retrieve_raises_on_error(retriever):
    """Retrieve raises RetrievalError on failure."""
    retriever.bm25_retriever.invoke = Mock(side_effect=Exception("Retrieval error"))

    with pytest.raises(RetrievalError, match="Retrieval failed"):
        retriever.retrieve("test query")


def test_retrieve_vector_only(retriever):
    """Retrieve vector only uses vectorstore."""
    mock_results = [
        (Document(page_content="Result 1"), 0.9),
        (Document(page_content="Result 2"), 0.8),
    ]
    retriever.vectorstore.similarity_search_with_score = Mock(return_value=mock_results)

    results = retriever.retrieve_vector_only("test query")

    assert len(results) == 2
    retriever.vectorstore.similarity_search_with_score.assert_called_once()


def test_retrieve_bm25_only(retriever):
    """Retrieve BM25 only uses BM25 retriever."""
    mock_docs = [Document(page_content="Result 1")]
    retriever.bm25_retriever.invoke = Mock(return_value=mock_docs)

    results = retriever.retrieve_bm25_only("test query")

    assert len(results) == 1
    retriever.bm25_retriever.invoke.assert_called_once_with("test query")
