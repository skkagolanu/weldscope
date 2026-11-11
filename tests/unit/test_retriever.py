"""Real BM25S ranking and hybrid result isolation."""

from unittest.mock import Mock
import pytest
from langchain_core.documents import Document
from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import RetrievalError
from weldscope.retrieval.retriever import HybridRetriever


@pytest.fixture
def documents():
    return [
        Document(
            page_content="Wormhole defects result from low heat input",
            metadata={"source": "one.pdf", "page": 0},
        ),
        Document(
            page_content="Surface cracks involve weld cooling",
            metadata={"source": "two.pdf", "page": 1},
        ),
    ]


@pytest.fixture
def retriever(documents):
    store = Mock()
    store.similarity_search.return_value = documents
    return HybridRetriever(store, documents, RAGConfig())


def test_keyword_search_ranks_relevant_document(retriever):
    assert "Wormhole" in retriever.retrieve_bm25_only("wormhole heat", k=1)[0].page_content


def test_hybrid_search_has_scores_without_mutating_shared_documents(retriever, documents):
    results = retriever.retrieve("wormhole", k=2)
    assert len(results) == 2
    assert all("score" in doc.metadata for doc in results)
    assert all("score" not in doc.metadata for doc in documents)


def test_filters_apply_to_both_paths_and_do_not_persist(retriever, documents):
    retriever.vectorstore.similarity_search.return_value = documents[:1]
    results = retriever.retrieve("wormhole", filters={"source": "one.pdf"})
    assert all(doc.metadata["source"] == "one.pdf" for doc in results)
    retriever.retrieve("cracks")
    assert retriever.vectorstore.similarity_search.call_args.kwargs["filter"] is None


def test_same_text_on_different_pages_is_citable_separately(retriever):
    docs = [
        Document(page_content="Text", metadata={"source": "one.pdf", "page": i}) for i in range(2)
    ]
    assert len(retriever._merge_results(docs, [])) == 2


def test_vector_only_uses_store(retriever):
    assert retriever.retrieve_vector_only("question", k=1)
    retriever.vectorstore.similarity_search.assert_called_with("question", k=1)


def test_empty_collection_rejected():
    with pytest.raises(RetrievalError, match="empty"):
        HybridRetriever(Mock(), [], RAGConfig())


def test_unsupported_filter_rejected(retriever):
    with pytest.raises(RetrievalError, match="exact metadata"):
        retriever.retrieve("question", filters={"source": {"$in": ["one.pdf"]}})


def test_store_failure_is_reported(retriever):
    retriever.vectorstore.similarity_search.side_effect = RuntimeError("unavailable")
    with pytest.raises(RetrievalError, match="Retrieval failed"):
        retriever.retrieve("question")
