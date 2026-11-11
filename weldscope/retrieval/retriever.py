"""Hybrid retriever combining vector and BM25 search."""

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import RetrievalError


class HybridRetriever:
    """Hybrid retrieval using vector + BM25 search."""

    def __init__(self, vectorstore: Chroma, documents: List[Document], config: RAGConfig):
        self.vectorstore = vectorstore
        self.config = config

        try:
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = config.retrieval_k

            self.vector_retriever = vectorstore.as_retriever(
                search_kwargs={"k": config.retrieval_k}
            )
        except Exception as e:
            raise RetrievalError(f"Failed to initialize hybrid retriever: {e}")

    def _merge_results(
        self, bm25_docs: List[Document], vector_docs: List[Document], bm25_weight: float = 0.5
    ) -> List[Document]:
        """Merge and deduplicate results from both retrievers."""
        doc_scores = {}

        for i, doc in enumerate(bm25_docs):
            key = doc.page_content
            score = bm25_weight * (1.0 / (i + 1))
            doc_scores[key] = (doc, score)

        for i, doc in enumerate(vector_docs):
            key = doc.page_content
            vector_score = (1.0 - bm25_weight) * (1.0 / (i + 1))

            if key in doc_scores:
                existing_doc, existing_score = doc_scores[key]
                doc_scores[key] = (existing_doc, existing_score + vector_score)
            else:
                doc_scores[key] = (doc, vector_score)

        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)

        results = []
        for doc, score in sorted_docs:
            doc.metadata["score"] = score
            results.append(doc)

        return results

    def retrieve(
        self, query: str, k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve documents using hybrid search."""
        try:
            k = k or self.config.retrieval_k

            if k != self.bm25_retriever.k:
                self.bm25_retriever.k = k

            if k != self.vector_retriever.search_kwargs.get("k"):
                self.vector_retriever.search_kwargs["k"] = k

            if filters:
                self.vector_retriever.search_kwargs["filter"] = filters

            bm25_docs = self.bm25_retriever.invoke(query)
            vector_docs = self.vector_retriever.invoke(query)

            merged = self._merge_results(bm25_docs, vector_docs)

            return merged[:k]
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {e}")

    def retrieve_vector_only(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve using vector search only."""
        try:
            k = k or self.config.retrieval_k
            return self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            raise RetrievalError(f"Vector retrieval failed: {e}")

    def retrieve_bm25_only(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve using BM25 search only."""
        try:
            if k:
                self.bm25_retriever.k = k
            return self.bm25_retriever.invoke(query)
        except Exception as e:
            raise RetrievalError(f"BM25 retrieval failed: {e}")
