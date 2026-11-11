"""Hybrid retrieval using maintained BM25S and Chroma integrations."""

from threading import RLock
from typing import List, Optional, Dict, Any
import bm25s
from langchain_core.documents import Document
from langchain_chroma import Chroma

from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import RetrievalError


class HybridRetriever:
    def __init__(self, vectorstore: Chroma, documents: List[Document], config: RAGConfig):
        self.vectorstore = vectorstore
        self.config = config
        self.documents = documents
        self._lock = RLock()
        try:
            if not documents:
                raise ValueError("The document collection is empty")
            self.bm25 = bm25s.BM25()
            self.bm25.index(
                bm25s.tokenize([doc.page_content for doc in documents], show_progress=False),
                show_progress=False,
            )
        except Exception as error:
            raise RetrievalError(f"Failed to initialize hybrid retriever: {error}") from error

    def _merge_results(self, bm25_docs, vector_docs, bm25_weight=0.5):
        scores = {}
        for docs, weight in [(bm25_docs, bm25_weight), (vector_docs, 1 - bm25_weight)]:
            for rank, doc in enumerate(docs, 1):
                # Keep repeated text from different pages independently citable.
                key = (
                    doc.id
                    or doc.metadata.get("chunk_id")
                    or (doc.page_content, doc.metadata.get("source"), doc.metadata.get("page"))
                )
                previous = scores.get(key, (doc, 0))[1]
                scores[key] = (doc, previous + weight / rank)
        results = []
        for doc, score in sorted(scores.values(), key=lambda item: item[1], reverse=True):
            results.append(
                Document(
                    id=doc.id,
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "score": float(score)},
                )
            )
        return results

    def _lexical(self, query, k, filters=None):
        indices, _ = self.bm25.retrieve(
            bm25s.tokenize(query, show_progress=False),
            k=len(self.documents) if filters else min(k, len(self.documents)),
            show_progress=False,
        )
        docs = [self.documents[int(index)] for index in indices[0]]
        if filters:
            docs = [
                doc
                for doc in docs
                if all(doc.metadata.get(key) == value for key, value in filters.items())
            ]
        return docs[:k]

    def retrieve(
        self, query: str, k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        if filters and any(
            key.startswith("$") or isinstance(value, dict) for key, value in filters.items()
        ):
            raise RetrievalError("Hybrid retrieval supports exact metadata filters only")
        try:
            with self._lock:
                k = k or self.config.retrieval_k
                lexical = self._lexical(query, k, filters)
                semantic = self.vectorstore.similarity_search(query, k=k, filter=filters)
                return self._merge_results(lexical, semantic)[:k]
        except Exception as error:
            raise RetrievalError(f"Retrieval failed: {error}") from error

    def retrieve_vector_only(self, query: str, k: Optional[int] = None) -> List[Document]:
        try:
            with self._lock:
                return self.vectorstore.similarity_search(query, k=k or self.config.retrieval_k)
        except Exception as error:
            raise RetrievalError(f"Vector retrieval failed: {error}") from error

    def retrieve_bm25_only(self, query: str, k: Optional[int] = None) -> List[Document]:
        try:
            with self._lock:
                return self._lexical(query, k or self.config.retrieval_k)
        except Exception as error:
            raise RetrievalError(f"BM25 retrieval failed: {error}") from error
