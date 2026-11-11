"""Cross-encoder reranker for retrieved documents."""

from typing import List, Tuple
from threading import RLock
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import RetrievalError


class Reranker:
    """Rerank documents using cross-encoder model."""

    def __init__(self, config: RAGConfig):
        self._lock = RLock()
        self.config = config
        self._model = None

    def _get_model(self) -> CrossEncoder:
        """Initialize cross-encoder model."""
        if self._model is None:
            try:
                self._model = CrossEncoder(self.config.reranker_model)
            except Exception as e:
                raise RetrievalError(f"Failed to initialize reranker: {e}")

        return self._model

    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        with self._lock:
            return self._rerank(query, documents, top_k)

    def _rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        """Rerank documents by relevance to query."""
        if not documents:
            return []

        try:
            model = self._get_model()
            top_k = top_k or self.config.retrieval_k

            pairs = [[query, doc.page_content] for doc in documents]
            scores = model.predict(pairs)

            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for doc, score in doc_scores[:top_k]:
                result = Document(
                    id=doc.id, page_content=doc.page_content, metadata=dict(doc.metadata)
                )
                result.metadata["rerank_score"] = float(score)
                reranked.append(result)

            return reranked
        except Exception as e:
            raise RetrievalError(f"Reranking failed: {e}")

    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        with self._lock:
            return self._score_pairs(query, texts)

    def _score_pairs(self, query: str, texts: List[str]) -> List[float]:
        """Score query-text pairs without document wrapping."""
        try:
            model = self._get_model()
            pairs = [[query, text] for text in texts]
            scores = model.predict(pairs)
            return [float(s) for s in scores]
        except Exception as e:
            raise RetrievalError(f"Scoring failed: {e}")
