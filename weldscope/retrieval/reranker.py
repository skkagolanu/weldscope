"""Cross-encoder reranker for retrieved documents."""

from typing import List, Optional
from threading import RLock
from time import perf_counter
from langchain_core.documents import Document
from weldscope.core.local_models import LocalModel

from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import RetrievalError


class Reranker:
    """Rerank documents using cross-encoder model."""

    def __init__(self, config: RAGConfig):
        self._lock = RLock()
        self.config = config
        self._model: Optional[LocalModel] = None

    def _get_model(self) -> LocalModel:
        """Initialize cross-encoder model."""
        if self._model is None:
            try:
                self._model = LocalModel(self.config.reranker_model)
            except Exception as e:
                raise RetrievalError(f"Failed to initialize reranker: {e}")

        return self._model

    def rerank(
        self, query: str, documents: List[Document], top_k: Optional[int] = None, *, timings=None
    ) -> List[Document]:
        with self._lock:
            return self._rerank(query, documents, top_k, timings=timings)

    def _rerank(
        self, query: str, documents: List[Document], top_k: Optional[int] = None, *, timings=None
    ) -> List[Document]:
        """Rerank documents by relevance to query."""
        if not documents:
            return []

        try:
            cold = self._model is None
            load_started = perf_counter()
            model = self._get_model()
            if timings is not None:
                timings["reranker_load_ms"] = (
                    (perf_counter() - load_started) * 1000 if cold else 0.0
                )
            top_k = top_k or self.config.retrieval_k

            pairs = [(query, doc.page_content) for doc in documents]
            score_started = perf_counter()
            scores = model.predict(pairs)
            if timings is not None:
                timings["reranker_compute_ms"] = (perf_counter() - score_started) * 1000

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
            pairs = [(query, text) for text in texts]
            scores = model.predict(pairs)
            return [float(s) for s in scores]
        except Exception as e:
            raise RetrievalError(f"Scoring failed: {e}")
