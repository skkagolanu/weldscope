"""Response formatter with source attribution."""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from weldscope.core.models import Source, RAGResponse


class ResponseFormatter:
    """Format RAG responses with source citations."""

    @staticmethod
    def format_sources(documents: List[Document]) -> List[Source]:
        """Extract source citations from retrieved documents."""
        sources = []

        for doc in documents:
            source = Source(
                doc_name=doc.metadata.get("source", "Unknown"),
                page_num=doc.metadata.get("page"),
                chunk_text=doc.page_content,
                fusion_score=doc.metadata.get("score"),
                rerank_score=doc.metadata.get("rerank_score"),
                chunk_id=doc.id or doc.metadata.get("chunk_id"),
                metadata=doc.metadata,
            )
            sources.append(source)

        return sources

    @staticmethod
    def create_response(
        query: str,
        answer: str,
        sources: List[Source],
        retrieved_count: int = 0,
        reranked_count: int = 0,
        latency_ms: float = 0.0,
        confidence: Optional[float] = None,
        faithfulness_score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        """Create complete RAG response with citations."""
        response_metadata = metadata or {}
        response_metadata.update(
            {"retrieved_count": retrieved_count, "reranked_count": reranked_count}
        )

        return RAGResponse(
            query=query,
            answer=answer,
            sources=sources,
            confidence=confidence,
            faithfulness_score=faithfulness_score,
            latency_ms=latency_ms,
            metadata=response_metadata,
        )

    @staticmethod
    def format_citation(source: Source, index: int) -> str:
        """Format single source citation for display."""
        page_info = f", Page {source.page_num + 1}" if source.page_num is not None else ""
        score = source.rerank_score if source.rerank_score is not None else source.fusion_score
        label = "Reranker" if source.rerank_score is not None else "Fusion"
        score_info = f" ({label} score: {score:.3f})" if score is not None else ""
        return f"[{index}] {source.doc_name}{page_info}{score_info}"
