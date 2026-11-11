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
                relevance_score=doc.metadata.get("score", 0.0),
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
        confidence: float = 1.0,
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
        page_info = f", Page {source.page_num}" if source.page_num else ""
        return f"[{index}] {source.doc_name}{page_info} (Score: {source.relevance_score:.3f})"
