"""RAG pipeline orchestrator."""

from typing import Callable, Iterator, Optional, List
from time import perf_counter
from langchain_core.documents import Document

from weldscope.core.config import RAGConfig
from weldscope.core.models import RAGResponse
from weldscope.core.exceptions import RetrievalError, GenerationError
from weldscope.retrieval.retriever import HybridRetriever
from weldscope.retrieval.reranker import Reranker
from weldscope.generation.generator import LLMGenerator
from weldscope.generation.formatter import ResponseFormatter
from weldscope.observability.tracer import RAGTracer


class RAGPipeline:
    """End-to-end RAG pipeline."""

    def __init__(
        self, retriever: HybridRetriever, config: RAGConfig, tracer: Optional[RAGTracer] = None
    ):
        self.retriever = retriever
        self.config = config
        self.reranker = Reranker(config)
        self.generator = LLMGenerator(config)
        self.tracer = tracer

    def _format_context(self, documents: List[Document]) -> str:
        """Format documents into context string."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f" (Page {page})" if page else ""
            context_parts.append(f"[{i}] {source}{page_info}:\n{doc.page_content}")

        return "\n\n".join(context_parts)

    def query(
        self, question: str, stream: bool = False, on_chunk: Optional[Callable[[str], None]] = None
    ) -> RAGResponse:
        """Execute RAG query."""
        if self.tracer:
            return self._query_with_tracing(question, stream, on_chunk)
        return self._query_internal(question, stream, on_chunk)

    def _query_internal(self, question: str, stream: bool, on_chunk=None) -> RAGResponse:
        """Internal query execution."""
        started = perf_counter()
        retrieved_docs = self.retriever.retrieve(question)

        if self.tracer:
            self.tracer.log_retrieval(
                question, len(retrieved_docs), (perf_counter() - started) * 1000
            )

        if not retrieved_docs:
            raise RetrievalError("No documents retrieved")

        reranked_docs = self.reranker.rerank(
            question, retrieved_docs, top_k=self.config.rerank_top_k
        )

        context = self._format_context(reranked_docs)
        prompt = self.generator.create_prompt(question, context)

        generation_started = perf_counter()
        if stream:
            answer_chunks = []
            for chunk in self.generator.stream(prompt):
                answer_chunks.append(chunk)
                if on_chunk:
                    on_chunk(chunk)
            answer = "".join(answer_chunks)
        else:
            answer = self.generator.generate(prompt)

        if self.tracer:
            self.tracer.log_generation(
                prompt, answer, None, (perf_counter() - generation_started) * 1000
            )

        sources = ResponseFormatter.format_sources(reranked_docs)

        return ResponseFormatter.create_response(
            query=question,
            answer=answer,
            sources=sources,
            retrieved_count=len(retrieved_docs),
            reranked_count=len(reranked_docs),
            latency_ms=(perf_counter() - started) * 1000,
        )

    def _query_with_tracing(self, question: str, stream: bool, on_chunk=None) -> RAGResponse:
        """Query execution with Langfuse tracing."""

        @self.tracer.trace_query(name="weldscope_pipeline")
        def traced_query():
            response = self._query_internal(question, stream, on_chunk)
            if self.tracer.client:
                response.metadata["trace_id"] = self.tracer.client.get_current_trace_id()
            return response

        return traced_query()

    def stream_query(self, question: str) -> Iterator[str]:
        """Stream RAG query response."""
        retrieved_docs = self.retriever.retrieve(question)

        if not retrieved_docs:
            raise RetrievalError("No documents retrieved")

        reranked_docs = self.reranker.rerank(
            question, retrieved_docs, top_k=self.config.rerank_top_k
        )

        context = self._format_context(reranked_docs)
        prompt = self.generator.create_prompt(question, context)

        for chunk in self.generator.stream(prompt):
            yield chunk
