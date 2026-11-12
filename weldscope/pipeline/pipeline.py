"""Conversation-aware RAG orchestration with isolated retrieval queries."""

from typing import Callable, Iterator, Optional, List
from time import perf_counter
from uuid import uuid4
from weldscope.monitoring.performance import PerformanceMonitor
from langchain_core.documents import Document

from weldscope.core.config import RAGConfig
from weldscope.core.models import RAGResponse
from weldscope.core.exceptions import RetrievalError
from weldscope.conversation import completed_history, plan_conversation
from weldscope.retrieval.retriever import HybridRetriever
from weldscope.retrieval.reranker import Reranker
from weldscope.generation.generator import LLMGenerator
from weldscope.generation.formatter import ResponseFormatter
from weldscope.observability.tracer import RAGTracer


class RAGPipeline:
    """Route social/product questions before loading or searching documents."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever],
        config: RAGConfig,
        tracer: Optional[RAGTracer] = None,
        retriever_loader=None,
    ):
        self.retriever = retriever
        self.retriever_loader = retriever_loader
        self.config = config
        self.reranker = Reranker(config)
        self.generator = LLMGenerator(config)
        self.tracer = tracer
        self.monitor = PerformanceMonitor(config.max_latency_ms)

    def _format_context(self, documents: List[Document]) -> str:
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page")
            page_info = f" (Page {page + 1})" if page is not None else ""
            context_parts.append(f"[{i}] {source}{page_info}:\n{doc.page_content}")
        return "\n\n".join(context_parts)

    def query(
        self,
        question: str,
        stream: bool = False,
        on_chunk: Optional[Callable[[str], None]] = None,
        *,
        history=None,
        sensor_context="",
    ) -> RAGResponse:
        started = perf_counter()
        history = completed_history(history or [])
        try:
            if self.tracer:
                response = self._query_with_tracing(
                    question, stream, on_chunk, history, sensor_context
                )
            else:
                response = self._query_internal(question, stream, on_chunk, history, sensor_context)
        except Exception as error:
            self.monitor.record_metrics(
                uuid4().hex,
                0,
                0,
                0,
                0,
                0,
                metadata={
                    "status": "error",
                    "error_type": type(error).__name__,
                    "stage_timings_available": False,
                },
                total_latency_ms=(perf_counter() - started) * 1000,
            )
            raise
        response.latency_ms = (perf_counter() - started) * 1000
        timings = response.metadata["timings"]
        timings["total_ms"] = response.latency_ms
        usage = (
            getattr(self.generator, "last_usage", None)
            if response.metadata["route"] == "research"
            else None
        )
        usage = usage if isinstance(usage, dict) else None
        response.metadata["token_usage"] = usage
        self.monitor.record_metrics(
            uuid4().hex,
            timings["retrieval_ms"],
            timings["reranking_ms"],
            timings["generation_ms"],
            response.metadata["retrieved_count"],
            response.metadata["reranked_count"],
            tokens_generated=usage.get("completion_tokens") if usage else None,
            metadata={
                "route": response.metadata["route"],
                "status": "success",
                "first_token_ms": timings["first_token_ms"],
            },
            total_latency_ms=response.latency_ms,
        )
        return response

    def _query_internal(
        self, question: str, stream: bool, on_chunk=None, history=None, sensor_context=""
    ) -> RAGResponse:
        started = perf_counter()
        timings = dict(
            initialization_ms=0.0,
            retrieval_ms=0.0,
            reranking_ms=0.0,
            generation_ms=0.0,
            render_ms=0.0,
            first_token_ms=None,
            first_token_total_ms=None,
        )
        plan = plan_conversation(question, history)
        retrieved_docs = []
        reranked_docs = []
        if plan.kind in {"local", "out_of_scope"}:
            answer = plan.answer
            if plan.kind == "local" and "model" in question.lower():
                answer += f"\n\nConfigured generation model: `{self.config.generation_model}`, served by Groq."
            if on_chunk:
                on_chunk(answer)
        elif plan.kind == "research":
            if self.retriever is None and self.retriever_loader:
                init_started = perf_counter()
                self.retriever = self.retriever_loader()
                timings["initialization_ms"] = (perf_counter() - init_started) * 1000
            if self.retriever is None:
                raise RetrievalError("The document collection is unavailable.")
            retrieval_started = perf_counter()
            retrieved_docs = self.retriever.retrieve(plan.retrieval_query)
            timings["retrieval_ms"] = (perf_counter() - retrieval_started) * 1000
            if self.tracer:
                self.tracer.log_retrieval(
                    plan.retrieval_query, len(retrieved_docs), (perf_counter() - started) * 1000
                )
            if not retrieved_docs:
                raise RetrievalError("No documents retrieved")
            rerank_started = perf_counter()
            reranked_docs = self.reranker.rerank(
                plan.retrieval_query, retrieved_docs, top_k=self.config.rerank_top_k
            )
            timings["reranking_ms"] = (perf_counter() - rerank_started) * 1000
            context = self._format_context(reranked_docs)
            if sensor_context:
                context += "\n\nBundled example sensor log (not a live feed):\n" + sensor_context
            # Only referential follow-ups carry the focused topic into generation.
            prompt = self.generator.create_prompt(plan.retrieval_query, context)
            generation_started = perf_counter()
            if stream:
                chunks = []
                for chunk in self.generator.stream(prompt, history=history):
                    if timings["first_token_ms"] is None:
                        timings["first_token_ms"] = (perf_counter() - generation_started) * 1000
                        timings["first_token_total_ms"] = (perf_counter() - started) * 1000
                    chunks.append(chunk)
                    if on_chunk:
                        render_started = perf_counter()
                        on_chunk(chunk)
                        timings["render_ms"] += (perf_counter() - render_started) * 1000
                answer = "".join(chunks)
            else:
                answer = self.generator.generate(prompt, history=history)
            timings["generation_ms"] = (perf_counter() - generation_started) * 1000
            if self.tracer:
                self.tracer.log_generation(
                    prompt, answer, None, (perf_counter() - generation_started) * 1000
                )
        return ResponseFormatter.create_response(
            query=question,
            answer=answer,
            sources=ResponseFormatter.format_sources(reranked_docs),
            retrieved_count=len(retrieved_docs),
            reranked_count=len(reranked_docs),
            latency_ms=(perf_counter() - started) * 1000,
            metadata={
                "route": plan.kind,
                "retrieval_query": plan.retrieval_query,
                "timings": timings,
            },
        )

    def _query_with_tracing(self, question, stream, on_chunk=None, history=None, sensor_context=""):
        @self.tracer.trace_query(name="weldscope_pipeline")
        def traced_query():
            response = self._query_internal(question, stream, on_chunk, history, sensor_context)
            if self.tracer.client:
                response.metadata["trace_id"] = self.tracer.client.get_current_trace_id()
            return response

        return traced_query()

    def stream_query(self, question: str, *, history=None) -> Iterator[str]:
        """Legacy chunk API; use query(on_chunk=...) when citations are needed."""
        history = completed_history(history or [])
        plan = plan_conversation(question, history)
        if plan.kind in {"local", "out_of_scope"}:
            yield plan.answer
            return
        if self.retriever is None and self.retriever_loader:
            self.retriever = self.retriever_loader()
        if self.retriever is None:
            raise RetrievalError("The document collection is unavailable.")
        docs = self.retriever.retrieve(plan.retrieval_query)
        if not docs:
            raise RetrievalError("No documents retrieved")
        ranked = self.reranker.rerank(plan.retrieval_query, docs, top_k=self.config.rerank_top_k)
        prompt = self.generator.create_prompt(plan.retrieval_query, self._format_context(ranked))
        yield from self.generator.stream(prompt, history=history)
