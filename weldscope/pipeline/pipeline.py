"""Conversation-aware RAG orchestration with isolated retrieval queries."""

from typing import Callable, Iterator, Optional, List, cast
from time import perf_counter
from uuid import uuid4
from datetime import datetime, timezone
from weldscope.monitoring.telemetry import record_event
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
from weldscope.generation.validation import check_answer
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
        request_id = uuid4().hex
        history = completed_history(history or [])
        try:
            if self.tracer:
                response = self._query_with_tracing(
                    question, stream, on_chunk, history, sensor_context, request_id=request_id
                )
            else:
                response = self._query_internal(question, stream, on_chunk, history, sensor_context)
        except Exception as error:
            record_event(
                self.config.telemetry_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "request_id": request_id,
                    "trace_id": request_id,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "timings": {"total_ms": (perf_counter() - started) * 1000},
                },
            )
            self.monitor.record_metrics(
                request_id,
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
        response.metadata["request_id"] = request_id
        if (
            not isinstance(response.metadata.get("trace_id"), str)
            or not response.metadata["trace_id"]
        ):
            response.metadata["trace_id"] = request_id
        record_event(
            self.config.telemetry_path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "trace_id": response.metadata["trace_id"],
                "status": "success",
                "route": response.metadata["route"],
                "timings": timings,
                "cold": bool(timings["reranker_load_ms"]),
                "retrieved_count": response.metadata.get("retrieved_count", 0),
                "reranked_count": response.metadata.get("reranked_count", 0),
                "model": self.config.generation_model,
                "token_usage": usage,
                "answer_check_status": response.metadata.get("answer_checks", {}).get("status"),
            },
        )
        self.monitor.record_metrics(
            request_id,
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
            reranker_load_ms=0.0,
            reranker_compute_ms=0.0,
            generation_ms=0.0,
            render_ms=0.0,
            first_token_ms=None,
            first_token_total_ms=None,
            first_visible_ms=None,
            validation_ms=0.0,
        )
        plan = plan_conversation(question, history)
        retrieved_docs = []
        reranked_docs = []
        answer_checks = {"status": "not_applicable"}
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
                plan.retrieval_query,
                retrieved_docs,
                top_k=self.config.rerank_top_k,
                timings=timings,
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
                answer = "".join(chunks)
            else:
                answer = self.generator.generate(prompt, history=history)
            timings["generation_ms"] = (perf_counter() - generation_started) * 1000
            validation_started = perf_counter()
            answer_checks = check_answer(answer, reranked_docs)
            if answer_checks["status"] == "withheld":
                answer = "The generated reply contained invalid source citations and has been withheld. Please rephrase the corpus question."
            timings["validation_ms"] = (perf_counter() - validation_started) * 1000
            if on_chunk:
                render_started = perf_counter()
                on_chunk(answer)
                timings["render_ms"] = (perf_counter() - render_started) * 1000
                timings["first_visible_ms"] = (perf_counter() - started) * 1000
            if self.tracer:
                reported_usage = getattr(self.generator, "last_usage", None)
                total_tokens = (
                    reported_usage.get("total_tokens") if isinstance(reported_usage, dict) else None
                )
                self.tracer.log_generation(
                    prompt, answer, total_tokens, (perf_counter() - generation_started) * 1000
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
                "answer_checks": answer_checks,
                "retrieval_query": plan.retrieval_query,
                "timings": timings,
            },
        )

    def _query_with_tracing(
        self, question, stream, on_chunk=None, history=None, sensor_context="", request_id=None
    ) -> RAGResponse:
        assert self.tracer is not None

        @self.tracer.trace_query(name="weldscope_pipeline", metadata={"request_id": request_id})
        def traced_query():
            response = self._query_internal(question, stream, on_chunk, history, sensor_context)
            if self.tracer.client:
                response.metadata["trace_id"] = self.tracer.client.get_current_trace_id()
            return response

        return cast(RAGResponse, traced_query())

    def stream_query(self, question: str, *, history=None) -> Iterator[str]:
        """Yield the final citation-checked reply; raw provider tokens remain internal."""
        response = self.query(question, stream=True, history=history)
        yield response.answer
