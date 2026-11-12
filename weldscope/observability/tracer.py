"""Langfuse tracer for RAG observability."""

from typing import Optional, Dict, Any, List
from functools import wraps
import time
from langfuse import Langfuse

from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import RAGException


class RAGTracer:
    """Observability tracer for RAG pipeline using Langfuse."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.enabled = config.langfuse_enabled
        self.client: Optional[Langfuse] = None

        if self.enabled:
            if not config.langfuse_public_key or not config.langfuse_secret_key:
                raise RAGException("Langfuse keys required when langfuse_enabled=True")

            self.client = Langfuse(
                public_key=config.langfuse_public_key.get_secret_value(),
                secret_key=config.langfuse_secret_key.get_secret_value(),
                host="https://cloud.langfuse.com",
            )

    def trace_query(self, name: str = "rag_query", metadata=None):
        """Decorator to trace RAG query execution."""

        def decorator(func):
            if not self.enabled:
                return func

            @wraps(func)
            def wrapper(*args, **kwargs):
                # SDK decorates a generator as a context manager; exercised by the real-span integration test.
                with self.client.start_as_current_observation(
                    name=name, metadata=metadata
                ):  # pylint: disable=not-context-manager
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def log_retrieval(self, query: str, num_chunks: int, latency_ms: float):
        """Log retrieval metrics."""
        if not self.enabled or not self.client:
            return

        self.client.start_observation(
            as_type="span",
            name="retrieval",
            input=query if self.config.trace_content else {"characters": len(query)},
            output={"num_chunks": num_chunks},
            metadata={"latency_ms": latency_ms},
        ).end()

    def log_generation(
        self, prompt: str, answer: str, tokens_used: Optional[int], latency_ms: float
    ):
        """Log generation metrics."""
        if not self.enabled or not self.client:
            return

        self.client.start_observation(
            as_type="generation",
            name="generation",
            input=prompt if self.config.trace_content else {"characters": len(prompt)},
            output=answer if self.config.trace_content else {"characters": len(answer)},
            usage_details={"total": tokens_used} if tokens_used is not None else None,
            metadata={"latency_ms": latency_ms},
        ).end()

    def score_feedback(self, trace_id: str, score: float, comment: Optional[str] = None):
        """Log user feedback score."""
        if not self.enabled or not self.client:
            return

        self.client.create_score(
            trace_id=trace_id, name="user_feedback", value=score, comment=comment
        )

    def flush(self):
        """Flush pending traces."""
        if self.enabled and self.client:
            self.client.flush()
