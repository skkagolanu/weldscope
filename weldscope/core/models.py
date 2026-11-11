"""Data models for RAG responses."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class Source:
    """Source document citation."""

    doc_name: str
    page_num: Optional[int]
    chunk_text: str
    relevance_score: float
    metadata: Dict[str, Any]


@dataclass
class RAGResponse:
    """Complete RAG pipeline response."""

    query: str
    answer: str
    sources: List[Source]
    confidence: float = 1.0
    faithfulness_score: Optional[float] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
