"""Configuration management with Pydantic validation."""

from pydantic_settings import BaseSettings
from pydantic import SecretStr, Field
from typing import Optional, Literal


class RAGConfig(BaseSettings):
    """RAG system configuration with validation."""

    hf_token: SecretStr = Field(..., description="HuggingFace API token")
    model_name: str = Field(default="meta-llama/Llama-3.1-8B-Instruct")
    temperature: float = Field(default=0.05, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)

    embedding_model: str = Field(default="sentence-transformers/paraphrase-MiniLM-L3-v2")

    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)

    retrieval_k: int = Field(default=5, ge=1, le=20)
    rerank_top_k: int = Field(default=3, ge=1, le=10)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    query_strategy: Literal["multi_query", "hyde", "step_back"] = "multi_query"
    num_query_variations: int = Field(default=3, ge=1, le=5)

    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    faithfulness_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    cache_enabled: bool = True
    cache_similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    max_latency_ms: int = Field(default=2000, ge=100)

    langfuse_enabled: bool = False
    langfuse_public_key: Optional[SecretStr] = None
    langfuse_secret_key: Optional[SecretStr] = None

    vectorstore_path: str = "./vectorstore"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }
