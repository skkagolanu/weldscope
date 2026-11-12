"""Configuration management with Pydantic validation."""

from pydantic_settings import BaseSettings
from pydantic import SecretStr, Field, field_validator
from typing import Optional, Literal


class RAGConfig(BaseSettings):
    """RAG system configuration with validation."""

    llm_provider: Literal["groq"] = "groq"
    groq_api_key: Optional[SecretStr] = None
    groq_model: str = "openai/gpt-oss-20b"
    temperature: float = Field(default=0.05, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)

    embedding_model: str = Field(default="sentence-transformers/paraphrase-MiniLM-L3-v2")

    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)

    retrieval_k: int = Field(default=5, ge=1, le=20)
    rerank_top_k: int = Field(default=3, ge=1, le=10)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    max_latency_ms: int = Field(default=2000, ge=100)

    langfuse_enabled: bool = False
    langfuse_public_key: Optional[SecretStr] = None
    langfuse_secret_key: Optional[SecretStr] = None

    vectorstore_path: str = "./vectorstore"
    telemetry_path: str = "artifacts/telemetry/events.jsonl"
    trace_content: bool = False

    @field_validator("groq_api_key", "langfuse_public_key", "langfuse_secret_key", mode="before")
    @classmethod
    def empty_credential_is_unset(cls, value):
        return None if isinstance(value, str) and not value.strip() else value

    @property
    def generation_provider(self) -> str:
        return "groq"

    @property
    def generation_model(self) -> str:
        return self.groq_model

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }
