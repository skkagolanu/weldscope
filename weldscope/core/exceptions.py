"""Core exceptions for RAG system."""


class RAGException(Exception):
    """Base exception for RAG system."""

    pass


class ConfigurationError(RAGException):
    """Configuration validation failed."""

    pass


class RetrievalError(RAGException):
    """Document retrieval failed."""

    pass


class GenerationError(RAGException):
    """LLM generation failed."""

    pass


class ValidationError(RAGException):
    """Response validation failed."""

    pass


class CacheError(RAGException):
    """Cache operation failed."""

    pass
