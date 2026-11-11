"""Test configuration module."""

import os
import pytest
from pydantic import ValidationError
from weldscope.core.config import RAGConfig


def test_config_requires_hf_token(monkeypatch):
    """HF token is required."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValidationError):
        RAGConfig()


def test_config_validates_temperature():
    """Temperature must be between 0 and 1."""
    with pytest.raises(ValidationError):
        RAGConfig(hf_token="test", temperature=1.5)


def test_config_validates_chunk_size():
    """Chunk size must be within bounds."""
    with pytest.raises(ValidationError):
        RAGConfig(hf_token="test", chunk_size=50)


def test_config_defaults():
    """Default values are set correctly."""
    config = RAGConfig(hf_token="test_token")
    assert config.model_name == "meta-llama/Llama-3.1-8B-Instruct"
    assert config.temperature == 0.05
    assert config.retrieval_k == 5
    assert config.cache_enabled is True


def test_config_query_strategy_validation():
    """Query strategy must be valid enum."""
    with pytest.raises(ValidationError):
        RAGConfig(hf_token="test", query_strategy="invalid")
