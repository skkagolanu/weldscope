"""Current configuration and validation tests."""

import pytest
from pydantic import ValidationError
from weldscope.core.config import RAGConfig


def test_config_allows_local_conversation_without_credentials():
    assert RAGConfig(_env_file=None, groq_api_key=None).groq_api_key is None


def test_blank_credential_is_unset():
    assert RAGConfig(groq_api_key=" ").groq_api_key is None


def test_current_defaults():
    config = RAGConfig()
    assert config.generation_provider == "groq"
    assert config.generation_model == "openai/gpt-oss-20b"
    assert config.retrieval_k == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 1.5},
        {"chunk_size": 50},
        {"query_strategy": "invalid"},
        {"llm_provider": "huggingface"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValidationError):
        RAGConfig(**kwargs)
