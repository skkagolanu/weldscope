"""Test LLM generator."""

import pytest
from unittest.mock import Mock, patch
from weldscope.generation.generator import LLMGenerator
from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import GenerationError


@pytest.fixture
def config():
    """Create test config."""
    return RAGConfig(hf_token="test_token")


@pytest.fixture
def generator(config):
    """Create generator instance."""
    return LLMGenerator(config)


def test_generator_initialization(generator):
    """Generator initializes with config."""
    assert generator.config is not None
    assert generator._llm is None
    assert generator._chat_model is None


@patch("weldscope.generation.generator.ChatHuggingFace")
@patch("weldscope.generation.generator.HuggingFaceEndpoint")
def test_get_llm_initializes_endpoint(mock_endpoint, mock_chat, generator):
    """Get LLM initializes HuggingFace endpoint."""
    mock_endpoint.return_value = Mock()
    mock_chat.return_value = Mock()

    llm = generator._get_llm()

    assert llm is not None
    mock_endpoint.assert_called_once()
    mock_chat.assert_called_once()
    assert generator._llm is not None


@patch("weldscope.generation.generator.ChatHuggingFace")
@patch("weldscope.generation.generator.HuggingFaceEndpoint")
def test_get_llm_caches_instance(mock_endpoint, mock_chat, generator):
    """Get LLM caches endpoint instance."""
    mock_endpoint.return_value = Mock()
    mock_chat.return_value = Mock()

    llm1 = generator._get_llm()
    llm2 = generator._get_llm()

    assert llm1 is llm2
    mock_endpoint.assert_called_once()
    mock_chat.assert_called_once()


@patch("weldscope.generation.generator.ChatHuggingFace")
@patch("weldscope.generation.generator.HuggingFaceEndpoint")
def test_get_llm_raises_on_error(mock_endpoint, mock_chat, generator):
    """Get LLM raises GenerationError on initialization failure."""
    mock_endpoint.side_effect = Exception("API error")

    with pytest.raises(GenerationError, match="Failed to initialize LLM"):
        generator._get_llm()


@patch("weldscope.generation.generator.ChatHuggingFace")
@patch("weldscope.generation.generator.HuggingFaceEndpoint")
def test_generate_returns_response(mock_endpoint, mock_chat, generator):
    """Generate returns LLM response."""
    mock_response = Mock()
    mock_response.content = "Test answer"
    mock_chat_model = Mock()
    mock_chat_model.invoke.return_value = mock_response
    mock_endpoint.return_value = Mock()
    mock_chat.return_value = mock_chat_model

    result = generator.generate("Test prompt")

    assert result == "Test answer"


@patch("weldscope.generation.generator.ChatHuggingFace")
@patch("weldscope.generation.generator.HuggingFaceEndpoint")
def test_generate_raises_on_error(mock_endpoint, mock_chat, generator):
    """Generate raises GenerationError on failure."""
    mock_chat_model = Mock()
    mock_chat_model.invoke.side_effect = Exception("API error")
    mock_endpoint.return_value = Mock()
    mock_chat.return_value = mock_chat_model

    with pytest.raises(GenerationError, match="Generation failed"):
        generator.generate("Test prompt")


@patch("weldscope.generation.generator.ChatHuggingFace")
@patch("weldscope.generation.generator.HuggingFaceEndpoint")
def test_stream_yields_chunks(mock_endpoint, mock_chat, generator):
    """Stream yields response chunks."""
    chunk1, chunk2, chunk3 = Mock(), Mock(), Mock()
    chunk1.content = "chunk1"
    chunk2.content = "chunk2"
    chunk3.content = "chunk3"
    mock_chat_model = Mock()
    mock_chat_model.stream.return_value = iter([chunk1, chunk2, chunk3])
    mock_endpoint.return_value = Mock()
    mock_chat.return_value = mock_chat_model

    chunks = list(generator.stream("Test prompt"))

    assert chunks == ["chunk1", "chunk2", "chunk3"]


@patch("weldscope.generation.generator.ChatHuggingFace")
@patch("weldscope.generation.generator.HuggingFaceEndpoint")
def test_stream_raises_on_error(mock_endpoint, mock_chat, generator):
    """Stream raises GenerationError on failure."""
    mock_chat_model = Mock()
    mock_chat_model.stream.side_effect = Exception("Streaming error")
    mock_endpoint.return_value = Mock()
    mock_chat.return_value = mock_chat_model

    with pytest.raises(GenerationError, match="Streaming failed"):
        list(generator.stream("Test prompt"))


def test_create_prompt_uses_default_template(generator):
    """Create prompt uses default FSW template."""
    prompt = generator.create_prompt(query="What causes defects?", context="Context text")

    assert "friction stir welding" in prompt
    assert "ISO 25239" in prompt
    assert "What causes defects?" in prompt
    assert "Context text" in prompt


def test_create_prompt_uses_custom_template(generator):
    """Create prompt accepts custom template."""
    custom_template = "Q: {question}\nC: {context}\nA:"

    prompt = generator.create_prompt(
        query="Test question", context="Test context", template=custom_template
    )

    assert "Q: Test question" in prompt
    assert "C: Test context" in prompt
