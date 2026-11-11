"""Prompt construction and provider lifecycle tests."""

from unittest.mock import Mock
import pytest
from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import GenerationError
from weldscope.generation.generator import LLMGenerator


def test_prompt_requires_source_support_and_does_not_assume_standards():
    prompt = LLMGenerator(RAGConfig()).create_prompt("What causes defects?", "Evidence")
    assert "Do not assume" in prompt
    assert "invent numerical settings" in prompt
    assert "Evidence" in prompt
    assert "What causes defects?" in prompt


def test_custom_prompt():
    assert LLMGenerator(RAGConfig()).create_prompt("Q", "C", "{question}: {context}") == "Q: C"


def test_empty_answer_is_an_error():
    generator = LLMGenerator(RAGConfig(groq_api_key="test"))
    generator._groq = Mock()
    generator._groq.chat.completions.create.return_value.choices = [Mock(message=Mock(content=""))]
    with pytest.raises(GenerationError, match="no answer"):
        generator.generate("Question")
    generator.close()
    generator._groq.close.assert_called_once()


def test_no_content_in_stream_is_an_error_and_closes_stream():
    generator = LLMGenerator(RAGConfig(groq_api_key="test"))
    response = Mock()
    response.__iter__ = Mock(return_value=iter([]))
    generator._groq = Mock()
    generator._groq.chat.completions.create.return_value = response
    with pytest.raises(GenerationError, match="no answer"):
        list(generator.stream("Question"))
    response.close.assert_called_once()
