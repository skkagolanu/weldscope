"""Exercise the actual Groq SDK serialization with a local HTTP transport."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from groq import Groq

from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import GenerationError
from weldscope.generation.generator import LLMGenerator, generation_error_message


def generator_with_transport(handler):
    generator = LLMGenerator(RAGConfig(groq_api_key="test-placeholder", llm_provider="groq"))
    generator._groq = Groq(
        api_key="test-placeholder",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return generator


def test_groq_request_uses_supported_model_options_and_structured_history():
    def handler(request):
        body = json.loads(request.content)
        assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
        assert body["model"] == "openai/gpt-oss-20b"
        assert body["include_reasoning"] is False
        assert "reasoning_format" not in body  # Unsupported by GPT-OSS.
        assert body["messages"][1] == {"role": "user", "content": "My name is Sam."}
        assert body["messages"][2]["role"] == "assistant"
        assert body["messages"][-1] == {"role": "user", "content": "What is my name?"}
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Your name is Sam."},
                    }
                ],
            },
        )

    generator = generator_with_transport(handler)
    try:
        assert (
            generator.generate(
                "What is my name?",
                history=[
                    {"role": "user", "content": "My name is Sam."},
                    {"role": "assistant", "content": "Hi Sam."},
                ],
            )
            == "Your name is Sam."
        )
    finally:
        generator._groq.close()


def test_groq_stream_delivers_content_and_ignores_reasoning():
    def handler(request):
        chunks = []
        for delta in [
            {"reasoning": "private reasoning"},
            {"content": "First "},
            {"content": "second"},
        ]:
            payload = {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "openai/gpt-oss-20b",
                "choices": [{"index": 0, "delta": delta}],
            }
            chunks.append("data: " + json.dumps(payload) + "\n\n")
        return httpx.Response(
            200,
            text="".join(chunks) + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    generator = generator_with_transport(handler)
    try:
        assert list(generator.stream("Question")) == ["First ", "second"]
    finally:
        generator._groq.close()


@pytest.mark.parametrize(
    "status, expected",
    [
        (402, "requires inference credits"),
        (429, "usage limit"),
        (401, "authenticate"),
        (404, "unavailable"),
    ],
)
def test_provider_errors_are_actionable_without_fallback_or_raw_details(status, expected):
    generator = generator_with_transport(
        lambda request: httpx.Response(
            status, json={"error": {"message": "sensitive upstream detail", "type": "test"}}
        )
    )
    generator._get_llm = Mock(side_effect=AssertionError("No fallback allowed"))
    try:
        with pytest.raises(GenerationError) as error:
            list(generator.stream("Question"))
        assert expected in str(error.value)
        assert "sensitive" not in str(error.value)
        generator._get_llm.assert_not_called()
    finally:
        generator._groq.close()


def test_missing_groq_configuration_does_not_call_huggingface():
    generator = LLMGenerator(RAGConfig(llm_provider="groq", groq_api_key=None))
    with pytest.raises(GenerationError, match="not configured"):
        generator.generate("Question")
