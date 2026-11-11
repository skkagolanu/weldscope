"""Exercise the actual Streamlit entrypoint without external inference calls."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest
import streamlit as st

from weldscope.core.models import RAGResponse, Source
from weldscope.core.config import RAGConfig


def test_chat_renders_streamed_answer_sources_and_clears_history():
    st.cache_resource.clear()
    pipeline = Mock()

    def query(question, stream, on_chunk, history, sensor_context):
        assert stream
        assert history == []
        assert sensor_context == ""
        for chunk in ["Low heat ", "can cause wormholes."]:
            on_chunk(chunk)
        return RAGResponse(
            question,
            "Low heat can cause wormholes.",
            [Source("fsw_doc1.pdf", 0, "Evidence from page one", 0.9, {})],
        )

    pipeline.query.side_effect = query
    with (
        patch(
            "weldscope.core.config.RAGConfig", return_value=SimpleNamespace(langfuse_enabled=False)
        ),
        patch(
            "weldscope.ingestion.builder.VectorStoreBuilder.load_vectorstore",
            return_value=(Mock(), []),
        ),
        patch("weldscope.retrieval.retriever.HybridRetriever"),
        patch("weldscope.pipeline.pipeline.RAGPipeline", return_value=pipeline),
    ):
        app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py")).run()
        assert not app.exception
        assert app.title[0].value == "WeldScope"
        app.chat_input[0].set_value("What causes wormhole defects?").run()
        assert not app.exception
        assert not app.error
        assert len(app.chat_message) == 2
        assert any(m.value == "Low heat can cause wormholes." for m in app.markdown)
        assert any(c.value == "Page 1" for c in app.caption)
        assert "Previous conversation" not in pipeline.query.call_args.args[0]
        app.button[0].click().run()
        assert not app.exception
        assert len(app.chat_message) == 0
    st.cache_resource.clear()


def test_greeting_and_about_work_without_credentials_or_document_initialization():
    st.cache_resource.clear()
    config = RAGConfig(_env_file=None, groq_api_key=None)
    with (
        patch("weldscope.core.config.RAGConfig", return_value=config),
        patch("weldscope.ingestion.builder.VectorStoreBuilder.load_vectorstore") as load,
        patch("weldscope.generation.generator.LLMGenerator.stream") as generate,
    ):
        app = AppTest.from_file(str(Path(__file__).parents[2] / "app.py")).run()
        app.chat_input[0].set_value("hi").run()
        assert not app.exception
        assert not app.error
        assert "Hi! I'm WeldScope" in app.chat_message[-1].markdown[0].value
        app.chat_input[0].set_value("What is the RAG based on?").run()
        assert not app.exception
        assert not app.error
        assert "BM25" in app.chat_message[-1].markdown[0].value
        assert not app.expander  # No fictitious source citations for a local reply.
        load.assert_not_called()
        generate.assert_not_called()
    st.cache_resource.clear()
