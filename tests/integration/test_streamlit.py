"""Exercise the actual Streamlit entrypoint without external inference calls."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from weldscope.core.models import RAGResponse, Source


def test_chat_renders_streamed_answer_sources_and_clears_history():
    pipeline = Mock()

    def query(question, stream, on_chunk):
        assert stream
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
