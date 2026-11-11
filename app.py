"""Streamlit UI for WeldScope with conversation memory and sensor analysis."""

import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import datetime

from weldscope.core.config import RAGConfig
from weldscope.conversation import completed_history, needs_sensor_context
from weldscope.core.exceptions import RetrievalError, GenerationError
from weldscope.ingestion.builder import VectorStoreBuilder
from weldscope.retrieval.retriever import HybridRetriever
from weldscope.retrieval.reranker import Reranker
from weldscope.pipeline.pipeline import RAGPipeline
from weldscope.observability.tracer import RAGTracer

st.set_page_config(
    page_title="WeldScope | Welding Research Assistant", page_icon="⚙", layout="wide"
)


@st.cache_resource
def load_config():
    """Load configuration."""
    return RAGConfig()


@st.cache_resource
def load_retriever(_config):
    """Load models and the index only when a research question needs them."""
    builder = VectorStoreBuilder(_config)
    vectorstore, documents = builder.load_vectorstore()
    return HybridRetriever(vectorstore, documents, _config)


@st.cache_resource
def load_reranker(_config):
    """Share model weights; Reranker serializes initialization and inference."""
    return Reranker(_config)


def release_system(resource):
    resource[0].generator.close()


@st.cache_resource(scope="session", on_release=release_system)
def initialize_system(_config):
    """Initialize conversation handling without loading the document index."""
    tracer = None
    if _config.langfuse_enabled:
        tracer = RAGTracer(_config)

    pipeline = RAGPipeline(
        None, _config, tracer=tracer, retriever_loader=lambda: load_retriever(_config)
    )
    pipeline.reranker = load_reranker(_config)
    return pipeline, tracer


@st.cache_data
def load_sensor_data():
    """Load sensor data from CSV."""
    sensor_file = Path(__file__).resolve().parent / "data/sensor_log.csv"
    if sensor_file.exists():
        return pd.read_csv(sensor_file)
    return None


def analyze_sensor_context(df, query):
    """Generate sensor data context for query."""
    if df is None:
        return ""

    defect_counts = df["defect_type"].value_counts()
    total_readings = len(df)
    defect_readings = len(df[df["defect_type"] != "none"])

    context = f"\n\nSensor Data Summary:\n"
    context += f"- Total readings: {total_readings}\n"
    context += f"- Defect occurrences: {defect_readings}\n"
    context += f"- Defect types: {', '.join(defect_counts.index.tolist())}\n"

    if "defect" in query.lower():
        recent_defects = df[df["defect_type"] != "none"].tail(5)
        if not recent_defects.empty:
            context += f"\nRecent defects detected:\n"
            for _, row in recent_defects.iterrows():
                context += f"- {row['defect_type']}: RPM={row['rpm']}, Force={row['force_kn']}kN, Temp={row['temperature_c']}°C\n"

    return context


def render_source(source, index):
    """Render single source citation."""
    with st.expander(f"Source {index}: {source.doc_name}"):
        if source.page_num is not None:
            st.caption(f"Page {source.page_num + 1}")
        if source.rerank_score is not None:
            st.caption(f"Reranker score: {source.rerank_score:.3f}")
        if source.fusion_score is not None:
            st.caption(f"Fusion rank score: {source.fusion_score:.3f}")
        st.text(source.chunk_text)


def main():
    """Main Streamlit application."""
    st.title("WeldScope")
    st.markdown(
        "Ask questions about friction stir welding defects, root causes, and corrective actions."
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sidebar
    with st.sidebar:
        st.header("Session Info")
        st.caption(f"Session ID: {st.session_state.session_id}")

        # Message counter
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.metric("Messages in conversation", total_messages)
        st.metric("Your questions", user_messages)

        remembered_turns = len(completed_history(st.session_state.messages)) // 2
        st.caption(f"Recent conversation context: {remembered_turns} completed exchanges")
        st.caption("Up to ten completed exchanges; long chats are shortened to fit.")

        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption(
            "Conversation context is specific to this session. Clear Conversation resets the chat."
        )

    # Load system
    try:
        config = load_config()
        pipeline, tracer = initialize_system(config)
    except Exception as e:
        st.error("WeldScope could not start. The app owner needs to check its configuration.")
        return

    # Display conversation history
    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message.get("route") == "error":
                st.error(message["content"])
                continue
            st.markdown(message["content"])
            if message.get("sources"):
                st.markdown("**Sources:**")
                for i, source in enumerate(message["sources"], 1):
                    render_source(source, i)
            if message.get("route") == "research" and message.get("timings"):
                t = message["timings"]
                st.caption(
                    f"Search {t['retrieval_ms']:.0f} ms · Rerank {t['reranking_ms']:.0f} ms · Total {t['total_ms']:.0f} ms"
                )
            if tracer and message.get("trace_id"):
                feedback = st.feedback("thumbs", key=f"feedback_{message_index}")
                if feedback is not None and feedback != message.get("feedback"):
                    tracer.score_feedback(message["trace_id"], float(feedback))
                    tracer.flush()
                    message["feedback"] = feedback

    # Chat input
    if prompt := st.chat_input("Ask about welding, sources, or WeldScope..."):
        history = completed_history(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                sensor_context = ""
                if needs_sensor_context(prompt):
                    sensor_context = analyze_sensor_context(load_sensor_data(), prompt)

                def display_chunk(chunk):
                    nonlocal full_response
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                with st.spinner("Thinking..."):
                    response = pipeline.query(
                        prompt,
                        stream=True,
                        on_chunk=display_chunk,
                        history=history,
                        sensor_context=sensor_context,
                    )

                message_placeholder.markdown(full_response)

                if response.sources:
                    st.markdown("**Sources:**")
                    for i, source in enumerate(response.sources, 1):
                        render_source(source, i)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                        "trace_id": response.metadata.get("trace_id"),
                        "route": response.metadata.get("route"),
                        "timings": response.metadata.get("timings"),
                        "retrieval_query": response.metadata.get("retrieval_query", ""),
                    }
                )
                st.rerun()

            except RetrievalError:
                error_message = (
                    "I couldn't search the document collection. Please try again shortly."
                )
            except GenerationError as error:
                error_message = str(error)
            except Exception:
                error_message = "I couldn't complete that reply. Please try again shortly."
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "route": "error"}
            )
            st.rerun()


if __name__ == "__main__":
    main()
