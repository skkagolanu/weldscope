"""Streamlit UI for WeldScope with conversation memory and sensor analysis."""

import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import datetime

from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import RetrievalError, GenerationError
from weldscope.ingestion.builder import VectorStoreBuilder
from weldscope.retrieval.retriever import HybridRetriever
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
def initialize_system(_config):
    """Initialize RAG system components."""
    builder = VectorStoreBuilder(_config)
    vectorstore, documents = builder.load_vectorstore()
    retriever = HybridRetriever(vectorstore, documents, _config)

    tracer = None
    if _config.langfuse_enabled:
        tracer = RAGTracer(_config)

    pipeline = RAGPipeline(retriever, _config, tracer=tracer)
    return pipeline, tracer


@st.cache_data
def load_sensor_data():
    """Load sensor data from CSV."""
    sensor_file = Path("data/sensor_log.csv")
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


def get_conversation_context(messages, max_turns=10):
    """Get last N conversation turns for context."""
    if len(messages) <= 1:
        return ""

    # The current prompt is already supplied separately.
    history = messages[:-1] if messages[-1]["role"] == "user" else messages
    recent = history[-(max_turns * 2) :]
    context = "\n\nPrevious conversation:\n"
    for msg in recent:
        role = msg["role"].capitalize()
        content = msg["content"][:200]
        context += f"{role}: {content}...\n"

    return context


def render_source(source, index):
    """Render single source citation."""
    with st.expander(f"Source {index}: {source.doc_name}"):
        if source.page_num is not None:
            st.caption(f"Page {source.page_num + 1}")
        st.caption(f"Relevance Score: {source.relevance_score:.3f}")
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

        # Memory retention indicator
        max_memory_turns = 10
        if user_messages > max_memory_turns:
            st.warning(f"⚠️ Remembering last {max_memory_turns} turns only")
            st.caption(f"Older messages: {user_messages - max_memory_turns}")
        else:
            remaining = max_memory_turns - user_messages
            st.success(f"✓ Full memory active")
            st.caption(f"Remaining turns: {remaining}")

        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.caption(
            "Note: Conversation history is temporary and deleted when you close the browser."
        )

    # Load system
    try:
        config = load_config()
        pipeline, tracer = initialize_system(config)
        sensor_df = load_sensor_data()
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        st.info("Make sure vector store exists. Run document ingestion first.")
        return

    # Display conversation history
    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                st.markdown("**Sources:**")
                for i, source in enumerate(message["sources"], 1):
                    render_source(source, i)
            if tracer and message.get("trace_id"):
                feedback = st.feedback("thumbs", key=f"feedback_{message_index}")
                if feedback is not None and feedback != message.get("feedback"):
                    tracer.score_feedback(message["trace_id"], float(feedback))
                    tracer.flush()
                    message["feedback"] = feedback

    # Chat input
    if prompt := st.chat_input("Ask about FSW defects..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Add conversation context
                conv_context = get_conversation_context(st.session_state.messages)

                # Add sensor context if relevant
                sensor_context = analyze_sensor_context(sensor_df, prompt)

                # Augment query with context
                augmented_query = prompt
                if conv_context:
                    augmented_query = f"{prompt}{conv_context}"
                if sensor_context:
                    augmented_query = f"{augmented_query}{sensor_context}"

                def display_chunk(chunk):
                    nonlocal full_response
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                with st.spinner("Retrieving evidence and writing an answer..."):
                    response = pipeline.query(augmented_query, stream=True, on_chunk=display_chunk)

                message_placeholder.markdown(full_response)

                st.markdown("**Sources:**")
                for i, source in enumerate(response.sources, 1):
                    render_source(source, i)

                st.caption(
                    f"Retrieved: {response.metadata.get('retrieved_count', 0)} | "
                    f"Reranked: {response.metadata.get('reranked_count', 0)}"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "sources": response.sources,
                        "trace_id": response.metadata.get("trace_id"),
                    }
                )
                st.rerun()

            except RetrievalError as e:
                st.error(f"Retrieval failed: {e}")
            except GenerationError as e:
                st.error(f"Generation failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
