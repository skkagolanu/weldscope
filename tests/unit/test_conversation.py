"""Regression cases for conversational intent and topic isolation."""

import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from weldscope.conversation import completed_history, plan_conversation, needs_sensor_context
from weldscope.core.config import RAGConfig
from weldscope.pipeline.pipeline import RAGPipeline


@pytest.mark.parametrize(
    "question",
    [
        "hi",
        "Hi!",
        "hello there",
        "hello, how are you?",
        "What's this app?",
        "Tell me about yourself",
        "how are you?",
        "thanks",
        "goodbye",
        "What is the RAG based on?",
        "What documents do you use?",
        "What can you do?",
        "What was this chatbot built for?",
        "How do you work?",
        "What is WeldScope?",
    ],
)
def test_social_and_product_questions_never_retrieve_or_generate(question):
    loader = Mock(side_effect=AssertionError("Must not load the index"))
    pipeline = RAGPipeline(None, RAGConfig(), retriever_loader=loader)
    pipeline.generator = Mock()
    response = pipeline.query(question)
    assert response.answer
    assert response.sources == []
    loader.assert_not_called()
    pipeline.generator.generate.assert_not_called()


@pytest.mark.parametrize(
    "question", ["High RPM causes which FSW defects?", "Hi, what causes wormholes?"]
)
def test_greeting_substrings_do_not_hide_welding_questions(question):
    assert plan_conversation(question).kind == "research"


def research_history():
    return [
        {"role": "user", "content": "What causes wormhole defects?"},
        {
            "role": "assistant",
            "content": "Prior answer that must not become search text.",
            "route": "research",
            "retrieval_query": "What causes wormhole defects?",
        },
    ]


def test_followup_retains_topic_without_searching_chat_transcripts():
    retriever = Mock()
    retriever.retrieve.return_value = [Document(page_content="Welding evidence")]
    pipeline = RAGPipeline(retriever, RAGConfig())
    pipeline.reranker = Mock()
    pipeline.reranker.rerank.return_value = retriever.retrieve.return_value
    pipeline.generator = Mock()
    pipeline.generator.generate.return_value = "Answer"
    response = pipeline.query(
        "What about temperature?", history=research_history(), sensor_context="Sensor summary"
    )
    searched = retriever.retrieve.call_args.args[0]
    assert "wormhole" in searched
    assert "temperature" in searched
    assert "Prior answer" not in searched
    assert "Sensor summary" not in searched
    assert pipeline.generator.generate.call_args.kwargs["history"] == research_history()
    assert response.metadata["route"] == "research"


def test_self_contained_question_does_not_inherit_previous_topic():
    question = "What causes surface cracking in FSW?"
    plan = plan_conversation(question, research_history())
    assert plan.retrieval_query == question


def test_unrelated_chat_resets_active_research_topic():
    history = research_history() + [
        {"role": "user", "content": "My name is Sam."},
        {"role": "assistant", "content": "Hi Sam", "route": "chat"},
    ]
    assert plan_conversation("Why?", history).kind == "local"


def test_unrelated_chat_uses_no_document_search():
    retriever = Mock()
    pipeline = RAGPipeline(retriever, RAGConfig())
    pipeline.generator = Mock()
    pipeline.generator.generate.return_value = "A conversational reply"
    response = pipeline.query("My name is Sam.")
    assert response.metadata["route"] == "out_of_scope"
    assert response.sources == []
    retriever.retrieve.assert_not_called()


def test_history_excludes_failed_turns_and_respects_limits():
    failed = [
        {"role": "user", "content": "Failed query"},
        {"role": "assistant", "content": "402 error", "route": "error"},
    ]
    messages = research_history() * 15 + failed + [{"role": "user", "content": "Current prompt"}]
    history = completed_history(messages)
    assert len(history) == 20
    assert not any("402" in m["content"] or "Current prompt" in m["content"] for m in history)
    short = completed_history(messages, character_budget=180)
    assert sum(len(m["content"]) for m in short) <= 180
    assert len(short) % 2 == 0


def test_sensor_data_is_opt_in():
    assert not needs_sensor_context("What causes wormhole defects?")
    assert needs_sensor_context("What do the sensor readings show?")
