"""Fast scope refusal, strict corpus routing, and bounded frontend redraws."""

from unittest.mock import Mock
import pytest
from weldscope.conversation import plan_conversation
from weldscope.core.config import RAGConfig
from weldscope.pipeline.pipeline import RAGPipeline
from weldscope.streaming import StreamDisplay


@pytest.mark.parametrize(
    "question",
    [
        "I am on a ship that was friction stir welded and the joints are cracking. We might sink soon.",
        "I'm aboard a vessel taking on water. What repairs should I do?",
        "Is my friction stir welded ship safe to sail?",
        "Who is the president?",
        "My name is Sam.",
    ],
)
def test_out_of_scope_requests_do_not_search_or_call_provider(question):
    loader = Mock(side_effect=AssertionError("Must not load models"))
    pipeline = RAGPipeline(None, RAGConfig(), retriever_loader=loader)
    pipeline.generator = Mock()
    answer = pipeline.query(question, stream=True)
    assert answer.metadata["route"] == "out_of_scope"
    assert not answer.sources
    assert "Mayday" not in answer.answer and "Channel 16" not in answer.answer
    loader.assert_not_called()
    pipeline.generator.stream.assert_not_called()
    pipeline.generator.generate.assert_not_called()
    assert list(pipeline.stream_query(question)) == [answer.answer]
    assert answer.metadata["timings"]["generation_ms"] == 0


@pytest.mark.parametrize(
    "question",
    [
        "What does the corpus say about cracks in friction stir welded ship panels?",
        "Explain FSW applications in ship construction.",
    ],
)
def test_general_ship_research_is_not_blocked(question):
    assert plan_conversation(question).kind == "research"


def test_stream_display_keeps_first_chunk_immediate_and_final_text_exact():
    paint = Mock()
    clock = Mock(return_value=0)
    display = StreamDisplay(paint, clock=clock)
    for part in ["a"] * 1000:
        display(part)
    assert paint.call_count == 1
    clock.return_value = 0.1
    display("b")
    assert paint.call_count == 2
    assert display.finish() == "a" * 1000 + "b"
    assert paint.call_args.args[0] == "a" * 1000 + "b"
    assert paint.call_count == 3


@pytest.mark.parametrize(
    "question",
    [
        "The ship is sinking soon, send me a recipe to cook my favourite meal.",
        "Given that FSW explanation, can you give me a pasta recipe?",
        "How can I cook dinner?",
        "i am in a ship that has fsw for welds. unfortunately i think there were lots of faults and now the joints that had defects are coming off. i would be drowning soon but i want to know how to cook butter chicken one last time before sinkign",
        "With FSW in mind, how to bake bread?",
        "After that FSW explanation, please cook me dinner.",
    ],
)
def test_recipe_request_does_not_inherit_fsw_retrieval_topic(question):
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hi!", "route": "local"},
        {"role": "user", "content": "What was this chatbot built for?"},
        {"role": "assistant", "content": "FSW corpus research", "route": "local"},
        {"role": "user", "content": "What is FSW?"},
        {
            "role": "assistant",
            "content": "Friction stir welding [1]",
            "route": "research",
            "retrieval_query": "What is FSW?",
        },
    ]
    loader = Mock(side_effect=AssertionError("No retrieval for a recipe"))
    pipeline = RAGPipeline(None, RAGConfig(), retriever_loader=loader)
    pipeline.generator = Mock()
    response = pipeline.query(question, history=history, stream=True)
    assert response.metadata["route"] == "out_of_scope"
    assert "Recipes" in response.answer
    assert not response.sources
    loader.assert_not_called()
    pipeline.generator.stream.assert_not_called()


def test_food_industry_welding_question_remains_research():
    assert (
        plan_conversation(
            "What does the FSW corpus say about welding food-processing equipment?"
        ).kind
        == "research"
    )
