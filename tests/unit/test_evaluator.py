"""Validation and aggregation of explicit model-judged evaluations."""

import json
from unittest.mock import Mock
import pytest
from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import ConfigurationError, GenerationError
from weldscope.evaluation.evaluator import GroundedAnswerEvaluator


@pytest.fixture
def evaluator():
    evaluator = GroundedAnswerEvaluator(RAGConfig())
    evaluator.generator = Mock()
    evaluator.generator.generate.return_value = json.dumps(
        dict(faithfulness=0.9, answer_relevancy=0.8, context_precision=0.7, context_recall=0.6)
    )
    return evaluator


@pytest.mark.parametrize(
    "questions,answers,contexts",
    [([], [], []), (["Q"], ["A", "B"], [["C"]]), (["Q"], ["A"], [["C"], ["D"]])],
)
def test_invalid_inputs(evaluator, questions, answers, contexts):
    with pytest.raises(ConfigurationError):
        evaluator.evaluate(questions, answers, contexts)
    evaluator.generator.generate.assert_not_called()


def test_reference_lengths_must_match(evaluator):
    with pytest.raises(ConfigurationError, match="must match"):
        evaluator.evaluate(["Q"], ["A"], [["C"]], ["R", "S"])


def test_no_recall_is_invented_without_reference(evaluator):
    result = evaluator.evaluate_single("Q", "A", ["C"])
    assert result.context_recall is None
    assert result.overall_score == pytest.approx(0.8)
    evaluator.generator.generate.assert_called_once()


def test_reference_allows_recall(evaluator):
    result = evaluator.evaluate_single("Q", "A", ["C"], "Reference")
    assert result.context_recall == 0.6
    assert result.overall_score == pytest.approx(0.75)


@pytest.mark.parametrize("output", ["not JSON", '{"faithfulness": 5}', '{"faithfulness":NaN}'])
def test_invalid_model_scores_are_rejected(evaluator, output):
    evaluator.generator.generate.return_value = output
    with pytest.raises(GenerationError, match="invalid scores"):
        evaluator.evaluate_single("Q", "A", ["C"])
