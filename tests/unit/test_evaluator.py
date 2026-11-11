"""Test RAGAS evaluator."""

import pytest
from unittest.mock import Mock, patch
from weldscope.evaluation.evaluator import RAGASEvaluator, EvaluationResult
from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import ConfigurationError


@pytest.fixture
def config():
    """Create test config."""
    return RAGConfig(hf_token="test_token")


@pytest.fixture
def evaluator(config):
    """Create evaluator instance."""
    return RAGASEvaluator(config)


def test_evaluator_initialization(evaluator):
    """Evaluator initializes with config and metrics."""
    assert evaluator.config is not None
    assert len(evaluator.metrics) == 4


def test_evaluate_raises_on_empty_inputs(evaluator):
    """Evaluate raises ConfigurationError on empty inputs."""
    with pytest.raises(ConfigurationError, match="required"):
        evaluator.evaluate([], [], [])


def test_evaluate_raises_on_mismatched_lengths(evaluator):
    """Evaluate raises ConfigurationError on mismatched lengths."""
    with pytest.raises(ConfigurationError, match="same length"):
        evaluator.evaluate(questions=["Q1", "Q2"], answers=["A1"], contexts=[["C1"], ["C2"]])


def test_evaluate_raises_on_mismatched_ground_truths(evaluator):
    """Evaluate raises ConfigurationError on mismatched ground truths."""
    with pytest.raises(ConfigurationError, match="must match"):
        evaluator.evaluate(
            questions=["Q1", "Q2"],
            answers=["A1", "A2"],
            contexts=[["C1"], ["C2"]],
            ground_truths=["GT1"],
        )


@patch("weldscope.evaluation.evaluator.evaluate")
def test_evaluate_returns_results(mock_evaluate, evaluator):
    """Evaluate returns EvaluationResult."""
    mock_evaluate.return_value = {
        "faithfulness": 0.9,
        "answer_relevancy": 0.85,
        "context_precision": 0.88,
        "context_recall": 0.92,
    }

    result = evaluator.evaluate(
        questions=["What causes defects?"],
        answers=["Insufficient heat input"],
        contexts=[["Context about defects"]],
    )

    assert isinstance(result, EvaluationResult)
    assert result.faithfulness == 0.9
    assert result.answer_relevancy == 0.85
    assert result.context_precision == 0.88
    assert result.context_recall == 0.92
    assert result.overall_score == pytest.approx(0.8875)


@patch("weldscope.evaluation.evaluator.evaluate")
def test_evaluate_with_ground_truths(mock_evaluate, evaluator):
    """Evaluate accepts ground truths."""
    mock_evaluate.return_value = {
        "faithfulness": 0.9,
        "answer_relevancy": 0.85,
        "context_precision": 0.88,
        "context_recall": 0.92,
    }

    result = evaluator.evaluate(
        questions=["Q1"], answers=["A1"], contexts=[["C1"]], ground_truths=["GT1"]
    )

    assert isinstance(result, EvaluationResult)
    mock_evaluate.assert_called_once()


@patch("weldscope.evaluation.evaluator.evaluate")
def test_evaluate_single(mock_evaluate, evaluator):
    """Evaluate single response."""
    mock_evaluate.return_value = {
        "faithfulness": 0.95,
        "answer_relevancy": 0.90,
        "context_precision": 0.85,
        "context_recall": 0.88,
    }

    result = evaluator.evaluate_single(
        question="What causes wormhole defects?",
        answer="Insufficient heat input during FSW",
        contexts=["Context about wormhole defects"],
        ground_truth="Wormholes are caused by insufficient heat",
    )

    assert isinstance(result, EvaluationResult)
    assert result.faithfulness == 0.95
