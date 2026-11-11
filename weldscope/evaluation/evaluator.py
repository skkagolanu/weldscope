"""RAGAS evaluation metrics for RAG system."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import ConfigurationError


@dataclass
class EvaluationResult:
    """RAGAS evaluation results."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float
    details: Dict[str, Any]


class RAGASEvaluator:
    """Evaluate RAG system using RAGAS metrics."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Evaluate RAG responses using RAGAS metrics."""
        if not questions or not answers or not contexts:
            raise ConfigurationError("Questions, answers, and contexts are required")

        if len(questions) != len(answers) != len(contexts):
            raise ConfigurationError("Questions, answers, and contexts must have same length")

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }

        if ground_truths:
            if len(ground_truths) != len(questions):
                raise ConfigurationError("Ground truths must match questions length")
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset,
            metrics=self.metrics,
        )

        overall = (
            sum(
                [
                    result["faithfulness"],
                    result["answer_relevancy"],
                    result["context_precision"],
                    result["context_recall"],
                ]
            )
            / 4
        )

        return EvaluationResult(
            faithfulness=result["faithfulness"],
            answer_relevancy=result["answer_relevancy"],
            context_precision=result["context_precision"],
            context_recall=result["context_recall"],
            overall_score=overall,
            details=result,
        )

    def evaluate_single(
        self, question: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate single RAG response."""
        return self.evaluate(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth] if ground_truth else None,
        )
