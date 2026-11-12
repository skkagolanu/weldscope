"""Exercise actual RAGAS metric classes; mocked judge tests are contracts only."""

import asyncio
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from ragas import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbedding
from evaluation.quality import answer_scores, page_id, retrieval_scores

ROOT = Path(__file__).resolve().parents[2]


class JudgeStub(InstructorBaseRagasLLM):
    def generate(self, prompt, response_model):
        raise AssertionError("Async judge expected")

    async def agenerate(self, prompt, response_model):
        if response_model.__name__ == "RubricScoreOutput":
            return response_model(score=5, feedback="Contract fixture only")
        raise RuntimeError("Deliberate failed metric fixture")


class EmbeddingStub(BaseRagasEmbedding):
    def embed_text(self, text, **kwargs):
        return [1.0, 0.0]

    async def aembed_text(self, text, **kwargs):
        return self.embed_text(text)


class NativeRagasTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_precision_recall_deduplicate_page_ids(self):
        sample = SingleTurnSample(
            retrieved_context_ids=["a", "a", "b"], reference_context_ids=["a", "c"]
        )
        self.assertEqual(await IDBasedContextPrecision().single_turn_ascore(sample), 0.5)
        self.assertEqual(await IDBasedContextRecall().single_turn_ascore(sample), 0.5)

    async def test_empty_native_precision_is_not_invented_zero(self):
        sample = SingleTurnSample(retrieved_context_ids=[], reference_context_ids=["a"])
        self.assertTrue(math.isnan(await IDBasedContextPrecision().single_turn_ascore(sample)))

    async def test_provisional_baseline_is_scored_by_ragas(self):
        report = await retrieval_scores(
            ROOT / "benchmarks/fsw-v1/verified/report.json",
            ROOT / "benchmarks/fsw-v1/questions.json",
        )
        self.assertEqual(report["framework"], "ragas")
        self.assertEqual(len(report["results"]), 160)
        self.assertEqual(report["summary"]["reranked"]["id_context_recall"], 0.575)

    async def test_fingerprint_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rankings.json"
            data = json.loads((ROOT / "benchmarks/fsw-v1/verified/report.json").read_text())
            data["provenance"]["dataset_sha256"] = "changed"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "fingerprints"):
                await retrieval_scores(path, ROOT / "benchmarks/fsw-v1/questions.json")

    async def test_incomplete_rankings_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rankings.json"
            data = json.loads((ROOT / "benchmarks/fsw-v1/verified/report.json").read_text())
            data["results"].pop()
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "coverage"):
                await retrieval_scores(path, ROOT / "benchmarks/fsw-v1/questions.json")

    async def test_failed_judge_metrics_are_explicit_not_average_zero(self):
        report = await answer_scores(
            [
                dict(
                    id="fixture",
                    question="What is FSW?",
                    response="A process [1]",
                    contexts=["Evidence"],
                )
            ],
            JudgeStub(),
            EmbeddingStub(),
        )
        self.assertFalse(report["complete"])
        self.assertEqual(report["summary"]["faithfulness"]["failed"], 1)
        self.assertIsNone(report["summary"]["faithfulness"]["mean"])
        self.assertEqual(report["summary"]["scope_adherence"]["mean"], 5)
        self.assertEqual(report["summary"]["context_recall"]["skipped"], 1)

    async def test_negative_case_only_uses_native_scope_rubric(self):
        report = await answer_scores(
            [
                dict(
                    id="fixture",
                    question="Cook dinner",
                    response="Outside the corpus",
                    contexts=[],
                    answerable=False,
                )
            ],
            JudgeStub(),
            EmbeddingStub(),
        )
        self.assertTrue(report["complete"])
        self.assertEqual(report["summary"]["faithfulness"]["skipped"], 1)
        self.assertEqual(report["summary"]["scope_adherence"]["mean"], 5)

    def test_page_identifiers_are_validated(self):
        self.assertEqual(page_id("data/a.pdf", 2), "a.pdf#page=2")
        for value in [0, -1, True, "2"]:
            with self.assertRaises(ValueError):
                page_id("a.pdf", value)


if __name__ == "__main__":
    unittest.main()


class GroqTransportContract(unittest.IsolatedAsyncioTestCase):
    async def test_native_wrapper_uses_one_token_limit_and_structured_json(self):
        import httpx
        from groq import AsyncGroq
        from pydantic import BaseModel
        from evaluation.run_answers import make_judge

        class Reply(BaseModel):
            answer: str

        def handle(request):
            body = json.loads(request.content)
            self.assertIn("max_completion_tokens", body)
            self.assertNotIn("max_tokens", body)
            return httpx.Response(
                200,
                json={
                    "id": "contract-test",
                    "created": 0,
                    "model": "openai/gpt-oss-120b",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": '{"answer":"fixture"}'},
                        }
                    ],
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport:
            async with AsyncGroq(api_key="local-test-placeholder", http_client=transport) as client:
                llm = make_judge(client, "openai/gpt-oss-120b")
                reply = await llm.agenerate("Return the contract fixture", Reply)
                self.assertEqual(reply.answer, "fixture")
