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

    async def test_checkpoint_and_error_budget_preserve_unassessed_cases(self):
        checkpoints = []
        samples = [
            dict(id=str(i), question="FSW?", response="Answer", contexts=["Evidence"])
            for i in range(3)
        ]
        report = await answer_scores(
            samples,
            JudgeStub(),
            EmbeddingStub(),
            on_progress=lambda rows: checkpoints.append(len(rows)),
            error_budget=1,
        )
        self.assertTrue(checkpoints)
        self.assertTrue(all(count == 1 for count in checkpoints))
        self.assertFalse(report["complete"])
        self.assertEqual(len(report["results"]), 3)
        self.assertEqual(
            report["results"][1]["skipped"]["faithfulness"], "assessment_aborted_after_errors"
        )

    def test_page_identifiers_are_validated(self):
        self.assertEqual(page_id("data/a.pdf", 2), "a.pdf#page=2")
        for value in [0, -1, True, "2"]:
            with self.assertRaises(ValueError):
                page_id("a.pdf", value)


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


class RatePacingContract(unittest.IsolatedAsyncioTestCase):
    async def test_quota_headers_delay_until_reset(self):
        import httpx
        from evaluation.pacing import RequestPacer

        now = [100.0]
        delays = []

        async def sleep(seconds):
            delays.append(seconds)
            now[0] += seconds

        pacer = RequestPacer(0)
        pacer.estimate = lambda request: 300
        with (
            patch("evaluation.pacing.monotonic", side_effect=lambda: now[0]),
            patch("evaluation.pacing.asyncio.sleep", side_effect=sleep),
        ):
            await pacer.response(
                httpx.Response(
                    200,
                    headers={
                        "x-ratelimit-remaining-tokens": "100",
                        "x-ratelimit-limit-tokens": "8000",
                        "x-ratelimit-reset-tokens": "1m2s",
                    },
                    json={"usage": {"total_tokens": 12}},
                )
            )
            await pacer.request(
                httpx.Request("POST", "https://example.invalid", content=b"fixture")
            )
        self.assertGreaterEqual(sum(delays), 62)
        self.assertEqual(pacer.observed_token_limit, 8000)

    async def test_retry_after_replays_a_429_then_succeeds(self):
        import httpx
        from evaluation.pacing import RequestPacer, QuotaTransport

        calls = []
        delays = []

        def handle(request):
            calls.append(request.content)
            return (
                httpx.Response(429, headers={"retry-after": ".5"}, json={})
                if len(calls) == 1
                else httpx.Response(200, json={"usage": {"total_tokens": 20}})
            )

        async def sleep(seconds):
            delays.append(seconds)

        pacer = RequestPacer(0)
        pacer.estimate = lambda request: 100
        transport = QuotaTransport(pacer, inner=httpx.MockTransport(handle))
        with patch("evaluation.pacing.asyncio.sleep", side_effect=sleep):
            async with httpx.AsyncClient(transport=transport) as client:
                result = await client.post("https://example.invalid", json={"messages": []})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(calls[0], calls[1])
        self.assertGreaterEqual(sum(delays), 0.5)
        self.assertEqual(pacer.status_counts, {"429": 1, "200": 1})
        self.assertEqual(pacer.usage["total_tokens"], 20)

    async def test_token_budget_defers_before_any_request(self):
        import httpx
        from evaluation.pacing import RequestPacer, QuotaDeferred

        pacer = RequestPacer(0, token_budget=10)
        pacer.estimate = lambda request: 20
        with self.assertRaises(QuotaDeferred):
            await pacer.request(httpx.Request("POST", "https://example.invalid"))
        self.assertEqual(pacer.requests, 0)

    async def test_completed_native_metric_is_reused_on_resume(self):
        judge = JudgeStub()
        with patch.object(judge, "agenerate", new_callable=AsyncMock) as generate:
            report = await answer_scores(
                [
                    dict(
                        id="one",
                        question="Cook",
                        response="Out of scope",
                        contexts=[],
                        answerable=False,
                    )
                ],
                judge,
                EmbeddingStub(),
                resume_rows=[dict(id="one", scores={"scope_adherence": 5}, errors={}, skipped={})],
            )
        self.assertTrue(report["complete"])
        generate.assert_not_called()

    async def test_resume_rejects_unbound_inputs(self):
        from evaluation.run_answers import assess
        from types import SimpleNamespace
        import os

        with tempfile.TemporaryDirectory() as directory:
            samples = Path(directory) / "samples.json"
            samples.write_text(json.dumps({"samples": [{"id": "one"}]}))
            resume = Path(directory) / "progress.json"
            resume.write_text(json.dumps({"assessment_fingerprint": "different"}))
            args = SimpleNamespace(
                samples=samples,
                limit=1,
                judge_model="judge",
                embedding_model="embedding",
                resume=resume,
            )
            with (
                patch.dict(os.environ, {"GROQ_API_KEY": "local-test-placeholder"}),
                self.assertRaisesRegex(ValueError, "fingerprint"),
            ):
                await assess(args)


if __name__ == "__main__":
    unittest.main()
