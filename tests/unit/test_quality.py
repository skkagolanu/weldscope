"""Metric definitions, source integrity, citation checks and telemetry regression cases."""

import json
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from langchain_core.documents import Document
from weldscope.conversation import plan_conversation
from weldscope.core.config import RAGConfig
from weldscope.generation.validation import check_answer
from weldscope.monitoring.telemetry import record_event
from weldscope.pipeline.pipeline import RAGPipeline
from weldscope.retrieval.retriever import HybridRetriever


def test_all_versioned_routing_cases():
    dataset = json.loads(
        (Path(__file__).parents[2] / "benchmarks/fsw-v1/questions.json").read_text()
    )
    assert len(dataset["cases"]) == 50
    for case in dataset["cases"]:
        assert (
            plan_conversation(case["question"], case["history"]).kind == case["expected_route"]
        ), case["id"]


@pytest.mark.parametrize(
    "answer,status",
    [
        ("FSW is solid-state [1].", "passed_structural_checks"),
        ("FSW is solid-state [0].", "withheld"),
        ("FSW is solid-state [999].", "withheld"),
        ("FSW is solid-state [1-999999].", "withheld"),
        ("FSW is solid-state [2-1].", "withheld"),
        ("FSW is solid-state [1-2].", "withheld"),
        ("FSW is solid-state [1, 1].", "passed_structural_checks"),
        ("FSW requires exactly 9372 RPM [1].", "needs_review"),
        ("FSW is solid-state.", "needs_review"),
        ("The sources do not contain that information.", "abstention"),
        (
            "The provided source evidence does not contain any information about changes in 2035.",
            "abstention",
        ),
    ],
)
def test_citation_and_numerical_support_diagnostics(answer, status):
    assert check_answer(answer, [Document(page_content="FSW is solid-state.")])["status"] == status


def test_citation_validation_withholds_final_invalid_reply():
    retriever = Mock()
    documents = [Document(page_content="FSW evidence")]
    retriever.retrieve.return_value = documents
    pipeline = RAGPipeline(retriever, RAGConfig(telemetry_path=""))
    pipeline.reranker.rerank = Mock(return_value=documents)
    pipeline.generator.generate = Mock(return_value="An unsupported claim [99].")
    result = pipeline.query("What is FSW?")
    assert "has been withheld" in result.answer
    assert result.metadata["answer_checks"]["invalid_citations"] == [99]


def test_allowlisted_sources_cannot_be_overridden_by_metadata_filter():
    documents = [Document(page_content="FSW evidence", metadata={"source": "current.pdf"})]
    store = Mock()
    store.similarity_search.return_value = []
    retriever = HybridRetriever(store, documents, RAGConfig(), allowed_sources=["current.pdf"])
    retriever.retrieve("FSW", filters={"source": "stale.pdf"})
    assert store.similarity_search.call_args.kwargs["filter"] == {
        "$and": [{"source": {"$in": ["current.pdf"]}}, {"source": "stale.pdf"}]
    }
    retriever.retrieve_vector_only("FSW")
    assert store.similarity_search.call_args.kwargs["filter"] == {
        "source": {"$in": ["current.pdf"]}
    }


def test_logs_exclude_content_and_unknown_usage_fields(tmp_path):
    destination = tmp_path / "events.jsonl"
    record_event(
        str(destination),
        {
            "request_id": "correlation",
            "question": "PRIVATE QUESTION",
            "answer": "PRIVATE ANSWER",
            "exception": "PRIVATE KEY",
            "status": "success",
            "token_usage": {"total_tokens": 10, "unknown": "PRIVATE", "prompt_tokens": "PRIVATE"},
        },
    )
    text = destination.read_text()
    assert "PRIVATE" not in text
    assert json.loads(text)["token_usage"] == {"total_tokens": 10}
    assert json.loads(text)["request_id"] == "correlation"


def test_telemetry_failure_does_not_break_answer():
    with patch("weldscope.monitoring.telemetry._sink", side_effect=OSError("disk failure")):
        record_event("somewhere", {"request_id": "correlation"})


def test_current_corpus_excludes_missing_files_and_out_of_range_pages(tmp_path):
    from pypdf import PdfWriter
    from weldscope.ingestion.builder import VectorStoreBuilder
    from weldscope.core.exceptions import ConfigurationError

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(str(corpus / "current.pdf"))
    store = Mock()
    store.get.return_value = {
        "ids": ["valid", "missing", "invalid-page"],
        "documents": ["current evidence", "stale evidence", "impossible page"],
        "metadatas": [
            {"source": "data/current.pdf", "page": 0},
            {"source": "data/missing.pdf", "page": 0},
            {"source": "data/current.pdf", "page": 5},
        ],
    }
    builder = VectorStoreBuilder(RAGConfig())
    with (
        patch.object(builder, "_get_embeddings"),
        patch("weldscope.ingestion.builder.Chroma", return_value=store),
    ):
        _, documents = builder.load_vectorstore(str(tmp_path), corpus_directory=str(corpus))
        assert [document.id for document in documents] == ["valid"]
        with pytest.raises(ConfigurationError, match="No index passages"):
            builder.load_vectorstore(str(tmp_path), corpus_directory=str(tmp_path / "absent"))
