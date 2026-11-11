"""Test vector store builder."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from weldscope.ingestion.builder import VectorStoreBuilder
from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import ConfigurationError


@pytest.fixture
def config():
    """Create test config."""
    return RAGConfig(groq_api_key="test_token")


@pytest.fixture
def builder(config):
    """Create builder instance."""
    return VectorStoreBuilder(config)


def test_builder_initialization(builder, config):
    """Builder initializes with config."""
    assert builder.config is config
    assert builder._embeddings is None


@patch("weldscope.ingestion.builder.LocalEmbeddings")
def test_get_embeddings_initializes_model(mock_embeddings, builder):
    """Get embeddings initializes HuggingFace model."""
    mock_embeddings.return_value = Mock()

    embeddings = builder._get_embeddings()

    assert embeddings is not None
    mock_embeddings.assert_called_once_with(model_name=builder.config.embedding_model)


@patch("weldscope.ingestion.builder.LocalEmbeddings")
def test_get_embeddings_caches_instance(mock_embeddings, builder):
    """Get embeddings caches model instance."""
    mock_embeddings.return_value = Mock()

    emb1 = builder._get_embeddings()
    emb2 = builder._get_embeddings()

    assert emb1 is emb2
    mock_embeddings.assert_called_once()


def test_load_pdfs_raises_on_missing_directory(builder, tmp_path):
    """Load PDFs raises ConfigurationError for missing directory."""
    nonexistent = tmp_path / "nonexistent"

    with pytest.raises(ConfigurationError, match="PDF directory not found"):
        builder.load_pdfs(str(nonexistent))


def test_load_pdfs_raises_on_no_pdfs(builder, tmp_path):
    """Load PDFs raises ConfigurationError when no PDFs found."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ConfigurationError, match="No PDF files found"):
        builder.load_pdfs(str(empty_dir))


@patch("weldscope.ingestion.builder.PdfReader")
def test_load_pdfs_loads_all_pdfs(mock_loader, builder, tmp_path):
    """Load PDFs loads all PDF files from directory."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "doc1.pdf").touch()
    (pdf_dir / "doc2.pdf").touch()

    mock_loader_instance = Mock()
    mock_loader_instance.pages = [Mock(extract_text=Mock(return_value="Content"))]
    mock_loader.return_value = mock_loader_instance

    documents = builder.load_pdfs(str(pdf_dir))

    assert len(documents) == 2
    assert mock_loader.call_count == 2


@patch("weldscope.ingestion.builder.RecursiveCharacterTextSplitter")
def test_split_documents_uses_config_params(mock_splitter, builder):
    """Split documents uses chunk size and overlap from config."""
    mock_splitter_instance = Mock()
    mock_splitter_instance.split_documents.return_value = []
    mock_splitter.return_value = mock_splitter_instance

    docs = [Document(page_content="Test content")]
    builder.split_documents(docs)

    mock_splitter.assert_called_once_with(
        chunk_size=builder.config.chunk_size,
        chunk_overlap=builder.config.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


@patch("weldscope.ingestion.builder.Chroma")
@patch("weldscope.ingestion.builder.LocalEmbeddings")
def test_build_vectorstore_creates_chroma(mock_embeddings, mock_chroma, builder):
    """Build vectorstore creates Chroma instance."""
    mock_embeddings.return_value = Mock()
    mock_chroma.from_documents.return_value = Mock()

    docs = [Document(page_content="Test content")]
    vectorstore = builder.build_vectorstore(docs)

    assert vectorstore is not None
    mock_chroma.from_documents.assert_called_once()


@patch("weldscope.ingestion.builder.Chroma")
@patch("weldscope.ingestion.builder.LocalEmbeddings")
def test_build_vectorstore_uses_persist_directory(mock_embeddings, mock_chroma, builder):
    """Build vectorstore uses specified persist directory."""
    mock_embeddings.return_value = Mock()
    mock_chroma.from_documents.return_value = Mock()

    docs = [Document(page_content="Test content")]
    builder.build_vectorstore(docs, persist_directory="/custom/path")

    call_args = mock_chroma.from_documents.call_args
    assert call_args[1]["persist_directory"] == "/custom/path"


@patch("weldscope.ingestion.builder.Chroma")
@patch("weldscope.ingestion.builder.LocalEmbeddings")
def test_load_vectorstore_raises_on_missing_directory(
    mock_embeddings, mock_chroma, builder, tmp_path
):
    """Load vectorstore raises ConfigurationError for missing directory."""
    mock_embeddings.return_value = Mock()
    nonexistent = tmp_path / "nonexistent"

    with pytest.raises(ConfigurationError, match="Vector store not found"):
        builder.load_vectorstore(str(nonexistent))


@patch("weldscope.ingestion.builder.Chroma")
@patch("weldscope.ingestion.builder.LocalEmbeddings")
def test_load_vectorstore_loads_existing(mock_embeddings, mock_chroma, builder, tmp_path):
    """Load vectorstore loads existing Chroma instance."""
    mock_embeddings.return_value = Mock()

    mock_vectorstore = Mock()
    mock_vectorstore.get.return_value = {
        "ids": ["id1", "id2"],
        "documents": ["Content 1", "Content 2"],
        "metadatas": [{"source": "doc1.pdf"}, {"source": "doc2.pdf"}],
    }
    mock_chroma.return_value = mock_vectorstore

    persist_dir = tmp_path / "vectorstore"
    persist_dir.mkdir()

    vectorstore, documents = builder.load_vectorstore(str(persist_dir))

    assert vectorstore is not None
    assert len(documents) == 2
    assert documents[0].page_content == "Content 1"
    assert documents[0].metadata["source"] == "doc1.pdf"
    mock_chroma.assert_called_once()


@patch("weldscope.ingestion.builder.VectorStoreBuilder.build_vectorstore")
@patch("weldscope.ingestion.builder.VectorStoreBuilder.split_documents")
@patch("weldscope.ingestion.builder.VectorStoreBuilder.load_pdfs")
def test_build_from_pdfs_executes_full_pipeline(
    mock_load, mock_split, mock_build, builder, tmp_path
):
    """Build from PDFs executes complete pipeline."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    mock_docs = [Document(page_content="Content")]
    mock_chunks = [Document(page_content="Chunk")]
    mock_vectorstore = Mock()

    mock_load.return_value = mock_docs
    mock_split.return_value = mock_chunks
    mock_build.return_value = mock_vectorstore

    result = builder.build_from_pdfs(str(pdf_dir))

    assert result is mock_vectorstore
    mock_load.assert_called_once_with(str(pdf_dir))
    mock_split.assert_called_once_with(mock_docs)
    mock_build.assert_called_once_with(mock_chunks, None)


def test_native_pdf_loading_and_stable_chunk_ids(builder, tmp_path):
    from pypdf import PdfWriter

    pdf = tmp_path / "empty.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with pdf.open("wb") as output:
        writer.write(output)
    # Empty PDF pages do not become searchable, fabricated text.
    assert builder.load_pdfs(str(tmp_path)) == []
    docs = [Document(page_content="Evidence", metadata={"source": "paper.pdf", "page": 0})]
    with (
        patch("weldscope.ingestion.builder.LocalEmbeddings"),
        patch("weldscope.ingestion.builder.Chroma") as chroma,
    ):
        builder.build_vectorstore(docs)
        first = chroma.from_documents.call_args.kwargs["ids"]
        builder.build_vectorstore(docs)
        assert chroma.from_documents.call_args.kwargs["ids"] == first
