"""Vector store builder for document ingestion."""

from pathlib import Path
from hashlib import sha256
from typing import List, Optional
from langchain_core.documents import Document
from pypdf import PdfReader
from weldscope.core.local_models import LocalModel
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from weldscope.ingestion.store import NumpyVectorStore

from weldscope.core.config import RAGConfig
from weldscope.core.exceptions import ConfigurationError


class LocalEmbeddings(Embeddings):
    """Local ONNX embeddings; no model downloads or hosted embedding provider."""

    def __init__(self, model_name):
        self.model = LocalModel(model_name)

    def embed_documents(self, texts):
        return self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=False
        ).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class VectorStoreBuilder:
    """Build and manage vector store from documents."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self._embeddings: Optional[LocalEmbeddings] = None

    def _get_embeddings(self) -> LocalEmbeddings:
        """Initialize embedding model."""
        if self._embeddings is None:
            self._embeddings = LocalEmbeddings(model_name=self.config.embedding_model)
        return self._embeddings

    def load_pdfs(self, pdf_dir: str) -> List[Document]:
        """Load all PDFs from directory."""
        pdf_path = Path(pdf_dir)

        if not pdf_path.exists():
            raise ConfigurationError(f"PDF directory not found: {pdf_dir}")

        documents = []
        pdf_files = list(pdf_path.glob("*.pdf"))

        if not pdf_files:
            raise ConfigurationError(f"No PDF files found in: {pdf_dir}")

        for pdf_file in pdf_files:
            reader = PdfReader(str(pdf_file))
            for page_index, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(pdf_file),
                                "page": page_index,
                            },
                        )
                    )

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        return splitter.split_documents(documents)

    def build_vectorstore(
        self, documents: List[Document], persist_directory: Optional[str] = None
    ) -> NumpyVectorStore:
        """Build vector store from documents."""
        embeddings = self._get_embeddings()
        persist_dir = persist_directory or self.config.vectorstore_path

        ids = [
            sha256(
                (
                    Path(doc.metadata.get("source", "")).name
                    + "\0"
                    + str(doc.metadata.get("page", ""))
                    + "\0"
                    + doc.page_content
                ).encode()
            ).hexdigest()
            for doc in documents
        ]
        unique = dict(zip(ids, documents))
        vectorstore = NumpyVectorStore.from_documents(
            documents=list(unique.values()),
            ids=list(unique),
            embedding=embeddings,
            persist_directory=persist_dir,
        )

        return vectorstore

    def load_vectorstore(
        self, persist_directory: Optional[str] = None, *, corpus_directory: Optional[str] = None
    ) -> tuple[NumpyVectorStore, List[Document]]:
        """Load existing vector store and reconstruct documents."""
        embeddings = self._get_embeddings()
        persist_dir = persist_directory or self.config.vectorstore_path

        if not Path(persist_dir).exists():
            raise ConfigurationError(f"Vector store not found: {persist_dir}")

        vectorstore = NumpyVectorStore(persist_directory=persist_dir, embedding_function=embeddings)

        # Reconstruct documents from vectorstore
        collection = vectorstore.get()
        documents = []

        if collection and collection["ids"]:
            for i, doc_id in enumerate(collection["ids"]):
                doc = Document(
                    id=doc_id,
                    page_content=collection["documents"][i],
                    metadata=collection["metadatas"][i] if collection["metadatas"] else {},
                )
                documents.append(doc)

        if corpus_directory is not None:
            current = {
                path.name: len(PdfReader(path).pages)
                for path in Path(corpus_directory).glob("*.pdf")
            }
            documents = [
                doc
                for doc in documents
                if (
                    Path(doc.metadata.get("source", "")).name in current
                    and isinstance(doc.metadata.get("page"), int)
                    and 0 <= doc.metadata["page"] < current[Path(doc.metadata["source"]).name]
                )
            ]
            if not documents:
                raise ConfigurationError("No index passages match the bundled PDF corpus")
        return vectorstore, documents

    def build_from_pdfs(
        self, pdf_dir: str, persist_directory: Optional[str] = None
    ) -> NumpyVectorStore:
        """Complete pipeline: load PDFs → split → build vectorstore."""
        documents = self.load_pdfs(pdf_dir)
        chunks = self.split_documents(documents)
        return self.build_vectorstore(chunks, persist_directory)
