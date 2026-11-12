"""CLI script for building vector store from PDFs."""

import argparse
from pathlib import Path

from weldscope.core.config import RAGConfig
from weldscope.ingestion.builder import VectorStoreBuilder


def main():
    """Build vector store from PDF directory."""
    parser = argparse.ArgumentParser(description="Build vector store from FSW PDF documents")
    parser.add_argument("--pdf-dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./indexes/fsw-v2",
        help="Output directory for vector store (default: ./indexes/fsw-v2)",
    )

    args = parser.parse_args()

    config = RAGConfig()
    builder = VectorStoreBuilder(config)

    print(f"Loading PDFs from: {args.pdf_dir}")
    documents = builder.load_pdfs(args.pdf_dir)
    print(f"Loaded {len(documents)} pages")

    print(f"Splitting documents (chunk_size={config.chunk_size}, overlap={config.chunk_overlap})")
    chunks = builder.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print(f"Building vector store with {config.embedding_model}")
    vectorstore = builder.build_vectorstore(chunks, args.output_dir)
    print(f"Vector store saved to: {args.output_dir}")
    print(f"Total documents in store: {len(vectorstore.get()["ids"])}")

    print("\nVector store build complete!")


if __name__ == "__main__":
    main()
