"""Small, credential-free environment snapshot for hosted release verification."""

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
import platform

FORBIDDEN_RUNTIME_PACKAGES = (
    "huggingface-hub",
    "transformers",
    "sentence-transformers",
    "tokenizers",
    "torch",
    "datasets",
    "chromadb",
    "langchain-chroma",
    "ragas",
    "hf-xet",
    "safetensors",
)


@lru_cache(maxsize=1)
def runtime_environment():
    present = []
    for name in FORBIDDEN_RUNTIME_PACKAGES:
        try:
            version(name)
        except PackageNotFoundError:
            continue
        present.append(name)
    return dict(
        python=platform.python_version(),
        embedding_backend="bundled_onnx",
        index_version="fsw-v2",
        unexpected_packages=present,
    )
