"""Query the maintained WeldScope pipeline using deployment configuration."""

from weldscope.core.config import RAGConfig
from weldscope.ingestion.builder import VectorStoreBuilder
from weldscope.retrieval.retriever import HybridRetriever
from weldscope.pipeline.pipeline import RAGPipeline


def create_pipeline():
    config = RAGConfig()
    store, documents = VectorStoreBuilder(config).load_vectorstore()
    return RAGPipeline(HybridRetriever(store, documents, config), config)


if __name__ == "__main__":
    response = create_pipeline().query("What causes wormhole defects in FSW?")
    print(response.answer)
    for source in response.sources:
        print(source.doc_name, source.page_num)
