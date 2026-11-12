"""RAGAS embedding adapter for the same bundled, network-free ONNX model as the app."""

import asyncio
from ragas.embeddings.base import BaseRagasEmbedding
from weldscope.core.local_models import LocalModel


class LocalRagasEmbeddings(BaseRagasEmbedding):
    def __init__(self, model="embedding"):
        super().__init__()
        self.model = LocalModel(model)

    def embed_text(self, text, **kwargs):
        return self.model.encode(text).tolist()

    async def aembed_text(self, text, **kwargs):
        return await asyncio.to_thread(self.embed_text, text)
