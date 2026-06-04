"""Local semantic embedding store — sentence-transformers on CPU.

Uses :class:`BaseNumpyVectorStore` for storage and search.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from memory_system.base_store import BaseNumpyVectorStore

logger = logging.getLogger(__name__)


class LocalEmbeddingStore(BaseNumpyVectorStore):
    """Semantic vector store backed by a local sentence-transformers model.

    Default: ``all-MiniLM-L6-v2`` (384-d, 80MB, CPU-friendly).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dim: int = 384,
    ) -> None:
        super().__init__(dim=dim)
        logger.info("Loading embedding model %s...", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded. dim=%d", dim)

    def embed(self, text: str) -> NDArray[np.float32]:
        if not text.strip():
            return np.zeros(self.dim, dtype=np.float32)
        self.call_count += 1
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def is_semantic(self) -> bool:
        return True
