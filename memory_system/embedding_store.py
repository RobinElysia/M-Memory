"""OpenAI-compatible cloud embedding store.

Uses :class:`BaseNumpyVectorStore` for storage and search.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI

from memory_system.base_store import BaseNumpyVectorStore

logger = logging.getLogger(__name__)


class OpenAIEmbeddingStore(BaseNumpyVectorStore):
    """Semantic vector store via OpenAI-compatible /v1/embeddings API.

    Args:
        api_key: API key (defaults to OPENAI_API_KEY).
        model: Embedding model name.
        base_url: API base URL.
        dim: Embedding dimensionality.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        dim: int = 1536,
    ) -> None:
        super().__init__(dim=dim)
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY required")
        self._client = OpenAI(api_key=key, base_url=base_url)
        self._model = model
        self.total_tokens: int = 0

    def embed(self, text: str) -> NDArray[np.float32]:
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        self.call_count += 1
        try:
            resp = self._client.embeddings.create(model=self._model, input=text)
        except Exception as exc:
            raise RuntimeError(f"Embedding API error: {exc}") from exc
        if resp.usage:
            self.total_tokens += resp.usage.total_tokens
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 1e-8 else vec

    def is_semantic(self) -> bool:
        return True
