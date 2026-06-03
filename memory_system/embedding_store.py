"""OpenAI-compatible semantic embedding store.

Uses OpenAI/DeepSeek/any OpenAI-compatible embedding API to produce
semantically meaningful vectors.  Drop-in replacement for NumpyVectorStore.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI

from memory_system.interfaces import VectorStore

logger = logging.getLogger(__name__)


class OpenAIEmbeddingStore(VectorStore):
    """Semantic vector store backed by OpenAI-compatible embedding API.

    Uses ``text-embedding-3-small`` by default (1536-d).  Works with any
    provider that exposes an OpenAI-compatible ``/v1/embeddings`` endpoint
    (DeepSeek, OpenAI, local vLLM, etc.).

    Args:
        api_key: API key (defaults to ``OPENAI_API_KEY`` env var).
        model: Embedding model name.
        base_url: API base URL.
        dim: Embedding dimensionality (must match model output).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        dim: int = 1536,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY env var or pass api_key=..."
            )
        self._client = OpenAI(api_key=key, base_url=base_url)
        self._model = model
        self.dim = dim
        self._vectors: NDArray[np.float32] | None = None
        self._metadata: list[dict[str, Any]] = []
        self.call_count: int = 0
        self.total_tokens: int = 0

    def embed(self, text: str) -> NDArray[np.float32]:
        """Produce semantic embedding via API."""
        if not text:
            return np.zeros(self.dim, dtype=np.float32)

        self.call_count += 1
        try:
            resp = self._client.embeddings.create(
                model=self._model,
                input=text,
            )
        except Exception as exc:
            logger.error("Embedding API error: %s", exc)
            raise RuntimeError(f"Embedding API error: {exc}") from exc

        if resp.usage:
            self.total_tokens += resp.usage.total_tokens

        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec

    def add(
        self, vectors: NDArray[np.float32], metadata: list[dict[str, Any]]
    ) -> list[str]:
        """Insert pre-computed vectors with metadata."""
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {vectors.shape}")
        if vectors.shape[0] != len(metadata):
            raise ValueError(
                f"Count mismatch: {vectors.shape[0]} vectors vs {len(metadata)} meta"
            )

        ids: list[str] = []
        for i, meta in enumerate(metadata):
            if "id" not in meta:
                raise ValueError(f"Metadata[{i}] missing 'id' key")
            ids.append(meta["id"])

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        normalized = vectors / norms

        if self._vectors is None:
            self._vectors = normalized.astype(np.float32)
        else:
            self._vectors = np.vstack([self._vectors, normalized]).astype(np.float32)

        self._metadata.extend(metadata)
        return ids

    def search(
        self, query_vector: NDArray[np.float32], top_k: int
    ) -> list[tuple[str, float]]:
        """Cosine similarity search."""
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if self._vectors is None or self._vectors.shape[0] == 0:
            return []

        q_norm = np.linalg.norm(query_vector)
        if q_norm > 1e-8:
            query_vector = query_vector / q_norm

        sims = np.dot(self._vectors, query_vector)
        top_k = min(top_k, len(sims))
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self._metadata[i]["id"], float(sims[i])) for i in top_indices]

    def remove(self, ids: list[str]) -> None:
        """Remove vectors by ID."""
        if self._vectors is None:
            raise KeyError(f"No vectors; cannot remove {ids}")
        existing = {m["id"] for m in self._metadata}
        for rid in ids:
            if rid not in existing:
                raise KeyError(f"ID '{rid}' not found")

        keep = np.ones(self._vectors.shape[0], dtype=bool)
        for i, meta in enumerate(self._metadata):
            if meta["id"] in ids:
                keep[i] = False

        if not keep.any():
            self._vectors = None
            self._metadata = []
        else:
            self._vectors = self._vectors[keep]
            self._metadata = [m for i, m in enumerate(self._metadata) if keep[i]]

    def count(self) -> int:
        """Number of stored vectors."""
        return 0 if self._vectors is None else self._vectors.shape[0]

    def is_semantic(self) -> bool:
        """API-based embeddings ARE semantically meaningful."""
        return True
