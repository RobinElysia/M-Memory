"""Local semantic embedding store using sentence-transformers.

Uses all-MiniLM-L6-v2 (384-d, lightweight, runs on CPU).
Drop-in replacement for NumpyVectorStore — enables _semantic_search().
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from memory_system.interfaces import VectorStore

logger = logging.getLogger(__name__)


class LocalEmbeddingStore(VectorStore):
    """Semantic vector store backed by a local sentence-transformers model.

    Default: ``all-MiniLM-L6-v2`` (384-d, 80MB, CPU-friendly).
    First ``embed()`` call downloads the model; subsequent calls are fast.

    Args:
        model_name: HuggingFace model identifier.
        dim: Embedding dimensionality (must match model).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dim: int = 384,
    ) -> None:
        logger.info("Loading embedding model %s...", model_name)
        self._model = SentenceTransformer(model_name)
        self.dim = dim
        self._vectors: NDArray[np.float32] | None = None
        self._metadata: list[dict[str, Any]] = []
        self.call_count: int = 0
        logger.info("Embedding model loaded. dim=%d", dim)

    def embed(self, text: str) -> NDArray[np.float32]:
        """Produce semantic embedding via local model."""
        if not text.strip():
            return np.zeros(self.dim, dtype=np.float32)
        self.call_count += 1
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def add(
        self, vectors: NDArray[np.float32], metadata: list[dict[str, Any]]
    ) -> list[str]:
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2-D, got {vectors.shape}")
        if vectors.shape[0] != len(metadata):
            raise ValueError(f"Count mismatch: {vectors.shape[0]} vs {len(metadata)}")
        ids = []
        for i, m in enumerate(metadata):
            if "id" not in m:
                raise ValueError(f"Metadata[{i}] missing 'id'")
            ids.append(m["id"])
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
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if self._vectors is None or self._vectors.shape[0] == 0:
            return []
        q = query_vector / (np.linalg.norm(query_vector) or 1.0)
        sims = np.dot(self._vectors, q)
        top_k = min(top_k, len(sims))
        idx = np.argsort(sims)[::-1][:top_k]
        return [(self._metadata[i]["id"], float(sims[i])) for i in idx]

    def remove(self, ids: list[str]) -> None:
        if self._vectors is None:
            raise KeyError(f"No vectors; cannot remove {ids}")
        existing = {m["id"] for m in self._metadata}
        for rid in ids:
            if rid not in existing:
                raise KeyError(f"ID '{rid}' not found")
        keep = np.ones(self._vectors.shape[0], dtype=bool)
        for i, m in enumerate(self._metadata):
            if m["id"] in ids:
                keep[i] = False
        if not keep.any():
            self._vectors = None
            self._metadata = []
        else:
            self._vectors = self._vectors[keep]
            self._metadata = [m for i, m in enumerate(self._metadata) if keep[i]]

    def count(self) -> int:
        return 0 if self._vectors is None else self._vectors.shape[0]

    def is_semantic(self) -> bool:
        return True
