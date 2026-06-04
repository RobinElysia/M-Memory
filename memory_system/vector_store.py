"""NumPy-based VectorStore implementation.

Uses in-memory numpy arrays with cosine similarity as the default backend.
Deterministic (no random), suitable as the default and for testing.
"""

from __future__ import annotations

import hashlib
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from memory_system.interfaces import VectorStore


class HashVectorStore(VectorStore):
    """In-memory hash-based vector store. NOT SEMANTIC. For testing only.

    Embedding is done via a deterministic hash-based projection (NOT a real
    language model) — this makes tests reproducible.  For production, inject
    a real embedding model via a different ``VectorStore`` implementation.

    Attributes:
        dim: Dimensionality of the vector space.
        _vectors: Stored vectors as (n, dim) array or None when empty.
        _metadata: List of metadata dicts (1:1 with rows of ``_vectors``).
    """

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim
        self._vectors: NDArray[np.float32] | None = None
        self._metadata: list[dict[str, Any]] = []

    # ── VectorStore interface ────────────────────────────────────────────

    def embed(self, text: str) -> NDArray[np.float32]:
        """Deterministic hash-based embedding.

        Uses SHA-256 to derive a pseudo-random but repeatable vector.
        This is NOT semantically meaningful — it is here to satisfy the
        interface for testing.  Real implementations should override.
        """
        if not text:
            # Empty string → zero vector
            return np.zeros(self.dim, dtype=np.float32)

        # Deterministic: hash each character position to derive vector components
        vec = np.zeros(self.dim, dtype=np.float32)
        for i, ch in enumerate(text.encode("utf-8")):
            h = hashlib.sha256(f"{i}:{ch}".encode()).digest()
            # Use 4 bytes at a time to build float32 components
            for j in range(min(self.dim, len(h) // 4)):
                val = int.from_bytes(h[j * 4 : (j + 1) * 4], "big")
                vec[j] += (val / 2**32) * 2.0 - 1.0  # scale to [-1, 1]

        # L2-normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec.astype(np.float32)

    def add(
        self, vectors: NDArray[np.float32], metadata: list[dict[str, Any]]
    ) -> list[str]:
        """Insert *vectors* with associated *metadata*."""
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"Expected vectors of shape (n, {self.dim}), got {vectors.shape}"
            )
        if vectors.shape[0] != len(metadata):
            raise ValueError(
                f"vectors count ({vectors.shape[0]}) != metadata count ({len(metadata)})"
            )

        ids: list[str] = []
        for i, meta in enumerate(metadata):
            if "id" not in meta:
                raise ValueError(f"Metadata at index {i} missing 'id' key")
            ids.append(meta["id"])

        # L2-normalize all incoming vectors
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
        """Cosine-similarity search over stored vectors."""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if self._vectors is None or self._vectors.shape[0] == 0:
            return []

        # Normalize query
        q_norm = np.linalg.norm(query_vector)
        if q_norm > 1e-8:
            query_vector = query_vector / q_norm

        # Cosine similarity = dot product of normalized vectors
        sims = np.dot(self._vectors, query_vector)
        top_k = min(top_k, len(sims))
        top_indices = np.argsort(sims)[::-1][:top_k]

        return [
            (self._metadata[i]["id"], float(sims[i]))
            for i in top_indices
        ]

    def remove(self, ids: list[str]) -> None:
        """Remove vectors by their IDs."""
        if self._vectors is None:
            raise KeyError(f"No vectors in store; cannot remove {ids}")

        existing_ids = {m["id"] for m in self._metadata}
        for rid in ids:
            if rid not in existing_ids:
                raise KeyError(f"ID '{rid}' not found in store")

        keep_mask = np.ones(self._vectors.shape[0], dtype=bool)
        for i, meta in enumerate(self._metadata):
            if meta["id"] in ids:
                keep_mask[i] = False

        if not keep_mask.any():
            self._vectors = None
            self._metadata = []
        else:
            self._vectors = self._vectors[keep_mask]
            self._metadata = [m for i, m in enumerate(self._metadata) if keep_mask[i]]

    def count(self) -> int:
        """Number of vectors currently stored."""
        if self._vectors is None:
            return 0
        return self._vectors.shape[0]

    def is_semantic(self) -> bool:
        """Hash-based vectors are NOT semantically meaningful."""
        return False

    # ── Utility ──────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(
        a: NDArray[np.float32], b: NDArray[np.float32]
    ) -> float:
        """Compute cosine similarity between two 1-D vectors."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm < 1e-8 or b_norm < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))


# Backward compatibility alias with deprecation warning
class NumpyVectorStore(HashVectorStore):
    """Deprecated alias for HashVectorStore. Use LocalEmbeddingStore instead."""

    def __init__(self, dim: int = 1536) -> None:
        warnings.warn(
            "NumpyVectorStore is deprecated. Use LocalEmbeddingStore for "
            "semantic embeddings or HashVectorStore for testing.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(dim=dim)
