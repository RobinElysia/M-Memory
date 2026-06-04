"""Hash-based vector store — deterministic, non-semantic, for testing only.

Uses :class:`BaseNumpyVectorStore` for storage and search.
"""

from __future__ import annotations

import hashlib
import logging
import warnings

import numpy as np
from numpy.typing import NDArray

from memory_system.base_store import BaseNumpyVectorStore

logger = logging.getLogger(__name__)


class HashVectorStore(BaseNumpyVectorStore):
    """In-memory hash-based vector store. NOT SEMANTIC. For testing only.

    Embeddings are deterministic SHA-256 hashes. Cosine similarity between
    hash-based vectors is random, not semantically meaningful.
    """

    def __init__(self, dim: int = 1536) -> None:
        super().__init__(dim=dim)

    def embed(self, text: str) -> NDArray[np.float32]:
        """Deterministic hash-based embedding (NOT semantic)."""
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        vec = np.zeros(self.dim, dtype=np.float32)
        for i, ch in enumerate(text.encode("utf-8")):
            h = hashlib.sha256(f"{i}:{ch}".encode()).digest()
            for j in range(min(self.dim, len(h) // 4)):
                val = int.from_bytes(h[j * 4 : (j + 1) * 4], "big")
                vec[j] += (val / 2**32) * 2.0 - 1.0
        norm = np.linalg.norm(vec)
        return ((vec / norm) if norm > 1e-8 else vec).astype(np.float32)

    def is_semantic(self) -> bool:
        return False


# Backward compatibility
class NumpyVectorStore(HashVectorStore):
    """Deprecated alias for HashVectorStore."""

    def __init__(self, dim: int = 1536) -> None:
        warnings.warn(
            "NumpyVectorStore is deprecated. Use HashVectorStore.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(dim=dim)
