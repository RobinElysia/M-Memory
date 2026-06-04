"""Shared utilities for the memory_system package."""

import numpy as np
from numpy.typing import NDArray

# Unified stopwords — single source of truth for lexical search + bucket matching
STOPWORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
    "in", "on", "at", "to", "of", "for", "with", "and", "or", "not",
    "be", "has", "have", "it", "its", "that", "this", "these", "those",
    "from", "by", "i", "my", "me", "you", "your", "now", "before",
    "after", "what", "how", "when", "where", "who", "why", "about",
}


def cosine_sim(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine similarity between two normalized vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-8 or b_norm < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))
