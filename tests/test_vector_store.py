"""Tests for NumPy-based VectorStore implementation."""

from __future__ import annotations

import numpy as np
import pytest

from memory_system.vector_store import NumpyVectorStore


class TestNumpyVectorStore:
    """Unit tests for NumpyVectorStore — happy path, edge cases, and error paths."""

    # ── Fixtures ──────────────────────────────────────────────────────────

    @pytest.fixture
    def store(self) -> NumpyVectorStore:
        return NumpyVectorStore(dim=8)

    @pytest.fixture
    def sample_texts(self) -> list[str]:
        return ["hello world", "goodbye world", "foo bar baz"]

    # ── embed ─────────────────────────────────────────────────────────────

    def test_embed_returns_correct_shape(self, store: NumpyVectorStore) -> None:
        result = store.embed("hello")
        assert isinstance(result, np.ndarray)
        assert result.shape == (8,)
        assert result.dtype == np.float32

    def test_embed_is_deterministic(self, store: NumpyVectorStore) -> None:
        a = store.embed("hello")
        b = store.embed("hello")
        np.testing.assert_array_equal(a, b)

    def test_embed_empty_string(self, store: NumpyVectorStore) -> None:
        result = store.embed("")
        assert result.shape == (8,)

    # ── add ───────────────────────────────────────────────────────────────

    def test_add_returns_ids(self, store: NumpyVectorStore) -> None:
        vecs = np.random.randn(3, 8).astype(np.float32)
        meta = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        ids = store.add(vecs, meta)
        assert ids == ["a", "b", "c"]

    def test_add_increments_count(self, store: NumpyVectorStore) -> None:
        vecs = np.random.randn(2, 8).astype(np.float32)
        meta = [{"id": "x"}, {"id": "y"}]
        store.add(vecs, meta)
        assert store.count() == 2

    def test_add_missing_id_raises(self, store: NumpyVectorStore) -> None:
        vecs = np.random.randn(1, 8).astype(np.float32)
        with pytest.raises(ValueError, match="id"):
            store.add(vecs, [{"not_id": "x"}])

    def test_add_shape_mismatch_raises(self, store: NumpyVectorStore) -> None:
        vecs = np.random.randn(3, 8).astype(np.float32)
        with pytest.raises(ValueError, match="metadata"):
            store.add(vecs, [{"id": "x"}, {"id": "y"}])

    # ── search ────────────────────────────────────────────────────────────

    def test_search_returns_top_results(self, store: NumpyVectorStore) -> None:
        # Three well-separated vectors
        v1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        store.add(
            np.stack([v1, v2, v3]),
            [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        )

        # Search for something very close to v1
        query = v1 + 0.01 * np.random.randn(8).astype(np.float32)
        results = store.search(query, top_k=2)
        assert len(results) == 2
        assert results[0][0] == "a"  # closest to v1
        assert results[0][1] > results[1][1]  # descending

    def test_search_top_k_zero_raises(self, store: NumpyVectorStore) -> None:
        with pytest.raises(ValueError, match="top_k"):
            store.search(np.ones(8, dtype=np.float32), 0)

    def test_search_empty_index(self, store: NumpyVectorStore) -> None:
        results = store.search(np.ones(8, dtype=np.float32), top_k=3)
        assert results == []

    # ── remove ────────────────────────────────────────────────────────────

    def test_remove_decrements_count(self, store: NumpyVectorStore) -> None:
        vecs = np.random.randn(3, 8).astype(np.float32)
        meta = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        store.add(vecs, meta)
        store.remove(["a", "c"])
        assert store.count() == 1

    def test_remove_missing_id_raises(self, store: NumpyVectorStore) -> None:
        with pytest.raises(KeyError):
            store.remove(["nonexistent"])

    # ── count ─────────────────────────────────────────────────────────────

    def test_count_starts_zero(self, store: NumpyVectorStore) -> None:
        assert store.count() == 0

    # ── cosine similarity ─────────────────────────────────────────────────

    def test_identical_vectors_similarity_one(self, store: NumpyVectorStore) -> None:
        v = np.ones(8, dtype=np.float32)
        sim = store._cosine_similarity(v, v)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_similarity_zero(self, store: NumpyVectorStore) -> None:
        a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        sim = store._cosine_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=1e-5)
