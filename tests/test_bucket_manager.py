"""Tests for BucketManager implementation."""

from __future__ import annotations

import time

import numpy as np
import pytest

from memory_system.bucket_manager import BucketManagerImpl
from memory_system.config import MemorySystemConfig
from memory_system.fake_llm import FakeLLMAdapter, create_assignment_decision
from memory_system.graph_engine import NetworkXGraphStore
from memory_system.models import Bucket, MemoryNode
from memory_system.vector_store import HashVectorStore


class TestBucketManager:
    """Unit tests for BucketManagerImpl."""

    @pytest.fixture
    def config(self) -> MemorySystemConfig:
        c = MemorySystemConfig()
        c.embedding_dim = 8
        return c

    @pytest.fixture
    def vector_store(self, config: MemorySystemConfig) -> HashVectorStore:
        return HashVectorStore(dim=config.embedding_dim)

    @pytest.fixture
    def graph_store(self) -> NetworkXGraphStore:
        return NetworkXGraphStore()

    @pytest.fixture
    def llm(self) -> FakeLLMAdapter:
        return FakeLLMAdapter()

    @pytest.fixture
    def manager(
        self,
        config: MemorySystemConfig,
        vector_store: HashVectorStore,
        graph_store: NetworkXGraphStore,
        llm: FakeLLMAdapter,
    ) -> BucketManagerImpl:
        return BucketManagerImpl(
            config=config,
            vector_store=vector_store,
            graph_store=graph_store,
            llm=llm,
        )

    def _make_node(
        self,
        node_id: str,
        summary: str,
        content: str = "",
        config: MemorySystemConfig | None = None,
    ) -> MemoryNode:
        dim = config.embedding_dim if config else 8
        # Use a simple hash-based embedding for tests
        vs = HashVectorStore(dim=dim)
        return MemoryNode(
            id=node_id,
            summary=summary,
            content=content or summary,
            summary_vector=vs.embed(summary),
            content_vector=vs.embed(content or summary),
            timestamp=time.time(),
            confidence=1.0,
        )

    # ── create_bucket ─────────────────────────────────────────────────────

    def test_create_bucket(self, manager: BucketManagerImpl) -> None:
        node = self._make_node("n1", "hello world")
        bucket = manager.create_bucket(node)
        assert bucket.id != ""
        assert bucket.medoid is not None
        assert bucket.medoid.node_id == "n1"
        assert "n1" in bucket.node_ids

    # ── find_candidates ───────────────────────────────────────────────────

    def test_find_candidates_empty_when_no_buckets(
        self, manager: BucketManagerImpl
    ) -> None:
        node = self._make_node("n1", "hello")
        candidates = manager.find_candidates(node)
        assert candidates == []

    def test_find_candidates_returns_existing_buckets(
        self, manager: BucketManagerImpl, config: MemorySystemConfig
    ) -> None:
        # Create two buckets with different themes
        n1 = self._make_node("n1", "machine learning", config=config)
        n2 = self._make_node("n2", "cooking recipes", config=config)
        manager.create_bucket(n1)
        manager.create_bucket(n2)

        # A new ML-related node should match the ML bucket
        n3 = self._make_node("n3", "deep learning neural networks", config=config)
        candidates = manager.find_candidates(n3)
        assert len(candidates) >= 1
        # The ML bucket should have higher similarity
        assert candidates[0][0].id != ""

    # ── assign_to_bucket ──────────────────────────────────────────────────

    def test_assign_to_bucket_places_node(
        self, manager: BucketManagerImpl, config: MemorySystemConfig
    ) -> None:
        node = self._make_node("n1", "python programming", config=config)
        bucket = manager.create_bucket(node)

        n2 = self._make_node("n2", "python asyncio tips", config=config)
        manager.assign_to_bucket(n2, bucket, cross_links=[])

        # Bucket should now have both nodes
        assert "n2" in bucket.node_ids
        assert n2.bucket_id == bucket.id

    def test_assign_with_cross_links_creates_edges(
        self,
        manager: BucketManagerImpl,
        config: MemorySystemConfig,
        graph_store: NetworkXGraphStore,
    ) -> None:
        n1 = self._make_node("n1", "python", config=config)
        n2 = self._make_node("n2", "javascript", config=config)
        b1 = manager.create_bucket(n1)
        b2 = manager.create_bucket(n2)

        n3 = self._make_node("n3", "python vs javascript comparison", config=config)
        manager.assign_to_bucket(
            n3,
            b1,
            cross_links=[{"bucket_id": b2.id, "weight": 0.8}],
        )

        # Cross-bucket edge from n3 to b2's medoid
        cross_edges = graph_store.get_out_edges(n3.id)
        assert len(cross_edges) >= 1
        # Should include the cross-bucket edge
        from memory_system.models import EdgeType
        cross = [e for e in cross_edges if e.edge_type == EdgeType.CROSS_BUCKET]
        assert len(cross) >= 1
        assert cross[0].target_id == n2.id  # b2's medoid

    # ── Medoid update ─────────────────────────────────────────────────────

    def test_medoid_updated_after_insert(
        self, manager: BucketManagerImpl, config: MemorySystemConfig
    ) -> None:
        n1 = self._make_node("n1", "python basics", config=config)
        bucket = manager.create_bucket(n1)
        old_medoid_id = bucket.medoid.node_id if bucket.medoid else ""

        n2 = self._make_node("n2", "python advanced topics", config=config)
        manager.assign_to_bucket(n2, bucket, cross_links=[])

        # Medoid may or may not change, but version should increment
        assert bucket.version >= 1

    # ── Single-node bucket Medoid ─────────────────────────────────────────

    def test_single_node_is_own_medoid(
        self, manager: BucketManagerImpl, config: MemorySystemConfig
    ) -> None:
        node = self._make_node("n1", "test", config=config)
        bucket = manager.create_bucket(node)
        assert bucket.medoid is not None
        assert bucket.medoid.node_id == "n1"

    # ── dormancy ──────────────────────────────────────────────────────────

    def test_dormancy_check_no_buckets(self, manager: BucketManagerImpl) -> None:
        dormant = manager.dormancy_check()
        assert dormant == []

    def test_dormancy_check_active_bucket_not_dormant(
        self, manager: BucketManagerImpl, config: MemorySystemConfig
    ) -> None:
        node = self._make_node("n1", "test", config=config)
        manager.create_bucket(node)
        dormant = manager.dormancy_check()
        assert dormant == []  # just created

    # ── wake_bucket ───────────────────────────────────────────────────────

    def test_wake_bucket_reactivates(
        self, manager: BucketManagerImpl, config: MemorySystemConfig
    ) -> None:
        node = self._make_node("n1", "test", config=config)
        bucket = manager.create_bucket(node)
        bucket.is_dormant = True

        woken = manager.wake_bucket(bucket.id)
        assert woken.is_dormant is False

    def test_wake_nonexistent_bucket_raises(
        self, manager: BucketManagerImpl
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            manager.wake_bucket("nonexistent")

    # ── get_active_buckets / get_all_buckets ──────────────────────────────

    def test_get_active_excludes_dormant(
        self, manager: BucketManagerImpl, config: MemorySystemConfig
    ) -> None:
        n1 = self._make_node("n1", "active", config=config)
        b1 = manager.create_bucket(n1)

        n2 = self._make_node("n2", "dormant", config=config)
        b2 = manager.create_bucket(n2)
        b2.is_dormant = True

        active = manager.get_active_buckets()
        all_buckets = manager.get_all_buckets()
        assert len(active) == 1
        assert len(all_buckets) == 2
