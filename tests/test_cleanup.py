"""Tests for CleanupScheduler."""

from __future__ import annotations

import time

import numpy as np
import pytest

from memory_system.bucket_manager import BucketManagerImpl
from memory_system.cleanup import CleanupScheduler
from memory_system.config import MemorySystemConfig
from memory_system.fake_llm import FakeLLMAdapter, create_conflict_response
from memory_system.graph_engine import NetworkXGraphStore
from memory_system.models import Medoid, MemoryNode
from memory_system.vector_store import HashVectorStore


class TestCleanupScheduler:
    """Tests for CleanupScheduler."""

    @pytest.fixture
    def config(self) -> MemorySystemConfig:
        c = MemorySystemConfig()
        c.embedding_dim = 8
        c.cleanup.interval_seconds = 0.1
        c.cleanup.node_similarity_for_contradiction = 0.5
        c.cleanup.medoid_similarity_floor = 0.3
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
    def nodes(self) -> dict[str, MemoryNode]:
        return {}

    @pytest.fixture
    def bucket_manager(
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

    @pytest.fixture
    def scheduler(
        self,
        config: MemorySystemConfig,
        bucket_manager: BucketManagerImpl,
        vector_store: HashVectorStore,
        llm: FakeLLMAdapter,
        nodes: dict[str, MemoryNode],
    ) -> CleanupScheduler:
        return CleanupScheduler(
            config=config,
            bucket_manager=bucket_manager,
            vector_store=vector_store,
            llm=llm,
            node_registry=nodes,
        )

    def _make_node(
        self,
        node_id: str,
        summary: str,
        content: str = "",
        config: MemorySystemConfig | None = None,
    ) -> MemoryNode:
        dim = config.embedding_dim if config else 8
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

    # ── lifecycle ────────────────────────────────────────────────────────

    def test_start_stop_sync(
        self, scheduler: CleanupScheduler
    ) -> None:
        scheduler.start_sync()
        assert scheduler._running
        scheduler.stop()
        assert not scheduler._running

    # ── dormancy check delegated ─────────────────────────────────────────

    def test_cleanup_cycle_marks_dormant(
        self,
        scheduler: CleanupScheduler,
        bucket_manager: BucketManagerImpl,
        config: MemorySystemConfig,
        nodes: dict[str, MemoryNode],
    ) -> None:
        node = self._make_node("n1", "test", config=config)
        bucket = bucket_manager.create_bucket(node)
        nodes["n1"] = node

        # Artificially age the bucket
        bucket.last_write_at = 0.0
        bucket.last_query_at = 0.0

        scheduler._cleanup_cycle()
        assert bucket.is_dormant

    # ── topic drift scan ─────────────────────────────────────────────────

    def test_topic_drift_marks_stale(
        self,
        scheduler: CleanupScheduler,
        bucket_manager: BucketManagerImpl,
        config: MemorySystemConfig,
        nodes: dict[str, MemoryNode],
    ) -> None:
        # Manually craft vectors: Medoid pointing one way, drift node orthogonal
        medoid_vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        drift_vec = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        n1 = MemoryNode(
            id="n1",
            summary="python",
            content="python",
            summary_vector=medoid_vec.copy(),
            content_vector=medoid_vec.copy(),
            timestamp=time.time(),
            confidence=1.0,
        )
        bucket = bucket_manager.create_bucket(n1)
        # Overwrite Medoid vector with our crafted one
        bucket.medoid = Medoid(
            node_id="n1",
            summary="python",
            vector=medoid_vec.copy(),
            version=0,
        )
        # Update vector index
        bucket_manager._vector_store.remove([f"medoid:{bucket.id}"])
        bucket_manager._vector_store.add(
            vectors=medoid_vec.reshape(1, -1),
            metadata=[{"id": f"medoid:{bucket.id}", "bucket_id": bucket.id}],
        )
        nodes["n1"] = n1
        bucket_manager._nodes["n1"] = n1

        n2 = MemoryNode(
            id="n2",
            summary="cooking",
            content="cooking",
            summary_vector=drift_vec.copy(),
            content_vector=drift_vec.copy(),
            timestamp=time.time(),
            confidence=1.0,
            bucket_id=bucket.id,
        )
        bucket.node_ids.append("n2")
        bucket_manager._nodes["n2"] = n2
        nodes["n2"] = n2

        # n2 is orthogonal → sim ≈ 0.0 < 0.3 floor → should be marked stale
        scheduler._scan_topic_drift(bucket)
        assert n2.is_stale

    # ── contradiction scan ───────────────────────────────────────────────

    def test_contradiction_scan_no_contradiction(
        self,
        scheduler: CleanupScheduler,
        bucket_manager: BucketManagerImpl,
        config: MemorySystemConfig,
        nodes: dict[str, MemoryNode],
    ) -> None:
        n1 = self._make_node("n1", "weather", "sunny", config=config)
        bucket = bucket_manager.create_bucket(n1)
        nodes["n1"] = n1

        # Same content → similar vectors but LLM should say no conflict
        n2 = self._make_node("n2", "weather", "sunny", config=config)
        bucket.node_ids.append("n2")
        bucket_manager._nodes["n2"] = n2
        nodes["n2"] = n2

        # Set up LLM to report no conflicts
        scheduler._llm.add_script(
            "fact-checking",
            create_conflict_response([]),
        )

        scheduler._scan_contradictions(bucket)
        # Neither should be stale
        assert not n1.is_stale
        assert not n2.is_stale

    def test_contradiction_scan_detects_conflict(
        self,
        scheduler: CleanupScheduler,
        bucket_manager: BucketManagerImpl,
        config: MemorySystemConfig,
        nodes: dict[str, MemoryNode],
    ) -> None:
        # Use nearly identical vectors to pass the similarity threshold (0.5)
        vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        near_vec = vec + 0.01 * np.random.RandomState(42).randn(8).astype(np.float32)
        near_vec = near_vec / np.linalg.norm(near_vec)

        n1 = MemoryNode(
            id="n1",
            summary="city",
            content="I live in Beijing",
            summary_vector=vec.copy(),
            content_vector=vec.copy(),
            timestamp=1000.0,
            confidence=0.8,
        )
        bucket = bucket_manager.create_bucket(n1)
        nodes["n1"] = n1
        bucket_manager._nodes["n1"] = n1

        n2 = MemoryNode(
            id="n2",
            summary="city",
            content="I moved to Shanghai",
            summary_vector=near_vec.copy(),
            content_vector=near_vec.copy(),
            timestamp=2000.0,
            confidence=0.9,
            bucket_id=bucket.id,
        )
        bucket.node_ids.append("n2")
        bucket_manager._nodes["n2"] = n2
        nodes["n2"] = n2

        # LLM detects contradiction: newer is n2 (idx 1), older is n1 (idx 0)
        scheduler._llm.add_script(
            "fact-checking",
            create_conflict_response(
                [{"newer_id": "1", "older_id": "0", "reason": "location changed"}]
            ),
        )

        scheduler._scan_contradictions(bucket)
        # n1 should be marked stale (older)
        assert n1.is_stale
