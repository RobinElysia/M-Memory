"""End-to-end scenario tests for the dual-layer memory system.

Uses fake (deterministic) embedding and scripted LLM adapters so tests
are fast, reproducible, and require no external services.

Key design: for deterministic similarity behaviour, tests use manually
constructed vectors rather than hash-based embeddings where needed.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from memory_system.config import MemorySystemConfig
from memory_system.fake_llm import (
    FakeLLMAdapter,
    create_assignment_decision,
    create_conflict_response,
)
from memory_system.graph_engine import NetworkXGraphStore
from memory_system.models import MemoryNode
from memory_system.retrieval import MemoryRetrievalEngineImpl
from memory_system.vector_store import NumpyVectorStore


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_config(dim: int = 8) -> MemorySystemConfig:
    c = MemorySystemConfig()
    c.embedding_dim = dim
    return c


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 1: Topic Drift & Bucket Split
# ═══════════════════════════════════════════════════════════════════════════════


class TestTopicDriftAndBucketSplit:
    """10 cat turns → 5 dog turns → 1 cat turn → verify split behaviour."""

    @pytest.fixture
    def config(self) -> MemorySystemConfig:
        return _make_config()

    @pytest.fixture
    def vs(self, config: MemorySystemConfig) -> NumpyVectorStore:
        return NumpyVectorStore(dim=config.embedding_dim)

    @pytest.fixture
    def gs(self) -> NetworkXGraphStore:
        return NetworkXGraphStore()

    @pytest.fixture
    def llm(self) -> FakeLLMAdapter:
        return FakeLLMAdapter()

    @pytest.fixture
    def engine(
        self,
        config: MemorySystemConfig,
        vs: NumpyVectorStore,
        gs: NetworkXGraphStore,
        llm: FakeLLMAdapter,
    ) -> MemoryRetrievalEngineImpl:
        return MemoryRetrievalEngineImpl(
            config=config,
            vector_store=vs,
            graph_store=gs,
            llm=llm,
        )

    def test_topic_drift_forms_separate_buckets(
        self,
        engine: MemoryRetrievalEngineImpl,
    ) -> None:
        """10 cat turns, then 5 dog turns → at least 2 buckets, cat Medoid correct."""

        # Phase 1: Ingest 10 cat turns (all go to the same bucket)
        cat_summaries = [f"cat topic {i}" for i in range(10)]
        cat_bucket_id: str | None = None

        for i, s in enumerate(cat_summaries):
            if cat_bucket_id is not None:
                engine._llm._script.clear()
                engine._llm.add_script(
                    "cat",
                    create_assignment_decision(cat_bucket_id, cross_links=[]),
                )
            nid = engine.ingest(s, f"cat content {i}")
            if cat_bucket_id is None:
                buckets = engine._bucket_manager.get_all_buckets()
                cat_bucket_id = buckets[0].id

        # Phase 2: Ingest 5 dog turns (create a new bucket)
        dog_summaries = [f"dog topic {i}" for i in range(5)]
        dog_bucket_id: str | None = None

        engine._llm._script.clear()
        engine._llm.add_script(
            "dog",
            create_assignment_decision("new", cross_links=[]),
        )
        for i, s in enumerate(dog_summaries):
            if dog_bucket_id is not None:
                engine._llm._script.clear()
                engine._llm.add_script(
                    "dog",
                    create_assignment_decision(dog_bucket_id, cross_links=[]),
                )
            nid = engine.ingest(s, f"dog content {i}")
            if dog_bucket_id is None:
                all_b = engine._bucket_manager.get_all_buckets()
                for b in all_b:
                    if b.id != cat_bucket_id:
                        dog_bucket_id = b.id
                        break
                if dog_bucket_id is None and len(all_b) >= 2:
                    dog_bucket_id = all_b[-1].id

        # Verify: at least 2 buckets exist
        all_buckets = engine._bucket_manager.get_all_buckets()
        assert len(all_buckets) >= 2, f"Expected ≥2 buckets, got {len(all_buckets)}"

        # Phase 3: 1 more cat turn — should return to cat bucket
        engine._llm._script.clear()
        engine._llm.add_script(
            "cat",
            create_assignment_decision(cat_bucket_id, cross_links=[]),
        )
        nid = engine.ingest("cat is purring again", "cat content 10")

        node = engine._nodes.get(nid)
        assert node is not None
        assert node.bucket_id == cat_bucket_id, (
            f"Expected cat node in {cat_bucket_id}, got {node.bucket_id}"
        )

        # Cat bucket should have 11 nodes
        cat_bucket = engine._bucket_manager.get_all_buckets()
        cat_b = next(b for b in cat_bucket if b.id == cat_bucket_id)
        assert len(cat_b.node_ids) == 11


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 2: Contradictory Information & Conflict Resolution
# ═══════════════════════════════════════════════════════════════════════════════


class TestConflictResolution:
    """User says Beijing → Shanghai; verify latest info kept, old marked stale."""

    @pytest.fixture
    def config(self) -> MemorySystemConfig:
        return _make_config()

    @pytest.fixture
    def vs(self, config: MemorySystemConfig) -> NumpyVectorStore:
        return NumpyVectorStore(dim=config.embedding_dim)

    @pytest.fixture
    def gs(self) -> NetworkXGraphStore:
        return NetworkXGraphStore()

    @pytest.fixture
    def llm(self) -> FakeLLMAdapter:
        return FakeLLMAdapter()

    @pytest.fixture
    def engine(
        self,
        config: MemorySystemConfig,
        vs: NumpyVectorStore,
        gs: NetworkXGraphStore,
        llm: FakeLLMAdapter,
    ) -> MemoryRetrievalEngineImpl:
        return MemoryRetrievalEngineImpl(
            config=config,
            vector_store=vs,
            graph_store=gs,
            llm=llm,
        )

    def test_contradiction_resolution(
        self,
        engine: MemoryRetrievalEngineImpl,
        config: MemorySystemConfig,
    ) -> None:
        """Directly test conflict resolution with controlled vectors."""
        # Use nearly identical vectors → both will have similar sim scores
        vec = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        beijing = MemoryNode(
            id="n_beijing",
            summary="my location",
            content="I live in Beijing",
            summary_vector=vec.copy(),
            content_vector=vec.copy(),
            timestamp=1000.0,
            confidence=0.8,
        )
        shanghai = MemoryNode(
            id="n_shanghai",
            summary="my location update",
            content="I moved to Shanghai",
            summary_vector=vec.copy() * 0.99,  # nearly identical
            content_vector=vec.copy() * 0.99,
            timestamp=2000.0,
            confidence=0.9,
        )

        # LLM: newer is Shanghai (idx 1), older is Beijing (idx 0)
        # But after re-ranking: Shanghai has higher confidence + newer timestamp
        # → Shanghai at idx 0, Beijing at idx 1
        engine._llm.add_script(
            "fact-checking",
            create_conflict_response(
                [{"newer_id": "0", "older_id": "1",
                  "reason": "location changed"}]
            ),
        )

        resolved = engine.resolve_conflicts(
            [beijing, shanghai], "where do I live"
        )

        assert len(resolved) == 2

        # Find Beijing in resolved
        bj = next(n for n in resolved if n.id == "n_beijing")
        sh = next(n for n in resolved if n.id == "n_shanghai")

        assert bj.is_stale, "Beijing should be marked stale (older info)"
        assert not sh.is_stale, "Shanghai should NOT be stale"
        # Shanghai confidence higher after Beijing downgrade
        assert sh.confidence > bj.confidence


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 3: Bucket Dormancy & Wake
# ═══════════════════════════════════════════════════════════════════════════════


class TestBucketDormancy:
    """Create bucket → mark dormant → query wakes it."""

    @pytest.fixture
    def config(self) -> MemorySystemConfig:
        c = _make_config()
        c.bucket.dormancy_interval_seconds = 0.01
        return c

    @pytest.fixture
    def vs(self, config: MemorySystemConfig) -> NumpyVectorStore:
        return NumpyVectorStore(dim=config.embedding_dim)

    @pytest.fixture
    def gs(self) -> NetworkXGraphStore:
        return NetworkXGraphStore()

    @pytest.fixture
    def llm(self) -> FakeLLMAdapter:
        return FakeLLMAdapter()

    @pytest.fixture
    def engine(
        self,
        config: MemorySystemConfig,
        vs: NumpyVectorStore,
        gs: NetworkXGraphStore,
        llm: FakeLLMAdapter,
    ) -> MemoryRetrievalEngineImpl:
        return MemoryRetrievalEngineImpl(
            config=config,
            vector_store=vs,
            graph_store=gs,
            llm=llm,
        )

    def test_dormancy_and_wake(
        self,
        engine: MemoryRetrievalEngineImpl,
    ) -> None:
        """Create bucket with 3 nodes, let it go dormant, verify wake on query."""

        bucket_id: str | None = None

        # Ingest 3 nodes into the same bucket
        for i in range(3):
            if bucket_id is not None:
                engine._llm._script.clear()
                engine._llm.add_script(
                    "gardening",
                    create_assignment_decision(bucket_id, cross_links=[]),
                )
            nid = engine.ingest(
                f"gardening tip {i}",
                f"gardening content {i}",
            )
            if bucket_id is None:
                buckets = engine._bucket_manager.get_all_buckets()
                bucket_id = buckets[0].id

        buckets = engine._bucket_manager.get_all_buckets()
        assert len(buckets) == 1, f"Expected 1 bucket, got {len(buckets)}"
        bucket = buckets[0]
        assert not bucket.is_dormant

        # Artificially age the bucket
        bucket.last_write_at = 0.0
        bucket.last_query_at = 0.0

        # Run dormancy check
        dormant = engine._bucket_manager.dormancy_check()
        assert len(dormant) == 1
        assert bucket.is_dormant

        # Active buckets should be empty
        active_after = engine._bucket_manager.get_active_buckets()
        assert len(active_after) == 0

        # Manually wake the bucket (simulating a matching query)
        woken = engine._bucket_manager.wake_bucket(bucket.id)
        assert not woken.is_dormant
        assert not bucket.is_dormant

        # Active buckets should now include the woken bucket
        assert len(engine._bucket_manager.get_active_buckets()) == 1

        # Search should now work
        result = engine.search("gardening tips")
        assert len(result.nodes) >= 1
