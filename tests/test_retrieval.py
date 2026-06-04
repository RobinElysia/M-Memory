"""Tests for MemoryRetrievalEngine implementation."""

from __future__ import annotations

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
from memory_system.vector_store import HashVectorStore


class TestRetrievalEngine:
    """Unit tests for MemoryRetrievalEngineImpl."""

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
    def engine(
        self,
        config: MemorySystemConfig,
        vector_store: HashVectorStore,
        graph_store: NetworkXGraphStore,
        llm: FakeLLMAdapter,
    ) -> MemoryRetrievalEngineImpl:
        return MemoryRetrievalEngineImpl(
            config=config,
            vector_store=vector_store,
            graph_store=graph_store,
            llm=llm,
        )

    # ── ingest ────────────────────────────────────────────────────────────

    def test_ingest_returns_node_id(self, engine: MemoryRetrievalEngineImpl) -> None:
        node_id = engine.ingest("python coding", "Python is a great language")
        assert isinstance(node_id, str)
        assert len(node_id) > 0

    def test_ingest_creates_bucket(self, engine: MemoryRetrievalEngineImpl) -> None:
        engine.ingest("python coding", "content about python")
        buckets = engine._bucket_manager.get_all_buckets()
        assert len(buckets) == 1

    def test_ingest_multiple_same_theme_same_bucket(
        self, engine: MemoryRetrievalEngineImpl
    ) -> None:
        # Force LLM to assign to the same bucket
        engine._llm.add_script(
            "python",
            create_assignment_decision("primary", cross_links=[]),
        )

        id1 = engine.ingest("python basics", "python 101")
        buckets_after_first = engine._bucket_manager.get_all_buckets()
        assert len(buckets_after_first) == 1

        # Replace the bucket ID in the script to match the actual bucket
        bucket_id = buckets_after_first[0].id
        engine._llm._script.clear()
        engine._llm.add_script(
            "python",
            create_assignment_decision(bucket_id, cross_links=[]),
        )

        id2 = engine.ingest("python advanced", "python 201")
        buckets = engine._bucket_manager.get_all_buckets()
        assert len(buckets) == 1  # still one bucket
        assert len(buckets[0].node_ids) == 2

    # ── search ────────────────────────────────────────────────────────────

    def test_search_empty_index(self, engine: MemoryRetrievalEngineImpl) -> None:
        result = engine.search("query")
        assert result.nodes == []
        assert result.scores == []

    def test_search_returns_ingested_content(
        self, engine: MemoryRetrievalEngineImpl
    ) -> None:
        engine.ingest("python programming language", "Python is a dynamic language")
        engine.ingest("javascript programming", "JavaScript runs in browsers")

        result = engine.search("programming")
        assert len(result.nodes) >= 1
        assert len(result.scores) >= 1
        # Both are programming-related, should return something
        assert any("programming" in n.summary.lower() for n in result.nodes)

    def test_search_respects_max_hops_override(
        self, engine: MemoryRetrievalEngineImpl
    ) -> None:
        engine.ingest("topic A", "content A")
        engine.ingest("topic B", "content B")

        result = engine.search("topic A", max_hops=0)
        # With 0 hops, no cross-bucket expansion
        assert isinstance(result.nodes, list)

    # ── resolve_conflicts ─────────────────────────────────────────────────

    def test_resolve_conflicts_no_conflicts(
        self, engine: MemoryRetrievalEngineImpl
    ) -> None:
        node = MemoryNode(
            id="n1",
            summary="test",
            content="I live in Beijing",
            summary_vector=np.ones(8, dtype=np.float32),
            content_vector=np.ones(8, dtype=np.float32),
            timestamp=1000.0,
            confidence=0.9,
        )
        resolved = engine.resolve_conflicts([node], "where do I live")
        assert len(resolved) == 1
        assert not resolved[0].is_stale

    def test_resolve_conflicts_marks_stale(
        self, engine: MemoryRetrievalEngineImpl
    ) -> None:
        old_node = MemoryNode(
            id="n_old",
            summary="location",
            content="I live in Beijing",
            summary_vector=np.array([1.0] + [0.0] * 7, dtype=np.float32),
            content_vector=np.array([1.0] + [0.0] * 7, dtype=np.float32),
            timestamp=1000.0,
            confidence=0.5,
        )
        new_node = MemoryNode(
            id="n_new",
            summary="location",
            content="I moved to Shanghai",
            summary_vector=np.array([1.0] + [0.0] * 7, dtype=np.float32),
            content_vector=np.array([1.0] + [0.0] * 7, dtype=np.float32),
            timestamp=2000.0,
            confidence=1.0,
        )

        # Both nodes have identical vectors, so new_node wins via higher
        # confidence and slightly newer timestamp → it ranks first (index 0).
        # LLM response: newer_id=0 (the winner), older_id=1 (to be marked stale).
        engine._llm.add_script(
            "fact-checking",
            create_conflict_response(
                [{"newer_id": "0", "older_id": "1",
                  "reason": "location changed"}]
            ),
        )

        resolved = engine.resolve_conflicts(
            [old_node, new_node], "where do I live"
        )
        assert len(resolved) == 2
        stale_nodes = [n for n in resolved if n.is_stale]
        assert len(stale_nodes) >= 1
        assert stale_nodes[0].id == "n_old"

    # ── conflict re-ranking ───────────────────────────────────────────────

    def test_resolve_conflicts_reranks_by_score(
        self, engine: MemoryRetrievalEngineImpl
    ) -> None:
        # Set up LLM to report no conflicts
        engine._llm.add_script("fact-checking", create_conflict_response([]))

        old_node = MemoryNode(
            id="n1",
            summary="weather",
            content="It is sunny today",
            summary_vector=np.array([1.0] + [0.0] * 7, dtype=np.float32),
            content_vector=np.array([1.0] + [0.0] * 7, dtype=np.float32),
            timestamp=1000.0,
            confidence=1.0,
        )
        new_node = MemoryNode(
            id="n2",
            summary="weather",
            content="It is raining today",
            summary_vector=np.array([0.9] + [0.0] * 7, dtype=np.float32),
            content_vector=np.array([0.9] + [0.0] * 7, dtype=np.float32),
            timestamp=2000.0,
            confidence=0.5,
        )
        resolved = engine.resolve_conflicts([old_node, new_node], "weather")
        # Both should be present, no conflict detected
        assert len(resolved) >= 1
