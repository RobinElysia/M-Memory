"""Smoke tests for production modules — verify import and basic functionality."""
import os

import pytest


class TestDeepSeekLLM:
    def test_import(self):
        from memory_system.deepseek_llm import DeepSeekAdapter
        assert DeepSeekAdapter is not None

    def test_requires_api_key(self):
        from memory_system.deepseek_llm import DeepSeekAdapter
        old = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
                DeepSeekAdapter()
        finally:
            if old:
                os.environ["DEEPSEEK_API_KEY"] = old


class TestLocalEmbedding:
    def test_import(self):
        from memory_system.local_embedding import LocalEmbeddingStore
        assert LocalEmbeddingStore is not None

    def test_is_semantic(self):
        from memory_system.local_embedding import LocalEmbeddingStore
        store = LocalEmbeddingStore()
        assert store.is_semantic() is True


class TestOpenAIEmbedding:
    def test_import(self):
        from memory_system.embedding_store import OpenAIEmbeddingStore
        assert OpenAIEmbeddingStore is not None

    def test_requires_api_key(self):
        from memory_system.embedding_store import OpenAIEmbeddingStore
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIEmbeddingStore()
        finally:
            if old:
                os.environ["OPENAI_API_KEY"] = old


class TestPersistence:
    def test_create_in_memory(self):
        from memory_system.persistence import PersistenceStore
        store = PersistenceStore(":memory:")
        nodes = store.load_all_nodes()
        assert nodes == []
        store.close()

    def test_persistence_round_trip(self):
        """Verify engine restores state from persistence after simulated restart."""
        import os
        from memory_system.persistence import PersistenceStore
        from memory_system.config import MemorySystemConfig
        from memory_system.graph_engine import NetworkXGraphStore
        from memory_system.fake_llm import FakeLLMAdapter
        from memory_system.vector_store import HashVectorStore
        from memory_system.retrieval import MemoryRetrievalEngineImpl

        db_path = "test_roundtrip.db"

        # ── First session: ingest ──
        config = MemorySystemConfig()
        config.embedding_dim = 1536
        store = PersistenceStore(db_path)
        engine1 = MemoryRetrievalEngineImpl(
            config=config, vector_store=HashVectorStore(dim=1536),
            graph_store=NetworkXGraphStore(), llm=FakeLLMAdapter(),
            persistence=store,
        )
        nid = engine1.ingest("hello", "Hello world", confidence=0.9)

        store.close()

        # ── Simulate restart: new engine, same db ──
        store2 = PersistenceStore(db_path)
        engine2 = MemoryRetrievalEngineImpl(
            config=config, vector_store=HashVectorStore(dim=1536),
            graph_store=NetworkXGraphStore(), llm=FakeLLMAdapter(),
            persistence=store2,
        )
        result = engine2.search("hello")
        found = any(nid == n.id or "Hello" in n.content for n in result.nodes)
        assert found, "Restored engine should find ingested node"

        store2.close()
        try:
            os.remove(db_path)
        except PermissionError:
            pass  # Windows file lock delay

    def test_save_and_load_node(self):
        import time

        from memory_system.models import MemoryNode
        from memory_system.persistence import PersistenceStore
        store = PersistenceStore(":memory:")
        node = MemoryNode(
            id="test-1", summary="test", content="hello world",
            summary_vector=None, content_vector=None,
            timestamp=time.time(), confidence=0.9,
        )
        store.save_node(node)
        nodes = store.load_all_nodes()
        assert len(nodes) == 1
        assert nodes[0]["id"] == "test-1"
        assert nodes[0]["content"] == "hello world"
        store.close()
