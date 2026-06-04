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
