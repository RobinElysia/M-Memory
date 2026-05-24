#!/usr/bin/env python3
"""Basic usage example for the m-memory dual-layer memory system.

Run with:
    python examples/basic_usage.py

No external API keys required — uses a fake (deterministic) embedding
backend and a scripted LLM adapter.
"""

from memory_system.config import MemorySystemConfig
from memory_system.fake_llm import (
    FakeLLMAdapter,
    create_assignment_decision,
    create_conflict_response,
)
from memory_system.graph_engine import NetworkXGraphStore
from memory_system.retrieval import MemoryRetrievalEngineImpl
from memory_system.vector_store import NumpyVectorStore


def main() -> None:
    # ── Setup ──────────────────────────────────────────────────────────
    config = MemorySystemConfig()
    config.embedding_dim = 8  # Small dimension for demo

    llm = FakeLLMAdapter()
    engine = MemoryRetrievalEngineImpl(
        config=config,
        vector_store=NumpyVectorStore(dim=config.embedding_dim),
        graph_store=NetworkXGraphStore(),
        llm=llm,
    )

    # ── Ingest memories ────────────────────────────────────────────────
    print("=== Ingesting memories ===\n")

    # First memory: creates a new bucket
    id1 = engine.ingest("my cat habits", "My cat loves sleeping in the sun")
    print(f"[ingest] id={id1}  summary='my cat habits'")

    # Set up LLM to assign subsequent cat-related nodes to the same bucket
    buckets = engine._bucket_manager.get_all_buckets()
    cat_bucket_id = buckets[0].id
    llm.add_script(
        "cat",
        create_assignment_decision(cat_bucket_id, cross_links=[]),
    )

    id2 = engine.ingest("cat food preference", "I feed my cat premium dry food")
    print(f"[ingest] id={id2}  summary='cat food preference'")

    # Create a separate topic (new bucket)
    llm.add_script("dog", create_assignment_decision("new", cross_links=[]))
    id3 = engine.ingest("my dog activities", "My dog enjoys running at the park")
    print(f"[ingest] id={id3}  summary='my dog activities'")

    # Cross-topic memory
    llm.add_script(
        "pets",
        create_assignment_decision(
            cat_bucket_id,
            cross_links=[{"bucket_id": "dog_bucket", "weight": 0.7}],
        ),
    )
    id4 = engine.ingest(
        "playing with my pets", "I play with both my cat and dog every evening"
    )
    print(f"[ingest] id={id4}  summary='playing with my pets'")

    # ── Search ─────────────────────────────────────────────────────────
    print("\n=== Searching ===\n")

    # Set up LLM to report no conflicts for search
    llm.add_script("fact-checking", create_conflict_response([]))

    result = engine.search("tell me about my cat")
    print("Query: 'tell me about my cat'")
    for node, score in zip(result.nodes, result.scores):
        print(f"  [{score:.4f}] {node.summary}: {node.content}")

    # ── Conflict resolution demo ───────────────────────────────────────
    print("\n=== Conflict Resolution Demo ===\n")

    import numpy as np
    from memory_system.models import MemoryNode

    vector = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    old_node = MemoryNode(
        id="old",
        summary="my city",
        content="I live in Beijing",
        summary_vector=vector.copy(),
        content_vector=vector.copy(),
        timestamp=1000.0,
        confidence=0.8,
    )
    new_node = MemoryNode(
        id="new",
        summary="my city update",
        content="I moved to Shanghai",
        summary_vector=vector.copy() * 0.99,
        content_vector=vector.copy() * 0.99,
        timestamp=2000.0,
        confidence=0.9,
    )

    llm.add_script(
        "fact-checking",
        create_conflict_response(
            [{"newer_id": "0", "older_id": "1", "reason": "location changed"}]
        ),
    )

    resolved = engine.resolve_conflicts([old_node, new_node], "where do I live")
    print("After conflict resolution:")
    for node in resolved:
        stale_tag = " [STALE]" if node.is_stale else ""
        print(f"  {node.content} (confidence={node.confidence:.2f}){stale_tag}")

    # ── Statistics ─────────────────────────────────────────────────────
    print("\n=== System State ===\n")
    all_buckets = engine._bucket_manager.get_all_buckets()
    print(f"Buckets: {len(all_buckets)}")
    for b in all_buckets:
        print(f"  {b.id[:8]}... nodes={len(b.node_ids)} dormant={b.is_dormant}")
    print(f"Graph nodes: {engine._graph_store.get_node_count()}")
    print(f"Graph edges: {engine._graph_store.get_edge_count()}")


if __name__ == "__main__":
    main()
