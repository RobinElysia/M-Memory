# MemoryRetrievalEngine

Top-level API for the dual-layer memory system.

## Setup

```python
from memory_system.config import MemorySystemConfig
from memory_system.vector_store import NumpyVectorStore
from memory_system.graph_engine import NetworkXGraphStore
from memory_system.fake_llm import FakeLLMAdapter
from memory_system.retrieval import MemoryRetrievalEngineImpl

config = MemorySystemConfig()
engine = MemoryRetrievalEngineImpl(
    config=config,
    vector_store=NumpyVectorStore(dim=1536),
    graph_store=NetworkXGraphStore(),
    llm=FakeLLMAdapter(),
)
```

## Methods

### `ingest(summary: str, content: str, confidence: float = 1.0) -> str`

Full ingestion pipeline:
1. Embed summary (A) and content (C) via `VectorStore.embed()`
2. Find candidate buckets via `BucketManager.find_candidates()`
3. Ask LLM to decide bucket assignment
4. Physically place node in primary bucket (`assign_to_bucket`)
5. Create cross-bucket edges if LLM recommends
6. Update bucket Medoid

Returns the new node's ID string.

```python
node_id = engine.ingest("my cat", "My cat loves sleeping in the sun")
```

### `search(query: str, max_hops: int | None = None, weight_threshold: float | None = None) -> SearchResult`

Two-layer retrieval pipeline:

**Layer 1** — coarse screening + graph expansion:
- Embed query → score all active-bucket Medoid vectors → keep top *m*
- Within each selected bucket, fine-search for top *p* A-nodes
- Graph associative expansion via cross-bucket edge traversal (BFS)

**Layer 2** — conflict resolution:
- Weighted re-rank: `score = α·sim(C,query) + β/(1+ΔT) + γ·confidence`
- LLM contradiction detection
- Mark older contradictory nodes as stale (downgraded, not deleted)

Returns `SearchResult(nodes=list[MemoryNode], scores=list[float])`.

```python
result = engine.search("tell me about my cat")
for node, score in zip(result.nodes, result.scores):
    print(f"[{score:.3f}] {node.summary}: {node.content}")
```

### `resolve_conflicts(candidates: list[MemoryNode], query: str) -> list[MemoryNode]`

Standalone conflict resolution on an arbitrary set of nodes.  Runs the full
re-rank → LLM check → stale-marking pipeline without bucket search.

```python
resolved = engine.resolve_conflicts([old_node, new_node], "where do I live")
```
