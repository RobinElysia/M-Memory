# API Reference

## MemoryRetrievalEngine

The top-level API. Create one instance, then ingest and search.

```python
from memory_system.retrieval import MemoryRetrievalEngineImpl

engine = MemoryRetrievalEngineImpl(
    config=MemorySystemConfig(),
    vector_store=NumpyVectorStore(dim=1536),
    graph_store=NetworkXGraphStore(),
    llm=FakeLLMAdapter(),
)
```

### `ingest(summary: str, content: str, confidence: float = 1.0) -> str`

Full ingestion pipeline:
1. Embed summary (A) and content (C)
2. Find candidate buckets via Medoid similarity
3. Ask LLM to decide bucket assignment
4. Physically place node in primary bucket
5. Create cross-bucket edges if LLM recommends
6. Update bucket Medoid

Returns the new node's ID string.

### `search(query: str, max_hops: int | None = None, weight_threshold: float | None = None) -> SearchResult`

Two-layer retrieval:
- **Layer 1**: Bucket coarse screen → in-bucket fine search → graph expansion
- **Layer 2**: Weighted re-rank → LLM conflict detection → stale marking

Returns `SearchResult(nodes=list[MemoryNode], scores=list[float])`.

### `resolve_conflicts(candidates: list[MemoryNode], query: str) -> list[MemoryNode]`

Standalone conflict resolution on an arbitrary set of nodes.

## BucketManager

```python
from memory_system.bucket_manager import BucketManagerImpl
```

### Key Methods

| Method | Description |
|--------|-------------|
| `create_bucket(node)` | New bucket with node as initial Medoid |
| `find_candidates(node)` | Top-k candidate buckets by Medoid similarity |
| `assign_to_bucket(node, bucket, cross_links)` | Place node + cross-bucket edges |
| `dormancy_check()` | Mark inactive buckets dormant |
| `wake_bucket(id)` | Reactivate dormant bucket |
| `split_bucket(bucket)` | Split overgrown bucket (placeholder) |

## VectorStore

```python
from memory_system.vector_store import NumpyVectorStore
from memory_system.interfaces import VectorStore
```

Implement `VectorStore` to swap backends (FAISS, Chroma, USearch, etc.).

### Required Methods

- `embed(text) -> NDArray` — text → dense vector
- `add(vectors, metadata) -> list[str]` — batch insert
- `search(query_vector, top_k) -> list[tuple[str, float]]` — similarity search
- `remove(ids)` — delete by ID
- `count() -> int` — total vectors

## GraphStore

```python
from memory_system.graph_engine import NetworkXGraphStore
from memory_system.interfaces import GraphStore
```

### Required Methods

- `add_node(id, attrs)` — register vertex
- `add_edge(from, to, type, weight) -> str` — create directed edge
- `traverse(starts, max_hops, threshold) -> list[TraversalPath]` — BFS walk
- `get_out_edges(id, type?)` — outgoing edges
- `remove_edge(id)` — delete edge

## Configuration

```python
from memory_system.config import MemorySystemConfig

config = MemorySystemConfig()
config.bucket.top_k = 3       # candidate buckets for assignment
config.bucket.top_m = 5       # buckets retrieved in search
config.graph.max_hops = 2     # graph walk depth
config.conflict.alpha = 0.5   # semantic similarity weight
config.conflict.beta = 0.3    # recency weight
config.conflict.gamma = 0.2   # confidence weight
```

See `memory_system/config.py` for all parameters and defaults.

## Data Models

```python
from memory_system.models import MemoryNode, Bucket, Medoid, Edge, SearchResult
```

See `memory_system/models.py` for field-level documentation.
