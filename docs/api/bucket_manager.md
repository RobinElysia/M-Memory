# BucketManager

Bucket lifecycle management and node-to-bucket assignment.

## Setup

```python
from memory_system.bucket_manager import BucketManagerImpl
from memory_system.config import MemorySystemConfig

manager = BucketManagerImpl(
    config=MemorySystemConfig(),
    vector_store=NumpyVectorStore(dim=1536),
    graph_store=NetworkXGraphStore(),
    llm=FakeLLMAdapter(),
)
```

## Methods

### `create_bucket(medoid_node: MemoryNode) -> Bucket`

Create a new bucket with *medoid_node* as its initial Medoid.
The node becomes the bucket's first member and its summary vector
is registered in the active index for semantic search.

### `find_candidates(node_a: MemoryNode) -> list[tuple[Bucket, float]]`

Compute top-k candidate buckets for *node_a* via cosine similarity
between the node's summary vector and each active bucket's Medoid vector.
Returns an empty list when no active buckets exist (first node in system).

### `assign_to_bucket(node_a: MemoryNode, bucket: Bucket, cross_links: list[dict]) -> None`

Physically place *node_a* into *bucket* and record cross-bucket edges:
- Sets `node_a.bucket_id = bucket.id`
- Creates an intra-bucket edge (node → Medoid)
- Creates cross-bucket edges per *cross_links* with out-degree eviction
- Recomputes bucket Medoid after insertion

### `split_bucket(bucket: Bucket) -> list[Bucket]`

Split an overgrown bucket into sub-buckets based on detected sub-clusters.
(Currently a placeholder — returns `[bucket]` unchanged.)

### `dormancy_check() -> list[Bucket]`

Identify buckets with no write or query activity within
`config.bucket.dormancy_interval_seconds`.  Marks them dormant and
removes their Medoid vectors from the active index.

### `wake_bucket(bucket_id: str) -> Bucket`

Reactivate a dormant bucket, restoring its Medoid vector to the
active index so it participates in future searches.

### `get_active_buckets() -> list[Bucket]`

Return all non-dormant buckets.

### `get_all_buckets() -> list[Bucket]`

Return every bucket including dormant ones.
