# GraphStore

Abstraction over graph storage and traversal.

Default implementation: `NetworkXGraphStore` (directed multi-graph via
`nx.MultiDiGraph`, supports parallel edges between the same two nodes).

## Interface

```python
from memory_system.interfaces import GraphStore, EdgeType

class GraphStore(ABC):
    @abstractmethod
    def add_node(self, node_id: str, attributes: dict[str, Any]) -> None: ...
    @abstractmethod
    def add_edge(self, from_id: str, to_id: str, edge_type: EdgeType, weight: float) -> str: ...
    @abstractmethod
    def traverse(self, start_nodes: list[str], max_hops: int, weight_threshold: float) -> list[TraversalPath]: ...
    @abstractmethod
    def get_out_edges(self, node_id: str, edge_type: EdgeType | None = None) -> list[Edge]: ...
    @abstractmethod
    def remove_edge(self, edge_id: str) -> None: ...
    @abstractmethod
    def get_node_count(self) -> int: ...
    @abstractmethod
    def get_edge_count(self, edge_type: EdgeType | None = None) -> int: ...
```

## Edge Types

| Type | Enum | Purpose |
|------|------|---------|
| Temporal | `EdgeType.TEMPORAL` | Chronological edge (dialogue order) |
| Intra-bucket | `EdgeType.INTRA_BUCKET` | Node → its bucket's Medoid |
| Cross-bucket | `EdgeType.CROSS_BUCKET` | Soft association across buckets |

## Default: NetworkXGraphStore

```python
from memory_system.graph_engine import NetworkXGraphStore
from memory_system.models import EdgeType

graph = NetworkXGraphStore()
```

| Method | Description |
|--------|-------------|
| `add_node(id, attrs)` | Register a vertex; raises if duplicate |
| `add_edge(from, to, type, weight)` | Directed edge; returns unique edge ID |
| `traverse(starts, hops, threshold)` | BFS following only `CROSS_BUCKET` edges ≥ threshold; best-single-path dedup |
| `get_out_edges(id, type?)` | Outgoing edges, optionally filtered |
| `remove_edge(id)` | Delete edge by ID |
| `get_node_count()` | Total vertices |
| `get_edge_count(type?)` | Total edges, optionally filtered by `EdgeType` |

## Traversal Behaviour

- Only `CROSS_BUCKET` edges are followed (temporal and intra-bucket edges are skipped)
- Edge weight must be ≥ *weight_threshold*
- **Best-single-path dedup**: when a node is reachable via multiple routes,
  only the path with the highest cumulative weight (product of edge weights) is kept
- Maximum depth capped by *max_hops*

```python
# Example: traverse from seed nodes with 2 hops, weight ≥ 0.5
paths = graph.traverse(["node-a", "node-b"], max_hops=2, weight_threshold=0.5)
for p in paths:
    print(f"Path: {' → '.join(p.node_ids)}  weight={p.total_weight:.3f}  hops={p.hops}")
```
