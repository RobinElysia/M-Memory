"""Domain data models for the dual-layer memory system.

Every model is a ``@dataclass`` (or Pydantic model when validation complexity
warrants it).  These are *plain data* objects — no business logic, no I/O,
no references to concrete backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from memory_system.interfaces import EdgeType  # canonical definition

# ═══════════════════════════════════════════════════════════════════════════════
# Core entities
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MemoryNode:
    """A single turn of conversation stored in the memory system.

    Attributes:
        id: Unique identifier (UUID string).
        summary: Summary paragraph / keywords (A) used for coarse screening.
        content: Full detailed content (C) used for precise retrieval.
        summary_vector: Dense embedding of *summary* (shape ``(dim,)``).
        content_vector: Dense embedding of *content* (shape ``(dim,)``).
        timestamp: Unix timestamp of creation time.
        confidence: Source confidence in ``[0.0, 1.0]``.
        bucket_id: Primary bucket this node physically resides in.
        is_stale: Whether this node has been marked outdated by the conflict-
            resolution pipeline.  Stale nodes are retained but downgraded.
    """

    id: str
    summary: str
    content: str
    summary_vector: NDArray[np.float32]
    content_vector: NDArray[np.float32]
    timestamp: float
    confidence: float = 1.0
    bucket_id: str = ""
    is_stale: bool = False

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryNode):
            return NotImplemented
        return self.id == other.id


@dataclass
class Medoid:
    """Representative node of a bucket.

    The Medoid is the actual node whose average cosine distance to all other
    nodes in the bucket is minimal.  When the bucket has exactly one node,
    that node *is* the Medoid.

    Attributes:
        node_id: The actual node serving as Medoid.
        summary: Cached copy of that node's summary text (A).
        vector: Cached copy of that node's summary vector.
        version: Monotonic version counter, incremented on every recomputation.
    """

    node_id: str
    summary: str
    vector: NDArray[np.float32]
    version: int = 0


@dataclass
class Bucket:
    """A dynamic cluster of memory nodes sharing a semantic theme.

    Attributes:
        id: Unique bucket identifier.
        medoid: Current Medoid (``None`` only briefly during initialisation
            before the first node is added).
        node_ids: Ordered list of node IDs belonging to this bucket.
        created_at: Unix timestamp of bucket creation.
        last_write_at: Unix timestamp of most recent write.
        last_query_at: Unix timestamp of most recent query hit.
        is_dormant: Whether this bucket has been put to sleep due to inactivity.
        version: Monotonic counter, incremented when Medoid changes.
    """

    id: str
    medoid: Medoid | None = None
    node_ids: list[str] = field(default_factory=list)
    created_at: float = 0.0
    last_write_at: float = 0.0
    last_query_at: float = 0.0
    is_dormant: bool = False
    version: int = 0


@dataclass
class Edge:
    """A directed edge in the memory graph.

    Attributes:
        id: Unique edge identifier.
        source_id: Source vertex (node) ID.
        target_id: Target vertex (node) ID.
        edge_type: Semantic category (temporal / intra-bucket / cross-bucket).
        weight: Edge weight in ``[0.0, 1.0]``.
        created_at: Unix timestamp of edge creation.
    """

    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float
    created_at: float


@dataclass
class TraversalPath:
    """Result of a single path discovered during graph traversal.

    Attributes:
        node_ids: Ordered list of node IDs along this path.
        total_weight: Cumulative confidence (product of edge weights).
        hops: Number of edges traversed.
    """

    node_ids: list[str]
    total_weight: float
    hops: int


@dataclass
class SearchResult:
    """Immutable result returned by ``MemoryRetrievalEngine.search``.

    Attributes:
        nodes: Ordered list of retrieved nodes, highest relevance first.
        scores: Per-node final relevance scores (1:1 with *nodes*).
    """

    nodes: list[MemoryNode]
    scores: list[float]

    def __post_init__(self) -> None:
        if len(self.nodes) != len(self.scores):
            raise ValueError(
                f"nodes and scores lengths must match: "
                f"{len(self.nodes)} vs {len(self.scores)}"
            )
