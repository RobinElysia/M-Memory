"""System-wide configuration parameters for the dual-layer memory system.

All tunable parameters extracted from ARCHITECTURE_DESIGN.md are defined here
with their types, default values, and rationale. No business logic lives in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BucketConfig:
    """Parameters governing bucket management and assignment."""

    # ── Candidate selection ────────────────────────────────────────────
    top_k: int = 3
    """Number of top candidate buckets considered during new-node assignment.
    The system computes vector similarity between the new node's summary A and
    all bucket Medoid vectors, then passes the top_k candidates (along with
    their context) to the LLM for final assignment decision.
    Per ARCHITECTURE_DESIGN.md: k=2 or 3.
    """

    # ── Semantic search ─────────────────────────────────────────────────
    top_m: int = 5
    """Number of buckets retrieved during first-layer semantic search.
    The query vector is compared against all bucket Medoid vectors to find
    the top_m most semantically relevant buckets.
    """

    top_p: int = 10
    """Number of A-nodes retrieved within each selected bucket during
    second-stage fine-grained search.
    """

    # ── Bucket lifecycle ────────────────────────────────────────────────
    split_threshold: int = 50
    """Minimum number of nodes in a bucket before split eligibility is checked.
    When exceeded, the system examines whether distinct sub-clusters have formed
    and triggers a split if so.
    """

    dormancy_interval_seconds: float = 3600.0
    """Time window (in seconds) without writes or queries after which a bucket
    is considered dormant. Dormant buckets have their Medoid vectors moved out
    of the active index to reduce maintenance overhead.
    """

    cold_storage_similarity_threshold: float = 0.3
    """Cosine similarity threshold below which a node's content C is considered
    to have drifted too far from its bucket's Medoid. Such nodes may be moved to
    cold storage or deleted during periodic cleanup.
    """


@dataclass
class GraphConfig:
    """Parameters governing the graph structure (edges, traversal)."""

    # ── Edge limits ─────────────────────────────────────────────────────
    max_out_degree: int = 5
    """Maximum number of cross-bucket edges a single node can have.
    When this limit is reached and a new higher-weight edge is proposed,
    the weakest existing edge is evicted (last-place elimination).
    Per ARCHITECTURE_DESIGN.md: prevents exponential edge growth.
    """

    weight_decay_threshold: float = 0.2
    """Minimum weight a cross-bucket edge must retain to stay in the graph.
    Edges below this threshold are pruned during periodic cleanup.
    """

    # ── Traversal ───────────────────────────────────────────────────────
    max_hops: int = 2
    """Maximum number of hops during graph walk for associative expansion.
    Starting from seed nodes, the traversal follows cross-bucket edges up to
    this many steps. Per ARCHITECTURE_DESIGN.md: controls '跨桶程度'.
    """

    edge_weight_threshold: float = 0.5
    """Minimum edge weight for traversal eligibility. Edges with weight below
    this threshold are ignored during graph walks.
    """


@dataclass
class ConflictConfig:
    """Parameters governing the conflict resolution pipeline (Layer 2)."""

    # ── Re-ranking weights ──────────────────────────────────────────────
    alpha: float = 0.5
    """Weight for semantic similarity score in the re-ranking formula:
    score = α·sim(C, query) + β/(1+ΔT) + γ·confidence.
    """

    beta: float = 0.3
    """Weight for temporal recency in the re-ranking formula.
    Higher β favours newer information.
    """

    gamma: float = 0.2
    """Weight for source confidence in the re-ranking formula.
    Higher γ favours information from higher-confidence sources.
    """

    top_n: int = 5
    """Maximum number of candidates retained after re-ranking and before
    the final conflict check by the LLM. Controls context window size.
    """

    # ── Staleness ───────────────────────────────────────────────────────
    stale_mark_downgrade_factor: float = 0.1
    """Factor by which a node's weight is multiplied when marked as 'stale'
    (outdated). This downgrades the node so it appears lower in results but
    remains retrievable for deep backtracking.
    """


@dataclass
class CleanupConfig:
    """Parameters governing periodic background maintenance tasks."""

    interval_seconds: float = 300.0
    """Interval (in seconds) between cleanup cycles. The background task wakes
    up every interval_seconds to scan for stale nodes, dormant buckets, and
    split candidates.
    """

    node_similarity_for_contradiction: float = 0.85
    """Cosine similarity threshold for detecting semantically similar nodes
    that may contain contradictory information. When two nodes exceed this
    threshold but contain conflicting key facts, the newer higher-confidence
    node is kept and the older one is marked stale.
    """

    medoid_similarity_floor: float = 0.3
    """If a node's similarity to its bucket Medoid falls below this value,
    the node is considered topic-drifted and is a candidate for cold storage
    or deletion.
    """


@dataclass
class MemorySystemConfig:
    """Top-level configuration aggregating all subsystem parameters."""

    bucket: BucketConfig = field(default_factory=BucketConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    conflict: ConflictConfig = field(default_factory=ConflictConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)

    # ── Embedding ───────────────────────────────────────────────────────
    embedding_dim: int = 1536
    """Dimensionality of embedding vectors. Default matches OpenAI ada-002.
    Must be consistent across the VectorStore implementation.
    """

    # ── Logging ─────────────────────────────────────────────────────────
    log_level: str = "INFO"
    """Structured logging level. LLM calls, bucket assignments, and cleanup
    operations are logged at this level.
    """
