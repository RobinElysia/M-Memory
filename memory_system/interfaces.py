"""Abstract interfaces (contracts) for the dual-layer memory system.

Every component is defined as an abstract base class so that implementations
are replaceable and testable. No concrete implementation details — not even
an ``import faiss`` — belong in this file.

All public methods carry full Google-style docstrings describing parameters,
return values, exceptions raised, and side effects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from memory_system.models import (
        Bucket,
        Edge,
        MemoryNode,
        SearchResult,
        TraversalPath,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════


class EdgeType(Enum):
    """Kinds of edges in the memory graph."""

    TEMPORAL = "temporal"
    """Chronological edge connecting successive dialogue turns."""

    INTRA_BUCKET = "intra_bucket"
    """Edge connecting a node to its primary bucket's Medoid."""

    CROSS_BUCKET = "cross_bucket"
    """Soft cross-bucket association edge — no physical node duplication."""


# ═══════════════════════════════════════════════════════════════════════════════
# Core interfaces
# ═══════════════════════════════════════════════════════════════════════════════


class VectorStore(ABC):
    """Abstraction over vector embedding and similarity search backends.

    Implementations may wrap FAISS, Chroma, USearch, NumPy, or a mock/fake
    backend for testing.  Callers must not depend on the concrete backend.
    """

    @abstractmethod
    def embed(self, text: str) -> NDArray[np.float32]:
        """Produce a dense embedding vector for *text*.

        Args:
            text: Arbitrary natural-language input.

        Returns:
            1-D ``np.ndarray`` of shape ``(dim,)`` and dtype ``float32``.

        Raises:
            RuntimeError: If the embedding backend is unavailable or the text
                exceeds the model's context window.
        """
        ...

    @abstractmethod
    def add(self, vectors: NDArray[np.float32], metadata: list[dict[str, Any]]) -> list[str]:
        """Insert *vectors* with associated *metadata* into the index.

        Args:
            vectors: 2-D array of shape ``(n, dim)``.
            metadata: List of *n* metadata dicts.  Each dict **must** contain
                at least the key ``"id"`` with a unique string value.

        Returns:
            List of stored vector IDs (same order as input).

        Raises:
            ValueError: If ``vectors.shape[0] != len(metadata)`` or any
                metadata dict is missing the ``"id"`` key.
        """
        ...

    @abstractmethod
    def search(
        self, query_vector: NDArray[np.float32], top_k: int
    ) -> list[tuple[str, float]]:
        """Return the *top_k* vectors most similar to *query_vector*.

        Args:
            query_vector: 1-D query embedding of shape ``(dim,)``.
            top_k: Maximum number of results to return.

        Returns:
            Ordered list of ``(id, similarity_score)`` tuples, highest
            similarity first.  Similarity is the raw backend score (e.g.
            cosine similarity or inner product depending on the index).

        Raises:
            ValueError: If *top_k* ≤ 0.
        """
        ...

    @abstractmethod
    def remove(self, ids: list[str]) -> None:
        """Remove vectors by their stored IDs.

        Args:
            ids: Vector IDs previously returned by :meth:`add`.

        Raises:
            KeyError: If any *id* is not present in the index.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of vectors currently in the index.

        Returns:
            Non-negative integer count.
        """
        ...

    @abstractmethod
    def is_semantic(self) -> bool:
        """Return whether embeddings are semantically meaningful.

        Hash/deterministic implementations MUST return ``False``.
        Model-based implementations (OpenAI, Cohere, local models) MUST
        return ``True``.  Callers may use this to choose between
        vector-based and lexical retrieval paths.
        """
        ...


class LLMAdapter(ABC):
    """Abstraction over a large-language-model backend.

    Implementations may wrap OpenAI, local models, or deterministic fake
    adapters used in testing.  All calls must be logged (prompt length,
    token consumption estimate, decision summary) — see the concrete
    implementation for logging details; this interface only mandates the
    behavioural contract.
    """

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Send a single-shot completion request.

        Args:
            prompt: The full prompt text (system + user combined).
            **kwargs: Backend-specific overrides (e.g. ``temperature``,
                ``max_tokens``).

        Returns:
            The model's text response.

        Raises:
            RuntimeError: If the backend is unreachable or returns an error.
        """
        ...

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Send a multi-turn chat-completion request.

        Args:
            messages: List of message dicts, each with ``"role"`` and
                ``"content"`` keys (OpenAI-compatible format).
            **kwargs: Backend-specific overrides.

        Returns:
            The assistant's text response.

        Raises:
            RuntimeError: If the backend is unreachable or returns an error.
        """
        ...


class GraphStore(ABC):
    """Abstraction over graph storage and traversal.

    The graph stores :class:`MemoryNode` references as vertices and
    :class:`Edge` instances as directed edges.  Implementations may use
    NetworkX, a custom adjacency list, or an external graph database.
    """

    @abstractmethod
    def add_node(self, node_id: str, attributes: dict[str, Any]) -> None:
        """Register a vertex in the graph.

        Args:
            node_id: Unique identifier for the node.
            attributes: Arbitrary key-value metadata attached to the vertex.

        Raises:
            ValueError: If *node_id* already exists.
        """
        ...

    @abstractmethod
    def add_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: EdgeType,
        weight: float,
    ) -> str:
        """Create a directed edge between two existing vertices.

        Args:
            from_id: Source vertex ID.
            to_id: Target vertex ID.
            edge_type: Semantic category for the edge.
            weight: Edge weight in ``[0.0, 1.0]``.

        Returns:
            Unique edge ID assigned by the store.

        Raises:
            ValueError: If either vertex does not exist.
        """
        ...

    @abstractmethod
    def traverse(
        self,
        start_nodes: list[str],
        max_hops: int,
        weight_threshold: float,
    ) -> list[TraversalPath]:
        """Walk the graph from *start_nodes* along eligible edges.

        Only :attr:`EdgeType.CROSS_BUCKET` edges whose weight ≥
        *weight_threshold* are followed.  Multi-path de-duplication uses a
        *best-single-path* strategy: when a node is reachable via more than
        one route, only the path with the highest cumulative weight is kept.

        Args:
            start_nodes: Seed vertex IDs.
            max_hops: Maximum number of edges to traverse from each seed.
            weight_threshold: Minimum edge weight for traversal eligibility.

        Returns:
            List of :class:`TraversalPath` instances representing all
            discovered routes.  Paths are not guaranteed to be sorted.
        """
        ...

    @abstractmethod
    def get_out_edges(
        self, node_id: str, edge_type: EdgeType | None = None
    ) -> list[Edge]:
        """Return outgoing edges from *node_id*, optionally filtered by type.

        Args:
            node_id: Source vertex ID.
            edge_type: If given, return only edges of this type.

        Returns:
            List of matching :class:`Edge` objects (may be empty).

        Raises:
            ValueError: If *node_id* does not exist.
        """
        ...

    @abstractmethod
    def remove_edge(self, edge_id: str) -> None:
        """Delete an edge by its ID.

        Args:
            edge_id: Edge ID previously returned by :meth:`add_edge`.

        Raises:
            KeyError: If *edge_id* is not present.
        """
        ...

    @abstractmethod
    def get_node_count(self) -> int:
        """Return the total number of vertices in the graph."""
        ...

    @abstractmethod
    def get_edge_count(self, edge_type: EdgeType | None = None) -> int:
        """Return the number of edges, optionally filtered by type."""
        ...


class BucketManager(ABC):
    """Abstraction over bucket lifecycle and node assignment.

    Responsible for cluster maintenance (Medoid updates, splits, dormancy)
    but **not** for the LLM decision itself — that is delegated to an
    :class:`LLMAdapter` injected at construction (see concrete impl).
    """

    @abstractmethod
    def find_candidates(
        self, node_a: MemoryNode
    ) -> list[tuple[Bucket, float]]:
        """Compute top-k candidate buckets for *node_a* via vector similarity.

        Args:
            node_a: The new memory node awaiting assignment.

        Returns:
            Ordered list of ``(Bucket, similarity_score)`` tuples, highest
            similarity first.  Length is at most ``config.bucket.top_k``.
            Returns an empty list when no active buckets exist (first node).
        """
        ...

    @abstractmethod
    def assign_to_bucket(
        self,
        node_a: MemoryNode,
        bucket: Bucket,
        cross_links: list[dict[str, Any]],
    ) -> None:
        """Physically place *node_a* into *bucket* and record cross-bucket edges.

        **Side effect**: updates *bucket*'s Medoid after insertion.

        Args:
            node_a: The memory node to place.
            bucket: The primary (physical) bucket.
            cross_links: List of ``{"bucket_id": str, "weight": float}`` dicts
                describing cross-bucket associations decided by the LLM.
                Each entry results in a :attr:`EdgeType.CROSS_BUCKET` edge
                from *node_a* to the target bucket's Medoid node.
        """
        ...

    @abstractmethod
    def create_bucket(self, medoid_node: MemoryNode) -> Bucket:
        """Create a new bucket whose initial Medoid is *medoid_node*.

        Args:
            medoid_node: The first (and currently only) node in the bucket.

        Returns:
            The newly created :class:`Bucket`.
        """
        ...

    @abstractmethod
    def split_bucket(self, bucket: Bucket) -> list[Bucket]:
        """Split a bucket into two or more sub-buckets based on sub-clusters.

        Args:
            bucket: The bucket to split (must have ≥ ``split_threshold`` nodes).

        Returns:
            List of replacement buckets (length ≥ 2).  The original *bucket*
            is **not** retained — callers must replace it with the returned list.
        """
        ...

    @abstractmethod
    def dormancy_check(self) -> list[Bucket]:
        """Identify and mark dormant buckets.

        A bucket is dormant when neither write nor query activity has
        occurred within ``config.bucket.dormancy_interval_seconds``.

        Returns:
            List of buckets that were **newly** marked dormant by this call.
        """
        ...

    @abstractmethod
    def wake_bucket(self, bucket_id: str) -> Bucket:
        """Reactivate a dormant bucket, restoring its Medoid to the active index.

        Args:
            bucket_id: The dormant bucket to wake.

        Returns:
            The reactivated :class:`Bucket`.

        Raises:
            ValueError: If *bucket_id* does not exist.
        """
        ...

    @abstractmethod
    def get_active_buckets(self) -> list[Bucket]:
        """Return all non-dormant buckets currently in the system."""
        ...

    @abstractmethod
    def get_all_buckets(self) -> list[Bucket]:
        """Return every bucket (including dormant ones)."""
        ...


class MemoryRetrievalEngine(ABC):
    """Top-level retrieval and conflict-resolution API.

    Composes :class:`VectorStore`, :class:`GraphStore`, :class:`BucketManager`,
    and :class:`LLMAdapter` to execute the full two-layer search pipeline.
    """

    @abstractmethod
    def search(
        self,
        query: str,
        max_hops: int | None = None,
        weight_threshold: float | None = None,
    ) -> SearchResult:
        """Execute the full two-layer retrieval pipeline.

        **Layer 1** — bucket-level coarse screening:
            1.  Embed *query*.
            2.  Score all active-bucket Medoid vectors, keep top *m*.
            3.  Within each selected bucket, fine-search for top *p* A-nodes.
            4.  Graph-associative expansion via cross-bucket edge traversal.

        **Layer 2** — conflict resolution:
            5.  Re-rank candidates with the :math:`α·sim + β/(1+ΔT) + γ·confidence` formula.
            6.  Run the LLM conflict-detection pass.
            7.  Mark stale nodes, return the ordered result.

        Args:
            query: Natural-language search query.
            max_hops: Overrides ``config.graph.max_hops`` when set.
            weight_threshold: Overrides ``config.graph.edge_weight_threshold``
                when set.

        Returns:
            :class:`SearchResult` with nodes ordered by final relevance score
            (highest first).  Stale nodes are included but downgraded.
        """
        ...

    @abstractmethod
    def resolve_conflicts(
        self, candidates: list[MemoryNode], query: str
    ) -> list[MemoryNode]:
        """Run the conflict-resolution pipeline on a set of candidate nodes.

        Steps:
            1.  Weighted re-rank (α, β, γ).
            2.  Truncate to top-N.
            3.  LLM contradiction check.
            4.  Mark older contradictory nodes as stale (side effect).

        Args:
            candidates: Unordered list of candidate nodes.
            query: The original query (used for semantic similarity scoring).

        Returns:
            Re-ranked list of nodes, stale-marked where applicable.
        """
        ...

    @abstractmethod
    def ingest(self, summary: str, content: str, confidence: float = 1.0) -> str:
        """Full ingestion pipeline: embed → assign bucket → store → return node ID.

        Args:
            summary: Summary paragraph / keywords (A).
            content: Detailed content to store (C).
            confidence: Source confidence in ``[0.0, 1.0]``.

        Returns:
            The newly created :class:`MemoryNode` ID.
        """
        ...
