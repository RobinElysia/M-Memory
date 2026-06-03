"""BucketManager implementation — cluster lifecycle and node assignment.

Handles Medoid computation, bucket creation/split/dormancy, and LLM-driven
node-to-bucket assignment via the injected :class:`LLMAdapter`.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import numpy as np
from numpy.typing import NDArray

from memory_system.config import MemorySystemConfig
from memory_system.interfaces import BucketManager, EdgeType, GraphStore, LLMAdapter, VectorStore
from memory_system.models import Bucket, Medoid, MemoryNode

logger = logging.getLogger(__name__)


class BucketManagerImpl(BucketManager):
    """Concrete bucket manager.

    Composes ``VectorStore`` for similarity computation, ``GraphStore`` for
    cross-bucket edges, and ``LLMAdapter`` for assignment decisions.

    **Medoid computation** is deterministic: the node whose average cosine
    distance to all other nodes in the bucket is minimal.  Ties are broken
    by ``node_id`` lexicographic order.
    """

    def __init__(
        self,
        config: MemorySystemConfig,
        vector_store: VectorStore,
        graph_store: GraphStore,
        llm: LLMAdapter,
    ) -> None:
        self._config = config
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._llm = llm

        # Internal registry
        self._buckets: dict[str, Bucket] = {}
        self._nodes: dict[str, MemoryNode] = {}

    # ── BucketManager interface ──────────────────────────────────────────

    def create_bucket(self, medoid_node: MemoryNode) -> Bucket:
        """Create a new bucket with *medoid_node* as its initial Medoid."""
        bucket = Bucket(
            id=uuid.uuid4().hex[:12],
            medoid=Medoid(
                node_id=medoid_node.id,
                summary=medoid_node.summary,
                vector=medoid_node.summary_vector.copy(),
                version=0,
            ),
            node_ids=[medoid_node.id],
            created_at=time.time(),
            last_write_at=time.time(),
            last_query_at=time.time(),
            version=0,
        )

        medoid_node.bucket_id = bucket.id

        self._buckets[bucket.id] = bucket
        self._nodes[medoid_node.id] = medoid_node

        # Register in graph
        self._graph_store.add_node(medoid_node.id, {"type": "memory_node"})

        # Register Medoid vector in the vector index for search
        self._vector_store.add(
            vectors=medoid_node.summary_vector.reshape(1, -1),
            metadata=[{"id": f"medoid:{bucket.id}", "bucket_id": bucket.id}],
        )

        return bucket

    def find_candidates(
        self, node_a: MemoryNode
    ) -> list[tuple[Bucket, float]]:
        """Find top-k candidate buckets for node assignment.

        In semantic mode: uses cosine similarity of summary vectors.
        In lexical mode (non-semantic embeddings): uses keyword overlap
        between the node's summary and the bucket Medoid's summary,
        avoiding random hash-based similarity.
        """
        active = self.get_active_buckets()
        if not active:
            return []

        if self._vector_store.is_semantic():
            return self._semantic_find_candidates(node_a, active)
        else:
            return self._lexical_find_candidates(node_a, active)

    def _semantic_find_candidates(
        self, node_a: MemoryNode, active: list[Bucket]
    ) -> list[tuple[Bucket, float]]:
        """Vector-based candidate selection (semantic embeddings)."""
        scored: list[tuple[Bucket, float]] = []
        for bucket in active:
            if bucket.medoid is None:
                continue
            sim = self._cosine_sim(
                node_a.summary_vector, bucket.medoid.vector
            )
            scored.append((bucket, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self._config.bucket.top_k]

    def _lexical_find_candidates(
        self, node_a: MemoryNode, active: list[Bucket]
    ) -> list[tuple[Bucket, float]]:
        """Keyword-based candidate selection (non-semantic fallback).

        Scores buckets by keyword overlap between the new node's summary+content
        and each bucket Medoid's summary+content.  This avoids the random
        similarity produced by hash-based embeddings.
        """
        node_words = set(
            node_a.summary.lower().split() + node_a.content.lower().split()
        )
        # Remove very common words
        stop = {"the","a","an","is","are","was","were","in","on","at","to",
                "of","for","with","and","or","i","my","me","you","your","it"}
        node_words -= stop

        if not node_words:
            return active[: self._config.bucket.top_k]

        scored: list[tuple[Bucket, float]] = []
        for bucket in active:
            if bucket.medoid is None:
                continue
            medoid_node = self._nodes.get(bucket.medoid.node_id)
            if medoid_node is None:
                continue
            bucket_words = set(
                medoid_node.summary.lower().split() +
                medoid_node.content.lower().split()
            )
            bucket_words -= stop
            overlap = len(node_words & bucket_words)
            if overlap > 0:
                scored.append((bucket, float(overlap)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self._config.bucket.top_k]

    def assign_to_bucket(
        self,
        node_a: MemoryNode,
        bucket: Bucket,
        cross_links: list[dict[str, Any]],
    ) -> None:
        """Place *node_a* into *bucket* and set up cross-bucket edges."""
        node_a.bucket_id = bucket.id
        bucket.node_ids.append(node_a.id)
        bucket.last_write_at = time.time()

        self._nodes[node_a.id] = node_a

        # Register node in graph
        self._graph_store.add_node(node_a.id, {"type": "memory_node"})

        # Intra-bucket edge: node → Medoid
        if bucket.medoid is not None:
            self._graph_store.add_edge(
                node_a.id,
                bucket.medoid.node_id,
                EdgeType.INTRA_BUCKET,
                weight=1.0,
            )

        # Cross-bucket edges
        for link in cross_links:
            target_bucket_id = link.get("bucket_id", "")
            weight = float(link.get("weight", 0.5))
            if target_bucket_id and target_bucket_id != bucket.id:
                target_bucket = self._buckets.get(target_bucket_id)
                if target_bucket is not None and target_bucket.medoid is not None:
                    self._add_cross_edge_with_eviction(
                        node_a.id,
                        target_bucket.medoid.node_id,
                        weight,
                    )

        # Update Medoid
        self._update_medoid(bucket)

        # Update Medoid vector in the index
        if bucket.medoid is not None:
            self._update_medoid_vector_in_index(bucket)

    def split_bucket(self, bucket: Bucket) -> list[Bucket]:
        """Split a bucket into two sub-buckets via spectral clustering.

        Builds a pairwise cosine similarity matrix for all nodes in the bucket,
        computes the second eigenvector of the Laplacian, and partitions nodes
        by sign.  Each partition becomes a new bucket with its own Medoid.

        Returns the original bucket unchanged if it has fewer than
        ``split_threshold`` nodes or if the partition is degenerate.
        """
        if len(bucket.node_ids) < self._config.bucket.split_threshold:
            return [bucket]

        # Build similarity matrix
        n = len(bucket.node_ids)
        nodes = [self._nodes[nid] for nid in bucket.node_ids if nid in self._nodes]
        if len(nodes) < 2:
            return [bucket]

        sim_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_sim(nodes[i].summary_vector, nodes[j].summary_vector)
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
            sim_matrix[i, i] = 1.0

        # Compute degree matrix and Laplacian
        degree = np.sum(sim_matrix, axis=1)
        # Normalized Laplacian: L = I - D^(-1/2) * S * D^(-1/2)
        d_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-8)))
        laplacian = np.eye(n) - d_sqrt_inv @ sim_matrix @ d_sqrt_inv

        # Second smallest eigenvector (Fiedler vector)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            fiedler = eigenvectors[:, 1]  # second eigenvector
        except np.linalg.LinAlgError:
            return [bucket]

        # Partition by sign
        cluster_a = [bucket.node_ids[i] for i in range(n) if fiedler[i] >= 0]
        cluster_b = [bucket.node_ids[i] for i in range(n) if fiedler[i] < 0]

        if len(cluster_a) < 1 or len(cluster_b) < 1:
            return [bucket]  # degenerate — no split

        # Create new buckets
        new_buckets: list[Bucket] = []
        for cluster_ids in (cluster_a, cluster_b):
            new_b = Bucket(
                id=uuid.uuid4().hex[:12],
                node_ids=list(cluster_ids),
                created_at=time.time(),
                last_write_at=time.time(),
                last_query_at=time.time(),
            )
            # Register nodes and graph nodes
            for nid in cluster_ids:
                node = self._nodes.get(nid)
                if node:
                    node.bucket_id = new_b.id
                    self._graph_store.add_node(nid, {"type": "memory_node"})
            # Compute Medoid
            self._update_medoid(new_b)
            # Register Medoid vector
            if new_b.medoid:
                self._vector_store.add(
                    vectors=new_b.medoid.vector.reshape(1, -1),
                    metadata=[{"id": f"medoid:{new_b.id}", "bucket_id": new_b.id}],
                )
            new_buckets.append(new_b)
            self._buckets[new_b.id] = new_b

        # Remove old bucket
        del self._buckets[bucket.id]
        try:
            self._vector_store.remove([f"medoid:{bucket.id}"])
        except KeyError:
            pass

        logger.info(
            "Split bucket %s into %s (%d nodes) and %s (%d nodes)",
            bucket.id[:8],
            new_buckets[0].id[:8], len(new_buckets[0].node_ids),
            new_buckets[1].id[:8], len(new_buckets[1].node_ids),
        )
        return new_buckets

    def dormancy_check(self) -> list[Bucket]:
        """Identify and mark dormant buckets."""
        now = time.time()
        newly_dormant: list[Bucket] = []
        interval = self._config.bucket.dormancy_interval_seconds

        for bucket in self._buckets.values():
            if bucket.is_dormant:
                continue
            if (
                now - bucket.last_write_at > interval
                and now - bucket.last_query_at > interval
            ):
                bucket.is_dormant = True
                # Remove Medoid from active index
                try:
                    self._vector_store.remove([f"medoid:{bucket.id}"])
                except KeyError:
                    pass
                newly_dormant.append(bucket)

        return newly_dormant

    def wake_bucket(self, bucket_id: str) -> Bucket:
        """Reactivate a dormant bucket."""
        bucket = self._buckets.get(bucket_id)
        if bucket is None:
            raise ValueError(f"Bucket '{bucket_id}' not found")

        bucket.is_dormant = False
        bucket.last_query_at = time.time()

        # Re-add Medoid to index
        if bucket.medoid is not None:
            self._vector_store.add(
                vectors=bucket.medoid.vector.reshape(1, -1),
                metadata=[{"id": f"medoid:{bucket.id}", "bucket_id": bucket.id}],
            )

        return bucket

    def get_active_buckets(self) -> list[Bucket]:
        """Return all non-dormant buckets."""
        return [b for b in self._buckets.values() if not b.is_dormant]

    def get_all_buckets(self) -> list[Bucket]:
        """Return every bucket."""
        return list(self._buckets.values())

    # ── Internal helpers ─────────────────────────────────────────────────

    def _update_medoid(self, bucket: Bucket) -> None:
        """Recompute the Medoid for *bucket* deterministically."""
        if len(bucket.node_ids) == 0:
            bucket.medoid = None
            return

        if len(bucket.node_ids) == 1:
            node_id = bucket.node_ids[0]
            node = self._nodes.get(node_id)
            if node is not None:
                bucket.medoid = Medoid(
                    node_id=node.id,
                    summary=node.summary,
                    vector=node.summary_vector.copy(),
                    version=bucket.version + 1,
                )
            bucket.version += 1
            return

        # Compute average cosine distance for each node
        best_node_id = bucket.node_ids[0]
        best_avg_dist = float("inf")

        for nid in bucket.node_ids:
            node = self._nodes.get(nid)
            if node is None:
                continue
            total_dist = 0.0
            count = 0
            for other_nid in bucket.node_ids:
                if other_nid == nid:
                    continue
                other = self._nodes.get(other_nid)
                if other is None:
                    continue
                # Cosine distance = 1 - cosine similarity
                sim = self._cosine_sim(node.summary_vector, other.summary_vector)
                total_dist += 1.0 - sim
                count += 1

            if count > 0:
                avg_dist = total_dist / count
                if avg_dist < best_avg_dist or (
                    abs(avg_dist - best_avg_dist) < 1e-9 and nid < best_node_id
                ):
                    best_avg_dist = avg_dist
                    best_node_id = nid

        best_node = self._nodes.get(best_node_id)
        if best_node is not None:
            bucket.medoid = Medoid(
                node_id=best_node_id,
                summary=best_node.summary,
                vector=best_node.summary_vector.copy(),
                version=bucket.version + 1,
            )
        bucket.version += 1

    def _update_medoid_vector_in_index(self, bucket: Bucket) -> None:
        """Replace the Medoid vector in the vector index."""
        try:
            self._vector_store.remove([f"medoid:{bucket.id}"])
        except KeyError:
            pass
        if bucket.medoid is not None:
            self._vector_store.add(
                vectors=bucket.medoid.vector.reshape(1, -1),
                metadata=[{"id": f"medoid:{bucket.id}", "bucket_id": bucket.id}],
            )

    def _add_cross_edge_with_eviction(
        self, from_node_id: str, to_node_id: str, weight: float
    ) -> None:
        """Add a cross-bucket edge, evicting the weakest if out-degree is full."""
        existing_cross = self._graph_store.get_out_edges(
            from_node_id, EdgeType.CROSS_BUCKET
        )

        if len(existing_cross) >= self._config.graph.max_out_degree:
            # Evict weakest
            weakest = min(existing_cross, key=lambda e: e.weight)
            self._graph_store.remove_edge(weakest.id)

        self._graph_store.add_edge(
            from_node_id,
            to_node_id,
            EdgeType.CROSS_BUCKET,
            weight=weight,
        )

    @staticmethod
    def _cosine_sim(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        """Cosine similarity between two vectors."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm < 1e-8 or b_norm < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))
