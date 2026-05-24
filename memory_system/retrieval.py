"""MemoryRetrievalEngine implementation — full two-layer retrieval pipeline.

Composes VectorStore, GraphStore, BucketManager, and LLMAdapter to execute
the complete search → graph expansion → conflict resolution flow.
"""

from __future__ import annotations

import logging
import time
import uuid

import numpy as np
from numpy.typing import NDArray

from memory_system.bucket_manager import BucketManagerImpl
from memory_system.config import MemorySystemConfig
from memory_system.interfaces import (
    GraphStore,
    LLMAdapter,
    MemoryRetrievalEngine,
    VectorStore,
)
from memory_system.llm_decision import (
    build_bucket_assignment_prompt,
    build_conflict_detection_prompt,
    parse_bucket_assignment_response,
    parse_conflict_detection_response,
)
from memory_system.models import (
    Bucket,
    MemoryNode,
    SearchResult,
)

logger = logging.getLogger(__name__)


class MemoryRetrievalEngineImpl(MemoryRetrievalEngine):
    """Full two-layer memory retrieval engine.

    Layer 1: Bucket-level coarse screening → in-bucket fine search → graph expansion.
    Layer 2: Weighted re-ranking → LLM conflict detection → stale marking.
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

        # BucketManager is composed, not injected — the engine owns it
        self._bucket_manager = BucketManagerImpl(
            config=config,
            vector_store=vector_store,
            graph_store=graph_store,
            llm=llm,
        )

        # Node storage
        self._nodes: dict[str, MemoryNode] = {}

    # ── MemoryRetrievalEngine interface ─────────────────────────────────

    def ingest(
        self,
        summary: str,
        content: str,
        confidence: float = 1.0,
    ) -> str:
        """Full ingestion pipeline: embed → assign → store."""
        # Embed
        summary_vec = self._vector_store.embed(summary)
        content_vec = self._vector_store.embed(content)

        node = MemoryNode(
            id=uuid.uuid4().hex[:12],
            summary=summary,
            content=content,
            summary_vector=summary_vec,
            content_vector=content_vec,
            timestamp=time.time(),
            confidence=confidence,
        )
        self._nodes[node.id] = node

        # Find candidate buckets
        candidates = self._bucket_manager.find_candidates(node)

        # LLM decision
        if candidates:
            prompt = build_bucket_assignment_prompt(
                node.summary,
                candidates,
            )
            response = self._llm.complete(prompt)
            try:
                decision = parse_bucket_assignment_response(response)
            except ValueError:
                logger.warning("Failed to parse LLM assignment, creating new bucket")
                decision = {
                    "primary_bucket": "new",
                    "reasoning": "parse failure fallback",
                    "cross_links": [],
                }
        else:
            decision = {
                "primary_bucket": "new",
                "reasoning": "no existing buckets",
                "cross_links": [],
            }

        # Assign
        primary_id = decision["primary_bucket"]
        cross_links = decision.get("cross_links", [])

        if primary_id == "new":
            bucket = self._bucket_manager.create_bucket(node)
        else:
            all_buckets = {
                b.id: b for b in self._bucket_manager.get_all_buckets()
            }
            existing = all_buckets.get(primary_id)
            if existing is None:
                bucket = self._bucket_manager.create_bucket(node)
            else:
                bucket = existing
                self._bucket_manager.assign_to_bucket(node, bucket, cross_links)

        # Store vectors for content-level search (Layer 2)
        self._vector_store.add(
            vectors=node.content_vector.reshape(1, -1),
            metadata=[{"id": f"content:{node.id}", "node_id": node.id}],
        )

        logger.info(
            "Ingested node %s into bucket %s: %s",
            node.id,
            bucket.id,
            summary[:80],
        )

        return node.id

    def search(
        self,
        query: str,
        max_hops: int | None = None,
        weight_threshold: float | None = None,
    ) -> SearchResult:
        """Execute the full two-layer retrieval pipeline."""
        hops = max_hops if max_hops is not None else self._config.graph.max_hops
        threshold = (
            weight_threshold
            if weight_threshold is not None
            else self._config.graph.edge_weight_threshold
        )

        # Step 1: Embed query
        query_vector = self._vector_store.embed(query)

        # Step 2: Bucket-level coarse screening (Layer 1)
        active_buckets = self._bucket_manager.get_active_buckets()

        # Also check dormant buckets — wake them if they match the query
        all_buckets = self._bucket_manager.get_all_buckets()
        dormant_buckets = [b for b in all_buckets if b.is_dormant]
        for db in dormant_buckets:
            if db.medoid is not None:
                sim = self._cosine_sim(query_vector, db.medoid.vector)
                # Use a similarity threshold from config or a sensible default
                if sim > 0.5:
                    self._bucket_manager.wake_bucket(db.id)
                    active_buckets.append(db)

        if not active_buckets:
            return SearchResult(nodes=[], scores=[])

        bucket_scores: list[tuple[Bucket, float]] = []
        for bucket in active_buckets:
            if bucket.medoid is None:
                continue
            sim = self._cosine_sim(query_vector, bucket.medoid.vector)
            bucket_scores.append((bucket, sim))

        bucket_scores.sort(key=lambda x: x[1], reverse=True)
        top_buckets = bucket_scores[: self._config.bucket.top_m]

        # Mark query hits for dormancy tracking
        now = time.time()
        for bucket, _ in top_buckets:
            bucket.last_query_at = now

        # Step 3: In-bucket fine search (A-node level)
        seed_nodes: list[tuple[MemoryNode, float]] = []
        for bucket, _ in top_buckets:
            for node_id in bucket.node_ids:
                node = self._nodes.get(node_id)
                if node is None:
                    continue
                sim = self._cosine_sim(query_vector, node.summary_vector)
                seed_nodes.append((node, sim))

        seed_nodes.sort(key=lambda x: x[1], reverse=True)
        limit = self._config.bucket.top_p * len(top_buckets)
        seed_nodes = seed_nodes[:limit]

        # Step 4: Graph expansion
        seed_ids = [n.id for n, _ in seed_nodes]
        paths = self._graph_store.traverse(seed_ids, hops, threshold)

        # Merge seed scores with traversal scores
        node_scores: dict[str, float] = {}
        for node, score in seed_nodes:
            if node.id not in node_scores or score > node_scores[node.id]:
                node_scores[node.id] = score

        for path in paths:
            target = path.node_ids[-1]
            if target not in node_scores or path.total_weight > node_scores[target]:
                node_scores[target] = path.total_weight

        # Step 5: Resolve conflicts (Layer 2)
        candidates = [
            self._nodes[nid]
            for nid in node_scores
            if nid in self._nodes
        ]
        resolved = self.resolve_conflicts(candidates, query)

        # Build result with final scores
        final_nodes: list[MemoryNode] = []
        final_scores: list[float] = []
        for node in resolved:
            final_nodes.append(node)
            final_scores.append(node_scores.get(node.id, 0.0))

        return SearchResult(nodes=final_nodes, scores=final_scores)

    def resolve_conflicts(
        self,
        candidates: list[MemoryNode],
        query: str,
    ) -> list[MemoryNode]:
        """Conflict resolution pipeline: re-rank → LLM check → mark stale."""
        if not candidates:
            return []

        # Step 1: Weighted re-ranking
        now = time.time()
        query_vector = self._vector_store.embed(query)

        alpha = self._config.conflict.alpha
        beta = self._config.conflict.beta
        gamma = self._config.conflict.gamma

        scored: list[tuple[MemoryNode, float]] = []
        for node in candidates:
            sim = self._cosine_sim(node.content_vector, query_vector)
            # Days since creation
            delta_t_days = (now - node.timestamp) / 86400.0
            time_factor = 1.0 / (1.0 + delta_t_days)
            confidence = node.confidence

            score = alpha * sim + beta * time_factor + gamma * confidence
            scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored[: self._config.conflict.top_n]

        # Step 2: LLM conflict detection
        prompt = build_conflict_detection_prompt(query, top_candidates)
        response = self._llm.complete(prompt)

        try:
            conflicts = parse_conflict_detection_response(response)
        except ValueError:
            logger.warning("Failed to parse conflict detection response")
            conflicts = []

        # Step 3: Mark stale nodes
        for conflict in conflicts:
            try:
                _newer_idx = int(conflict["newer_id"])
                older_idx = int(conflict["older_id"])
                if 0 <= older_idx < len(top_candidates):
                    older_node = top_candidates[older_idx][0]
                    older_node.is_stale = True
                    older_node.confidence *= self._config.conflict.stale_mark_downgrade_factor
                    logger.info(
                        "Marked node as stale: %s reason=%s",
                        older_node.id,
                        conflict.get("reason", ""),
                    )
            except (ValueError, IndexError):
                continue

        return [node for node, _ in top_candidates]

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_sim(
        a: NDArray[np.float32], b: NDArray[np.float32]
    ) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm < 1e-8 or b_norm < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))
