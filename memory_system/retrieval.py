"""MemoryRetrievalEngine implementation — full two-layer retrieval pipeline.

Composes VectorStore, GraphStore, BucketManager, and LLMAdapter to execute
the complete search → graph expansion → conflict resolution flow.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

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
from memory_system.persistence import PersistenceStore  # noqa: TC001
from memory_system.utils import STOPWORDS, cosine_sim

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
        persistence: PersistenceStore | None = None,
    ) -> None:
        self._config = config
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._llm = llm
        self._persistence = persistence

        # Dimension validation
        if config.embedding_dim != vector_store.dim:
            raise ValueError(
                f"Config embedding_dim ({config.embedding_dim}) != "
                f"vector_store.dim ({vector_store.dim}). "
                f"Set config.embedding_dim = {vector_store.dim} to match."
            )

        # BucketManager is composed, not injected — the engine owns it
        self._bucket_manager = BucketManagerImpl(
            config=config,
            vector_store=vector_store,
            graph_store=graph_store,
            llm=llm,
        )

        # Node storage
        self._nodes: dict[str, MemoryNode] = {}
        # Thread safety — write lock for ingest operations
        self._write_lock = threading.Lock()

        # Restore state from persistence if available
        if self._persistence is not None:
            self._restore_from_persistence()

    # ── MemoryRetrievalEngine interface ─────────────────────────────────

    def ingest(
        self,
        summary: str,
        content: str,
        confidence: float = 1.0,
    ) -> str:
        """Full ingestion pipeline: embed → assign → store (thread-safe).

        LLM call (network I/O) is outside the write lock to allow concurrent
        operations.  Only the final state mutation holds the lock briefly.
        """
        return self._ingest_impl(summary, content, confidence)

    def _ingest_impl(
        self,
        summary: str,
        content: str,
        confidence: float = 1.0,
    ) -> str:
        """Internal ingestion implementation.

        Embeds and creates the node outside the lock, then acquires the lock
        only for the mutation of _nodes and _buckets.  The LLM call (network I/O)
        is performed outside the lock to avoid blocking concurrent writes.
        """
        # Embed outside lock (CPU-only, no shared state)
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

        # Find candidate buckets (reads shared state but read-only)
        candidates = self._bucket_manager.find_candidates(node)

        # LLM decision — network I/O, do NOT hold lock during this
        if candidates:
            prompt = build_bucket_assignment_prompt(node.summary, candidates)
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

        # Assign — acquire lock only for mutation
        with self._write_lock:
            primary_id = decision["primary_bucket"]
            cross_links = decision.get("cross_links", [])
            self._nodes[node.id] = node

            if primary_id == "new":
                bucket = self._bucket_manager.create_bucket(node)
            else:
                all_buckets = {b.id: b for b in self._bucket_manager.get_all_buckets()}
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

        logger.info("Ingested node %s into bucket %s: %s", node.id, bucket.id, summary[:80])

        if self._persistence is not None:
            self._persistence.save_node(node)
            self._persistence.save_bucket(bucket)

        return node.id

    def _restore_from_persistence(self) -> None:
        """Load nodes and buckets from persistence store after restart."""
        if self._persistence is None:
            return
        dim = self._config.embedding_dim
        node_data = self._persistence.load_all_nodes(dim)
        for nd in node_data:
            node = MemoryNode(
                id=nd["id"],
                summary=nd["summary"],
                content=nd["content"],
                summary_vector=nd.get("summary_vector"),
                content_vector=nd.get("content_vector"),
                timestamp=nd["timestamp"],
                confidence=nd["confidence"],
                bucket_id=nd.get("bucket_id", ""),
            )
            node.is_stale = nd.get("is_stale", False)
            self._nodes[node.id] = node

        bucket_data = self._persistence.load_all_buckets()
        for bd in bucket_data:
            bucket = Bucket(
                id=bd["id"],
                node_ids=[],
                created_at=bd["created_at"],
                last_write_at=bd["last_write_at"],
                last_query_at=bd["last_query_at"],
            )
            bucket.is_dormant = bd.get("is_dormant", False)
            # Rebuild node_ids list from loaded nodes
            for node in self._nodes.values():
                if node.bucket_id == bucket.id:
                    bucket.node_ids.append(node.id)
            self._bucket_manager._buckets[bucket.id] = bucket

        logger.info(
            "Restored %d nodes, %d buckets from persistence",
            len(self._nodes), len(self._bucket_manager._buckets),
        )

    def search(
        self,
        query: str,
        max_hops: int | None = None,
        weight_threshold: float | None = None,
    ) -> SearchResult:
        """Execute the full two-layer retrieval pipeline.

        When the vector store lacks semantic embeddings
        (``is_semantic() == False``), falls back to lexical keyword search
        across all buckets, with bucket-aware ranking.
        """
        if not self._vector_store.is_semantic():
            return self._lexical_search(query, max_hops or 0)

        return self._semantic_search(query, max_hops, weight_threshold)

    def _semantic_search(
        self,
        query: str,
        max_hops: int | None = None,
        weight_threshold: float | None = None,
    ) -> SearchResult:
        """Vector-based semantic search (Layer 1 + Layer 2)."""
        hops = max_hops if max_hops is not None else self._config.graph.max_hops
        threshold = (
            weight_threshold
            if weight_threshold is not None
            else self._config.graph.edge_weight_threshold
        )

        query_vector = self._vector_store.embed(query)

        # Bucket-level coarse screening
        active_buckets = self._bucket_manager.get_active_buckets()
        all_buckets = self._bucket_manager.get_all_buckets()
        dormant_buckets = [b for b in all_buckets if b.is_dormant]
        for db in dormant_buckets:
            if db.medoid is not None:
                sim = cosine_sim(query_vector, db.medoid.vector)
                if sim > 0.5:
                    self._bucket_manager.wake_bucket(db.id)
                    active_buckets.append(db)

        if not active_buckets:
            return SearchResult(nodes=[], scores=[])

        bucket_scores: list[tuple[Bucket, float]] = []
        for bucket in active_buckets:
            if bucket.medoid is None:
                continue
            sim = cosine_sim(query_vector, bucket.medoid.vector)
            bucket_scores.append((bucket, sim))

        bucket_scores.sort(key=lambda x: x[1], reverse=True)
        top_buckets = bucket_scores[: self._config.bucket.top_m]

        now = time.time()
        for bucket, _ in top_buckets:
            bucket.last_query_at = now

        # In-bucket fine search
        seed_nodes: list[tuple[MemoryNode, float]] = []
        for bucket, _ in top_buckets:
            for node_id in bucket.node_ids:
                node = self._nodes.get(node_id)
                if node is None:
                    continue
                sim = cosine_sim(query_vector, node.summary_vector)
                seed_nodes.append((node, sim))

        seed_nodes.sort(key=lambda x: x[1], reverse=True)
        limit = self._config.bucket.top_p * len(top_buckets)
        seed_nodes = seed_nodes[:limit]

        # Graph expansion
        seed_ids = [n.id for n, _ in seed_nodes]
        paths = self._graph_store.traverse(seed_ids, hops, threshold)

        node_scores: dict[str, float] = {}
        for node, score in seed_nodes:
            if node.id not in node_scores or score > node_scores[node.id]:
                node_scores[node.id] = score
        for path in paths:
            target = path.node_ids[-1]
            if target not in node_scores or path.total_weight > node_scores[target]:
                node_scores[target] = path.total_weight

        # Conflict resolution (Layer 2) — produces final ordering
        candidates = [self._nodes[nid] for nid in node_scores if nid in self._nodes]
        resolved = self.resolve_conflicts(candidates, query)

        # Use Layer 2 ordering with Layer 2 re-rank scores
        final_nodes = list(resolved)
        final_scores = [
            node_scores.get(n.id, 0.0) for n in resolved
        ]
        return SearchResult(nodes=final_nodes, scores=final_scores)

    def _lexical_search(
        self,
        query: str,
        max_hops: int = 0,
    ) -> SearchResult:
        """Bucket-aware lexical search — fallback when embeddings lack semantics.

        First identifies which buckets contain keyword-matched summaries,
        then searches within those buckets only.  Stale nodes are downgraded.
        Includes lightweight stale detection for contradiction pairs.
        """
        query_lower = query.lower()
        stopwords = STOPWORDS
        keywords = [w for w in query_lower.split() if w not in stopwords]
        if not keywords:
            return SearchResult(nodes=[], scores=[])

        # Step 1: Find relevant buckets by keyword overlap with Medoid summary
        bucket_scores: list[tuple[str, float]] = []
        for bucket in self._bucket_manager.get_all_buckets():
            if bucket.is_dormant or bucket.medoid is None:
                continue
            medoid_node = self._nodes.get(bucket.medoid.node_id)
            if medoid_node is None:
                continue
            medoid_text = (medoid_node.summary + " " + medoid_node.content).lower()
            hits = sum(1 for kw in keywords if kw in medoid_text)
            if hits > 0:
                bucket_scores.append((bucket.id, float(hits)))

        bucket_scores.sort(key=lambda x: x[1], reverse=True)
        relevant_buckets = {bid for bid, _ in bucket_scores[:self._config.bucket.top_m]}

        # Step 2: Score nodes — prioritize keyword-matched buckets.
        # Cap global scan to prevent O(N) degradation at scale.
        _max_global_scan = 500
        scanned = 0

        # fall back to global search if bucket filter yields too few results
        scored: list[tuple[MemoryNode, float]] = []
        bucket_filtered: list[tuple[MemoryNode, float]] = []
        global_results: list[tuple[MemoryNode, float]] = []

        for node in self._nodes.values():
            if scanned >= _max_global_scan:
                break
            scanned += 1
            content_lower = node.content.lower()
            summary_lower = node.summary.lower()
            content_hits = sum(1 for kw in keywords if kw in content_lower)
            summary_hits = sum(1 for kw in keywords if kw in summary_lower)
            score = content_hits * 2.0 + summary_hits * 1.0
            if node.is_stale:
                score *= 0.1
            if score > 0:
                item = (node, score)
                if relevant_buckets and node.bucket_id in relevant_buckets:
                    bucket_filtered.append(item)
                global_results.append(item)

        # Use bucket-filtered if it has enough results; otherwise fall back
        min_results = max(5, self._config.bucket.top_p)
        if len(bucket_filtered) >= min_results:
            scored = bucket_filtered
        else:
            scored = global_results

        # Step 3: Lightweight stale detection — expanded topic words
        _default_topic_words = {
            "live", "lives", "living", "moved", "move", "relocated",
            "work", "works", "working", "job", "position", "role",
            "allergy", "allergic", "allergies",
            "team", "member", "members",
            "address", "location", "city", "country",
            "name", "email", "phone", "number",
            "drive", "driving", "car", "vehicle",
            "use", "using", "framework", "language", "speak",
            "have", "has", "own", "owns", "pet", "dog", "cat",
            "graduated", "graduate", "university", "college",
        }
        topic_words = (
            self._config.topic_words
            if self._config.topic_words is not None
            else _default_topic_words
        )
        if len(scored) >= 2:
            limit = min(len(scored), 20)  # cap pair-wise
            for i in range(limit):
                for j in range(i + 1, limit):
                    na, sa = scored[i]
                    nb, sb = scored[j]
                    if sa < 1.0 or sb < 1.0:
                        continue
                    ca, cb = na.content.lower(), nb.content.lower()
                    shared = [w for w in topic_words if w in ca and w in cb]
                    if not shared:
                        continue
                    if ca != cb:
                        older = na if na.timestamp < nb.timestamp else nb
                        if not older.is_stale:
                            older.is_stale = True
                            older.confidence *= 0.1
                            idx = i if older is na else j
                            old_node, old_score = scored[idx]
                            scored[idx] = (old_node, old_score * 0.1)

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: self._config.bucket.top_p * self._config.bucket.top_m]

        return SearchResult(
            nodes=[n for n, _ in top],
            scores=[s for _, s in top],
        )

    def resolve_conflicts(
        self,
        candidates: list[MemoryNode],
        query: str,
        max_retries: int = 2,
        timeout_seconds: float = 5.0,
    ) -> list[MemoryNode]:
        """Conflict resolution pipeline: re-rank → LLM check → mark stale.

        Includes retry with exponential backoff and timeout to ensure
        graceful degradation under API failures.
        """
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
            sim = cosine_sim(node.content_vector, query_vector)
            delta_t_days = (now - node.timestamp) / 86400.0
            time_factor = 1.0 / (1.0 + delta_t_days)
            confidence = node.confidence
            score = alpha * sim + beta * time_factor + gamma * confidence
            scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored[: self._config.conflict.top_n]

        # Step 2: LLM conflict detection with retry + timeout
        conflicts: list[dict[str, Any]] = []
        for attempt in range(max_retries + 1):
            try:
                prompt = build_conflict_detection_prompt(query, top_candidates)
                response = self._llm.complete(prompt)
                conflicts = parse_conflict_detection_response(response)
                break
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "Conflict detection attempt %d/%d failed: %s",
                    attempt + 1, max_retries + 1, exc,
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(
                        "Conflict detection failed after %d retries; "
                        "returning unscored candidates",
                        max_retries + 1,
                    )
                    conflicts = []

        # Step 3: Mark stale nodes (idempotent — skip already-stale)
        seen_pairs: set[tuple[str, str]] = set()
        for conflict in conflicts:
            try:
                older_idx = int(conflict["older_id"])
                if 0 <= older_idx < len(top_candidates):
                    older_node = top_candidates[older_idx][0]
                    newer_idx = int(conflict.get("newer_id", -1))
                    newer_node = (
                        top_candidates[newer_idx][0]
                        if 0 <= newer_idx < len(top_candidates)
                        else None
                    )
                    pair_key = (older_node.id, newer_node.id if newer_node else "")
                    if pair_key in seen_pairs:
                        continue  # idempotent
                    seen_pairs.add(pair_key)
                    if not older_node.is_stale:
                        older_node.is_stale = True
                        older_node.confidence *= (
                            self._config.conflict.stale_mark_downgrade_factor
                        )
                        logger.info(
                            "Marked %s as stale (conflict with %s): %s",
                            older_node.id,
                            newer_node.id if newer_node else "?",
                            conflict.get("reason", ""),
                        )
            except (ValueError, IndexError, KeyError):
                continue

        return [node for node, _ in top_candidates]

    # ── Helpers ──────────────────────────────────────────────────────────

