"""Periodic background cleanup tasks.

Runs bucket dormancy checks, stale-node scanning, contradiction detection,
and topic-drift detection on a configurable interval.  Uses ``asyncio`` or
a daemon thread so it never blocks the main write path.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import numpy as np
from numpy.typing import NDArray

from memory_system.config import MemorySystemConfig
from memory_system.interfaces import BucketManager, LLMAdapter, VectorStore
from memory_system.llm_decision import (
    build_conflict_detection_prompt,
    parse_conflict_detection_response,
)
from memory_system.models import Bucket, MemoryNode

logger = logging.getLogger(__name__)


class CleanupScheduler:
    """Orchestrates periodic background maintenance.

    Can run either as an ``asyncio.Task`` or a daemon ``threading.Thread``.

    Responsibilities:
    - Bucket dormancy checks.
    - Intra-bucket contradiction scanning.
    - Topic-drift detection (node vs Medoid similarity).
    - Bucket split eligibility assessment (deferred to BucketManager).
    """

    def __init__(
        self,
        config: MemorySystemConfig,
        bucket_manager: BucketManager,
        vector_store: VectorStore,
        llm: LLMAdapter,
        node_registry: dict[str, MemoryNode],
    ) -> None:
        self._config = config
        self._bucket_manager = bucket_manager
        self._vector_store = vector_store
        self._llm = llm
        self._nodes = node_registry

        self._running = False
        self._thread: threading.Thread | None = None
        self._task: asyncio.Task[None] | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start_sync(self) -> None:
        """Start cleanup as a daemon thread (suitable for sync contexts)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("CleanupScheduler started (thread)")

    def stop(self) -> None:
        """Stop the cleanup loop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._task is not None:
            self._task.cancel()
        logger.info("CleanupScheduler stopped")

    async def start_async(self) -> None:
        """Start cleanup as an asyncio task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop_async())
        logger.info("CleanupScheduler started (async)")

    def _run_loop(self) -> None:
        """Synchronous loop body."""
        while self._running:
            try:
                self._cleanup_cycle()
            except Exception:
                logger.exception("Cleanup cycle failed")
            time.sleep(self._config.cleanup.interval_seconds)

    async def _run_loop_async(self) -> None:
        """Asynchronous loop body."""
        while self._running:
            try:
                self._cleanup_cycle()
            except Exception:
                logger.exception("Cleanup cycle failed")
            await asyncio.sleep(self._config.cleanup.interval_seconds)

    # ── Cycle logic ──────────────────────────────────────────────────────

    def _cleanup_cycle(self) -> None:
        """Execute one full cleanup pass."""
        # 1. Dormancy check
        dormant = self._bucket_manager.dormancy_check()
        if dormant:
            logger.info("Marked dormant buckets: %d", len(dormant))

        # 2. Per-bucket scans
        for bucket in self._bucket_manager.get_all_buckets():
            if bucket.is_dormant:
                continue

            self._scan_contradictions(bucket)
            self._scan_topic_drift(bucket)

        # 3. Bucket split check (deferred — split_bucket is a placeholder)

    def _scan_contradictions(self, bucket: Bucket) -> None:
        """Scan node pairs within *bucket* for contradictions."""
        node_ids = list(bucket.node_ids)
        if len(node_ids) < 2:
            return

        similarity_threshold = self._config.cleanup.node_similarity_for_contradiction

        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                node_a = self._nodes.get(node_ids[i])
                node_b = self._nodes.get(node_ids[j])
                if node_a is None or node_b is None:
                    continue
                if node_a.is_stale or node_b.is_stale:
                    continue

                # Check content-vector similarity
                sim = self._cosine_sim(
                    node_a.content_vector, node_b.content_vector
                )
                if sim < similarity_threshold:
                    continue

                # Ask LLM whether there's a factual contradiction
                prompt = build_conflict_detection_prompt(
                    "contradiction scan",
                    [(node_a, sim), (node_b, sim)],
                )
                response = self._llm.complete(prompt)
                try:
                    conflicts = parse_conflict_detection_response(response)
                except ValueError:
                    continue

                for conflict in conflicts:
                    try:
                        older_idx = int(conflict["older_id"])
                        if older_idx == 0:
                            older = node_a
                        elif older_idx == 1:
                            older = node_b
                        else:
                            continue
                        older.is_stale = True
                        older.confidence *= self._config.conflict.stale_mark_downgrade_factor
                        logger.info(
                            "Cleanup: marked stale node %s", older.id
                        )
                    except (ValueError, IndexError):
                        continue

    def _scan_topic_drift(self, bucket: Bucket) -> None:
        """Check whether any node has drifted too far from its bucket Medoid."""
        if bucket.medoid is None:
            return

        floor = self._config.cleanup.medoid_similarity_floor

        for node_id in list(bucket.node_ids):
            node = self._nodes.get(node_id)
            if node is None:
                continue

            sim = self._cosine_sim(node.content_vector, bucket.medoid.vector)
            if sim < floor:
                # Mark as stale (soft removal — node stays but is downgraded)
                node.is_stale = True
                logger.info(
                    "Topic drift detected node=%s sim=%.4f bucket=%s",
                    node_id,
                    round(sim, 4),
                    bucket.id,
                )

    @staticmethod
    def _cosine_sim(
        a: NDArray[np.float32],
        b: NDArray[np.float32],
    ) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm < 1e-8 or b_norm < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))
