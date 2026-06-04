"""SQLite-based persistence for nodes and buckets.

Provides save/load/delete for MemoryNode and Bucket state.
Engine can optionally inject a PersistenceStore for durability.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from memory_system.models import Bucket, MemoryNode

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    content TEXT NOT NULL,
    bucket_id TEXT,
    timestamp REAL NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    is_stale INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS buckets (
    id TEXT PRIMARY KEY,
    medoid_node_id TEXT,
    created_at REAL NOT NULL,
    last_write_at REAL NOT NULL,
    last_query_at REAL NOT NULL,
    is_dormant INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_nodes_bucket ON nodes(bucket_id);
CREATE INDEX IF NOT EXISTS idx_nodes_stale ON nodes(is_stale);
"""


class PersistenceStore:
    """SQLite-backed persistence for m-memory state."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def save_node(self, node: MemoryNode) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?)",
            (
                node.id, node.summary, node.content, node.bucket_id,
                node.timestamp, node.confidence, int(node.is_stale),
            ),
        )
        self._conn.commit()

    def save_bucket(self, bucket: Bucket) -> None:
        medoid_id = bucket.medoid.node_id if bucket.medoid else None
        self._conn.execute(
            "INSERT OR REPLACE INTO buckets VALUES (?,?,?,?,?,?)",
            (
                bucket.id, medoid_id, bucket.created_at,
                bucket.last_write_at, bucket.last_query_at, int(bucket.is_dormant),
            ),
        )
        self._conn.commit()

    def delete_node(self, node_id: str) -> None:
        self._conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))
        self._conn.commit()

    def delete_bucket(self, bucket_id: str) -> None:
        self._conn.execute("DELETE FROM buckets WHERE id=?", (bucket_id,))
        self._conn.commit()

    def load_all_nodes(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, summary, content, bucket_id, timestamp, confidence, is_stale FROM nodes"
        ).fetchall()
        return [
            {
                "id": r[0], "summary": r[1], "content": r[2],
                "bucket_id": r[3], "timestamp": r[4],
                "confidence": r[5], "is_stale": bool(r[6]),
            }
            for r in rows
        ]

    def load_all_buckets(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, medoid_node_id, created_at, last_write_at, "
            "last_query_at, is_dormant FROM buckets"
        ).fetchall()
        return [
            {
                "id": r[0], "medoid_node_id": r[1], "created_at": r[2],
                "last_write_at": r[3], "last_query_at": r[4],
                "is_dormant": bool(r[5]),
            }
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
