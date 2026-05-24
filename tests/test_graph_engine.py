"""Tests for NetworkX-based GraphStore implementation."""

from __future__ import annotations

import pytest

from memory_system.graph_engine import NetworkXGraphStore
from memory_system.models import Edge, EdgeType, TraversalPath


class TestNetworkXGraphStore:
    """Unit tests for NetworkXGraphStore."""

    @pytest.fixture
    def store(self) -> NetworkXGraphStore:
        return NetworkXGraphStore()

    # ── add_node ──────────────────────────────────────────────────────────

    def test_add_node_succeeds(self, store: NetworkXGraphStore) -> None:
        store.add_node("n1", {"label": "test"})
        assert store.get_node_count() == 1

    def test_add_duplicate_node_raises(self, store: NetworkXGraphStore) -> None:
        store.add_node("n1", {})
        with pytest.raises(ValueError, match="already exists"):
            store.add_node("n1", {})

    # ── add_edge ──────────────────────────────────────────────────────────

    def test_add_edge_returns_id(self, store: NetworkXGraphStore) -> None:
        store.add_node("a", {})
        store.add_node("b", {})
        eid = store.add_edge("a", "b", EdgeType.CROSS_BUCKET, 0.8)
        assert isinstance(eid, str)
        assert len(eid) > 0

    def test_add_edge_missing_node_raises(self, store: NetworkXGraphStore) -> None:
        store.add_node("a", {})
        with pytest.raises(ValueError, match="does not exist"):
            store.add_edge("a", "nonexistent", EdgeType.TEMPORAL, 1.0)

    def test_add_edge_zero_weight(self, store: NetworkXGraphStore) -> None:
        store.add_node("a", {})
        store.add_node("b", {})
        eid = store.add_edge("a", "b", EdgeType.TEMPORAL, 0.0)
        assert store.get_edge_count() == 1

    # ── get_out_edges ─────────────────────────────────────────────────────

    def test_get_out_edges_filtered(self, store: NetworkXGraphStore) -> None:
        store.add_node("a", {})
        store.add_node("b", {})
        store.add_node("c", {})
        store.add_edge("a", "b", EdgeType.TEMPORAL, 0.5)
        store.add_edge("a", "c", EdgeType.CROSS_BUCKET, 0.9)

        temporal = store.get_out_edges("a", EdgeType.TEMPORAL)
        assert len(temporal) == 1
        assert temporal[0].edge_type == EdgeType.TEMPORAL

        cross = store.get_out_edges("a", EdgeType.CROSS_BUCKET)
        assert len(cross) == 1
        assert cross[0].target_id == "c"

    def test_get_out_edges_all(self, store: NetworkXGraphStore) -> None:
        store.add_node("a", {})
        store.add_node("b", {})
        store.add_edge("a", "b", EdgeType.TEMPORAL, 0.5)
        store.add_edge("a", "b", EdgeType.CROSS_BUCKET, 0.7)

        all_edges = store.get_out_edges("a")
        assert len(all_edges) == 2

    def test_get_out_edges_nonexistent_node(self, store: NetworkXGraphStore) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            store.get_out_edges("nonexistent")

    # ── remove_edge ───────────────────────────────────────────────────────

    def test_remove_edge(self, store: NetworkXGraphStore) -> None:
        store.add_node("a", {})
        store.add_node("b", {})
        eid = store.add_edge("a", "b", EdgeType.TEMPORAL, 0.5)
        store.remove_edge(eid)
        assert store.get_edge_count() == 0

    def test_remove_nonexistent_edge_raises(self, store: NetworkXGraphStore) -> None:
        with pytest.raises(KeyError):
            store.remove_edge("nonexistent")

    # ── traverse ──────────────────────────────────────────────────────────

    def test_traverse_single_hop(self, store: NetworkXGraphStore) -> None:
        # a → b (cross, 0.9), a → c (cross, 0.3)
        for nid in ("a", "b", "c"):
            store.add_node(nid, {})
        store.add_edge("a", "b", EdgeType.CROSS_BUCKET, 0.9)
        store.add_edge("a", "c", EdgeType.CROSS_BUCKET, 0.3)

        paths = store.traverse(["a"], max_hops=1, weight_threshold=0.5)
        # Only b is reachable (c is below threshold)
        assert len(paths) >= 1
        target_ids = {p.node_ids[-1] for p in paths}
        assert "b" in target_ids
        assert "c" not in target_ids

    def test_traverse_multi_hop(self, store: NetworkXGraphStore) -> None:
        # a → b (0.9) → c (0.8)
        for nid in ("a", "b", "c"):
            store.add_node(nid, {})
        store.add_edge("a", "b", EdgeType.CROSS_BUCKET, 0.9)
        store.add_edge("b", "c", EdgeType.CROSS_BUCKET, 0.8)

        paths = store.traverse(["a"], max_hops=2, weight_threshold=0.5)
        target_ids = {p.node_ids[-1] for p in paths}
        assert "b" in target_ids
        assert "c" in target_ids

    def test_traverse_ignores_non_cross_edges(self, store: NetworkXGraphStore) -> None:
        for nid in ("a", "b"):
            store.add_node(nid, {})
        store.add_edge("a", "b", EdgeType.TEMPORAL, 1.0)

        paths = store.traverse(["a"], max_hops=2, weight_threshold=0.1)
        # Temporal edges should NOT be followed
        target_ids = {p.node_ids[-1] for p in paths}
        assert "b" not in target_ids

    def test_traverse_best_single_path_dedup(self, store: NetworkXGraphStore) -> None:
        # a → b (0.9), a → c (0.8), c → b (0.6)
        # b reachable via a→b (weight 0.9) and a→c→b (weight 0.8*0.6=0.48)
        for nid in ("a", "b", "c"):
            store.add_node(nid, {})
        store.add_edge("a", "b", EdgeType.CROSS_BUCKET, 0.9)
        store.add_edge("a", "c", EdgeType.CROSS_BUCKET, 0.8)
        store.add_edge("c", "b", EdgeType.CROSS_BUCKET, 0.6)

        paths = store.traverse(["a"], max_hops=2, weight_threshold=0.1)
        # b should appear with best score (0.9, not 0.48)
        b_paths = [p for p in paths if p.node_ids[-1] == "b"]
        assert len(b_paths) >= 1
        # The best path should have weight 0.9
        best = max(b_paths, key=lambda p: p.total_weight)
        assert best.total_weight == pytest.approx(0.9)

    def test_traverse_empty_start(self, store: NetworkXGraphStore) -> None:
        paths = store.traverse([], max_hops=2, weight_threshold=0.5)
        assert paths == []

    # ── counts ────────────────────────────────────────────────────────────

    def test_get_node_count(self, store: NetworkXGraphStore) -> None:
        assert store.get_node_count() == 0
        store.add_node("a", {})
        assert store.get_node_count() == 1

    def test_get_edge_count_filtered(self, store: NetworkXGraphStore) -> None:
        store.add_node("a", {})
        store.add_node("b", {})
        store.add_edge("a", "b", EdgeType.TEMPORAL, 0.5)
        store.add_edge("a", "b", EdgeType.CROSS_BUCKET, 0.7)
        assert store.get_edge_count() == 2
        assert store.get_edge_count(EdgeType.TEMPORAL) == 1
        assert store.get_edge_count(EdgeType.CROSS_BUCKET) == 1
