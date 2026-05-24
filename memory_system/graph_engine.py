"""NetworkX-based GraphStore implementation.

Stores the memory graph as a directed NetworkX graph with edge attributes.
Traversal follows only CROSS_BUCKET edges and uses best-single-path dedup.
"""

from __future__ import annotations

import uuid
from collections import deque
from typing import Any

import networkx as nx

from memory_system.interfaces import EdgeType, GraphStore
from memory_system.models import Edge, TraversalPath


class NetworkXGraphStore(GraphStore):
    """Directed-graph store backed by NetworkX.

    Nodes are stored as vertices with an ``attributes`` dict.
    Edges carry ``edge_type``, ``weight``, and ``id`` as edge attributes.
    """

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()

    # ── GraphStore interface ─────────────────────────────────────────────

    def add_node(self, node_id: str, attributes: dict[str, Any]) -> None:
        """Register a vertex."""
        if self._graph.has_node(node_id):
            raise ValueError(f"Node '{node_id}' already exists")
        self._graph.add_node(node_id, **attributes)

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: EdgeType,
        weight: float,
    ) -> str:
        """Create a directed edge."""
        if not self._graph.has_node(from_id):
            raise ValueError(f"Source node '{from_id}' does not exist")
        if not self._graph.has_node(to_id):
            raise ValueError(f"Target node '{to_id}' does not exist")

        edge_id = uuid.uuid4().hex[:12]
        self._graph.add_edge(
            from_id,
            to_id,
            id=edge_id,
            edge_type=edge_type,
            weight=weight,
        )
        return edge_id

    def traverse(
        self,
        start_nodes: list[str],
        max_hops: int,
        weight_threshold: float,
    ) -> list[TraversalPath]:
        """BFS traversal following CROSS_BUCKET edges only.

        Uses best-single-path dedup: when a node is reachable via multiple
        routes, only the path with the highest cumulative weight is kept.
        """
        if not start_nodes:
            return []

        # best_score[node_id] = highest cumulative weight seen so far
        best_score: dict[str, float] = {}
        # best_path[node_id] = the node_ids list for the best path
        best_path: dict[str, list[str]] = {}

        queue: deque[tuple[str, float, int, list[str]]] = deque()
        # (current_node_id, cumulative_weight, depth, path_so_far)

        for start in start_nodes:
            if not self._graph.has_node(start):
                continue
            best_score[start] = 1.0  # seed nodes start at weight 1.0
            best_path[start] = [start]
            queue.append((start, 1.0, 0, [start]))

        while queue:
            current, cum_weight, depth, path = queue.popleft()

            if depth >= max_hops:
                continue

            for _, neighbor, _key, data in self._graph.out_edges(current, data=True, keys=True):
                etype: EdgeType = data["edge_type"]
                eweight: float = data["weight"]

                # Only follow CROSS_BUCKET edges above the threshold
                if etype != EdgeType.CROSS_BUCKET:
                    continue
                if eweight < weight_threshold:
                    continue

                new_weight = cum_weight * eweight
                new_path = path + [neighbor]
                new_depth = depth + 1

                # Best-single-path dedup
                if neighbor not in best_score or new_weight > best_score[neighbor]:
                    best_score[neighbor] = new_weight
                    best_path[neighbor] = new_path
                    if new_depth < max_hops:
                        queue.append((neighbor, new_weight, new_depth, new_path))

        # Build traversal paths (exclude seeds themselves)
        results: list[TraversalPath] = []
        for node_id, score in best_score.items():
            if node_id in start_nodes:
                continue
            results.append(
                TraversalPath(
                    node_ids=list(best_path[node_id]),
                    total_weight=score,
                    hops=len(best_path[node_id]) - 1,
                )
            )

        return results

    def get_out_edges(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
    ) -> list[Edge]:
        """Return outgoing edges, optionally filtered by type."""
        if not self._graph.has_node(node_id):
            raise ValueError(f"Node '{node_id}' does not exist")

        edges: list[Edge] = []
        for _, target, _key, data in self._graph.out_edges(node_id, data=True, keys=True):
            etype: EdgeType = data["edge_type"]
            if edge_type is not None and etype != edge_type:
                continue
            edges.append(
                Edge(
                    id=data["id"],
                    source_id=node_id,
                    target_id=target,
                    edge_type=etype,
                    weight=data["weight"],
                    created_at=0.0,  # Not tracked in this implementation
                )
            )
        return edges

    def remove_edge(self, edge_id: str) -> None:
        """Remove an edge by its ID."""
        for u, v, key, data in list(self._graph.edges(data=True, keys=True)):
            if data.get("id") == edge_id:
                self._graph.remove_edge(u, v, key=key)
                return
        raise KeyError(f"Edge '{edge_id}' not found")

    def get_node_count(self) -> int:
        """Number of vertices."""
        return int(self._graph.number_of_nodes())

    def get_edge_count(self, edge_type: EdgeType | None = None) -> int:
        """Number of edges, optionally filtered."""
        if edge_type is None:
            return int(self._graph.number_of_edges())

        count = 0
        for _, _, _key, data in self._graph.edges(data=True, keys=True):
            if data.get("edge_type") == edge_type:
                count += 1
        return count
