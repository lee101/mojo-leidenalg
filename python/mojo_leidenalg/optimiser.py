"""The optimisation loop: local moves followed by connectivity refinement."""

from __future__ import annotations

import numpy as np
import igraph as ig

from ._lib import connected_refine, local_move


class Optimiser:
    """Optimise supported partitions with deterministic Leiden-style passes.

    Each round greedily improves the selected objective, splits disconnected
    pieces, then moves again.  The split is the Leiden connectivity invariant.
    """

    def __init__(self):
        self.max_comm_size = 0
        self._seed = 0
        self._calls = 0

    def set_rng_seed(self, seed):
        self._seed = int(seed)

    def _move(self, partition):
        n = partition.n
        degree = np.empty(n, dtype=np.float64)
        total = np.empty(n, dtype=np.float64)
        size = np.empty(n, dtype=np.int64)
        marks = np.empty(n, dtype=np.int64)
        touched = np.empty(n, dtype=np.int64)
        kin = np.empty(n, dtype=np.float64)
        order = np.random.default_rng(self._seed + self._calls).permutation(n).astype(np.int64)
        self._calls += 1
        return int(local_move(
            partition._row, partition._col, partition._weight, partition._membership,
            degree, total, size, marks, touched, kin, order, partition._mode,
            partition.resolution_parameter, 32, int(self.max_comm_size),
        ))

    def move_nodes(self, partition, is_membership_fixed=None):
        if is_membership_fixed is not None and any(is_membership_fixed):
            raise NotImplementedError("fixed memberships are not yet supported")
        old = partition.quality()
        self._move(partition)
        partition.renumber_communities()
        return partition.quality() - old

    def merge_nodes(self, partition, is_membership_fixed=None):
        return self.move_nodes(partition, is_membership_fixed)

    def refine_partition(self, partition, is_membership_fixed=None):
        if is_membership_fixed is not None and any(is_membership_fixed):
            raise NotImplementedError("fixed memberships are not yet supported")
        n = partition.n
        result = np.empty(n, dtype=np.int64)
        seen = np.empty(n, dtype=np.int64)
        stack = np.empty(n, dtype=np.int64)
        connected_refine(partition._row, partition._col, partition._membership,
                         result, seen, stack)
        partition._membership = result
        return partition

    @staticmethod
    def _aggregate(partition):
        """Contract a partition, preserving each undirected CSR edge once."""
        membership = partition._membership
        n_communities = len(partition)
        edges = partition._edges
        left = membership[edges[:, 0]]
        right = membership[edges[:, 1]]
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        keys = low * n_communities + high
        unique, inverse = np.unique(keys, return_inverse=True)
        weights = np.bincount(inverse, weights=partition._edge_weight)
        edges = np.column_stack((unique // n_communities, unique % n_communities)).tolist()
        graph = ig.Graph(n=n_communities, edges=edges, directed=False)
        cls = partition.__class__
        if partition._mode == 0:
            return cls(graph, weights=weights)
        return cls(graph, weights=weights, resolution_parameter=partition.resolution_parameter)

    def optimise_partition(self, partition, n_iterations=2, is_membership_fixed=None):
        if is_membership_fixed is not None and any(is_membership_fixed):
            raise NotImplementedError("fixed memberships are not yet supported")
        old = partition.quality()
        current = partition
        # `base_map` maps original vertices to the vertices of `current`.
        base_map = np.arange(partition.n, dtype=np.int64)
        rounds = 0
        while n_iterations < 0 or rounds < n_iterations:
            moved = self._move(current)
            self.refine_partition(current)
            moved += self._move(current)
            self.refine_partition(current)
            current.renumber_communities()
            base_map = current._membership[base_map]
            rounds += 1
            # Aggregated vertices need their original node weights to enforce
            # this constraint. Keep the guarantee exact until that extension.
            if moved == 0 or len(current) == current.n or self.max_comm_size > 0:
                break
            current = self._aggregate(current)
        partition._membership = base_map
        partition.renumber_communities()
        return partition.quality() - old
