"""Upstream-shaped vertex partitions backed by a CSR graph representation."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ._lib import quality


def _compact(values, n: int) -> np.ndarray:
    if values is None:
        return np.arange(n, dtype=np.int64)
    values = list(values)
    if len(values) != n:
        raise ValueError("initial_membership must have one entry per vertex")
    labels = {}
    return np.asarray([labels.setdefault(x, len(labels)) for x in values], dtype=np.int64)


def _float64_weights(weights):
    raw = np.asarray(weights)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise TypeError("weights must be a one-dimensional array of real numbers")
    if raw.dtype.kind == "f" and raw.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError("weights with precision wider than float64 are not supported")
    if raw.dtype.kind in "iu" and raw.size and np.max(np.abs(raw.astype(object))) > 2**53:
        raise ValueError("integer weights must be exactly representable as float64")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("weights must be finite")
    return result


def _csr(graph, weights):
    if graph.is_directed():
        raise NotImplementedError("mojo_leidenalg currently supports undirected igraph graphs")
    n = graph.vcount()
    edges = np.asarray(graph.get_edgelist(), dtype=np.int64)
    if weights is None:
        edge_weight = np.ones(len(edges), dtype=np.float64)
    elif isinstance(weights, str):
        edge_weight = _float64_weights(graph.es[weights])
    else:
        edge_weight = _float64_weights(weights)
    if len(edge_weight) != len(edges):
        raise ValueError("weights must have one entry per edge")
    if not len(edges):
        return (np.zeros(n + 1, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64))
    endpoints = np.concatenate((edges[:, 0], edges[:, 1]))
    neighbours = np.concatenate((edges[:, 1], edges[:, 0]))
    values = np.concatenate((edge_weight, edge_weight))
    degree = np.bincount(endpoints, minlength=n)
    row = np.empty(n + 1, dtype=np.int64)
    row[0] = 0
    np.cumsum(degree, out=row[1:])
    order = np.argsort(endpoints, kind="stable")
    return row, neighbours[order], values[order]


class VertexPartition:
    """Common Python facade for the supported mutable vertex partitions."""

    _mode = 0

    def __init__(self, graph, initial_membership=None, weights=None, resolution_parameter=1.0, **kwargs):
        if kwargs:
            unknown = ", ".join(kwargs)
            raise TypeError(f"unsupported partition option(s): {unknown}")
        self.graph = graph
        self.n = int(graph.vcount())
        self.resolution_parameter = float(resolution_parameter)
        if not np.isfinite(self.resolution_parameter):
            raise ValueError("resolution_parameter must be finite")
        self._edges = np.asarray(graph.get_edgelist(), dtype=np.int64)
        self._row, self._col, self._weight = _csr(graph, weights)
        self._edge_weight = (np.ones(len(self._edges), dtype=np.float64) if weights is None
                             else (_float64_weights(graph.es[weights]) if isinstance(weights, str)
                                   else _float64_weights(weights)))
        self._membership = _compact(initial_membership, self.n)

    @property
    def membership(self):
        return self._membership.tolist()

    @membership.setter
    def membership(self, value):
        self._membership = _compact(value, self.n)

    def set_membership(self, membership):
        self.membership = membership

    def __len__(self):
        return len(self.sizes())

    def __iter__(self):
        return iter([self[i] for i in range(len(self))])

    def __getitem__(self, community):
        return np.flatnonzero(self._membership == community).tolist()

    def sizes(self):
        return [int(x) for x in np.bincount(self._membership)]

    def size(self, community):
        return len(self[community])

    def _scratch(self):
        return (np.empty(self.n, dtype=np.float64), np.empty(self.n, dtype=np.float64),
                np.empty(self.n, dtype=np.int64))

    def quality(self, resolution_parameter=None):
        resolution = self.resolution_parameter if resolution_parameter is None else float(resolution_parameter)
        if not np.isfinite(resolution):
            raise ValueError("resolution_parameter must be finite")
        degree, total, size = self._scratch()
        return float(quality(self._row, self._col, self._weight, self._membership,
                             degree, total, size, self._mode, resolution))

    @property
    def q(self):
        return self.quality()

    @property
    def modularity(self):
        # This deliberately follows leidenalg's property: it asks igraph for
        # the graph modularity, rather than reusing a partition-only weight
        # sequence passed to the constructor.
        return float(self.graph.modularity(self.membership))

    def diff_move(self, v, new_comm):
        v, new_comm = int(v), int(new_comm)
        if not 0 <= v < self.n or not 0 <= new_comm < self.n:
            raise ValueError("vertex and community must be valid indices")
        before = self.quality()
        saved = self._membership[v]
        self._membership[v] = new_comm
        after = self.quality()
        self._membership[v] = saved
        return after - before

    def move_node(self, v, new_comm):
        self._membership[int(v)] = int(new_comm)
        self.renumber_communities()

    def renumber_communities(self):
        self._membership = _compact(self._membership, self.n)

    def summary(self):
        return f"{self.__class__.__name__} with {len(self)} communities"


class MutableVertexPartition(VertexPartition):
    pass


class ModularityVertexPartition(MutableVertexPartition):
    _mode = 0

    def __init__(self, graph, initial_membership=None, weights=None):
        super().__init__(graph, initial_membership, weights, resolution_parameter=1.0)


class CPMVertexPartition(MutableVertexPartition):
    _mode = 1

    def __init__(self, graph, initial_membership=None, weights=None, node_sizes=None,
                 resolution_parameter=1.0, correct_self_loops=None):
        if node_sizes is not None:
            raise NotImplementedError("CPM node_sizes are not yet supported")
        super().__init__(graph, initial_membership, weights, resolution_parameter)


class RBConfigurationVertexPartition(MutableVertexPartition):
    _mode = 2

    def __init__(self, graph, initial_membership=None, weights=None, resolution_parameter=1.0):
        super().__init__(graph, initial_membership, weights, resolution_parameter)
