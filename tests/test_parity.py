"""Objective values and public behaviour are checked against leidenalg."""

import igraph as ig
import leidenalg as upstream
import numpy as np
import pytest

import mojo_leidenalg as mla


@pytest.fixture(scope="module")
def weighted_graph():
    return ig.Graph(
        n=8,
        edges=[(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
               (2, 3), (5, 6), (6, 7)],
    )


@pytest.mark.parametrize(
    ("ours_type", "upstream_type", "kwargs"),
    [
        (mla.ModularityVertexPartition, upstream.ModularityVertexPartition, {}),
        (mla.CPMVertexPartition, upstream.CPMVertexPartition, {"resolution_parameter": 0.35}),
        (mla.RBConfigurationVertexPartition, upstream.RBConfigurationVertexPartition,
         {"resolution_parameter": 0.7}),
    ],
)
def test_quality_matches_upstream_for_weighted_partitions(weighted_graph, ours_type, upstream_type, kwargs):
    weights = [1.0, 2.0, 1.5, 3.0, 1.0, 2.5, 0.4, 1.2, 0.8]
    memberships = ([0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 1, 1, 1, 2, 2],
                   [3, 3, 0, 0, 2, 2, 1, 1])
    for membership in memberships:
        ours = ours_type(weighted_graph, initial_membership=membership, weights=weights, **kwargs)
        theirs = upstream_type(weighted_graph, initial_membership=membership, weights=weights, **kwargs)
        assert ours.quality() == pytest.approx(theirs.quality(), abs=1e-12)
        assert ours.modularity == pytest.approx(theirs.modularity, abs=1e-12)


def test_move_delta_matches_upstream(weighted_graph):
    weights = [1.0, 2.0, 1.5, 3.0, 1.0, 2.5, 0.4, 1.2, 0.8]
    membership = [0, 0, 0, 1, 1, 1, 2, 2]
    for ours_type, upstream_type, kwargs in [
        (mla.ModularityVertexPartition, upstream.ModularityVertexPartition, {}),
        (mla.CPMVertexPartition, upstream.CPMVertexPartition, {"resolution_parameter": 0.3}),
        (mla.RBConfigurationVertexPartition, upstream.RBConfigurationVertexPartition,
         {"resolution_parameter": 0.8}),
    ]:
        ours = ours_type(weighted_graph, membership, weights, **kwargs)
        theirs = upstream_type(weighted_graph, membership, weights, **kwargs)
        assert ours.diff_move(2, 1) == pytest.approx(theirs.diff_move(2, 1), abs=1e-12)


def test_quality_simd_weight_scan_with_tail_matches_upstream():
    graph = ig.Graph.Star(6, mode="undirected")
    weights = [0.5, 1.5, 2.5, 3.5, 4.5]
    membership = [0, 0, 0, 1, 1, 1]
    ours = mla.ModularityVertexPartition(graph, membership, weights)
    theirs = upstream.ModularityVertexPartition(graph, membership, weights)
    assert ours.quality() == pytest.approx(theirs.quality(), abs=1e-12)


def test_find_partition_improves_and_returns_connected_communities():
    graph = ig.Graph.Famous("Zachary")
    start = mla.ModularityVertexPartition(graph)
    ours = mla.find_partition(graph, mla.ModularityVertexPartition, seed=0, n_iterations=2)
    theirs = upstream.find_partition(graph, upstream.ModularityVertexPartition, seed=0)
    assert ours.quality() > start.quality()
    # Heuristics may choose a different local optimum, but must be competitive.
    assert ours.quality() >= theirs.quality() * 0.85
    for community in ours:
        if len(community) > 1:
            assert graph.induced_subgraph(community).is_connected()


def test_cpm_resolution_and_maximum_community_size():
    graph = ig.Graph.Full(10)
    part = mla.find_partition(graph, mla.CPMVertexPartition,
                              resolution_parameter=0.01, max_comm_size=3, seed=4)
    assert max(part.sizes()) <= 3
    assert part.quality() > 0


def test_public_partition_shape_and_membership_setter(weighted_graph):
    part = mla.ModularityVertexPartition(weighted_graph, [10, 10, 7, 7, 7, 2, 2, 2])
    assert len(part) == 3
    assert sorted(sum(list(part), [])) == list(range(weighted_graph.vcount()))
    part.set_membership([0] * weighted_graph.vcount())
    assert part.sizes() == [weighted_graph.vcount()]


def test_public_usage_example_and_optimiser_methods():
    graph = ig.Graph.Famous("Zachary")
    partition = mla.find_partition(graph, mla.ModularityVertexPartition, seed=0)
    assert len(partition.membership) == graph.vcount()
    assert partition.q == pytest.approx(partition.quality())
    assert partition.modularity == pytest.approx(graph.modularity(partition.membership))

    start = mla.ModularityVertexPartition(graph)
    optimiser = mla.Optimiser()
    assert optimiser.move_nodes(start) >= 0
    assert optimiser.merge_nodes(start) >= 0
    assert optimiser.refine_partition(start) is start
    start.move_node(0, 1)
    assert sum(start.sizes()) == graph.vcount()


def test_rejects_unsafe_labels_and_lossy_weights():
    graph = ig.Graph(n=2, edges=[(0, 1)])
    partition = mla.ModularityVertexPartition(graph)
    with pytest.raises(ValueError, match="valid indices"):
        partition.diff_move(0, 2)
    with pytest.raises(TypeError, match="wider than float64"):
        mla.ModularityVertexPartition(graph, weights=np.array([1], dtype=np.longdouble))
    with pytest.raises(ValueError, match="exactly representable"):
        mla.ModularityVertexPartition(graph, weights=[2**53 + 1])


def test_self_loops_remain_correct_after_contraction():
    graph = ig.Graph(n=4, edges=[(0, 0), (0, 1), (1, 2), (2, 3), (3, 3)])
    weights = [0.5, 1.0, 2.0, 1.0, 0.25]
    ours = mla.find_partition(graph, mla.ModularityVertexPartition, weights=weights, seed=1,
                              n_iterations=2)
    assert np.isfinite(ours.quality())


def test_directed_graph_is_rejected_explicitly():
    with pytest.raises(NotImplementedError, match="undirected"):
        mla.ModularityVertexPartition(ig.Graph(n=2, edges=[(0, 1)], directed=True))
