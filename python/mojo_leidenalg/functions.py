"""Public convenience functions matching leidenalg's covered entry point."""

from __future__ import annotations

from .optimiser import Optimiser


def find_partition(graph, partition_type, initial_membership=None, weights=None,
                   n_iterations=2, max_comm_size=0, seed=None, **kwargs):
    """Detect communities using a supported partition class.

    `seed` is accepted for source compatibility. The current kernel traverses
    vertices in stable igraph order, making results deterministic by design.
    """
    partition = partition_type(graph, initial_membership=initial_membership,
                               weights=weights, **kwargs)
    optimiser = Optimiser()
    optimiser.max_comm_size = int(max_comm_size)
    if seed is not None:
        optimiser.set_rng_seed(seed)
    optimiser.optimise_partition(partition, n_iterations=n_iterations)
    return partition
