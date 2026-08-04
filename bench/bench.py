"""Measure the full public optimisation call against upstream leidenalg."""

from __future__ import annotations

import math
import platform
import random
import time

import igraph as ig
import leidenalg as upstream
import mojo_leidenalg as mla


def elapsed(fn, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def graph():
    ig.set_random_number_generator(random.Random(0))
    return ig.Graph.SBM(pref_matrix=[[0.045, 0.002], [0.002, 0.045]],
                        block_sizes=[750, 750], directed=False)


def main():
    data = graph()
    # Force shared-library loading and compilation outside measurement.
    mla.find_partition(data, mla.ModularityVertexPartition, seed=0)
    cases = [
        ("Modularity find_partition (1,500-node SBM)",
         lambda: mla.find_partition(data, mla.ModularityVertexPartition, seed=0),
         lambda: upstream.find_partition(data, upstream.ModularityVertexPartition, seed=0)),
        ("CPM find_partition (1,500-node SBM, gamma=0.01)",
         lambda: mla.find_partition(data, mla.CPMVertexPartition, seed=0, resolution_parameter=0.01),
         lambda: upstream.find_partition(data, upstream.CPMVertexPartition, seed=0, resolution_parameter=0.01)),
    ]
    print(f"machine: {platform.platform()} / {platform.processor() or 'unknown CPU'}")
    print(f"{'case':<53}{'mojo-leidenalg':>16}{'leidenalg':>14}{'ratio':>10}")
    print("-" * 93)
    for name, ours, theirs in cases:
        a, b = elapsed(ours), elapsed(theirs)
        state = "faster" if a < b else "slower"
        print(f"{name:<53}{a * 1e3:>14.1f}ms{b * 1e3:>12.1f}ms{b / a:>9.2f}x {state}")


if __name__ == "__main__":
    main()
