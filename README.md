# mojo-leidenalg

`mojo-leidenalg` is a small, experimental Mojo implementation of the sparse
community-objective kernels used by [leidenalg](https://github.com/vtraag/leidenalg).
It accepts undirected `igraph.Graph` instances. It is not a drop-in replacement
for upstream `leidenalg`.

## Install and use

This repository is currently run from a checkout with Pixi. The example below
is exercised by the test suite.

```bash
pixi install
pixi run build
pixi run test
```

```python
import igraph as ig
import mojo_leidenalg as leidenalg

graph = ig.Graph.Famous("Zachary")
partition = leidenalg.find_partition(
    graph, leidenalg.ModularityVertexPartition, seed=0,
)
print(partition.membership)
print(partition.quality())
```

## Scope

Covered and tested:

| API | Coverage |
| --- | --- |
| `find_partition` | Modularity, CPM, and RB Configuration partitions on weighted undirected graphs |
| `Optimiser.optimise_partition`, `move_nodes`, `merge_nodes`, `refine_partition` | greedy moves and connected-component refinement |
| `ModularityVertexPartition`, `CPMVertexPartition`, `RBConfigurationVertexPartition` | `membership`, `quality`, `q`, `modularity`, `sizes`, `diff_move`, and `move_node` |

The implementation accepts finite float64-compatible edge weights. It rejects
directed graphs, non-finite weights, lossy wider-than-float64 weights, and
unsafe community labels before they reach native code.

Not covered: directed, multiplex, or bipartite optimisation; fixed
memberships; CPM `node_sizes`; resolution profiles; and the remaining upstream
partition families. Unsupported cases raise `NotImplementedError` where there
is an explicit option; this project does not approximate them.

The optimiser is deterministic for a supplied seed, but it is a heuristic and
need not return upstream's exact partition. `optimise_partition` refines after
each move phase, so its returned communities are connected.

## Benchmark

Run it yourself with `pixi run bench`. The following is the actual output from
that command on this machine (Linux 6.8.0-136-generic, x86_64), using the best
of three complete public calls. Compilation and initial loading are excluded.

| case | mojo-leidenalg | leidenalg | result |
| --- | ---: | ---: | --- |
| Modularity `find_partition` (1,500-node SBM) | 45.1 ms | 34.2 ms | 0.76x, slower |
| CPM `find_partition` (1,500-node SBM, gamma=0.01) | 40.1 ms | 30.8 ms | 0.77x, slower |

These figures are a point-in-time measurement, not a general performance
claim. Upstream `leidenalg` is substantially more complete.

## How it works

Python converts the undirected graph into a contiguous CSR layout and retains
every NumPy buffer for the native call. The ctypes boundary validates lengths,
strides, dtypes, index ranges, writeability, and non-null addresses. Mojo then
uses caller-provided scratch buffers for degree/community totals and performs
greedy local moves; Python contracts the graph between rounds and Mojo splits
disconnected components.

## License

MIT. See [LICENSE](LICENSE).
