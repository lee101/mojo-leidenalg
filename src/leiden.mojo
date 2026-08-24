"""Sparse, in-place community-objective kernels.

CSR stores each undirected edge twice.  Community ids are always in [0, n),
which lets the caller provide O(n) scratch rather than allocating in Mojo.
"""

from max.algorithm import parallelize
from std.runtime import initialize_runtime
from std.sys import simd_width_of


comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime DEGREE_PARALLEL_EDGES = 1_000_000
comptime DEGREE_PARALLEL_NODES = 32_768
comptime DEGREE_CHUNK = 1_024
comptime DEGREE_WORKERS = 16


def fp(address: Int) -> FPtr:
    return FPtr(unsafe_from_address=address)


def ip(address: Int) -> IPtr:
    return IPtr(unsafe_from_address=address)


def degree_range(row: IPtr, weight: FPtr, degree: FPtr, start: Int, stop: Int):
    for i in range(start, stop):
        var d = 0.0
        var e = row[i]
        var end = row[i + 1]
        while e < end and e % W != 0:
            d += weight[e]
            e += 1
        while e + W <= end:
            d += weight.load[width=W](e).reduce_add()
            e += W
        while e < end:
            d += weight[e]
            e += 1
        degree[i] = d


def initialise(
    row: IPtr, col: IPtr, weight: FPtr, membership: IPtr, degree: FPtr,
    total: FPtr, size: IPtr, n: Int
):
    var c = 0
    while c + W <= n:
        total.store(c, SIMD[DType.float64, W](0.0))
        size.store(c, SIMD[DType.int, W](0))
        c += W
    while c < n:
        total[c] = 0.0
        size[c] = 0
        c += 1
    if n >= DEGREE_PARALLEL_NODES and row[n] >= DEGREE_PARALLEL_EDGES:
        initialize_runtime()
        var chunks = (n + DEGREE_CHUNK - 1) // DEGREE_CHUNK
        def work(chunk: Int) capturing:
            var start = chunk * DEGREE_CHUNK
            degree_range(row, weight, degree, start, min(start + DEGREE_CHUNK, n))
        parallelize[work](chunks, min(chunks, DEGREE_WORKERS))
    else:
        degree_range(row, weight, degree, 0, n)
    for i in range(n):
        var d = degree[i]
        var c = membership[i]
        total[c] += d
        size[c] += 1


def local_move(
    row: IPtr, col: IPtr, weight: FPtr, membership: IPtr, degree: FPtr,
    total: FPtr, size: IPtr, marks: IPtr, touched: IPtr, kin: FPtr,
    order: IPtr, n: Int, mode: Int, resolution: Float64, max_passes: Int, max_comm_size: Int
) -> Int:
    """Greedy local moves, aggregating adjacent-community weights in O(E)."""
    initialise(row, col, weight, membership, degree, total, size, n)
    for c in range(n):
        marks[c] = 0
    var moved_total = 0
    var m2 = 0.0
    var i = 0
    while i + W <= n:
        m2 += degree.load[width=W](i).reduce_add()
        i += W
    while i < n:
        m2 += degree[i]
        i += 1
    if m2 == 0.0:
        return 0
    for pass_id in range(max_passes):
        var moved = 0
        for position in range(n):
            var i = order[position]
            var old = membership[i]
            var stamp = i + 1
            var ntouched = 0
            for e in range(row[i], row[i + 1]):
                var c = membership[col[e]]
                if marks[c] != stamp:
                    marks[c] = stamp
                    touched[ntouched] = c
                    kin[c] = 0.0
                    ntouched += 1
                kin[c] += weight[e]
            var old_kin = 0.0
            if marks[old] == stamp:
                old_kin = kin[old]
            var best = old
            var best_gain = 0.0
            var d = degree[i]
            for q in range(ntouched):
                var candidate = touched[q]
                if candidate == old:
                    continue
                if max_comm_size > 0 and size[candidate] >= max_comm_size:
                    continue
                var gain = 0.0
                if mode == 1:
                    gain = kin[candidate] - old_kin - resolution * Float64(size[candidate] - (size[old] - 1))
                else:
                    gain = kin[candidate] - old_kin - resolution * d * (total[candidate] - (total[old] - d)) / m2
                if gain > best_gain:
                    best_gain = gain
                    best = candidate
            if best != old:
                membership[i] = best
                total[old] -= d
                total[best] += d
                size[old] -= 1
                size[best] += 1
                moved += 1
        moved_total += moved
        if moved == 0:
            break
    return moved_total


def connected_refine(
    row: IPtr, col: IPtr, membership: IPtr, result: IPtr, seen: IPtr,
    stack: IPtr, n: Int
) -> Int:
    """Split each community into graph-connected components.

    This is the refinement invariant: an output community never consists of
    disconnected pieces. `result` is separate so labels being assigned cannot
    change the membership predicate while a component is traversed.
    """
    for i in range(n):
        seen[i] = 0
    var next_label = 0
    for root in range(n):
        if seen[root] != 0:
            continue
        var original = membership[root]
        var top = 0
        stack[top] = root
        top += 1
        seen[root] = 1
        while top > 0:
            top -= 1
            var node = stack[top]
            result[node] = next_label
            for e in range(row[node], row[node + 1]):
                var other = col[e]
                if seen[other] == 0 and membership[other] == original:
                    seen[other] = 1
                    stack[top] = other
                    top += 1
        next_label += 1
    return next_label


def quality(
    row: IPtr, col: IPtr, weight: FPtr, membership: IPtr, degree: FPtr,
    total: FPtr, size: IPtr, n: Int, mode: Int, resolution: Float64
) -> Float64:
    initialise(row, col, weight, membership, degree, total, size, n)
    var m2 = 0.0
    var internal = 0.0
    var i = 0
    while i + W <= n:
        m2 += degree.load[width=W](i).reduce_add()
        i += W
    while i < n:
        m2 += degree[i]
        i += 1
    for i in range(n):
        for e in range(row[i], row[i + 1]):
            if membership[i] == membership[col[e]]:
                internal += weight[e]
    if mode == 1:
        var answer = internal
        for c in range(n):
            answer -= resolution * Float64(size[c] * (size[c] - 1))
        return answer
    if m2 == 0.0:
        return 0.0
    var expected = 0.0
    var c = 0
    while c + W <= n:
        var values = total.load[width=W](c)
        expected += (values * values).reduce_add() / m2
        c += W
    while c < n:
        expected += total[c] * total[c] / m2
        c += 1
    if mode == 0:
        return (internal - resolution * expected) / m2
    return internal - resolution * expected


@export("mla_local_move")
def mla_local_move(
    row_addr: Int, col_addr: Int, weight_addr: Int, membership_addr: Int,
    degree_addr: Int, total_addr: Int, size_addr: Int, marks_addr: Int,
    touched_addr: Int, kin_addr: Int, order_addr: Int, n: Int, mode: Int, resolution: Float64,
    max_passes: Int, max_comm_size: Int
) abi("C") -> Int:
    return local_move(ip(row_addr), ip(col_addr), fp(weight_addr), ip(membership_addr),
        fp(degree_addr), fp(total_addr), ip(size_addr), ip(marks_addr), ip(touched_addr),
        fp(kin_addr), ip(order_addr), n, mode, resolution, max_passes, max_comm_size)


@export("mla_connected_refine")
def mla_connected_refine(
    row_addr: Int, col_addr: Int, membership_addr: Int, result_addr: Int,
    seen_addr: Int, stack_addr: Int, n: Int
) abi("C") -> Int:
    return connected_refine(ip(row_addr), ip(col_addr), ip(membership_addr),
        ip(result_addr), ip(seen_addr), ip(stack_addr), n)


@export("mla_quality")
def mla_quality(
    row_addr: Int, col_addr: Int, weight_addr: Int, membership_addr: Int,
    degree_addr: Int, total_addr: Int, size_addr: Int, n: Int, mode: Int,
    resolution: Float64
) abi("C") -> Float64:
    return quality(ip(row_addr), ip(col_addr), fp(weight_addr), ip(membership_addr),
        fp(degree_addr), fp(total_addr), ip(size_addr), n, mode, resolution)
