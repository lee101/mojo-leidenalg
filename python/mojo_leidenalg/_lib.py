"""ctypes binding for the one-file Mojo community kernel."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-leidenalg.so")
I = ctypes.c_int64
F = ctypes.c_double
_SIGNATURES = {
    "mla_local_move": ([I] * 13 + [F, I, I], I),
    "mla_connected_refine": ([I] * 7, I),
    "mla_quality": ([I] * 9 + [F], F),
}
_loaded = None


def build() -> str:
    sources = [os.path.join(ROOT, "src", "leiden.mojo")]
    if os.path.exists(LIB) and os.path.getmtime(LIB) >= max(map(os.path.getmtime, sources)):
        return LIB
    script = os.path.join(ROOT, "build", "build.sh")
    proc = subprocess.run(["bash", script], cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (args, result) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes, fn.restype = args, result
    return _loaded


def _array(array, dtype, length, name, *, writable=True):
    """Validate an ndarray before its address crosses the untyped C ABI."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.dtype != dtype or array.ndim != 1 or not array.flags.c_contiguous:
        raise TypeError(f"{name} must be a one-dimensional C-contiguous {dtype} array")
    if len(array) != length:
        raise ValueError(f"{name} must contain exactly {length} elements")
    if writable and not array.flags.writeable:
        raise ValueError(f"{name} must be writable")
    if not array.ctypes.data:
        raise ValueError(f"{name} must have a non-null data pointer")
    return array


def _csr(row, col, weight, membership):
    n = len(membership)
    _array(row, np.dtype(np.int64), n + 1, "row", writable=False)
    if row[0] != 0 or np.any(row[1:] < row[:-1]):
        raise ValueError("row must be nondecreasing and start at zero")
    edges = int(row[-1])
    _array(col, np.dtype(np.int64), edges, "col", writable=False)
    _array(weight, np.dtype(np.float64), edges, "weight", writable=False)
    _array(membership, np.dtype(np.int64), n, "membership")
    if np.any(col < 0) or np.any(col >= n):
        raise ValueError("col contains an invalid vertex index")
    if not np.all(np.isfinite(weight)):
        raise ValueError("weight must be finite")
    if np.any(membership < 0) or np.any(membership >= n):
        raise ValueError("membership contains an invalid community index")
    return n


def addr(array) -> int:
    """Return a validated array address for a ctypes call.

    Callers keep the array references alive for the complete native call.
    """
    return int(array.ctypes.data)


def local_move(row, col, weight, membership, degree, total, size, marks, touched,
               kin, order, mode, resolution, max_passes, max_comm_size):
    n = _csr(row, col, weight, membership)
    _array(degree, np.dtype(np.float64), n, "degree")
    _array(total, np.dtype(np.float64), n, "total")
    _array(size, np.dtype(np.int64), n, "size")
    _array(marks, np.dtype(np.int64), n, "marks")
    _array(touched, np.dtype(np.int64), n, "touched")
    _array(kin, np.dtype(np.float64), n, "kin")
    _array(order, np.dtype(np.int64), n, "order", writable=False)
    if np.any(order < 0) or np.any(order >= n) or len(np.unique(order)) != n:
        raise ValueError("order must be a permutation of the vertex indices")
    return lib().mla_local_move(
        addr(row), addr(col), addr(weight), addr(membership), addr(degree), addr(total),
        addr(size), addr(marks), addr(touched), addr(kin), addr(order), n, int(mode),
        float(resolution), int(max_passes), int(max_comm_size),
    )


def connected_refine(row, col, membership, result, seen, stack):
    n = len(membership)
    _array(row, np.dtype(np.int64), n + 1, "row", writable=False)
    edges = int(row[-1])
    _array(col, np.dtype(np.int64), edges, "col", writable=False)
    _array(membership, np.dtype(np.int64), n, "membership")
    _array(result, np.dtype(np.int64), n, "result")
    _array(seen, np.dtype(np.int64), n, "seen")
    _array(stack, np.dtype(np.int64), n, "stack")
    if row[0] != 0 or np.any(row[1:] < row[:-1]) or np.any(col < 0) or np.any(col >= n):
        raise ValueError("invalid CSR graph")
    return lib().mla_connected_refine(addr(row), addr(col), addr(membership), addr(result),
                                      addr(seen), addr(stack), n)


def quality(row, col, weight, membership, degree, total, size, mode, resolution):
    n = _csr(row, col, weight, membership)
    _array(degree, np.dtype(np.float64), n, "degree")
    _array(total, np.dtype(np.float64), n, "total")
    _array(size, np.dtype(np.int64), n, "size")
    return lib().mla_quality(addr(row), addr(col), addr(weight), addr(membership), addr(degree),
                             addr(total), addr(size), n, int(mode), float(resolution))
