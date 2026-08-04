"""Leiden-style community detection accelerated by Mojo sparse kernels."""

from .partition import (
    CPMVertexPartition,
    ModularityVertexPartition,
    RBConfigurationVertexPartition,
    MutableVertexPartition,
    VertexPartition,
)
from .optimiser import Optimiser
from .functions import find_partition

__version__ = "0.1.0"

__all__ = [
    "CPMVertexPartition", "ModularityVertexPartition",
    "RBConfigurationVertexPartition", "MutableVertexPartition", "VertexPartition",
    "Optimiser", "find_partition",
]
