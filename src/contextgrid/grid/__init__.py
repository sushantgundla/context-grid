"""The experiment matrix and the runner that walks it."""

from __future__ import annotations

from contextgrid.grid.matrix import (
    AXIS_ORDER,
    Matrix,
    MatrixError,
    SweepMode,
    canonicalise,
    deduplicate,
    matrix,
)
from contextgrid.grid.runner import Budget, Runner, estimate_cost

__all__ = [
    "AXIS_ORDER",
    "Budget",
    "Matrix",
    "MatrixError",
    "Runner",
    "SweepMode",
    "canonicalise",
    "deduplicate",
    "estimate_cost",
    "matrix",
]
