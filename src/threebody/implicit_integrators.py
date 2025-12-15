"""Implicit integrators (stubs).

You requested a separate implicit integrator module and to expose it via
`threebody.integrators`. This file is intentionally minimal: it provides the
expected function signatures, but does not implement a nonlinear solver yet.

When you’re ready, we can implement a robust Newton / quasi-Newton solve with
finite-difference Jacobians (still NumPy-only) and good convergence diagnostics.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

VectorField = Callable[[float, np.ndarray], np.ndarray]


def implicit_midpoint_step(
    f: VectorField,
    t: float,
    y: np.ndarray,
    h: float,
    *,
    max_iter: int = 20,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, int]:
    """One step of the implicit midpoint rule.

    Method:
        y_{n+1} = y_n + h * f(t_n + h/2, (y_n + y_{n+1})/2)

    Returns:
        y_next: the (approximate) implicit solution at t+h
        nfev: number of RHS evaluations

    Notes:
        This requires solving a nonlinear system. Implementation is pending.
    """
    raise NotImplementedError(
        "Implicit midpoint requires a nonlinear solve; not implemented yet."
    )
