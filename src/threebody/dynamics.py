"""Three-body problem dynamics, using Newton's law of gravitation F=G m1 m2 / r^2.

State vector convention (flat np array, length 18):
    y = [x1,y1,z1,x2,y2,z2,x3,y3,z3,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3]

All bodies have mass m_i and move in three-dimensional space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

_STATE_DIM = 18
_N_BODIES = 3
_DIM = 3

@dataclass(frozen=True)
class DynamicsParams:
    """Parameters for the equal-mass planar three-body problem."""

    G: float = 1.0
    softening: float = 0.0
    masses: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=float)

def split_state(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to split flat state into positions and velocities.

    Args:
        y: Array of shape (18,) representing the flat state.
    Returns:
        r: 3D positions of each body, shape (3,3)
        v: 3D velocities of each body, shape (3,3)
    """
    y = np.asarray(y, dtype=float)
    if y.shape != (_STATE_DIM,):
        raise ValueError(f"Expected y.shape == ({_STATE_DIM},), got {y.shape}")

    r = y[: 3 * _N_BODIES].reshape(_N_BODIES, _DIM)
    v = y[3 * _N_BODIES :].reshape(_N_BODIES, _DIM)
    return r, v


def pack_state(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Helper function to pack position and velocity matrices into the flat 18D state vector."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    if r.shape != (_N_BODIES, _DIM):
        raise ValueError(f"Expected r.shape == ({_N_BODIES},{_DIM}), got {r.shape}")
    if v.shape != (_N_BODIES, _DIM):
        raise ValueError(f"Expected v.shape == ({_N_BODIES},{_DIM}), got {v.shape}")

    return np.concatenate([r.reshape(-1), v.reshape(-1)])


def accelerations(r: np.ndarray, *, params: DynamicsParams | None = None) -> np.ndarray:
    """Compute gravitational accelerations for each body.
    G=1 by default.

    Args:
        r: 3D positions of each body, shape (3,3)
        params: Dynamics parameters (masses, G, optional softening).

    Returns:
        a: Accelerations, shape (3,3)
    Notes:
        The acceleration on body i is:
            a_i = G * sum_{j!=i} m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)

        softening=0.0 gives the true Newtonian force.
    """
    if params is None:
        params = DynamicsParams()

    # Enforce array shapes
    masses = np.asarray(params.masses, dtype=float)
    r = np.asarray(r, dtype=float)
    if masses.shape != (_N_BODIES,):
        raise ValueError(f"Expected masses.shape == ({_N_BODIES},), got {masses.shape}")
    if r.shape != (_N_BODIES, _DIM):
        raise ValueError(f"Expected r.shape == ({_N_BODIES},{_DIM}), got {r.shape}")

    # Initialize accelerations
    a = np.zeros_like(r)
    eps2 = float(params.softening) ** 2
    G = float(params.G)

    # Explicit pairwise sum; small N keeps this clear and reliable.
    for i in range(_N_BODIES):
        for j in range(_N_BODIES):
            if j == i:
                continue
            dr = r[j] - r[i]
            dist2 = float(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]) + eps2
            if dist2 <= 0.0:
                raise FloatingPointError("Non-positive pair distance squared encountered")
            inv_dist3 = dist2 ** (-1.5)
            a[i] += G * masses[j] * dr * inv_dist3

    return a


def rhs(t: float, y: np.ndarray, *, params: DynamicsParams | None = None) -> np.ndarray:
    """Right-hand side for the ODE y' = f(t, y)."""
    r, v = split_state(y)
    a = accelerations(r, params=params)
    return pack_state(v, a)


def energy(y: np.ndarray, *, params: DynamicsParams | None = None) -> float:
    """Compute total (kinetic + potential) energy for the state.

    Args:
        y: Flat state, shape (18,)
        params: Dynamics parameters (masses, G, optional softening).

    Returns:
        Total energy as a float.

    Notes:
        For masses m_i, the total energy is:
            T = 1/2 * sum_i m_i * ||v_i||^2
            U = -G * sum_{i<j} m_i * m_j / sqrt(||r_i-r_j||^2 + eps^2)
    """
    if params is None:
        params = DynamicsParams()

    r, v = split_state(y)
    G = float(params.G)
    eps2 = float(params.softening) ** 2
    masses = np.asarray(params.masses, dtype=float)

    kinetic = 0.5 * float(np.sum(masses * np.sum(v * v, axis=1)))

    potential = 0.0
    for i in range(_N_BODIES):
        for j in range(i + 1, _N_BODIES):
            dr = r[j] - r[i]
            dist2 = float(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]) + eps2
            if dist2 <= 0.0:
                raise FloatingPointError("Non-positive pair distance squared encountered")
            potential += -G * masses[i] * masses[j] / np.sqrt(dist2)

    return kinetic + float(potential)
