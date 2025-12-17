"""Simulation driver for ODE IVPs with adaptive stepping.

This module is integration- and physics-agnostic:
- You provide the vector field f(t, y)
- You provide a numerical stepper (e.g., DOPRI5) that returns (y_trial, err, nfev)
- You provide an adaptive controller that decides accept/reject and next h

Recorded diagnostics:
- time points
- states
- accepted step sizes
- total RHS evaluation count (includes rejected steps)
- optional energy + drift from initial value
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence, Tuple

import numpy as np


VectorField = Callable[[float, np.ndarray], np.ndarray]
Stepper = Callable[[VectorField, float, np.ndarray, float], Tuple[np.ndarray, np.ndarray, int]]
EnergyFn = Callable[[np.ndarray], float]


class StepSizeController(Protocol):
    """Interface expected by `simulate_adaptive`.

    Your implementation in `controllers.py` should follow this protocol.
    """

    order: int

    def init(self, *, t0: float, y0: np.ndarray, h0: float) -> None:
        ...

    def error_norm(
        self,
        *,
        t: float,
        y: np.ndarray,
        y_trial: np.ndarray,
        err: np.ndarray,
        h: float,
    ) -> float:
        ...

    def accept(self, *, err_norm: float) -> bool:
        ...

    def propose_step_size(self, *, h: float, err_norm: float, accepted: bool) -> float:
        ...


@dataclass(frozen=True)
class SimulationResult:
    t: np.ndarray
    y: np.ndarray
    h: np.ndarray
    nfev: int
    nstepper: int
    err_norm: np.ndarray
    energy: Optional[np.ndarray]
    energy_drift: Optional[np.ndarray]


def simulate_adaptive(
    f: VectorField,
    y0: np.ndarray,
    t_span: Sequence[float],
    *,
    J_fy: VectorField | None = None,
    h0: float,
    stepper: Stepper,
    controller: StepSizeController,
    history: tuple[Sequence[float], np.ndarray] | None = None,
    energy_fn: EnergyFn | None = None,
    nfev0: int = 0,
    max_steps: int = 1_000_000_000,
) -> SimulationResult:
    """Integrate y' = f(t, y) on [t0, t1] with adaptive time stepping.

    Args:
        f: Vector field f(t, y).
        J_fy: Jacobian of f with respect to y, for implicit solvers.
        y0: Initial state (1D array).
        t_span: (t0, t1).
        h0: Initial step size magnitude.
        stepper: Step function returning (y_trial, err, nfev). For multistep
            methods, the stepper can use the full accepted history.
        controller: Adaptive step-size controller.
        energy_fn: Optional energy function E(y) for diagnostics.
        max_steps: Hard limit on accepted steps.

    Returns:
        SimulationResult with recorded trajectory and diagnostics.
    """
    if len(t_span) != 2:
        raise ValueError("t_span must be a 2-sequence (t0, t1)")

    t0 = float(t_span[0])
    t1 = float(t_span[1])
    if t0 == t1:
        y0 = np.asarray(y0, dtype=float)
        energy = None
        drift = None
        if energy_fn is not None:
            e0 = float(energy_fn(y0))
            energy = np.array([e0], dtype=float)
            drift = np.array([0.0], dtype=float)
        return SimulationResult(
            t=np.array([t0], dtype=float),
            y=y0.reshape(1, -1),
            h=np.array([], dtype=float),
            nfev=int(nfev0),
            err_norm=np.array([], dtype=float),
            energy=energy,
            energy_drift=drift,
        )

    direction = 1.0 if t1 > t0 else -1.0
    if h0 <= 0.0:
        raise ValueError("h0 must be positive")

    y = np.asarray(y0, dtype=float)
    if y.ndim != 1:
        raise ValueError("y0 must be a 1D array")

    t = t0
    h = direction * float(h0)

    # Accepted history (can be pre-populated for multistep methods).
    t_hist: list[float] = [float(t0)]
    y_hist: list[np.ndarray] = [y.copy()]
    h_hist: list[float] = []
    err_hist: list[float] = []

    nfev_total = int(nfev0)

    if history is not None:
        t_hist = history[0]
        y_hist = (history[1])
        t = float(t_hist[-1])
        y = np.asarray(y_hist[-1], dtype=float)

        # Populate diagnostics for the pre-accepted history so arrays align.
        if len(t_hist) >= 2:
            dt = np.diff(np.asarray(t_hist, dtype=float))
            h_hist = [float(x) for x in dt]
            err_hist = [float("nan") for _ in range(dt.size)]

    e0 = None
    e_hist: list[float] = []
    if energy_fn is not None:
        e_hist = [float(energy_fn(yi)) for yi in y_hist]
        e0 = float(e_hist[0])

    n_stepper = max(0, len(t_hist) - 1)
    # Main loop: accept/reject steps until reaching t1.
    while (t - t1) * direction < 0.0:
        if n_stepper >= max_steps:
            raise RuntimeError(f"Exceeded max_steps={max_steps}")

        remaining = t1 - t
        # Never step past the end.
        if abs(h) > abs(remaining):
            h = remaining

        if h == 0.0:
            raise RuntimeError("Step size underflow (h==0)")

        y_trial, err, nfev = stepper(f, J_fy, t_hist, y_hist, h)
        nfev_total += int(nfev)
        err_norm = float(
            controller.error_norm(t=t, y=y, y_trial=y_trial, err=np.asarray(err, dtype=float), h=h)
        )
        accepted = bool(controller.accept(err_norm=err_norm))

        h_next = float(controller.propose_step_size(h=h, err_norm=err_norm, accepted=accepted))
        if h_next == 0.0 or not np.isfinite(h_next):
            raise RuntimeError("Controller proposed invalid next step size")

        if accepted:
            t = t + h
            y = np.asarray(y_trial, dtype=float)

            t_hist.append(float(t))
            y_hist.append(y.copy())
            h_hist.append(float(h))
            err_hist.append(err_norm)

            if energy_fn is not None:
                e_hist.append(float(energy_fn(y)))

        h = h_next
        n_stepper += 1

    t_arr = np.asarray(t_hist, dtype=float)
    y_arr = np.vstack([yi.reshape(1, -1) for yi in y_hist]).astype(float, copy=False)

    energy_arr = None
    drift_arr = None
    if energy_fn is not None:
        energy_arr = np.asarray(e_hist, dtype=float)
        drift_arr = energy_arr - float(e0)

    return SimulationResult(
        t=t_arr,
        y=y_arr,
        h=np.asarray(h_hist, dtype=float),
        nfev=int(nfev_total),
        nstepper=int(n_stepper),
        err_norm=np.asarray(err_hist, dtype=float),
        energy=energy_arr,
        energy_drift=drift_arr,
    )


def simulate_rk(
    problem,
    t_span: Sequence[float],
    *,
    tol: float = 1e-10,
    h0: float = 1e-3,
    max_steps: int = 1_000_000,
) -> SimulationResult:
    """Convenience wrapper to simulate a `ThreeBodyProblem` with using 
    embedded Runge-Kutta with sensible defaults.

    Scripts are expected to call this function and should not import numerical
    modules like integrators/controllers/dynamics directly.
    """
    # Local imports keep the core adaptive driver physics-agnostic.
    from .controllers import ControllerConfig, AdaptiveController
    from .dynamics import DynamicsParams, energy as _energy, rhs as _rhs
    from .integrators import dormand_prince54, rk_step_embedded

    tableau = dormand_prince54()

    # For dormand_prince5(4): error is O(h^(order_low+1)).
    controller = AdaptiveController(
        order=int(tableau.order_low),
        config=ControllerConfig(rtol=float(tol), atol=float(tol)),
    )

    def stepper(f: VectorField, J_fy: VectorField, t: list[np.ndarray], y: list[np.ndarray], h: float):
        return rk_step_embedded(f, t[-1], y[-1], h, tableau)

    params = DynamicsParams(G=float(getattr(problem, "G", 1.0)), masses=np.asarray(problem.masses, dtype=float))

    def f(t: float, y: np.ndarray) -> np.ndarray:
        return _rhs(t, y, params=params)

    def E(y: np.ndarray) -> float:
        return float(_energy(y, params=params))

    return simulate_adaptive(
        f,
        np.asarray(problem.y0, dtype=float),
        t_span,
        h0=float(h0),
        stepper=stepper,
        controller=controller,
        energy_fn=E,
        max_steps=int(max_steps),
    )


def simulate_bdf(
    problem,
    t_span: Sequence[float],
    *,
    tol: float = 1e-10,
    h0: float = 1e-3,
    max_steps: int = 1_000_000,
) -> SimulationResult:
    """Convenience wrapper to simulate a `ThreeBodyProblem` with using 
    backward differentiation formula (BDF) methods with sensible defaults.

    Scripts are expected to call this function and should not import numerical
    modules like integrators/controllers/dynamics directly.
    """
    # Local imports keep the core adaptive driver physics-agnostic.
    from .controllers import ControllerConfig, AdaptiveController
    from .dynamics import DynamicsParams, energy as _energy, rhs as _rhs, jacobian as J_fy
    from .integrators import BDF_step, dormand_prince54, rk_step_embedded
    
    # Use order 5 w/ 5th order explicit startup.
    controller = AdaptiveController(
        order=5,
        config=ControllerConfig(rtol=float(tol), atol=float(tol)),
    )

    def stepper(f: VectorField, J_fy: VectorField, t: list[np.ndarray], y: list[np.ndarray], h: float):
        t_past = np.array(t[-controller.order:], dtype=float)
        y_past = np.column_stack(y[-controller.order:]).astype(float, copy=False)
        return BDF_step(f, J_fy, t_past, y_past, h)

    params = DynamicsParams(G=float(getattr(problem, "G", 1.0)), masses=np.asarray(problem.masses, dtype=float))

    def f(t: float, y: np.ndarray) -> np.ndarray:
        return _rhs(t, y, params=params)

    def E(y: np.ndarray) -> float:
        return float(_energy(y, params=params))

    # Build startup history: need (order+1) points total.
    tableau = dormand_prince54()
    h = h0
    t_init: list[float] = [t_span[0]]
    y_init: list[np.ndarray] = [np.asarray(problem.y0, dtype=float)]
    nfev_startup_total = 0
    while len(t_init) < controller.order + 1:
        y_next, _err, _nfev = rk_step_embedded(f, t_init[-1], y_init[-1], h, tableau)
        nfev_startup_total += int(_nfev)
        err_norm = controller.error_norm(t=t_init[-1], y=y_init[-1], y_trial=y_next, err=_err, h=h)
        if controller.accept(err_norm=err_norm):
            t_init.append(float(t_init[-1] + h))
            y_init.append(np.asarray(y_next, dtype=float))
        
        h = controller.propose_step_size(h=h, err_norm=err_norm, accepted=True)

    return simulate_adaptive(
        f,
        np.asarray(problem.y0, dtype=float),
        t_span,
        J_fy=J_fy,
        h0=float(h0),
        stepper=stepper,
        controller=controller,
        history=(t_init, y_init),
        energy_fn=E,
        nfev0=int(nfev_startup_total),
        max_steps=int(max_steps),
    )