"""Run a single named orbit (thin CLI glue).

Usage:
  C:/.../python.exe scripts/run_orbit.py figure_eight --t-mult 1.0 --tol 1e-10 --h0 1e-3 --visualize --save out/figure8.npz

Policy:
- No numerics here: no RHS/integrators/controllers.
- Orchestration only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running directly from a src-layout repo without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from threebody.problems import load_problem
from threebody.simulate import simulate_rk, simulate_bdf
from threebody.visualize import visualize


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single three-body orbit")
    p.add_argument("orbit_name", help="Orbit key (JSON filename stem)")
    p.add_argument("--t-mult", type=float, default=1.0, help="Multiple of orbit period (default: 1.0)")
    p.add_argument("--tol", type=float, default=1e-10, help="Adaptive tolerance (default: 1e-10)")
    p.add_argument("--h0", type=float, default=1e-3, help="Initial step size (default: 1e-3)")
    p.add_argument("--visualizer", action="store_true", help="Launch interactive visualizer")
    p.add_argument("--save", type=str, default=None, help="Save trajectory to .npz")
    return p.parse_args()

def run(
    problem_name: str,
    t_mult: float,
    tol: float,
    h0: float,
    visualizer: bool,
    method: str = "RK",
    output_dir=None,
    show_full_trajectory: bool = False,
):
    problem = load_problem(problem_name)
    if problem.period is None:
        raise SystemExit(f"Orbit '{problem.name}' has no period; cannot use --t-mult")

    T = float(problem.period) * float(t_mult)
    match method:
        case "RK":
            result = simulate_rk(problem, (0.0, T), tol=float(tol), h0=float(h0))
        case "BDF":
            result = simulate_bdf(problem, (0.0, T), tol=float(tol), h0=float(h0))
        case _:
            raise ValueError(f"Unknown method: {method}")

    if output_dir is not None:
        out = Path(output_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            t=result.t,
            y=result.y,
            energy=(result.energy if result.energy is not None else np.array([])),
            h=result.h,
            err_norm=result.err_norm,
            nfev=result.nfev,
        )
        print(f"Saved: {out}")

    if visualizer:
        visualize(
            result.y,
            t=result.t,
            energy=result.energy,
            show_energy=(result.energy is not None),
            show_full_trajectory=bool(show_full_trajectory),
        )


def main() -> int:
    args = _parse_args()
    
    run(
        problem_name=args.orbit_name,
        t_mult=args.t_mult,
        tol=args.tol,
        h0=args.h0,
        visualizer=args.visualizer,
        output_dir=args.output_dir,
    )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
