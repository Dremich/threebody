"""Batch render multiple orbits (thin CLI glue).

Runs orbits headlessly, samples a fixed number of frames uniformly in time,
and writes a per-orbit GIF using ffmpeg (no intermediate PNGs).

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

from threebody.problems import list_orbits, load_problem
from threebody.simulate import simulate
from threebody.visualize import render_gif_ffmpeg


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch render three-body orbits")
    p.add_argument("--orbits", nargs="*", default=None, help="Subset of orbit names (default: all)")
    p.add_argument("--frames", type=int, default=500, help="Number of frames per orbit (default: 600)")
    p.add_argument("--tol", type=float, default=1e-10, help="Adaptive tolerance (default: 1e-10)")
    p.add_argument("--h0", type=float, default=1e-3, help="Initial step size (default: 1e-3)")
    p.add_argument("--out-dir", type=str, default="renders", help="Output directory (default: renders)")
    p.add_argument("--fps", type=float, default=120.0, help="GIF frames per second (default: 120.0)")
    return p.parse_args()


def _sample_uniform_in_time(t: np.ndarray, y: np.ndarray, n_frames: int) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or y.ndim != 2 or y.shape[0] != t.shape[0]:
        raise ValueError("Expected t:(n,), y:(n,d)")

    n_frames = int(n_frames)
    if n_frames < 2:
        raise ValueError("frames must be >= 2")

    t_grid = np.linspace(t[0], t[-1], n_frames)
    ys = np.empty((n_frames, y.shape[1]), dtype=float)
    for j in range(y.shape[1]):
        ys[:, j] = np.interp(t_grid, t, y[:, j])
    return ys


def main() -> int:
    args = _parse_args()

    orbit_names = args.orbits if args.orbits else list_orbits()
    if not orbit_names:
        raise SystemExit("No orbit definitions found")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for name in orbit_names:
        problem = load_problem(name)
        if problem.period is None:
            print(f"Skipping {name}: missing period")
            continue

        T = float(problem.period)
        result = simulate(problem, (0.0, T), tol=float(args.tol), h0=float(args.h0))

        sampled_states = _sample_uniform_in_time(result.t, result.y, int(args.frames))

        out_gif = out_root / f"{name}.gif"
        render_gif_ffmpeg(sampled_states, out_gif, fps=float(args.fps), dpi=150)
        print(f"Wrote: {out_gif}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
