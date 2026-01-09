"""Export precomputed three-body trajectories to a browser-friendly JSON format.

GitHub Pages cannot run Python, so this script converts the existing `.npz`
outputs in `data/computations/` into static JSON assets that a JS viewer can load.

Examples:
    Export one file:
        C:/.../.venv/Scripts/python.exe scripts/export_web_data.py \
            --input data/computations/yarn_RK.npz \
            --output visualizer/data/yarn_RK.json \
            --max-frames 5000

    Export all computations and generate a manifest for the web viewer:
        C:/.../.venv/Scripts/python.exe scripts/export_web_data.py \
            --input-dir data/computations \
            --output-dir visualizer/data \
            --manifest visualizer/data/manifest.json \
            --max-frames 6000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _extract_xy(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=float)
    if states.ndim != 2:
        raise ValueError("states must have shape (n, d)")

    d = states.shape[1]
    if d == 12:
        r = states[:, :6].reshape(-1, 3, 2)
        return r[:, :, 0], r[:, :, 1]
    if d == 18:
        r = states[:, :9].reshape(-1, 3, 3)
        return r[:, :, 0], r[:, :, 1]

    raise ValueError(f"Unsupported state dimension {d}; expected 12 or 18")


def _downsample_indices(n: int, max_frames: int) -> np.ndarray:
    if max_frames <= 0 or n <= max_frames:
        return np.arange(n, dtype=int)
    # Uniform index sampling (keeps start/end)
    return np.linspace(0, n - 1, int(max_frames), dtype=int)


def export_npz_to_json(input_path: Path, output_path: Path, *, max_frames: int) -> None:
    data = np.load(input_path)

    t = data.get("t")
    y = data.get("y")
    energy = data.get("energy")

    if t is None or y is None:
        raise ValueError("Expected arrays 't' and 'y' in the .npz")

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if y.ndim != 2 or y.shape[0] != t.shape[0]:
        raise ValueError("y must be 2D with y.shape[0] == t.shape[0]")

    idx = _downsample_indices(t.shape[0], int(max_frames))

    t_ds = t[idx]
    y_ds = y[idx]

    x, y2 = _extract_xy(y_ds)

    payload: dict[str, object] = {
        "name": input_path.stem,
        "t": t_ds.tolist(),
        "x": x.tolist(),
        "y": y2.tolist(),
        "meta": {
            "source": str(input_path.as_posix()),
            "frames": int(t_ds.shape[0]),
            "state_dim": int(y_ds.shape[1]),
        },
    }

    if energy is not None:
        energy = np.asarray(energy, dtype=float)
        if energy.ndim == 1 and energy.shape[0] == t.shape[0]:
            payload["energy"] = energy[idx].tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def export_dir(
    input_dir: Path,
    output_dir: Path,
    *,
    max_frames: int,
) -> list[dict[str, str]]:
    input_dir = input_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, str]] = []
    for npz_path in sorted(input_dir.glob("*.npz")):
        key = npz_path.stem
        out_path = output_dir / f"{key}.json"
        export_npz_to_json(npz_path, out_path, max_frames=max_frames)
        entries.append({"key": key, "label": key, "url": f"data/{key}.json"})

    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-frames", type=int, default=5000)
    args = parser.parse_args()

    if args.input is not None or args.output is not None:
        if args.input is None or args.output is None:
            raise SystemExit("--input and --output must be provided together")
        export_npz_to_json(args.input, args.output, max_frames=args.max_frames)
        return

    if args.input_dir is not None or args.output_dir is not None:
        if args.input_dir is None or args.output_dir is None:
            raise SystemExit("--input-dir and --output-dir must be provided together")
        entries = export_dir(args.input_dir, args.output_dir, max_frames=args.max_frames)
        if args.manifest is not None:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(
                json.dumps({"entries": entries}, indent=2) + "\n",
                encoding="utf-8",
            )
        return

    raise SystemExit("Provide either --input/--output or --input-dir/--output-dir")


if __name__ == "__main__":
    main()
