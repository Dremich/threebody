"""Named three-body initial value problems loaded from JSON.

This module reads orbit definitions from the repository data directory:
    data/orbits/*.json

JSON schema (minimal):
{
  "name": "figure_eight",
  "masses": [1, 1, 1],
  "G": 
  "y0": [...],
  "period": 6.3259,
  "description": "The Chenciner-Mongomery figure-eight orbit",
  "reference": "https://link.to.paper"
}

Notes:
- `period`, `description`, and `reference` are optional (can be null).
- `y0` is the flat state vector used by the rest of the codebase. If 18 elements,
  it's a full spatial problem; if 12 elements, it's a planar problem with z and vz = 0.
   y0_3D = [x1,y1,z1,x2,y2,z2,x3,y3,z3,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3]
   y0_2D = [x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3]
- `G` is the gravitational constant, defaults to 1.0 if missing.
- `masses` is a length-3 array giving the masses of each body.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import json
import numpy as np


@dataclass(frozen=True)
class ThreeBodyProblem:
    name: str
    masses: np.ndarray
    y0: np.ndarray
    period: float | None
    G: float = 1.0
    description: str | None = None
    reference: str | None = None


def _repo_root() -> Path:
    # .../src/threebody/problems.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def orbits_dir() -> Path:
    """Return the path to the on-disk orbit JSON directory."""
    return _repo_root() / "data" / "orbit_definitions"


def list_orbits() -> List[str]:
    """List available orbit names (derived from JSON filenames)."""
    root = orbits_dir()
    if not root.exists():
        return []
    return sorted(p.stem for p in root.glob("*.json") if p.is_file())


def _as_float_array(x: Any, *, name: str) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"{name} must be array-like of numbers") from e
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def _lift_planar_y0_to_spatial(y0_12: np.ndarray) -> np.ndarray:
    """Convert 12D planar state to 18D spatial state by inserting zeros.

    Planar convention:
        [x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3]

    Spatial convention:
        [x1,y1,z1,x2,y2,z2,x3,y3,z3,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3]

    We set all z and vz components to 0.
    """
    y0_12 = np.asarray(y0_12, dtype=float)
    if y0_12.shape != (12,):
        raise ValueError(f"Expected planar y0 of shape (12,), got {y0_12.shape}")

    r2 = y0_12[:6].reshape(3, 2)
    v2 = y0_12[6:].reshape(3, 2)

    r3 = np.zeros((3, 3), dtype=float)
    v3 = np.zeros((3, 3), dtype=float)
    r3[:, :2] = r2
    v3[:, :2] = v2

    return np.concatenate([r3.reshape(-1), v3.reshape(-1)])


def _parse_problem_dict(d: Dict[str, Any], *, fallback_name: str) -> ThreeBodyProblem:
    if not isinstance(d, dict):
        raise ValueError("Orbit JSON must be an object")

    name = d.get("name", fallback_name)
    if not isinstance(name, str) or not name:
        raise ValueError("'name' must be a non-empty string")

    masses = _as_float_array(d.get("masses", None), name="masses")
    if masses.shape != (3,):
        raise ValueError(f"masses must have shape (3,), got {masses.shape}")

    y0 = _as_float_array(d.get("initial_state", None), name="initial_state")
    if y0.shape == (12,):
        y0 = _lift_planar_y0_to_spatial(y0)
    elif y0.shape != (18,):
        raise ValueError(
            f"initial_state must have length 12 (planar) or 18 (spatial); got {y0.shape}"
        )

    period_raw = d.get("period", None)
    period: Optional[float]
    if period_raw is None:
        period = None
    else:
        try:
            period = float(period_raw)
        except Exception as e:
            raise ValueError("'period' must be a number or null") from e

    G_raw = d.get("G", 1.0)
    if G_raw is None:
        G = 1.0
    else:
        try:
            G = float(G_raw)
        except Exception as e:
            raise ValueError("'G' must be a number or null") from e

    description_raw = d.get("description", None)
    description = None if description_raw is None else str(description_raw)

    reference_raw = d.get("reference", None)
    reference = None if reference_raw is None else str(reference_raw)

    return ThreeBodyProblem(
        name=name,
        masses=masses,
        y0=y0,
        period=period,
        G=G,
        description=description,
        reference=reference,
    )


def load_orbit(name: str) -> ThreeBodyProblem:
    """Load an orbit definition from `data/orbits/{name}.json`."""
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")

    path = orbits_dir() / f"{name}.json"
    if not path.exists():
        available = ", ".join(list_orbits())
        raise FileNotFoundError(
            f"Orbit '{name}' not found at {path}. Available: {available or '(none)'}"
        )

    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)

    return _parse_problem_dict(d, fallback_name=path.stem)


# Backwards/CLI-friendly alias.
def load_problem(name: str) -> ThreeBodyProblem:
    """Alias for `load_orbit` used by CLI scripts."""
    return load_orbit(name)


def load_all_orbits() -> List[ThreeBodyProblem]:
    """Load all orbit definitions found under `data/orbits/*.json`."""
    root = orbits_dir()
    if not root.exists():
        return []

    problems: List[ThreeBodyProblem] = []
    for p in sorted(root.glob("*.json")):
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        problems.append(_parse_problem_dict(d, fallback_name=p.stem))

    return problems
