"""Generate Suvakov–Dmitrašinović-style orbit JSON from a few inputs.

This script is intentionally small and interactive.

It prompts for:
  - name
  - x_dot_1
  - y_dot_1
  - period

Then it writes a new JSON file in the same format as `butterfly_I.json`, using
the same default values for all other fields.

State vector format (length 18):
  [r1(3), r2(3), r3(3), v1(3), v2(3), v3(3)]

Here we keep the same default positions as butterfly_I:
  r1 = (-1, 0, 0)
  r2 = ( 1, 0, 0)
  r3 = ( 0, 0, 0)

And compute velocities from the provided (x_dot_1, y_dot_1):
  v1 = ( x_dot_1,  y_dot_1, 0)
  v2 = ( x_dot_1,  y_dot_1, 0)
  v3 = (-2*x_dot_1, -2*y_dot_1, 0)
"""

from __future__ import annotations

import json
from pathlib import Path


TEMPLATE = {
	"name": "butterfly_I",
	"G": 1.0,
	"masses": [1.0, 1.0, 1.0],
	"initial_state": [
		-1.0,
		0.0,
		0.0,
		1.0,
		0.0,
		0.0,
		0.0,
		0.0,
		0.0,
		0.4662036850,
		0.4323657300,
		0.0,
		0.4662036850,
		0.4323657300,
		0.0,
		-0.9324073700,
		-0.8647314600,
		0.0,
	],
	"period": 6.32591398,
	"description": "Suvakov-Dmitrasinovic butterfly I periodic orbit",
	"reference": "http://dx.doi.org/10.1103/PhysRevLett.110.114301",
}


def _prompt_str(label: str) -> str:
	while True:
		s = input(f"{label}: ").strip()
		if s:
			return s
		print("Please enter a non-empty value.")


def _prompt_float(label: str) -> float:
	while True:
		s = input(f"{label}: ").strip()
		try:
			return float(s)
		except Exception:
			print("Please enter a valid number.")


def build_initial_state(*, x_dot_1: float, y_dot_1: float) -> list[float]:
	# Positions match butterfly_I defaults.
	r1 = (-1.0, 0.0, 0.0)
	r2 = (1.0, 0.0, 0.0)
	r3 = (0.0, 0.0, 0.0)

	x1 = float(x_dot_1)
	y1 = float(y_dot_1)
	v1 = (x1, y1, 0.0)
	v2 = (x1, y1, 0.0)
	v3 = (-2.0 * x1, -2.0 * y1, 0.0)

	state = [*r1, *r2, *r3, *v1, *v2, *v3]
	if len(state) != 18:
		raise RuntimeError("internal error: initial_state must have length 18")
	return [float(x) for x in state]


def main() -> int:
	name = _prompt_str("name")
	x_dot_1 = _prompt_float("x_dot_1")
	y_dot_1 = _prompt_float("y_dot_1")
	period = _prompt_float("period")

	data = dict(TEMPLATE)
	data["name"] = name
	data["period"] = float(period)
	data["initial_state"] = build_initial_state(x_dot_1=x_dot_1, y_dot_1=y_dot_1)

	# Keep template description/reference by default; user can edit later.

	out_path = Path(__file__).resolve().parent / f"{name}.json"
	with out_path.open("w", encoding="utf-8", newline="\n") as f:
		json.dump(data, f, indent=2)
		f.write("\n")

	print(f"Wrote: {out_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

