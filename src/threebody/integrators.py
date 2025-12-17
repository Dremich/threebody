"""Integrator facade.

This module re-exports integration algorithms from submodules so the rest of the
project can depend on a stable import path:

	from threebody import integrators

You renamed the implementation modules to use underscores, so we can use normal
Python imports (no dynamic loading).
"""

from __future__ import annotations

from .runge_kutta_integrators import (
	EmbeddedRKTableau,
	VectorField,
	dormand_prince54,
	rk_step_embedded,
)

from .implicit_integrators import BDF_step

__all__ = [
	"VectorField",
	"EmbeddedRKTableau",
	"dormand_prince54",
	"rk_step_embedded",
	"BDF_step",
]

