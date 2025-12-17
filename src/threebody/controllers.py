"""Adaptive step-size controller for IVP integration methods.

Core ideas (embedded RK):
- Take a time step producing (y_high, err) using error estimation from the method.
- Convert the error vector to a scalar using a scaled norm
- Accept the step if err_norm <= tolerance
- Update h using the method order and a safety factor
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ControllerConfig:
	"""Configuration for adaptive step size control."""

	rtol: float = 1e-9
	atol: float = 1e-12
	safety: float = 0.9
	min_factor: float = 0.2
	max_factor: float = 5.0

	# Max number of consecutive rejected steps before aborting.
	max_reject: int = 20


@dataclass
class ControllerState:
	"""State carried between steps (for PI/PID controllers)."""

	n_reject: int = 0
	prev_err: float | None = None
	prev_h: float | None = None


def scaled_rms_error_norm(err: np.ndarray, y: np.ndarray, y_trial: np.ndarray, *, atol: float, rtol: float, h: float) -> float:
	"""Compute a scalar error norm used for accept/reject.

	Uses an RMS norm with componentwise scaling:
		scale_i = atol + rtol * max(|y_i|, |y_trial_i|)
		err_norm = sqrt( mean( (err_i / scale_i)^2 ) )
	"""
	err = np.asarray(err, dtype=float)
	y = np.asarray(y, dtype=float)
	y_trial = np.asarray(y_trial, dtype=float)

	if err.shape != y.shape or y.shape != y_trial.shape:
		raise ValueError("err, y, y_trial must have the same shape")
	if err.ndim != 1:
		raise ValueError("Expected 1D state vectors")

	max_allowed_error = float(atol) + float(rtol) * np.maximum(np.abs(y), np.abs(y_trial))
	# Avoid division by zero if atol=rtol=0.
	scale = np.maximum(max_allowed_error, np.finfo(float).tiny)

	if h == 0.0:
		z = err / max_allowed_error
	else: 
		z = err / (h * max_allowed_error)
	return float(np.sqrt(np.mean(z * z)))

def max_error_norm(err: np.ndarray, y: np.ndarray, y_trial: np.ndarray, *, atol: float, rtol: float, h: float) -> float:
    """Computes a scalar error norm used to determine whether to accept/reject y_trial.
    y_trial is accepted when err_norm <= 1.

    Uses a max norm with componentwise scaling:
        err_limit = atol + rtol * max(|y_i|, |y_trial_i|)
        err_norm = max( |err_i| / err_limit )
    """
    # Standard error check using scale term with max y
    max_allowed_error = atol + rtol * np.maximum(np.abs(y), np.abs(y_trial))
    
    # Check for division by zero
    max_allowed_error = np.maximum(max_allowed_error, np.finfo(float).tiny)
    
    # Compute max norm of error ratio
    error_ratio = np.abs(err) / h / max_allowed_error
    return float(np.max(error_ratio))

class StepSizeController(Protocol):
	"""Controller protocol expected by `threebody.simulate.simulate_adaptive`.
	Implementations typically store a ControllerConfig and ControllerState.
	"""

	order: int

	def error_norm(
		self,
		*,
		t: float,
		y: np.ndarray,
		y_trial: np.ndarray,
		err: np.ndarray,
	) -> float:
		...

	def accept(self, *, err_norm: float) -> bool:
		...

	def propose_step_size(self, *, h: float, err_norm: float, accepted: bool) -> float:
		...


@dataclass
class AdaptiveController:
    """Adaptive step-size controller for variable stepsize numerical integration.
    """
    
    order: int
    config: ControllerConfig = ControllerConfig()

    def error_norm(self, *, t: float, y: np.ndarray, y_trial: np.ndarray, err: np.ndarray, h: float) -> float:
        return scaled_rms_error_norm(err, y, y_trial, atol=self.config.atol, rtol=self.config.rtol, h=h)

    def accept(self, *, err_norm: float) -> bool:
        return err_norm <= 1.0

    def propose_step_size(self, *, h: float, err_norm: float, accepted: bool) -> float:
        # Classic controller (simple, robust):
        #   factor = safety * err_norm^(-1/(order+1))
        #   factor clipped to [min_factor, max_factor]
        # Use order = embedded_low_order (the order of the error estimate).

        if np.abs(err_norm) < 1e-20:
            return h * self.config.max_factor

        # Adjust step size based on error rate.
        # Smaller err_norm -> increase h; larger err_norm -> decrease h.
        safety = float(self.config.safety)
        factor = safety * float(err_norm) ** (-1.0 / (self.order + 1))
        factor = float(np.clip(factor, self.config.min_factor, self.config.max_factor))
        return h * factor