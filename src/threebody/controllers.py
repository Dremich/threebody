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


def scaled_rms_error_norm(err: np.ndarray, y: np.ndarray, y_trial: np.ndarray, *, atol: float, rtol: float) -> float:
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

	scale = float(atol) + float(rtol) * np.maximum(np.abs(y), np.abs(y_trial))
	# Avoid division by zero if atol=rtol=0.
	scale = np.maximum(scale, np.finfo(float).tiny)
	z = err / scale
	return float(np.sqrt(np.mean(z * z)))

def max_error_norm(err: np.ndarray, y: np.ndarray, y_trial: np.ndarray, *, atol: float, rtol: float) -> float:
    """Computes a scalar error norm used to determine whether to accept/reject y_trial.
    y_trial is accepted when err_norm <= 1.

    Uses a max norm with componentwise scaling:
        err_limit = atol + rtol * max(|y_i|, |y_trial_i|)
        err_norm = max( |err_i| / err_limit )
    """
    # Standard error check using scale term with max y
    scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_trial))
    
    # Check for division by zero
    scale = np.maximum(scale, np.finfo(float).tiny)
    
    # Compute max norm of error ratio
    error_ratio = np.abs(err) / scale
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
class RKAdaptiveController:
	"""Simple, robust step-size controller for embedded explicit RK.

	Uses a classic one-step controller:
		factor = safety * err_norm^(-1/(order+1))
	with clipping to [min_factor, max_factor].
	"""

	order: int
	config: ControllerConfig = ControllerConfig()

	def error_norm(
		self,
		*,
		t: float,
		y: np.ndarray,
		y_trial: np.ndarray,
		err: np.ndarray,
	) -> float:
		return scaled_rms_error_norm(
			err,
			y,
			y_trial,
			atol=self.config.atol,
			rtol=self.config.rtol,
		)

	def accept(self, *, err_norm: float) -> bool:
		return float(err_norm) <= 1.0

	def propose_step_size(self, *, h: float, err_norm: float, accepted: bool) -> float:
		h = float(h)
		err = float(err_norm)

		if err <= 0.0:
			factor = self.config.max_factor
		else:
			exponent = -1.0 / float(self.order + 1)
			factor = self.config.safety * (err ** exponent)

		factor = float(np.clip(factor, self.config.min_factor, self.config.max_factor))

		# If rejected, always reduce (even if computed factor was >1 due to err<1).
		if not accepted:
			factor = min(factor, 1.0)

		h_new = h * factor
		# Preserve direction.
		if h_new == 0.0:
			h_new = np.copysign(np.finfo(float).tiny, h)

		return float(h_new)

@dataclass
class AdaptiveController:
    """Adaptive step-size controller for variable stepsize numerical integration.
    """
    
    order: int
    config: ControllerConfig = ControllerConfig()

    def error_norm(self, *, t: float, y: np.ndarray, y_trial: np.ndarray, err: np.ndarray) -> float:
        return max_error_norm(err, y, y_trial, atol=self.config.atol, rtol=self.config.rtol)

    def accept(self, *, err_norm: float) -> bool:
        return err_norm <= 1.0

    def propose_step_size(self, *, h: float, err_norm: float, accepted: bool) -> float:
        # Classic controller (simple, robust):
        #   factor = safety * err_norm^(-1/(order+1))
        #   factor clipped to [min_factor, max_factor]
        # Use order = embedded_low_order (the order of the error estimate).
        
        # TODO: UPDATE THIS TO USE ERR_NORM
        # TODO: IMPLEMENT ITERATIVE CONTROLLER
        # TODO: CREATE 2 MORE PROBLEMS
        # TODO: WRITE REPORT
        # Adjust step size based on error rate
        rate_ratio = (max_error_rate / error_rate)**(1/(order + 1))  # Adjust step size based on error rate
        h = h * min(5, max(0.2, safety * np.min(rate_ratio)))  # Limit step size changes to factor of 5 increase or 0.2 decrease
        

