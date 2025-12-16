"""Explicit Runge-Kutta IVP integration. 
This module implements the embedded Runge-Kutta pair Dormand-Prince5(4) for adaptive
stepsize IVP integration.

Author: Andrew Tolton, University of California, Los Angeles
Date: 12-15-2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

# Define type for forcing function f(t, y) -> dy/dt
VectorField = Callable[[float, np.ndarray], np.ndarray]

@dataclass(frozen=True)
class EmbeddedRKTableau:
    """Butcher tableau for an embedded explicit RK pair.
    This class allows for the generalization of any embedded RK pair by specifying the
    Butcher tableau coefficients.

    Attributes:
        a: Coefficients for the intermediate stages.
        b_high: Coefficients for the high order final stage.
        b_low: Coefficients for the low order final stage.
        c: Coefficients for the intermediate time steps.
        order_high: Order of the high solution (used by controllers)
        order_low: Order of the low solution (used by controllers)
    """
    # Initialize butcher tableau arrays
    c: np.ndarray # [s,]
    a: np.ndarray # [s, s]
    b_high: np.ndarray # [s,]
    b_low: np.ndarray # [s,]
    order_high: int
    order_low: int

def dormand_prince54() -> EmbeddedRKTableau:
    """Dormand-Prince 5(4) embedded RK pair. This is a 7-stage method computing a 5th and 4th order solution.
    Coefficients from https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method

    Returns:
        EmbeddedRKTableau configured for Dormand-Prince.
    """

    a_coeffs = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0, 0, 0],
    [44/45, -56/15, 32/9, 0, 0, 0, 0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ])
    b5_coeffs = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    b4_coeffs = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    c_coeffs = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    n_stages = 7

    return EmbeddedRKTableau(
        c=c_coeffs,
        a=a_coeffs,
        b_high=b5_coeffs,
        b_low=b4_coeffs,
        order_high=5,
        order_low=4,
    )

def rk_step_embedded(f: VectorField, t: float, y: np.ndarray, h: float, tableau: EmbeddedRKTableau,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """General method for taking a single step with an embedded Runge-Kutta method. 
    The method computes:
    
    y_{n+1} = y_n + h * sum_{i=1}^s b_i * k_i
    
    where 
      k_1 = f(t_n, y_n)
      k_2 = f(t_n + c_2*h, y_n + h*(a_21*k_1))
      k_3 = f(t_n + c_3*h, y_n + h*(a_31*k_1 + a_32*k_2))
      ...
      k_s = f(t_n + c_s*h, y_n + h*sum_{j=1}^{s} a_sj*k_j)
    
    are the stages given by the Butcher tableau. The method also computes an embedded lower-order solution for error estimation.
       
    Find more info at https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    
    Parameters:
        f: Vector field f(t, y) -> dy/dt.
        t: Current time.
        y: Current state, 1D array.
        h: Step size.
        tableau: Embedded RK tableau.
            
    Returns:
        y_high: Higher-order solution at t+h.
        err: Error estimate vector (y_high - y_low).
        nfev: Number of RHS evaluations.
    """
    
    # Check input shape, important for numpy matrix operations
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    
    # Extract coefficients from tableau
    c = tableau.c
    a = tableau.a
    bH = tableau.b_high
    bL = tableau.b_low
    n_stages = c.shape[0]
    
    # Build stages
    k = np.zeros((n_stages, y.shape[0]), dtype=float)
    k[0] = f(t,y) 
    
    for ii in range(1, n_stages):
        t_stage = t + c[ii] * h
        y_stage = y.copy()
        for jj in range(ii):
            y_stage += h * a[ii,jj] * k[jj]
            
        k[ii] = f(t_stage, y_stage)
        
    # Combine stages to get high and low order solutions
    y_high = y + h * (bH @ k)
    y_low = y + h * (bL @ k)
    
    # Estimate error as difference between high and low order solutions
    err = y_high - y_low
    
    return y_high, err, int(n_stages)
        
        
    