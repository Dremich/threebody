"""Linear-Multistep IVP integrator.
This module provides a general class for linear-multistep methods, and 
implements the trapezoidal method via Newton-Raphson for solving IVPs.

Author: Andrew Tolton, University of California, Los Angeles
Date: 12-15-2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import math

# Define type for forcing function f(t, y) -> dy/dt
VectorField = Callable[[float, np.ndarray], np.ndarray]
    
def adaptive_bdf_coeffs(t_past) -> np.ndarray:
    """Computes the adaptive BDF y-coeffs for BDF-k and the error coefficients for
    the leading error term using the past k+2 times. 
    The coefficients are computed using the matrix equation:
    [1, 1,      1,      ...,  1       ] {c0}   {0}
    [0, dt1,    dt2,    ...,  dtk-1   ] {c1}   {1}
    [0, dt1**2, dt2**2, ...,  dtk-1**2] {c2} = {0}
    [              ...                ] {..}   {0}
    [0, dt1**k, dt2**k, ...,  dtk-1**k] {ck}   {0}
    
    The local error rate = d_k+1 (y_past @ err_coeffs)
    where d_k+1 = -1^(k+1) sum(i=1:k){c_i dt_i^k+1 / (k+1)!}
    
    Compute the local truncation error as error_rate * h.
    
    Parameters:
        t_past: np.ndarray [k+1,]
         Time vector of length k+1, where t_past[-1] is the current time step.
    
    Returns: 
        y_coeffs: np.ndarray [k+1,]
            The coefficients for the y terms in the multistep method.
        d_kp1: float
            The coefficient for the local truncation error estimate.
        err_coeffs: np.ndarray [k+1,]
            The weights for computing the k+1th derivative of y from y_past.
    """
    
    k = len(t_past) - 1 # order k of the BDF method
    
    # Build dt = [0, dt1, dt2, ..., dtk] where dt_i = t_{n-i} - t_n
    t_reversed = np.flip(t_past)
    dt = t_reversed - t_reversed[0]
    
    ## Solve for the y coefficients (uses most recent k+1 points)
    # Build the Vandermonde matrix
    V = np.transpose(np.vander(dt[:(k)], increasing=True)) # exclude last point, saved for error
    
    # Build the right-hand side vector
    rhs = np.zeros(len(t_past)-1)
    rhs[1] = 1
    
    # Solve the matrix equation V * y_coeffs = rhs to find y_coeffs
    y_coeffs = np.linalg.solve(V, rhs)
    
    ## Solve for the error coefficients (uses all k+1 points)
    W = np.transpose(np.vander(dt[:(k+1)], increasing=True))
    rhs_err = np.zeros(len(t_past))
    rhs_err[-1] = math.factorial(k)
    
    err_coeffs = np.linalg.solve(W, rhs_err) # Solve for the error coefficients
    d_kp1 = np.sum(y_coeffs[1:] * dt[1:(k)]**(k)) / math.factorial(k)
    
    return y_coeffs, d_kp1, err_coeffs  
    
def BDF_step(f: VectorField, J_fy: VectorField, t_past: np.ndarray, y_past: np.ndarray, h: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Backwards differentiation method for solving ODE IVPs, defined by the equation
    sum_{j=0}^{p} cy_j * y_{n-j} = h f(t_n, y_n)
    where cy_j are the y_coeffs and p is the number of previous steps used (should = order of method).
    
    t_past and y_past are vectors of length p+1, where t_past[-1] is t_n-1. 
    p+1 th point is used for error estimation. 

    Parameters:
        f: function
            The function defining the ODE (dy/dt = f(t, y)).
        y_past: np.ndarray [m,p+1]
            Vector of previous steps y_{n+j} (j=-p-1,...,-1).
        t_past: np. [p+1,]
            Vector of previous times t_{n+j} (j=-p-1,...,-1).
        h: float
            Current proposed step size.
        tableau: LinearMultistepTableau
            The tableau for the linear-multistep method.
        J_fy: callable
            Jacobian of f with respect to y, required for implicit methods (cf_0 != 0).

    Returns:
        y_n: np.ndarray [m,] 
            Numerical solution for y(t_n).
        err: np.ndarray [m,] 
            Error estimate.
        nfev: int
            Number of RHS evaluations.
    """
    # Ensure y is a [m,p+1] numpy array (even if 0-d scalar) to handle math uniformly
    y_past = np.array(y_past, dtype=float)
    if y_past.ndim == 1:
        y_past = y_past.reshape(1, -1)
    
    t_past = np.array(t_past, dtype=float)
    t_vec = np.concatenate((t_past, [t_past[-1] + h]))
    p = y_past.shape[1] # order p of the BDF method (-1 because extra point for error)
    
    # Determine adaptive y_coefficients
    y_coeffs, d_kp1, err_coeffs = adaptive_bdf_coeffs(t_vec)

    # Build the linear combination of previous steps sum_{j=1}^{p} cy_j * y_{n-j}
    y_prev = np.zeros(y_past.shape[0])
    for jj, cy_j in enumerate(y_coeffs[1:], start=1):
        y_prev += cy_j * y_past[:, -jj]
    
    # Find y_n self-consistently using Newton-Raphson
    # y_ceoffs were computed such that f(t_n, y_n) = sum_{j=0:k} (c_j * y_{n-j})
    t_n = t_vec[-1]
    
    def g(y_n):
        return y_coeffs[0]*y_n + y_prev - f(t_n, y_n)
    def J_g(y_n):
        return y_coeffs[0]*np.eye(y_past.shape[0]) - J_fy(t_n, y_n)
    
    y_guess = y_past[:, -1].copy()  # Use previous y-value as initial guess
    
    ideal_tolerance = 0.1 * h**(p + 1) # We want Newton error O(local truncation error)
    floating_point_error = 10 * np.finfo(float).eps
    safe_tol = max(ideal_tolerance, floating_point_error)  # Use the larger of the two
    tol = min(safe_tol, 1e-5)  # Cap maximum tolerance in case of large step size 

    # debug_check_jacobian(g, J_g, y_guess)  # Debugging function to check Jacobian consistency
    y_n, nr_iter = newton_raphson(g, J_g, y_guess, tol=tol)
    
    # Warn if Newton-Raphson did not converge
    if nr_iter == newton_raphson.__defaults__[1]:  # max_iter default
        print(f"Warning: Newton-Raphson did not converge at t={t_n:.4f}")

    # Compute error estimate:
    # LTE = d_kp1 * h * d^{k+1}y/dt^{k+1} 
    y_all = np.concatenate((y_past, y_n.reshape(-1, 1)), axis=1)
    error = d_kp1 * h * y_all @ np.flip(err_coeffs)

    return y_n, error, nr_iter

def newton_raphson(f: callable, J_fy: VectorField, y0: np.ndarray, tol: float = 1e-10, max_iter: int = 20, verbose: bool = False):
    """Newton-Raphson root-finding method for nonlinear equations f(y) = 0.
    Parameters:
        f: function
            The function for which we want to find the root (f(y) = 0).
        J_fy: function
            The Jacobian of the function f.
        y0: np.ndarray
            The initial guess for the root.
        tol: float
            The tolerance for convergence.
        max_iter: int
            The maximum number of iterations.
        verbose: bool
            If True, prints iteration details.
            
    Returns:
        y: np.ndarray
            The approximate root.
        iter_stop: int
            Iteration i when method terminated
    """
    # Ensure y is a numpy array (even if 0-d scalar) to handle math uniformly
    y = np.array(y0, dtype=float)
    
    # Determine if we are in scalar mode based on input shape
    is_scalar = y.ndim == 0
    
    for i in range(max_iter):        
        # Calculate update step: y_new = y - J_fy(y)^(-1) @ f(y) = y - delta
        if is_scalar:
            delta = f(y) / J_fy(y)
        else:
            delta = np.linalg.solve(J_fy(y), f(y)) # Solves J * delta = f(y) for delta
        y_new = y - delta
        delta_max = np.abs(delta) if is_scalar else np.max(np.abs(delta)) # Define for termination check

        # Print output for debugging if verbose
        if verbose:
            y_print = f"{y_new:.6f}" if is_scalar else str(np.round(y_new, 4))
            print(f"{i:<5} | {delta_max:<15.2e} | {y_print}")

        # Check termination condition
        if delta_max < tol:
            return y_new, i + 1
        
        # Proceed to next iteration
        y = y_new
        
    # If we pass max_iter without convergence, return last estimate with failure flag
    return y, max_iter

def debug_check_jacobian(g, J_g, y_point, epsilon=1e-6):
    """
    Checks if the analytic Jacobian J_g matches the numerical finite difference of g.
    """
    # print("\n--- DEBUG: Jacobian Check ---")
    
    # 1. Analytic Jacobian
    J_analytic = J_g(y_point)
    if np.ndim(J_analytic) == 0: J_analytic = np.array([[J_analytic]])
    
    # 2. Numerical Jacobian (Finite Difference)
    dim = y_point.size
    J_num = np.zeros_like(J_analytic)
    y_flat = y_point.flatten()
    
    # Perturb each dimension
    for i in range(dim):
        y_perturb = y_flat.copy()
        y_perturb[i] += epsilon
        
        # Calculate slope: (g(y+e) - g(y)) / e
        g_plus = g(y_perturb.reshape(y_point.shape))
        g_base = g(y_point)
        col = (g_plus - g_base) / epsilon
        
        J_num[:, i] = col.flatten()
        
    # 3. Compare
    diff = np.linalg.norm(J_analytic - J_num)
    # print(f"Analytic J:\n{J_analytic}")
    # print(f"Numerical J:\n{J_num}")
    # print(f"Difference Norm: {diff:.2e}")
    
    if diff > 1e-4:
        print(">>> CRITICAL WARNING: Jacobian is incorrect! <<<")
    else:
        print(">>> Jacobian looks consistent. <<<")