## Script to test the implementation of the analytical Jacobian matrix in dynamics.py
## This script compares the analytical Jacobian matrix with a numerical Jacobian 
## matrix computed using finite differences.
##
## Written by Gemini 3 Pro





import sys
from pathlib import Path
# Go up to the parent directory (..), then down into "src/threebody"
# This adds "../src/threebody" to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "threebody"))

from dynamics import rhs, jacobian, DynamicsParams
import numpy as np


def test_jacobian_accuracy():
    print("--- Starting Jacobian Verification ---")
    
    # A. Setup Random State
    np.random.seed(42)
    # Random positions [-10, 10] and velocities [-1, 1]
    y0 = np.random.rand(18) * 20 - 10 
    y0[9:18] = (np.random.rand(9) * 2) - 1
    
    params = DynamicsParams()
    
    # B. Compute Analytic Jacobian (Your Code)
    # Note: Ensure your `jacobian` function is defined!
    try:
        J_analytic = jacobian(0.0, y0, params=params)
    except NameError:
        print("Error: 'jacobian' function not defined. Please paste your code in.")
        return

    # C. Compute Numerical Jacobian (Finite Difference)
    # J_ij = (f(y_j + h) - f(y_j - h)) / 2h
    J_num = np.zeros((18, 18))
    h = 1e-7 # Perturbation step size
    
    print(f"Computing numerical gradients (perturbation h={h})...")
    
    # We loop through every input variable (18 of them)
    for j in range(18):
        y_plus = y0.copy()
        y_minus = y0.copy()
        
        y_plus[j] += h
        y_minus[j] -= h
        
        # Evaluate Dynamics
        f_plus = rhs(0.0, y_plus, params=params)
        f_minus = rhs(0.0, y_minus, params=params)
        
        # Central Difference
        col_deriv = (f_plus - f_minus) / (2 * h)
        J_num[:, j] = col_deriv

    # D. Compare
    diff = np.abs(J_analytic - J_num)
    max_error = np.max(diff)
    
    print("\n--- Results ---")
    print(f"Max Absolute Error: {max_error:.2e}")
    
    # Check Blocks
    print("\nBlock Check:")
    print(f"Top-Right (Velocity dependence of Position): Max Error = {np.max(diff[0:9, 9:18]):.2e}")
    print(f"Bottom-Left (Position dependence of Gravity): Max Error = {np.max(diff[9:18, 0:9]):.2e}")
    print(f"Top-Left (Position dependence of Position):   Max Error = {np.max(diff[0:9, 0:9]):.2e} (Should be 0)")
    print(f"Bottom-Right (Velocity dependence of Acceleration):   Max Error = {np.max(diff[9:18, 9:18]):.2e} (Should be 0)")
    
    if np.allclose(J_analytic, J_num, rtol=1e-5, atol=1e-7):
        print("\n✅ SUCCESS: Analytical Jacobian matches Numerical Jacobian!")
    else:
        print("\n❌ FAILURE: Mismatch detected.")
        
# Run the test
test_jacobian_accuracy()