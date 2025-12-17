import sys
from pathlib import Path
# Go up to the parent directory (..), then down into "src/threebody"
# This adds "../src/threebody" to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "threebody"))

from implicit_integrators import adaptive_bdf_coeffs, BDF_step
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def test_bdf2_convergence():
    print("Running BDF-2 Convergence Test on y' = y...")
    
    # 1. Define Problem (y' = y)
    def f(t, y): return y
    def J_f(t, y): return 1.0 if np.ndim(y)==0 else np.eye(len(y))
    
    # 2. Setup Exact Solution for History
    #    BDF-2 requires 2 history points for the method + 1 extra for error est
    #    Total 3 points needed in history: [t_n-3, t_n-2, t_n-1]
    t_start = 1.0
    y_exact = lambda t: np.exp(t)
    
    # 3. Run Step for decreasing h
    h_values = [0.2, 0.1, 0.05, 0.025, 0.0125]
    errors = []
    
    print(f"{'h':<10} | {'Exact Err':<12} | {'Est Err':<12} | {'Ratio':<6}")
    print("-" * 50)

    for h in h_values:
        # Generate history [t-3h, t-2h, t-h]
        t_past = np.array([t_start - 3*h, t_start - 2*h, t_start - h])
        y_past = y_exact(t_past) # Scalar input
        
        # Take one step
        y_next, err_est, _ = BDF_step(f, J_f, t_past, y_past, h)
        y_next = y_next[0]  # Scalar output
        err_est = err_est[0]  # Scalar output
        
        # Calculate Exact LTE (Difference from exact solution at t_start)
        # Note: We stepped FROM t_start-h TO t_start.
        # So y_next corresponds to time t_start.
        exact_val = y_exact(t_start)
        exact_err = np.abs(y_next - exact_val)
        
        errors.append(exact_err)
        print(f"{h:<10.4f} | {exact_err:<12.4e} | {np.abs(err_est):<12.4e} | {exact_err/np.abs(err_est):<6.2f}")

    # 4. Analyze Convergence Rate
    #    Slope = log(err2/err1) / log(h2/h1)
    #    Expected: ~3.0 for LTE of BDF-2
    log_h = np.log(h_values)
    log_err = np.log(errors)
    slope, intercept = np.polyfit(log_h, log_err, 1)
    
    print("\nConvergence Analysis:")
    print(f"Calculated Slope (Order + 1): {slope:.4f}")
    print("Expected Slope for BDF-2 LTE: 3.0000")
    
    if 2.8 < slope < 3.2:
        print("✅ PASSED: O(h^3) Local Error Convergence confirmed.")
    else:
        print("❌ FAILED: Convergence rate incorrect.")

    # Visualization
    plt.figure()
    plt.loglog(h_values, errors, 'o-', label='Actual Error')
    plt.loglog(h_values, [np.exp(intercept)*h**3 for h in h_values], 'k--', label='O($h^3$) ref')
    plt.xlabel('Step size h')
    plt.ylabel('Local Truncation Error')
    plt.title(f'BDF-2 Convergence (Slope={slope:.2f})')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.show()


test_bdf2_convergence()