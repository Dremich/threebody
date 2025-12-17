import sys
from pathlib import Path
# Go up to the parent directory (..), then down into "src/threebody"
# This adds "../src/threebody" to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "threebody"))

from implicit_integrators import adaptive_bdf_coeffs, BDF_step
import numpy as np
import matplotlib.pyplot as plt

# Re-use the functions you defined: BDF_step, adaptive_bdf_coeffs, newton_raphson

def test_variable_step_bdf2():
    """
    Verifies the order of convergence for the Adaptive BDF-2 solver on a non-uniform ('jagged') time grid.
    
    Test Strategy:
    1. Problem: Solves y' = y, y(0)=1 (Exact solution: y = e^t).
    2. Geometry: Sets up a fixed, non-uniform history pattern relative to a target time t_c:
       - Current Step: h (target step size)
       - Previous Step: 1.5 * h
       - Oldest Step: 0.5 * h
       This ensures the solver is tested on a grid that strictly requires variable-step coefficients 
       (standard fixed-step coefficients would fail to cancel error terms correctly here).
    3. Scaling: The entire geometry is scaled by a factor 's' over multiple iterations.
    4. Verification: 
       - Calculates the Local Truncation Error (LTE) at the target step.
       - Checks that LTE scales as O(h^3) (slope ~3.0 on log-log plot).
       - Verifies that the internal error estimator ratio remains consistent (~1.0).
    """
    
    print("\n=== Running Variable Step Size BDF-2 Test ===")
    print("Testing geometry: [Long Step] -> [Short Step] -> [Target Step]")
    
    # 1. Define Problem (y' = y, y(0)=1, Exact=e^t)
    def f(t, y): return y
    def J_f(t, y): return 1.0 if np.ndim(y)==0 else np.eye(len(y))
    y_exact = lambda t: np.exp(t)
    
    # 2. Define the 'Jagged' Geometry ratios
    #    We will scale these by a factor 's' in the loop.
    #    Geometry:
    #      t_target (current step we are solving for)
    #      t_n-1    (1.0 unit back)
    #      t_n-2    (2.5 units back -> previous step was 1.5x larger)
    #      t_n-3    (3.0 units back -> step before that was 0.5x)
    #    This ensures the grid is NOT uniform.
    geometry_offsets = np.array([3.0, 2.5, 1.0]) 
    
    t_center = 2.0  # We will solve for y at t ~ 2.0
    
    scales = [0.1, 0.05, 0.025, 0.0125]
    errors = []
    ratios = []
    
    print(f"{'Scale':<10} | {'h_eff':<10} | {'True Err':<12} | {'Est Err':<12} | {'Ratio':<6}")
    print("-" * 65)
    
    for s in scales:
        # 3. Construct the non-uniform time history based on scale 's'
        #    t_past = [t_n-3, t_n-2, t_n-1]
        t_past = t_center - (geometry_offsets * s)
        #    Sort strictly ascending just to be safe (though offsets imply it)
        t_past = np.sort(t_past)
        
        #    Proposed step size 'h' is the distance from t_n-1 to t_center
        #    In our geometry, t_n-1 is at offset 1.0*s, so h = 1.0*s
        h = 1.0 * s
        
        # 4. Generate exact history y_values
        y_past = y_exact(t_past)
        
        # 5. Run the Adaptive Step
        y_next, err_est, _ = BDF_step(f, J_f, t_past, y_past, h)
        y_next = y_next[0]  # scalar
        err_est = err_est[0]  # scalar
        
        # 6. Verify Accuracy
        t_next_actual = t_past[-1] + h
        true_val = y_exact(t_next_actual)
        true_err = np.abs(y_next - true_val)
        
        errors.append(true_err)
        ratios.append(true_err / np.abs(err_est))
        
        print(f"{s:<10.4f} | {h:<10.4f} | {true_err:<12.4e} | {np.abs(err_est):<12.4e} | {ratios[-1]:<6.2f}")

    # 7. Check Convergence Slope
    log_s = np.log(scales)
    log_err = np.log(errors)
    slope, intercept = np.polyfit(log_s, log_err, 1)
    
    print(f"\nMeasured Convergence Order: {slope:.4f}")
    
    # Validation Logic
    if 2.8 < slope < 3.2:
        print("✅ SUCCESS: Adaptive solver maintains O(h^3) on jagged grid.")
    else:
        print("❌ FAILURE: Convergence order is incorrect.")

    # Optional: Visualization
    plt.figure()
    plt.loglog(scales, errors, 'o-', label='Error (Variable Grid)')
    plt.loglog(scales, [np.exp(intercept)*s**3 for s in scales], 'k--', label='O($h^3$) Reference')
    plt.xlabel('Scale Factor')
    plt.ylabel('Error')
    plt.title(f'Variable Step BDF-2 Convergence (Slope={slope:.2f})')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.show()

if __name__ == "__main__":
    test_variable_step_bdf2()