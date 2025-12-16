
import sys
from pathlib import Path
# Go up to the parent directory (..), then down into "src/threebody"
# This adds "../src/threebody" to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "threebody"))

from implicit_integrators import adaptive_bdf_coeffs
import numpy as np

def test_bdf_coeffs():
    # --- Example Usage for BDF-2 ---
    # Suppose t = [1.5, 1.2, 1.0] 
    # (current time 1.5, last step 0.3, previous step 0.2)
    t_points = [0.6, 1.0, 1.2, 1.5]

    y_coeffs, _, _ = adaptive_bdf_coeffs(np.array(t_points))
    print(f"Coefficients: {y_coeffs}")

    # Verify against the manual formula derived earlier:
    h_n = 0.3
    h_prev = 0.2
    c0 = (2*h_n + h_prev) / (h_n * (h_n + h_prev))
    c1 = -(h_n + h_prev) / (h_n * h_prev)
    c2 = h_n / (h_prev * (h_n + h_prev))
    print(f"Manual Check: {[c0, c1, c2]}")
    
    # Check if the computed coefficients match the manual check
    assert np.allclose(y_coeffs, [c0, c1, c2]), "BDF coefficients do not match manual check"
    print("Test 1 Passed: BDF coefficients match manual check")
    
def test_bdf2_uniform_grid():
    # Setup: Uniform grid t = [0, 1, 2, 3]
    # Current time t_n = 3. Step size h = 1.
    # k = 2 (Requires 4 points total: 3 for method, 1 extra for error)
    t_past = np.array([0.0, 1.0, 2.0, 3.0])
    
    y_coeffs, d_kp1, err_coeffs = adaptive_bdf_coeffs(t_past)
    
    print("\n--- Test Results (BDF-2 Uniform) ---")
    print(f"Y Coeffs: {y_coeffs}")
    print(f"Error Constant: {d_kp1}")
    print(f"Error Weights: {err_coeffs}")

    # 1. Verify BDF Coefficients
    # Standard BDF-2: 1.5*y_n - 2*y_n-1 + 0.5*y_n-2
    expected_y = np.array([1.5, -2.0, 0.5])
    np.testing.assert_allclose(y_coeffs, expected_y, atol=1e-10)
    
    # 2. Verify Error Constant (d_kp1)
    # Residual of t^3 for BDF-2 coeffs:
    # term = 1.5(0)^3 - 2(-1)^3 + 0.5(-2)^3
    #      = 0 - 2(-1) + 0.5(-8) = 2 - 4 = -2
    # d_kp1 = -2 / 3! = -2/6 = -1/3
    expected_d = -1.0 / 3.0
    assert abs(d_kp1 - expected_d) < 1e-10, "Error constant does not match expected value"
    
    # 3. Verify Error Weights (Approximating 3rd derivative)
    # Standard FD for y''' on uniform grid: [1, -3, 3, -1]
    # (Applied to t_n, t_n-1, t_n-2, t_n-3)
    expected_err = np.array([1.0, -3.0, 3.0, -1.0])
    np.testing.assert_allclose(err_coeffs, expected_err, atol=1e-10)
    print("Test 2 Passed: Error coefficients match expected values")
    
# Run the test
test_bdf_coeffs()
test_bdf2_uniform_grid()