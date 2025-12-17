import sys
from pathlib import Path
# Go up to the parent directory (..), then down into "src/threebody"
# This adds "../src/threebody" to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "threebody"))

from implicit_integrators import newton_raphson

import numpy as np
import unittest

class TestNewtonRaphson(unittest.TestCase):

    def test_scalar_quadratic(self):
        """
        Test finding the square root of 2: f(y) = y^2 - 2 = 0
        """
        print("\n--- Test: Scalar Quadratic (y^2 - 2) ---")
        
        def f(y): 
            return y**2 - 2.0
            
        def J(y): 
            # Jacobian must return a 1x1 matrix (array of arrays) or scalar
            # to be compatible with linalg.solve? 
            # Ideally your NR handles scalar returns gracefully.
            return 2.0 * y

        # Guess 1.0, Expect ~1.414
        y0 = 1.0
        root, iters = newton_raphson(f, J, y0, tol=1e-8, verbose=True)
        
        self.assertAlmostEqual(root, np.sqrt(2), places=7)
        print(f"Scalar converged in {iters} iters.")

    def test_vector_system(self):
        """
        Test intersection of a circle and a hyperbola:
        1) x^2 + y^2 - 4 = 0  (Circle radius 2)
        2) xy - 1 = 0         (Hyperbola)
        
        Exact roots include approx (1.93, 0.518)
        """
        print("\n--- Test: 2D System (Circle + Hyperbola) ---")
        
        def f(vec):
            x, y = vec
            return np.array([
                x**2 + y**2 - 4.0,
                x*y - 1.0
            ])
            
        def J(vec):
            x, y = vec
            # Row 0: df1/dx, df1/dy
            # Row 1: df2/dx, df2/dy
            return np.array([
                [2*x, 2*y],
                [y,   x]
            ])

        y0 = np.array([2.0, 0.5]) # Guess close to a root
        root, iters = newton_raphson(f, J, y0, tol=1e-8, verbose=True)
        
        # Verify result satisfies f(root) approx 0
        res = f(root)
        np.testing.assert_allclose(res, np.zeros(2), atol=1e-6)
        print(f"Vector system converged in {iters} iters to {root}")

if __name__ == '__main__':
    # Assume your newton_raphson is defined in the same file or imported
    # If copy-pasting, put your definition above this block
    unittest.main(argv=['first-arg-is-ignored'], exit=False)