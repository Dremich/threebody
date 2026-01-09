## Test Results

# Testing Jacobian
Results of my analytical jacobian agree with a numerically computed jacobian, suggesting both are accurate.

# Testing Newton-Raphson
Newton-Raphson converges for quadratic case, and intersection between circle and hyperbola.

# Testing Adaptive BDF
Solution demonstrates O(h^(k+1)) local truncation error on the model problem e^x for both uniform and non-uniform grids. Error estimate matches theoretical error estimate from exact Taylor series.