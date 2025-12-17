import sys
from pathlib import Path

# Go up to the parent directory (..), then down into "scripts"
# This adds "../scripts" to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# Now run_orbit is directly visible to Python
from run_orbit import run

## THIS SCRIPT SOLVES THE PROBLEM BELOW USING METHOD AND SAVES THE OUTPUT TO DATA/COMPUTATIONS 

problem = "yarn"
method = "RK" # RK or BDF

##############################################################################################

run(
    problem_name=problem,
    t_mult=1.0,
    tol=1e-10,
    h0=1e-3,
    visualizer=True,        # critical for tests
    method=method,
    output_dir=f"data/computations/{problem}_{method}",  # critical for tests
    show_full_trajectory=False,
)
