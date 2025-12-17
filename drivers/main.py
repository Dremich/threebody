import sys
from pathlib import Path

# Go up to the parent directory (..), then down into "scripts"
# This adds "../scripts" to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# Now run_orbit is directly visible to Python
from run_orbit import run

problem = "moth_I"
run(
    problem_name=problem,
    t_mult=1.0,
    tol=1e-10,
    h0=1e-3,
    visualizer=True,        # critical for tests
    method="BDF",
    output_dir=f"tests/{problem}",  # critical for tests
    show_full_trajectory=False,
)

# outputs = list(Path("tests/figure_eight_test").glob("*"))
# assert len(outputs) > 0
