import sys
from pathlib import Path

# Go up to the parent directory (..), then down into "scripts"
# This adds "../scripts" to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# Now run_orbit is directly visible to Python
from run_orbit import run

run(
    problem_name="triangle_attempt",
    t_mult=10.0,
    tol=1e-10,
    h0=1e-3,
    visualizer=True,        # critical for tests
    output_dir="tests/triangle_attempt_test",  # critical for tests
)

# outputs = list(Path("tests/triangle_attempt_test").glob("*"))
# assert len(outputs) > 0
