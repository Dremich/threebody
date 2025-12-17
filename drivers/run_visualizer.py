import sys
import numpy as np
from pathlib import Path

# Go up to the parent directory (..), then down into "scripts"
# This adds "../scripts" to the python search path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Now run_orbit is directly visible to Python
from src.threebody.visualize import visualize



## THIS SCRIPT VISUALIZES THE OUTPUT OF A PREVIOUSLY SOLVED PROBLEM, SAVED IN DATA/COMPUTATIONS 

datastr = "goggles_RK" # name of problem, without .npz. Example: "figure_eight_RK"

##############################################################################################



# Loading precomputed data
filepath = Path(f"data/computations/{datastr}.npz")
data = np.load(filepath)

t = data['t']
y = data['y']
energy = data['energy']
nfev = data['nfev']
nstepper = data['nstepper']
time_elapsed = data.get('time_elapsed', 0.0)
if energy.size == 0:
        energy = None

# Print output
method = "RK" if "RK" in datastr else "BDF"
print(f"Solved {datastr} using {method} in {time_elapsed:.6f} seconds")
print(f"Initial state: {y[0]}")
print(f"Final state: {y[-1]}\n")
print(f"Initial energy: {energy[0] if energy is not None else 'N/A'}")
print(f"Final energy: {energy[-1] if energy is not None else 'N/A'}")
print(f"Energy drift: {energy[-1] - energy[0] if energy is not None else 'N/A'}\n")

if method == "RK":    
    print("Total function evaluations (RK):", nfev, "\n")
    print("Total attempted steps (RK):", nstepper, "\n")
else:
    print("Total number of Newton Raphson evaluations (BDF):", nfev)
    print("Total attempted steps (BDF):", nstepper, "\n")

# Visualize the result
visualize(
    y,
    t=t,
    energy=energy,
    show_energy=(energy is not None),
    show_full_trajectory=bool(False),
)


    
