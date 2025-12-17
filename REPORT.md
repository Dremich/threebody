# threebody
Adaptive Numerical Simulation of Periodic Three-Body Orbits

- Description of the 3-body problem
- Why it is a difficult and interesting problem
- Overview of the numerical approach

Mathematical model

Numerical methods

Validation
 - Figure-eight orbit
 - Period error
 - Energy drift

Examples
 - Figure-eight, interactive
 - Long-term stability
 - Multiple orbits

Usage examples
```python scripts/run_orbit.py figure_eight --tol 1e-10 --visualize
python scripts/batch_render.py --out-dir renders/
python scripts/make_mosaic.py --input-dir renders/ --out mosaic.gif```

AI Transparency Statement
I used AI-assisted tools (GitHub Copilot) to build the auxiliary project components (visualizer, file I/O, JSON parsing) and scaffold the project. All of the numerical algorithms, (equations of motion, Runge-Kutta/implicit IVP solvers, adaptive step-controllers) I implemented myself. All analyses, verification, and results reflect my own independent work and judgement.