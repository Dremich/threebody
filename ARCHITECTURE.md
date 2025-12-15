# Architecture

## 1. High-level data flow

- Orbit definition (JSON) → `threebody.problems` → `ThreeBodyProblem`
- `ThreeBodyProblem` → `threebody.simulate.simulate(...)` → `SimulationResult`
- `SimulationResult` →
  - diagnostics export (`np.savez`) via scripts
  - interactive visualization via `threebody.visualize`
  - headless frame rendering via `threebody.visualize.render_frames`

One-way dependencies (enforced by intent):

- `problems` → (data only)
- `dynamics` → (physics RHS + invariants only)
- `integrators` → (time stepping only)
- `controllers` → (adaptivity policy only)
- `simulate` → composes `dynamics` + `integrators` + `controllers`
- `visualize` → consumes arrays only; does not call numerics
- `scripts/*` → orchestration; composes `problems` + `simulate` + `visualize`


## 2. Core numerical pipeline

### `dynamics.py`

- Role: vector field and diagnostics for the three-body ODE.
- Primary interfaces:
  - `rhs(t: float, y: np.ndarray, *, params=...) -> np.ndarray`
  - `energy(y: np.ndarray, *, params=...) -> float`
  - `split_state(y) -> (r, v)` and `pack_state(r, v) -> y`
- Inputs/outputs:
  - Input `y` is a flat 1D state vector (see invariants below).
  - Output is a flat derivative vector of the same length.
- No integration logic, no adaptivity, no I/O.

### `integrators.py`

- Role: stable import surface for time-stepping algorithms.
- Interfaces exposed to the rest of the code:
  - Embedded explicit RK: `dopri5()`, `embedded_rk_step(...)`
  - Implicit placeholder: `implicit_midpoint_step(...)` (not implemented)
- Stepping interface used by the adaptive loop:
  - `stepper(f, t, y, h) -> (y_trial, err, nfev)`
  - `err` is an *error estimate vector* compatible with norm-based control.
- No knowledge of three-body state layout; works on generic 1D arrays.

### `controllers.py`

- Role: convert an error estimate vector into accept/reject decisions and a next step size.
- Key pieces:
  - `scaled_rms_error_norm(err, y, y_trial, atol, rtol) -> float`
  - `StepSizeController` protocol:
    - `init(t0, y0, h0)`
    - `error_norm(t, y, y_trial, err) -> float`
    - `accept(err_norm) -> bool`
    - `propose_step_size(h, err_norm, accepted) -> float`
  - `RKAdaptiveController`: minimal controller implementing the protocol.
- Policy only: does not evaluate `rhs`, does not allocate trajectories, does not plot.

### `simulate.py`

- Role: run the adaptive loop, record diagnostics, return arrays.
- Two layers:
  - `simulate_adaptive(f, y0, t_span, h0, stepper, controller, energy_fn) -> SimulationResult`
    - Generic, physics-agnostic driver.
    - Owns the accept/reject loop, end-time handling, and recording.
  - `simulate(problem, t_span, tol, h0) -> SimulationResult`
    - Project-level wrapper that wires together:
      - `dynamics.rhs` and `dynamics.energy`
      - `integrators.dopri5` + `integrators.embedded_rk_step`
      - `controllers.RKAdaptiveController`
- Output structure:
  - `SimulationResult.t`: monotone time samples (accepted steps)
  - `SimulationResult.y`: accepted states, shape `(n, d)`
  - `SimulationResult.h`: accepted step sizes, length `n-1`
  - `SimulationResult.err_norm`, `SimulationResult.nfev`
  - optional `energy` and `energy_drift`


## 3. Problem definitions

### `problems.py`

- Role: load orbit definitions from JSON and validate/normalize them.
- Primary interfaces:
  - `list_orbits() -> list[str]`
  - `load_orbit(name) -> ThreeBodyProblem`
  - `load_problem(name) -> ThreeBodyProblem` (alias used by scripts)
- JSON consumption:
  - Required: `name` (or inferred from filename), `masses`, `y0`
  - Optional: `period`, `G`, `description`, `reference`
- State normalization:
  - If `y0` is length 12 (planar), it is lifted to length 18 by inserting `z=0` and `vz=0`.
  - If `y0` is length 18, it is used as-is.


## 4. Visualization and rendering

### `visualize.py`

- Role: render trajectories from arrays; no calls into numerics.
- Two modes implemented with shared drawing/update logic:
  - Interactive:
    - one figure, three colored traces, fading trail
    - blitting-based redraw (cached backgrounds), no `FuncAnimation`
    - keyboard/mouse controls update an index `i` and redraw artists
  - Headless rendering:
    - `render_frames(states, out_dir, dpi=150)`
    - fixed axis limits computed from the full orbit
    - saves deterministic PNG filenames per orbit directory
- Input contract:
  - `states` is `(n, d)` with `d` in `{12, 18}`; only `(x,y)` are visualized.


## 5. Script-level orchestration

Scripts are thin glue: argument parsing + composition of library functions.

### `scripts/run_orbit.py`

- Composes: `problems.load_problem` → `simulate.simulate` → optional visualization / export.
- Outputs:
  - optional `.npz` containing arrays (`t`, `y`, `energy`, `h`, `err_norm`, `nfev`).
- Does not import `dynamics`, `integrators`, or `controllers`.

### `scripts/batch_render.py`

- Composes: orbit list → simulation → uniform-in-time resampling → `visualize.render_frames`.
- Uniform sampling is by time interpolation (not by adaptive step count).
- Produces:
  - `renders/<orbit_name>/frame_000000.png`, ...
  - `renders/<orbit_name>.gif` assembled from frames (Pillow).
- Does not import `dynamics`, `integrators`, or `controllers`.

### `scripts/make_mosaic.py`

- Pure image processing:
  - loads per-orbit GIFs
  - normalizes frame count and tile resolution
  - tiles into a grid and writes a final mosaic GIF
- No imports from `threebody.*` are required for correctness.


## 6. Data formats and invariants

### State vector layout

- Spatial (internal canonical):
  - `y = [x1,y1,z1,x2,y2,z2,x3,y3,z3,vx1,vy1,vz1,vx2,vy2,vz2,vx3,vy3,vz3]`
- Planar (accepted at input only):
  - `y0 = [x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3]`
  - lifted to 18D with `z=vz=0`

### Diagnostics collected

- Always (adaptive loop): `t`, `y`, `h`, `err_norm`, `nfev`
- Optional (if energy function is provided): `energy`, `energy_drift = energy - energy[0]`

### Assumptions

- Planar subspace is represented as `z=vz=0` at initialization; there is no additional constraint enforcement.
- Center-of-mass or momentum constraints are not imposed by the library; they may be encoded in the input data.


## 7. Design constraints and non-goals

- No SciPy integrators; stepping is implemented directly with NumPy.
- No event handling (collisions, close-approach termination, root finding).
- No regularization, no special handling of near-singular configurations.
- No symplectic or geometric integration guarantees (the default integrator is an embedded explicit RK pair).
- No automatic periodic-orbit search; the code consumes pre-defined initial conditions.
- No general N-body abstraction layer; the focus is a fixed three-body layout and simple interfaces.
