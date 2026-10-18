# Project Handoff — 3D Autonomous Ground-Robot Planning and Safety

> **Project name:** Deliberately undecided. Do not invent, infer, or brand the project with a name until the team chooses one.
>
> **Status:** Concept committed; architecture and implementation details below are a proposed execution plan, not claims of completed work.
>
> **Audience:** Teammates and AI coding agents (including Claude). Read this before proposing architecture, writing code, or changing scope.
>
> **Timeframe:** Competitive hackathon; assume approximately 36 hours as a planning baseline, adaptable to 24–48 hours. The aim is an impressive, technically defensible working prototype, not a paper reproduction or full autonomous-driving stack.

## 0. Read this first: the project in plain English

We are building a **fully interactive 3D robotics simulation of an autonomous four-wheel ground rover / inspection cart navigating a changing industrial environment**. The robot must drive to a target while respecting steering geometry, speed, braking distance, and obstacles, including moving objects.

The technical centerpiece is a **trained conditional flow-matching neural network** that generates candidate **time-parameterized driving trajectories**, plus a **Control Barrier Function (CBF)-based safety filter** that can override unsafe steering or braking commands at execution time. When the environment changes, the system detects invalid plans, samples new candidate trajectories, and continues navigating. The interface lets judges interfere with the environment and visualize exactly what the neural planner proposed versus what the safety controller executed.

**Pitch in two sentences:** We train a generative motion planner to create feasible driving trajectories for an autonomous ground rover in a 3D industrial environment. A mathematical safety layer continuously checks and corrects its actions, so the user can introduce moving obstacles and watch the robot replan, brake, or steer while seeing the AI's intended path and the safety interventions.

**Not:** a drone, drone swarm, security-defense game, 2D pathfinding toy, A* wrapped in an AI interface, generic chatbot, or a scripted car following predetermined tracks.

**Important:** The ML should actually generate novel trajectories on held-out maps. The safety layer should actually influence control commands, not be a decorative red warning label. The 3D physics world should constrain movement and determine whether collisions occurred, not merely render a 2D path as a 3D animation.

## 1. Intent, context, and non-negotiables

### Why we're building it

We want a project with technical ambition comparable to recent robotics/ML research: modern generative modeling, control theory, robotic motion planning, and convincing 3D presentation. The result should both **look impressive** and stand up to technical questions about its actual algorithmic implementation.

The team previously built [Swarm](https://github.com/ian-kr/Swarm), a 3D drone-swarm defense trainer with PPO-trained agents and A*-style baselines. **We do not want another drone project.** Differentiate this one through an autonomous **ground rover**, **flow-matching trajectory generation rather than PPO**, **continuous nonholonomic vehicle dynamics**, and an **explicit mathematical safety filter**. 3D graphics alone do not differentiate it from Swarm.

### Fixed decisions

- Primary agent: **one autonomous wheeled ground rover / inspection cart**. Use a compact, believable four-wheel, car-like rover by default; final 3D chassis style can be changed without changing the algorithm.
- Main scene: **real 3D industrial or inspection facility**: corridors, equipment, shelves, crates, loading areas, a crossing vehicle, possibly ramps or uneven terrain.
- Core ML: **conditional flow matching** trained using expert driving trajectories generated in simulation; actual PyTorch training required.
- Core non-ML technical element: **CBF-based online safety filtering**, preferably with a small quadratic program (QP), backed by physically meaningful dynamics and a safe fallback.
- Meaningful comparison against a **classical vehicle-aware planner**, not only a straight-line or random baseline.
- No real hardware required; real robot deployment, SLAM, learned camera perception, and multi-robot coordination are out of scope for MVP.
- Product name is TBD. Do **not** reuse the previous working title or introduce any new title in UI, code headers, slides, or repo name unless the team chooses one.

### Flexible decisions

- Exact rendering stack, asset selection, 3D map design, vehicle model detail, model size, training volume, and visual theme can be adjusted for reliability.
- If the team chooses a warehouse cart rather than an inspection rover, retain the underlying car-like steering, safety, and generative planning design.
- The core route-planning mathematics may initially use a ground-plane projection, **provided** the simulation runs in 3D and incorporates the robot's real footprint/dynamics. Ramps, grade limits, roll stability, and elevation-aware costs are high-value extensions, not reasons to break the MVP.

## 2. User experience / intended hackathon demo

1. The application opens to a **polished 3D industrial facility** with a rover, obstacles, and a selectable destination. Free camera, cinematic follow camera, and optional overhead inspection view.
2. The judge clicks a target location. The system samples **multiple different candidate paths** from a trained flow model. Their geometry, curvature, relative safety, and estimated arrival times are visualized in-world as thin glowing ribbons.
3. A selected trajectory is tracked by the simulated rover with **continuous acceleration, braking, and steering**, visibly respecting its turning radius and speed limits.
4. The judge enables an obstacle event: e.g. a forklift crosses the route, a moving barrier starts to close, or a cart enters the lane. This is an actual change to obstacle state in the simulator.
5. The robot's **safety filter intervenes**: reduce throttle, brake, or choose an allowed steering action. The interface indicates the neural planner's proposed command and the corrected command. If the route is blocked, replan and show the replacement path.
6. Toggle **safety filter on/off** in a *deterministic replay*, using the same scenario seed and obstacle schedule. Show real collision outcomes and safety intervention events. Do not fake or force a crash; use scenarios where the unfiltered controller really does collide or violate a safety margin.
7. Show a lightweight experiment summary: goal reach rate, collision rate, minimum clearance, path efficiency, planning latency, and CBF interventions. Judges should be able to understand the difference between *generating a path* and *controlling it safely*.

**Visual direction:** industrial 3D robotics laboratory / modern warehouse, strong lighting, sensible materials and geometry, high legibility. The rover should look like a real machine rather than a 2D dot or hovering sprite. The primary UI is the simulation, not a dashboard. Overlays should expose algorithm behavior: candidate trajectories, upcoming hazards, safe-distance envelopes, velocity, controller corrections, and optionally a compact neural-network inference panel.

**Hero demonstration:** a robot approaches a narrow aisle while a crossing forklift enters its route. We show the learned path, the hazardous projected action, the safety adjustment, and a new path around the obstruction (if a feasible path exists). Playback and camera controls make the event easy to inspect.

## 3. High-level system architecture

```text
Browser (React + React Three Fiber / Three.js)
  - 3D world, rover visualization, camera, object placement, controls
  - selected goal, spawned obstacles, paths, status, deterministic replays
                         | WebSocket / HTTP
                         v
Python backend (FastAPI or equivalent)
  - session state / scenario seed / simulation clock
  - robot state, map geometry, obstacle states, collision queries
  - expert trajectory planner (offline dataset generation + baseline)
  - conditional flow-matching PyTorch model (inference)
  - route feasibility / ranking / path tracking
  - CBF-style safety QP + emergency fallback
  - synchronized physics stepping and telemetry
                         |
                         v
Physics/simulation layer (PyBullet or practical alternative)
  - ground, solid obstacles, collision detection, robot movement
  - moving obstacles with scripted trajectories in MVP
  - definitive realized state and collisions, not neural guesses

Offline ML pipeline (Python): scene generation -> expert plans -> dataset
  -> flow matching training -> checkpoints -> held-out evaluation
```

**Keep simulation authoritative.** The backend owns positions, timing, world state, and collision outcomes; the frontend renders state snapshots and interpolates between them. The frontend must not independently invent physics outcomes.

### Coordinate conventions (define and test immediately)

- Backend: `(x, y, z)` world coordinates with **z up**, distances in meters, time in seconds; heading `psi` rotates in the ground XY plane.
- Frontend/Three.js: Y up. One valid conversion is `(x, y, z)_backend -> (x, z, -y)_frontend`; centralize it in a tested coordinate utility, including yaw and path points.
- Model local frame: x-forward and y-left relative to rover; normalize all model inputs/outputs consistently.
- Use a single simulation clock and deterministic seed for scenario replays.

## 4. What makes this a real 3D robotics project?

An autonomous ground rover isn't free to fly in XYZ like a drone. It moves on surfaces under contact and nonholonomic steering constraints. **Do not pretend it has arbitrary 3D movement.** Instead:

- The **world and collision objects are 3D**: shelves, corners, forklifts, boxes, beams, barriers, elevated surfaces.
- The rover has a **3D rigid-body footprint and ground contact**: obstacle clearance and turning width matter.
- Commands are **steering and acceleration/braking**, not arbitrary teleportation or instantaneous changes in heading.
- Physical tracking limits matter: maximum steering angle, maximum acceleration/deceleration, vehicle size, turning radius, and limits under load/slope if enabled.
- A 2D/2.5D occupancy or height-map projection can be used for efficient route conditioning, but vertical clearance, slope, and 3D collision checking should feed into route feasibility where implemented.
- For MVP, use mostly level terrain with genuinely 3D obstacles, moving entities, and continuous car-like motion. Add ramps/terrain-aware speed and grade constraints only after the integrated core works.

### Recommended motion model for first working implementation

A **kinematic bicycle model** is a good starting point (not a balancing two-wheel bicycle: it is a standard approximation of four-wheel, front-steered vehicles):

```text
state s = (x, y, psi, v)
control u = (a, delta)

x_dot   = v * cos(psi)
y_dot   = v * sin(psi)
psi_dot = (v / L) * tan(delta)
v_dot   = a

where:
  L = wheelbase
  a = longitudinal acceleration or deceleration
  delta = front-wheel steering angle
```

Enforce bounds on speed, acceleration, braking, steering angle, and ideally steering rate. This prevents unrealistically instantaneous sideways movement or heading changes. For a low-speed inspection rover, this model plus 3D collision checks is a defensible MVP. If PyBullet has a stable suitable wheeled URDF and wheel-control setup, use it as the authoritative vehicle simulation; otherwise prefer a correct, deterministic kinematic rover with 3D collision checks over an unstable wheel-physics implementation.

Suggested **starting** parameters (tune, not scientific constants): max speed 2–3 m/s, wheelbase 0.55–0.9 m, acceleration about 1 m/s², braking 2–3 m/s², steering limit 25–35 degrees, control loop 10–30 Hz, physics/render updates decoupled. Never assume a turn is feasible without validation.

## 5. The ML: conditional flow matching for trajectories

### Concept

Train a generative model on many **successful, timed, physically feasible trajectories**. At inference it receives the robot's current situation and generates new possible motion trajectories that fit the destination and nearby obstacles.

This isn't a language model, pretrained chatbot, or a manually encoded A* policy. A trained network must actually produce path candidates, preferably several per query. Paths should be generated on maps **not seen in training**.

### A. Generate expert data

Procedure:

1. Procedurally construct diverse navigable map layouts (corridors, obstacles, tight turns, optional crossings); maintain a split by **map layout/seed**, not only by individual examples.
2. Sample feasible start states and goals.
3. Use a vehicle-aware planner (ideally **Hybrid A*** or RRT with motion primitives) to find collision-free routes in `(x, y, heading)`; plain grid A* alone is insufficient to guarantee drivable vehicle motion.
4. Smooth/time-parameterize candidate routes, impose velocity/steering/acceleration constraints, and validate by **actually rolling them out** through the chosen motion model / simulator.
5. Keep only validated expert trajectories, label map/vehicle/start/goal conditions and timing. Record failure cases separately for evaluation; avoid training impossible routes as positives.
6. Include variety: straight aisles, S-bends, U-shaped obstacles, narrow turning corridors, multiple equivalent routes, and random obstacle locations.

Important: static expert paths do not automatically teach behavior around moving obstacles. For the MVP, train first on **static scenes** and handle new moving hazards via the online controller plus replanning; afterward add time-indexed obstacle predictions to the model's input and generate dynamic-obstacle-aware expert trajectories.

### B. Data representation

An implementable starting choice:

```text
Condition c:
  local obstacle/occupancy raster in rover coordinates (e.g. 32x32 or 48x48)
  optional local terrain-height / drivable-surface channel
  current speed, heading / steering state and vehicle limits
  goal relative to rover (dx, dy, goal heading optional)
  optional nearby moving obstacle states (positions, velocities, sizes)

Target trajectory tau (for K steps, e.g. K=16 or 24):
  each step = [local_x, local_y, heading_sin, heading_cos, target_speed]
  OR controls [acceleration, steering] with corresponding rollout
  fixed horizon and sample interval; use consistent timestamps
```

**Preferred MVP choice:** timed local-frame **waypoints + heading + target speed**, because they are easier to generate/train/visualize. A separate vehicle-aware trajectory tracker converts these into nominal steering and acceleration, which the safety filter can correct. **Validate dynamics after generation**—a waypoint predictor alone does not make a feasible vehicle path. A trajectory-of-controls model is a stronger extension with more difficult training.

Include normalization, consistent wraparound representation for heading, and a fixed horizon. A path with merely pretty XY coordinates but no physically trackable controls is not an acceptable final result.

### C. Training objective (actual flow matching)

Let `tau` be a flattened expert trajectory and `epsilon ~ N(0, I)` a random noise trajectory representation. Sample `t ~ Uniform(0,1)` and interpolate:

```text
z_t = (1 - t) * epsilon + t * tau
v_target = tau - epsilon
loss = mean_squared_error(v_theta(z_t, t, condition), v_target)
```

The neural network learns a conditional *velocity field in trajectory space*: which direction noisy trajectory representations should move to become expert-like paths. At inference, begin with new random noise, integrate the learned field from `t=0` to `t=1` (e.g. a small Euler/Heun step count), decode and validate candidates.

This simplified straight-line conditional flow-matching recipe is a starting implementation, not a claim that every generated path will be safe or valid. Collision constraints and dynamics must still be checked separately.

### D. Lightweight architecture

- Small CNN encoder for obstacle/terrain raster; concatenate goal, robot state, time embedding, and current noisy trajectory.
- A compact MLP, residual MLP, or 1D temporal convolution network predicts the trajectory-space velocity field.
- Aim for a **small model first** (hundreds of thousands to low millions of parameters), not huge pretrained diffusion/video models.
- Train in PyTorch. Save model checkpoints, normalization statistics, splits, and training config.
- Measure generated diversity; sampling multiple noise seeds should produce more than identical jittered copies when alternatives exist.

### E. Candidate ranking / tracking / replanning

At planning time:

1. Encode the current local map, robot state, goal, and (if supported) moving obstacle estimates.
2. Sample N candidate trajectories using different noise seeds (start N around 4–16; tune latency).
3. Reject candidates colliding with **static/dynamic predicted obstacles**, impossible curvature, speed bounds, missing target, or irrecoverable tracking violations.
4. Rank accepted candidates for target progress, collision margin, smoothness, travel time, and dynamics/terrain cost.
5. Select a trajectory, generate nominal actions using a path tracker (pure pursuit or Stanley-style steering plus speed PID is acceptable), and execute through safety filtering.
6. On obstacle changes, failed tracking, stale plans, or fixed replanning intervals, generate a new set. Make repeated replanning conservative enough not to oscillate between paths.

If no valid learned candidate exists, expose it honestly and **fall back to a vehicle-aware classical planner or controlled stop**. This should be logged as a fallback, not credited as a neural success.

## 6. The control-theory centerpiece: safety filter

### Concept

The generative model proposes a desired trajectory. A lower-level controller generates its preferred steering/acceleration action (`u_nom`). The **safety layer** receives that action and returns a corrected action (`u_safe`) based on vehicle dynamics and obstacle positions / velocities.

```text
Flow model -> selected planned trajectory
         -> vehicle path tracker -> nominal command (steer, accelerate)
         -> CBF-style safety optimizer -> safe command (or emergency stop)
         -> physics simulator -> next real state
```

A CBF describes an allowable region by a function `h(state) >= 0`. For a simple circular obstacle approximation:

```text
h(p) = ||p_robot - p_obstacle||^2 - R_safe^2
R_safe = robot collision radius + obstacle radius + margin
```

**Very important mathematical caveat:** For a car-like vehicle with acceleration and steering as the controls, simply checking `h >= 0` or applying the first-order formula `h_dot + alpha*h >= 0` does *not* generally create valid constraints on both controls. Position typically has a higher relative degree with respect to acceleration/steering; obstacle velocity and braking distance matter. Claude must not implement a bogus “CBF” that ignores this and call it a formal safety guarantee.

### Feasible implementation options — choose one deliberately

**Option 1 (recommended early build): discrete-time, model-predictive CBF-style filter.**

- Take the current state, nominal control, obstacle states, and a short rollout horizon.
- Predict the next state(s) under the actual vehicle motion model.
- Require a discrete barrier condition such as `h(s_next) >= (1 - alpha)*h(s_now)` (with well-defined conditions, margins, and control bounds). For nonlinearity, linearize near the nominal control for a small QP, or solve a small constrained optimization; then verify candidates with a full simulation rollout.
- Optimize minimal deviation from nominal control: `min (u - u_nom)^T W (u - u_nom)` subject to safety and actuator constraints.
- Add a speed-aware stopping margin and safety buffers for dynamic obstacles; if constraints are infeasible or solver fails, command emergency braking / stop and flag the event.
- Call this a **CBF-inspired or discrete-time CBF safety filter**, not a formally certified collision-free controller unless proof assumptions and the implemented discretization justify the claim.

**Option 2 (advanced / research-grade): high-order CBF / exponential CBF QP.**

- Derive the barrier constraints from the chosen continuous-time dynamics and the control inputs, including obstacle motion and correct relative degree.
- One option is a speed/yaw-rate unicycle surrogate where acceleration and yaw rate affect the second derivative of distance. Enforce realistic car steering feasibility when converting yaw rate to steering angle. Another is a high-order / sampled-data derivation directly for the bicycle model.
- Validate feasibility, actuator bounds, sensing delay, and mismatch against the actual simulator. Formal guarantees belong only to the precise model and stated assumptions.

**Never** silently substitute a hard-coded `distance < threshold -> brake` and represent it as Control Barrier Functions. That can be an emergency fallback, but the intended technical contribution is optimization-based intervention from mathematical constraints.

### Minimum safety events to handle

- Static obstacle approached too closely.
- Moving cart/forklift crossing intended route at different speeds.
- Infeasible narrow corridor: robot should stop or choose a new plan instead of clipping geometry.
- Actuator limits / high-speed stopping distance.
- Model candidate that is geometrically plausible but dynamically impossible.

### What the interface should show

- Nominal model-generated route (distinct visual style).
- Selected feasible/tracked route and actually executed trail.
- Current and predicted obstacle envelopes / margins.
- Controller mode: **tracking / intervention / emergency stop / replanning / goal reached**.
- Numerical inputs/outputs when safety intervenes: proposed throttle/steering vs corrected throttle/steering.
- Toggle/replay with and without the filter, under identical starting states and obstacle schedules.

## 7. Dynamics, time, and simulation integration

**Single source of truth:** a Python simulator. The render loop is not authoritative.

Suggested time architecture (adjust to performance):

- Physical integration: fixed timestep ~1/60–1/120 s.
- Safety controller and tracking: ~10–30 Hz, or once each controlled physics step.
- Neural replanning: as-needed / ~1–5 Hz, conditioned on map changes and latency.
- Frontend updates: 20–30 Hz backend snapshots, interpolate visually to 60 FPS if possible.

### Moving obstacles

Initially use deterministic scripted moving obstacles (crossing cart / vehicle); the motion schedules are known to the simulator and can be replayed identically. They are **not** learned autonomous agents. The rover should sense or receive observable obstacle position/velocity through a defined state interface.

Do **not** feed future ground-truth obstacle trajectories to the planner unless explicitly disclosed as an oracle baseline. For an honest MVP, use known constant-velocity extrapolation over a short horizon from current positions/velocities and replan as observations change. Learned prediction is a stretch goal.

### Collision geometry

The visual mesh and collision footprint must remain aligned. For a minimal car-like rover, use a rectangle/swept footprint approximation rather than an unrealistically small point. In 3D, use vehicle and obstacle colliders and confirm collisions from simulation state. When doing fast route checks in XY, inflate obstacles by vehicle footprint/safety margin, then validate with a full rollout.

## 8. Recommended stack and implementation boundaries

| Subsystem | Preferred implementation | Why |
|---|---|---|
| ML and training | Python 3.11+, PyTorch, NumPy | Simple, GPU-optional, debuggable |
| Procedural map generation | Python + NumPy | Reproducible seeded environments |
| Expert planner | Hybrid A* / car-like motion primitives, optional smoothing | Vehicle turning constraints |
| Physics/collision | PyBullet (or deterministic custom kinematic dynamics + Bullet collision queries) | 3D bodies, collision detection |
| Safety optimization | NumPy + OSQP / CVXPY if compatible | Small constrained QP; fallback solver needed |
| Backend service | FastAPI, WebSocket, Pydantic | Control intents + streamed simulator state |
| Frontend | React + TypeScript + Three.js / React Three Fiber | Polished interactive 3D scene |
| Visual assets | Simple licensed industrial GLB models or primitives | High quality with minimal integration risk |
| Experiments | Python scripts -> CSV/JSON, optional plots | Honest baselines and metrics |

Avoid adding ROS 2, Isaac Sim, heavy CUDA-only plugins, a full SLAM pipeline, Unity, or a second physics engine unless there is a strong demonstrated need. A local browser app with a Python process is sufficient for judges.

**Hardware:** The team has an AMD Radeon 780M integrated GPU and 32 GB DDR5 in a ThinkPad P16s; another teammate may have an 880-series integrated Radeon GPU. Plan to support **CPU PyTorch** for the first trained model. PyTorch CUDA-specific operations cannot be assumed to work on AMD iGPUs. DirectML and ROCm depend on versions, device support, and op compatibility; do not make them project blockers. A compact trajectory model should be trainable/benchmarkable locally. More intensive training can use a cloud GPU if available, but the app should still demo locally with a saved checkpoint.

## 9. Proposed repository structure

```text
/
├── Handoff.md
├── README.md
├── backend/
│   ├── app.py                   # FastAPI app; websocket / HTTP
│   ├── api/schemas.py           # Messages and DTOs
│   ├── sim/
│   │   ├── world.py             # Scenario ownership and physics tick
│   │   ├── rover.py             # Rover state, kinematic bicycle / PyBullet adapter
│   │   ├── obstacles.py         # Static + moving objects, deterministic scripts
│   │   ├── collisions.py        # Ground-truth and footprint checks
│   │   └── scenarios.py         # Seeded scene generation
│   ├── planning/
│   │   ├── expert_planner.py    # Hybrid A* / vehicle motion primitives
│   │   ├── trajectory.py        # Representations / normalization / validation
│   │   ├── tracker.py           # Convert selected path to nominal controls
│   │   ├── ranking.py           # Score and choose generated routes
│   │   └── safety.py            # CBF-style QP + fail-safe
│   ├── ml/
│   │   ├── model.py             # Conditional flow-matching network
│   │   ├── dataset.py           # Expert trajectories and conditions
│   │   ├── generate.py          # Deterministic synthetic-data pipeline
│   │   ├── train.py             # Training, validation, checkpointing
│   │   └── inference.py         # Candidate generation
│   ├── evaluation/
│   │   ├── run_benchmark.py     # Held-out scenario evaluations
│   │   └── metrics.py
│   └── artifacts/              # Small example checkpoints/normalizers
├── frontend/
│   ├── src/
│   │   ├── scene/              # Three.js world and rover renderers
│   │   ├── components/         # Controls / overlays / legend
│   │   ├── overlays/           # Paths, safety volumes, interventions
│   │   ├── hooks/              # Websocket state and interpolation
│   │   └── App.tsx
│   └── public/assets/
├── tests/                      # Dynamics, collision, optimizer, serialization
├── scripts/                    # Start, bootstrap, benchmark, record demo
└── docs/                       # Additional technical notes as needed
```

This is a **suggested** structure; do not create empty complexity for its own sake. Keep code modular enough for teammates to work independently with clearly agreed schemas.

## 10. Data and API contracts (freeze early)

Pick minimal shared schemas before parallel work. Example:

```json
{
  "type": "state",
  "tick": 1750,
  "sim_time": 29.1667,
  "seed": 42,
  "rover": {"x": 8.1, "y": 4.2, "z": 0.4, "heading": 1.57, "speed": 1.2, "steering": 0.12},
  "goal": {"x": 17.0, "y": 10.0, "z": 0.0},
  "obstacles": [{"id": "forklift-1", "position": [10.0, 5.0, 0.0], "velocity": [-0.4, 0.0, 0.0], "shape": "box", "size": [1.3, 2.2, 1.6]}],
  "candidate_paths": [],
  "selected_path": [],
  "executed_trail": [],
  "controller": {"mode": "tracking", "nominal": {"a": 0.4, "steer": 0.12}, "actual": {"a": 0.4, "steer": 0.12}, "barrier_min": 1.8},
  "metrics": {"collisions": 0, "min_clearance": 0.85, "replans": 2, "interventions": 0}
}
```

Example frontend intents: `reset(seed)`, `set_goal(x,y,z)`, `spawn_obstacle(type,pose)`, `toggle_safety(enabled)`, `start`, `pause`, `step`, `start_replay(seed, policy_mode)`.

Do not send a huge mesh or entire ML embedding on every frame. Send authoritative state, compact candidate waypoints and event messages when changed. Include schema versioning if messages evolve.

## 11. Implementation sequence / integration strategy

**Do not build the most glamorous component first.** First prove the real control-to-physics loop and front-to-back data flow. Then replace the expert planner output with learned trajectories, then add the safety QP, then polish.

### Stage 0 — first 2–3 hours: decisions and skeleton

- Agree on simulated coordinate system, timestep, state/control representation, trajectory horizon, and API contracts.
- Create repo, runnable Python backend, minimal React Three Fiber frontend, and one-click local start instructions.
- Decide whether PyBullet wheel dynamics are viable immediately; if not, use deterministic kinematic bicycle state integration and 3D collision geometry.
- Render rover and a small environment with correct physics position updates.

**Exit criterion:** Rover can move according to actual control commands in the 3D scene; backend and UI agree on coordinates.

### Stage 1 — working classical autonomous rover

- Build static map and collision boundaries; add reproducible random layouts.
- Implement Hybrid A* or another vehicle-feasible expert planner and trajectory tracker.
- Click goal, get trajectory, drive the rover there; respect turning radius and speed limits.
- Log time and collision outcomes. This is the **baseline**, not the final ML feature.

**Exit criterion:** Deterministic successful runs on several maps, no camera/physics desynchronization.

### Stage 2 — synthetic expert dataset and first flow model

- Generate validated expert trajectories for many diverse maps and split held-out maps.
- Train an intentionally small conditional flow-matching network from scratch.
- Inference samples candidate paths from random noise; render in 3D and report invalid path rate.
- Add route ranking and tracking; if learned routes are bad, inspect learned path error and conditioning before increasing model size.

**Exit criterion:** Saved trained checkpoint produces nontrivial feasible trajectories on **unseen** maps and can drive the rover on at least some test scenarios without resorting to the classical planner.

### Stage 3 — safety filter + dynamic scenario

- Script a moving cart or forklift crossing the route.
- Implement a model-aware barrier filter / QP with actuator constraints and explicit emergency braking fallback.
- Visually differentiate nominal versus corrected commands, safety boundary, and interventions.
- Replan after disturbances; carefully handle invalid path/no feasible route.

**Exit criterion:** Deterministically replay a scenario with a meaningful controller intervention; collision/min-clearance metrics come from the simulator.

### Stage 4 — competitive demo + evaluation

- Improve industrial art, environment lighting, materials, vehicle movement, camera, and path visualization.
- Build two or three short seeded scenarios: normal navigation, blocked aisle, moving crossing obstacle.
- Record repeatable comparison runs; add speed/latency and safety graphs/overlays without drowning the scene in dashboards.
- Prepare backups: pretrained checkpoint stored locally, deterministic replay if live training fails, video fallback if an environment install fails.

**Exit criterion:** A judge can experience the technical difference in about 60–90 seconds and see meaningful quantitative results.

### Suggested 36-hour schedule (parallelize)

| Period | Primary goal |
|---|---|
| 0–4 h | Freeze contracts, runnable physics + browser integration |
| 4–12 h | Expert planner/data generator + controllable 3D scene |
| 12–22 h | Flow-model training/inference; dynamic scene + initial safety controller |
| 22–30 h | Integrated replanning + benchmark comparisons; fix core failure modes |
| 30–36 h | Visual polish, rehearsed scenarios, presentation, backup recording |

This is an allocation guide, **not** a promise that training, CBF integration, and rendering will all work on the first attempt. Keep a runnable demo throughout.

## 12. Recommended team ownership (adjust for team size)

- **ML / data lead:** procedural dataset design, expert trajectories, conditional flow model, training scripts, inference latency, held-out tests.
- **Control / simulation lead:** physics backend, vehicle dynamics, collision envelopes, path tracking, CBF-QP, obstacle events, deterministic replay.
- **Frontend / experience lead:** React Three Fiber world, vehicle animation, industrial scene, path/safety overlays, cameras, judge controls.
- **Integration / experiments lead (if 4 people):** API schemas, cross-team integration, benchmark protocol, logs, numerical metrics, demo script, fallback plans.

All branches integrate early. The ML and frontend developers must use the **same shared trajectory and coordinate schema** as the backend; do not independently invent a second robot motion representation.

## 13. Baselines, ablations, and actual research questions

The project should be able to answer real questions, not only look good:

1. Can a flow-matching planner generate **trackable** trajectories on previously unseen ground-robot environments?
2. How does its runtime, feasibility, and route quality compare with a classical vehicle-aware planner?
3. Does adding the online safety filter reduce collisions/unsafe proximity while preserving route completion?
4. How much efficiency is sacrificed when it must maintain safety?
5. How often must the system fall back to the classical planner or stop when no learned candidate is feasible?

**Test these modes on identical held-out scenario seeds:**

| Mode | What it measures |
|---|---|
| Classical feasible planner + tracker | A credible non-ML reference |
| Trained flow planner + tracker, safety off | Whether the learned generator actually works |
| Trained flow planner + tracker, safety on | Benefit/cost of the CBF-style safety layer |
| Classical planner + same safety layer (if time) | Separates planner quality from safety-layer benefit |

**Metrics:** goal success rate; collision count; minimum obstacle clearance; path length and time to goal; average curvature / curvature-limit violations; braking and safety intervention count; generated candidate feasibility; frequency of fallback; model sampling latency; planning wall-clock cost; and 3D simulation FPS.

**Evaluation protocol:** use a specified number of held-out map layouts and seeded dynamic obstacle schedules; report exactly N episodes, failures, mean/median, and preferably variability. No cherry-picked percentages or claims of SOTA performance. Measure safety-off/on with identical initial states and obstacle motion schedules. Do not claim flow matching must beat classical planning; negative/ambiguous results are still valid engineering findings.

Potential educational/visual ablation: show early untrained trajectory samples, post-training samples, raw predicted route, rejected candidates, and safety-corrected execution. Never mislabel a fallback plan as a learned plan.

## 14. Risks and realistic fallbacks

| Failure / risk | Mitigation |
|---|---|
| Flow model produces collisions or highly curved paths | Train on simpler maps, normalize data carefully, reject infeasible candidates, use feasible route ranking, compare expert dataset distribution |
| Not enough expert data / generation too slow | Reduce map/obstacle complexity and trajectory horizon, batch simple motion-primitive expert generation, train small model first |
| Dynamic obstacles are not covered by static expert data | Explicit short-horizon obstacle extrapolation + online safety filter + replanning; adding time-dependent conditions is stretch |
| CBF-QP infeasible / numerical instability | Bounded controls, solver timeouts, conservative emergency brake/stop and honest status, log solver failures |
| Naive distance CBF doesn't constrain acceleration/steering correctly | Use properly derived sampled-data or higher-order constraints; do not advertise proof without derivation |
| 3D PyBullet wheel dynamics unstable | Use deterministic kinematic bicycle backend + honest 3D collision integration; keep visual wheels synchronized |
| Model too slow on AMD integrated GPU | Use small PyTorch CPU model, shorter horizon, fewer sample steps/candidates, preload checkpoint |
| Vehicle gets trapped / repeated replanning oscillates | Progress timeouts, goal tolerance, route hysteresis, classical fallback and stuck detector |
| Browser render and backend disagree | Backend-authoritative state; fixed simulation timestep; controlled interpolation; schema tests |
| Unrealistic final presentation | Prioritize one excellent 3D industrial scene and two reproducible interactive scenarios, not ten untested levels |

**Fallback hierarchy:**

1. Full flow planner + filter + dynamic 3D environment (target).
2. Flow planner on static scenes + real-time safety/replanning for one scripted moving obstacle.
3. Flow planner on static scenes + interactive 3D rover and verified static-obstacle safety.
4. If the learned model fails, keep an honest classical planner demo and show its failed-versus-attempted ML experiment, rather than falsely presenting it as neural navigation. This is a partial delivery, not equivalent to the intended project.

## 15. MVP vs stretch goals

### MVP — must work

- Beautiful enough **3D industrial scene** with rover, solid obstacles, and controllable camera.
- Actual rover kinematic motion and physical collision validation.
- Start/goal selection and a feasible classical baseline.
- **Trained-from-scratch conditional flow-matching model** producing candidate trajectories conditioned on environment and goal.
- Candidate collision/curvature checks and tracking of a selected path.
- **Real** safety intervention with correctly documented CBF-style mathematical optimization and stop fallback.
- At least one **moving obstacle event** with replanning or controlled stop.
- Reproducible baseline/safety-off/safety-on comparisons and real metrics.

### Stretch — only after MVP works

- Height-map-aware control on ramps and slopes, grade/stability constraints.
- Several classes of 3D robots or vehicles with different physical dynamics.
- Multiple autonomous rovers, shared traffic and multi-agent collision constraints.
- Flow-policy conditioning on tracked moving-obstacle future predictions.
- Lidar/depth-sensor simulation with partial observability and dynamic occupancy estimation.
- Learning from sparse onboard camera perception rather than perfect simulator state.
- Risk-sensitive / uncertainty-aware trajectory generation.
- Improved control-theoretic formulation with proven safe-set assumptions.
- Full rigid-body wheel dynamics, suspension, terrain traction and slip.

**Avoid scope explosion:** no drones, no war game, no generative-video world model, no user-auth/database CRUD, no chatbot, no VR requirement, no real hardware deployment, no broad autonomous vehicle autonomy claims.

## 16. Demo story and short presentation outline

**Opening (10–15 seconds):** “Most navigation systems optimize where a robot should go, but a feasible path can become unsafe when the world changes—and a ground vehicle cannot instantly steer or stop. We combine a learned generative motion planner with a mathematical safety filter.”

**Show the model (15–25 seconds):** Click destination; generate visible alternative trajectories. Briefly explain it was trained on thousands of expert simulations and is producing paths in a new environment. Show selected path and moving rover.

**Introduce danger (20–25 seconds):** Spawn crossing forklift or moving barrier. Show predicted risk, the CBF-style action correction, replanning/stopping, and actual robot motion. Optionally replay identical scenario safety-disabled to demonstrate real effect.

**Results (10–15 seconds):** Show measured success/collision/clearance/latency comparisons. Highlight that the model plans, the control system filters, and the simulator tests what actually happens.

**Closing (5 seconds):** Potential applications: industrial inspection, warehouse automation, low-speed autonomous delivery, and safety research. Frame this as a research prototype, not a road-certified autonomous-driving system.

## 17. Definition of done / acceptance tests

- [ ] From a clean clone, one clear set of commands launches backend and frontend; optional saved checkpoint is available.
- [ ] Rover obeys turning and speed constraints, collides with solid objects, and cannot move sideways instantaneously.
- [ ] New seeded layouts can be generated and are not all copies of one map.
- [ ] A trained flow-matching checkpoint produces candidate trajectories; they are distinguishable from fallback planner results.
- [ ] At least one held-out map is demonstrated and evaluated.
- [ ] A vehicle-aware baseline can drive the same scenario.
- [ ] Nominal vs corrected actions are numerically logged; the safety filter actually modifies a command in a reproduced hazard.
- [ ] Actuator bounds and missing/infeasible solutions have an explicit fallback.
- [ ] A real dynamic-obstacle scenario triggers stopping/replanning or another meaningful safety response.
- [ ] Replaying identical scenario seeds yields equivalent safety-on/off comparisons.
- [ ] Visual path, actual robot state, and backend collision events stay synchronized.
- [ ] Evaluation metrics come from measured runs, not fabricated placeholders.
- [ ] README clearly explains training, physics, baselines, limitations, and how to run the demo.
- [ ] No unearned claims of mathematical guarantees, pretrained intelligence, on-road autonomy, or superior performance.

## 18. Instructions for Claude / future coding agents

1. **Read this handoff as a design brief, not a completed implementation.** Check the actual repository state before assuming any file, component, dataset, or model already exists.
2. Do not rename the project. Treat the name as undecided and avoid branding.
3. Do not substitute a drone, 2D-only dot navigator, graph-search animation, or an AI wrapper for the described project.
4. Keep the primary ML technique **conditional flow matching** unless explicitly instructed to change it. The user chose this direction because of its modern generative ML plus mathematical safety combination.
5. Be rigorous about whether the neural network **actually** predicts route candidates versus whether a classical planner is generating them at inference.
6. Be rigorous about the **relative degree / correct dynamics** for CBF constraints. A simple distance threshold is not a sufficient implementation.
7. Prefer measurable training, replayable tests, and honest failures over scripted “successful” cinematic footage.
8. Suggest pragmatic simplifications that **preserve the core research idea**. A smaller network and limited map distribution are acceptable; replacing the learned planner with A* is not equivalent.
9. On AMD iGPU hardware, start with portable dependencies and CPU-compatible PyTorch; don't assume CUDA works. Detect and test any proposed GPU acceleration instead of assuming support.
10. Before large code generation, explicitly define module interfaces, state representation, timestep, and test scenario. Build incrementally; don't start by installing a massive simulation stack.
11. When asked to implement, maintain a working, reproducible vertical slice and report what is actually functional versus planned.
12. If there are tradeoffs, prioritize **real training + correct vehicle/control logic + reproducible demo + polished 3D experience**, in that order without neglecting the other two.

## 19. Research foundation / reading list

These are **inspirations and technical references**, not claims that our team invented the underlying methods. Do not assume a listed paper's exact formal guarantees transfer to our implementation.

1. **Yang, Jang, Han — *SafeFlowMatcher: Safe and Fast Planning using Flow Matching with Control Barrier Functions* (ICLR 2026).** Closest conceptual reference: learned flow-based trajectory generation plus safety correction. Its predictor–corrector method and barrier certificate are specific to its formulation, not automatically ours. Paper: https://proceedings.iclr.cc/paper_files/paper/2026/hash/58b1b19bbbcce03bfb66ea60344e41ea-Abstract-Conference.html ; code: https://github.com/takahashi-seiryu/SafeFlowMatcher
2. **Lipman et al. — *Flow Matching for Generative Modeling* (ICLR 2023).** Foundation for learning generative velocity fields via flow matching. https://arxiv.org/abs/2210.02747
3. **Ames et al. — *Control Barrier Function Based Quadratic Programs for Safety Critical Systems* (IEEE TAC 2017).** Fundamental CBF/QP framework, including automotive-style examples and assumptions. https://authors.library.caltech.edu/records/jnhr0-1ww05
4. **Hybrid A* references for car-like vehicles:** car-like state constraints and path smoothing. Example contemporary implementation: https://github.com/MedAmineElhafsi/HybridA-Path-Planning ; see also https://www.mdpi.com/2076-3417/15/22/12298
5. **PyBullet Quickstart Guide:** Python robotics simulator, rigid bodies, URDF, contacts and collision queries. https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html
6. **Earlier project for differentiation only:** https://github.com/ian-kr/Swarm — PPO-trained drone attack swarm / defense sim. Avoid merely reskinning its concept.

## 20. One-paragraph handoff summary

Build an unnamed, polished **3D autonomous ground-rover robotics prototype** in a changing industrial environment. Procedurally generate expert, physically feasible car-like trajectories and train a small **conditional flow-matching PyTorch network** to sample new time-parameterized routes conditioned on local obstacles, vehicle state, and destination. A vehicle-aware tracker executes selected trajectories while a **dynamics-aware CBF-style constrained safety filter** adjusts steering/acceleration in response to static and moving obstacles, with replanning and honest emergency fallback. Use a backend-authoritative simulator, a beautiful interactive Three.js frontend, deterministic hazard scenarios, and rigorous comparisons against a classical feasible planner and safety-disabled mode. Keep it demonstrably different from the team's prior drone-swarm project, hardware-conscious for AMD integrated GPUs, and careful about the mathematical limits of safety guarantees. **Project name deliberately TBD.**
