# CBF-style discrete-time safety filter

This is a local safety layer for the kinematic bicycle in `northstar/vehicle/bicycle.py`. It sits between a trajectory tracker and the plant. It does not replace the flow-matching planner, and it does not claim that every scene has a safe command.

Dataset coordinates are used throughout: horizontal X/Z, elevation Y, yaw about +Y from +X toward +Z. The bicycle state is `(x, z, yaw, speed, steering)`.

## Plant and the command that is filtered

`BicycleModel.step` consumes `ControlInput`:

- `accel` is clipped to `[-max_decel, max_accel]` = `[-2.5, 1.5]` m/s².
- `steer_cmd`, when set, is an absolute steering target. It is clipped to `±max_steer` (`±0.55` rad) and then rate-limited by `max_steer_rate * dt` (`1.2` rad/s).
- If `steer_cmd` is omitted, `steer_rate` is integrated and clipped the same way.
- Speed is clipped to `[0, 2.5]` m/s. The rover does not reverse.
- Wheelbase is `1.0` m. The footprint passed to clearance is the body (`1.3 × 0.8` m) inflated by `footprint_margin` (`0.05` m).

The decision variable is `u = (accel, steer_cmd)`. A pure rate command is converted with `steer_cmd = steering + steer_rate * dt` before the quadratic program. The certified command is returned as `ControlInput(accel=..., steer_cmd=...)`.

One integration step uses the average of the old and new speed and the updated steering angle, so both inputs move the pose inside a single sample. That is a property of this discrete plant. It is not a reason to treat geometric distance as a relative-degree-1 barrier on the continuous bicycle.

## Why a distance threshold is not the constraint

Continuous-time kinematics:

```
x_dot = v cos(yaw)
z_dot = v sin(yaw)
yaw_dot = (v / L) tan(delta)
v_dot = a
delta_dot = omega
```

A clearance `c(x, z, yaw)` depends on position and heading. Its first derivative depends on `v` and `yaw`, not on `a` or `omega`. The inputs show up in the second derivative. Relative degree is 2. A constraint written only on `c` does not define a control barrier function for `(a, omega)`.

Braking distance makes the same point concrete. From `2.5` m/s the rover needs `v² / (2 * 2.5) = 1.25` m to stop. A check that the rover is not currently inside an obstacle accepts a command that cannot be braked before contact.

## What is enforced

For each nearby obstacle, let `c_i(s, t)` be the signed separation between the inflated footprint and the obstacle geometry at time `t` (oriented box, or a circle for columns). Positive means separation. Negative means penetration depth.

The filter certifies `u` by rolling the discrete plant forward:

1. Apply `u` for one plant step of `dt`, then for the rest of `preview_time` (default `0.7` s) at a backup step of `max(dt, 0.1)` s. This is the commitment being checked. The coarser backup step is a computational approximation; the command that is applied to the plant still uses the real `dt`.
2. Apply the backup controller: `accel = -max_decel`, steering held at the angle reached after the commitment.
3. Hold the stopped pose for `idle_time` (default `1.5` s) so a predicted moving obstacle can sweep the stopping point.

Define

```
phi_i(u) = min_k [ c_i(s_k, t_k) - m ]
```

with safety margin `m` (default `0.12` m). The obstacle pose at `t_k` is the constant-velocity forecast supplied in the scene, not a ground-truth future sample.

The quadratic program enforces the linearization of

```
phi_i(u) >= 0
```

about the nominal command after it has been clipped into the actuator box. This is a backup-controller approximation of a discrete control-invariant set (the same role a higher-order CBF plays for relative degree 2). It is not a proof that the continuous-time barrier stays positive under every disturbance, every linearization error, or every forecast error.

When the current slack `h_i(s) = c_i(s, t) - m` is already inside the activation distance, a one-step discrete-time CBF is added:

```
h_i(F(s, u)) >= (1 - gamma) h_i(s)
```

with `gamma` in `(0, 1]` (default `0.4`). On this plant `F` depends on both `accel` and `steer_cmd`, so the one-step inequality is a well-posed discrete barrier for the implemented integrator. It stops a single sample from closing a small gap faster than the class-K decay allows. The backup constraint is what accounts for stopping distance beyond that sample.

Both maps are linearized with one-sided finite differences inside the actuator box. Inactive obstacles (backup slack and current slack above `activation_clearance`, default `2` m, and outside the reachable horizon) are dropped.

## Quadratic program

```
minimize    w_a (a - a_nom)^2 + w_delta (delta_cmd - delta_nom)^2
subject to  linearized phi_i(u) >= 0
            linearized one-step decay, when active
            a in [-max_decel, max_accel]
            delta_cmd in the one-step reachable steering interval
```

Default weights are `w_a = 1` and `w_delta = 25`, so a `1` m/s² acceleration change costs the same as a `0.2` rad steering change. Braking is preferred to a large swerve. Leaving the route is the planner's job.

The feasible set is a convex polygon in two variables. The solver in `northstar/control/qp.py` returns the exact weighted projection onto that polygon by checking the unconstrained point, its projection onto each edge, and every vertex. No iterative tolerance is involved. SciPy/OSQP are not required at this dimension; the enumeration is the solver.

If the true (nonlinear) backup slack at the QP solution is still below `-violation_tol` (`0.03` m), a short search from that point toward maximum braking and a few steering angles looks for a command that restores the slack. The QP remains the primary step. The search only runs when the linear model was wrong or the linear constraints were infeasible.

## Infeasible commands

If every sampled command, including maximum braking, still drives the backup slack below the tolerance, the filter:

- applies maximum braking,
- chooses the rate-limited steering angle among the sampled angles that leaves the largest backup slack,
- sets `infeasible=True`, `fallback="max_brake"`, `solver_status="infeasible_fallback"`.

That fallback is not a certificate. An obstacle inside the braking distance can still be hit. Callers should read `impact` behavior from the simulator, not from this flag alone.

## Predictions, not ground truth

`filter_control` only reads a `SafetyScene`: box or circle poses, plus `vx`, `vz`. During the rollout, centers move as `p + v * (t - t_ref)`.

`build_scene(..., dynamics_mode="causal_cv")` fills those velocities from the sensed motion:

- current pose is the ground-truth sample at the current time (what a sensor measures now),
- after the first sample, velocity is a backward difference over `dt_obs` (default `0.2` s),
- the stored `DynamicObject.prediction` polyline is not consulted when ground truth exists, so a route-conditioned or oracle forecast cannot leak in.

`dynamics_mode="snapshot_gt"` is for scoring only. It pins each agent to its true pose and sets velocity to zero.

Constant velocity is wrong when an agent turns or stops after the current sample. The idle segment of the backup makes the filter conservative about agents that are about to cross the stopping pose, and it is blind to a turn that has not started.

## Replan signal

`replan_required` is local. It is set only while the filter is infeasible or is withholding acceleration that the tracker still wants, and a footprint placed ahead along the current heading (1.2 m to 4.8 m) stays inside the margin for the whole prediction horizon (default 4 s). A forklift that occupies the heading for only part of the horizon does not set the flag. A barrier that never moves does, once the rover is braking for it. Steering away from a nearby wall while still making progress does not, by itself, request a new route.

The filter will not drive around a globally blocked route. The generative planner should emit a new trajectory when this flag stays true.

## Integration with a flow-matching trajectory

The network is not an input. A follower turns a timed polyline into the same nominal command the pure-pursuit tracker produces.

```python
from northstar.control import (
    TrajectoryTracker,
    build_scene,
    closed_loop_step,
    filter_control,
)
from northstar.vehicle.bicycle import ControlInput, VehicleParams, VehicleState

params = VehicleParams()
tracker = TrajectoryTracker(predicted_xz_polyline, params=params, cruise_speed=1.6)

# Once per control cycle. `dynamics` are sensed tracks, not future ground truth.
scene = build_scene(
    state.t,
    facility=facility,
    static_obstacles=static_obstacles,
    dynamic_obstacles=dynamics,
    dynamics_mode="causal_cv",
)

# Either the one-step helper:
state, nominal, applied, diag = closed_loop_step(
    state, tracker, scene, dt, params=params, use_cbf=True
)

# Or the pieces, if the follower is not TrajectoryTracker:
nominal = ControlInput(accel=a_nom, steer_cmd=delta_nom)
applied, diag = filter_control(state, nominal, scene, dt, params=params)
state = BicycleModel(params).step(state, applied, dt)

if diag.replan_required:
    # Ask the generative model (or the logged Hybrid A* fallback) for a new route.
    ...
frontend_payload = diag.to_dict()
```

`diag.to_dict()` uses schema `cbf-diagnostics-1`. Fields the frontend can render directly:

| Field | Meaning |
| --- | --- |
| `intervened` | Safe command differs from the raw nominal, including actuator clipping. |
| `cbf_active` | An obstacle constraint moved the command beyond actuator clipping. |
| `infeasible` | No sampled command kept the backup slack above the tolerance. |
| `fallback` | `""` or `"max_brake"`. |
| `replan_required` | Forward heading stays blocked under the prediction. |
| `nominal_accel`, `nominal_steer_cmd` | Tracker command. |
| `safe_accel`, `safe_steer_cmd` | Command sent to the bicycle. |
| `accel_correction`, `steer_correction` | `safe - nominal`. |
| `min_clearance_m` | Instantaneous inflated-footprint clearance. |
| `min_backup_clearance_m` | Predicted slack of the command that was applied, before adding the margin back in (slack, so 0 is the margin). |
| `closing_speed_mps` | Positive when the gap to the nearest obstacle is shrinking. |
| `active_obstacle_ids` | Obstacles that produced a linear constraint. |
| `solve_time_s` | Wall time of this call. |
| `solver_status` | `nominal_safe`, `qp_feasible`, `line_search`, or `infeasible_fallback`. |
| `pose_threejs` | `dataset_to_threejs(x, z, y)`. |
| `yaw_dataset` | Bicycle yaw. |
| `yaw_threejs` | `-yaw_dataset`, because `dataset_to_threejs` mirrors Z. |

Contract the tracker must satisfy:

- State is `VehicleState` in the dataset XZ frame, including the current `steering` and `speed`.
- Nominal control is `ControlInput`. `steer_cmd` is an absolute angle. `steer_rate` is accepted and converted.
- `dt` is the simulation step, in seconds. The same `dt` has to be the one passed to `BicycleModel.step`.
- Obstacle forecasts are velocities in m/s on X and Z. Do not pass ground-truth future waypoints into `filter_control`.

The filter does not read network weights, occupancy tensors, or trajectory samples. Replacing `TrajectoryTracker` with a follower that tracks the flow-matching polyline is the entire integration change.

## What was measured

`python scripts/eval_cbf_safety.py` runs each scenario in `northstar/control/scenarios.py` twice, with the same pure-pursuit tracker. The plant does not resolve contact, so an unfiltered rover that hits an obstacle keeps driving through it. Goal progress on those runs is the progress of that ghost, not of a rover that stopped at impact. Body clearance uses the physical footprint (`footprint_margin = 0`) against ground-truth poses. The filter sees only the causal constant-velocity forecast.

`docs/cbf_eval/results.json` and the PNGs next to it are the run this section quotes. `tests/test_cbf_safety.py` (15 tests) covers the same scenes plus clearance geometry, the 2-variable QP, actuator limits, a facility wall, and the rule that a stored prediction polyline is ignored.

| Scenario | Collisions off / on | Min body clearance off / on (m) | Obstacle interventions | Infeasible steps | Replan steps | Goal progress on | Mean solve time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static barrier | 11 / 0 | -0.800 / 0.387 | 57 | 0 | 55 | 0.47 | 20.4 ms |
| crossing forklift | 15 / 0 | -0.800 / 0.211 | 33 | 0 | 0 | 1.00 | 21.8 ms |
| narrow corridor | 66 / 0 | -0.080 / 0.170 | 79 | 0 | 17 | 0.78 | 117 ms |
| follow lead | 25 / 0 | -0.800 / 0.371 | 66 | 0 | 5 | 0.44 | 51.1 ms |
| unavoidable wall | 15 / 21 | -0.500 / -0.500 | 24 | 24 | 17 | 0.12 | 34.0 ms |
| open floor | 0 / 0 | none | 0 | 0 | 0 | 0.97 | 0.02 ms |

Mean solve time across the six filtered runs is 40.7 ms. Open-floor time is the cost of an empty scene. The corridor is slower because the rover yaws and clearance falls through to the separating-axis test against the walls.

Reading the rows:

- The barrier case stops with 0.39 m of body clearance and asks for a new route. The unfiltered tracker hits at 2.0 m/s and continues through the barrier.
- The forklift case yields, then finishes the 18 m route (progress 1.00) with no replan flag. The unfiltered tracker hits at 1.6 m/s.
- In the corridor the mean absolute steering correction is 0.28 rad per step and the body stays 0.17 m off the wall. The reference path itself is inside the wall, so the unfiltered tracker collides. Seventeen steps still raise `replan_required` when the heading points at the wall and the filter is briefly holding speed. The corridor is passable; that flag is conservative there.
- Behind the slower AGV the filtered rover finishes at 0.59 m/s, just above the lead's 0.55 m/s, with 0.37 m of body clearance. Five replan steps fire while it is braking. The unfiltered tracker hits at 1.6 m/s and drives through.
- The unavoidable wall is inside the braking distance at 2.4 m/s (about 0.35 m of gap, stopping distance about 1.15 m). Every filtered step is `infeasible_fallback` and commands maximum braking. Impact speed is 1.9 m/s versus 2.4 m/s unfiltered. Both collide. The filter does not claim otherwise.
- On the empty floor the obstacle constraints never activate. Four interventions are steering-limit clips, not obstacle corrections. Progress matches the unfiltered tracker (0.97).

## Assumptions and limits

- The backup policy is maximum braking with steering held. It is not a search over swerves. A narrow opening that requires a deliberate lateral shift before braking can be marked infeasible even when a longer maneuver would fit.
- The linearization of `min` clearance along a trajectory is nonsmooth. The nonlinear check and the braking search exist because the QP can be optimistic. They can also be conservative.
- Forecasts are constant velocity. A pedestrian who steps into the lane after the current sample is invisible until the velocity estimate changes. An agent who stops earlier than the forecast keeps the rover waiting or, if the rover is already committed, can still cause contact.
- The discrete CBF and the backup set are enforced on the nominal model with no slip, no delay beyond the preview window, and no actuator bias. That is not a formal guarantee on hardware.
- `replan_required` looks only a few meters ahead along the current heading. A blocked aisle beyond that horizon is the planner's problem until the rover gets closer.
- The rover cannot reverse (`min_speed = 0`). Some conflicts that a reverse maneuver could undo are infeasible for this filter.
