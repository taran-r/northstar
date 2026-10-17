# Northstar Synthetic Trajectory Data Pipeline

## Purpose

This package generates expert navigation trajectories for training a **conditional flow-matching** neural network that proposes short-horizon rover motions inside one persistent industrial facility.

Out of scope (handled by teammates / later work):

- Flow-matching network architecture and PyTorch training loop
- Full Control Barrier Function (CBF) safety controller
- Three.js frontend rendering

## Algorithm selection

| Component | Choice | Why |
|-----------|--------|-----|
| Static routing | **Hybrid A\*** with bicycle motion primitives | Kinematically feasible paths (no sideways motion / instant yaw flips); standard for Ackermann/bicycle vehicles ([ackermann-path-planning](https://github.com/sezer-muhammed/ackermann-path-planning)) |
| Dynamic interaction (primary) | **Timed path following** on the Hybrid A\* route | Brake / wait when predicted dynamic occupancy blocks the lookahead; yields wait/yield/follow behaviors at dataset scale |
| Dynamic interaction (fallback) | **Time-augmented Hybrid A\*** | Full space–time search when following fails; collision checks at node time \(t\) |
| Prediction for planner / NN | **Constant-velocity** (default); route-conditioned optional | Avoids silently giving the network perfect foresight |
| Oracle scenarios | Flagged `oracle_dynamics=true` | Expert may use GT futures for hard timing; conditioning channels still store CV predictions |
| Smoothing | B-spline + curvature-aware velocity profile | Produces continuous, executable demos similar in spirit to FlowMP’s smooth expert motions |
| Validation | GT dynamic collision + bicycle rollout | Ensures saved trajectories are physically consistent |

**Why not plain grid A\*?** It ignores minimum turning radius and steering limits, producing paths a real rover cannot track.

**Why not Hybrid A\* alone?** It only sees static geometry. Moving forklifts/workers require a time dimension (or frequent replan). Our generator supports both offline time-aware planning and advancing/replanning from updated states via `plan_expert_trajectory` on a mutated `ScenarioSpec.start`.

References used for guidance (not copied):

- FlowMP: https://github.com/mkhangg/flow_mp
- Hybrid A\*: https://github.com/sezer-muhammed/ackermann-path-planning
- Time-aware planning: https://arxiv.org/abs/2508.06404
- Scenario patterns: https://github.com/carla-simulator/scenario_runner

## Shared facility geometry

Authoritative JSON:

```
facilities/industrial_facility_v1.json
```

Schema version `1.0.0`.

### Coordinates

This generator stores the ground plane as **X/Z** with **Y** elevation (meters; angles in radians; time in seconds), matching the original data-gen brief.

Team backend convention (`CLAUDE.md`) is `(x, y, z)` with **z up** and heading in the **XY** plane. When integrating:

- `(x, z, yaw)_dataset → (x, y, psi)_backend`
- Use the single tested helper for Three.js Y-up conversion (see `CLAUDE.md`)

Do not invent a separate training map. The same facility JSON is intended for Three.js loading after coordinate conversion.

Scenario JSON files under `data/.../scenarios/` add temporary static + dynamic obstacles on top of the base facility.

## Vehicle model

Module: `northstar.vehicle.bicycle.BicycleModel`

Default rover parameters (configurable):

- length 1.3 m, width 0.8 m, wheelbase 1.0 m
- max speed 2.5 m/s, max steer 0.55 rad
- accel / brake / steer-rate limits

This module is designed for reuse by the neural follower and future CBF layer.

## Trajectory / training sample format

### Full route

`trajectories/*.json` — variable-length expert route with fields per state:

`t, x, z, y, yaw, speed, steering, accel`

### Local window (NN sample)

Default: **H = 32** states @ **dt = 0.2 s** → **6.4 s** horizon (configurable).

Each `samples/<id>.npz` + sidecar `.json` contains:

| Field | Shape / type | Role |
|-------|--------------|------|
| `future` | `[32, 7]` | Expert future: `x,z,yaw,speed,steering,accel,t_rel` |
| `ego0` | dict/vector `[5]` | Initial state conditioning |
| `subgoal` | `[3]` | Local goal (window end if global goal beyond horizon) |
| `global_goal` | `[3]` | Scenario destination |
| `facility_occ` | `[H,W]` | Permanent structure local occupancy |
| `static_occ` | `[H,W]` | Facility + temporary static obstacles |
| `dynamic_occ` | `[T,H,W]` | **Predicted** dynamic occupancy (CV), not GT |
| `oracle_dynamics` | bool | Expert used GT futures |
| `behavior_tag` | str | Scenario behavior label |
| `mode_id` | int | Multi-modal expert index |

Local maps default to **20 m × 20 m** @ **0.25 m** resolution, ego-centered.

### PyTorch loader

```python
from torch.utils.data import DataLoader
from northstar.dataset import NorthstarTrajectoryDataset, collate_trajectory_batch

ds = NorthstarTrajectoryDataset("data/dataset_v1", split="train")
loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_trajectory_batch)
batch = next(iter(loader))
# batch["future"]      [B, 32, 7]  world frame
# batch["future_rel"]  [B, 32, 7]  ego frame (recommended training target)
# batch["static_occ"]  [B, 1, H, W]
# batch["dynamic_occ"] [B, T, H, W]
# batch["ego0"], batch["subgoal"], batch["global_goal"]
```

Suggested flow-matching target: `future_rel[:, :, :3]` (or include speed/accel channels to match FlowMP-style second-order supervision).

## Behaviors covered

Scenarios cycle through tags such as: straight travel, sharp/long turns, tight corridors, intersections, static avoidance, braking, waiting for traffic, yielding, accelerating when clear, detours around blockages, following another vehicle. A fraction of scenarios request **multiple expert modes** (e.g. wait vs. detour).

## Replanning

Advance a scenario by updating `ScenarioSpec.start` (and optionally truncating dynamic trajectories to `t >= now`), then call `plan_expert_trajectory` again. The same stack used offline supports live replan for the eventual demo.

## Generated dataset (`data/dataset_v1`)

Current build: **5021** windows from **363** scenarios. Splits are by **scenario** (see `dataset_meta.json` → `split_policy`). Target generation flag: `--min-windows 5000`.

**`data/dataset_v1` is the immutable initial corpus** for training handoff. Do not overwrite it; see `docs/TRAINING_HANDOFF.md`.

The `data/` directory is tracked in git (~60 MB). Ordinary Git is fine at this size; move to Git LFS if the corpus grows much larger.

### Subgoal leakage note

When the global goal lies beyond the 6.4 s window, `subgoal` is set to the **expert window endpoint** (`is_local=1`). That is convenient for training but is **not** automatically available at inference unless the live stack supplies an equivalent local goal. Prefer `global_goal` + maps as primary conditioning, or define a reproducible subgoal policy.

## Commands

```bash
pip install -e .
python scripts/smoke_test.py
python scripts/generate_dataset.py --out data/dataset_v1 --n-scenarios 360 --min-windows 5000
python scripts/validate_dataset.py --root data/dataset_v1
```

## Ground truth vs prediction (critical)

1. **Ground truth** dynamic trajectories: used by the simulator / validation; optionally by the **oracle** expert planner.
2. **Predictions** (constant-velocity): stored in `dynamic_occ` and used by non-oracle planning.

Never train the network as if it receives GT futures unless you deliberately build an ablation and document it.
