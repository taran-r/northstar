"""Frontend bridge: closed-loop episodes emitted as the UI's state frames.

The frontend declares its contract in `frontend/src/types/sim.ts` (schema
version 1) and leaves `frontend/src/sim/mockSource.ts` as "the seam the real
client will replace". This module produces exactly those `StateFrame` objects
from a real run of the pipeline, so the remaining frontend work is transport
rather than translation.

What a frame carries comes from four places:

  - rover pose and executed trail: the bicycle plant, so the rendered motion is
    kinematically valid whatever the network proposed;
  - candidate and selected paths: `inference.plan` plus `inference.rank_candidates`;
  - controller mode, nominal/actual commands and barrier margin: the CBF
    diagnostics returned by `northstar.control.closed_loop_step`;
  - metrics: accumulated over the episode, scored against snapshot ground truth
    the same way `northstar.control.simulate.simulate_scenario` scores.

Coordinates are converted once, here. The corpus and the model work in the
dataset frame (X and Z horizontal, Y up); the frontend's schema is the backend
frame (X and Y horizontal, Z up) and its scene layer maps that to Three.js
itself. `northstar.coords.dataset_xz_to_backend_xy` is the only conversion used,
and scene components are expected not to re-derive it.

`measured` is True on these frames: every number is from the run that produced
them. Frames built by the UI's own `frames.ts` set it False.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

from backend.ml.dataset import (
    REAL_DATA_ROOT,
    V2_DATA_ROOT,
    load_vehicle_params,
    world_to_local,
)
from backend.ml.evaluation import _vehicle_limits, footprint_centres
from backend.ml.inference import FlowMatchingPlanner, best_candidate
from northstar.control.clearance import oriented_box_clearance
from northstar.control.scene import build_scene
from northstar.control.simulate import closed_loop_step
from northstar.control.tracking import TrajectoryTracker
from northstar.coords import dataset_xz_to_backend_xy
from northstar.facility.loader import load_facility, load_scenario
from northstar.facility.occupancy import (
    build_facility_occupancy,
    dynamic_occupancy_at_time,
    sample_dynamic_pose,
)
from northstar.geometry import OrientedBox
from northstar.vehicle.bicycle import VehicleParams, VehicleState

SCHEMA_VERSION = 1

FACILITY_PATH = Path("facilities/industrial_facility_v1.json")
MAP_SIZE_M = 20.0
MAP_RESOLUTION = 0.25
DYNAMIC_SLICE_DT = 0.8
DYNAMIC_SLICES = 8

CONTROL_DT = 0.1
REPLAN_PERIOD_S = 1.0
GOAL_RADIUS_M = 1.5
CRUISE_SPEED = 1.6
TRACKER_LOOKAHEAD = 1.5
TRACKER_STEER_GAIN = 1.7
CANDIDATES = 8
SOLVER_STEPS = 16

CONTROLLER_MODES = ("tracking", "intervention", "estop", "replanning", "goal")


def _xy(x: float, z: float) -> tuple[float, float]:
    """Dataset horizontal (x, z) -> backend horizontal (x, y)."""
    bx, by, _, _ = dataset_xz_to_backend_xy(x, z)
    return bx, by


def rover_state(state: VehicleState) -> dict:
    x, y = _xy(state.x, state.z)
    return {
        "x": x,
        "y": y,
        "z": 0.0,
        "heading": float(state.yaw),
        "speed": float(state.speed),
        "steering": float(state.steering),
    }


def path_points(
    polyline_xz: Sequence[Sequence[float]], dt: float | None = None, t0: float = 0.0
) -> list[dict]:
    """Dataset XZ polyline -> the frontend's PathPoint list."""
    points = []
    for index, point in enumerate(polyline_xz):
        x, y = _xy(float(point[0]), float(point[1]))
        entry = {"x": x, "y": y}
        if dt is not None:
            entry["t"] = t0 + index * dt
        points.append(entry)
    return points


def obstacle_states(static_obstacles, dynamic_obstacles, t: float) -> list[dict]:
    """Static and moving obstacles at time `t`.

    Facility walls, columns and zones are left out: the scene renders the hall
    from the facility spec and would draw them twice. Dynamic poses come from
    the prediction track, never the ground truth, so the UI shows what the
    controller was allowed to see.
    """
    out = []
    for obstacle in static_obstacles:
        x, y = _xy(obstacle.x, obstacle.z)
        out.append(
            {
                "id": obstacle.id,
                "position": [x, y, float(getattr(obstacle, "y", 0.0))],
                "velocity": [0.0, 0.0, 0.0],
                "shape": "box",
                "size": [
                    float(obstacle.length),
                    float(obstacle.width),
                    float(getattr(obstacle, "height", 1.0)),
                ],
                "kind": obstacle.type,
                "yaw": float(obstacle.yaw),
            }
        )

    for agent in dynamic_obstacles:
        pose = sample_dynamic_pose(agent, t, use_prediction=True)
        if pose is None:
            continue
        ax, az, ayaw = pose
        later = sample_dynamic_pose(agent, t + CONTROL_DT, use_prediction=True)
        if later is None:
            vx = vy = 0.0
        else:
            vx, vy = (
                (later[0] - ax) / CONTROL_DT,
                (later[1] - az) / CONTROL_DT,
            )
        x, y = _xy(ax, az)
        out.append(
            {
                "id": agent.id,
                "position": [x, y, float(getattr(agent, "y", 0.0))],
                "velocity": [float(vx), float(vy), 0.0],
                "shape": "box",
                "size": [
                    float(agent.length),
                    float(agent.width),
                    float(getattr(agent, "height", 1.0)),
                ],
                "kind": agent.type,
                "yaw": float(ayaw),
            }
        )
    return out


def controller_mode(
    diagnostics, reached_goal: bool, replanned: bool
) -> str:
    """Collapse the filter's diagnostics into the UI's single mode enum.

    Checked most severe first: arriving ends the run, an infeasible program is
    the emergency stop, a standing replan request outranks a routine
    correction, and anything else that changed the command is an intervention.
    """
    if reached_goal:
        return "goal"
    if diagnostics is not None and diagnostics.infeasible:
        return "estop"
    if replanned or (diagnostics is not None and diagnostics.replan_required):
        return "replanning"
    if diagnostics is not None and diagnostics.intervened:
        return "intervention"
    return "tracking"


def _steer_of(command) -> float:
    if command.steer_cmd is not None:
        return float(command.steer_cmd)
    return float(command.steer_rate)


def controller_state(
    nominal,
    applied,
    diagnostics,
    mode: str,
    source: str,
    latency_ms: float | None,
) -> dict:
    return {
        "mode": mode,
        "nominal": {"a": float(nominal.accel), "steer": _steer_of(nominal)},
        "actual": {"a": float(applied.accel), "steer": _steer_of(applied)},
        "barrier_min": (
            None
            if diagnostics is None or not np.isfinite(diagnostics.min_clearance_m)
            else float(diagnostics.min_clearance_m)
        ),
        "latency_ms": None if latency_ms is None else float(latency_ms),
        "source": source,
    }


@dataclass
class EpisodeMetrics:
    collisions: int = 0
    min_clearance: Optional[float] = None
    replans: int = 0
    interventions: int = 0
    goal_reached: bool = False
    start_gap_m: float = 0.0
    travelled_m: float = 0.0

    def path_efficiency(self) -> Optional[float]:
        """Straight-line distance over distance driven, once the goal is met.

        Reported only for a run that arrived. For one that stalls, travelled
        distance is small while the remaining gap is large, which inverts the
        ratio and reads as a suspiciously efficient run - a stalled episode
        measured 12.18 before this guard.
        """
        if not self.goal_reached or self.travelled_m <= 1e-6:
            return None
        return self.start_gap_m / self.travelled_m

    def to_dict(self) -> dict:
        return {
            "collisions": self.collisions,
            "min_clearance": self.min_clearance,
            "replans": self.replans,
            "interventions": self.interventions,
            "path_efficiency": self.path_efficiency(),
            "goal_reached": self.goal_reached,
        }


def state_frame(
    tick: int,
    sim_time: float,
    seed: int,
    state: VehicleState,
    goal_xz: tuple[float, float] | None,
    obstacles: list[dict],
    candidate_paths: list[list[dict]],
    selected_path: list[dict],
    fallback_path: list[dict],
    trail: list[dict],
    controller: dict,
    metrics: dict,
    flow: list | None = None,
    conditioning: dict | None = None,
) -> dict:
    if goal_xz is None:
        goal = None
    else:
        goal_x, goal_y = _xy(goal_xz[0], goal_xz[1])
        goal = [goal_x, goal_y, 0.0]
    frame = {
        "type": "state",
        "version": SCHEMA_VERSION,
        "tick": tick,
        "sim_time": float(sim_time),
        "seed": seed,
        "rover": rover_state(state),
        "goal": goal,
        "obstacles": obstacles,
        "candidate_paths": candidate_paths,
        "selected_path": selected_path,
        "fallback_paths": fallback_path,
        "executed_trail": trail,
        "controller": controller,
        "metrics": metrics,
        "measured": True,
    }
    if flow:
        frame["flow"] = flow
    if conditioning:
        frame["conditioning"] = conditioning
    return frame


def _body_clearance(state: VehicleState, scene, params: VehicleParams) -> float:
    body = OrientedBox(state.x, state.z, state.yaw, params.length, params.width)
    worst = float("inf")
    for box in scene.boxes:
        other = OrientedBox(box.x, box.z, box.yaw, box.length, box.width)
        gap, _ = oriented_box_clearance(body, other)
        worst = min(worst, float(gap))
    for circle in scene.circles:
        gap = (
            float(np.hypot(circle.x - state.x, circle.z - state.z))
            - float(circle.radius)
            - 0.5 * max(params.length, params.width)
        )
        worst = min(worst, gap)
    return worst


def _candidate_collisions(candidates, state: VehicleState, grid, limits) -> tuple[list, list]:
    cos_yaw, sin_yaw = np.cos(state.yaw), np.sin(state.yaw)
    flags, clearances = [], []
    for candidate in candidates:
        flat = footprint_centres(candidate.trajectory, limits).reshape(-1, 2)
        world_x = state.x + flat[:, 0] * cos_yaw - flat[:, 1] * sin_yaw
        world_z = state.z + flat[:, 0] * sin_yaw + flat[:, 1] * cos_yaw
        hit = False
        for wx, wz in zip(world_x, world_z):
            gx, gz = grid.world_to_grid(float(wx), float(wz))
            if not grid.in_bounds(gx, gz) or grid.data[gz, gx] > 0:
                hit = True
                break
        flags.append(not hit)
        clearances.append(0.0 if hit else 1.0)
    return flags, clearances


def episode(
    scenario_path: Path,
    planner: FlowMatchingPlanner,
    use_cbf: bool = True,
    duration_s: float = 45.0,
    seed: int = 0,
    facility_path: Path = FACILITY_PATH,
) -> Iterator[dict]:
    """Run one scenario and yield a `StateFrame` per control step.

    The model proposes every `REPLAN_PERIOD_S`; candidates are ranked and the
    best collision-free one becomes the tracker's reference. When every
    candidate is rejected the previous reference is held and the frame reports
    `source: "fallback"`, which is the point at which a deployment should call
    the classical planner instead - CLAUDE.md requires that path be logged and
    labelled rather than presented as the learned result.
    """
    spec = load_scenario(scenario_path)
    facility = load_facility(facility_path)
    params = VehicleParams()
    limits = _vehicle_limits()
    statics = list(spec.static_obstacles)
    dynamics = list(spec.dynamic_obstacles)

    base_facility = build_facility_occupancy(facility, MAP_RESOLUTION)
    base_static = build_facility_occupancy(
        facility, MAP_RESOLUTION, static_obstacles=statics
    )

    state = VehicleState(
        x=spec.start.x,
        z=spec.start.z,
        yaw=spec.start.yaw,
        speed=spec.start.speed,
        steering=spec.start.steering,
        t=0.0,
    )
    goal_xz = (spec.goal.x, spec.goal.z)

    metrics = EpisodeMetrics(
        start_gap_m=float(np.hypot(goal_xz[0] - state.x, goal_xz[1] - state.z))
    )
    min_clearance = float("inf")

    tracker: TrajectoryTracker | None = None
    source = "fallback"
    candidate_paths: list[list[dict]] = []
    selected_path: list[dict] = []
    fallback_path: list[dict] = []
    trail: list[dict] = []
    plan_latency_ms: float | None = None
    last_plan_t = -1e9
    previous = (state.x, state.z)
    dt_traj = float(planner.schema.get("dt", 0.2))

    for tick in range(int(duration_s / CONTROL_DT)):
        gap = float(np.hypot(goal_xz[0] - state.x, goal_xz[1] - state.z))
        if gap <= GOAL_RADIUS_M:
            metrics.goal_reached = True

        replanned = False
        if not metrics.goal_reached and (
            tracker is None or state.t - last_plan_t >= REPLAN_PERIOD_S
        ):
            replanned = True
            metrics.replans += 1
            last_plan_t = state.t

            facility_crop = base_facility.crop_local(state.x, state.z, MAP_SIZE_M)
            static_crop = base_static.crop_local(state.x, state.z, MAP_SIZE_M)
            dynamic_stack = None
            if planner.uses_dynamic:
                dynamic_stack = np.stack(
                    [
                        dynamic_occupancy_at_time(
                            base_static,
                            dynamics,
                            state.t + k * DYNAMIC_SLICE_DT,
                            use_prediction=True,
                        )
                        .crop_local(state.x, state.z, MAP_SIZE_M)
                        .data
                        for k in range(DYNAMIC_SLICES)
                    ]
                )

            ego = {"x": state.x, "z": state.z, "yaw": state.yaw}
            gx, gy, gyaw = world_to_local(
                np.array(goal_xz[0]), np.array(goal_xz[1]), np.array(0.0), ego
            )
            result = planner.plan(
                facility_occ=facility_crop.data,
                static_occ=static_crop.data,
                dynamic_occ=dynamic_stack,
                goal_local=(float(gx), float(gy)),
                goal_yaw_local=float(gyaw),
                speed=state.speed,
                steering=state.steering,
                ego_yaw=state.yaw,
                num_candidates=CANDIDATES,
                num_steps=SOLVER_STEPS,
                seed=seed,
            )
            plan_latency_ms = result.latency_ms_total

            snapshot = dynamic_occupancy_at_time(
                base_static, dynamics, state.t, use_prediction=True
            )
            flags, clearances = _candidate_collisions(
                result.candidates, state, snapshot, limits
            )
            candidate_paths = [
                path_points(c.world_xz(ego), dt=dt_traj, t0=state.t)
                for c in result.candidates
            ]
            pick = best_candidate(result, collision_free=flags, clearance_m=clearances)

            if pick is None:
                source = "fallback"
                selected_path = []
                fallback_path = path_points([(state.x, state.z), goal_xz])
                if tracker is None:
                    tracker = TrajectoryTracker(
                        [(state.x, state.z), goal_xz],
                        params=params,
                        cruise_speed=CRUISE_SPEED,
                        lookahead=TRACKER_LOOKAHEAD,
                        steer_gain=TRACKER_STEER_GAIN,
                    )
            else:
                source = "learned"
                fallback_path = []
                polyline = pick.candidate.world_xz(ego)
                selected_path = path_points(polyline, dt=dt_traj, t0=state.t)
                tracker = TrajectoryTracker(
                    [(float(a), float(b)) for a, b in polyline],
                    params=params,
                    cruise_speed=CRUISE_SPEED,
                    lookahead=TRACKER_LOOKAHEAD,
                    steer_gain=TRACKER_STEER_GAIN,
                )

        perceived = build_scene(
            state.t,
            facility=facility,
            static_obstacles=statics,
            dynamic_obstacles=dynamics,
            dynamics_mode="causal_cv",
        )
        nxt, nominal, applied, diagnostics = closed_loop_step(
            state,
            tracker,
            perceived,
            CONTROL_DT,
            params=params,
            use_cbf=use_cbf,
        )

        truth = build_scene(
            nxt.t,
            facility=facility,
            static_obstacles=statics,
            dynamic_obstacles=dynamics,
            dynamics_mode="snapshot_gt",
        )
        clearance = _body_clearance(nxt, truth, params)
        if clearance < -1e-6:
            metrics.collisions += 1
        min_clearance = min(min_clearance, clearance)
        metrics.min_clearance = (
            None if not np.isfinite(min_clearance) else float(min_clearance)
        )

        if diagnostics is not None and diagnostics.intervened:
            metrics.interventions += 1

        metrics.travelled_m += float(
            np.hypot(nxt.x - previous[0], nxt.z - previous[1])
        )
        previous = (nxt.x, nxt.z)

        trail_x, trail_y = _xy(nxt.x, nxt.z)
        trail.append({"x": trail_x, "y": trail_y, "t": float(nxt.t)})

        mode = controller_mode(diagnostics, metrics.goal_reached, replanned)
        latency = plan_latency_ms if replanned else (
            None
            if diagnostics is None
            else float(diagnostics.solve_time_s) * 1000.0
        )

        yield state_frame(
            tick=tick,
            sim_time=nxt.t,
            seed=seed,
            state=nxt,
            goal_xz=goal_xz,
            obstacles=obstacle_states(statics, dynamics, nxt.t),
            candidate_paths=candidate_paths,
            selected_path=selected_path,
            fallback_path=fallback_path,
            trail=list(trail),
            controller=controller_state(
                nominal, applied, diagnostics, mode, source, latency
            ),
            metrics=metrics.to_dict(),
        )

        state = nxt
        if metrics.goal_reached:
            break


def write_episode(
    scenario_path: Path,
    checkpoint: Path,
    out_path: Path,
    use_cbf: bool = True,
    duration_s: float = 45.0,
    seed: int = 0,
) -> dict:
    """Run an episode and write one JSON frame per line.

    A frame log is what the UI can consume without a socket: a `SimSource` that
    replays these lines is indistinguishable, to the screens, from a live
    simulator, because the frames are the schema they already read.
    """
    planner = FlowMatchingPlanner.load(checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = 0
    last: dict | None = None
    with out_path.open("w", encoding="utf-8") as handle:
        for frame in episode(
            scenario_path, planner, use_cbf=use_cbf, duration_s=duration_s, seed=seed
        ):
            handle.write(json.dumps(frame) + "\n")
            frames += 1
            last = frame
    return {
        "path": str(out_path),
        "frames": frames,
        "final_metrics": None if last is None else last["metrics"],
        "use_cbf": use_cbf,
    }


VEC3 = "Vec3"


def validate_frame(frame: dict) -> list[str]:
    """Check a frame against `frontend/src/types/sim.ts`, field by field.

    The schema lives in TypeScript, so nothing checks it at runtime on this
    side. Emitting a field the UI does not declare, or omitting one it reads,
    would surface as a blank panel rather than an error, so the contract is
    asserted here instead.
    """
    problems: list[str] = []

    def want(container: dict, key: str, kinds, where: str):
        if key not in container:
            problems.append(f"{where}: missing {key}")
            return None
        value = container[key]
        if not isinstance(value, kinds) or isinstance(value, bool) and kinds is not bool:
            problems.append(f"{where}.{key}: {type(value).__name__} not {kinds}")
        return value

    def vec3(value, where: str):
        if not isinstance(value, list) or len(value) != 3:
            problems.append(f"{where}: expected Vec3")
            return
        for item in value:
            if not isinstance(item, (int, float)) or isinstance(item, bool):
                problems.append(f"{where}: non-numeric component")

    def path(points, where: str):
        if not isinstance(points, list):
            problems.append(f"{where}: expected PathPoint[]")
            return
        for index, point in enumerate(points):
            if not isinstance(point, dict):
                problems.append(f"{where}[{index}]: expected object")
                continue
            for axis in ("x", "y"):
                if not isinstance(point.get(axis), (int, float)):
                    problems.append(f"{where}[{index}].{axis}: missing or non-numeric")
            extra = set(point) - {"x", "y", "t"}
            if extra:
                problems.append(f"{where}[{index}]: undeclared keys {sorted(extra)}")

    expected = {
        "type", "version", "tick", "sim_time", "seed", "rover", "goal",
        "obstacles", "candidate_paths", "selected_path", "fallback_paths",
        "executed_trail", "controller", "metrics", "measured",
    }
    missing = expected - set(frame)
    undeclared = set(frame) - expected - {"flow", "conditioning"}
    flow = frame.get("flow", "absent")
    if flow != "absent":
        if not isinstance(flow, list):
            problems.append("flow: expected a list")
        else:
            for index, curve in enumerate(flow):
                where = f"flow[{index}]"
                if not isinstance(curve, dict):
                    problems.append(f"{where}: expected object")
                    continue
                if not isinstance(curve.get("t"), (int, float)):
                    problems.append(f"{where}.t: non-numeric")
                if not isinstance(curve.get("index"), int):
                    problems.append(f"{where}.index: expected int")
                if not isinstance(curve.get("selected"), bool):
                    problems.append(f"{where}.selected: expected bool")
                points = curve.get("points")
                if not isinstance(points, list):
                    problems.append(f"{where}.points: expected list")
                else:
                    for point in points:
                        if not isinstance(point, dict) or not isinstance(
                            point.get("x"), (int, float)
                        ) or not isinstance(point.get("y"), (int, float)):
                            problems.append(f"{where}.points: expected {{x, y}}")
                            break
    conditioning = frame.get("conditioning", "absent")
    if conditioning != "absent":
        if not isinstance(conditioning, dict):
            problems.append("conditioning: expected object")
        else:
            size = conditioning.get("size")
            extra = set(conditioning) - {"size", "facility", "static"}
            if extra:
                problems.append(f"conditioning: undeclared {sorted(extra)}")
            if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= 48:
                problems.append("conditioning.size: expected int")
            else:
                for key in ("facility", "static"):
                    cells = conditioning.get(key)
                    if (
                        not isinstance(cells, list)
                        or len(cells) != size * size
                        or any(cell not in (0, 1) or isinstance(cell, bool) for cell in cells)
                    ):
                        problems.append(f"conditioning.{key}: expected {size * size} cells of 0/1")
    if missing:
        problems.append(f"frame: missing {sorted(missing)}")
    if undeclared:
        problems.append(f"frame: undeclared {sorted(undeclared)}")

    if frame.get("type") != "state":
        problems.append("frame.type: must be 'state'")
    if frame.get("version") != SCHEMA_VERSION:
        problems.append(f"frame.version: must be {SCHEMA_VERSION}")
    if not isinstance(frame.get("measured"), bool):
        problems.append("frame.measured: must be boolean")
    want(frame, "tick", int, "frame")
    want(frame, "sim_time", (int, float), "frame")
    want(frame, "seed", int, "frame")

    rover = frame.get("rover")
    if isinstance(rover, dict):
        for key in ("x", "y", "z", "heading", "speed", "steering"):
            want(rover, key, (int, float), "rover")
        extra = set(rover) - {"x", "y", "z", "heading", "speed", "steering"}
        if extra:
            problems.append(f"rover: undeclared keys {sorted(extra)}")
    else:
        problems.append("rover: expected object")

    if frame.get("goal") is not None:
        vec3(frame["goal"], "goal")

    for index, obstacle in enumerate(frame.get("obstacles", [])):
        where = f"obstacles[{index}]"
        want(obstacle, "id", str, where)
        vec3(obstacle.get("position"), f"{where}.position")
        vec3(obstacle.get("velocity"), f"{where}.velocity")
        vec3(obstacle.get("size"), f"{where}.size")
        if obstacle.get("shape") not in ("box", "cylinder"):
            problems.append(f"{where}.shape: must be box or cylinder")
        extra = set(obstacle) - {
            "id", "position", "velocity", "shape", "size", "kind", "yaw"
        }
        if extra:
            problems.append(f"{where}: undeclared keys {sorted(extra)}")

    for index, candidate in enumerate(frame.get("candidate_paths", [])):
        path(candidate, f"candidate_paths[{index}]")
    path(frame.get("selected_path"), "selected_path")
    path(frame.get("fallback_paths"), "fallback_paths")
    path(frame.get("executed_trail"), "executed_trail")

    controller = frame.get("controller")
    if isinstance(controller, dict):
        if controller.get("mode") not in CONTROLLER_MODES:
            problems.append(f"controller.mode: {controller.get('mode')!r} not in enum")
        if controller.get("source") not in ("learned", "fallback"):
            problems.append("controller.source: must be learned or fallback")
        for key in ("nominal", "actual"):
            block = controller.get(key)
            if not isinstance(block, dict) or set(block) != {"a", "steer"}:
                problems.append(f"controller.{key}: expected {{a, steer}}")
                continue
            for inner in ("a", "steer"):
                if not isinstance(block[inner], (int, float)):
                    problems.append(f"controller.{key}.{inner}: non-numeric")
        for key in ("barrier_min", "latency_ms"):
            value = controller.get(key, "absent")
            if value != "absent" and value is not None and not isinstance(
                value, (int, float)
            ):
                problems.append(f"controller.{key}: must be number or null")
        extra = set(controller) - {
            "mode", "nominal", "actual", "barrier_min", "latency_ms", "source"
        }
        if extra:
            problems.append(f"controller: undeclared keys {sorted(extra)}")
    else:
        problems.append("controller: expected object")

    metrics = frame.get("metrics")
    numeric = ("collisions", "min_clearance", "replans", "interventions", "path_efficiency")
    if isinstance(metrics, dict):
        for key in numeric:
            value = metrics.get(key, "absent")
            if value == "absent":
                problems.append(f"metrics: missing {key}")
            elif value is not None and not isinstance(value, (int, float)):
                problems.append(f"metrics.{key}: must be number or null")
        reached = metrics.get("goal_reached", "absent")
        if reached != "absent" and reached is not None and not isinstance(reached, bool):
            problems.append("metrics.goal_reached: must be boolean or null")
        extra = set(metrics) - set(numeric) - {"goal_reached"}
        if extra:
            problems.append(f"metrics: undeclared keys {sorted(extra)}")
    else:
        problems.append("metrics: expected object")

    return problems


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("backend/artifacts/frames.jsonl"))
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--safety",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run with the CBF filter engaged",
    )
    args = parser.parse_args()

    summary = write_episode(
        args.scenario,
        args.checkpoint,
        args.out,
        use_cbf=args.safety,
        duration_s=args.duration,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
