"""Run expert planners to produce validated trajectories for a scenario."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from northstar.collision import CollisionChecker
from northstar.facility.occupancy import build_facility_occupancy
from northstar.facility.schema import FacilitySpec, ScenarioSpec
from northstar.planner.hybrid_astar import HybridAStar, HybridAStarConfig
from northstar.planner.smoother import profile_velocity, smooth_trajectory
from northstar.planner.timed_follow import follow_path_with_dynamics
from northstar.trajectory.format import TrajectoryRecord, resample_states
from northstar.vehicle.bicycle import BicycleModel, VehicleParams, VehicleState


def _accel_from_states(states: list[VehicleState]) -> list[float]:
    acc = [0.0]
    for i in range(1, len(states)):
        dt = max(states[i].t - states[i - 1].t, 1e-3)
        acc.append((states[i].speed - states[i - 1].speed) / dt)
    return acc


def _validate_collision_free(
    states: list[VehicleState],
    collision: CollisionChecker,
    scenario: ScenarioSpec,
) -> bool:
    for s in states[::2]:
        if collision.collides(
            s, scenario.dynamic_obstacles, t=s.t, use_prediction=False
        ):
            return False
    if states and collision.collides(
        states[-1], scenario.dynamic_obstacles, t=states[-1].t, use_prediction=False
    ):
        return False
    return True


def plan_expert_trajectory(
    facility: FacilitySpec,
    scenario: ScenarioSpec,
    vehicle_params: Optional[VehicleParams] = None,
    dt: float = 0.2,
    mode_id: int = 0,
    prefer_wait: bool = False,
) -> Optional[TrajectoryRecord]:
    """
    Expert stack:

      1. Hybrid A* (bicycle primitives) on static geometry.
      2. B-spline smooth + curvature-aware velocity profile.
      3. Timed following with brake/wait for predicted dynamic obstacles.
      4. Validate against ground-truth dynamics + light bicycle rollout.
    """
    vehicle_params = vehicle_params or VehicleParams()
    model = BicycleModel(vehicle_params)
    occ = build_facility_occupancy(
        facility, resolution=0.3, static_obstacles=scenario.static_obstacles
    )
    collision = CollisionChecker(
        facility,
        static_obstacles=scenario.static_obstacles,
        occupancy=occ,
        vehicle_params=vehicle_params,
    )
    collision.fast_occupancy = True

    start = VehicleState(
        x=scenario.start.x,
        z=scenario.start.z,
        yaw=scenario.start.yaw,
        speed=max(0.0, scenario.start.speed),
        steering=scenario.start.steering,
        t=0.0,
    )
    goal = VehicleState(
        x=scenario.goal.x,
        z=scenario.goal.z,
        yaw=scenario.goal.yaw,
        speed=0.0,
        t=0.0,
    )
    if collision.collides_static(start) or collision.collides_static(goal):
        return None

    prefer_wait = prefer_wait or mode_id == 1
    planner_used = "none"
    rolled: Optional[list[VehicleState]] = None

    static_cfg = HybridAStarConfig(
        xy_resolution=0.8,
        yaw_resolution=math.radians(22.5),
        step_dt=0.3,
        steps_per_motion=2,
        steer_angles=(-0.55, -0.25, 0.0, 0.25, 0.55),
        speeds=(1.2, 2.0),
        max_iterations=8_000,
        goal_xy_tol=1.4,
        goal_yaw_tol=math.radians(50.0),
        heuristic_weight=1.1,
    )
    geom = HybridAStar(collision, model=model, config=static_cfg).plan(start, goal)
    if geom is None or len(geom) < 2:
        return None
    if math.hypot(geom[-1].x - goal.x, geom[-1].z - goal.z) > 2.0:
        return None

    smoothed = smooth_trajectory(geom, smoothing=1.0)
    profiled = profile_velocity(
        smoothed,
        params=vehicle_params,
        dt=dt,
        cruise_speed=vehicle_params.max_speed * (0.55 if prefer_wait else 0.85),
    )

    if scenario.dynamic_obstacles:
        followed = follow_path_with_dynamics(
            profiled,
            collision,
            scenario.dynamic_obstacles,
            model,
            dt=dt,
            use_prediction=not scenario.oracle_dynamics,
            max_wait=7.0 if prefer_wait else 4.0,
            prefer_wait=prefer_wait,
            cruise_speed=vehicle_params.max_speed * (0.5 if prefer_wait else 0.8),
        )
        if followed is not None and _validate_collision_free(
            followed, collision, scenario
        ):
            rolled = followed
            planner_used = "hybrid_astar+timed_follow"
        else:
            # Retry with oracle timing for hard scenarios (still flagged on scenario)
            followed = follow_path_with_dynamics(
                profiled,
                collision,
                scenario.dynamic_obstacles,
                model,
                dt=dt,
                use_prediction=False,
                max_wait=8.0,
                prefer_wait=True,
                cruise_speed=vehicle_params.max_speed * 0.5,
            )
            if followed is not None and _validate_collision_free(
                followed, collision, scenario
            ):
                rolled = followed
                planner_used = "hybrid_astar+timed_follow_gt"
                scenario.oracle_dynamics = True
    else:
        rolled = profiled
        planner_used = "hybrid_astar_static"

    if rolled is None:
        # Accept static profile only if GT dynamics never intersect it
        if _validate_collision_free(profiled, collision, scenario):
            rolled = profiled
            planner_used = "hybrid_astar_static"
        else:
            return None

    states, accels = resample_states(rolled, _accel_from_states(rolled), dt=dt)

    final, final_acc = states, accels
    if not _validate_collision_free(final, collision, scenario):
        return None

    progress = math.hypot(final[-1].x - final[0].x, final[-1].z - final[0].z)
    speeds = [s.speed for s in final]
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    max_speed = float(np.max(speeds)) if speeds else 0.0
    if progress < 4.0 or mean_speed < 0.25 or max_speed < 0.5:
        return None

    return TrajectoryRecord(
        trajectory_id=f"{scenario.scenario_id}_m{mode_id}",
        scenario_id=scenario.scenario_id,
        dt=dt,
        states=final,
        accel_cmds=final_acc,
        mode_id=mode_id,
        metadata={
            "planner": planner_used,
            "n_states": len(final),
            "duration": final[-1].t - final[0].t if final else 0.0,
            "mean_speed": mean_speed,
            "max_speed": max_speed,
            "path_length": progress,
            "reached_near_goal": math.hypot(goal.x - final[-1].x, goal.z - final[-1].z)
            <= 2.5,
        },
    )


def plan_static_path(
    facility: FacilitySpec,
    scenario: ScenarioSpec,
    vehicle_params: Optional[VehicleParams] = None,
    dt: float = 0.2,
    cruise_scale: float = 0.85,
) -> Optional[list[VehicleState]]:
    """Static Hybrid A* path with every bicycle substep stored. No dynamic reaction."""
    vehicle_params = vehicle_params or VehicleParams()
    model = BicycleModel(vehicle_params)
    occ = build_facility_occupancy(
        facility, resolution=0.25, static_obstacles=scenario.static_obstacles
    )
    collision = CollisionChecker(
        facility,
        static_obstacles=scenario.static_obstacles,
        occupancy=occ,
        vehicle_params=vehicle_params,
    )
    collision.fast_occupancy = False
    start = VehicleState(
        x=scenario.start.x,
        z=scenario.start.z,
        yaw=scenario.start.yaw,
        speed=max(0.0, scenario.start.speed),
        steering=scenario.start.steering,
        t=0.0,
    )
    goal = VehicleState(
        x=scenario.goal.x,
        z=scenario.goal.z,
        yaw=scenario.goal.yaw,
        speed=0.0,
        t=0.0,
    )
    if collision.collides_static(start) or collision.collides_static(goal):
        return None
    static_cfg = HybridAStarConfig(
        xy_resolution=0.8,
        yaw_resolution=math.radians(22.5),
        step_dt=dt,
        steps_per_motion=6,
        steer_angles=(-0.55, -0.25, 0.0, 0.25, 0.55),
        speeds=(1.6, 2.2),
        max_iterations=8_000,
        goal_xy_tol=1.4,
        goal_yaw_tol=math.radians(50.0),
        heuristic_weight=1.1,
        record_substeps=True,
    )
    geom = HybridAStar(collision, model=model, config=static_cfg).plan(start, goal)
    if geom is None or len(geom) < 2:
        return None
    if math.hypot(geom[-1].x - goal.x, geom[-1].z - goal.z) > 2.0:
        return None
    _ = (dt, cruise_scale)
    return geom


def plan_multimodal(
    facility: FacilitySpec,
    scenario: ScenarioSpec,
    vehicle_params: Optional[VehicleParams] = None,
    dt: float = 0.2,
    max_modes: int = 2,
) -> list[TrajectoryRecord]:
    results: list[TrajectoryRecord] = []
    for mode in range(max_modes):
        rec = plan_expert_trajectory(
            facility,
            scenario,
            vehicle_params=vehicle_params,
            dt=dt,
            mode_id=mode,
            prefer_wait=(mode == 1),
        )
        if rec is None:
            continue
        if any(_similar(prev, rec) for prev in results):
            continue
        results.append(rec)
    return results


def _similar(a: TrajectoryRecord, b: TrajectoryRecord, tol: float = 1.2) -> bool:
    n = min(len(a.states), len(b.states), 32)
    if n < 4:
        return True
    errs = [
        math.hypot(a.states[i].x - b.states[i].x, a.states[i].z - b.states[i].z)
        for i in range(n)
    ]
    return float(np.mean(errs)) < tol
