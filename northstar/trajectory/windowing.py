"""Extract fixed-horizon local windows from full expert routes."""

from __future__ import annotations

from typing import Optional

import numpy as np

from northstar.facility.occupancy import (
    OccupancyGrid,
    build_facility_occupancy,
    sample_dynamic_pose,
)
from northstar.facility.schema import FacilitySpec, ScenarioSpec
from northstar.geometry import OrientedBox
from northstar.trajectory.format import (
    DEFAULT_DT,
    DEFAULT_HORIZON_STEPS,
    TrajectoryRecord,
    TrajectoryWindow,
)


def _local_subgoal(
    states_xyz: np.ndarray,
    global_goal: dict[str, float],
    horizon: int,
) -> dict[str, float]:
    """If global goal is beyond the window, use the window's final pose as subgoal."""
    last = states_xyz[min(horizon - 1, len(states_xyz) - 1)]
    gx, gz = global_goal["x"], global_goal["z"]
    # Distance from window start to global goal
    dist_goal = float(np.hypot(gx - states_xyz[0, 0], gz - states_xyz[0, 1]))
    window_span = float(
        np.hypot(last[0] - states_xyz[0, 0], last[1] - states_xyz[0, 1])
    )
    if dist_goal > window_span + 1.0:
        return {
            "x": float(last[0]),
            "z": float(last[1]),
            "yaw": float(last[2]),
            "is_local": 1.0,
        }
    return {
        "x": float(gx),
        "z": float(gz),
        "yaw": float(global_goal.get("yaw", last[2])),
        "is_local": 0.0,
    }


def extract_windows(
    facility: FacilitySpec,
    scenario: ScenarioSpec,
    record: TrajectoryRecord,
    horizon: int = DEFAULT_HORIZON_STEPS,
    dt: float = DEFAULT_DT,
    map_size_m: float = 20.0,
    map_resolution: float = 0.25,
    pred_steps: int = 8,
    stride: int = 4,
    facility_occ: Optional[OccupancyGrid] = None,
    static_occ: Optional[OccupancyGrid] = None,
) -> list[TrajectoryWindow]:
    states = record.states
    accels = record.accel_cmds
    if len(states) < horizon:
        return []

    facility_occ = facility_occ or build_facility_occupancy(facility, resolution=map_resolution)
    static_occ = static_occ or build_facility_occupancy(
        facility, resolution=map_resolution, static_obstacles=scenario.static_obstacles
    )

    global_goal = {
        "x": scenario.goal.x,
        "z": scenario.goal.z,
        "yaw": scenario.goal.yaw,
    }
    behavior = scenario.metadata.get("behavior_tag", "unknown")
    windows: list[TrajectoryWindow] = []

    for start_i in range(0, len(states) - horizon, stride):
        ego = states[start_i]
        future_rows = []
        for k in range(horizon):
            s = states[start_i + k]
            a = accels[min(start_i + k, len(accels) - 1)] if accels else 0.0
            future_rows.append(
                [s.x, s.z, s.yaw, s.speed, s.steering, a, s.t - ego.t]
            )
        future = np.asarray(future_rows, dtype=np.float32)
        subgoal = _local_subgoal(future, global_goal, horizon)

        # Local maps centered on ego
        fac_local = facility_occ.crop_local(ego.x, ego.z, map_size_m)
        stat_local = static_occ.crop_local(ego.x, ego.z, map_size_m)

        # Predicted dynamic occupancy (stamp onto local static crop)
        base_local = stat_local.data
        dyn_stack = []
        for j in range(pred_steps):
            t = ego.t + j * dt
            local = OccupancyGrid(
                stat_local.origin_x,
                stat_local.origin_z,
                stat_local.resolution,
                stat_local.width,
                stat_local.height,
                data=base_local.copy(),
            )
            for dyn in scenario.dynamic_obstacles:
                pose = sample_dynamic_pose(dyn, t, use_prediction=True)
                if pose is None:
                    continue
                x, z, yaw = pose
                box = OrientedBox(x, z, yaw, dyn.length, dyn.width)
                aabb = box.aabb()
                gx0, gz0 = local.world_to_grid(aabb.min_x, aabb.min_z)
                gx1, gz1 = local.world_to_grid(aabb.max_x, aabb.max_z)
                for gz in range(gz0, gz1 + 1):
                    for gx in range(gx0, gx1 + 1):
                        if not local.in_bounds(gx, gz):
                            continue
                        wx, wz = local.grid_to_world(gx, gz)
                        if box.contains_point(wx, wz):
                            local.set_occupied(gx, gz)
            dyn_stack.append(local.data)
        dynamic_occ = np.stack(dyn_stack, axis=0).astype(np.float32)

        windows.append(
            TrajectoryWindow(
                sample_id=f"{record.trajectory_id}_w{start_i:04d}",
                scenario_id=scenario.scenario_id,
                trajectory_id=record.trajectory_id,
                horizon=horizon,
                dt=dt,
                ego0={
                    "x": ego.x,
                    "z": ego.z,
                    "yaw": ego.yaw,
                    "speed": ego.speed,
                    "steering": ego.steering,
                    "t": ego.t,
                },
                future=future,
                subgoal=subgoal,
                global_goal=global_goal,
                static_occ=stat_local.data,
                dynamic_occ=dynamic_occ,
                facility_occ=fac_local.data,
                oracle_dynamics=scenario.oracle_dynamics,
                behavior_tag=behavior,
                mode_id=record.mode_id,
                metadata={
                    "start_index": start_i,
                    "map_size_m": map_size_m,
                    "map_resolution": map_resolution,
                    "pred_steps": pred_steps,
                },
            )
        )
    return windows
