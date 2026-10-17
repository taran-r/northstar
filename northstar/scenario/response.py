"""Measured expert responses. Behavior tags are not treated as evidence."""

from __future__ import annotations

import math
from typing import Any, Sequence, Union

from northstar.facility.occupancy import sample_dynamic_pose
from northstar.facility.schema import DynamicObject
from northstar.geometry import OrientedBox, oriented_boxes_collide
from northstar.vehicle.bicycle import VehicleParams, VehicleState


StateLike = Union[VehicleState, dict[str, Any]]


def _fields(state: StateLike) -> tuple[float, float, float, float, float, float]:
    if isinstance(state, dict):
        return (
            float(state["x"]),
            float(state["z"]),
            float(state.get("yaw", 0.0)),
            float(state.get("speed", 0.0)),
            float(state.get("steering", 0.0)),
            float(state.get("t", 0.0)),
        )
    return state.x, state.z, state.yaw, state.speed, state.steering, state.t


def motion_response(
    states: Sequence[StateLike],
    goal_x: float,
    goal_z: float,
    *,
    stop_speed: float = 0.12,
    brake_accel: float = -0.7,
    goal_clearance_m: float = 2.5,
) -> dict[str, float | bool]:
    """Mid-route stopping and braking, ignoring the final stop at the goal."""
    stop_time = 0.0
    longest_stop = 0.0
    current = 0.0
    braked = False
    min_speed = 1e9
    samples = 0
    for i, state in enumerate(states):
        x, z, _yaw, speed, _steer, t = _fields(state)
        if math.hypot(x - goal_x, z - goal_z) < goal_clearance_m:
            current = 0.0
            continue
        samples += 1
        min_speed = min(min_speed, speed)
        if speed < stop_speed:
            dt = 0.2
            if i > 0:
                _, _, _, _, _, prev_t = _fields(states[i - 1])
                dt = max(t - prev_t, 1e-3)
            current += dt
            stop_time += dt
            longest_stop = max(longest_stop, current)
        else:
            current = 0.0
        if i > 0:
            _, _, _, prev_speed, _, prev_t = _fields(states[i - 1])
            dt = max(t - prev_t, 1e-3)
            if (speed - prev_speed) / dt <= brake_accel:
                braked = True
    return {
        "midroute_stop_s": float(stop_time),
        "longest_stop_s": float(longest_stop),
        "min_midroute_speed": float(min_speed if samples else 0.0),
        "yielded": bool(longest_stop >= 0.6),
        "braked": bool(braked),
    }


def polyline_deviation(reference: Sequence[StateLike], candidate: Sequence[StateLike]) -> float:
    """Mean distance from candidate samples to the reference polyline."""
    if len(reference) < 2 or not candidate:
        return 0.0
    ref = [(_fields(s)[0], _fields(s)[1]) for s in reference]
    total = 0.0
    far = 0
    for state in candidate:
        x, z, *_ = _fields(state)
        best = min(math.hypot(x - rx, z - rz) for rx, rz in ref)
        total += best
        if best > 1.8:
            far += 1
    mean = total / len(candidate)
    # Require a sustained offset, not a single corner cut.
    if far / len(candidate) < 0.2:
        return min(mean, 1.0)
    return mean


def route_conflict(
    path: Sequence[StateLike],
    dynamics: Sequence[DynamicObject],
    params: VehicleParams | None = None,
    *,
    use_prediction: bool = True,
    stride: int = 2,
) -> dict[str, float | bool]:
    """True when a dynamic footprint overlaps the rover footprint on this route."""
    params = params or VehicleParams()
    if not dynamics or len(path) < 2:
        return {"conflict": False, "min_separation_m": float("inf"), "n_conflict_samples": 0}
    min_sep = float("inf")
    hits = 0
    for state in path[::stride]:
        x, z, yaw, _speed, _steer, t = _fields(state)
        ego = VehicleState(x=x, z=z, yaw=yaw, t=t)
        box = ego.footprint(params)
        for dyn in dynamics:
            pose = sample_dynamic_pose(dyn, t, use_prediction=use_prediction)
            if pose is None:
                continue
            dx, dz, dyaw = pose
            sep = math.hypot(x - dx, z - dz)
            min_sep = min(min_sep, sep)
            other = OrientedBox(dx, dz, dyaw, dyn.length, dyn.width)
            if oriented_boxes_collide(box, other, margin=0.05):
                hits += 1
    return {
        "conflict": hits > 0,
        "min_separation_m": float(min_sep),
        "n_conflict_samples": hits,
    }
