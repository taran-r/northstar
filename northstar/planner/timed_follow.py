"""Follow a static geometric path while reacting to predicted dynamic obstacles.

Much faster than full space-time Hybrid A* for dataset generation, while still
producing waiting, braking, and yielding behaviors.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from northstar.collision import CollisionChecker
from northstar.facility.schema import DynamicObject
from northstar.vehicle.bicycle import BicycleModel, ControlInput, VehicleParams, VehicleState


def follow_path_with_dynamics(
    path: list[VehicleState],
    collision: CollisionChecker,
    dynamics: Sequence[DynamicObject],
    model: BicycleModel,
    dt: float = 0.2,
    use_prediction: bool = True,
    max_wait: float = 6.0,
    cruise_speed: Optional[float] = None,
    prefer_wait: bool = False,
) -> Optional[list[VehicleState]]:
    """
    Track polyline `path` with the bicycle model.

    On predicted collision: brake / wait up to `max_wait`, then continue.
    Returns None if the path cannot be completed without GT-invalid motion
    (caller should validate with ground truth separately).
    """
    if len(path) < 2:
        return None

    params = model.params
    v_cruise = cruise_speed or params.max_speed * (0.65 if prefer_wait else 0.85)
    look_ahead = 1.8

    # Polyline arrays
    pts = np.array([[p.x, p.z] for p in path], dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s_cum[-1])
    if total < 0.5:
        return None

    s = VehicleState(
        x=path[0].x,
        z=path[0].z,
        yaw=path[0].yaw,
        speed=min(path[0].speed, v_cruise),
        steering=0.0,
        t=0.0,
    )
    out = [s.copy()]
    s_along = 0.0
    waited = 0.0
    max_steps = int((total / max(v_cruise * 0.25, 0.2) + max_wait * 3) / dt) + 100

    for _ in range(max_steps):
        if s_along >= total - 0.35:
            # Brake to stop near end
            if s.speed < 0.05:
                break
            s = model.step(s, ControlInput(accel=-params.max_decel, steer_cmd=0.0), dt)
            out.append(s.copy())
            continue

        # Pure-pursuit style target
        target_s = min(total, s_along + look_ahead)
        tx, tz = _point_at_s(pts, s_cum, target_s)
        desired_yaw = math.atan2(tz - s.z, tx - s.x)
        yaw_err = (desired_yaw - s.yaw + math.pi) % (2 * math.pi) - math.pi
        steer = float(
            np.clip(1.8 * yaw_err, -params.max_steer, params.max_steer)
        )

        # Progress speed with terminal slowdown
        remain = total - s_along
        v_tgt = min(v_cruise, math.sqrt(max(0.0, 2 * params.max_decel * remain)))
        if abs(yaw_err) > 0.6:
            v_tgt = min(v_tgt, 0.9)

        # Predictive collision check a short horizon ahead
        blocked = _is_blocked_ahead(
            s, steer, v_tgt, model, collision, dynamics, dt, use_prediction, prefer_wait
        )
        if blocked:
            # Brake / wait
            accel = -params.max_decel if s.speed > 0.05 else 0.0
            s = model.step(s, ControlInput(accel=accel, steer_cmd=0.0), dt)
            if s.speed < 0.05:
                s.speed = 0.0
                waited += dt
            out.append(s.copy())
            if waited > max_wait:
                return None
            continue

        waited = 0.0
        accel = float(
            np.clip((v_tgt - s.speed) / dt, -params.max_decel, params.max_accel)
        )
        prev = s.copy()
        s = model.step(s, ControlInput(accel=accel, steer_cmd=steer), dt)
        # Advance arc-length by projected motion
        s_along = min(total, s_along + math.hypot(s.x - prev.x, s.z - prev.z))
        out.append(s.copy())

        if collision.collides(s, dynamics, t=s.t, use_prediction=use_prediction):
            return None

    if math.hypot(out[-1].x - path[-1].x, out[-1].z - path[-1].z) > 3.5:
        return None
    return out


def _point_at_s(pts: np.ndarray, s_cum: np.ndarray, s: float) -> tuple[float, float]:
    if s <= 0:
        return float(pts[0, 0]), float(pts[0, 1])
    if s >= s_cum[-1]:
        return float(pts[-1, 0]), float(pts[-1, 1])
    i = int(np.searchsorted(s_cum, s, side="right") - 1)
    i = max(0, min(i, len(pts) - 2))
    span = max(float(s_cum[i + 1] - s_cum[i]), 1e-6)
    u = (s - s_cum[i]) / span
    x = pts[i, 0] + u * (pts[i + 1, 0] - pts[i, 0])
    z = pts[i, 1] + u * (pts[i + 1, 1] - pts[i, 1])
    return float(x), float(z)


def _is_blocked_ahead(
    state: VehicleState,
    steer: float,
    v_tgt: float,
    model: BicycleModel,
    collision: CollisionChecker,
    dynamics: Sequence[DynamicObject],
    dt: float,
    use_prediction: bool,
    prefer_wait: bool,
) -> bool:
    if not dynamics:
        return False
    horizon = 1.6 if prefer_wait else 1.2
    steps = max(1, int(horizon / dt))
    s = state.copy()
    for _ in range(steps):
        accel = float(
            np.clip((v_tgt - s.speed) / dt, -model.params.max_decel, model.params.max_accel)
        )
        s = model.step(s, ControlInput(accel=accel, steer_cmd=steer), dt)
        if collision.collides(s, dynamics, t=s.t, use_prediction=use_prediction):
            return True
    return False
