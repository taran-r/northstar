"""Obstacle-unaware trajectory tracker.

The nominal command is pure pursuit on a polyline plus a speed profile that
slows for the remaining path length. It does not look at obstacles. The safety
filter is what rejects commands that the backup rollout cannot certify.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from northstar.geometry import wrap_angle
from northstar.vehicle.bicycle import ControlInput, VehicleParams, VehicleState


def polyline_from_points(points: Sequence[Sequence[float]], spacing: float = 0.5) -> np.ndarray:
    """Resample an XZ polyline at approximately ``spacing`` meters."""
    raw = np.asarray(points, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] < 2:
        raise ValueError("path needs at least one [x, z] point")
    if raw.shape[0] == 1:
        return raw[:, :2].copy()
    out = [raw[0, :2].copy()]
    for i in range(1, len(raw)):
        a = raw[i - 1, :2]
        b = raw[i, :2]
        dist = float(np.linalg.norm(b - a))
        if dist < 1e-6:
            continue
        steps = max(1, int(math.ceil(dist / max(spacing, 1e-3))))
        for k in range(1, steps + 1):
            out.append(a + (k / steps) * (b - a))
    return np.vstack(out)


class TrajectoryTracker:
    """Pure-pursuit steering and a kinematic speed target along a fixed path."""

    def __init__(
        self,
        waypoints: Sequence[Sequence[float]],
        params: Optional[VehicleParams] = None,
        cruise_speed: float = 1.6,
        lookahead: float = 1.5,
        steer_gain: float = 1.7,
        turn_speed: float = 0.8,
        turn_slow_angle: float = 0.7,
    ):
        self.params = params or VehicleParams()
        self.points = polyline_from_points(waypoints)
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1) if len(self.points) > 1 else np.zeros(0)
        self.s_cum = np.concatenate([[0.0], np.cumsum(seg)]) if len(self.points) > 1 else np.zeros(1)
        self.total = float(self.s_cum[-1])
        self.cruise_speed = cruise_speed
        self.lookahead = lookahead
        self.steer_gain = steer_gain
        self.turn_speed = turn_speed
        self.turn_slow_angle = turn_slow_angle

    def closest_s(self, state: VehicleState) -> float:
        if len(self.points) == 1:
            return 0.0
        d2 = (self.points[:, 0] - state.x) ** 2 + (self.points[:, 1] - state.z) ** 2
        return float(self.s_cum[int(np.argmin(d2))])

    def point_at(self, s_along: float) -> tuple[float, float]:
        if s_along <= 0.0:
            return float(self.points[0, 0]), float(self.points[0, 1])
        if s_along >= self.total:
            return float(self.points[-1, 0]), float(self.points[-1, 1])
        i = int(np.searchsorted(self.s_cum, s_along, side="right") - 1)
        i = max(0, min(i, len(self.points) - 2))
        span = max(float(self.s_cum[i + 1] - self.s_cum[i]), 1e-6)
        u = (s_along - self.s_cum[i]) / span
        x = self.points[i, 0] + u * (self.points[i + 1, 0] - self.points[i, 0])
        z = self.points[i, 1] + u * (self.points[i + 1, 1] - self.points[i, 1])
        return float(x), float(z)

    def nominal_control(self, state: VehicleState, dt: float) -> ControlInput:
        params = self.params
        if self.total < 1e-3:
            return ControlInput(accel=-params.max_decel, steer_cmd=0.0)
        s_along = self.closest_s(state)
        remain = max(0.0, self.total - s_along)
        tx, tz = self.point_at(min(self.total, s_along + self.lookahead))
        desired = math.atan2(tz - state.z, tx - state.x)
        yaw_err = wrap_angle(desired - state.yaw)
        steer = float(np.clip(self.steer_gain * yaw_err, -params.max_steer, params.max_steer))
        v_tgt = min(self.cruise_speed, math.sqrt(max(0.0, 2.0 * params.max_decel * remain)))
        if abs(yaw_err) > self.turn_slow_angle:
            v_tgt = min(v_tgt, self.turn_speed)
        if remain < 0.35:
            v_tgt = 0.0
        accel = (v_tgt - state.speed) / max(dt, 1e-3)
        accel = float(np.clip(accel, -params.max_decel, params.max_accel))
        return ControlInput(accel=accel, steer_cmd=steer)
