"""Configurable kinematic bicycle model for a four-wheel ground rover.

Reusable by the expert planner, neural trajectory follower, and future CBF controller.
Ground-plane navigation on XZ; Y is preserved for rendering.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from northstar.geometry import OrientedBox, wrap_angle


@dataclass
class VehicleParams:
    length: float = 1.3
    width: float = 0.8
    wheelbase: float = 1.0
    max_speed: float = 2.5
    min_speed: float = 0.0  # forward-only for expert demos
    max_steer: float = 0.55
    max_accel: float = 1.5
    max_decel: float = 2.5
    max_steer_rate: float = 1.2
    # collision footprint slightly larger than body for safety margin
    footprint_margin: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VehicleParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class VehicleState:
    x: float
    z: float
    yaw: float
    speed: float = 0.0
    steering: float = 0.0
    y: float = 0.0
    t: float = 0.0

    def copy(self) -> "VehicleState":
        return VehicleState(
            x=self.x,
            z=self.z,
            yaw=self.yaw,
            speed=self.speed,
            steering=self.steering,
            y=self.y,
            t=self.t,
        )

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x, self.z, self.yaw, self.speed, self.steering], dtype=np.float64
        )

    def footprint(self, params: VehicleParams) -> OrientedBox:
        return OrientedBox(
            self.x,
            self.z,
            self.yaw,
            params.length + 2 * params.footprint_margin,
            params.width + 2 * params.footprint_margin,
        )


@dataclass
class ControlInput:
    accel: float
    steer_rate: float = 0.0
    # Alternatively set absolute steering target; rate limit is applied in step()
    steer_cmd: Optional[float] = None


class BicycleModel:
    """Continuous-time kinematic bicycle with accel / steer-rate limits."""

    def __init__(self, params: Optional[VehicleParams] = None):
        self.params = params or VehicleParams()

    def clamp_controls(self, u: ControlInput, state: VehicleState) -> ControlInput:
        p = self.params
        accel = float(np.clip(u.accel, -p.max_decel, p.max_accel))
        if u.steer_cmd is not None:
            target = float(np.clip(u.steer_cmd, -p.max_steer, p.max_steer))
            err = target - state.steering
            steer_rate = float(np.clip(err / 1e-3, -p.max_steer_rate, p.max_steer_rate))
            # Prefer finite rate when integrating with dt
            steer_rate = float(np.clip(target - state.steering, -p.max_steer_rate, p.max_steer_rate))
            # The actual rate will be (target-steer)/dt clipped; store desired rate as clipped delta intent
            steer_rate = float(
                np.clip(
                    (target - state.steering),
                    -p.max_steer_rate,
                    p.max_steer_rate,
                )
            )
            # Reinterpret: caller should pass steer_rate via step with dt-aware clipping
            return ControlInput(accel=accel, steer_rate=0.0, steer_cmd=target)
        steer_rate = float(np.clip(u.steer_rate, -p.max_steer_rate, p.max_steer_rate))
        return ControlInput(accel=accel, steer_rate=steer_rate, steer_cmd=None)

    def step(self, state: VehicleState, u: ControlInput, dt: float) -> VehicleState:
        p = self.params
        if dt <= 0:
            return state.copy()

        accel = float(np.clip(u.accel, -p.max_decel, p.max_accel))
        if u.steer_cmd is not None:
            target = float(np.clip(u.steer_cmd, -p.max_steer, p.max_steer))
            max_delta = p.max_steer_rate * dt
            steering = state.steering + float(
                np.clip(target - state.steering, -max_delta, max_delta)
            )
        else:
            steer_rate = float(np.clip(u.steer_rate, -p.max_steer_rate, p.max_steer_rate))
            steering = float(
                np.clip(state.steering + steer_rate * dt, -p.max_steer, p.max_steer)
            )

        speed = float(np.clip(state.speed + accel * dt, p.min_speed, p.max_speed))
        # Average speed for integration
        v_avg = 0.5 * (state.speed + speed)
        yaw_rate = (v_avg / p.wheelbase) * math.tan(steering) if abs(steering) > 1e-6 else 0.0
        yaw = wrap_angle(state.yaw + yaw_rate * dt)
        x = state.x + v_avg * math.cos(yaw) * dt
        # Using mid-yaw is slightly better; re-integrate with average yaw
        yaw_mid = wrap_angle(state.yaw + 0.5 * yaw_rate * dt)
        x = state.x + v_avg * math.cos(yaw_mid) * dt
        z = state.z + v_avg * math.sin(yaw_mid) * dt

        return VehicleState(
            x=x,
            z=z,
            yaw=yaw,
            speed=speed,
            steering=steering,
            y=state.y,
            t=state.t + dt,
        )

    def simulate(
        self,
        state: VehicleState,
        controls: list[ControlInput],
        dt: float,
    ) -> list[VehicleState]:
        traj = [state.copy()]
        s = state
        for u in controls:
            s = self.step(s, u, dt)
            traj.append(s)
        return traj

    def min_turning_radius(self) -> float:
        return self.params.wheelbase / math.tan(self.params.max_steer)

    def curvature(self, steering: float) -> float:
        return math.tan(steering) / self.params.wheelbase
