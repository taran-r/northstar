"""Trajectory serialization formats for expert demos and NN windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from northstar.vehicle.bicycle import VehicleState


# Fixed local horizon defaults (configurable in pipeline)
DEFAULT_HORIZON_STEPS = 32
DEFAULT_DT = 0.2  # => 6.4 s


@dataclass
class TrajectoryRecord:
    """Full feasible route for one expert solution of a scenario."""

    trajectory_id: str
    scenario_id: str
    dt: float
    states: list[VehicleState]
    # Per-step accel commands aligned with states[1:] (last repeated)
    accel_cmds: list[float]
    mode_id: int = 0  # multi-modal expert index
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "scenario_id": self.scenario_id,
            "dt": self.dt,
            "mode_id": self.mode_id,
            "metadata": self.metadata,
            "states": [
                {
                    "t": s.t,
                    "x": s.x,
                    "z": s.z,
                    "y": s.y,
                    "yaw": s.yaw,
                    "speed": s.speed,
                    "steering": s.steering,
                    "accel": self.accel_cmds[min(i, len(self.accel_cmds) - 1)]
                    if self.accel_cmds
                    else 0.0,
                }
                for i, s in enumerate(self.states)
            ],
        }


@dataclass
class TrajectoryWindow:
    """
    One training sample: H future states at fixed dt, plus conditioning.

    Conditioning fields are stored as numpy-friendly arrays in `cond`.
    """

    sample_id: str
    scenario_id: str
    trajectory_id: str
    horizon: int
    dt: float
    # Initial rover state (conditioning)
    ego0: dict[str, float]
    # Future H x features: x, z, yaw, speed, steering, accel, t_rel
    future: np.ndarray
    # Local subgoal (may differ from global goal if beyond horizon)
    subgoal: dict[str, float]
    global_goal: dict[str, float]
    # Conditioning tensors
    static_occ: np.ndarray  # [H_map, W_map]
    dynamic_occ: np.ndarray  # [T_pred, H_map, W_map] predicted occupancy
    facility_occ: np.ndarray  # permanent only
    oracle_dynamics: bool
    behavior_tag: str
    mode_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_npz(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "scenario_id": self.scenario_id,
            "trajectory_id": self.trajectory_id,
            "horizon": self.horizon,
            "dt": self.dt,
            "ego0": self.ego0,
            "future": self.future.astype(np.float32),
            "subgoal": self.subgoal,
            "global_goal": self.global_goal,
            "static_occ": self.static_occ.astype(np.float32),
            "dynamic_occ": self.dynamic_occ.astype(np.float32),
            "facility_occ": self.facility_occ.astype(np.float32),
            "oracle_dynamics": self.oracle_dynamics,
            "behavior_tag": self.behavior_tag,
            "mode_id": self.mode_id,
            "metadata": self.metadata,
        }


def states_to_array(states: list[VehicleState], accel_cmds: list[float]) -> np.ndarray:
    """Nx7: x, z, yaw, speed, steering, accel, t."""
    rows = []
    for i, s in enumerate(states):
        a = accel_cmds[min(i, len(accel_cmds) - 1)] if accel_cmds else 0.0
        rows.append([s.x, s.z, s.yaw, s.speed, s.steering, a, s.t])
    return np.asarray(rows, dtype=np.float32)


def resample_states(
    states: list[VehicleState],
    accel_cmds: list[float],
    dt: float,
) -> tuple[list[VehicleState], list[float]]:
    """Linearly resample a trajectory onto a uniform time grid."""
    if len(states) < 2:
        return states, accel_cmds
    t0, t1 = states[0].t, states[-1].t
    if t1 <= t0:
        return states, accel_cmds
    times = np.arange(t0, t1 + 0.5 * dt, dt)
    xs = np.array([s.x for s in states])
    zs = np.array([s.z for s in states])
    yaws = np.array([s.yaw for s in states])
    speeds = np.array([s.speed for s in states])
    steers = np.array([s.steering for s in states])
    ts = np.array([s.t for s in states])
    acc = np.array(
        [accel_cmds[min(i, len(accel_cmds) - 1)] if accel_cmds else 0.0 for i in range(len(states))]
    )

    def lerp_yaw(t: float) -> float:
        i = int(np.searchsorted(ts, t, side="right") - 1)
        i = max(0, min(i, len(ts) - 2))
        u = 0.0 if ts[i + 1] <= ts[i] else (t - ts[i]) / (ts[i + 1] - ts[i])
        dy = (yaws[i + 1] - yaws[i] + np.pi) % (2 * np.pi) - np.pi
        return float(yaws[i] + u * dy)

    out_s: list[VehicleState] = []
    out_a: list[float] = []
    for t in times:
        out_s.append(
            VehicleState(
                x=float(np.interp(t, ts, xs)),
                z=float(np.interp(t, ts, zs)),
                yaw=lerp_yaw(float(t)),
                speed=float(np.interp(t, ts, speeds)),
                steering=float(np.interp(t, ts, steers)),
                t=float(t),
            )
        )
        out_a.append(float(np.interp(t, ts, acc)))
    return out_s, out_a
