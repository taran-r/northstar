"""Obstacle scene viewed by the safety filter.

The filter consumes positions and predicted velocities. It does not read future
ground-truth waypoints. ``build_scene(..., dynamics_mode='causal_cv')`` estimates
each agent's current pose and velocity from samples at or before the current
time, then extrapolates that velocity. ``snapshot_gt`` places agents at their
true current pose for scoring and does not forecast them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from northstar.collision import wall_as_box
from northstar.facility.occupancy import sample_dynamic_pose
from northstar.facility.schema import DynamicObject, FacilitySpec, StaticObject


@dataclass
class BoxObstacle:
    """Oriented rectangle on the XZ plane, optionally moving at constant velocity."""

    id: str
    x: float
    z: float
    yaw: float
    length: float
    width: float
    vx: float = 0.0
    vz: float = 0.0
    t_ref: float = 0.0
    kind: str = "box"

    def pose_at(self, t: float) -> tuple[float, float, float]:
        dt = t - self.t_ref
        return self.x + self.vx * dt, self.z + self.vz * dt, self.yaw


@dataclass
class CircleObstacle:
    """Vertical column approximated as a circle on the XZ plane."""

    id: str
    x: float
    z: float
    radius: float
    vx: float = 0.0
    vz: float = 0.0
    t_ref: float = 0.0
    kind: str = "circle"

    def center_at(self, t: float) -> tuple[float, float]:
        dt = t - self.t_ref
        return self.x + self.vx * dt, self.z + self.vz * dt


@dataclass
class SafetyScene:
    """Static geometry plus constant-velocity obstacle forecasts at ``t``."""

    t: float
    boxes: list[BoxObstacle] = field(default_factory=list)
    circles: list[CircleObstacle] = field(default_factory=list)

    def obstacles(self) -> list[BoxObstacle | CircleObstacle]:
        return [*self.boxes, *self.circles]


def _interp_pose(
    dyn: DynamicObject, t: float, use_prediction: bool
) -> Optional[tuple[float, float, float]]:
    return sample_dynamic_pose(dyn, t, use_prediction=use_prediction)


def causal_kinematics(
    dyn: DynamicObject,
    t: float,
    *,
    dt_obs: float = 0.2,
    allow_prediction_fallback: bool = True,
) -> Optional[tuple[float, float, float, float, float]]:
    """Current pose and a causal velocity estimate.

    Returns ``(x, z, yaw, vx, vz)`` or ``None`` when the agent has no samples.
    Ground truth is the sensed current motion. The stored ``prediction`` list is
    used only when ground truth is missing. After the first sample, velocity is a
    backward difference, so a later change of direction is not anticipated.
    """
    use_prediction = False
    traj = dyn.ground_truth
    if not traj and allow_prediction_fallback:
        traj = dyn.prediction
        use_prediction = True
    if not traj:
        return None

    if t <= traj[0].t + 1e-9:
        w0 = traj[0]
        if len(traj) == 1 or traj[1].t <= w0.t + 1e-9:
            vx = w0.speed * math.cos(w0.yaw)
            vz = w0.speed * math.sin(w0.yaw)
        else:
            w1 = traj[1]
            dt = max(w1.t - w0.t, 1e-3)
            vx = (w1.x - w0.x) / dt
            vz = (w1.z - w0.z) / dt
        return w0.x, w0.z, w0.yaw, vx, vz

    pose = _interp_pose(dyn, t, use_prediction)
    if pose is None:
        return None
    t_prev = max(traj[0].t, t - dt_obs)
    prev = _interp_pose(dyn, t_prev, use_prediction)
    if prev is None:
        return pose[0], pose[1], pose[2], 0.0, 0.0
    dt = max(t - t_prev, 1e-3)
    vx = (pose[0] - prev[0]) / dt
    vz = (pose[1] - prev[1]) / dt
    return pose[0], pose[1], pose[2], vx, vz


def snapshot_pose(
    dyn: DynamicObject, t: float
) -> Optional[tuple[float, float, float]]:
    """True pose at ``t`` for scoring. Does not forecast."""
    pose = sample_dynamic_pose(dyn, t, use_prediction=False)
    if pose is None:
        pose = sample_dynamic_pose(dyn, t, use_prediction=True)
    return pose


def build_scene(
    t: float,
    *,
    boxes: Optional[Sequence[BoxObstacle]] = None,
    circles: Optional[Sequence[CircleObstacle]] = None,
    facility: Optional[FacilitySpec] = None,
    static_obstacles: Optional[Sequence[StaticObject]] = None,
    dynamic_obstacles: Optional[Sequence[DynamicObject]] = None,
    dynamics_mode: str = "causal_cv",
) -> SafetyScene:
    """Assemble the geometry the filter or the scorer is allowed to see.

    ``dynamics_mode='causal_cv'`` forecasts constant velocity from the current
    sensed motion. ``dynamics_mode='snapshot_gt'`` pins each agent to its true
    pose at ``t`` with zero velocity so clearance is the instantaneous gap.
    """
    if dynamics_mode not in {"causal_cv", "snapshot_gt"}:
        raise ValueError(f"unknown dynamics_mode {dynamics_mode!r}")

    out_boxes: list[BoxObstacle] = []
    out_circles: list[CircleObstacle] = []

    for box in boxes or []:
        out_boxes.append(
            BoxObstacle(
                id=box.id,
                x=box.x,
                z=box.z,
                yaw=box.yaw,
                length=box.length,
                width=box.width,
                vx=box.vx,
                vz=box.vz,
                t_ref=box.t_ref,
                kind=box.kind,
            )
        )
    for circ in circles or []:
        out_circles.append(
            CircleObstacle(
                id=circ.id,
                x=circ.x,
                z=circ.z,
                radius=circ.radius,
                vx=circ.vx,
                vz=circ.vz,
                t_ref=circ.t_ref,
                kind=circ.kind,
            )
        )

    if facility is not None:
        for wall in facility.walls:
            wb = wall_as_box(wall)
            out_boxes.append(
                BoxObstacle(
                    id=wall.id,
                    x=wb.x,
                    z=wb.z,
                    yaw=wb.yaw,
                    length=wb.length,
                    width=wb.width,
                    t_ref=t,
                    kind="wall",
                )
            )
        for obj in facility.permanent_objects:
            out_boxes.append(
                BoxObstacle(
                    id=obj.id,
                    x=obj.x,
                    z=obj.z,
                    yaw=obj.yaw,
                    length=obj.length,
                    width=obj.width,
                    t_ref=t,
                    kind="permanent",
                )
            )
        for col in facility.columns:
            out_circles.append(
                CircleObstacle(
                    id=col.id,
                    x=col.x,
                    z=col.z,
                    radius=col.radius,
                    t_ref=t,
                    kind="column",
                )
            )

    for obj in static_obstacles or []:
        out_boxes.append(
            BoxObstacle(
                id=obj.id,
                x=obj.x,
                z=obj.z,
                yaw=obj.yaw,
                length=obj.length,
                width=obj.width,
                t_ref=t,
                kind=obj.type or "static",
            )
        )

    for dyn in dynamic_obstacles or []:
        if dynamics_mode == "snapshot_gt":
            pose = snapshot_pose(dyn, t)
            if pose is None:
                continue
            x, z, yaw = pose
            vx, vz = 0.0, 0.0
        else:
            kin = causal_kinematics(dyn, t)
            if kin is None:
                continue
            x, z, yaw, vx, vz = kin
        out_boxes.append(
            BoxObstacle(
                id=dyn.id,
                x=x,
                z=z,
                yaw=yaw,
                length=dyn.length,
                width=dyn.width,
                vx=vx,
                vz=vz,
                t_ref=t,
                kind=dyn.type or "dynamic",
            )
        )

    return SafetyScene(t=t, boxes=out_boxes, circles=out_circles)
