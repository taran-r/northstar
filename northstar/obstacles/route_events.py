"""Dynamic events placed on a planned route so the conflict is geometric, not accidental."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from northstar.facility.schema import DynamicObject, DynamicWaypoint, StaticObject
from northstar.obstacles.dynamic import DYNAMIC_TYPES, constant_velocity_prediction
from northstar.vehicle.bicycle import VehicleState


def _point(path: Sequence[VehicleState], index: int) -> VehicleState:
    index = max(0, min(len(path) - 1, index))
    return path[index]


def _index_for_fraction(path: Sequence[VehicleState], frac: float) -> int:
    return int(np.clip(round(frac * (len(path) - 1)), 1, len(path) - 2))


def _track(
    x0: float,
    z0: float,
    yaw: float,
    speed: float,
    duration: float,
    dt: float,
    t0: float = 0.0,
) -> list[DynamicWaypoint]:
    out: list[DynamicWaypoint] = []
    t = t0
    x, z = x0, z0
    while t <= duration + 1e-9:
        out.append(DynamicWaypoint(t=t, x=x, z=z, yaw=yaw, speed=speed))
        x += speed * math.cos(yaw) * dt
        z += speed * math.sin(yaw) * dt
        t += dt
    return out


def _agent(
    obj_id: str,
    obj_type: str,
    ground_truth: list[DynamicWaypoint],
    *,
    prediction_model: str,
    sudden_change: bool,
    rng: np.random.Generator,
) -> DynamicObject:
    length, width, height, _nom = DYNAMIC_TYPES.get(obj_type, (1.5, 0.8, 1.5, 1.0))
    if prediction_model == "constant_velocity":
        pred = constant_velocity_prediction(
            ground_truth,
            horizon=max(40.0, ground_truth[-1].t if ground_truth else 40.0),
            dt=0.2,
            noise_std=0.0,
            rng=rng,
        )
    else:
        pred = list(ground_truth)
    return DynamicObject(
        id=obj_id,
        type=obj_type,
        length=length,
        width=width,
        height=height,
        behavior="route_conflict",
        ground_truth=ground_truth,
        prediction=pred,
        prediction_model=prediction_model,
        metadata={"sudden_change": sudden_change, "placed_on_route": True},
    )


def crossing_agent(
    path: Sequence[VehicleState],
    rng: np.random.Generator,
    *,
    fraction: float | None = None,
    obj_type: str = "forklift",
    duration: float = 45.0,
    dt: float = 0.2,
    sudden_change: bool = False,
) -> DynamicObject:
    """Slow crosser timed to occupy a route point when the geometric plan is there.

    Constant-velocity prediction matches the crossing itself. An optional later
    turn changes only the ground truth after the conflict, so the encounter the
    rover reacts to is still predicted.
    """
    frac = float(fraction if fraction is not None else rng.uniform(0.32, 0.68))
    idx = _index_for_fraction(path, frac)
    point = _point(path, idx)
    t_hit = max(float(point.t), 1.5)
    speed = float(rng.uniform(1.05, 1.55))
    side = 1.0 if rng.random() < 0.5 else -1.0
    yaw = point.yaw + side * math.pi / 2
    x0 = point.x - math.cos(yaw) * speed * t_hit
    z0 = point.z - math.sin(yaw) * speed * t_hit
    gt = _track(x0, z0, yaw, speed, duration, dt)
    if sudden_change:
        turn_t = t_hit + float(rng.uniform(3.0, 5.0))
        yaw2 = yaw + float(rng.choice([-1.0, 1.0])) * math.pi / 2
        kept = [w for w in gt if w.t <= turn_t]
        if kept:
            tail = _track(kept[-1].x, kept[-1].z, yaw2, speed, duration, dt, t0=kept[-1].t + dt)
            gt = kept + tail
    return _agent(
        f"dyn_cross_{obj_type}",
        obj_type,
        gt,
        prediction_model="constant_velocity",
        sudden_change=sudden_change,
        rng=rng,
    )


def stopping_crosser(
    path: Sequence[VehicleState],
    rng: np.random.Generator,
    *,
    hold_s: float = 3.0,
    duration: float = 45.0,
    dt: float = 0.2,
) -> DynamicObject:
    """Crosser whose ground truth stops in the lane. The CV prediction does not.

    The expert may use the ground truth (oracle). The stored prediction does not
    include that stop, and the scenario must be flagged oracle_dynamics.
    """
    idx = _index_for_fraction(path, float(rng.uniform(0.34, 0.6)))
    point = _point(path, idx)
    t_hit = max(float(point.t), 1.5)
    speed = float(rng.uniform(1.05, 1.45))
    side = 1.0 if rng.random() < 0.5 else -1.0
    yaw = point.yaw + side * math.pi / 2
    x0 = point.x - math.cos(yaw) * speed * t_hit
    z0 = point.z - math.sin(yaw) * speed * t_hit
    moving = _track(x0, z0, yaw, speed, duration, dt)
    before = [w for w in moving if w.t <= t_hit + 1e-9]
    hold: list[DynamicWaypoint] = []
    t = t_hit
    while t < t_hit + hold_s - 1e-9:
        t += dt
        hold.append(DynamicWaypoint(t=t, x=point.x, z=point.z, yaw=yaw, speed=0.0))
    after = _track(point.x, point.z, yaw, speed, duration, dt, t0=(hold[-1].t + dt if hold else t_hit))
    gt = before + hold + after
    length, width, height, _nom = DYNAMIC_TYPES["forklift"]
    pred = constant_velocity_prediction(moving, horizon=duration, dt=dt, noise_std=0.0, rng=rng)
    return DynamicObject(
        id="dyn_oracle_stop_forklift",
        type="forklift",
        length=length,
        width=width,
        height=height,
        behavior="route_conflict",
        ground_truth=gt,
        prediction=pred,
        prediction_model="constant_velocity",
        metadata={"sudden_change": True, "placed_on_route": True, "hidden_stop": True},
    )


def oncoming_agent(
    path: Sequence[VehicleState],
    rng: np.random.Generator,
    *,
    fraction: float | None = None,
    duration: float = 45.0,
    dt: float = 0.2,
) -> DynamicObject:
    """Vehicle approaching along the route from ahead. CV prediction matches GT."""
    frac = float(fraction if fraction is not None else rng.uniform(0.28, 0.48))
    idx = _index_for_fraction(path, frac)
    point = _point(path, idx)
    yaw = point.yaw + math.pi
    speed = float(rng.uniform(1.2, 1.7))
    # Start ahead and drive toward the rover; at t=0 it is already on the route.
    gt = _track(point.x, point.z, yaw, speed, duration, dt)
    return _agent(
        "dyn_oncoming_crossing_vehicle",
        "crossing_vehicle",
        gt,
        prediction_model="constant_velocity",
        sudden_change=False,
        rng=rng,
    )


def lead_then_exit_agent(
    path: Sequence[VehicleState],
    rng: np.random.Generator,
    *,
    duration: float = 45.0,
    dt: float = 0.2,
) -> DynamicObject:
    """Slower cart ahead that leaves the lane. Prediction follows that visible exit.

    The exit is part of the published prediction (not a hidden ground-truth-only
    turn), so the expert and the network see the same lane-clearing motion.
    """
    idx = _index_for_fraction(path, float(rng.uniform(0.22, 0.4)))
    point = _point(path, idx)
    speed = float(rng.uniform(0.45, 0.75))
    along = _track(point.x, point.z, point.yaw, speed, 4.0, dt)
    exit_yaw = point.yaw + float(rng.choice([-1.0, 1.0])) * (math.pi / 2)
    last = along[-1]
    tail = _track(last.x, last.z, exit_yaw, speed, duration, dt, t0=last.t + dt)
    gt = along + tail
    return _agent(
        "dyn_lead_industrial_cart",
        "industrial_cart",
        gt,
        prediction_model="route_conditioned",
        sudden_change=False,
        rng=rng,
    )


def barrier_on_path(path: Sequence[VehicleState], rng: np.random.Generator) -> StaticObject:
    """Temporary barrier across the geometric route, forcing a later replan."""
    idx = _index_for_fraction(path, float(rng.uniform(0.35, 0.62)))
    point = _point(path, idx)
    return StaticObject(
        id="route_block_barrier",
        type="barrier",
        x=point.x,
        z=point.z,
        yaw=point.yaw + math.pi / 2,
        length=2.6,
        width=0.5,
        height=1.2,
        permanent=False,
        metadata={"blocks_route": True},
    )
