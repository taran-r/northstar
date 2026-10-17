"""Time-indexed dynamic obstacle trajectories and motion prediction."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from northstar.facility.schema import DynamicObject, DynamicWaypoint, FacilitySpec, Zone
from northstar.geometry import point_in_polygon, wrap_angle


def _sample_in_zone(zone: Zone, rng: np.random.Generator) -> tuple[float, float]:
    xs = [p[0] for p in zone.polygon]
    zs = [p[1] for p in zone.polygon]
    for _ in range(40):
        x = float(rng.uniform(min(xs), max(xs)))
        z = float(rng.uniform(min(zs), max(zs)))
        if point_in_polygon(x, z, zone.polygon):
            return x, z
    return float(np.mean(xs)), float(np.mean(zs))


def _interpolate_route(
    waypoints_xz: list[tuple[float, float]],
    speeds: list[float],
    dt: float,
    t0: float = 0.0,
    hold_times: Optional[list[float]] = None,
) -> list[DynamicWaypoint]:
    """Build a dense ground-truth trajectory along a polyline with optional holds."""
    if len(waypoints_xz) < 2:
        x, z = waypoints_xz[0]
        return [DynamicWaypoint(t=t0, x=x, z=z, yaw=0.0, speed=0.0)]

    hold_times = hold_times or [0.0] * len(waypoints_xz)
    out: list[DynamicWaypoint] = []
    t = t0
    for i in range(len(waypoints_xz) - 1):
        x0, z0 = waypoints_xz[i]
        x1, z1 = waypoints_xz[i + 1]
        yaw = math.atan2(z1 - z0, x1 - x0)
        # Hold at waypoint
        hold = hold_times[i]
        th = 0.0
        while th < hold - 1e-9:
            out.append(DynamicWaypoint(t=t, x=x0, z=z0, yaw=yaw, speed=0.0))
            t += dt
            th += dt
        dist = math.hypot(x1 - x0, z1 - z0)
        speed = max(speeds[min(i, len(speeds) - 1)], 0.2)
        n = max(1, int(math.ceil(dist / (speed * dt))))
        for k in range(n):
            u = (k + 1) / n
            out.append(
                DynamicWaypoint(
                    t=t,
                    x=x0 + u * (x1 - x0),
                    z=z0 + u * (z1 - z0),
                    yaw=yaw,
                    speed=speed,
                )
            )
            t += dt
    return out


def constant_velocity_prediction(
    ground_truth: list[DynamicWaypoint],
    horizon: float = 8.0,
    dt: float = 0.2,
    noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> list[DynamicWaypoint]:
    """Predict future from initial velocity; used as NN / non-oracle planner input."""
    if not ground_truth:
        return []
    rng = rng or np.random.default_rng(0)
    w0 = ground_truth[0]
    # Estimate velocity from early GT
    if len(ground_truth) >= 2:
        w1 = ground_truth[min(3, len(ground_truth) - 1)]
        dt0 = max(w1.t - w0.t, 1e-3)
        vx = (w1.x - w0.x) / dt0
        vz = (w1.z - w0.z) / dt0
        yaw = math.atan2(vz, vx) if abs(vx) + abs(vz) > 1e-3 else w0.yaw
        speed = math.hypot(vx, vz)
    else:
        yaw, speed = w0.yaw, w0.speed
        vx, vz = speed * math.cos(yaw), speed * math.sin(yaw)

    pred: list[DynamicWaypoint] = []
    steps = int(horizon / dt) + 1
    for i in range(steps):
        t = w0.t + i * dt
        x = w0.x + vx * (t - w0.t)
        z = w0.z + vz * (t - w0.t)
        if noise_std > 0:
            x += float(rng.normal(0, noise_std))
            z += float(rng.normal(0, noise_std))
        pred.append(DynamicWaypoint(t=t, x=x, z=z, yaw=yaw, speed=speed))
    return pred


def route_conditioned_prediction(
    ground_truth: list[DynamicWaypoint],
    horizon: float = 8.0,
) -> list[DynamicWaypoint]:
    """Oracle-route prediction clipped to horizon (still not sudden intent changes)."""
    if not ground_truth:
        return []
    t_end = ground_truth[0].t + horizon
    return [w for w in ground_truth if w.t <= t_end + 1e-9]


DYNAMIC_TYPES = {
    "forklift": (2.4, 1.2, 2.2, 1.4),
    "agv": (1.2, 0.8, 1.0, 1.0),
    "industrial_cart": (1.5, 0.8, 1.2, 0.9),
    "worker": (0.5, 0.5, 1.7, 1.2),
    "crossing_vehicle": (2.6, 1.3, 1.6, 1.6),
}


def build_route_follower(
    facility: FacilitySpec,
    rng: np.random.Generator,
    obj_id: str,
    obj_type: str = "forklift",
    dt: float = 0.2,
    duration: float = 30.0,
    prediction_model: str = "constant_velocity",
    sudden_change: bool = False,
) -> DynamicObject:
    """Create a dynamic agent that follows corridor/intersection waypoints."""
    length, width, height, nom_speed = DYNAMIC_TYPES.get(
        obj_type, (1.5, 0.8, 1.5, 1.0)
    )
    corridors = [z for z in facility.zones if z.kind in {"corridor", "intersection"}]
    if len(corridors) < 2:
        # Fallback horizontal crossing
        route = [(5.0, 14.0), (55.0, 14.0)]
    else:
        n_wp = int(rng.integers(3, 6))
        route = []
        for _ in range(n_wp):
            zone = corridors[int(rng.integers(0, len(corridors)))]
            route.append(_sample_in_zone(zone, rng))

    speed = float(rng.uniform(0.6, nom_speed))
    speeds = [speed] * max(1, len(route) - 1)
    # Occasional stop at an intersection
    holds = [0.0] * len(route)
    if rng.random() < 0.4 and len(holds) > 1:
        holds[int(rng.integers(1, len(holds) - 1))] = float(rng.uniform(1.0, 3.5))

    gt = _interpolate_route(route, speeds, dt=dt, t0=0.0, hold_times=holds)
    # Trim / extend to duration
    if gt and gt[-1].t < duration:
        last = gt[-1]
        t = last.t
        while t < duration:
            t += dt
            gt.append(
                DynamicWaypoint(t=t, x=last.x, z=last.z, yaw=last.yaw, speed=0.0)
            )
    else:
        gt = [w for w in gt if w.t <= duration]

    if sudden_change and len(gt) > 10:
        # Mid-horizon intent change: reverse or divert (GT differs from CV prediction)
        mid = len(gt) // 2
        divert = [(gt[mid].x + float(rng.uniform(-4, 4)), gt[mid].z + float(rng.uniform(-4, 4)))]
        divert.append(
            (
                float(np.clip(divert[0][0] + rng.uniform(-8, 8), 2, 58)),
                float(np.clip(divert[0][1] + rng.uniform(-6, 6), 2, 38)),
            )
        )
        tail = _interpolate_route(
            [(gt[mid].x, gt[mid].z)] + divert,
            [speed * 0.8, speed * 0.8],
            dt=dt,
            t0=gt[mid].t,
        )
        gt = gt[:mid] + tail

    if prediction_model == "route_conditioned":
        pred = route_conditioned_prediction(gt, horizon=8.0)
    else:
        noise = 0.15 if sudden_change else 0.0
        pred = constant_velocity_prediction(gt, horizon=8.0, dt=dt, noise_std=noise, rng=rng)

    return DynamicObject(
        id=obj_id,
        type=obj_type,
        length=length,
        width=width,
        height=height,
        behavior="route_follow",
        ground_truth=gt,
        prediction=pred,
        prediction_model=prediction_model,
        metadata={"sudden_change": sudden_change},
    )


def generate_dynamic_obstacles(
    facility: FacilitySpec,
    rng: np.random.Generator,
    n_min: int = 1,
    n_max: int = 4,
    oracle_fraction: float = 0.25,
) -> tuple[list[DynamicObject], bool]:
    """
    Returns (obstacles, oracle_dynamics_flag).

    oracle_dynamics_flag is True if the expert planner should use ground-truth futures
    for this scenario (hard multi-agent timing). The dataset still stores CV predictions
    for the neural conditioning channels.
    """
    n = int(rng.integers(n_min, n_max + 1))
    types = list(DYNAMIC_TYPES.keys())
    use_oracle = bool(rng.random() < oracle_fraction)
    dyns: list[DynamicObject] = []
    for i in range(n):
        typ = types[int(rng.integers(0, len(types)))]
        pred_model = "constant_velocity"
        sudden = bool(rng.random() < 0.2)
        dyns.append(
            build_route_follower(
                facility,
                rng,
                obj_id=f"dyn_{i}_{typ}",
                obj_type=typ,
                prediction_model=pred_model,
                sudden_change=sudden,
            )
        )
    return dyns, use_oracle
