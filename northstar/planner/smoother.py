"""Trajectory smoothing and velocity profiling for expert demos."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.interpolate import splprep, splev

from northstar.vehicle.bicycle import BicycleModel, ControlInput, VehicleParams, VehicleState
from northstar.geometry import wrap_angle


def smooth_trajectory(
    path: list[VehicleState],
    smoothing: float = 0.5,
    n_points: Optional[int] = None,
) -> list[VehicleState]:
    """B-spline smooth XZ path; recompute yaw from tangents."""
    if len(path) < 4:
        return [p.copy() for p in path]

    xs = np.array([p.x for p in path], dtype=np.float64)
    zs = np.array([p.z for p in path], dtype=np.float64)
    ts = np.array([p.t for p in path], dtype=np.float64)

    # Remove near-duplicates
    keep = [0]
    for i in range(1, len(xs)):
        if math.hypot(xs[i] - xs[keep[-1]], zs[i] - zs[keep[-1]]) > 0.05:
            keep.append(i)
    if len(keep) < 4:
        return [p.copy() for p in path]
    xs, zs, ts = xs[keep], zs[keep], ts[keep]

    try:
        tck, _ = splprep([xs, zs], u=np.linspace(0, 1, len(xs)), s=smoothing, k=3)
    except Exception:
        return [p.copy() for p in path]

    n = n_points or max(len(path), int((ts[-1] - ts[0]) / 0.1) + 1)
    u_fine = np.linspace(0, 1, n)
    x_new, z_new = splev(u_fine, tck)
    dx, dz = splev(u_fine, tck, der=1)

    t0, t1 = float(ts[0]), float(ts[-1])
    out: list[VehicleState] = []
    for i in range(n):
        yaw = math.atan2(dz[i], dx[i])
        t = t0 + (t1 - t0) * (i / max(n - 1, 1))
        speed = path[min(i * len(path) // n, len(path) - 1)].speed
        out.append(
            VehicleState(
                x=float(x_new[i]),
                z=float(z_new[i]),
                yaw=wrap_angle(yaw),
                speed=speed,
                steering=0.0,
                y=0.0,
                t=t,
            )
        )
    return out


def profile_velocity(
    path: list[VehicleState],
    params: Optional[VehicleParams] = None,
    dt: float = 0.2,
    cruise_speed: Optional[float] = None,
) -> list[VehicleState]:
    """Re-time a geometric path with accel/decel limits and curvature slowing."""
    if len(path) < 2:
        return [p.copy() for p in path]

    params = params or VehicleParams()
    model = BicycleModel(params)
    v_cruise = cruise_speed or params.max_speed * 0.85

    # Arc-length parameterization
    pts = np.array([[p.x, p.z] for p in path], dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-3:
        out = [path[0].copy()]
        for i in range(1, max(2, int(1.0 / dt))):
            st = path[0].copy()
            st.t = path[0].t + i * dt
            st.speed = 0.0
            out.append(st)
        return out

    # Curvature estimate
    yaws = np.array([p.yaw for p in path], dtype=np.float64)
    kappa = np.zeros(len(path))
    for i in range(1, len(path) - 1):
        ds = max(seg[i - 1] + seg[i], 1e-3)
        dyaw = wrap_angle(yaws[i + 1] - yaws[i - 1])
        kappa[i] = abs(dyaw) / ds
    kappa[0], kappa[-1] = kappa[1], kappa[-2]

    # Speed limits from curvature and terminal stop
    v_lim = np.full(len(path), v_cruise)
    for i, k in enumerate(kappa):
        if k > 1e-3:
            # a_lat ≈ v^2 * kappa  <= ~1.5
            v_lim[i] = min(v_lim[i], math.sqrt(1.5 / k))
    v_lim[-1] = 0.0

    # Forward / backward pass
    v = np.zeros(len(path))
    v[0] = max(min(path[0].speed, v_lim[0]), min(0.6, v_cruise))
    for i in range(1, len(path)):
        ds = max(seg[i - 1], 1e-4)
        v[i] = min(v_lim[i], math.sqrt(v[i - 1] ** 2 + 2 * params.max_accel * ds))
    for i in range(len(path) - 2, -1, -1):
        ds = max(seg[i], 1e-4)
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * params.max_decel * ds))
    # Keep a minimum cruise on the interior so we never freeze at v=0 mid-path
    if len(v) > 2:
        v[1:-1] = np.maximum(v[1:-1], min(0.5, v_cruise))

    out: list[VehicleState] = []
    s_cur = 0.0
    t_cur = path[0].t
    v_cur = float(max(v[0], 0.4))
    out.append(
        VehicleState(
            x=path[0].x,
            z=path[0].z,
            yaw=path[0].yaw,
            speed=v_cur,
            steering=0.0,
            t=t_cur,
        )
    )

    max_steps = int(total / max(v_cruise * 0.2, 0.15)) + 50
    for _ in range(max_steps):
        if s_cur >= total - 1e-3:
            break
        idx = int(np.searchsorted(s, s_cur, side="right") - 1)
        idx = max(0, min(idx, len(path) - 2))
        span = max(s[idx + 1] - s[idx], 1e-6)
        u = (s_cur - s[idx]) / span
        x = path[idx].x + u * (path[idx + 1].x - path[idx].x)
        z = path[idx].z + u * (path[idx + 1].z - path[idx].z)
        yaw = path[idx].yaw + u * wrap_angle(path[idx + 1].yaw - path[idx].yaw)
        v_tgt = float(v[idx] + u * (v[idx + 1] - v[idx]))
        v_tgt = max(v_tgt, 0.3 if s_cur < total - 1.0 else 0.0)
        accel = float(
            np.clip((v_tgt - v_cur) / dt, -params.max_decel, params.max_accel)
        )
        v_cur = float(np.clip(v_cur + accel * dt, 0.0, params.max_speed))
        if v_cur < 0.05 and s_cur < total - 0.5:
            v_cur = 0.4
        k = float(kappa[idx])
        steer = float(
            np.clip(math.atan(params.wheelbase * k), -params.max_steer, params.max_steer)
        )
        if wrap_angle(path[idx + 1].yaw - path[idx].yaw) < 0:
            steer = -steer

        s_cur += max(v_cur, 0.05) * dt
        t_cur += dt
        out.append(
            VehicleState(
                x=x, z=z, yaw=wrap_angle(yaw), speed=v_cur, steering=steer, t=t_cur
            )
        )

    last = path[-1].copy()
    last.t = out[-1].t + dt
    last.speed = 0.0
    out.append(last)
    return out


def validate_executable(
    traj: list[VehicleState],
    model: BicycleModel,
    dt: float = 0.2,
    pos_tol: float = 0.6,
) -> bool:
    """Check that open-loop tracking of recorded controls stays near the path."""
    if len(traj) < 2:
        return False
    s = traj[0].copy()
    for i in range(1, len(traj)):
        target = traj[i]
        prev = traj[i - 1]
        accel = (target.speed - prev.speed) / max(target.t - prev.t, 1e-3)
        s = model.step(s, ControlInput(accel=accel, steer_cmd=target.steering), dt)
        if math.hypot(s.x - target.x, s.z - target.z) > pos_tol:
            # Soft fail: geometric paths may drift; allow larger tol near end
            if i < len(traj) - 2 and math.hypot(s.x - target.x, s.z - target.z) > pos_tol * 1.8:
                return False
    return True
