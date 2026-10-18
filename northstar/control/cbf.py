"""Discrete-time predictive safety filter for the kinematic bicycle.

The continuous-time clearance of a kinematic bicycle has relative degree 2 in
acceleration and steering rate: those inputs appear in the second derivative of
position, not the first. A threshold on the current distance is not a CBF for
this model. This module instead certifies a command with a backup controller
(maximum braking, steering held) and, when the gap is already small, a one-step
discrete-time CBF on clearance. The certified command is the actuator-feasible
point closest to the nominal command in a weighted squared distance. That
projection is a 2-variable quadratic program.

If no command keeps the backup clearance above the margin, the filter applies
maximum braking, reports the problem infeasible, and does not claim the rover
can still avoid contact. A persistent block along the forward heading sets
``replan_required`` so a planner can replace the route. This filter is local.

See ``docs/CBF_SAFETY.md`` for the constraint derivation and the assumptions.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from northstar.control.clearance import box_circle_clearance, oriented_box_clearance
from northstar.control.qp import solve_weighted_qp_2d
from northstar.control.scene import BoxObstacle, CircleObstacle, SafetyScene
from northstar.coords import dataset_to_threejs
from northstar.geometry import OrientedBox
from northstar.vehicle.bicycle import BicycleModel, ControlInput, VehicleParams, VehicleState


@dataclass
class CBFConfig:
    """Tunable parameters of the predictive filter. Distances are meters."""

    safety_margin: float = 0.12
    preview_time: float = 0.7
    idle_time: float = 1.5
    activation_clearance: float = 2.0
    gamma: float = 0.4
    violation_tol: float = 0.03
    weight_accel: float = 1.0
    weight_steer: float = 25.0
    fd_accel: float = 0.25
    fd_steer: float = 0.06
    use_one_step_dcbf: bool = True
    replan_horizon: float = 4.0
    replan_distances: tuple[float, ...] = (1.2, 2.2, 3.4, 4.8)
    nearby_extra: float = 2.0


@dataclass
class CBFDiagnostics:
    """Intervention record for logs and the Three.js frontend.

    Positions in ``pose_threejs`` use ``dataset_to_threejs``. Dataset yaw is a
    right-handed rotation about +Y from +X toward +Z. Three.js mirrors Z, so a
    heading rendered with the same right-handed Y rotation is ``-yaw``.
    """

    intervened: bool
    cbf_active: bool
    infeasible: bool
    fallback: str
    replan_required: bool
    limit_clamped: bool
    nominal_accel: float
    nominal_steer_cmd: float
    safe_accel: float
    safe_steer_cmd: float
    accel_correction: float
    steer_correction: float
    min_clearance_m: float
    min_backup_clearance_m: float
    nominal_backup_clearance_m: float
    closing_speed_mps: float
    nearest_obstacle_id: str
    active_obstacle_ids: list[str] = field(default_factory=list)
    solve_time_s: float = 0.0
    solver_status: str = "nominal_safe"
    pose_threejs: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_dataset: float = 0.0
    yaw_threejs: float = 0.0

    def to_dict(self) -> dict:
        return {
            "schema_version": "cbf-diagnostics-1",
            "intervened": self.intervened,
            "cbf_active": self.cbf_active,
            "infeasible": self.infeasible,
            "fallback": self.fallback,
            "replan_required": self.replan_required,
            "limit_clamped": self.limit_clamped,
            "nominal_accel": self.nominal_accel,
            "nominal_steer_cmd": self.nominal_steer_cmd,
            "safe_accel": self.safe_accel,
            "safe_steer_cmd": self.safe_steer_cmd,
            "accel_correction": self.accel_correction,
            "steer_correction": self.steer_correction,
            "min_clearance_m": self.min_clearance_m,
            "min_backup_clearance_m": self.min_backup_clearance_m,
            "nominal_backup_clearance_m": self.nominal_backup_clearance_m,
            "closing_speed_mps": self.closing_speed_mps,
            "nearest_obstacle_id": self.nearest_obstacle_id,
            "active_obstacle_ids": list(self.active_obstacle_ids),
            "solve_time_s": self.solve_time_s,
            "solver_status": self.solver_status,
            "pose_threejs": {
                "x": self.pose_threejs[0],
                "y": self.pose_threejs[1],
                "z": self.pose_threejs[2],
            },
            "yaw_dataset": self.yaw_dataset,
            "yaw_threejs": self.yaw_threejs,
        }


def _nominal_vector(control: ControlInput, state: VehicleState, dt: float) -> np.ndarray:
    if control.steer_cmd is not None:
        steer = float(control.steer_cmd)
    else:
        steer = float(state.steering + control.steer_rate * dt)
    return np.array([float(control.accel), steer], dtype=np.float64)


def _bounds(state: VehicleState, dt: float, params: VehicleParams) -> tuple[np.ndarray, np.ndarray]:
    window = params.max_steer_rate * max(dt, 0.0)
    lo_steer = max(-params.max_steer, state.steering - window)
    hi_steer = min(params.max_steer, state.steering + window)
    if hi_steer < lo_steer:
        mid = float(np.clip(state.steering, -params.max_steer, params.max_steer))
        lo_steer = hi_steer = mid
    lo = np.array([-params.max_decel, lo_steer], dtype=np.float64)
    hi = np.array([params.max_accel, hi_steer], dtype=np.float64)
    return lo, hi


def _obstacle_clearance(
    footprint, obs: BoxObstacle | CircleObstacle, t: float
) -> tuple[float, np.ndarray]:
    if isinstance(obs, CircleObstacle):
        cx, cz = obs.center_at(t)
        return box_circle_clearance(footprint, cx, cz, obs.radius)
    x, z, yaw = obs.pose_at(t)
    other = OrientedBox(x, z, yaw, obs.length, obs.width)
    return oriented_box_clearance(footprint, other)


def _slacks(
    state: VehicleState,
    obstacles: Sequence[BoxObstacle | CircleObstacle],
    params: VehicleParams,
    margin: float,
) -> np.ndarray:
    if not obstacles:
        return np.zeros(0, dtype=np.float64)
    foot = state.footprint(params)
    out = np.empty(len(obstacles), dtype=np.float64)
    for i, obs in enumerate(obstacles):
        clearance, _ = _obstacle_clearance(foot, obs, state.t)
        out[i] = clearance - margin
    return out


def instantaneous_clearance(
    state: VehicleState,
    scene: SafetyScene,
    params: VehicleParams,
) -> tuple[float, float, str]:
    """Geometric footprint clearance, closing speed, and nearest obstacle id.

    Closing speed is positive when the gap to the nearest obstacle is shrinking.
    The normal used here is the direction that moves the rover away from the
    obstacle.
    """
    obstacles = scene.obstacles()
    if not obstacles:
        return math.inf, 0.0, ""
    foot = state.footprint(params)
    v_body = np.array(
        [state.speed * math.cos(state.yaw), state.speed * math.sin(state.yaw)],
        dtype=np.float64,
    )
    best = math.inf
    best_closing = 0.0
    best_id = ""
    for obs in obstacles:
        clearance, normal = _obstacle_clearance(foot, obs, state.t)
        if clearance < best:
            v_obs = np.array([obs.vx, obs.vz], dtype=np.float64)
            # Positive when relative velocity points into the obstacle.
            best_closing = max(0.0, float(-normal @ (v_body - v_obs)))
            best = clearance
            best_id = obs.id
    return float(best), float(best_closing), best_id


def _within_reach(
    state: VehicleState,
    obs: BoxObstacle | CircleObstacle,
    reach: float,
    horizon_t: float,
) -> bool:
    if isinstance(obs, CircleObstacle):
        radius = obs.radius
        ox, oz = obs.x, obs.z
    else:
        radius = 0.5 * math.hypot(obs.length, obs.width)
        ox, oz = obs.x, obs.z
    gap = math.hypot(ox - state.x, oz - state.z) - radius
    obs_speed = math.hypot(obs.vx, obs.vz)
    return gap < reach + obs_speed * horizon_t


def _select_obstacles(
    state: VehicleState,
    scene: SafetyScene,
    params: VehicleParams,
    config: CBFConfig,
) -> list[BoxObstacle | CircleObstacle]:
    horizon_t = (
        config.preview_time
        + state.speed / max(params.max_decel, 1e-3)
        + config.idle_time
        + 0.4
    )
    travel = state.speed * config.preview_time + (state.speed**2) / (
        2.0 * max(params.max_decel, 1e-3)
    )
    travel += 0.5 * params.max_accel * config.preview_time**2
    reach = travel + config.activation_clearance + config.nearby_extra
    return [
        obs
        for obs in scene.obstacles()
        if _within_reach(state, obs, reach, horizon_t)
    ]


def _rollout_slack(
    state: VehicleState,
    control: np.ndarray,
    obstacles: Sequence[BoxObstacle | CircleObstacle],
    dt: float,
    params: VehicleParams,
    config: CBFConfig,
    model: BicycleModel,
    abort_below: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Backup-clearance slack and one-step slack for each obstacle.

    The first sample uses the plant ``dt``. Later samples use at least 0.1 s,
    which is the backup integration step and not a second plant model. The
    rollout applies ``control`` for ``preview_time``, brakes at ``-max_decel``
    with steering held, then holds the stopped pose. Slack is clearance minus
    ``safety_margin``. ``abort_below`` stops once the minimum is already past
    that value; the returned minimum is then an upper bound on the true one.
    """
    n = len(obstacles)
    min_slack = np.full(n, np.inf, dtype=np.float64)
    next_slack = np.full(n, np.inf, dtype=np.float64)
    if n == 0:
        return min_slack, next_slack

    def absorb(slack: np.ndarray) -> bool:
        nonlocal min_slack
        min_slack = np.minimum(min_slack, slack)
        return abort_below is not None and float(np.min(min_slack)) < abort_below

    s = state.copy()
    command = ControlInput(accel=float(control[0]), steer_cmd=float(control[1]))
    s = model.step(s, command, dt)
    next_slack = _slacks(s, obstacles, params, config.safety_margin)
    if absorb(next_slack):
        return min_slack, next_slack

    roll_dt = max(dt, 0.1)
    preview_left = max(0.0, config.preview_time - dt)
    for _ in range(int(round(preview_left / roll_dt))):
        s = model.step(s, command, roll_dt)
        if absorb(_slacks(s, obstacles, params, config.safety_margin)):
            return min_slack, next_slack

    held_steer = s.steering
    max_brake_steps = (
        int(math.ceil((params.max_speed / max(params.max_decel, 1e-3) + 0.4) / roll_dt)) + 1
    )
    taken = 0
    while s.speed > 0.05 and taken < max_brake_steps:
        s = model.step(
            s, ControlInput(accel=-params.max_decel, steer_cmd=held_steer), roll_dt
        )
        held_steer = s.steering
        taken += 1
        if absorb(_slacks(s, obstacles, params, config.safety_margin)):
            return min_slack, next_slack

    parked = s.copy()
    parked.speed = 0.0
    t = parked.t
    idle_steps = max(0, int(round(config.idle_time / roll_dt)))
    for _ in range(idle_steps):
        t += roll_dt
        parked.t = t
        if absorb(_slacks(parked, obstacles, params, config.safety_margin)):
            return min_slack, next_slack
    return min_slack, next_slack


def _perturb(u: np.ndarray, index: int, step: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    up = u.copy()
    up[index] = min(hi[index], u[index] + step)
    if abs(up[index] - u[index]) < 0.25 * step:
        up[index] = max(lo[index], u[index] - step)
    return up


def _weighted_cost(u: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    d = u - target
    return float(np.sum(weights * d * d))


def _min_slack(slack: np.ndarray) -> float:
    if len(slack) == 0:
        return math.inf
    return float(np.min(slack))


def _unique_steers(values: list[float], lo: float, hi: float) -> list[float]:
    out: list[float] = []
    for raw in values:
        steer = float(np.clip(raw, lo, hi))
        if all(abs(steer - prev) > 1e-3 for prev in out):
            out.append(steer)
    return out


def _recover_command(
    sol: Optional[np.ndarray],
    u_box: np.ndarray,
    u_raw: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    evaluate,
    weights: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, bool, float]:
    """Back off acceleration, then fall back to maximum braking.

    A feasible command is the one closest to the nominal command. If none of
    the sampled commands keep the backup slack above ``tol``, the return value
    is maximum braking with the steering angle that leaves the largest slack.
    The boolean is True when that command is backup-feasible.
    """
    steers = _unique_steers(
        [
            float(sol[1]) if sol is not None else float(u_box[1]),
            float(u_box[1]),
            0.0,
            float(lo[1]),
            float(hi[1]),
        ],
        float(lo[1]),
        float(hi[1]),
    )
    start_accel = float(sol[0]) if sol is not None else float(u_box[0])
    mid_accel = 0.5 * (start_accel + float(lo[0]))
    # Straight steering has to be in this list. A wall-seeking nominal can be
    # infeasible at every acceleration while zero steer at the same speed is safe.
    candidates = [
        np.array([accel, steer], dtype=np.float64)
        for steer in steers[:3]
        for accel in (start_accel, mid_accel, float(lo[0]))
    ]
    candidates.extend(
        np.array([float(lo[0]), steer], dtype=np.float64) for steer in steers[3:]
    )
    candidates.sort(key=lambda u: _weighted_cost(u, u_raw, weights))
    best_bad = np.array([float(lo[0]), steers[0]], dtype=np.float64)
    best_bad_slack = -math.inf
    seen: set[tuple[float, float]] = set()
    for candidate in candidates:
        key = (round(float(candidate[0]), 3), round(float(candidate[1]), 4))
        if key in seen:
            continue
        seen.add(key)
        slack = _min_slack(evaluate(candidate, abort_below=-0.35)[0])
        if slack >= -tol:
            return candidate, True, slack
        if slack > best_bad_slack and abs(float(candidate[0]) - float(lo[0])) < 1e-6:
            best_bad_slack = slack
            best_bad = candidate
    return best_bad, False, best_bad_slack


def _persistent_forward_block(
    state: VehicleState,
    scene: SafetyScene,
    params: VehicleParams,
    config: CBFConfig,
) -> bool:
    """True when a pose ahead of the rover stays inside the margin for the horizon.

    A crossing agent that only occupies the heading briefly does not count: some
    predicted time must open a gap, or the route is treated as blocked.
    """
    if not scene.obstacles():
        return False
    n_times = max(2, int(round(config.replan_horizon / 0.5)))
    times = [scene.t + i * (config.replan_horizon / (n_times - 1)) for i in range(n_times)]
    cy = math.cos(state.yaw)
    sy = math.sin(state.yaw)
    for dist in config.replan_distances:
        probe = state.copy()
        probe.x = state.x + dist * cy
        probe.z = state.z + dist * sy
        probe.speed = 0.0
        worst_open = -math.inf
        for t in times:
            probe.t = t
            slack = _slacks(probe, scene.obstacles(), params, config.safety_margin)
            worst_open = max(worst_open, _min_slack(slack))
        if worst_open < 0.0:
            return True
    return False


def filter_control(
    state: VehicleState,
    nominal_control: ControlInput,
    obstacles: SafetyScene,
    dt: float,
    params: Optional[VehicleParams] = None,
    config: Optional[CBFConfig] = None,
) -> tuple[ControlInput, CBFDiagnostics]:
    """Return the closest backup-safe command and a diagnostic record.

    ``obstacles`` must already contain predicted velocities. Future ground-truth
    samples are not consulted here.
    """
    started = time.perf_counter()
    params = params or VehicleParams()
    config = config or CBFConfig()
    u_raw = _nominal_vector(nominal_control, state, dt)
    lo, hi = _bounds(state, dt, params)
    if dt <= 0.0:
        u_box = np.clip(u_raw, lo, hi)
        diag = _diagnostics(
            state,
            u_raw,
            u_box,
            u_box,
            obstacles,
            params,
            infeasible=False,
            fallback="",
            replan=False,
            active=[],
            nominal_backup=math.inf,
            safe_backup=math.inf,
            status="nominal_safe",
            solve_time=time.perf_counter() - started,
        )
        return ControlInput(accel=float(u_box[0]), steer_cmd=float(u_box[1])), diag

    u_box = np.clip(u_raw, lo, hi)
    relevant = _select_obstacles(state, obstacles, params, config)
    model = BicycleModel(params)
    weights = np.array([config.weight_accel, config.weight_steer], dtype=np.float64)

    cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def evaluate(
        u: np.ndarray, abort_below: Optional[float] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (round(float(u[0]), 3), round(float(u[1]), 4), abort_below)
        hit = cache.get(key)
        if hit is None:
            hit = _rollout_slack(
                state,
                u,
                relevant,
                dt,
                params,
                config,
                model,
                abort_below=abort_below,
            )
            cache[key] = hit
        return hit

    current_slack = _slacks(state, relevant, params, config.safety_margin)
    min0, next0 = evaluate(u_box)
    nominal_backup = _min_slack(min0)

    hot = nominal_backup <= config.activation_clearance or _min_slack(current_slack) <= config.activation_clearance
    rows: list[tuple[np.ndarray, float]] = []
    active: list[str] = []
    if hot and relevant:
        u_a = _perturb(u_box, 0, config.fd_accel, lo, hi)
        u_d = _perturb(u_box, 1, config.fd_steer, lo, hi)
        min_a, next_a = evaluate(u_a) if abs(u_a[0] - u_box[0]) > 1e-7 else (min0, next0)
        min_d, next_d = evaluate(u_d) if abs(u_d[1] - u_box[1]) > 1e-7 else (min0, next0)
        g_min = np.zeros((len(relevant), 2), dtype=np.float64)
        g_next = np.zeros((len(relevant), 2), dtype=np.float64)
        da = float(u_a[0] - u_box[0])
        dd = float(u_d[1] - u_box[1])
        if abs(da) > 1e-8:
            g_min[:, 0] = (min_a - min0) / da
            g_next[:, 0] = (next_a - next0) / da
        if abs(dd) > 1e-8:
            g_min[:, 1] = (min_d - min0) / dd
            g_next[:, 1] = (next_d - next0) / dd

        for i, obs in enumerate(relevant):
            if min0[i] <= config.activation_clearance and float(np.linalg.norm(g_min[i])) > 1e-5:
                g = g_min[i]
                rhs = -float(min0[i]) + float(g @ u_box)
                rows.append((g, rhs))
                active.append(obs.id)
            if (
                config.use_one_step_dcbf
                and current_slack[i] <= config.activation_clearance
                and float(np.linalg.norm(g_next[i])) > 1e-5
            ):
                g = g_next[i]
                target_h = (1.0 - config.gamma) * float(current_slack[i])
                rhs = target_h - float(next0[i]) + float(g @ u_box)
                rows.append((g, rhs))
                if obs.id not in active:
                    active.append(obs.id)

    status = "nominal_safe"
    infeasible = False
    fallback = ""
    u_safe = u_box.copy()
    safe_backup = nominal_backup

    if hot and (rows or nominal_backup < -config.violation_tol):
        sol = solve_weighted_qp_2d(u_raw, weights, rows, lo, hi) if rows else None
        verified = False
        if sol is not None:
            sol_slack, _ = evaluate(sol)
            if _min_slack(sol_slack) >= -config.violation_tol:
                u_safe = sol
                safe_backup = _min_slack(sol_slack)
                status = "qp_feasible"
                verified = True
        if not verified:
            recovered, ok, recovered_slack = _recover_command(
                sol, u_box, u_raw, lo, hi, evaluate, weights, config.violation_tol
            )
            u_safe = recovered
            safe_backup = recovered_slack
            if ok:
                status = "line_search"
            else:
                infeasible = True
                fallback = "max_brake"
                status = "infeasible_fallback"

    min_clearance, closing, nearest = instantaneous_clearance(state, obstacles, params)
    replan = False
    # Only while the filter is holding the rover back. A passable corridor whose
    # reference line sits near a wall should not request a new route every step.
    withholding = float(u_safe[0]) <= 0.05 and float(u_raw[0]) > 0.2
    if (infeasible or withholding) and min_clearance <= config.replan_distances[-1] + 2.0:
        replan = _persistent_forward_block(state, obstacles, params, config)

    diag = _diagnostics(
        state,
        u_raw,
        u_box,
        u_safe,
        obstacles,
        params,
        infeasible=infeasible,
        fallback=fallback,
        replan=replan,
        active=active,
        nominal_backup=nominal_backup,
        safe_backup=safe_backup,
        status=status,
        solve_time=time.perf_counter() - started,
        min_clearance=min_clearance,
        closing=closing,
        nearest=nearest,
    )
    return ControlInput(accel=float(u_safe[0]), steer_cmd=float(u_safe[1])), diag


def _diagnostics(
    state: VehicleState,
    u_raw: np.ndarray,
    u_box: np.ndarray,
    u_safe: np.ndarray,
    scene: SafetyScene,
    params: VehicleParams,
    *,
    infeasible: bool,
    fallback: str,
    replan: bool,
    active: list[str],
    nominal_backup: float,
    safe_backup: float,
    status: str,
    solve_time: float,
    min_clearance: Optional[float] = None,
    closing: Optional[float] = None,
    nearest: Optional[str] = None,
) -> CBFDiagnostics:
    if min_clearance is None or closing is None or nearest is None:
        min_clearance, closing, nearest = instantaneous_clearance(state, scene, params)
    tx, ty, tz = dataset_to_threejs(state.x, state.z, state.y)
    limit_clamped = bool(np.any(np.abs(u_raw - u_box) > 1e-6))
    cbf_active = bool(np.linalg.norm(u_safe - u_box) > 1e-3)
    intervened = bool(np.linalg.norm(u_safe - u_raw) > 1e-3)

    def _finite(value: float) -> float:
        if math.isinf(value):
            return 1e9 if value > 0 else -1e9
        return float(value)

    return CBFDiagnostics(
        intervened=intervened,
        cbf_active=cbf_active,
        infeasible=bool(infeasible),
        fallback=fallback,
        replan_required=bool(replan),
        limit_clamped=limit_clamped,
        nominal_accel=float(u_raw[0]),
        nominal_steer_cmd=float(u_raw[1]),
        safe_accel=float(u_safe[0]),
        safe_steer_cmd=float(u_safe[1]),
        accel_correction=float(u_safe[0] - u_raw[0]),
        steer_correction=float(u_safe[1] - u_raw[1]),
        min_clearance_m=_finite(min_clearance),
        min_backup_clearance_m=_finite(safe_backup),
        nominal_backup_clearance_m=_finite(nominal_backup),
        closing_speed_mps=float(closing),
        nearest_obstacle_id=nearest,
        active_obstacle_ids=list(dict.fromkeys(active)),
        solve_time_s=float(solve_time),
        solver_status=status,
        pose_threejs=(float(tx), float(ty), float(tz)),
        yaw_dataset=float(state.yaw),
        yaw_threejs=float(-state.yaw),
    )
