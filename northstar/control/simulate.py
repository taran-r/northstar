"""Closed loop: path tracker, safety filter, bicycle plant.

Collision metrics use the rover body (no footprint inflation) against the
ground-truth obstacle pose at the current time. The filter, when enabled, sees
only the causal constant-velocity scene from ``build_scene``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from northstar.control.cbf import CBFConfig, CBFDiagnostics, filter_control
from northstar.control.scene import BoxObstacle, CircleObstacle, SafetyScene, build_scene
from northstar.facility.schema import DynamicObject, FacilitySpec, StaticObject
from northstar.vehicle.bicycle import BicycleModel, ControlInput, VehicleParams, VehicleState


class NominalTracker(Protocol):
    def nominal_control(self, state: VehicleState, dt: float) -> ControlInput: ...


@dataclass
class SimWorld:
    """Geometry available to a scenario. Dynamics carry ground-truth samples."""

    boxes: list[BoxObstacle] = field(default_factory=list)
    circles: list[CircleObstacle] = field(default_factory=list)
    facility: Optional[FacilitySpec] = None
    static_obstacles: list[StaticObject] = field(default_factory=list)
    dynamic_obstacles: list[DynamicObject] = field(default_factory=list)


@dataclass
class StepRecord:
    t: float
    x: float
    z: float
    yaw: float
    speed: float
    steering: float
    nominal_accel: float
    nominal_steer: float
    applied_accel: float
    applied_steer: float
    clearance_m: float
    collided: bool
    intervened: bool
    cbf_active: bool
    infeasible: bool
    replan_required: bool
    solve_time_s: float
    solver_status: str


@dataclass
class SimResult:
    name: str
    use_cbf: bool
    dt: float
    collided: bool
    collision_steps: int
    min_clearance_m: float
    intervention_count: int
    cbf_active_count: int
    mean_abs_accel_correction: float
    mean_abs_steer_correction: float
    max_abs_accel_correction: float
    max_abs_steer_correction: float
    nominal_limit_violations: int
    applied_limit_violations: int
    state_limit_violations: int
    infeasible_count: int
    replan_count: int
    goal_progress: float
    final_goal_distance_m: float
    mean_solve_time_s: float
    max_solve_time_s: float
    impact_speed_mps: Optional[float]
    final_speed_mps: float
    final_x: float
    final_z: float
    steps_n: int
    steps: list[StepRecord] = field(default_factory=list)

    def metrics(self) -> dict:
        return {
            "use_cbf": self.use_cbf,
            "dt": self.dt,
            "collided": self.collided,
            "collision_steps": self.collision_steps,
            "min_clearance_m": self.min_clearance_m,
            "intervention_count": self.intervention_count,
            "cbf_active_count": self.cbf_active_count,
            "mean_abs_accel_correction": self.mean_abs_accel_correction,
            "mean_abs_steer_correction": self.mean_abs_steer_correction,
            "max_abs_accel_correction": self.max_abs_accel_correction,
            "max_abs_steer_correction": self.max_abs_steer_correction,
            "nominal_limit_violations": self.nominal_limit_violations,
            "applied_limit_violations": self.applied_limit_violations,
            "state_limit_violations": self.state_limit_violations,
            "infeasible_count": self.infeasible_count,
            "replan_count": self.replan_count,
            "goal_progress": self.goal_progress,
            "final_goal_distance_m": self.final_goal_distance_m,
            "mean_solve_time_s": self.mean_solve_time_s,
            "max_solve_time_s": self.max_solve_time_s,
            "impact_speed_mps": self.impact_speed_mps,
            "final_speed_mps": self.final_speed_mps,
            "final_x": self.final_x,
            "final_z": self.final_z,
            "steps": self.steps_n,
        }

    def summary(self) -> str:
        m = self.metrics()
        bits = [f"{k}={m[k]}" for k in (
            "collided",
            "collision_steps",
            "min_clearance_m",
            "intervention_count",
            "cbf_active_count",
            "infeasible_count",
            "replan_count",
            "goal_progress",
            "impact_speed_mps",
            "mean_solve_time_s",
            "max_solve_time_s",
            "nominal_limit_violations",
            "applied_limit_violations",
            "state_limit_violations",
            "final_speed_mps",
        )]
        return f"{self.name} cbf={self.use_cbf} " + " ".join(bits)


def scene_for(
    world: SimWorld,
    t: float,
    *,
    dynamics_mode: str,
) -> SafetyScene:
    return build_scene(
        t,
        boxes=world.boxes,
        circles=world.circles,
        facility=world.facility,
        static_obstacles=world.static_obstacles,
        dynamic_obstacles=world.dynamic_obstacles,
        dynamics_mode=dynamics_mode,
    )


def body_clearance(state: VehicleState, scene: SafetyScene, params: VehicleParams) -> float:
    """Clearance of the physical body, without the footprint inflation."""
    from northstar.control.cbf import instantaneous_clearance

    bare = VehicleParams(**{**params.to_dict(), "footprint_margin": 0.0})
    clearance, _, _ = instantaneous_clearance(state, scene, bare)
    return clearance


def _command_within_limits(
    state: VehicleState, control: ControlInput, dt: float, params: VehicleParams
) -> bool:
    if control.accel < -params.max_decel - 1e-6 or control.accel > params.max_accel + 1e-6:
        return False
    if control.steer_cmd is None:
        return abs(control.steer_rate) <= params.max_steer_rate + 1e-6
    if abs(control.steer_cmd) > params.max_steer + 1e-6:
        return False
    if abs(control.steer_cmd - state.steering) > params.max_steer_rate * dt + 1e-5:
        return False
    return True


def _state_within_limits(state: VehicleState, params: VehicleParams) -> bool:
    if state.speed < params.min_speed - 1e-6 or state.speed > params.max_speed + 1e-6:
        return False
    if abs(state.steering) > params.max_steer + 1e-6:
        return False
    return True


def closed_loop_step(
    state: VehicleState,
    tracker: NominalTracker,
    scene: SafetyScene,
    dt: float,
    params: Optional[VehicleParams] = None,
    config: Optional[CBFConfig] = None,
    use_cbf: bool = True,
) -> tuple[VehicleState, ControlInput, ControlInput, Optional[CBFDiagnostics]]:
    """One integration step: tracker, optional safety filter, bicycle update.

    This is the hook a flow-matching follower should call. ``tracker`` stands in
    for whatever turns a generated trajectory into ``(accel, steer_cmd)``. The
    filter does not read that trajectory or the network.
    """
    params = params or VehicleParams()
    nominal = tracker.nominal_control(state, dt)
    diagnostics: Optional[CBFDiagnostics] = None
    if use_cbf:
        applied, diagnostics = filter_control(
            state, nominal, scene, dt, params=params, config=config
        )
    else:
        applied = nominal
    nxt = BicycleModel(params).step(state, applied, dt)
    return nxt, nominal, applied, diagnostics


def simulate_scenario(
    name: str,
    start: VehicleState,
    goal_xy: tuple[float, float],
    tracker: NominalTracker,
    world: SimWorld,
    dt: float,
    duration: float,
    use_cbf: bool,
    params: Optional[VehicleParams] = None,
    config: Optional[CBFConfig] = None,
) -> SimResult:
    """Roll a scenario with the filter on or off. Scoring uses ground truth."""
    params = params or VehicleParams()
    state = start.copy()
    steps: list[StepRecord] = []
    min_clearance = math.inf
    collision_steps = 0
    interventions = 0
    cbf_active = 0
    infeasible = 0
    replans = 0
    nominal_violations = 0
    applied_violations = 0
    state_violations = 0
    solve_times: list[float] = []
    accel_corr: list[float] = []
    steer_corr: list[float] = []
    impact: Optional[float] = None
    max_steps = int(duration / max(dt, 1e-3)) + 2

    for _ in range(max_steps):
        if state.t >= duration - 1e-9:
            break
        perceived = scene_for(world, state.t, dynamics_mode="causal_cv")
        nxt, nominal, applied, diag = closed_loop_step(
            state, tracker, perceived, dt, params=params, config=config, use_cbf=use_cbf
        )
        truth = scene_for(world, nxt.t, dynamics_mode="snapshot_gt")
        clearance = body_clearance(nxt, truth, params)
        hit = clearance < -1e-6
        if hit:
            collision_steps += 1
            if impact is None:
                impact = float(nxt.speed)
        min_clearance = min(min_clearance, clearance)
        if not _command_within_limits(state, nominal, dt, params):
            nominal_violations += 1
        if not _command_within_limits(state, applied, dt, params):
            applied_violations += 1
        if not _state_within_limits(nxt, params):
            state_violations += 1
        accel_corr.append(abs(applied.accel - nominal.accel))
        steer_delta = (applied.steer_cmd if applied.steer_cmd is not None else 0.0) - (
            nominal.steer_cmd if nominal.steer_cmd is not None else nominal.steer_rate
        )
        steer_corr.append(abs(steer_delta))
        intervened = bool(diag and diag.intervened)
        active = bool(diag and diag.cbf_active)
        bad = bool(diag and diag.infeasible)
        replan = bool(diag and diag.replan_required)
        interventions += int(intervened)
        cbf_active += int(active)
        infeasible += int(bad)
        replans += int(replan)
        solve = float(diag.solve_time_s) if diag else 0.0
        solve_times.append(solve)
        steps.append(
            StepRecord(
                t=nxt.t,
                x=nxt.x,
                z=nxt.z,
                yaw=nxt.yaw,
                speed=nxt.speed,
                steering=nxt.steering,
                nominal_accel=nominal.accel,
                nominal_steer=float(nominal.steer_cmd if nominal.steer_cmd is not None else nominal.steer_rate),
                applied_accel=applied.accel,
                applied_steer=float(applied.steer_cmd if applied.steer_cmd is not None else applied.steer_rate),
                clearance_m=float(clearance) if math.isfinite(clearance) else 1e9,
                collided=hit,
                intervened=intervened,
                cbf_active=active,
                infeasible=bad,
                replan_required=replan,
                solve_time_s=solve,
                solver_status=diag.solver_status if diag else "disabled",
            )
        )
        state = nxt

    start_dist = math.hypot(goal_xy[0] - start.x, goal_xy[1] - start.z)
    final_dist = math.hypot(goal_xy[0] - state.x, goal_xy[1] - state.z)
    progress = (start_dist - final_dist) / max(start_dist, 1e-6)
    if not math.isfinite(min_clearance):
        min_clearance = 1e9
    return SimResult(
        name=name,
        use_cbf=use_cbf,
        dt=dt,
        collided=collision_steps > 0,
        collision_steps=collision_steps,
        min_clearance_m=float(min_clearance),
        intervention_count=interventions,
        cbf_active_count=cbf_active,
        mean_abs_accel_correction=float(np.mean(accel_corr)) if accel_corr else 0.0,
        mean_abs_steer_correction=float(np.mean(steer_corr)) if steer_corr else 0.0,
        max_abs_accel_correction=float(np.max(accel_corr)) if accel_corr else 0.0,
        max_abs_steer_correction=float(np.max(steer_corr)) if steer_corr else 0.0,
        nominal_limit_violations=nominal_violations,
        applied_limit_violations=applied_violations,
        state_limit_violations=state_violations,
        infeasible_count=infeasible,
        replan_count=replans,
        goal_progress=float(progress),
        final_goal_distance_m=float(final_dist),
        mean_solve_time_s=float(np.mean(solve_times)) if solve_times else 0.0,
        max_solve_time_s=float(np.max(solve_times)) if solve_times else 0.0,
        impact_speed_mps=impact,
        final_speed_mps=float(state.speed),
        final_x=float(state.x),
        final_z=float(state.z),
        steps_n=len(steps),
        steps=steps,
    )
