"""Trajectory checks that go beyond the v1 window heuristics.

Collision checks use the rover footprint, including swept samples between
saved states, against permanent geometry, temporary static obstacles, and
ground-truth dynamic poses. A closed-loop bicycle step checks that the saved
accel and steering are locally consistent with the kinematic model.

Linear resampling of the expert path means the bicycle model will not match
the saved polyline to millimeter precision. ``ROLLOUT_STEP_TOL_M`` is the
accepted per-step position error for that reason. Collision checks are not
relaxed to meet a dataset-size target.
"""

from __future__ import annotations

import math

from northstar.collision import CollisionChecker
from northstar.facility.occupancy import build_facility_occupancy
from northstar.facility.schema import FacilitySpec, ScenarioSpec
from northstar.geometry import wrap_angle
from northstar.vehicle.bicycle import BicycleModel, ControlInput, VehicleParams, VehicleState


ROLLOUT_STEP_TOL_M = 0.45
OPEN_LOOP_END_TOL_M = 1.5


def _interp(a: VehicleState, b: VehicleState, u: float) -> VehicleState:
    dyaw = (b.yaw - a.yaw + math.pi) % (2 * math.pi) - math.pi
    return VehicleState(
        x=a.x + u * (b.x - a.x),
        z=a.z + u * (b.z - a.z),
        yaw=wrap_angle(a.yaw + u * dyaw),
        speed=a.speed + u * (b.speed - a.speed),
        steering=a.steering + u * (b.steering - a.steering),
        t=a.t + u * (b.t - a.t),
    )


def validate_trajectory(
    facility: FacilitySpec,
    scenario: ScenarioSpec,
    states: list[VehicleState],
    accels: list[float],
    params: VehicleParams | None = None,
) -> tuple[bool, str]:
    params = params or VehicleParams()
    if len(states) < 5:
        return False, "too_short"
    for s in states:
        if not all(math.isfinite(v) for v in (s.x, s.z, s.yaw, s.speed, s.steering, s.t)):
            return False, "nonfinite"
    for i in range(1, len(states)):
        dt = states[i].t - states[i - 1].t
        if dt <= 1e-6 or abs(dt - 0.2) > 0.05:
            return False, "timestamp"
        if states[i].speed > params.max_speed + 0.05 or states[i].speed < -1e-3:
            return False, "speed_limit"
        if abs(states[i].steering) > params.max_steer + 0.05:
            return False, "steer_limit"
        steer_rate = abs(states[i].steering - states[i - 1].steering) / dt
        if steer_rate > params.max_steer_rate + 0.25:
            return False, "steer_rate"
        accel = (states[i].speed - states[i - 1].speed) / dt
        if accel > params.max_accel + 0.35 or accel < -(params.max_decel + 0.35):
            return False, "accel_limit"
        disp = math.hypot(states[i].x - states[i - 1].x, states[i].z - states[i - 1].z)
        vmax = max(states[i].speed, states[i - 1].speed)
        if disp > vmax * dt * 1.6 + 0.2:
            return False, "heading_or_step"
        if vmax > 0.8 and disp > 0.05:
            heading = math.atan2(states[i].z - states[i - 1].z, states[i].x - states[i - 1].x)
            err = abs((heading - states[i - 1].yaw + math.pi) % (2 * math.pi) - math.pi)
            if err > 1.1:
                return False, "heading_or_step"

    occ = build_facility_occupancy(
        facility, resolution=0.25, static_obstacles=scenario.static_obstacles
    )
    collision = CollisionChecker(
        facility,
        static_obstacles=scenario.static_obstacles,
        occupancy=occ,
        vehicle_params=params,
    )
    collision.fast_occupancy = False
    for i, state in enumerate(states):
        if collision.collides_static(state):
            return False, "static_collision"
        if collision.collides_dynamic(
            state, scenario.dynamic_obstacles, t=state.t, use_prediction=False, margin=0.05
        ):
            return False, "dynamic_collision"
        if i == 0:
            continue
        for u in (0.25, 0.5, 0.75):
            mid = _interp(states[i - 1], state, u)
            if collision.collides_static(mid):
                return False, "swept_static_collision"
            if collision.collides_dynamic(
                mid, scenario.dynamic_obstacles, t=mid.t, use_prediction=False, margin=0.0
            ):
                return False, "swept_dynamic_collision"

    model = BicycleModel(params)
    for i in range(len(states) - 1):
        dt = states[i + 1].t - states[i].t
        accel = accels[min(i, len(accels) - 1)] if accels else 0.0
        pred = model.step(
            states[i],
            ControlInput(accel=accel, steer_cmd=states[i + 1].steering),
            dt,
        )
        err = math.hypot(pred.x - states[i + 1].x, pred.z - states[i + 1].z)
        if err > ROLLOUT_STEP_TOL_M:
            return False, "rollout"

    sim = states[0].copy()
    for i in range(len(states) - 1):
        dt = states[i + 1].t - states[i].t
        accel = accels[min(i, len(accels) - 1)] if accels else 0.0
        sim = model.step(sim, ControlInput(accel=accel, steer_cmd=states[i + 1].steering), dt)
    end_err = math.hypot(sim.x - states[-1].x, sim.z - states[-1].z)
    if end_err > OPEN_LOOP_END_TOL_M:
        return False, "open_loop_drift"
    return True, "ok"
