"""Safety-filter checks that do not depend on the flow-matching model."""

from __future__ import annotations

import math

import numpy as np

from northstar.control.cbf import filter_control
from northstar.control.clearance import box_circle_clearance, oriented_box_clearance
from northstar.control.qp import solve_weighted_qp_2d
from northstar.control.scenarios import (
    AggressiveTracker,
    crossing_forklift,
    follow_lead,
    narrow_corridor,
    open_floor,
    static_barrier,
    unavoidable_wall,
)
from northstar.control.scene import BoxObstacle, build_scene
from northstar.control.simulate import simulate_scenario
from northstar.coords import dataset_to_threejs
from northstar.facility.loader import load_facility
from northstar.facility.schema import DynamicObject, DynamicWaypoint, FacilitySpec, WallSegment
from northstar.geometry import OrientedBox
from northstar.vehicle.bicycle import ControlInput, VehicleParams, VehicleState

ROOT_FACILITY = "facilities/industrial_facility_v1.json"


def _run(scenario, use_cbf: bool, params: VehicleParams | None = None):
    params = params or VehicleParams()
    return simulate_scenario(
        scenario.name,
        scenario.start,
        scenario.goal_xy,
        scenario.tracker,
        scenario.world,
        scenario.dt,
        scenario.duration,
        use_cbf=use_cbf,
        params=params,
    )


def test_box_clearance_gap_and_penetration():
    a = OrientedBox(0.0, 0.0, 0.0, 2.0, 2.0)
    separated = OrientedBox(3.0, 0.0, 0.0, 2.0, 2.0)
    gap, normal = oriented_box_clearance(a, separated)
    assert gap == pytest_approx(1.0, abs=1e-6)
    assert normal[0] < 0.0

    overlapping = OrientedBox(1.5, 0.0, 0.0, 2.0, 2.0)
    pen, _ = oriented_box_clearance(a, overlapping)
    assert pen == pytest_approx(-0.5, abs=1e-6)

    # Circle center 2 m ahead of a box face at x=1, radius 0.5 -> gap 0.5.
    circ, _ = box_circle_clearance(a, 2.0, 0.0, 0.5)
    assert circ == pytest_approx(0.5, abs=1e-6)
    inside, _ = box_circle_clearance(a, 0.0, 0.0, 0.3)
    assert inside < 0.0

    rotated = OrientedBox(0.0, 3.0, math.pi / 2.0, 2.0, 2.0)
    gap_r, _ = oriented_box_clearance(a, rotated)
    assert gap_r == pytest_approx(1.0, abs=1e-6)

    tilted = OrientedBox(3.0, 0.0, 0.4, 2.0, 2.0)
    gap_t, _ = oriented_box_clearance(a, tilted)
    assert 0.2 < gap_t < 1.5


def test_qp_projects_onto_box_and_halfplane():
    weights = np.array([1.0, 1.0])
    lo = np.array([-2.0, -2.0])
    hi = np.array([2.0, 2.0])
    inside = solve_weighted_qp_2d(np.array([0.2, -0.1]), weights, [], lo, hi)
    assert inside is not None
    assert np.allclose(inside, [0.2, -0.1])

    clipped = solve_weighted_qp_2d(np.array([10.0, -8.0]), weights, [], lo, hi)
    assert clipped is not None
    assert np.allclose(clipped, [2.0, -2.0])

    # a + d >= 1, target at origin -> (0.5, 0.5)
    half = solve_weighted_qp_2d(
        np.array([0.0, 0.0]),
        weights,
        [(np.array([1.0, 1.0]), 1.0)],
        lo,
        hi,
    )
    assert half is not None
    assert np.allclose(half, [0.5, 0.5], atol=1e-6)

    empty = solve_weighted_qp_2d(
        np.array([0.0, 0.0]),
        weights,
        [(np.array([1.0, 0.0]), 5.0)],
        lo,
        hi,
    )
    assert empty is None


def test_actuator_limits_are_enforced_without_obstacles():
    state = VehicleState(x=0.0, z=0.0, yaw=0.0, speed=1.0, steering=0.1, t=0.0)
    scene = build_scene(0.0)
    safe, diag = filter_control(
        state,
        ControlInput(accel=8.0, steer_cmd=2.0),
        scene,
        dt=0.1,
    )
    params = VehicleParams()
    assert -params.max_decel - 1e-9 <= safe.accel <= params.max_accel + 1e-9
    assert safe.steer_cmd is not None
    assert abs(safe.steer_cmd) <= params.max_steer + 1e-9
    assert abs(safe.steer_cmd - state.steering) <= params.max_steer_rate * 0.1 + 1e-6
    assert diag.limit_clamped
    assert diag.intervened
    assert not diag.cbf_active
    assert not diag.infeasible
    tx, ty, tz = dataset_to_threejs(state.x, state.z, state.y)
    payload = diag.to_dict()
    assert payload["schema_version"] == "cbf-diagnostics-1"
    assert payload["pose_threejs"] == {"x": tx, "y": ty, "z": tz}
    assert payload["yaw_threejs"] == pytest_approx(-state.yaw)


def test_single_step_brakes_before_a_wall():
    wall = BoxObstacle(id="wall", x=3.6, z=0.0, yaw=0.0, length=0.4, width=6.0, kind="wall")
    # Face at x=3.4. Rover front at 0.65, gap about 2.7 m, speed 2.2.
    # Preview plus stopping distance reaches the wall; nominal accel does not.
    state = VehicleState(x=0.0, z=0.0, yaw=0.0, speed=2.2, steering=0.0, t=0.0)
    scene = build_scene(0.0, boxes=[wall])
    safe, diag = filter_control(
        state,
        ControlInput(accel=1.5, steer_cmd=0.0),
        scene,
        dt=0.1,
    )
    assert safe.accel < 0.0, diag.summary() if hasattr(diag, "summary") else diag.to_dict()
    assert diag.cbf_active
    assert diag.nominal_backup_clearance_m < diag.min_backup_clearance_m + 0.5
    assert not diag.infeasible


def test_static_barrier_stops_instead_of_colliding():
    params = VehicleParams()
    scenario = static_barrier(params)
    blind = _run(scenario, False, params)
    guarded = _run(scenario, True, params)
    assert blind.collided, blind.summary()
    assert not guarded.collided, guarded.summary()
    assert guarded.min_clearance_m > 0.0
    assert guarded.cbf_active_count > 0
    assert guarded.replan_count > 0
    assert guarded.applied_limit_violations == 0
    assert guarded.state_limit_violations == 0
    assert guarded.final_speed_mps < 0.3
    assert guarded.mean_solve_time_s < 0.05


def test_crossing_forklift_yields_using_prediction():
    params = VehicleParams()
    scenario = crossing_forklift(params)
    blind = _run(scenario, False, params)
    guarded = _run(scenario, True, params)
    assert blind.collided, blind.summary()
    assert not guarded.collided, guarded.summary()
    assert guarded.min_clearance_m > 0.0
    assert guarded.cbf_active_count > 0
    assert guarded.goal_progress > 0.45, guarded.summary()
    assert guarded.applied_limit_violations == 0


def test_narrow_corridor_rejects_wall_seeking_steer():
    params = VehicleParams()
    scenario = narrow_corridor(params)
    blind = _run(scenario, False, params)
    guarded = _run(scenario, True, params)
    assert blind.collided, blind.summary()
    assert not guarded.collided, guarded.summary()
    assert guarded.min_clearance_m > 0.0
    assert guarded.max_abs_steer_correction > 0.05
    assert guarded.final_x > 3.0, guarded.summary()
    assert guarded.state_limit_violations == 0


def test_following_slower_vehicle_keeps_a_gap():
    params = VehicleParams()
    scenario = follow_lead(params)
    blind = _run(scenario, False, params)
    guarded = _run(scenario, True, params)
    assert blind.collided, blind.summary()
    assert not guarded.collided, guarded.summary()
    assert guarded.min_clearance_m > 0.0
    lead_x = 6.0 + 0.55 * guarded.steps[-1].t
    assert guarded.final_x < lead_x - 0.5
    assert guarded.final_x > 3.0, guarded.summary()
    assert guarded.cbf_active_count > 0


def test_unavoidable_collision_is_flagged_and_slower():
    params = VehicleParams()
    scenario = unavoidable_wall(params)
    blind = _run(scenario, False, params)
    guarded = _run(scenario, True, params)
    assert blind.collided and guarded.collided
    assert guarded.infeasible_count > 0, guarded.summary()
    assert guarded.impact_speed_mps is not None and blind.impact_speed_mps is not None
    assert guarded.impact_speed_mps < blind.impact_speed_mps - 0.15
    assert any(step.solver_status == "infeasible_fallback" for step in guarded.steps)


def test_open_floor_tracks_without_intervention():
    params = VehicleParams()
    scenario = open_floor(params)
    guarded = _run(scenario, True, params)
    assert not guarded.collided
    assert guarded.cbf_active_count == 0, guarded.summary()
    assert guarded.replan_count == 0
    assert guarded.infeasible_count == 0
    assert guarded.goal_progress > 0.9
    assert guarded.applied_limit_violations == 0
    assert guarded.state_limit_violations == 0


def test_aggressive_tracker_limits_are_corrected_in_the_loop():
    params = VehicleParams()
    result = simulate_scenario(
        "aggressive",
        VehicleState(x=0.0, z=1.0, yaw=0.0, speed=0.5, steering=0.0, t=0.0),
        (4.0, 1.0),
        AggressiveTracker(),
        build_empty_world(),
        dt=0.1,
        duration=0.6,
        use_cbf=True,
        params=params,
    )
    blind = simulate_scenario(
        "aggressive",
        VehicleState(x=0.0, z=1.0, yaw=0.0, speed=0.5, steering=0.0, t=0.0),
        (4.0, 1.0),
        AggressiveTracker(),
        build_empty_world(),
        dt=0.1,
        duration=0.6,
        use_cbf=False,
        params=params,
    )
    assert result.nominal_limit_violations > 0
    assert result.applied_limit_violations == 0
    assert result.state_limit_violations == 0
    assert blind.applied_limit_violations > 0
    assert blind.state_limit_violations == 0


def test_causal_velocity_ignores_stored_prediction():
    gt = []
    t = 0.0
    x = 0.0
    while t <= 4.0:
        gt.append(DynamicWaypoint(t=t, x=x, z=6.0, yaw=0.0, speed=1.0))
        x += 0.1
        t += 0.1
    # Oracle-looking forecast that dives into the rover's lane. Must be ignored.
    bad = [
        DynamicWaypoint(t=0.0, x=0.0, z=6.0, yaw=-math.pi / 2, speed=3.0),
        DynamicWaypoint(t=2.0, x=0.0, z=0.0, yaw=-math.pi / 2, speed=3.0),
    ]
    agent = DynamicObject(
        id="ghost",
        type="forklift",
        length=2.4,
        width=1.2,
        ground_truth=gt,
        prediction=bad,
        prediction_model="do_not_use",
    )
    scene = build_scene(0.0, dynamic_obstacles=[agent], dynamics_mode="causal_cv")
    assert len(scene.boxes) == 1
    assert abs(scene.boxes[0].vz) < 1e-6
    assert scene.boxes[0].vx == pytest_approx(1.0, abs=1e-6)

    state = VehicleState(x=0.0, z=0.0, yaw=0.0, speed=1.2, steering=0.0, t=0.0)
    safe, diag = filter_control(
        state, ControlInput(accel=0.4, steer_cmd=0.0), scene, dt=0.1
    )
    assert safe.accel > 0.0, diag.to_dict()
    assert not diag.infeasible


def test_facility_wall_is_visible_to_the_filter():
    facility = FacilitySpec(
        schema_version="1.0.0",
        facility_id="cbf-test",
        name="cbf-test",
        dimensions={"width_x": 30.0, "depth_z": 20.0, "height_y": 4.0},
        origin={"x": 0.0, "y": 0.0, "z": 0.0},
        walls=[WallSegment(id="gate", x1=6.0, z1=-3.0, x2=6.0, z2=3.0, thickness=0.3)],
        columns=[],
        zones=[],
    )
    state = VehicleState(x=3.5, z=0.0, yaw=0.0, speed=2.0, steering=0.0, t=0.0)
    scene = build_scene(0.0, facility=facility)
    assert any(box.kind == "wall" for box in scene.boxes)
    safe, diag = filter_control(
        state, ControlInput(accel=0.8, steer_cmd=0.0), scene, dt=0.1
    )
    assert safe.accel < 0.0, diag.to_dict()
    assert diag.cbf_active


def test_real_facility_scene_builds():
    facility = load_facility(ROOT_FACILITY)
    scene = build_scene(0.0, facility=facility)
    assert len(scene.circles) == len(facility.columns)
    assert len(scene.boxes) >= len(facility.walls)


def test_closed_loop_step_advances_bicycle_state():
    from northstar.control import closed_loop_step
    from northstar.control.tracking import TrajectoryTracker

    params = VehicleParams()
    tracker = TrajectoryTracker([(0.0, 0.0), (8.0, 0.0)], params=params, cruise_speed=1.0)
    state = VehicleState(x=0.0, z=0.0, yaw=0.0, speed=0.5, steering=0.0, t=0.0)
    wall = BoxObstacle(id="ahead", x=3.5, z=0.0, yaw=0.0, length=0.6, width=4.0)
    scene = build_scene(0.0, boxes=[wall])
    nxt, nominal, applied, diag = closed_loop_step(
        state, tracker, scene, dt=0.1, params=params, use_cbf=True
    )
    assert nxt.t == pytest_approx(0.1)
    assert diag is not None
    assert nominal.steer_cmd is not None and applied.steer_cmd is not None
    assert abs(nxt.x - state.x) > 0.0


def pytest_approx(value: float, abs: float = 1e-6):  # noqa: A002 - match pytest.approx keyword
    class _Cmp:
        def __eq__(self, other: object) -> bool:
            return isinstance(other, (int, float)) and math.isclose(float(other), value, abs_tol=abs)

        def __repr__(self) -> str:
            return f"approx({value} ± {abs})"

    return _Cmp()


def build_empty_world():
    from northstar.control.simulate import SimWorld

    return SimWorld()

