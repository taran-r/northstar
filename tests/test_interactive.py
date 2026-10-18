"""Closed-loop checks for the interactive rover session.

These do not require a trained checkpoint. A scripted proposer stands in for
the network so the test can force a clean candidate or a colliding one, and
Hybrid A* is the real fallback.
"""

import numpy as np

from backend.ml.bridge import validate_frame
from backend.ml.inference import Candidate, CandidateDiagnostics, PlanResult
from backend.ml.interactive import InteractiveSim
from northstar.facility.occupancy import OccupancyGrid


class ScriptedPlanner:
    uses_dynamic = False
    schema = {"dt": 0.2}

    def __init__(self, mode: str = "forward"):
        self.mode = mode
        self.calls = 0

    def plan(self, **kwargs):
        self.calls += 1
        count = 12
        trajectory = np.zeros((count, 7), dtype=np.float32)
        if self.mode == "forward":
            trajectory[:, 0] = np.linspace(0.0, 5.0, count)
        else:
            trajectory[:, 0] = np.linspace(0.0, 0.5, count)
            trajectory[:, 1] = np.linspace(0.0, -3.2, count)
        trajectory[:, 3] = 1.0
        trajectory[:, 4] = 1.2
        candidate = Candidate(trajectory=trajectory, seed=0, index=0, dt=0.2)
        report = CandidateDiagnostics(
            index=0,
            max_speed=1.2,
            speed_ok=True,
            max_abs_accel=0.2,
            accel_ok=True,
            max_abs_steer=0.0,
            steer_ok=True,
            max_curvature=0.0,
            curvature_ok=True,
            path_length_m=5.0,
            goal_distance_m=1.0,
            heading_error_deg_mean=0.0,
            speed_mismatch_ms=0.0,
            dynamics_ok=True,
        )
        return PlanResult(
            candidates=[candidate],
            diagnostics=[report],
            latency_ms_total=1.0,
            latency_ms_per_candidate=1.0,
            diversity_m=0.0,
            uses_dynamic=False,
            solver="euler",
            solver_steps=1,
            seed=0,
            dt=0.2,
        )


def test_crop_local_matches_truncating_index():
    grid = OccupancyGrid(0.0, 0.0, 0.5, 8, 6)
    grid.data[1, 2] = 0.75
    crop = grid.crop_local(1.0, 0.5, 2.0)
    assert crop.data.shape[0] == crop.data.shape[1]
    wx, wz = crop.grid_to_world(0, 0)
    sgx = int((wx - grid.origin_x) / grid.resolution)
    sgz = int((wz - grid.origin_z) / grid.resolution)
    if grid.in_bounds(sgx, sgz):
        assert crop.data[0, 0] == grid.data[sgz, sgx]
    else:
        assert crop.data[0, 0] == 1.0


def test_forward_candidate_is_labelled_learned():
    planner = ScriptedPlanner("forward")
    sim = InteractiveSim(planner=planner, dynamics=[])
    assert sim.try_goal(14.0, 14.0)
    frame = sim.step()
    assert validate_frame(frame) == []
    assert frame["measured"] is True
    assert frame["controller"]["source"] == "learned"
    assert len(frame["selected_path"]) >= 2
    assert frame["fallback_paths"] == []
    assert planner.calls == 1
    assert frame["goal"] == [14.0, 14.0, 0.0]


def test_colliding_candidate_falls_back_to_hybrid_astar():
    planner = ScriptedPlanner("into_wall")
    sim = InteractiveSim(planner=planner, dynamics=[])
    assert sim.try_goal(16.0, 14.0)
    frame = sim.step()
    assert validate_frame(frame) == []
    assert frame["controller"]["source"] == "fallback"
    assert frame["selected_path"] == []
    assert len(frame["fallback_paths"]) >= 2
    assert planner.calls == 1


def test_barrier_is_not_driven_through():
    planner = ScriptedPlanner("forward")
    sim = InteractiveSim(planner=planner, dynamics=[])
    assert sim.try_spawn("barrier", 8.0, 14.0, yaw=np.pi / 2)
    assert sim.try_goal(20.0, 14.0)
    frame = None
    for _ in range(40):
        frame = sim.step()
    assert frame is not None
    assert validate_frame(frame) == []
    assert frame["metrics"]["collisions"] == 0
    assert frame["metrics"]["min_clearance"] > -0.05
    assert frame["metrics"]["replans"] >= 1
    # The barrier sits on the straight lane, so a rover that moved has gone
    # around it or stopped short of it. Sitting on the start pose is a miss.
    moved = abs(frame["rover"]["x"] - 4.0) + abs(frame["rover"]["y"] - 14.0)
    assert moved > 0.4
    assert frame["rover"]["x"] < 7.6 or abs(frame["rover"]["y"] - 14.0) > 0.7


def test_spawned_obstacle_triggers_another_plan():
    planner = ScriptedPlanner("forward")
    sim = InteractiveSim(planner=planner, dynamics=[])
    assert sim.try_goal(18.0, 14.0)
    sim.step()
    calls = planner.calls
    assert sim.try_spawn("pallet", 10.0, 18.0, yaw=0.0)
    sim.step()
    assert planner.calls > calls


def test_forklift_moves_and_rover_does_not_start_inside_it():
    sim = InteractiveSim(planner=None, dynamics=None)
    first = sim.step()
    ids = {item["id"]: item for item in first["obstacles"]}
    assert "forklift_lane" in ids
    for _ in range(15):
        frame = sim.step()
    later = {item["id"]: item for item in frame["obstacles"]}
    moved = abs(later["forklift_lane"]["position"][0] - ids["forklift_lane"]["position"][0])
    assert moved > 0.2
    assert frame["metrics"]["collisions"] == 0


def test_hybrid_fallback_reaches_an_open_goal():
    sim = InteractiveSim(planner=None, dynamics=[])
    assert sim.try_goal(12.0, 14.0)
    frame = None
    for _ in range(70):
        frame = sim.step()
        if frame["metrics"]["goal_reached"]:
            break
    assert frame is not None
    assert frame["metrics"]["goal_reached"] is True
    assert frame["metrics"]["collisions"] == 0
    assert frame["controller"]["source"] == "fallback"
    assert frame["metrics"]["min_clearance"] > 0.5


def test_goal_inside_a_wall_is_rejected():
    sim = InteractiveSim(planner=None, dynamics=[])
    assert sim.try_goal(0.2, 0.2) is False
    frame = sim.step()
    assert frame["goal"] is None


def test_hall_crossing_does_not_run_the_rover_over():
    """The aisle forklift must stay in the filter and never produce an estop hit."""
    from backend.ml.hall import scene_to_dataset

    sim = InteractiveSim(planner=None, dynamics=None, layout="hall", async_plan=False)
    assert sim.try_goal(*scene_to_dataset(0.0, 70.0))
    frame = None
    for _ in range(220):
        frame = sim.step()
        assert frame["controller"]["mode"] != "estop"
        assert frame["metrics"]["collisions"] == 0
        h = frame["controller"]["barrier_min"]
        if h is not None:
            assert h > -0.05
    assert frame is not None
    assert frame["metrics"]["min_clearance"] > -0.05
    assert frame["rover"]["y"] < -40.0


def test_placed_forklift_patrols_and_rover_is_not_pinned():
    """A drawn forklift must move. A static body in the lane is what froze the run."""
    import math

    from backend.ml.hall import scene_to_dataset

    sim = InteractiveSim(planner=None, dynamics=[], layout="hall", async_plan=False)
    # Backend heading −π/2 faces down the aisle (same as the setup conversion).
    assert sim.place_obstacle("forklift", *scene_to_dataset(0.0, 36.0), -math.pi / 2)
    assert sim.try_goal(*scene_to_dataset(0.0, 70.0))
    first = sim.step()
    forks = [o for o in first["obstacles"] if o["kind"] == "forklift"]
    assert len(forks) == 1
    start = forks[0]["position"][:2]
    moved = False
    stuck = 0
    frame = first
    for _ in range(160):
        frame = sim.step()
        assert frame["controller"]["mode"] != "estop"
        assert frame["metrics"]["collisions"] == 0
        pose = next(o for o in frame["obstacles"] if o["kind"] == "forklift")["position"]
        if math.hypot(pose[0] - start[0], pose[1] - start[1]) > 1.0:
            moved = True
        if frame["rover"]["speed"] < 0.05:
            stuck += 1
        else:
            stuck = 0
        assert stuck < 45
    assert moved
    assert frame["rover"]["y"] < -20.0


def test_hall_matches_the_drawn_aisle_and_moves():
    from backend.ml.hall import scene_to_dataset

    assert scene_to_dataset(-1.5, 8) == (-1.5, -8.0)
    sim = InteractiveSim(planner=None, dynamics=[], layout="hall")
    # A rack footprint from the visual hall is solid. The aisle centre is not.
    assert sim.base_static.is_occupied(-12.6, -14.0)
    assert not sim.base_static.is_occupied(0.0, -40.0)
    assert sim.try_goal(-1.0, -22.0)
    frame = None
    for _ in range(40):
        frame = sim.step()
    assert frame is not None
    assert frame["metrics"]["collisions"] == 0
    assert frame["rover"]["y"] < -8.5
    assert frame["controller"]["source"] == "fallback"
    validate_frame(frame)


def test_hall_forklift_box_is_long_along_its_heading():
    """A placed forklift must not become a wall across the aisle.

    The mesh is long along the forks. Swapping length/width left an invisible
    slab that the rover entered while the drawn forklift still looked clear;
    the filter then reported h ≈ −0.9 m and emergency-stopped.
    """
    import math

    from backend.ml.interactive import HALL_SIZES
    from northstar.control.clearance import oriented_box_clearance
    from northstar.geometry import OrientedBox
    from northstar.vehicle.bicycle import VehicleParams, VehicleState

    length, width, _height = HALL_SIZES["forklift"]
    assert length > width
    # Scene yaw from seed 427, converted the same way as RunScreen.
    scene_yaw = 1.3375
    backend_yaw = scene_yaw - math.pi / 2
    box = OrientedBox(5.1, -21.5, backend_yaw, length, width)
    params = VehicleParams()
    # Just beside the forks: the swapped box already penetrates, the corrected
    # one still has a gap. Deeper toward the body both are inside — that is a
    # real overlap, not the phantom wall.
    beside = VehicleState(x=1.5, z=-21.5, yaw=-math.pi / 2, speed=0.0)
    fixed, _ = oriented_box_clearance(beside.footprint(params), box)
    legacy = OrientedBox(5.1, -21.5, scene_yaw, 2.97, 6.14)
    bad, _ = oriented_box_clearance(beside.footprint(params), legacy)
    assert bad < 0.0
    assert fixed > 0.15
    assert fixed > bad + 0.3


def test_closest_scene_keeps_the_face_that_is_already_close():
    from backend.ml.interactive import _closest_scene
    from northstar.control.scene import BoxObstacle, CircleObstacle, SafetyScene
    from northstar.vehicle.bicycle import VehicleState

    state = VehicleState(x=0.0, z=0.0, yaw=0.0, speed=0.0)
    # Six small centres nearby, plus one long box whose face is closer than any centre.
    boxes = [
        BoxObstacle(id=f"near_{i}", x=1.2, z=-1.5 + 0.5 * i, yaw=0.0, length=0.6, width=0.6)
        for i in range(6)
    ]
    boxes.append(
        BoxObstacle(id="long_face", x=4.0, z=0.0, yaw=0.0, length=6.0, width=1.0)
    )
    scene = SafetyScene(t=0.0, boxes=boxes, circles=[
        CircleObstacle(id="col", x=-8.0, z=0.0, radius=0.55),
    ])
    kept = _closest_scene(scene, state, limit=6)
    assert any(obs.id == "long_face" for obs in kept.obstacles())


def test_closest_scene_always_keeps_a_moving_agent():
    from backend.ml.interactive import _closest_scene
    from northstar.control.scene import BoxObstacle, SafetyScene
    from northstar.vehicle.bicycle import VehicleState

    state = VehicleState(x=0.0, z=0.0, yaw=-1.57, speed=1.5)
    boxes = [
        BoxObstacle(id=f"near_{i}", x=0.2, z=-1.0 - 0.4 * i, yaw=0.0, length=0.5, width=0.5)
        for i in range(8)
    ]
    boxes.append(
        BoxObstacle(
            id="hall-crossing",
            x=6.0,
            z=-4.0,
            yaw=3.14,
            length=4.4,
            width=2.0,
            vx=-1.5,
            vz=0.0,
        )
    )
    scene = SafetyScene(t=0.0, boxes=boxes, circles=[])
    kept = _closest_scene(scene, state, limit=6)
    assert any(obs.id == "hall-crossing" for obs in kept.obstacles())


def test_hall_pallet_is_solid_and_the_fallback_stays_local():
    sim = InteractiveSim(planner=None, dynamics=[], layout="hall")
    assert sim.place_obstacle("pallet", -1.5, -18.0, 0.0)
    assert sim.try_goal(-1.5, -40.0)
    poses = []
    frame = None
    for _ in range(80):
        frame = sim.step()
        poses.append((frame["rover"]["x"], frame["rover"]["y"]))
    assert frame is not None
    assert validate_frame(frame) == []
    pallet = next(item for item in frame["obstacles"] if item["kind"] == "pallet")
    assert pallet["size"][0] > 1.8
    # Just inside the padded stack along its width.
    assert sim.base_static.is_occupied(-1.5, -18.7)
    assert frame["metrics"]["collisions"] == 0
    assert frame["metrics"]["min_clearance"] > -0.05
    half_x, half_z = pallet["size"][0] / 2, pallet["size"][1] / 2
    for x, z in poses:
        assert abs(x - (-1.5)) >= half_x or abs(z - (-18.0)) >= half_z
    route = frame["fallback_paths"]
    if len(route) >= 2:
        length = sum(
            ((route[i + 1]["x"] - route[i]["x"]) ** 2 + (route[i + 1]["y"] - route[i]["y"]) ** 2)
            ** 0.5
            for i in range(len(route) - 1)
        )
        assert length <= 11.0


def test_fallback_detour_hands_back_to_the_network():
    planner = ScriptedPlanner("into_wall")
    sim = InteractiveSim(planner=planner, dynamics=[])
    assert sim.try_goal(22.0, 14.0)
    first = sim.step()
    assert first["controller"]["source"] == "fallback"
    route = first["fallback_paths"]
    assert len(route) >= 2
    length = sum(
        ((route[i + 1]["x"] - route[i]["x"]) ** 2 + (route[i + 1]["y"] - route[i]["y"]) ** 2) ** 0.5
        for i in range(len(route) - 1)
    )
    assert length <= 11.0
    planner.mode = "forward"
    frame = first
    for _ in range(160):
        frame = sim.step()
        if frame["controller"]["source"] == "learned" and planner.calls >= 2:
            break
    assert frame["controller"]["source"] == "learned"
    assert planner.calls >= 2
    assert validate_frame(frame) == []
    conditioning = frame["conditioning"]
    assert conditioning["size"] == 20
    assert len(conditioning["facility"]) == 400
    assert set(conditioning["static"]).issubset({0, 1})


def test_a_correction_does_not_flicker_into_watching_or_replan():
    from backend.ml.interactive import _latch_mode

    shown, until = _latch_mode("tracking", 0.0, "intervention", 0.0)
    assert shown == "intervention"
    assert _latch_mode(shown, until, "tracking", 0.2)[0] == "intervention"
    assert _latch_mode(shown, until, "replanning", 0.4)[0] == "intervention"
    assert _latch_mode(shown, until, "tracking", until + 0.1)[0] == "tracking"
    assert _latch_mode(shown, until, "goal", 0.2)[0] == "goal"


def test_a_stopped_rover_creeps_when_the_rollout_allows_it():
    """The program is linearized at full throttle, so a stopped rover only got the brake.

    At zero speed that brake also freezes the steering. A crawl the backup
    rollout accepts has to be kept, and it is not a reason to replan.
    """
    from northstar.control.cbf import filter_control
    from northstar.control.scene import BoxObstacle, SafetyScene
    from northstar.vehicle.bicycle import ControlInput, VehicleState

    from backend.ml.interactive import HALL_CBF

    # Tight ribbon that forces the crawl path (not hall mesh pads).
    boxes = [
        BoxObstacle(id="barrier", x=0.4, z=-14.0, yaw=0.15, length=4.56, width=2.11),
        BoxObstacle(id="crates", x=-4.4, z=-13.0, yaw=0.0, length=3.48, width=3.24),
        BoxObstacle(id="pallet", x=2.4, z=-12.4, yaw=0.0, length=2.69, width=2.40),
        BoxObstacle(id="rear", x=-1.8, z=-9.2, yaw=1.1, length=2.97, width=6.14),
    ]
    state = VehicleState(x=-1.468, z=-12.38, yaw=-1.406, speed=0.0, steering=0.51, t=2.0)
    scene = SafetyScene(t=2.0, boxes=boxes, circles=[])
    safe, diag = filter_control(
        state,
        ControlInput(accel=1.5, steer_cmd=0.55),
        scene,
        0.2,
        config=HALL_CBF,
    )
    # Either the crawl path or a QP that keeps a little forward motion and
    # the commanded steering — not a hard brake that freezes yaw at v=0.
    assert safe.accel >= 0.0
    assert safe.steer_cmd > 0.3
    assert diag.solver_status in ("creep", "qp_feasible", "line_search")
    assert diag.min_clearance_m > -0.05
    assert not diag.infeasible


def test_creep_does_not_enter_a_wall():
    from northstar.control.cbf import filter_control
    from northstar.control.scene import BoxObstacle, SafetyScene
    from northstar.vehicle.bicycle import ControlInput, VehicleState

    from backend.ml.interactive import HALL_CBF

    # Bumper is about 0.65 m ahead. This face sits just beyond it, so a crawl
    # reaches the wall and the brake has to remain.
    wall = BoxObstacle(id="wall", x=1.05, z=0.0, yaw=0.0, length=0.4, width=6.0)
    state = VehicleState(x=0.0, z=0.0, yaw=0.0, speed=0.15, steering=0.0, t=0.0)
    scene = SafetyScene(t=0.0, boxes=[wall], circles=[])
    safe, diag = filter_control(
        state,
        ControlInput(accel=1.5, steer_cmd=0.0),
        scene,
        0.2,
        config=HALL_CBF,
    )
    assert safe.accel <= 0.05
    assert diag.min_clearance_m > -0.05


def test_braking_stays_a_correction_when_a_replan_is_also_requested():
    from types import SimpleNamespace

    from backend.ml.interactive import _display_mode

    braking = SimpleNamespace(
        intervened=True, infeasible=False, replan_required=True, min_clearance_m=0.2
    )
    assert _display_mode(braking, False, planning=True) == "intervention"
    assert _display_mode(braking, False, planning=False) == "intervention"
    waiting = SimpleNamespace(
        intervened=False, infeasible=False, replan_required=True, min_clearance_m=0.5
    )
    assert _display_mode(waiting, False, planning=True) == "replanning"
    # Infeasible is a hold / replan, never an emergency stop — even with
    # negative clearance. That label froze the hall demo inside a soft pad.
    holding = SimpleNamespace(
        intervened=True, infeasible=True, replan_required=True, min_clearance_m=0.1
    )
    assert _display_mode(holding, False, planning=False) == "intervention"
    stopped = SimpleNamespace(
        intervened=True, infeasible=True, replan_required=True, min_clearance_m=-0.2
    )
    assert _display_mode(stopped, False, planning=False) == "intervention"
    assert _display_mode(stopped, False, planning=True) == "replanning"
    assert _display_mode(stopped, False, planning=False) != "estop"


def test_hall_barrier_is_passed_without_emergency_stop():
    """The striped guard is thin. An oversized collision box made the filter estop."""
    from backend.ml.hall import scene_to_dataset
    from backend.ml.interactive import HALL_SIZES

    length, width, _ = HALL_SIZES["barrier"]
    assert width < 1.2
    assert length > width
    sim = InteractiveSim(planner=None, dynamics=[], layout="hall", async_plan=False)
    assert sim.place_obstacle("barrier", *scene_to_dataset(-0.5, 20.0), 0.0)
    assert sim.try_goal(*scene_to_dataset(2.0, 45.0))
    frame = None
    for _ in range(100):
        frame = sim.step()
        assert frame["controller"]["mode"] != "estop"
        assert frame["metrics"]["collisions"] == 0
    assert frame is not None
    assert frame["rover"]["y"] < -30.0
    assert frame["metrics"]["min_clearance"] > -0.05


def test_hall_soft_overlap_never_estops_and_requests_a_new_route():
    """A pad that overlaps the body must not freeze the run on Emergency stop."""
    from backend.ml.hall import HALL_START_X, HALL_START_Z, scene_to_dataset

    sim = InteractiveSim(planner=None, dynamics=[], layout="hall", async_plan=False)
    assert sim.place_obstacle("crates", HALL_START_X + 0.4, HALL_START_Z + 0.2, 0.0)
    assert sim.try_goal(*scene_to_dataset(2.0, 45.0))
    modes = set()
    saw_recovery = False
    t0 = None
    frame = None
    for _ in range(40):
        frame = sim.step()
        assert frame["controller"]["mode"] != "estop"
        modes.add(frame["controller"]["mode"])
        if sim._recovery_replan or sim.replan_hold_s > 0.0:
            saw_recovery = True
        if t0 is None:
            t0 = frame["sim_time"]
    assert frame is not None
    assert frame["sim_time"] > t0
    assert saw_recovery or "replanning" in modes or "intervention" in modes


def test_no_network_still_labels_hybrid_fallback():
    sim = InteractiveSim(planner=None, dynamics=[])
    assert sim.try_goal(12.0, 14.0)
    frame = sim.step()
    assert frame["controller"]["source"] == "fallback"
    assert frame["candidate_paths"] == []
    assert len(frame["fallback_paths"]) >= 2
