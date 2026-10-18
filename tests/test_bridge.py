"""The frontend contract, asserted on this side.

`frontend/src/types/sim.ts` is TypeScript, so nothing checks it at runtime
against what the backend emits. A field the UI does not declare, or one it
reads that is absent, shows up as a blank panel rather than an error. These
tests pin the shape and the enums, and one of them reads the .ts file so the
Python mirror cannot drift from it silently.
"""

import json
import re
from pathlib import Path

import pytest

from backend.ml.bridge import (
    CONTROLLER_MODES,
    SCHEMA_VERSION,
    EpisodeMetrics,
    controller_mode,
    obstacle_states,
    path_points,
    rover_state,
    state_frame,
    validate_frame,
)
from backend.ml.dataset import REAL_DATA_ROOT
from northstar.facility.loader import load_scenario
from northstar.vehicle.bicycle import VehicleState

SIM_TS = Path("frontend/src/types/sim.ts")
SCENARIO = Path("data/dataset_v2_additions/scenarios/v2_scn_00001.json")


class _Command:
    def __init__(self, accel, steer_cmd=None, steer_rate=0.0):
        self.accel = accel
        self.steer_cmd = steer_cmd
        self.steer_rate = steer_rate


class _Diagnostics:
    def __init__(self, **kwargs):
        self.intervened = kwargs.get("intervened", False)
        self.cbf_active = kwargs.get("cbf_active", False)
        self.infeasible = kwargs.get("infeasible", False)
        self.replan_required = kwargs.get("replan_required", False)
        self.min_clearance_m = kwargs.get("min_clearance_m", 1.0)
        self.solve_time_s = kwargs.get("solve_time_s", 0.004)


def _frame(**overrides) -> dict:
    state = VehicleState(x=10.0, z=4.0, yaw=0.5, speed=1.2, steering=0.1, t=2.0)
    base = dict(
        tick=7,
        sim_time=2.0,
        seed=0,
        state=state,
        goal_xz=(20.0, 9.0),
        obstacles=[],
        candidate_paths=[path_points([(1.0, 2.0), (2.0, 3.0)], dt=0.2)],
        selected_path=path_points([(1.0, 2.0), (2.0, 3.0)], dt=0.2),
        fallback_path=[],
        trail=path_points([(0.0, 0.0)]),
        controller={
            "mode": "tracking",
            "nominal": {"a": 0.4, "steer": 0.02},
            "actual": {"a": 0.4, "steer": 0.02},
            "barrier_min": 1.5,
            "latency_ms": 31.0,
            "source": "learned",
        },
        metrics=EpisodeMetrics().to_dict(),
    )
    base.update(overrides)
    return state_frame(**base)


def test_emitted_frame_satisfies_the_schema():
    assert validate_frame(_frame()) == []


def test_validator_rejects_an_undeclared_field():
    frame = _frame()
    frame["extra_panel"] = 1
    assert any("undeclared" in problem for problem in validate_frame(frame))


def test_validator_rejects_a_missing_field():
    frame = _frame()
    del frame["controller"]
    assert any("missing" in problem for problem in validate_frame(frame))


def test_validator_rejects_an_unknown_controller_mode():
    frame = _frame()
    frame["controller"]["mode"] = "cruising"
    assert any("mode" in problem for problem in validate_frame(frame))


@pytest.mark.skipif(not SIM_TS.exists(), reason="frontend not present")
def test_python_mirror_matches_the_typescript_schema():
    """Guard against drift: the .ts file is the source of truth.

    If the frontend adds a StateFrame field or a ControllerMode, this fails
    rather than letting the backend keep emitting the old shape.
    """
    source = SIM_TS.read_text(encoding="utf-8")

    version = re.search(r"SCHEMA_VERSION\s*=\s*(\d+)", source)
    assert version is not None
    assert int(version.group(1)) == SCHEMA_VERSION

    modes = re.search(r"ControllerMode\s*=\s*([^;]+);", source)
    assert modes is not None
    declared = set(re.findall(r"'([a-z_]+)'", modes.group(1)))
    assert declared == set(CONTROLLER_MODES)

    frame_block = re.search(r"StateFrame\s*=\s*\{(.*?)\n\};", source, re.S)
    assert frame_block is not None
    block = frame_block.group(1)
    required = set(re.findall(r"^\s*([a-z_]+):", block, re.M))
    optional = set(re.findall(r"^\s*([a-z_]+)\?:", block, re.M))
    assert required.isdisjoint(optional)
    assert required == set(_frame())
    full = _frame(
        flow=[
            {
                "t": 0.0,
                "index": 0,
                "selected": True,
                "points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}],
            }
        ],
        conditioning={"size": 2, "facility": [0, 1, 0, 1], "static": [1, 0, 0, 0]},
    )
    assert required | optional == set(full)
    assert validate_frame(full) == []


def test_rover_state_maps_dataset_xz_to_backend_xy():
    """Dataset is XZ-horizontal; the frontend schema is XY-horizontal, z up."""
    state = VehicleState(x=3.0, z=7.0, yaw=0.25, speed=0.0, steering=0.0, t=0.0)
    emitted = rover_state(state)
    assert emitted["x"] == pytest.approx(3.0)
    assert emitted["y"] == pytest.approx(7.0)
    assert emitted["z"] == pytest.approx(0.0)
    assert emitted["heading"] == pytest.approx(0.25)


def test_path_points_carry_timing_when_asked():
    points = path_points([(0.0, 0.0), (1.0, 2.0)], dt=0.2, t0=1.0)
    assert points[0] == {"x": 0.0, "y": 0.0, "t": 1.0}
    assert points[1]["t"] == pytest.approx(1.2)
    assert path_points([(0.0, 0.0)])[0] == {"x": 0.0, "y": 0.0}


def test_controller_mode_order_is_most_severe_first():
    assert controller_mode(_Diagnostics(intervened=True), True, False) == "goal"
    assert controller_mode(_Diagnostics(infeasible=True), False, True) == "estop"
    assert controller_mode(_Diagnostics(intervened=True), False, True) == "replanning"
    assert controller_mode(_Diagnostics(intervened=True), False, False) == "intervention"
    assert controller_mode(_Diagnostics(), False, False) == "tracking"
    assert controller_mode(None, False, False) == "tracking"


def test_path_efficiency_is_withheld_until_the_goal_is_reached():
    """A stalled run travels little while the gap stays large, which inverts
    the ratio: one measured 12.18 before this guard."""
    stalled = EpisodeMetrics(start_gap_m=22.0, travelled_m=1.8, goal_reached=False)
    assert stalled.path_efficiency() is None

    arrived = EpisodeMetrics(start_gap_m=22.0, travelled_m=24.0, goal_reached=True)
    assert arrived.path_efficiency() == pytest.approx(22.0 / 24.0)


@pytest.mark.skipif(not SCENARIO.exists(), reason="v2 corpus not present")
def test_obstacles_exclude_the_facility_and_use_predicted_poses():
    """The scene draws the hall from the facility spec, so emitting walls and
    columns here would render them twice. Dynamic poses come from the
    prediction track, never ground truth."""
    spec = load_scenario(SCENARIO)
    emitted = obstacle_states(
        list(spec.static_obstacles), list(spec.dynamic_obstacles), 0.0
    )
    assert len(emitted) == len(spec.static_obstacles) + len(spec.dynamic_obstacles)
    assert all(len(o["position"]) == 3 and len(o["size"]) == 3 for o in emitted)
    assert all(o["shape"] in ("box", "cylinder") for o in emitted)

    moving = [o for o in emitted if any(abs(v) > 1e-9 for v in o["velocity"])]
    assert moving, "a scenario with dynamic agents should report a velocity"


@pytest.mark.skipif(not SCENARIO.exists(), reason="v2 corpus not present")
def test_frames_are_json_serialisable():
    """The transport is JSON, so numpy scalars must not leak into a frame."""
    frame = _frame(obstacles=obstacle_states(
        list(load_scenario(SCENARIO).static_obstacles), [], 0.0
    ))
    round_tripped = json.loads(json.dumps(frame))
    assert validate_frame(round_tripped) == []
