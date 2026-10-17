"""Checks for v2 generation helpers. Does not read or write dataset_v1 samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from northstar.dataset.combined import CombinedTrainDataset
from northstar.obstacles.route_events import crossing_agent
from northstar.pipeline.expand import _refuse_v1
from northstar.scenario.response import motion_response, route_conflict
from northstar.trajectory.windowing import subgoal_audit_fields
from northstar.vehicle.bicycle import VehicleState


def test_subgoal_expert_label_is_distinguished_from_inference_goal():
    fields = subgoal_audit_fields(
        {"x": 1.0, "z": 2.0, "yaw": 0.3, "is_local": 1.0},
        {"x": 9.0, "z": 8.0, "yaw": 0.1},
    )
    assert fields["expert_subgoal_is_label"] is True
    assert fields["subgoal_source"] == "expert_window_endpoint"
    assert fields["inference_goal"]["x"] == 9.0
    assert fields["inference_goal"]["source"] == "scenario_goal"


def test_motion_response_counts_midroute_stop_not_goal_stop():
    states = []
    for i in range(20):
        speed = 0.0 if 8 <= i <= 12 else 1.5
        states.append({"x": float(i), "z": 0.0, "speed": speed, "t": i * 0.2})
    # Goal is the last point, so the early stop is mid-route.
    out = motion_response(states, goal_x=19.0, goal_z=0.0)
    assert out["yielded"] is True
    assert out["longest_stop_s"] >= 0.6


def test_crossing_agent_conflicts_with_the_route():
    path = [
        VehicleState(x=0.4 * i, z=12.0, yaw=0.0, speed=1.4, t=0.2 * i) for i in range(50)
    ]
    agent = crossing_agent(path, np.random.default_rng(1), fraction=0.5, sudden_change=False)
    predicted = route_conflict(path, [agent], use_prediction=True)
    assert predicted["conflict"] is True
    assert predicted["min_separation_m"] < 2.0


def test_generator_refuses_to_write_dataset_v1():
    v1 = Path("data/dataset_v1")
    with pytest.raises(SystemExit):
        _refuse_v1(v1, v1)


def test_combined_loader_rejects_held_out_splits():
    with pytest.raises(ValueError):
        CombinedTrainDataset(["data/dataset_v1"], split="test")
