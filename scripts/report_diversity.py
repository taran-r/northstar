#!/usr/bin/env python
"""Recompute obstacle mix and realized interactions from saved scenarios.

Behavior tags are ignored. Stopping, braking, and route deviation are measured
from trajectories. Route conflict uses ground-truth dynamic poses.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from northstar.facility.schema import DynamicObject, DynamicWaypoint
from northstar.scenario.response import motion_response, route_conflict
from northstar.vehicle.bicycle import VehicleParams


def _dyn(raw: dict) -> DynamicObject:
    return DynamicObject(
        id=raw["id"],
        type=raw["type"],
        length=raw["length"],
        width=raw["width"],
        height=raw.get("height", 1.5),
        behavior=raw.get("behavior", ""),
        ground_truth=[DynamicWaypoint(**w) for w in raw.get("ground_truth", [])],
        prediction=[DynamicWaypoint(**w) for w in raw.get("prediction", [])],
        prediction_model=raw.get("prediction_model", "constant_velocity"),
        metadata=raw.get("metadata", {}),
    )


def report(root: Path) -> dict:
    scenarios = sorted((root / "scenarios").glob("*.json"))
    mix = Counter()
    layouts = Counter()
    realized = Counter()
    endpoint_cells = set()
    for path in scenarios:
        data = json.loads(path.read_text(encoding="utf-8"))
        ns = len(data.get("static_obstacles") or [])
        nd = len(data.get("dynamic_obstacles") or [])
        if ns and nd:
            mix["mixed"] += 1
        elif nd:
            mix["dynamic_only"] += 1
        elif ns:
            mix["static_only"] += 1
        else:
            mix["empty"] += 1
        layouts[data.get("metadata", {}).get("layout", "untagged")] += 1
        sx, sz = data["start"]["x"], data["start"]["z"]
        gx, gz = data["goal"]["x"], data["goal"]["z"]
        endpoint_cells.add((round(sx / 4.0), round(sz / 4.0), round(gx / 4.0), round(gz / 4.0)))

    traj_dir = root / "trajectories"
    n_traj = 0
    measured = Counter()
    planners = Counter()
    for path in sorted(traj_dir.glob("*.json")):
        traj = json.loads(path.read_text(encoding="utf-8"))
        n_traj += 1
        planners[traj.get("metadata", {}).get("planner", "?")] += 1
        states = traj["states"]
        if not states:
            continue
        goal = states[-1]
        motion = motion_response(states, goal["x"], goal["z"])
        scen_path = root / "scenarios" / f"{traj['scenario_id']}.json"
        dynamics = []
        if scen_path.exists():
            scen = json.loads(scen_path.read_text(encoding="utf-8"))
            dynamics = [_dyn(d) for d in scen.get("dynamic_obstacles") or []]
            goal = {"x": scen["goal"]["x"], "z": scen["goal"]["z"]}
            motion = motion_response(states, goal["x"], goal["z"])
        conflict = route_conflict(states, dynamics, VehicleParams(), use_prediction=False)
        if motion["yielded"]:
            measured["yielded"] += 1
        if motion["braked"] and (conflict["conflict"] or conflict["min_separation_m"] < 3.5):
            measured["braked_near_dynamic"] += 1
        if conflict["conflict"]:
            measured["gt_route_conflict"] += 1
        stored = traj.get("metadata", {}).get("realized", {})
        if stored.get("detoured"):
            measured["detoured"] += 1
        if stored.get("meaningful") or motion["yielded"] or (
            motion["braked"] and conflict["conflict"]
        ):
            measured["meaningful"] += 1

    n_scen = len(scenarios)
    windows = list((root / "samples").glob("*.npz")) if (root / "samples").exists() else []
    splits = {}
    leaked = None
    if (root / "splits" / "train.json").exists():
        sets = {}
        for split in ("train", "val", "test"):
            names = json.loads((root / "splits" / f"{split}.json").read_text(encoding="utf-8"))
            sets[split] = {n.split("_m")[0] for n in names}
            splits[split] = {"windows": len(names), "scenarios": len(sets[split])}
        leaked = {
            "train_val": len(sets["train"] & sets["val"]),
            "train_test": len(sets["train"] & sets["test"]),
            "val_test": len(sets["val"] & sets["test"]),
        }
    return {
        "root": str(root).replace("\\", "/"),
        "scenarios": n_scen,
        "trajectories": n_traj,
        "windows": len(windows),
        "mix": dict(mix),
        "layouts": dict(layouts),
        "endpoint_cells_4m": len(endpoint_cells),
        "measured_trajectories": dict(measured),
        "meaningful_scenario_fraction": (measured["meaningful"] / n_traj) if n_traj else 0.0,
        "planners": dict(planners),
        "splits": splits,
        "scenario_split_leaks": leaked,
        "yield_rate": (measured["yielded"] / n_traj) if n_traj else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/dataset_v2_additions")
    args = p.parse_args()
    print(json.dumps(report(Path(args.root)), indent=2))


if __name__ == "__main__":
    main()
