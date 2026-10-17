"""Resumable generation of dataset_v2_additions. Never writes data/dataset_v1."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from northstar.facility.loader import load_facility, save_scenario
from northstar.facility.occupancy import build_facility_occupancy
from northstar.facility.schema import FACILITY_SCHEMA_VERSION, ScenarioSpec
from northstar.obstacles.route_events import (
    barrier_on_path,
    crossing_agent,
    lead_then_exit_agent,
    oncoming_agent,
    stopping_crosser,
)
from northstar.obstacles.static_gen import generate_static_obstacles
from northstar.pipeline.expert import _accel_from_states, plan_expert_trajectory, plan_static_path
from northstar.pipeline.generate import _save_sample
from northstar.pipeline.strict_validate import validate_trajectory
from northstar.planner.timed_follow import yield_on_vertices
from northstar.scenario.generator import ScenarioGenerator
from northstar.scenario.response import motion_response, polyline_deviation, route_conflict
from northstar.trajectory.format import TrajectoryRecord, resample_states
from northstar.trajectory.windowing import extract_windows
from northstar.vehicle.bicycle import BicycleModel, VehicleParams, VehicleState


V2_SEED0 = 8_000_000
SEED_STRIDE = 7919
# At least half of attempts are ordinary static routes from the existing expert.
LAYOUT_CYCLE = (
    ["ordinary"] * 12
    + ["yield_cross"] * 3
    + ["detour"] * 2
    + ["brake_oncoming"] * 2
    + ["mixed"] * 1
)


@dataclass
class ExpandConfig:
    out_dir: str = "data/dataset_v2_additions"
    v1_dir: str = "data/dataset_v1"
    n_scenarios: int = 1300
    seed0: int = V2_SEED0
    dt: float = 0.2
    horizon: int = 32
    window_stride: int = 4
    map_size_m: float = 20.0
    map_resolution: float = 0.25
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    facility_path: str = "facilities/industrial_facility_v1.json"
    max_attempts_factor: int = 12
    oracle_fraction: float = 0.1
    use_time_aware: bool = True


def _refuse_v1(out: Path, v1: Path) -> None:
    out_r, v1_r = out.resolve(), v1.resolve()
    if out_r == v1_r or v1_r in out_r.parents or out_r in v1_r.parents:
        raise SystemExit(f"Refusing to write into or over {v1_r}")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))


def _load_v1_seeds(v1: Path) -> set[int]:
    seeds: set[int] = set()
    scen = v1 / "scenarios"
    if not scen.exists():
        return seeds
    for path in scen.glob("*.json"):
        # Avoid parsing full dynamic trajectories.
        for line in path.read_text(encoding="utf-8").splitlines():
            if '"seed"' in line:
                digits = "".join(ch for ch in line if ch.isdigit())
                if digits:
                    seeds.add(int(digits))
                break
    return seeds


def _scan_v2(out: Path) -> dict:
    scenarios = []
    seeds = set()
    samples: list[str] = []
    if (out / "scenarios").exists():
        for path in sorted((out / "scenarios").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            scenarios.append(data)
            seeds.add(int(data["seed"]))
    if (out / "samples").exists():
        samples = sorted(p.name for p in (out / "samples").glob("*.npz"))
    return {"scenarios": scenarios, "seeds": seeds, "samples": samples}


def _scenario_id(index: int) -> str:
    return f"v2_scn_{index:05d}"


def _behavior_tag(layout: str, realized: dict) -> str:
    if realized.get("detoured"):
        return "detour_blockage"
    if realized.get("yielded") and layout in {"yield_cross", "mixed", "detour"}:
        return "yield_intersection"
    if realized.get("yielded"):
        return "wait_traffic"
    if realized.get("braked") and layout == "follow_lead":
        return "follow_vehicle"
    if realized.get("braked"):
        return "brake_for_obstacle"
    if layout == "ordinary":
        return "straight"
    return "static_avoid"


def _endpoints_close(a: dict, b: dict, tol: float = 1.25) -> bool:
    return (
        math.hypot(a["start"]["x"] - b["start"]["x"], a["start"]["z"] - b["start"]["z"]) < tol
        and math.hypot(a["goal"]["x"] - b["goal"]["x"], a["goal"]["z"] - b["goal"]["z"]) < tol
        and a.get("metadata", {}).get("layout") == b.get("metadata", {}).get("layout")
    )


def _make_record(
    scenario: ScenarioSpec,
    states: list[VehicleState],
    accels: list[float],
    *,
    mode_id: int,
    planner: str,
    realized: dict,
) -> TrajectoryRecord:
    progress = math.hypot(states[-1].x - states[0].x, states[-1].z - states[0].z)
    speeds = [s.speed for s in states]
    return TrajectoryRecord(
        trajectory_id=f"{scenario.scenario_id}_m{mode_id}",
        scenario_id=scenario.scenario_id,
        dt=0.2,
        states=states,
        accel_cmds=accels,
        mode_id=mode_id,
        metadata={
            "planner": planner,
            "n_states": len(states),
            "duration": states[-1].t - states[0].t,
            "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            "max_speed": float(np.max(speeds)) if speeds else 0.0,
            "path_length": progress,
            "reached_near_goal": math.hypot(scenario.goal.x - states[-1].x, scenario.goal.z - states[-1].z)
            <= 2.5,
            "realized": realized,
        },
    )


def _drive(
    facility,
    scenario: ScenarioSpec,
    path: list[VehicleState],
    params: VehicleParams,
    *,
    use_prediction: bool,
    prefer_wait: bool,
    dt: float,
) -> Optional[list[VehicleState]]:
    from northstar.collision import CollisionChecker

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
    return yield_on_vertices(
        path,
        collision,
        scenario.dynamic_obstacles,
        dt=dt,
        use_prediction=use_prediction,
        max_wait=12.0 if prefer_wait else 6.0,
        cruise_speed=params.max_speed,
    )


def _accept(
    facility,
    scenario: ScenarioSpec,
    rolled: Optional[list[VehicleState]],
    reference: list[VehicleState],
    params: VehicleParams,
    *,
    layout: str,
    planner: str,
    mode_id: int,
    dt: float,
) -> tuple[Optional[TrajectoryRecord], str, dict]:
    if rolled is None:
        return None, "timed_follow_failed", {}
    if math.hypot(rolled[-1].x - scenario.goal.x, rolled[-1].z - scenario.goal.z) > 3.5:
        return None, "missed_goal", {}
    states, accels = resample_states(rolled, _accel_from_states(rolled), dt=dt)
    ok, reason = validate_trajectory(facility, scenario, states, accels, params)
    if not ok:
        return None, reason, {}
    progress = math.hypot(states[-1].x - states[0].x, states[-1].z - states[0].z)
    if progress < 4.0 or float(np.mean([s.speed for s in states])) < 0.2:
        return None, "insufficient_progress", {}
    motion = motion_response(states, scenario.goal.x, scenario.goal.z)
    predicted = route_conflict(reference, scenario.dynamic_obstacles, params, use_prediction=True)
    observed = route_conflict(states, scenario.dynamic_obstacles, params, use_prediction=False)
    deviation = polyline_deviation(reference, states)
    conflict = bool(predicted["conflict"] or observed["conflict"])
    close = min(float(predicted["min_separation_m"]), float(observed["min_separation_m"])) < 3.5
    realized = {
        "layout": layout,
        "yielded": bool(motion["yielded"] and (conflict or close)),
        "braked": bool(motion["braked"] and (conflict or close)),
        "detoured": bool(deviation >= 1.25),
        "midroute_stop_s": motion["midroute_stop_s"],
        "longest_stop_s": motion["longest_stop_s"],
        "predicted_route_conflict": bool(predicted["conflict"]),
        "observed_gt_conflict": bool(observed["conflict"]),
        "min_separation_m": float(min(predicted["min_separation_m"], observed["min_separation_m"])),
        "path_deviation_m": float(deviation),
        "oracle_dynamics": bool(scenario.oracle_dynamics),
    }
    realized["meaningful"] = bool(
        realized["yielded"] or realized["braked"] or realized["detoured"]
    )
    tag = _behavior_tag(layout, realized)
    scenario.metadata["behavior_tag"] = tag
    scenario.metadata["realized"] = realized
    record = _make_record(scenario, states, accels, mode_id=mode_id, planner=planner, realized=realized)
    record.metadata["behavior_tag"] = tag
    return record, "ok", realized


def _try_time_aware(facility, scenario, params, dt: float) -> Optional[list[VehicleState]]:
    from northstar.collision import CollisionChecker
    from northstar.planner.time_aware import TimeAwareConfig, TimeAwarePlanner

    occ = build_facility_occupancy(facility, resolution=0.35, static_obstacles=scenario.static_obstacles)
    collision = CollisionChecker(
        facility,
        static_obstacles=scenario.static_obstacles,
        occupancy=occ,
        vehicle_params=params,
    )
    collision.fast_occupancy = True
    start = VehicleState(
        x=scenario.start.x, z=scenario.start.z, yaw=scenario.start.yaw, speed=0.0, t=0.0
    )
    goal = VehicleState(x=scenario.goal.x, z=scenario.goal.z, yaw=scenario.goal.yaw, speed=0.0)
    cfg = TimeAwareConfig(
        xy_resolution=1.0,
        yaw_resolution=math.radians(30),
        time_resolution=0.8,
        max_time=30.0,
        max_iterations=4_000,
        speeds=(0.0, 1.2, 2.0),
        steer_angles=(-0.5, 0.0, 0.5),
        goal_xy_tol=1.6,
    )
    path = TimeAwarePlanner(collision, model=BicycleModel(params), config=cfg).plan(
        start,
        goal,
        scenario.dynamic_obstacles,
        use_oracle=scenario.oracle_dynamics,
    )
    if path is None:
        return None
    if math.hypot(path[-1].x - goal.x, path[-1].z - goal.z) > 2.5:
        return None
    return path


def _interaction_attempt(facility, gen, scenario, layout, rng, params, cfg: ExpandConfig, reference):
    """Attach route-coupled obstacles and return (records, reason)."""
    scenario.dynamic_obstacles = []
    oracle = layout != "ordinary" and float(rng.random()) < cfg.oracle_fraction
    if layout == "yield_cross":
        if oracle:
            scenario.dynamic_obstacles = [stopping_crosser(reference, rng)]
            scenario.oracle_dynamics = True
        else:
            scenario.dynamic_obstacles = [
                crossing_agent(
                    reference,
                    rng,
                    obj_type=str(rng.choice(["forklift", "agv", "worker", "industrial_cart"])),
                    sudden_change=bool(rng.random() < 0.2),
                )
            ]
            scenario.oracle_dynamics = False
    elif layout == "brake_oncoming":
        scenario.dynamic_obstacles = [oncoming_agent(reference, rng)]
        scenario.oracle_dynamics = False
    elif layout == "follow_lead":
        scenario.dynamic_obstacles = [lead_then_exit_agent(reference, rng)]
        scenario.oracle_dynamics = False
    elif layout == "mixed":
        scenario.dynamic_obstacles = [
            crossing_agent(reference, rng, fraction=0.38, obj_type="forklift"),
            oncoming_agent(reference, rng, fraction=0.62),
        ]
        scenario.oracle_dynamics = False
    elif layout == "detour":
        scenario.dynamic_obstacles = [crossing_agent(reference, rng, fraction=0.48, obj_type="agv")]
        scenario.oracle_dynamics = False
        # Plan around the predicted blockage, then remove the virtual box so the
        # saved scenario only contains the real dynamic obstacle.
        block = barrier_on_path(reference, rng)
        block.id = "virtual_route_block"
        scenario.static_obstacles.append(block)
        alt = plan_static_path(facility, scenario, params, dt=cfg.dt)
        scenario.static_obstacles.pop()
        if alt is not None and polyline_deviation(reference, alt) >= 1.25:
            driven = _drive(
                facility, scenario, alt, params,
                use_prediction=True, prefer_wait=False, dt=cfg.dt,
            )
            rec, _, _ = _accept(
                facility, scenario, driven, reference, params,
                layout=layout, planner="hybrid_astar_detour", mode_id=0, dt=cfg.dt,
            )
            records = []
            if rec is not None:
                records.append(rec)
                waited = _drive(
                    facility, scenario, reference, params,
                    use_prediction=True, prefer_wait=True, dt=cfg.dt,
                )
                wait_rec, _, wait_realized = _accept(
                    facility, scenario, waited, reference, params,
                    layout="yield_cross", planner="hybrid_astar+timed_follow", mode_id=1, dt=cfg.dt,
                )
                if wait_rec is not None and wait_realized.get("meaningful"):
                    records.append(wait_rec)
                return records, "ok"
        if cfg.use_time_aware:
            ta = _try_time_aware(facility, scenario, params, cfg.dt)
            if ta is not None and polyline_deviation(reference, ta) >= 1.25:
                rec, _, _ = _accept(
                    facility, scenario, ta, reference, params,
                    layout=layout, planner="time_aware_detour", mode_id=0, dt=cfg.dt,
                )
                if rec is not None:
                    return [rec], "ok"
        layout = "yield_cross"
    rolled = _drive(
        facility,
        scenario,
        reference,
        params,
        use_prediction=not scenario.oracle_dynamics,
        prefer_wait=layout in {"yield_cross", "mixed", "brake_oncoming", "follow_lead"},
        dt=cfg.dt,
    )
    planner = "hybrid_astar+timed_follow"
    if scenario.oracle_dynamics:
        planner = "hybrid_astar+timed_follow_gt"
    rec, reason, _ = _accept(
        facility, scenario, rolled, reference, params,
        layout=layout, planner=planner, mode_id=0, dt=cfg.dt,
    )
    if rec is None:
        return [], reason
    return [rec], "ok"


def _assign_split(layout: str, realized: dict, counts: Counter, cfg: ExpandConfig) -> str:
    total = sum(counts.values()) + 1
    test_target = cfg.val_fraction  # 0.10 of scenarios; val uses the same fraction
    # The dataclass stores val_fraction 0.1 and train 0.8, so test is the remainder.
    test_frac = max(0.0, 1.0 - cfg.train_fraction - cfg.val_fraction)
    difficult = bool(realized.get("meaningful")) and layout != "ordinary"
    if difficult and counts["test"] / total < test_frac:
        return "test"
    if counts["val"] / total < cfg.val_fraction:
        return "val"
    return "train"


def _write_split_lists(out: Path, assignments: dict[str, str], samples_by_scenario: dict[str, list[str]]) -> dict[str, int]:
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for sid, split in sorted(assignments.items()):
        splits[split].extend(sorted(samples_by_scenario.get(sid, [])))
    split_dir = out / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, files in splits.items():
        tmp = split_dir / f"{name}.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(files, f, indent=2)
        os.replace(tmp, split_dir / f"{name}.json")
    _atomic_json(split_dir / "scenario_assignments.json", assignments)
    return {k: len(v) for k, v in splits.items()}


def _combined_manifest(out: Path, v1: Path) -> None:
    v1_train = []
    train_path = v1 / "splits" / "train.json"
    if train_path.exists():
        v1_train = json.loads(train_path.read_text(encoding="utf-8"))
    v2_train = []
    v2_path = out / "splits" / "train.json"
    if v2_path.exists():
        v2_train = json.loads(v2_path.read_text(encoding="utf-8"))
    manifest = {
        "policy": "train_only",
        "note": (
            "Fine-tuning may load these files only. "
            "v1 and v2 validation/test files are excluded."
        ),
        "v1_root": str(v1).replace("\\", "/"),
        "v2_root": str(out).replace("\\", "/"),
        "v1_train": v1_train,
        "v2_train": v2_train,
    }
    _atomic_json(out / "combined_train_manifest.json", manifest)


def expand_dataset(cfg: ExpandConfig) -> dict:
    out = Path(cfg.out_dir)
    v1 = Path(cfg.v1_dir)
    _refuse_v1(out, v1)
    for sub in ("scenarios", "trajectories", "samples", "splits", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    facility_src = Path(cfg.facility_path)
    facility_dst = out / "facility" / facility_src.name
    if not facility_dst.exists():
        facility_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(facility_src, facility_dst)
    facility = load_facility(facility_src)
    params = VehicleParams()
    gen = ScenarioGenerator(facility, vehicle_params=params)
    facility_occ = build_facility_occupancy(facility, resolution=cfg.map_resolution)

    existing = _scan_v2(out)
    v1_seeds = _load_v1_seeds(v1)
    used_seeds = set(existing["seeds"]) | set(v1_seeds)
    assignments: dict[str, str] = {}
    assign_path = out / "splits" / "scenario_assignments.json"
    if assign_path.exists():
        assignments = json.loads(assign_path.read_text(encoding="utf-8"))
    samples_by_scenario: dict[str, list[str]] = {}
    for name in existing["samples"]:
        sid = name.split("_m")[0]
        samples_by_scenario.setdefault(sid, []).append(name)
    known_endpoints = []
    for data in existing["scenarios"]:
        known_endpoints.append(data)
        sid = data["scenario_id"]
        if sid not in assignments and sid in samples_by_scenario:
            assignments[sid] = "train"

    state_path = out / "generation_state.json"
    attempt = 0
    if state_path.exists():
        attempt = int(json.loads(state_path.read_text(encoding="utf-8")).get("next_attempt", 0))

    failures: Counter = Counter()
    if (out / "failure_reasons.json").exists():
        failures.update(json.loads((out / "failure_reasons.json").read_text(encoding="utf-8")))

    t0 = time.time()
    target = cfg.n_scenarios
    max_attempts = attempt + max(target, 1) * cfg.max_attempts_factor
    pbar = tqdm(total=target, initial=len(samples_by_scenario), desc="v2 scenarios")

    def _n_ok() -> int:
        return len(samples_by_scenario)

    while _n_ok() < target and attempt < max_attempts:
        seed = cfg.seed0 + attempt * SEED_STRIDE
        attempt += 1
        if seed in used_seeds:
            failures["duplicate_seed"] += 1
            continue
        used_seeds.add(seed)
        layout = LAYOUT_CYCLE[(attempt - 1) % len(LAYOUT_CYCLE)]
        rng = np.random.default_rng(seed)
        sid = _scenario_id(_n_ok())
        if (out / "scenarios" / f"{sid}.json").exists():
            failures["id_collision"] += 1
            continue

        if layout == "ordinary":
            ends = gen.sample_endpoints(seed, min_separation_m=10.0)
            if ends is None:
                failures["no_endpoints"] += 1
                continue
            start, goal = ends
            static = generate_static_obstacles(
                facility, rng, n_min=4, n_max=12, keepout=[start, goal], block_corridor_prob=0.15
            )
            scenario = gen._assemble(sid, seed, "straight", False, start, goal, static, [], False)
            scenario.metadata["layout"] = "ordinary"
            rec = plan_expert_trajectory(facility, scenario, params, dt=cfg.dt)
            if rec is None:
                failures["ordinary_plan_failed"] += 1
                continue
            yaw_change = 0.0
            for i in range(1, len(rec.states)):
                dy = (rec.states[i].yaw - rec.states[i - 1].yaw + math.pi) % (2 * math.pi) - math.pi
                yaw_change += abs(dy)
            if yaw_change > 2.2:
                scenario.metadata["behavior_tag"] = "sharp_turn"
            elif yaw_change > 0.9:
                scenario.metadata["behavior_tag"] = "long_curve"
            else:
                scenario.metadata["behavior_tag"] = "straight"
            realized = {
                "layout": "ordinary",
                "yielded": False,
                "braked": False,
                "detoured": False,
                "meaningful": False,
                "predicted_route_conflict": False,
                "observed_gt_conflict": False,
                "min_separation_m": float("inf"),
                "path_deviation_m": 0.0,
                "oracle_dynamics": False,
            }
            scenario.metadata["realized"] = realized
            rec.metadata["realized"] = realized
            rec.metadata["behavior_tag"] = scenario.metadata["behavior_tag"]
            records = [rec]
        else:
            ends = gen.sample_endpoints(seed, min_separation_m=12.0)
            if ends is None:
                failures["no_endpoints"] += 1
                continue
            start, goal = ends
            static = generate_static_obstacles(
                facility,
                rng,
                n_min=2,
                n_max=8,
                keepout=[start, goal],
                block_corridor_prob=0.05,
            )
            scenario = gen._assemble(sid, seed, layout, False, start, goal, static, [], False)
            scenario.metadata["layout"] = layout
            reference = plan_static_path(facility, scenario, params, dt=cfg.dt)
            if reference is None:
                failures["static_plan_failed"] += 1
                continue
            records, reason = _interaction_attempt(
                facility, gen, scenario, layout, rng, params, cfg, reference
            )
            if not records:
                failures[reason] += 1
                scenario.dynamic_obstacles = []
                scenario.oracle_dynamics = False
                scenario.metadata["interaction_fallback_from"] = layout
                scenario.metadata["layout"] = "ordinary"
                layout = "ordinary"
                rec = plan_expert_trajectory(facility, scenario, params, dt=cfg.dt)
                if rec is None:
                    failures["ordinary_plan_failed"] += 1
                    continue
                scenario.metadata["behavior_tag"] = "straight"
                realized = {
                    "layout": "ordinary",
                    "yielded": False,
                    "braked": False,
                    "detoured": False,
                    "meaningful": False,
                    "predicted_route_conflict": False,
                    "observed_gt_conflict": False,
                    "min_separation_m": float("inf"),
                    "path_deviation_m": 0.0,
                    "oracle_dynamics": False,
                }
                scenario.metadata["realized"] = realized
                rec.metadata["realized"] = realized
                rec.metadata["behavior_tag"] = "straight"
                records = [rec]
            else:
                realized = records[0].metadata.get("realized", {})

        if any(_endpoints_close(scenario.to_dict(), prev) for prev in known_endpoints):
            failures["near_duplicate"] += 1
            continue

        save_scenario(scenario, out / "scenarios" / f"{sid}.json")
        static_occ = build_facility_occupancy(
            facility,
            resolution=cfg.map_resolution,
            static_obstacles=scenario.static_obstacles,
        )
        written: list[str] = []
        for rec in records:
            with open(out / "trajectories" / f"{rec.trajectory_id}.json", "w", encoding="utf-8") as f:
                json.dump(rec.to_dict(), f)
            windows = extract_windows(
                facility,
                scenario,
                rec,
                horizon=cfg.horizon,
                dt=cfg.dt,
                map_size_m=cfg.map_size_m,
                map_resolution=cfg.map_resolution,
                stride=cfg.window_stride,
                facility_occ=facility_occ,
                static_occ=static_occ,
            )
            for window in windows:
                written.append(_save_sample(out / "samples", window))
        if not written:
            failures["no_windows"] += 1
            (out / "scenarios" / f"{sid}.json").unlink(missing_ok=True)
            for rec in records:
                traj_path = out / "trajectories" / f"{rec.trajectory_id}.json"
                traj_path.unlink(missing_ok=True)
            continue

        split = _assign_split(layout, realized, Counter(assignments.values()), cfg)
        assignments[sid] = split
        samples_by_scenario[sid] = written
        known_endpoints.append(scenario.to_dict())
        pbar.update(1)
        pbar.set_postfix(layout=layout, windows=sum(len(v) for v in samples_by_scenario.values()))

        if _n_ok() % 5 == 0 or _n_ok() >= target:
            _write_split_lists(out, assignments, samples_by_scenario)
            _combined_manifest(out, v1)
            _atomic_json(out / "failure_reasons.json", dict(failures))
            _atomic_json(
                state_path,
                {
                    "next_attempt": attempt,
                    "scenarios_ok": _n_ok(),
                    "windows": sum(len(v) for v in samples_by_scenario.values()),
                    "seed0": cfg.seed0,
                },
            )

    pbar.close()
    split_counts = _write_split_lists(out, assignments, samples_by_scenario)
    _combined_manifest(out, v1)
    _atomic_json(out / "failure_reasons.json", dict(failures))
    _atomic_json(
        state_path,
        {
            "next_attempt": attempt,
            "scenarios_ok": _n_ok(),
            "windows": sum(len(v) for v in samples_by_scenario.values()),
            "seed0": cfg.seed0,
        },
    )

    by_layout = Counter()
    realized_counts = Counter()
    for data in _scan_v2(out)["scenarios"]:
        meta = data.get("metadata", {})
        by_layout[meta.get("layout", "?")] += 1
        real = meta.get("realized", {})
        for key in ("yielded", "braked", "detoured", "meaningful"):
            if real.get(key):
                realized_counts[key] += 1
    n_windows = sum(len(v) for v in samples_by_scenario.values())
    meta = {
        "schema_version": "1.0.0",
        "role": "dataset_v2_additions",
        "immutable_predecessor": str(v1).replace("\\", "/"),
        "facility_id": facility.facility_id,
        "facility_path": str(facility_src).replace("\\", "/"),
        "coordinate_note": "Same XZ-horizontal facility and tensor schema as dataset_v1.",
        "generalization_note": (
            "Every scenario uses industrial_facility_v1. Held-out splits measure "
            "unseen obstacle configurations and routes inside this building, not a new building."
        ),
        "subgoal_policy": {
            "subgoal": "May be the expert window endpoint when is_local=1. Not an inference input.",
            "inference_goal": "Scenario global goal, stored in sample metadata. Safe at inference.",
            "global_goal": "Scenario destination. Safe at inference.",
        },
        "vehicle_params": params.to_dict(),
        "horizon": cfg.horizon,
        "dt": cfg.dt,
        "map_size_m": cfg.map_size_m,
        "map_resolution": cfg.map_resolution,
        "future_features": ["x", "z", "yaw", "speed", "steering", "accel", "t_rel"],
        "stats": {
            "scenarios_ok": _n_ok(),
            "windows": n_windows,
            "attempts": attempt,
            "failures": dict(failures),
            "by_layout": dict(by_layout),
            "realized": dict(realized_counts),
        },
        "elapsed_sec": time.time() - t0,
        "config": asdict(cfg),
        "splits": split_counts,
        "split_policy": {
            "type": "scenario",
            "test_preference": "meaningful dynamic interactions",
            "train_excludes": "all v1 and v2 val/test",
        },
        "resume_command": (
            "python scripts/generate_v2_additions.py "
            f"--out {cfg.out_dir} --n-scenarios {cfg.n_scenarios}"
        ),
    }
    _atomic_json(out / "dataset_meta.json", meta)
    return meta
