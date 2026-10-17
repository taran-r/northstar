"""End-to-end synthetic dataset generation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from northstar.facility.build_base import write_default_facility
from northstar.facility.loader import load_facility, save_scenario
from northstar.facility.occupancy import build_facility_occupancy
from northstar.pipeline.expert import plan_expert_trajectory, plan_multimodal
from northstar.scenario.generator import BEHAVIOR_TAGS, ScenarioGenerator
from northstar.trajectory.format import DEFAULT_DT, DEFAULT_HORIZON_STEPS
from northstar.trajectory.windowing import extract_windows
from northstar.vehicle.bicycle import VehicleParams


@dataclass
class GenerateConfig:
    out_dir: str = "data/dataset_v1"
    n_scenarios: int = 360
    min_windows: int = 5000
    seed: int = 42
    dt: float = DEFAULT_DT
    horizon: int = DEFAULT_HORIZON_STEPS
    multimodal_fraction: float = 0.25
    max_modes: int = 2
    window_stride: int = 4
    map_size_m: float = 20.0
    map_resolution: float = 0.25
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    # remainder = test
    facility_path: Optional[str] = None
    max_attempts_factor: int = 15


def _save_sample(sample_dir: Path, window) -> str:
    arr = window.to_npz()
    fname = f"{window.sample_id}.npz"
    np.savez_compressed(
        sample_dir / fname,
        future=arr["future"],
        static_occ=arr["static_occ"],
        dynamic_occ=arr["dynamic_occ"],
        facility_occ=arr["facility_occ"],
    )
    meta = {
        "sample_id": arr["sample_id"],
        "scenario_id": arr["scenario_id"],
        "trajectory_id": arr["trajectory_id"],
        "horizon": arr["horizon"],
        "dt": arr["dt"],
        "ego0": arr["ego0"],
        "subgoal": arr["subgoal"],
        "global_goal": arr["global_goal"],
        "oracle_dynamics": arr["oracle_dynamics"],
        "behavior_tag": arr["behavior_tag"],
        "mode_id": arr["mode_id"],
        "metadata": arr["metadata"],
    }
    with open(sample_dir / f"{window.sample_id}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return fname


def generate_dataset(cfg: GenerateConfig) -> dict:
    out = Path(cfg.out_dir)
    scen_dir = out / "scenarios"
    traj_dir = out / "trajectories"
    sample_dir = out / "samples"
    split_dir = out / "splits"
    for d in (scen_dir, traj_dir, sample_dir, split_dir):
        d.mkdir(parents=True, exist_ok=True)

    if cfg.facility_path:
        facility_path = Path(cfg.facility_path)
    else:
        facility_path = write_default_facility(out / "facility" / "industrial_facility_v1.json")
        # Also write canonical copy under facilities/
        write_default_facility()

    facility = load_facility(facility_path)
    # Canonical facilities folder
    write_default_facility()

    vehicle = VehicleParams()
    gen = ScenarioGenerator(facility, vehicle_params=vehicle)
    facility_occ = build_facility_occupancy(facility, resolution=cfg.map_resolution)

    rng = np.random.default_rng(cfg.seed)
    behavior_cycle = list(BEHAVIOR_TAGS)

    stats = {
        "scenarios_requested": cfg.n_scenarios,
        "min_windows": cfg.min_windows,
        "scenarios_ok": 0,
        "trajectories_ok": 0,
        "windows": 0,
        "failures": 0,
        "by_behavior": {},
        "oracle_scenarios": 0,
        "multimodal_scenarios": 0,
    }

    sample_names: list[str] = []
    t0 = time.time()

    attempts = 0
    target_scen = cfg.n_scenarios
    target_win = cfg.min_windows
    # Allow extra scenarios if window quota not yet met
    max_attempts = max(target_scen, target_win // 5) * cfg.max_attempts_factor
    pbar = tqdm(total=max(target_scen, 1), desc="scenarios")

    scen_i = 0

    def _quota_met() -> bool:
        return stats["scenarios_ok"] >= target_scen and stats["windows"] >= target_win

    while not _quota_met() and attempts < max_attempts:
        attempts += 1
        seed = int(cfg.seed + attempts * 9973)
        behavior = behavior_cycle[scen_i % len(behavior_cycle)]
        multi = bool(rng.random() < cfg.multimodal_fraction)

        scenario = gen.generate(
            seed=seed,
            scenario_id=f"scn_{stats['scenarios_ok']:05d}",
            force_behavior=behavior,
            multi_modal=multi,
        )
        if scenario is None:
            stats["failures"] += 1
            continue

        if multi:
            records = plan_multimodal(
                facility,
                scenario,
                vehicle_params=vehicle,
                dt=cfg.dt,
                max_modes=cfg.max_modes,
            )
        else:
            rec = plan_expert_trajectory(
                facility, scenario, vehicle_params=vehicle, dt=cfg.dt, mode_id=0
            )
            records = [rec] if rec is not None else []

        if not records:
            stats["failures"] += 1
            continue

        save_scenario(scenario, scen_dir / f"{scenario.scenario_id}.json")
        static_occ = build_facility_occupancy(
            facility,
            resolution=cfg.map_resolution,
            static_obstacles=scenario.static_obstacles,
        )

        n_win = 0
        for rec in records:
            with open(traj_dir / f"{rec.trajectory_id}.json", "w", encoding="utf-8") as f:
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
            for w in windows:
                fname = _save_sample(sample_dir, w)
                sample_names.append(fname)
                n_win += 1

        if n_win == 0:
            # Keep scenario/traj for debugging but don't count as success for training quota
            stats["failures"] += 1
            continue

        stats["scenarios_ok"] += 1
        stats["trajectories_ok"] += len(records)
        stats["windows"] += n_win
        stats["by_behavior"][behavior] = stats["by_behavior"].get(behavior, 0) + 1
        if scenario.oracle_dynamics:
            stats["oracle_scenarios"] += 1
        if len(records) > 1:
            stats["multimodal_scenarios"] += 1
        scen_i += 1
        if stats["scenarios_ok"] <= target_scen:
            pbar.update(1)
        pbar.set_postfix(windows=stats["windows"])

    pbar.close()

    # Splits by scenario (no window from the same scenario across splits)
    from collections import defaultdict

    by_scenario: dict[str, list[str]] = defaultdict(list)
    for fname in sample_names:
        # scn_00000_m0_w0000.npz -> scn_00000
        sid = fname.split("_m")[0]
        by_scenario[sid].append(fname)
    scenarios = list(by_scenario.keys())
    rng.shuffle(scenarios)
    n_scen = len(scenarios)
    n_train_s = int(n_scen * cfg.train_fraction)
    n_val_s = int(n_scen * cfg.val_fraction)
    train_s = scenarios[:n_train_s]
    val_s = scenarios[n_train_s : n_train_s + n_val_s]
    test_s = scenarios[n_train_s + n_val_s :]
    splits = {
        "train": sorted(f for s in train_s for f in by_scenario[s]),
        "val": sorted(f for s in val_s for f in by_scenario[s]),
        "test": sorted(f for s in test_s for f in by_scenario[s]),
    }
    for name, files in splits.items():
        with open(split_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(files, f, indent=2)

    meta = {
        "schema_version": "1.0.0",
        "facility_id": facility.facility_id,
        "facility_schema_version": facility.schema_version,
        "facility_path": str(facility_path),
        "vehicle_params": vehicle.to_dict(),
        "horizon": cfg.horizon,
        "dt": cfg.dt,
        "map_size_m": cfg.map_size_m,
        "map_resolution": cfg.map_resolution,
        "future_features": [
            "x",
            "z",
            "yaw",
            "speed",
            "steering",
            "accel",
            "t_rel",
        ],
        "conditioning": [
            "ego0",
            "subgoal",
            "global_goal",
            "facility_occ",
            "static_occ",
            "dynamic_occ (predicted, not GT)",
        ],
        "planner": {
            "primary": "time-aware Hybrid A*",
            "fallback": "Hybrid A* + velocity profile",
            "smoothing": "B-spline + curvature-aware speed profile",
            "oracle_note": (
                "Scenarios with oracle_dynamics=true used ground-truth dynamic "
                "futures in the expert planner; NN conditioning still uses "
                "constant-velocity predictions."
            ),
        },
        "stats": stats,
        "elapsed_sec": time.time() - t0,
        "config": asdict(cfg),
        "splits": {k: len(v) for k, v in splits.items()},
        "split_policy": {
            "type": "scenario",
            "train_fraction": cfg.train_fraction,
            "val_fraction": cfg.val_fraction,
            "n_scenarios": {
                "train": len(train_s),
                "val": len(val_s),
                "test": len(test_s),
            },
            "note": "Windows from the same scenario_id never cross splits.",
        },
    }
    with open(out / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta
