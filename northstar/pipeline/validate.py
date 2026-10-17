"""Validate generated dataset quality for the training teammate."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def validate_dataset(root: str | Path) -> dict:
    root = Path(root)
    meta_path = root / "dataset_meta.json"
    assert meta_path.exists(), f"Missing {meta_path}"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    sample_dir = root / "samples"
    files = sorted(sample_dir.glob("*.npz"))
    assert files, "No samples found"

    behaviors = Counter()
    oracle = 0
    horizons = []
    speed_max = []
    collide_feats = 0
    bad = 0

    for fp in files:
        data = np.load(fp)
        future = data["future"]
        static = data["static_occ"]
        dynamic = data["dynamic_occ"]
        facility = data["facility_occ"]
        info = json.loads(fp.with_suffix(".json").read_text(encoding="utf-8"))

        if future.shape[0] != meta["horizon"]:
            bad += 1
            continue
        if not np.isfinite(future).all():
            bad += 1
            continue

        # Dynamics feasibility heuristics
        dt = float(info["dt"])
        speeds = future[:, 3]
        if speeds.max() > meta["vehicle_params"]["max_speed"] + 0.05:
            bad += 1
            continue
        window_disp = float(
            np.hypot(future[-1, 0] - future[0, 0], future[-1, 1] - future[0, 1])
        )
        # Reject frozen / near-stationary windows
        if speeds.max() < 0.15 and window_disp < 0.5:
            bad += 1
            continue

        # Path continuity
        disp = np.hypot(np.diff(future[:, 0]), np.diff(future[:, 1]))
        if disp.max() > meta["vehicle_params"]["max_speed"] * dt * 1.5 + 0.3:
            bad += 1
            continue

        behaviors[info.get("behavior_tag", "?")] += 1
        oracle += int(bool(info.get("oracle_dynamics")))
        horizons.append(future.shape[0])
        speed_max.append(float(speeds.max()))

        # Maps sane
        if static.shape != facility.shape:
            bad += 1
        if dynamic.ndim != 3:
            bad += 1

        # Ego cell roughly free on facility map center
        cy, cx = facility.shape[0] // 2, facility.shape[1] // 2
        if facility[cy, cx] > 0.5 and static[cy, cx] > 0.5:
            collide_feats += 1

    report = {
        "n_samples": len(files),
        "bad_samples": bad,
        "oracle_windows": oracle,
        "behaviors": dict(behaviors),
        "mean_max_speed": float(np.mean(speed_max)) if speed_max else 0.0,
        "center_occupied_count": collide_feats,
        "splits": meta.get("splits", {}),
        "stats": meta.get("stats", {}),
        "ok": bad == 0 and len(files) > 0,
    }

    out_path = root / "validation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/dataset_v1")
    args = p.parse_args(argv)
    report = validate_dataset(args.root)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
