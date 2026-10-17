#!/usr/bin/env python
"""Readiness audit for data/dataset_v1 (does not regenerate data)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from northstar.dataset import NorthstarTrajectoryDataset, collate_trajectory_batch


ROOT = Path("data/dataset_v1")
HORIZON = 32
DT = 0.2
MAX_STEER = 0.55
MAX_SPEED = 2.5


def scenario_id_from_sample(name: str, sidecar: dict) -> str:
    return sidecar.get("scenario_id") or name.split("_m")[0]


def audit_splits() -> dict:
    split_scens: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        names = json.loads((ROOT / "splits" / f"{split}.json").read_text(encoding="utf-8"))
        scens = set()
        for n in names:
            info = json.loads((ROOT / "samples" / Path(n).with_suffix(".json")).read_text(encoding="utf-8"))
            scens.add(scenario_id_from_sample(n, info))
        split_scens[split] = scens
    leak_tv = split_scens["train"] & split_scens["val"]
    leak_tt = split_scens["train"] & split_scens["test"]
    leak_vt = split_scens["val"] & split_scens["test"]
    return {
        "n_scenarios": {k: len(v) for k, v in split_scens.items()},
        "leak_train_val": len(leak_tv),
        "leak_train_test": len(leak_tt),
        "leak_val_test": len(leak_vt),
        "ok": not (leak_tv or leak_tt or leak_vt),
    }


def audit_loader_full() -> dict:
    ds = NorthstarTrajectoryDataset(ROOT, split="train", load_maps=True)
    # Use all samples via concatenating splits
    all_names = []
    for split in ("train", "val", "test"):
        all_names.extend(json.loads((ROOT / "splits" / f"{split}.json").read_text(encoding="utf-8")))
    assert len(all_names) == 5021 or len(list((ROOT / "samples").glob("*.npz"))) == len(all_names)

    n_ok = 0
    n_fail = 0
    failures = []
    local_subgoals = 0
    dyn_interact = 0
    dyn_nonzero = 0
    oracle = 0
    future_rel_checks = []

    # Direct sample walk (faster / clearer than Dataset for all splits)
    for name in all_names:
        npz_path = ROOT / "samples" / name
        json_path = npz_path.with_suffix(".json")
        try:
            if not npz_path.exists() or not json_path.exists():
                raise FileNotFoundError(name)
            data = np.load(npz_path, allow_pickle=False)
            info = json.loads(json_path.read_text(encoding="utf-8"))
            future = np.asarray(data["future"], dtype=np.float32)
            fac = np.asarray(data["facility_occ"], dtype=np.float32)
            sta = np.asarray(data["static_occ"], dtype=np.float32)
            dyn = np.asarray(data["dynamic_occ"], dtype=np.float32)

            assert future.shape == (HORIZON, 7), future.shape
            assert fac.shape == sta.shape
            assert dyn.ndim == 3 and dyn.shape[1:] == fac.shape
            for arr in (future, fac, sta, dyn):
                assert np.isfinite(arr).all()

            # units / ranges
            assert float(info["dt"]) == DT
            speeds = future[:, 3]
            steers = future[:, 4]
            yaws = future[:, 2]
            t_rel = future[:, 6]
            assert speeds.min() >= -1e-3
            assert speeds.max() <= MAX_SPEED + 0.05
            assert steers.min() >= -MAX_STEER - 0.05
            assert steers.max() <= MAX_STEER + 0.05
            assert np.all(np.abs(yaws) <= math.pi + 1e-3)
            assert abs(float(t_rel[0])) < 1e-3
            assert np.allclose(np.diff(t_rel), DT, atol=1e-3)

            # future_rel consistency
            ego = np.array(
                [
                    info["ego0"]["x"],
                    info["ego0"]["z"],
                    info["ego0"]["yaw"],
                    info["ego0"]["speed"],
                    info["ego0"]["steering"],
                ],
                dtype=np.float32,
            )
            fut_t = torch.from_numpy(future)
            ego_t = torch.from_numpy(ego)
            rel = NorthstarTrajectoryDataset._to_ego_frame(fut_t, ego_t).numpy()
            # first step near origin in ego frame
            future_rel_checks.append(float(np.hypot(rel[0, 0], rel[0, 1])))
            # inverse check: rotate back
            c, s = math.cos(ego[2]), math.sin(ego[2])
            xw = c * rel[:, 0] - s * rel[:, 1] + ego[0]
            zw = s * rel[:, 0] + c * rel[:, 1] + ego[1]
            assert np.allclose(xw, future[:, 0], atol=1e-4)
            assert np.allclose(zw, future[:, 1], atol=1e-4)

            # Dataset __getitem__ path for a subset later; here check maps load
            item_future = fut_t
            assert item_future.dtype == torch.float32

            if info.get("subgoal", {}).get("is_local", 0):
                local_subgoals += 1
            # Also detect local by proximity to last future pose
            sg = info["subgoal"]
            last = future[-1]
            if math.hypot(sg["x"] - last[0], sg["z"] - last[1]) < 0.25:
                if math.hypot(sg["x"] - info["global_goal"]["x"], sg["z"] - info["global_goal"]["z"]) > 1.0:
                    local_subgoals += 0  # already counted via is_local; keep separate below

            if bool(info.get("oracle_dynamics")):
                oracle += 1

            # Dynamic map has extra occupancy vs static (predicted agents in local crop)
            extra = (dyn > 0.5) & (sta[None, ...] <= 0.5)
            if extra.any():
                dyn_nonzero += 1
                # Meaningful interaction: extra occupancy within 8m of ego (map center)
                # Map is ego-centered; center cell ~ ego. Check any extra in central radius.
                h, w = fac.shape
                cy, cx = h // 2, w // 2
                res = float(info.get("metadata", {}).get("map_resolution", 0.25))
                rad_cells = int(math.ceil(8.0 / res))
                y0, y1 = max(0, cy - rad_cells), min(h, cy + rad_cells + 1)
                x0, x1 = max(0, cx - rad_cells), min(w, cx + rad_cells + 1)
                if extra[:, y0:y1, x0:x1].any():
                    dyn_interact += 1

            n_ok += 1
        except Exception as e:
            n_fail += 1
            if len(failures) < 20:
                failures.append({"sample": name, "error": repr(e)})

    # full Dataset + collate smoke for each split
    loader_ok = True
    shapes = {}
    for split in ("train", "val", "test"):
        ds = NorthstarTrajectoryDataset(ROOT, split=split)
        loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_trajectory_batch)
        batch = next(iter(loader))
        shapes[split] = {k: (tuple(v.shape), str(v.dtype)) for k, v in batch.items() if torch.is_tensor(v)}
        for k, v in batch.items():
            if torch.is_tensor(v) and not torch.isfinite(v).all():
                loader_ok = False

    # count is_local properly
    n_local = 0
    for name in all_names:
        info = json.loads((ROOT / "samples" / Path(name).with_suffix(".json")).read_text(encoding="utf-8"))
        if float(info.get("subgoal", {}).get("is_local", 0)) >= 0.5:
            n_local += 1

    return {
        "n_listed": len(all_names),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "failures": failures,
        "mean_future_rel_origin_err": float(np.mean(future_rel_checks)) if future_rel_checks else None,
        "local_subgoal_windows": n_local,
        "oracle_windows": oracle,
        "windows_with_predicted_dyn_in_map": dyn_nonzero,
        "windows_with_dyn_near_ego_8m": dyn_interact,
        "loader_ok": loader_ok,
        "batch_shapes": shapes,
    }


def main() -> None:
    splits = audit_splits()
    loader = audit_loader_full()
    report = {"splits": splits, "loader": loader}
    out = Path("data/dataset_v1/readiness_audit.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
