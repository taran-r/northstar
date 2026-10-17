"""Automated checks for dataset loader, split integrity, and coords."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from northstar.coords import (
    backend_to_threejs,
    backend_xy_to_dataset_xz,
    dataset_to_threejs,
    dataset_xz_to_backend_xy,
    threejs_to_backend,
)
from northstar.dataset import NorthstarTrajectoryDataset, collate_trajectory_batch

ROOT = Path(__file__).resolve().parents[1] / "data" / "dataset_v1"


def test_coords_roundtrip():
    x, z, y, yaw = 3.0, -2.5, 0.0, 0.7
    bx, by, bz, psi = dataset_xz_to_backend_xy(x, z, y, yaw)
    assert (bx, by, bz, psi) == (3.0, -2.5, 0.0, 0.7)
    x2, z2, y2, yaw2 = backend_xy_to_dataset_xz(bx, by, bz, psi)
    assert (x2, z2, y2, yaw2) == (x, z, y, yaw)

    fx, fy, fz = backend_to_threejs(1.0, 2.0, 3.0)
    assert (fx, fy, fz) == (1.0, 3.0, -2.0)
    assert threejs_to_backend(fx, fy, fz) == (1.0, 2.0, 3.0)

    tx, ty, tz = dataset_to_threejs(4.0, 5.0, 0.0)
    assert (tx, ty, tz) == (4.0, 0.0, -5.0)


def test_splits_no_scenario_leakage():
    assert ROOT.exists(), "data/dataset_v1 missing — clone with dataset assets"
    split_scens: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        names = json.loads((ROOT / "splits" / f"{split}.json").read_text(encoding="utf-8"))
        scens = set()
        for n in names:
            info = json.loads((ROOT / "samples" / Path(n).with_suffix(".json")).read_text(encoding="utf-8"))
            scens.add(info["scenario_id"])
        split_scens[split] = scens
    assert not (split_scens["train"] & split_scens["val"])
    assert not (split_scens["train"] & split_scens["test"])
    assert not (split_scens["val"] & split_scens["test"])


def test_loader_batch_shapes_and_finite():
    ds = NorthstarTrajectoryDataset(ROOT, split="train")
    assert len(ds) > 0
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_trajectory_batch)
    batch = next(iter(loader))
    assert batch["future"].shape == (4, 32, 7)
    assert batch["future_rel"].shape == (4, 32, 7)
    assert batch["ego0"].shape == (4, 5)
    assert batch["subgoal"].shape == (4, 3)
    assert batch["global_goal"].shape == (4, 3)
    assert batch["facility_occ"].shape[0] == 4
    assert batch["static_occ"].shape == batch["facility_occ"].shape
    assert batch["dynamic_occ"].ndim == 4
    for k, v in batch.items():
        if torch.is_tensor(v):
            assert torch.isfinite(v).all(), k
            assert v.dtype in (torch.float32, torch.int64)


def test_future_rel_origin_and_inverse():
    ds = NorthstarTrajectoryDataset(ROOT, split="val")
    item = ds[0]
    rel = item["future_rel"]
    assert float(torch.hypot(rel[0, 0], rel[0, 1])) < 1e-4
    # yaw wrapped
    assert float(item["future"][:, 2].abs().max()) <= math.pi + 1e-5


def test_dynamic_occ_is_not_required_to_match_gt_channel():
    """Sanity: dynamic_occ is a separate tensor; loader does not expose GT futures."""
    ds = NorthstarTrajectoryDataset(ROOT, split="test")
    item = ds[0]
    assert "dynamic_occ" in item
    assert "dynamic_gt_occ" not in item
    assert set(item.keys()) >= {
        "future",
        "future_rel",
        "ego0",
        "subgoal",
        "global_goal",
        "facility_occ",
        "static_occ",
        "dynamic_occ",
        "oracle_dynamics",
    }
