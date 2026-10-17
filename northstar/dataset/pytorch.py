"""PyTorch-friendly dataset for conditional flow-matching training.

Each sample provides:
  - ego0: current rover state
  - future: [H, 7] expert future (x, z, yaw, speed, steering, accel, t_rel)
  - subgoal / global_goal
  - facility_occ: permanent structure local map
  - static_occ: facility + temporary static obstacles
  - dynamic_occ: [T, H, W] predicted dynamic occupancy (NOT ground-truth futures)
  - oracle_dynamics: flag if expert planner used GT dynamics
  - behavior_tag / mode_id
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class NorthstarTrajectoryDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        load_maps: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.load_maps = load_maps
        index_path = self.root / "splits" / f"{split}.json"
        if not index_path.exists():
            # Fall back to all samples
            sample_dir = self.root / "samples"
            self.sample_files = sorted(sample_dir.glob("*.npz"))
        else:
            with open(index_path, "r", encoding="utf-8") as f:
                names = json.load(f)
            self.sample_files = [self.root / "samples" / n for n in names]

        meta_path = self.root / "dataset_meta.json"
        self.meta: dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.sample_files[idx]
        data = np.load(path, allow_pickle=False)
        # npz may store JSON-like fields as arrays; sidecar for dicts
        sidecar = path.with_suffix(".json")
        info: dict[str, Any] = {}
        if sidecar.exists():
            with open(sidecar, "r", encoding="utf-8") as f:
                info = json.load(f)

        future = torch.from_numpy(np.asarray(data["future"], dtype=np.float32))
        ego0 = torch.tensor(
            [
                info["ego0"]["x"],
                info["ego0"]["z"],
                info["ego0"]["yaw"],
                info["ego0"]["speed"],
                info["ego0"]["steering"],
            ],
            dtype=torch.float32,
        )
        subgoal = torch.tensor(
            [info["subgoal"]["x"], info["subgoal"]["z"], info["subgoal"]["yaw"]],
            dtype=torch.float32,
        )
        global_goal = torch.tensor(
            [
                info["global_goal"]["x"],
                info["global_goal"]["z"],
                info["global_goal"]["yaw"],
            ],
            dtype=torch.float32,
        )

        item: dict[str, Any] = {
            "future": future,  # [H, 7]
            "ego0": ego0,  # [5]
            "subgoal": subgoal,  # [3]
            "global_goal": global_goal,  # [3]
            "oracle_dynamics": torch.tensor(
                float(info.get("oracle_dynamics", False)), dtype=torch.float32
            ),
            "mode_id": torch.tensor(int(info.get("mode_id", 0)), dtype=torch.int64),
            "sample_id": info.get("sample_id", path.stem),
            "behavior_tag": info.get("behavior_tag", "unknown"),
            "dt": torch.tensor(float(info.get("dt", 0.2)), dtype=torch.float32),
        }

        if self.load_maps:
            item["facility_occ"] = torch.from_numpy(
                np.asarray(data["facility_occ"], dtype=np.float32)
            ).unsqueeze(0)
            item["static_occ"] = torch.from_numpy(
                np.asarray(data["static_occ"], dtype=np.float32)
            ).unsqueeze(0)
            item["dynamic_occ"] = torch.from_numpy(
                np.asarray(data["dynamic_occ"], dtype=np.float32)
            )

        # Relative future in ego frame (convenient for flow matching)
        item["future_rel"] = self._to_ego_frame(future, ego0)
        return item

    @staticmethod
    def _to_ego_frame(future: torch.Tensor, ego0: torch.Tensor) -> torch.Tensor:
        """Translate/rotate x,z,yaw into ego frame; keep speed/steer/accel/t."""
        x0, z0, yaw0 = ego0[0], ego0[1], ego0[2]
        c, s = torch.cos(yaw0), torch.sin(yaw0)
        dx = future[:, 0] - x0
        dz = future[:, 1] - z0
        x_r = c * dx + s * dz
        z_r = -s * dx + c * dz
        yaw_r = (future[:, 2] - yaw0 + np.pi) % (2 * np.pi) - np.pi
        out = future.clone()
        out[:, 0] = x_r
        out[:, 1] = z_r
        out[:, 2] = yaw_r
        return out


def collate_trajectory_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Default collate that stacks tensors and keeps string fields as lists."""
    out: dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out
