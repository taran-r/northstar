"""Expert dataset loader, normalization, and scenario-level splits.

Reads `data/dataset_v1` (schema 1.0.0). Two deliberate departures from the
files on disk:

1. `data/dataset_v1/splits/*.json` in the working tree are NOT used. That copy
   predates the audit fix and is shuffled per window, so 233 of 239 test
   scenarios also appear in train and overlapping windows straddle the split.
   The canonical scenario-grouped manifests were extracted from `origin/main`
   into `splits_v1_canonical.json` and are frozen there with provenance.
   `assert_no_scenario_overlap` rejects any manifest that leaks.
2. `future` is stored in world coordinates. It is converted to the rover's
   local frame (x forward, y left, origin and heading at the window start).
   The rotation matches `northstar.dataset.pytorch._to_ego_frame`.

`subgoal` is deliberately NOT part of the conditioning. In 4414 of 5021
windows it equals the expert trajectory's final waypoint to within 5 cm
(median distance 0.000 m), and `is_local` flags exactly those windows, so
feeding either would hand the model the answer it is meant to generate.
Conditioning uses `global_goal`, which never coincides with the endpoint.

Normalization statistics are fit on the training split only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

REAL_DATA_ROOT = Path("data/dataset_v1")
V2_DATA_ROOT = Path("data/dataset_v2_additions")
CANONICAL_SPLITS = Path(__file__).parent / "splits_v1_canonical.json"

HORIZON = 32
DT = 0.2
STEP_FEATURES = 7

RAW_GRID = 80
OCC_POOL = 2
OCC_GRID = RAW_GRID // OCC_POOL
OCC_CHANNELS = 2
DYNAMIC_SLICES = 8

VEHICLE_PARAM_ORDER = (
    "length",
    "width",
    "wheelbase",
    "max_speed",
    "max_steer",
    "max_accel",
    "max_decel",
    "max_steer_rate",
)

GLOBAL_GOAL_FEATURES = 4
EGO_FEATURES = 4
COND_FEATURES = (
    GLOBAL_GOAL_FEATURES + EGO_FEATURES + len(VEHICLE_PARAM_ORDER)
)

STD_FLOOR = 1e-4


def _wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def load_vehicle_params(root: Path = REAL_DATA_ROOT) -> np.ndarray:
    meta = json.loads((root / "dataset_meta.json").read_text())
    params = meta["vehicle_params"]
    return np.array(
        [params[name] for name in VEHICLE_PARAM_ORDER], dtype=np.float32
    )


def world_to_local(
    x: np.ndarray, z: np.ndarray, yaw: np.ndarray, ego: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cos_yaw = np.cos(ego["yaw"])
    sin_yaw = np.sin(ego["yaw"])
    dx = x - ego["x"]
    dz = z - ego["z"]
    return (
        dx * cos_yaw + dz * sin_yaw,
        -dx * sin_yaw + dz * cos_yaw,
        _wrap(yaw - ego["yaw"]),
    )


def _pool_max(grid: np.ndarray, factor: int = OCC_POOL) -> np.ndarray:
    size = grid.shape[0] // factor
    return grid.reshape(size, factor, size, factor).max(axis=(1, 3))


@dataclass
class Sample:
    sample_id: str
    scenario_id: str
    occupancy: np.ndarray
    dynamic: np.ndarray
    cond: np.ndarray
    trajectory: np.ndarray
    oracle: bool
    behavior: str
    ego: dict


def dynamic_only(
    dynamic_occ: np.ndarray, facility: np.ndarray, static: np.ndarray
) -> np.ndarray:
    """Isolate the moving-obstacle forecast from the stored dynamic channels.

    `dynamic_occ` is the union of building, static clutter and predicted
    moving obstacles: measured over 600 random windows it covers 100% of the
    facility and static cells, the moving part is ~7% of occupied cells, and
    all eight slices are bit-identical in 76% of windows. Encoding it raw
    would spend the temporal branch re-encoding walls, so subtract the static
    union and keep only what actually moves.

    Returns uint8 because the stored maps are strictly binary.
    """
    obstructed = np.maximum(facility, static) > 0
    isolated = (dynamic_occ > 0) & ~obstructed[None]
    return isolated.astype(np.uint8)


def load_sample(
    sample_id: str, vehicle: np.ndarray, root: Path = REAL_DATA_ROOT
) -> Sample:
    arrays = np.load(root / "samples" / f"{sample_id}.npz")
    meta = json.loads((root / "samples" / f"{sample_id}.json").read_text())

    future = arrays["future"]
    ego = meta["ego0"]
    local_x, local_y, local_yaw = world_to_local(
        future[:, 0], future[:, 1], future[:, 2], ego
    )
    trajectory = np.stack(
        [
            local_x,
            local_y,
            np.sin(local_yaw),
            np.cos(local_yaw),
            future[:, 3],
            future[:, 4],
            future[:, 5],
        ],
        axis=-1,
    ).astype(np.float32)

    facility = arrays["facility_occ"]
    static = arrays["static_occ"]
    occupancy = np.stack([_pool_max(facility), _pool_max(static)]).astype(np.float32)

    isolated = dynamic_only(arrays["dynamic_occ"], facility, static)
    dynamic = np.stack([_pool_max(slice_) for slice_ in isolated]).astype(np.uint8)

    goal = meta["global_goal"]
    goal_x, goal_y, goal_yaw = world_to_local(
        np.array(goal["x"]), np.array(goal["z"]), np.array(goal["yaw"]), ego
    )

    cond = np.concatenate(
        [
            [goal_x, goal_y, np.sin(goal_yaw), np.cos(goal_yaw)],
            [
                ego["speed"],
                ego["steering"],
                np.sin(ego["yaw"]),
                np.cos(ego["yaw"]),
            ],
            vehicle,
        ]
    ).astype(np.float32)

    return Sample(
        sample_id=sample_id,
        scenario_id=meta["scenario_id"],
        occupancy=occupancy,
        dynamic=dynamic,
        cond=cond,
        trajectory=trajectory,
        oracle=bool(meta["oracle_dynamics"]),
        behavior=meta["behavior_tag"],
        ego=ego,
    )


@dataclass
class SampleSplits:
    """Frozen window-level assignment, plus the scenarios each split covers."""

    train: list[str]
    val: list[str]
    test: list[str]
    provenance: dict = field(default_factory=dict)

    def scenarios(self, name: str) -> set[str]:
        return {s.split("_m")[0] for s in getattr(self, name)}

    def to_dict(self) -> dict:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
            "provenance": self.provenance,
            "scenario_counts": {
                name: len(self.scenarios(name)) for name in ("train", "val", "test")
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> SampleSplits:
        return cls(
            train=payload["train"],
            val=payload["val"],
            test=payload["test"],
            provenance=payload.get("provenance", {}),
        )


def assert_no_scenario_overlap(splits: SampleSplits) -> None:
    """Refuse to train on a split that leaks scenarios between partitions.

    The stale manifests in the dataset directory leak 233 of 239 test
    scenarios into train, which makes held-out numbers meaningless. Failing
    loudly here is the only reliable guard, since the symptom is a suspiciously
    good validation score rather than an error.
    """
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = splits.scenarios(left) & splits.scenarios(right)
        if shared:
            raise ValueError(
                f"{len(shared)} scenarios appear in both {left} and {right} "
                f"(e.g. {sorted(shared)[:3]}); refusing to use a leaking split"
            )


def load_canonical_splits(path: Path = CANONICAL_SPLITS) -> SampleSplits:
    payload = json.loads(path.read_text())
    splits = SampleSplits(
        train=payload["samples"]["train"],
        val=payload["samples"]["val"],
        test=payload["samples"]["test"],
        provenance=payload["provenance"],
    )
    assert_no_scenario_overlap(splits)
    return splits


def sample_ids_for_scenarios(
    scenarios: list[str], root: Path = REAL_DATA_ROOT
) -> list[str]:
    wanted = set(scenarios)
    return sorted(
        p.stem
        for p in (root / "samples").glob("*.npz")
        if p.stem.split("_m")[0] in wanted
    )


def _floored_std(std: np.ndarray) -> np.ndarray:
    """Pass constant features through unscaled.

    Statistics are accumulated in float64: in float32 a constant column picks
    up enough summation error to clear a small floor, and dividing by that
    rounding noise turns a constant feature into garbage.
    """
    return np.where(std < STD_FLOOR, 1.0, std).astype(np.float32)


@dataclass
class Normalizer:
    cond_mean: np.ndarray
    cond_std: np.ndarray
    traj_mean: np.ndarray
    traj_std: np.ndarray

    @classmethod
    def fit(cls, samples: list[Sample]) -> Normalizer:
        cond = np.stack([s.cond for s in samples]).astype(np.float64)
        traj = np.stack([s.trajectory for s in samples]).astype(np.float64)
        return cls(
            cond_mean=cond.mean(axis=0).astype(np.float32),
            cond_std=_floored_std(cond.std(axis=0)),
            traj_mean=traj.mean(axis=(0, 1)).astype(np.float32),
            traj_std=_floored_std(traj.std(axis=(0, 1))),
        )

    def normalize_cond(self, cond: np.ndarray) -> np.ndarray:
        return ((cond - self.cond_mean) / self.cond_std).astype(np.float32)

    def normalize_traj(self, trajectory: np.ndarray) -> np.ndarray:
        return ((trajectory - self.traj_mean) / self.traj_std).astype(np.float32)

    def denormalize_traj(self, trajectory: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.traj_mean, device=trajectory.device)
        std = torch.as_tensor(self.traj_std, device=trajectory.device)
        return trajectory * std + mean

    def to_dict(self) -> dict:
        return {
            "cond_mean": self.cond_mean.tolist(),
            "cond_std": self.cond_std.tolist(),
            "traj_mean": self.traj_mean.tolist(),
            "traj_std": self.traj_std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Normalizer:
        return cls(
            cond_mean=np.array(payload["cond_mean"], dtype=np.float32),
            cond_std=np.array(payload["cond_std"], dtype=np.float32),
            traj_mean=np.array(payload["traj_mean"], dtype=np.float32),
            traj_std=np.array(payload["traj_std"], dtype=np.float32),
        )


class FlowMatchingDataset(Dataset):
    def __init__(
        self,
        sample_ids: list[str],
        normalizer: Normalizer | None = None,
        include_oracle: bool = True,
        root: Path = REAL_DATA_ROOT,
    ):
        vehicle = load_vehicle_params(root)
        loaded = (load_sample(sample_id, vehicle, root) for sample_id in sample_ids)
        self.samples = [s for s in loaded if include_oracle or not s.oracle]
        self.normalizer = normalizer

    @classmethod
    def from_samples(
        cls, samples: list[Sample], normalizer: Normalizer | None = None
    ) -> FlowMatchingDataset:
        """Wrap already-loaded samples, bypassing the per-id reader."""
        obj = cls.__new__(cls)
        obj.samples = samples
        obj.normalizer = normalizer
        return obj

    @property
    def scenario_ids(self) -> set[str]:
        return {s.scenario_id for s in self.samples}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        cond = sample.cond
        trajectory = sample.trajectory

        if self.normalizer is not None:
            cond = self.normalizer.normalize_cond(cond)
            trajectory = self.normalizer.normalize_traj(trajectory)

        return {
            "occupancy": torch.from_numpy(sample.occupancy),
            "dynamic": torch.from_numpy(sample.dynamic).float(),
            "cond": torch.from_numpy(np.ascontiguousarray(cond)),
            "trajectory": torch.from_numpy(np.ascontiguousarray(trajectory)),
        }


@dataclass
class DatasetBundle:
    train: FlowMatchingDataset
    val: FlowMatchingDataset
    test: FlowMatchingDataset
    normalizer: Normalizer
    splits: SampleSplits
    include_oracle: bool


def build_datasets(
    root: Path = REAL_DATA_ROOT,
    splits_path: Path = CANONICAL_SPLITS,
    include_oracle: bool = True,
) -> DatasetBundle:
    splits = load_canonical_splits(splits_path)

    train = FlowMatchingDataset(splits.train, None, include_oracle, root)
    normalizer = Normalizer.fit(train.samples)
    train.normalizer = normalizer

    return DatasetBundle(
        train=train,
        val=FlowMatchingDataset(splits.val, normalizer, include_oracle, root),
        test=FlowMatchingDataset(splits.test, normalizer, include_oracle, root),
        normalizer=normalizer,
        splits=splits,
        include_oracle=include_oracle,
    )


def load_official_splits(root: Path) -> SampleSplits:
    """Read the scenario-grouped manifests shipped inside a corpus.

    Both corpora ship `splits/{train,val,test}.json` listing window filenames.
    The v1 copies in the working tree were leaky before the upstream audit fix;
    they are clean as of `origin/main` commit `b32b87a7`, and
    `assert_no_scenario_overlap` is kept as the standing guard rather than a
    one-off check.
    """

    def ids(name: str) -> list[str]:
        payload = json.loads((root / "splits" / f"{name}.json").read_text())
        if isinstance(payload, dict):
            payload = payload.get("samples") or payload.get("windows") or []
        return [i[:-4] if i.endswith(".npz") else i for i in payload]

    splits = SampleSplits(
        train=ids("train"),
        val=ids("val"),
        test=ids("test"),
        provenance={"source": str(root / "splits"), "kind": "official_shipped"},
    )
    assert_no_scenario_overlap(splits)
    return splits


def sample_from_item(item: dict, vehicle: np.ndarray) -> Sample:
    """Build a training sample from a `northstar.dataset.pytorch` item.

    `future_rel` is already expressed in the rover frame and agrees with
    `world_to_local` to float32 precision, so it is used directly. `ego0` and
    `global_goal` remain in world coordinates and are rotated here.

    The two heading channels come from the relative yaw, while `sin/cos` of the
    world ego yaw enter the condition vector: the occupancy rasters are
    world-axis-aligned while the trajectory is rover-frame, so without the
    world heading the model cannot align the map to the path it predicts.
    """
    ego_array = item["ego0"].numpy()
    ego = {
        "x": float(ego_array[0]),
        "z": float(ego_array[1]),
        "yaw": float(ego_array[2]),
        "speed": float(ego_array[3]),
        "steering": float(ego_array[4]),
    }

    future_rel = item["future_rel"].numpy()
    trajectory = np.stack(
        [
            future_rel[:, 0],
            future_rel[:, 1],
            np.sin(future_rel[:, 2]),
            np.cos(future_rel[:, 2]),
            future_rel[:, 3],
            future_rel[:, 4],
            future_rel[:, 5],
        ],
        axis=-1,
    ).astype(np.float32)

    facility = np.asarray(item["facility_occ"].numpy()).squeeze()
    static = np.asarray(item["static_occ"].numpy()).squeeze()
    occupancy = np.stack([_pool_max(facility), _pool_max(static)]).astype(np.float32)

    isolated = dynamic_only(item["dynamic_occ"].numpy(), facility, static)
    dynamic = np.stack([_pool_max(slice_) for slice_ in isolated]).astype(np.uint8)

    goal = item["global_goal"].numpy()
    goal_x, goal_y, goal_yaw = world_to_local(
        np.array(float(goal[0])), np.array(float(goal[1])), np.array(float(goal[2])), ego
    )

    cond = np.concatenate(
        [
            [goal_x, goal_y, np.sin(goal_yaw), np.cos(goal_yaw)],
            [ego["speed"], ego["steering"], np.sin(ego["yaw"]), np.cos(ego["yaw"])],
            vehicle,
        ]
    ).astype(np.float32)

    sample_id = item["sample_id"]
    return Sample(
        sample_id=sample_id,
        scenario_id=sample_id.rsplit("_m", 1)[0],
        occupancy=occupancy,
        dynamic=dynamic,
        cond=cond,
        trajectory=trajectory,
        oracle=bool(float(item["oracle_dynamics"])),
        behavior=item.get("behavior_tag", ""),
        ego=ego,
    )


@dataclass
class CombinedBundle:
    """Train on both corpora, evaluate each held-out set on its own.

    v1 and v2 test sets are kept apart on purpose: v1 is the baseline every
    earlier checkpoint was measured against, and v2 covers harder interaction
    scenarios, so pooling them would hide which of the two moved.
    """

    train: FlowMatchingDataset
    val_v1: FlowMatchingDataset
    val_v2: FlowMatchingDataset
    test_v1: FlowMatchingDataset
    test_v2: FlowMatchingDataset
    normalizer: Normalizer
    splits: dict
    include_oracle: bool
    counts: dict


FEATURE_VERSION = "step7_cond16_dyn8"


def _transform_all(source, vehicle: np.ndarray, include_oracle: bool) -> list[Sample]:
    samples = []
    for index in range(len(source)):
        sample = sample_from_item(source[index], vehicle)
        if include_oracle or not sample.oracle:
            samples.append(sample)
    return samples


def build_combined_datasets(
    roots: tuple[Path, ...] = (REAL_DATA_ROOT, V2_DATA_ROOT),
    include_oracle: bool = False,
    cache_dir: Path | None = Path("backend/artifacts/cache"),
) -> CombinedBundle:
    """Train on v1+v2 through `CombinedTrainDataset`, evaluate each set apart.

    `CombinedTrainDataset` refuses anything but the train split by design, so a
    v1 or v2 held-out scenario cannot reach training. Validation and test come
    from `NorthstarTrajectoryDataset` on each root separately.

    Oracle windows are excluded by default. `oracle_dynamics=1` marks windows
    whose expert planner saw ground-truth obstacle futures - 23.7% of v1 train
    and 3.6% of v2 train - which is privileged information the deployed model
    will not have.

    Transformed samples are cached per feature version, since re-reading 22k
    npz files costs more than the training run itself.
    """
    from northstar.dataset.combined import CombinedTrainDataset
    from northstar.dataset.pytorch import NorthstarTrajectoryDataset

    roots = tuple(Path(r) for r in roots)
    vehicle = load_vehicle_params(roots[0])
    splits = {str(root): load_official_splits(root) for root in roots}

    for left in splits:
        for right in splits:
            if left >= right:
                continue
            for a in ("train", "val", "test"):
                for b in ("train", "val", "test"):
                    shared = splits[left].scenarios(a) & splits[right].scenarios(b)
                    if shared:
                        raise ValueError(
                            f"{len(shared)} scenarios shared between {left}:{a} "
                            f"and {right}:{b}; refusing to train"
                        )

    tag = f"{FEATURE_VERSION}_oracle{int(include_oracle)}"
    cache = (cache_dir / f"samples_{tag}.pt") if cache_dir else None
    if cache is not None and cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        parts = payload["parts"]
    else:
        parts = {
            "train": _transform_all(
                CombinedTrainDataset(list(roots), split="train"), vehicle, include_oracle
            )
        }
        for root in roots:
            key = "v1" if root == REAL_DATA_ROOT else "v2"
            for split in ("val", "test"):
                parts[f"{split}_{key}"] = _transform_all(
                    NorthstarTrajectoryDataset(root, split=split), vehicle, include_oracle
                )
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"parts": parts, "feature_version": FEATURE_VERSION}, cache)

    normalizer = Normalizer.fit(parts["train"])
    made = {
        name: FlowMatchingDataset.from_samples(samples, normalizer)
        for name, samples in parts.items()
    }
    counts = {name: len(samples) for name, samples in parts.items()}
    counts["train_scenarios"] = len({s.scenario_id for s in parts["train"]})

    return CombinedBundle(
        train=made["train"],
        val_v1=made["val_v1"],
        val_v2=made["val_v2"],
        test_v1=made["test_v1"],
        test_v2=made["test_v2"],
        normalizer=normalizer,
        splits={k: v.to_dict() for k, v in splits.items()},
        include_oracle=include_oracle,
        counts=counts,
    )
