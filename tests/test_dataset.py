import json
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.ml.dataset import (
    COND_FEATURES,
    HORIZON,
    OCC_CHANNELS,
    OCC_GRID,
    REAL_DATA_ROOT,
    STEP_FEATURES,
    CANONICAL_SPLITS,
    DYNAMIC_SLICES,
    FlowMatchingDataset,
    Normalizer,
    SampleSplits,
    assert_no_scenario_overlap,
    dynamic_only,
    load_canonical_splits,
    load_official_splits,
    load_sample,
    load_vehicle_params,
    world_to_local,
)

pytestmark = pytest.mark.skipif(
    not REAL_DATA_ROOT.exists(), reason="expert dataset not present"
)


@pytest.fixture(scope="module")
def splits():
    return load_canonical_splits()


@pytest.fixture(scope="module")
def held_out(splits):
    data = FlowMatchingDataset(splits.test)
    data.normalizer = Normalizer.fit(data.samples)
    return data


def test_canonical_split_matches_the_audit_counts(splits):
    assert (len(splits.train), len(splits.val), len(splits.test)) == (4203, 420, 398)
    assert len(splits.scenarios("train")) == 290
    assert len(splits.scenarios("val")) == 36
    assert len(splits.scenarios("test")) == 37


def test_canonical_split_records_its_provenance(splits):
    assert splits.provenance["commit"]
    assert "origin/main" in splits.provenance["source"]


def test_canonical_split_shares_no_scenario(splits):
    assert_no_scenario_overlap(splits)


def test_every_window_belongs_to_exactly_one_split(splits):
    combined = splits.train + splits.val + splits.test
    on_disk = {p.stem for p in (REAL_DATA_ROOT / "samples").glob("*.npz")}
    assert len(combined) == len(set(combined))
    assert set(combined) == on_disk


def test_shipped_manifests_are_scenario_grouped():
    """The working-tree copy of data/dataset_v1/splits leaked 233 of 239 test
    scenarios into train before the upstream audit fix. It is clean as of
    origin/main b32b87a7, so the shipped manifests must now pass the guard."""
    shipped = load_official_splits(REAL_DATA_ROOT)
    assert_no_scenario_overlap(shipped)
    assert len(shipped.scenarios("train")) == 290
    assert len(shipped.scenarios("val")) == 36
    assert len(shipped.scenarios("test")) == 37


def test_a_leaking_manifest_is_still_rejected():
    """The guard itself, exercised on a split built to leak. This is the test
    that must keep failing loudly if a future corpus reintroduces the bug."""
    shipped = load_official_splits(REAL_DATA_ROOT)
    leaking = SampleSplits(
        train=shipped.train + shipped.test[:1],
        val=shipped.val,
        test=shipped.test,
    )
    with pytest.raises(ValueError, match="refusing to use a leaking split"):
        assert_no_scenario_overlap(leaking)


def test_sample_tensor_shapes(held_out):
    item = held_out[0]
    assert item["occupancy"].shape == (OCC_CHANNELS, OCC_GRID, OCC_GRID)
    assert item["cond"].shape == (COND_FEATURES,)
    assert item["trajectory"].shape == (HORIZON, STEP_FEATURES)


def test_trajectory_starts_at_local_origin(held_out):
    for index in range(0, len(held_out), 37):
        step = held_out.samples[index].trajectory[0]
        assert np.allclose(step[:2], 0.0, atol=1e-5)
        assert np.allclose(step[2:4], [0.0, 1.0], atol=1e-5)


def test_heading_is_a_unit_vector(held_out):
    for index in range(0, len(held_out), 37):
        trajectory = held_out.samples[index].trajectory
        assert np.allclose(np.hypot(trajectory[:, 2], trajectory[:, 3]), 1.0, atol=1e-4)


def test_local_conversion_preserves_path_length(held_out):
    vehicle = load_vehicle_params()
    for sample in held_out.samples[:25]:
        raw = np.load(REAL_DATA_ROOT / "samples" / f"{sample.sample_id}.npz")["future"]
        world = np.linalg.norm(np.diff(raw[:, :2], axis=0), axis=1).sum()
        local = np.linalg.norm(np.diff(sample.trajectory[:, :2], axis=0), axis=1).sum()
        assert local == pytest.approx(world, abs=1e-3)


def test_world_to_local_is_identity_at_the_ego_pose():
    ego = {"x": 3.0, "z": -7.0, "yaw": 0.9}
    x, y, yaw = world_to_local(
        np.array([ego["x"]]), np.array([ego["z"]]), np.array([ego["yaw"]]), ego
    )
    assert x[0] == pytest.approx(0.0, abs=1e-6)
    assert y[0] == pytest.approx(0.0, abs=1e-6)
    assert yaw[0] == pytest.approx(0.0, abs=1e-6)


def test_speed_respects_the_vehicle_limit(held_out):
    max_speed = load_vehicle_params()[3]
    for sample in held_out.samples:
        assert sample.trajectory[:, 4].max() <= max_speed + 1e-4


def test_oracle_windows_can_be_excluded(splits):
    scenarios = splits.test
    everything = FlowMatchingDataset(scenarios)
    filtered = FlowMatchingDataset(scenarios, include_oracle=False)
    assert not any(s.oracle for s in filtered.samples)
    assert len(filtered) <= len(everything)


def test_normalized_features_are_finite_and_bounded(held_out):
    cond = torch.stack([held_out[i]["cond"] for i in range(len(held_out))])
    trajectory = torch.stack([held_out[i]["trajectory"] for i in range(len(held_out))])
    assert torch.isfinite(cond).all()
    assert torch.isfinite(trajectory).all()
    assert cond.abs().max() < 50.0


def test_normalizer_roundtrip_is_lossless(held_out):
    raw = held_out.samples[3].trajectory
    restored = held_out.normalizer.denormalize_traj(held_out[3]["trajectory"]).numpy()
    assert np.allclose(restored, raw, atol=1e-4)


def test_normalizer_survives_serialisation(held_out):
    revived = Normalizer.from_dict(held_out.normalizer.to_dict())
    assert np.allclose(revived.traj_std, held_out.normalizer.traj_std)
    assert np.allclose(revived.cond_mean, held_out.normalizer.cond_mean)


def test_splits_survive_serialisation(splits):
    assert SampleSplits.from_dict(splits.to_dict()).test == splits.test


def test_conditioning_excludes_the_expert_derived_subgoal(held_out):
    """In 4414/5021 windows `subgoal` IS the expert trajectory endpoint, so it
    must never reach the model. Compare the raw cond vector against the
    endpoint the model is supposed to generate."""
    for sample in held_out.samples[:60]:
        meta = json.loads(
            (REAL_DATA_ROOT / "samples" / f"{sample.sample_id}.json").read_text()
        )
        if not meta["subgoal"]["is_local"]:
            continue
        endpoint = sample.trajectory[-1, :2]
        for offset in range(len(sample.cond) - 1):
            pair = sample.cond[offset:offset + 2]
            assert not np.allclose(pair, endpoint, atol=0.05), (
                f"cond[{offset}:{offset+2}] leaks the trajectory endpoint"
            )


def test_subgoal_leak_premise_still_holds():
    """Documents why the exclusion above exists; fails loudly if a future
    dataset version stops deriving subgoal from the expert future."""
    leaking = 0
    sample_ids = sorted(p.stem for p in (REAL_DATA_ROOT / "samples").glob("*.npz"))[:400]
    for sample_id in sample_ids:
        meta = json.loads((REAL_DATA_ROOT / "samples" / f"{sample_id}.json").read_text())
        future = np.load(REAL_DATA_ROOT / "samples" / f"{sample_id}.npz")["future"]
        subgoal = np.array([meta["subgoal"]["x"], meta["subgoal"]["z"]])
        if np.linalg.norm(subgoal - future[-1, :2]) < 0.05:
            leaking += 1
    assert leaking > 0


def test_global_goal_is_not_the_trajectory_endpoint(held_out):
    for sample in held_out.samples[:60]:
        meta = json.loads(
            (REAL_DATA_ROOT / "samples" / f"{sample.sample_id}.json").read_text()
        )
        raw = np.load(REAL_DATA_ROOT / "samples" / f"{sample.sample_id}.npz")["future"]
        goal = np.array([meta["global_goal"]["x"], meta["global_goal"]["z"]])
        assert np.linalg.norm(goal - raw[-1, :2]) > 0.05


def test_constant_vehicle_features_normalize_to_zero(held_out):
    cond = torch.stack([held_out[i]["cond"] for i in range(len(held_out))])
    vehicle_block = cond[:, -len(load_vehicle_params()):]
    assert torch.allclose(vehicle_block, torch.zeros_like(vehicle_block), atol=1e-5)


def test_dynamic_channels_shape_and_binary(held_out):
    for sample in held_out.samples[:20]:
        assert sample.dynamic.shape == (DYNAMIC_SLICES, OCC_GRID, OCC_GRID)
        assert sample.dynamic.dtype == np.uint8
        assert set(np.unique(sample.dynamic).tolist()) <= {0, 1}


def test_dynamic_only_removes_static_structure():
    """Raw dynamic_occ is the union of building, clutter and moving obstacles:
    it covers 100% of facility/static cells. Isolating must strip those."""
    facility = np.zeros((4, 4), dtype=np.float32)
    static = np.zeros((4, 4), dtype=np.float32)
    facility[0, 0] = 1.0
    static[1, 1] = 1.0

    raw = np.zeros((2, 4, 4), dtype=np.float32)
    raw[:, 0, 0] = 1.0
    raw[:, 1, 1] = 1.0
    raw[0, 3, 3] = 1.0

    isolated = dynamic_only(raw, facility, static)
    assert isolated[0, 0, 0] == 0
    assert isolated[0, 1, 1] == 0
    assert isolated[0, 3, 3] == 1
    assert isolated[1, 3, 3] == 0


def test_isolating_dynamic_drops_most_occupied_cells(held_out):
    raw_fraction, isolated_fraction = [], []
    for sample in held_out.samples[:30]:
        arrays = np.load(REAL_DATA_ROOT / "samples" / f"{sample.sample_id}.npz")
        raw_fraction.append(float((arrays["dynamic_occ"] > 0).mean()))
        isolated_fraction.append(float(sample.dynamic.mean()))
    assert np.mean(raw_fraction) > 10 * np.mean(isolated_fraction)


def test_dynamic_tensor_is_float_for_the_model(held_out):
    item = held_out[0]
    assert item["dynamic"].dtype == torch.float32
    assert item["dynamic"].shape == (DYNAMIC_SLICES, OCC_GRID, OCC_GRID)
