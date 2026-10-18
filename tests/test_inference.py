import numpy as np
import pytest
import torch

from backend.ml.dataset import (
    HORIZON,
    OCC_CHANNELS,
    OCC_GRID,
    RAW_GRID,
    REAL_DATA_ROOT,
    STEP_FEATURES,
    FlowMatchingDataset,
    Normalizer,
    load_canonical_splits,
)
from backend.ml.inference import (
    SOLVERS,
    Candidate,
    FlowMatchingPlanner,
    pairwise_diversity,
    prepare_occupancy,
)
from backend.ml.model import HIDDEN_DIM, NUM_BLOCKS, FlowMatchingModel

pytestmark = pytest.mark.skipif(
    not REAL_DATA_ROOT.exists(), reason="expert dataset not present"
)

GOAL = (6.0, 2.0)


@pytest.fixture(scope="module")
def dataset():
    data = FlowMatchingDataset(load_canonical_splits().test[:40])
    data.normalizer = Normalizer.fit(data.samples)
    return data


@pytest.fixture(scope="module")
def planner(dataset):
    torch.manual_seed(0)
    return FlowMatchingPlanner(
        model=FlowMatchingModel(),
        normalizer=dataset.normalizer,
        schema={
            "horizon": HORIZON,
            "step_features": STEP_FEATURES,
            "dt": 0.2,
            "occ_channels": OCC_CHANNELS,
            "occ_grid": OCC_GRID,
        },
        vehicle=np.zeros(8, dtype=np.float32),
        device=torch.device("cpu"),
    )


@pytest.fixture(scope="module")
def occupancy(dataset):
    return dataset.samples[0].occupancy


@pytest.fixture(scope="module")
def live_field_planner(dataset):
    """A planner whose velocity field is non-zero.

    `FlowMatchingModel` zero-initialises its output projection, so a freshly
    built model predicts exactly zero velocity and the ODE solver is a no-op.
    Tests about integration behaviour need a field that actually moves.
    """
    torch.manual_seed(1)
    model = FlowMatchingModel()
    torch.nn.init.normal_(model.output_projection.weight, std=0.02)
    return FlowMatchingPlanner(
        model=model,
        normalizer=dataset.normalizer,
        schema={
            "horizon": HORIZON,
            "step_features": STEP_FEATURES,
            "dt": 0.2,
            "occ_channels": OCC_CHANNELS,
            "occ_grid": OCC_GRID,
        },
        vehicle=np.zeros(8, dtype=np.float32),
        device=torch.device("cpu"),
    )


def test_candidate_count_and_shape(planner, occupancy):
    candidates = planner.generate(occupancy, GOAL, num_candidates=5, num_steps=4)
    assert len(candidates) == 5
    for candidate in candidates:
        assert candidate.trajectory.shape == (HORIZON, STEP_FEATURES)
        assert np.isfinite(candidate.trajectory).all()


def test_same_seed_is_bit_identical(planner, occupancy):
    first = planner.generate(occupancy, GOAL, num_candidates=4, num_steps=4, seed=7)
    second = planner.generate(occupancy, GOAL, num_candidates=4, num_steps=4, seed=7)
    for a, b in zip(first, second):
        assert np.array_equal(a.trajectory, b.trajectory)


def test_different_seeds_give_different_candidates(planner, occupancy):
    first = planner.generate(occupancy, GOAL, num_candidates=4, num_steps=4, seed=1)
    second = planner.generate(occupancy, GOAL, num_candidates=4, num_steps=4, seed=2)
    assert not np.allclose(first[0].trajectory, second[0].trajectory)


def test_seeds_within_one_call_are_not_collapsed(planner, occupancy):
    candidates = planner.generate(occupancy, GOAL, num_candidates=6, num_steps=4)
    assert pairwise_diversity(candidates) > 1e-3


@pytest.mark.parametrize("solver", SOLVERS)
def test_both_solvers_run(planner, occupancy, solver):
    candidates = planner.generate(
        occupancy, GOAL, num_candidates=3, num_steps=4, solver=solver
    )
    assert all(np.isfinite(c.trajectory).all() for c in candidates)


def test_unknown_solver_is_rejected(planner, occupancy):
    with pytest.raises(ValueError, match="solver must be one of"):
        planner.generate(occupancy, GOAL, solver="midpoint")


def test_decoded_heading_is_a_unit_vector(planner, occupancy):
    for candidate in planner.generate(occupancy, GOAL, num_candidates=4, num_steps=4):
        norms = np.hypot(candidate.trajectory[:, 2], candidate.trajectory[:, 3])
        assert np.allclose(norms, 1.0, atol=1e-5)


def test_candidate_accessors_agree_with_the_raw_array(planner, occupancy):
    candidate = planner.generate(occupancy, GOAL, num_candidates=1, num_steps=4)[0]
    assert np.array_equal(candidate.positions, candidate.trajectory[:, :2])
    assert np.array_equal(candidate.speeds, candidate.trajectory[:, 4])
    assert np.allclose(
        candidate.headings,
        np.arctan2(candidate.trajectory[:, 2], candidate.trajectory[:, 3]),
    )


def test_more_solver_steps_changes_the_result(live_field_planner, occupancy):
    coarse = live_field_planner.generate(
        occupancy, GOAL, num_candidates=2, num_steps=2, seed=3
    )
    fine = live_field_planner.generate(
        occupancy, GOAL, num_candidates=2, num_steps=16, seed=3
    )
    assert not np.allclose(coarse[0].trajectory, fine[0].trajectory)


def test_solvers_disagree_under_a_live_field(live_field_planner, occupancy):
    euler = live_field_planner.generate(
        occupancy, GOAL, num_candidates=2, num_steps=4, seed=3, solver="euler"
    )
    heun = live_field_planner.generate(
        occupancy, GOAL, num_candidates=2, num_steps=4, seed=3, solver="heun"
    )
    assert not np.allclose(euler[0].trajectory, heun[0].trajectory)


def test_zero_velocity_field_leaves_the_noise_untouched(planner, occupancy):
    """Documents the fixture quirk above: an untrained model integrates to a
    no-op, so solver tests must use a perturbed model."""
    coarse = planner.generate(occupancy, GOAL, num_candidates=2, num_steps=2, seed=3)
    fine = planner.generate(occupancy, GOAL, num_candidates=2, num_steps=16, seed=3)
    assert np.allclose(coarse[0].trajectory, fine[0].trajectory)


def test_prepare_occupancy_accepts_raw_and_pooled_grids():
    raw = np.zeros((RAW_GRID, RAW_GRID), dtype=np.float32)
    raw[0, 0] = 1.0
    stacked = prepare_occupancy(raw, raw)
    assert stacked.shape == (OCC_CHANNELS, OCC_GRID, OCC_GRID)
    assert stacked[0, 0, 0] == 1.0

    pooled = np.zeros((OCC_GRID, OCC_GRID), dtype=np.float32)
    assert prepare_occupancy(pooled, pooled).shape == (
        OCC_CHANNELS,
        OCC_GRID,
        OCC_GRID,
    )


def test_conditioning_excludes_subgoal(planner):
    """Inference must build the same 14-feature vector training used, with no
    subgoal slot: see dataset.py for why subgoal is a leak."""
    from backend.ml.dataset import COND_FEATURES

    assert planner.condition_vector(GOAL, 0.0, 1.0, 0.0).shape == (COND_FEATURES,)


def test_checkpoint_roundtrip_reproduces_candidates(planner, occupancy, tmp_path):
    payload = {
        "model_state": planner.model.state_dict(),
        "normalizer": planner.normalizer.to_dict(),
        "schema": planner.schema,
        "vehicle": planner.vehicle.tolist(),
    }
    path = tmp_path / "roundtrip.pt"
    torch.save(payload, path)

    revived = FlowMatchingPlanner.load(path)
    original = planner.generate(occupancy, GOAL, num_candidates=3, num_steps=4, seed=5)
    restored = revived.generate(occupancy, GOAL, num_candidates=3, num_steps=4, seed=5)
    for a, b in zip(original, restored):
        assert np.allclose(a.trajectory, b.trajectory, atol=1e-6)


def test_generate_from_sample_matches_explicit_conditioning(planner, dataset):
    sample = dataset.samples[0]
    from_sample = planner.generate_from_sample(
        sample, num_candidates=2, num_steps=4, seed=11
    )
    explicit = planner.generate(
        sample.occupancy,
        goal_local=(float(sample.cond[0]), float(sample.cond[1])),
        goal_yaw_local=float(np.arctan2(sample.cond[2], sample.cond[3])),
        speed=float(sample.cond[4]),
        steering=float(sample.cond[5]),
        num_candidates=2,
        num_steps=4,
        seed=11,
    )
    for a, b in zip(from_sample, explicit):
        assert np.array_equal(a.trajectory, b.trajectory)


def test_pairwise_diversity_is_zero_for_identical_paths():
    trajectory = np.zeros((HORIZON, STEP_FEATURES), dtype=np.float32)
    twins = [Candidate(trajectory, seed=0, index=i, dt=0.2) for i in range(3)]
    assert pairwise_diversity(twins) == 0.0


def _write_checkpoint(path, model, normalizer, schema_extra):
    torch.save(
        {
            "model_state": model.state_dict(),
            "normalizer": normalizer.to_dict(),
            "vehicle": np.zeros(8, dtype=np.float32).tolist(),
            "schema": {
                "horizon": HORIZON,
                "step_features": STEP_FEATURES,
                "dt": 0.2,
                "occ_channels": OCC_CHANNELS,
                "occ_grid": OCC_GRID,
                "use_dynamic": False,
                **schema_extra,
            },
        },
        path,
    )
    return path


def test_load_rebuilds_non_default_architecture(dataset, tmp_path):
    """A model trained at a smaller width must load without a shape mismatch.

    `load` used to rebuild at the module defaults (256/3) regardless of what
    was trained, so any architecture sweep arm failed here.
    """
    model = FlowMatchingModel(hidden_dim=96, num_blocks=2)
    path = _write_checkpoint(
        tmp_path / "small.pt",
        model,
        dataset.normalizer,
        {"hidden_dim": 96, "num_blocks": 2},
    )

    planner = FlowMatchingPlanner.load(path, device=torch.device("cpu"))

    assert len(planner.model.blocks) == 2
    assert planner.model.input_projection.out_features == 96


def test_load_falls_back_to_defaults_for_legacy_schema(dataset, tmp_path):
    """Checkpoints written before the schema carried sizes were all 256/3."""
    path = _write_checkpoint(
        tmp_path / "legacy.pt", FlowMatchingModel(), dataset.normalizer, {}
    )

    planner = FlowMatchingPlanner.load(path, device=torch.device("cpu"))

    assert len(planner.model.blocks) == NUM_BLOCKS
    assert planner.model.input_projection.out_features == HIDDEN_DIM
