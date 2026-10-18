import numpy as np
import pytest

from backend.ml.dataset import (
    DYNAMIC_SLICES,
    OCC_CHANNELS,
    OCC_GRID,
    RAW_GRID,
    REAL_DATA_ROOT,
    FlowMatchingDataset,
    Sample,
    load_canonical_splits,
)
from backend.ml.evaluation import (
    MAP_SIZE_M,
    RAW_RESOLUTION,
    _vehicle_limits,
    assess_trajectory,
    footprint_centres,
    load_raw_occupancy,
    local_to_world,
)
from backend.ml.train import consistency_losses

pytestmark = pytest.mark.skipif(
    not REAL_DATA_ROOT.exists(), reason="expert dataset not present"
)

HORIZON = 32


def _straight_trajectory(speed: float = 1.0, dt: float = 0.2) -> np.ndarray:
    trajectory = np.zeros((HORIZON, 5), dtype=np.float32)
    trajectory[:, 0] = np.arange(HORIZON) * speed * dt
    trajectory[:, 3] = 1.0
    trajectory[:, 4] = speed
    return trajectory


def _blank_sample(goal=(10.0, 0.0)) -> Sample:
    cond = np.zeros(14, dtype=np.float32)
    cond[0], cond[1] = goal
    cond[3] = 1.0
    return Sample(
        sample_id="synthetic",
        scenario_id="synthetic",
        occupancy=np.zeros((OCC_CHANNELS, OCC_GRID, OCC_GRID), dtype=np.float32),
        dynamic=np.zeros((DYNAMIC_SLICES, OCC_GRID, OCC_GRID), dtype=np.uint8),
        cond=cond,
        trajectory=_straight_trajectory(),
        oracle=False,
        behavior="synthetic",
        ego={"x": 0.0, "z": 0.0, "yaw": 0.0},
    )


@pytest.fixture(scope="module")
def held_out():
    data = FlowMatchingDataset(load_canonical_splits().test[:40])
    return data


def test_local_to_world_is_identity_without_rotation_or_offset():
    points = np.array([[1.0, 2.0], [-3.0, 0.5]])
    ego = {"x": 0.0, "z": 0.0, "yaw": 0.0}
    assert np.allclose(local_to_world(points, ego), points)


def test_local_to_world_applies_the_ego_pose():
    ego = {"x": 5.0, "z": -2.0, "yaw": np.pi / 2}
    world = local_to_world(np.array([[1.0, 0.0]]), ego)
    assert np.allclose(world, [[5.0, -1.0]], atol=1e-6)


def test_local_to_world_preserves_distances():
    ego = {"x": 3.0, "z": 7.0, "yaw": 0.9}
    points = np.array([[0.0, 0.0], [2.0, 1.0], [5.0, -1.0]])
    world = local_to_world(points, ego)
    for a, b in ((0, 1), (1, 2)):
        assert np.linalg.norm(world[a] - world[b]) == pytest.approx(
            np.linalg.norm(points[a] - points[b]), abs=1e-9
        )


def test_footprint_has_one_disc_per_waypoint_spanning_the_body():
    limits = _vehicle_limits()
    discs = footprint_centres(_straight_trajectory(), limits)
    assert discs.shape[0] == HORIZON
    assert discs.shape[2] == 2
    spread = np.linalg.norm(discs[0, -1] - discs[0, 0])
    assert spread == pytest.approx(limits["length"], abs=1e-5)


def test_footprint_follows_heading():
    limits = _vehicle_limits()
    sideways = _straight_trajectory()
    sideways[:, 2], sideways[:, 3] = 1.0, 0.0
    discs = footprint_centres(sideways, limits)
    offset = discs[0, -1] - discs[0, 0]
    assert abs(offset[1]) > abs(offset[0])


def test_clean_straight_path_is_feasible_and_collision_free():
    sample = _blank_sample()
    occupancy = np.zeros((RAW_GRID, RAW_GRID), dtype=np.float32)
    report = assess_trajectory(_straight_trajectory(), sample, occupancy, 0.2)
    assert not report.collides
    assert report.feasible
    assert report.heading_consistent
    assert report.speed_ok and report.accel_ok and report.curvature_ok


def test_obstacle_on_the_path_is_detected():
    sample = _blank_sample()
    occupancy = np.zeros((RAW_GRID, RAW_GRID), dtype=np.float32)
    centre = RAW_GRID // 2
    ahead = centre + int(3.0 / RAW_RESOLUTION)
    occupancy[centre - 2 : centre + 2, ahead - 2 : ahead + 2] = 1.0
    report = assess_trajectory(_straight_trajectory(), sample, occupancy, 0.2)
    assert report.collides
    assert report.collision_steps > 0


def test_overspeed_is_flagged():
    limits = _vehicle_limits()
    trajectory = _straight_trajectory(speed=limits["max_speed"] + 2.0)
    report = assess_trajectory(
        trajectory, _blank_sample(), np.zeros((RAW_GRID, RAW_GRID), np.float32), 0.2
    )
    assert not report.speed_ok
    assert not report.feasible


def test_heading_decoupled_from_motion_is_flagged():
    """The baseline's actual failure mode: heading that ignores travel."""
    trajectory = _straight_trajectory()
    trajectory[:, 2], trajectory[:, 3] = 1.0, 0.0
    report = assess_trajectory(
        trajectory, _blank_sample(), np.zeros((RAW_GRID, RAW_GRID), np.float32), 0.2
    )
    assert not report.heading_consistent
    assert not report.feasible


def test_goal_progress_is_positive_when_driving_at_the_goal():
    sample = _blank_sample(goal=(10.0, 0.0))
    report = assess_trajectory(
        _straight_trajectory(), sample, np.zeros((RAW_GRID, RAW_GRID), np.float32), 0.2
    )
    assert report.goal_progress_m > 5.0


def test_goal_progress_is_negative_when_driving_away():
    sample = _blank_sample(goal=(-10.0, 0.0))
    report = assess_trajectory(
        _straight_trajectory(), sample, np.zeros((RAW_GRID, RAW_GRID), np.float32), 0.2
    )
    assert report.goal_progress_m < 0.0


def test_expert_trajectories_are_mostly_consistent(held_out):
    """Calibration check: the metric must pass real expert paths, otherwise
    the model numbers it produces mean nothing."""
    limits = _vehicle_limits()
    consistent = []
    for sample in held_out.samples[:25]:
        occupancy = load_raw_occupancy(sample.sample_id)
        report = assess_trajectory(sample.trajectory, sample, occupancy, 0.2, limits)
        consistent.append(report.heading_consistent)
    assert np.mean(consistent) > 0.9


def test_expert_speeds_and_accelerations_are_within_limits(held_out):
    limits = _vehicle_limits()
    for sample in held_out.samples[:25]:
        occupancy = load_raw_occupancy(sample.sample_id)
        report = assess_trajectory(sample.trajectory, sample, occupancy, 0.2, limits)
        assert report.speed_ok
        assert report.accel_ok


def test_consistency_loss_is_near_zero_for_a_clean_path():
    import torch

    trajectory = torch.from_numpy(_straight_trajectory()).unsqueeze(0)
    mean = torch.zeros(5)
    std = torch.ones(5)
    heading, speed, unit = consistency_losses(trajectory, mean, std, 0.2)
    assert float(heading) < 1e-4
    assert float(speed) < 1e-4
    assert float(unit) < 1e-6


def test_consistency_loss_punishes_decoupled_heading():
    import torch

    good = _straight_trajectory()
    bad = good.copy()
    bad[:, 2], bad[:, 3] = 1.0, 0.0

    mean, std = torch.zeros(5), torch.ones(5)
    good_heading, _, _ = consistency_losses(
        torch.from_numpy(good).unsqueeze(0), mean, std, 0.2
    )
    bad_heading, _, _ = consistency_losses(
        torch.from_numpy(bad).unsqueeze(0), mean, std, 0.2
    )
    assert float(bad_heading) > float(good_heading) + 0.5


def test_consistency_loss_punishes_speed_mismatch():
    import torch

    trajectory = _straight_trajectory(speed=1.0)
    trajectory[:, 4] = 3.0

    mean, std = torch.zeros(5), torch.ones(5)
    _, speed_loss, _ = consistency_losses(
        torch.from_numpy(trajectory).unsqueeze(0), mean, std, 0.2
    )
    assert float(speed_loss) > 1.0
