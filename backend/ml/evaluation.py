"""Physical evaluation of generated trajectories on held-out scenarios.

Flow-matching loss says nothing about whether a path is drivable, so this
module measures collision, kinematic feasibility, goal progress and latency.

Two calibration decisions matter for reading the numbers:

1. The stored occupancy rasters are footprint-inflated approximations, and
   `crop_local` marks everything outside the facility as occupied. Point or
   footprint checks against them therefore flag some genuinely collision-free
   expert paths: measured on 200 random windows, 21% of EXPERT trajectories
   touch an occupied cell. So every metric here is also computed for the
   expert trajectory on the identical checker, and only the model-versus-expert
   comparison is meaningful. An absolute collision rate quoted alone would be
   wrong.
2. The rasters are world-axis-aligned (see `OccupancyGrid.crop_local`, a pure
   translation), while trajectories are in the rover frame. Local waypoints are
   therefore rotated back to world before being indexed into the grid.

The rover is approximated by three discs along its length rather than a single
centre point, so the width of the vehicle counts.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from backend.ml.dataset import (
    RAW_GRID,
    REAL_DATA_ROOT,
    Sample,
    load_vehicle_params,
)
from backend.ml.inference import Candidate, FlowMatchingPlanner, pairwise_diversity

RAW_RESOLUTION = 0.25
MAP_SIZE_M = 20.0
FOOTPRINT_DISCS = 3
HEADING_TOLERANCE_RAD = 0.6
MIN_SPEED_FOR_HEADING_CHECK = 0.2
TOLERANCE = 1e-6


def _vehicle_limits() -> dict[str, float]:
    length, width, wheelbase, max_speed, max_steer, max_accel, max_decel, _ = (
        load_vehicle_params()
    )
    return {
        "length": float(length),
        "width": float(width),
        "wheelbase": float(wheelbase),
        "max_speed": float(max_speed),
        "max_steer": float(max_steer),
        "max_accel": float(max_accel),
        "max_decel": float(max_decel),
        "max_curvature": float(np.tan(max_steer) / wheelbase),
    }


def load_raw_occupancy(sample_id: str, root: Path = REAL_DATA_ROOT) -> np.ndarray:
    """Full-resolution static occupancy union, for accurate collision checks."""
    arrays = np.load(root / "samples" / f"{sample_id}.npz")
    return np.maximum(arrays["facility_occ"], arrays["static_occ"])


def footprint_centres(
    trajectory: np.ndarray, limits: dict[str, float]
) -> np.ndarray:
    """Disc centres approximating the rover body at every waypoint."""
    position = trajectory[:, :2]
    heading = np.stack(
        [trajectory[:, 3], trajectory[:, 2]], axis=-1
    )  # (cos, sin) -> forward unit vector
    offsets = np.linspace(
        -limits["length"] / 2, limits["length"] / 2, FOOTPRINT_DISCS
    )
    return position[:, None, :] + heading[:, None, :] * offsets[None, :, None]


def local_to_world(points: np.ndarray, ego: dict) -> np.ndarray:
    cos_yaw, sin_yaw = np.cos(ego["yaw"]), np.sin(ego["yaw"])
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    return points @ rotation.T + np.array([ego["x"], ego["z"]])


def _grid_indices(world: np.ndarray, ego: dict) -> tuple[np.ndarray, np.ndarray]:
    origin = np.array([ego["x"], ego["z"]]) - MAP_SIZE_M / 2
    cells = ((world - origin) / RAW_RESOLUTION).astype(int)
    return cells[..., 0], cells[..., 1]


@dataclass
class TrajectoryReport:
    collides: bool
    collision_steps: int
    clearance_m: float
    leaves_window: bool
    max_speed: float
    speed_ok: bool
    max_abs_accel: float
    accel_ok: bool
    max_curvature: float
    curvature_ok: bool
    heading_consistent: bool
    feasible: bool
    goal_progress_m: float
    final_displacement_m: float
    path_length_m: float
    # Continuous companions to the pass/fail flags above. `heading_consistent`
    # demands every moving step within tolerance, so it saturates at 0 for any
    # model that is broadly misaligned and then hides real improvement: the
    # auxiliary-loss runs cut mean heading error 42.3 -> 27.0 degrees while the
    # flag stayed at 0 throughout. Track these when comparing models.
    heading_error_deg_mean: float
    heading_error_deg_max: float
    speed_mismatch_ms: float


CURVATURE_MIN_STEP_M = 0.02


def _curvature(trajectory: np.ndarray) -> np.ndarray:
    """Path curvature over the steps where the rover actually moves.

    Curvature is yaw change per unit distance, which is undefined when the
    rover is stationary: a hair of yaw noise over a near-zero step produced
    values up to 17 per metre against a 0.613 limit, and because the gate took
    the per-trajectory maximum a single stopped step failed the whole path.
    Many windows start from rest, so this rejected the expert as infeasible -
    27% of v1 and 61% of v2 expert trajectories - even though the expert comes
    from a kinematic planner and is drivable by construction.

    Steps shorter than `CURVATURE_MIN_STEP_M` are dropped, matching the mask
    the heading check already uses.

    The estimator is the exact one for a circular arc sampled at its endpoints,
    `k = 2 sin(dyaw / 2) / chord`, not `dyaw / chord`. The naive form measures
    yaw change against the chord while the yaw is gained along the arc, which
    overestimates curvature by `1 + dyaw^2 / 24`. That is only 0.12% at this
    sample spacing, but the gate compares against `tan(max_steer) / wheelbase`
    with a 1e-6 tolerance, and the expert planner drives at full steering lock,
    so the bias alone failed 61% of v2 expert trajectories whose true curvature
    sits exactly on the limit.
    """
    heading = np.arctan2(trajectory[:, 2], trajectory[:, 3])
    step = np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
    delta = np.arctan2(np.sin(np.diff(heading)), np.cos(np.diff(heading)))
    moving = step > CURVATURE_MIN_STEP_M
    if not moving.any():
        return np.zeros(1)
    return 2.0 * np.abs(np.sin(delta[moving] / 2.0)) / step[moving]


def assess_trajectory(
    trajectory: np.ndarray,
    sample: Sample,
    occupancy: np.ndarray,
    dt: float,
    limits: dict[str, float] | None = None,
) -> TrajectoryReport:
    limits = limits or _vehicle_limits()

    discs = footprint_centres(trajectory, limits)
    world = local_to_world(discs.reshape(-1, 2), sample.ego).reshape(discs.shape)
    grid_x, grid_z = _grid_indices(world, sample.ego)

    inside = (
        (grid_x >= 0) & (grid_x < RAW_GRID) & (grid_z >= 0) & (grid_z < RAW_GRID)
    )
    occupied = np.zeros_like(inside)
    occupied[inside] = occupancy[grid_z[inside], grid_x[inside]] > 0
    collision_steps = int(occupied.any(axis=1).sum())

    free = np.argwhere(occupancy == 0)
    blocked = np.argwhere(occupancy > 0)
    if len(blocked) and len(free):
        centres = np.stack([grid_z[:, FOOTPRINT_DISCS // 2], grid_x[:, FOOTPRINT_DISCS // 2]], axis=-1)
        distances = np.linalg.norm(
            blocked[None, :, :] - centres[:, None, :], axis=-1
        ).min(axis=1)
        clearance = float(distances.min() * RAW_RESOLUTION)
    else:
        clearance = float("inf")

    speed = trajectory[:, 4]
    accel = np.diff(speed) / dt
    curvature = _curvature(trajectory)

    travel = np.diff(trajectory[:, :2], axis=0)
    moving = np.linalg.norm(travel, axis=1) > MIN_SPEED_FOR_HEADING_CHECK * dt
    heading = np.arctan2(trajectory[:-1, 2], trajectory[:-1, 3])
    travel_angle = np.arctan2(travel[:, 1], travel[:, 0])
    heading_error = np.abs(
        np.arctan2(np.sin(travel_angle - heading), np.cos(travel_angle - heading))
    )
    heading_consistent = bool(
        not moving.any() or heading_error[moving].max() <= HEADING_TOLERANCE_RAD
    )
    observed = heading_error[moving] if moving.any() else np.zeros(1)
    speed_mismatch = float(
        np.abs(
            np.linalg.norm(travel, axis=1) / dt
            - 0.5 * (speed[1:] + speed[:-1])
        )[moving].mean()
        if moving.any()
        else 0.0
    )

    speed_ok = bool(speed.max() <= limits["max_speed"] + TOLERANCE and speed.min() >= -TOLERANCE)
    accel_ok = bool(
        accel.max(initial=0.0) <= limits["max_accel"] + TOLERANCE
        and -accel.min(initial=0.0) <= limits["max_decel"] + TOLERANCE
    )
    curvature_ok = bool(curvature.max(initial=0.0) <= limits["max_curvature"] + TOLERANCE)

    goal = sample.cond[:2].astype(np.float64)
    start_gap = float(np.linalg.norm(goal))
    end_gap = float(np.linalg.norm(goal - trajectory[-1, :2]))

    return TrajectoryReport(
        collides=collision_steps > 0,
        collision_steps=collision_steps,
        clearance_m=clearance,
        leaves_window=bool((~inside).any()),
        max_speed=float(speed.max()),
        speed_ok=speed_ok,
        max_abs_accel=float(np.abs(accel).max(initial=0.0)),
        accel_ok=accel_ok,
        max_curvature=float(curvature.max(initial=0.0)),
        curvature_ok=curvature_ok,
        heading_consistent=heading_consistent,
        feasible=speed_ok and accel_ok and curvature_ok and heading_consistent,
        goal_progress_m=start_gap - end_gap,
        final_displacement_m=float(
            np.linalg.norm(trajectory[-1, :2] - sample.trajectory[-1, :2])
        ),
        path_length_m=float(
            np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1).sum()
        ),
        heading_error_deg_mean=float(np.degrees(observed.mean())),
        heading_error_deg_max=float(np.degrees(observed.max())),
        speed_mismatch_ms=speed_mismatch,
    )


def _summarise(reports: list[TrajectoryReport]) -> dict:
    if not reports:
        return {}
    rows = [asdict(r) for r in reports]
    summary: dict[str, float] = {}
    for key, value in rows[0].items():
        series = np.array([row[key] for row in rows], dtype=np.float64)
        series = series[np.isfinite(series)]
        if len(series) == 0:
            continue
        if isinstance(value, bool):
            summary[f"{key}_rate"] = float(series.mean())
        else:
            summary[f"{key}_mean"] = float(series.mean())
            summary[f"{key}_median"] = float(np.median(series))
    return summary


@dataclass
class EvaluationResult:
    checkpoint: str
    scenarios: int
    windows: int
    candidates_per_window: int
    solver: str
    solver_steps: int
    model: dict
    expert: dict
    diversity_m: float
    latency_ms_per_candidate: float
    latency_ms_per_window: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_checkpoint(
    checkpoint_path: Path,
    samples: list[Sample],
    num_candidates: int = 8,
    num_steps: int = 8,
    solver: str = "heun",
    seed: int = 0,
    root: Path = REAL_DATA_ROOT,
) -> EvaluationResult:
    planner = FlowMatchingPlanner.load(checkpoint_path)
    limits = _vehicle_limits()
    dt = float(planner.schema.get("dt", 0.2))

    model_reports: list[TrajectoryReport] = []
    expert_reports: list[TrajectoryReport] = []
    diversities: list[float] = []
    elapsed = 0.0

    for sample in samples:
        occupancy = load_raw_occupancy(sample.sample_id, root)

        started = time.perf_counter()
        candidates = planner.generate_from_sample(
            sample,
            num_candidates=num_candidates,
            num_steps=num_steps,
            seed=seed,
            solver=solver,
        )
        elapsed += time.perf_counter() - started

        diversities.append(pairwise_diversity(candidates))
        for candidate in candidates:
            model_reports.append(
                assess_trajectory(candidate.trajectory, sample, occupancy, dt, limits)
            )
        expert_reports.append(
            assess_trajectory(sample.trajectory, sample, occupancy, dt, limits)
        )

    windows = max(len(samples), 1)
    return EvaluationResult(
        checkpoint=str(checkpoint_path),
        scenarios=len({s.scenario_id for s in samples}),
        windows=len(samples),
        candidates_per_window=num_candidates,
        solver=solver,
        solver_steps=num_steps,
        model=_summarise(model_reports),
        expert=_summarise(expert_reports),
        diversity_m=float(np.mean(diversities)) if diversities else 0.0,
        latency_ms_per_candidate=elapsed * 1000 / max(len(model_reports), 1),
        latency_ms_per_window=elapsed * 1000 / windows,
    )


def _row(label: str, model: float | None, expert: float | None) -> str:
    def cell(value: float | None) -> str:
        return "    n/a" if value is None else f"{value:7.3f}"

    return f"  {label:28s} {cell(model)}   {cell(expert)}"


def print_report(result: EvaluationResult) -> None:
    print(f"checkpoint: {result.checkpoint}")
    print(
        f"held-out: {result.windows} windows / {result.scenarios} scenarios, "
        f"{result.candidates_per_window} candidates each, "
        f"{result.solver_steps} {result.solver} steps"
    )
    print(f"\n{'':30s} {'model':>7s}   {'expert':>7s}")
    keys = [
        ("heading error (deg, mean)", "heading_error_deg_mean_mean"),
        ("heading error (deg, max)", "heading_error_deg_max_mean"),
        ("speed mismatch (m/s)", "speed_mismatch_ms_mean"),
        ("collision rate", "collides_rate"),
        ("feasible rate", "feasible_rate"),
        ("speed within limit", "speed_ok_rate"),
        ("accel within limit", "accel_ok_rate"),
        ("curvature within limit", "curvature_ok_rate"),
        ("heading consistent", "heading_consistent_rate"),
        ("leaves 20 m window", "leaves_window_rate"),
        ("goal progress (m, mean)", "goal_progress_m_mean"),
        ("final displacement (m)", "final_displacement_m_mean"),
        ("path length (m, mean)", "path_length_m_mean"),
        ("clearance (m, median)", "clearance_m_median"),
        ("max curvature (mean)", "max_curvature_mean"),
    ]
    for label, key in keys:
        print(_row(label, result.model.get(key), result.expert.get(key)))

    print(f"\n  candidate diversity:         {result.diversity_m:.3f} m")
    print(f"  latency per candidate:       {result.latency_ms_per_candidate:.1f} ms")
    print(f"  latency per window:          {result.latency_ms_per_window:.1f} ms")
    print(
        "\n  Collision and window figures are measured against footprint-inflated\n"
        "  rasters that also mark outside-facility as occupied, so the expert\n"
        "  column is the reference, not zero."
    )


def main() -> None:
    import argparse

    from backend.ml.dataset import FlowMatchingDataset, load_canonical_splits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--windows", type=int, default=120)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--solver", default="heun")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    splits = load_canonical_splits()
    held_out = FlowMatchingDataset(splits.test)
    stride = max(1, len(held_out) // args.windows)
    samples = held_out.samples[::stride][: args.windows]
    print(f"test pool: {len(held_out)} windows, evaluating {len(samples)}\n")

    results = []
    for checkpoint in args.checkpoints:
        result = evaluate_checkpoint(
            checkpoint,
            samples,
            num_candidates=args.candidates,
            num_steps=args.steps,
            solver=args.solver,
            seed=args.seed,
        )
        print_report(result)
        print()
        results.append(result.to_dict())

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"written: {args.out}")


if __name__ == "__main__":
    main()
