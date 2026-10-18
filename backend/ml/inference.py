"""Candidate trajectory generation from a trained flow-matching checkpoint.

Sampling starts from gaussian noise in trajectory space and integrates the
learned velocity field from t=0 to t=1. Nothing from a classical planner is
mixed in: if the model produces a poor candidate, it is returned as-is for the
ranking stage to reject. Fallbacks belong upstream and must be logged as such.

Given a seed, output is bit-identical across runs.

Usage:
    python -m backend.ml.inference backend/artifacts/flow_matching_v1_best.pt
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from backend.ml.dataset import (
    DT,
    OCC_POOL,
    RAW_GRID,
    REAL_DATA_ROOT,
    Normalizer,
    _pool_max,
    dynamic_only,
    load_vehicle_params,
)
from backend.ml.model import HIDDEN_DIM, NUM_BLOCKS, FlowMatchingModel

DEFAULT_CANDIDATES = 8
DEFAULT_SOLVER_STEPS = 8
DEFAULT_SOLVER = "heun"
SOLVERS = ("euler", "heun")


@dataclass
class Candidate:
    """One generated trajectory in the rover's local frame, physical units.

    `trajectory` is (K, 7): x forward (m), y left (m), sin(yaw), cos(yaw),
    speed (m/s), steering (rad), accel (m/s^2). Step k is at time k * `dt`
    from now. Step 0 is the rover's current pose, so it should sit near
    (0, 0, 0, 1). The last two channels are the control targets the CBF filter
    corrects rather than re-derives.
    """

    trajectory: np.ndarray
    seed: int
    index: int
    dt: float

    @property
    def positions(self) -> np.ndarray:
        return self.trajectory[:, :2]

    @property
    def headings(self) -> np.ndarray:
        return np.arctan2(self.trajectory[:, 2], self.trajectory[:, 3])

    @property
    def speeds(self) -> np.ndarray:
        return self.trajectory[:, 4]

    @property
    def steering(self) -> np.ndarray:
        return self.trajectory[:, 5]

    def world_xz(self, ego: dict) -> np.ndarray:
        """(K, 2) polyline in dataset world coordinates, XZ-horizontal.

        `northstar.control.TrajectoryTracker` consumes a world XZ polyline, not
        a rover-frame array, so this is the conversion at the safety-filter
        boundary. `ego` needs `x`, `z` and `yaw` as stored in sample metadata.
        """
        cos_yaw, sin_yaw = np.cos(ego["yaw"]), np.sin(ego["yaw"])
        local = self.positions
        return np.stack(
            [
                ego["x"] + local[:, 0] * cos_yaw - local[:, 1] * sin_yaw,
                ego["z"] + local[:, 0] * sin_yaw + local[:, 1] * cos_yaw,
            ],
            axis=-1,
        )

    def world_yaw(self, ego: dict) -> np.ndarray:
        return np.arctan2(
            np.sin(self.headings + ego["yaw"]), np.cos(self.headings + ego["yaw"])
        )

    @property
    def accel(self) -> np.ndarray:
        return self.trajectory[:, 6]


def prepare_occupancy(facility: np.ndarray, static: np.ndarray) -> np.ndarray:
    """Stack and downsample the two rasters the model was trained on.

    Accepts either the raw 80x80 grids from the dataset or grids already
    pooled to 40x40.
    """
    grids = []
    for grid in (facility, static):
        grid = np.asarray(grid, dtype=np.float32).squeeze()
        grids.append(_pool_max(grid) if grid.shape[0] == RAW_GRID else grid)
    return np.stack(grids)


class FlowMatchingPlanner:
    def __init__(
        self,
        model: FlowMatchingModel,
        normalizer: Normalizer,
        schema: dict,
        vehicle: np.ndarray,
        device: torch.device,
    ):
        self.model = model.eval()
        self.normalizer = normalizer
        self.schema = schema
        self.vehicle = vehicle
        self.device = device

    @classmethod
    def load(
        cls, checkpoint_path: Path, device: torch.device | None = None
    ) -> FlowMatchingPlanner:
        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        schema = payload["schema"]

        # Sizes come from the checkpoint, not the module defaults: a model
        # trained at any other width would otherwise be rebuilt at 256/3 and
        # fail to load with a shape mismatch. Checkpoints written before these
        # schema fields existed were all trained at the defaults.
        model = FlowMatchingModel(
            hidden_dim=int(schema.get("hidden_dim", HIDDEN_DIM)),
            num_blocks=int(schema.get("num_blocks", NUM_BLOCKS)),
            use_dynamic=bool(schema.get("use_dynamic", False)),
        ).to(device)
        model.load_state_dict(payload["model_state"])

        vehicle = payload.get("vehicle")
        vehicle = (
            np.array(vehicle, dtype=np.float32)
            if vehicle is not None
            else load_vehicle_params(REAL_DATA_ROOT)
        )

        return cls(
            model=model,
            normalizer=Normalizer.from_dict(payload["normalizer"]),
            schema=schema,
            vehicle=vehicle,
            device=device,
        )

    @property
    def uses_dynamic(self) -> bool:
        return self.model.use_dynamic

    def condition_vector(
        self,
        goal_local: tuple[float, float],
        goal_yaw_local: float = 0.0,
        speed: float = 0.0,
        steering: float = 0.0,
        ego_yaw: float = 0.0,
    ) -> np.ndarray:
        """Assemble the 16-feature condition vector.

        `ego_yaw` is the rover's heading in world coordinates, not the local
        frame where it is zero by construction. The occupancy rasters are
        world-axis-aligned while the trajectory is rover-frame, so the world
        heading is what lets the model relate the two.
        """
        return np.concatenate(
            [
                [
                    goal_local[0],
                    goal_local[1],
                    np.sin(goal_yaw_local),
                    np.cos(goal_yaw_local),
                ],
                [speed, steering, np.sin(ego_yaw), np.cos(ego_yaw)],
                self.vehicle,
            ]
        ).astype(np.float32)

    @torch.no_grad()
    def plan(
        self,
        facility_occ: np.ndarray,
        static_occ: np.ndarray,
        goal_local: tuple[float, float],
        goal_yaw_local: float = 0.0,
        speed: float = 0.0,
        steering: float = 0.0,
        ego_yaw: float = 0.0,
        dynamic_occ: np.ndarray | None = None,
        num_candidates: int = DEFAULT_CANDIDATES,
        num_steps: int = DEFAULT_SOLVER_STEPS,
        seed: int = 0,
        solver: str = DEFAULT_SOLVER,
        record_flow: bool = False,
    ) -> PlanResult:
        """Plan from raw rover state and occupancy, returning candidates plus
        diagnostics.

        This is the entry point for the safety filter and the renderer, so it
        takes the inputs they already hold - rover state, goal, the two static
        rasters and the predicted dynamic forecast - and does the pooling and
        dynamic isolation internally. Pass `dynamic_occ` as the raw [T, H, W]
        forecast; the static union is subtracted here, matching training.

        Given a seed the result is bit-identical across runs.
        """
        occupancy = prepare_occupancy(facility_occ, static_occ)

        dynamic = None
        if self.uses_dynamic:
            if dynamic_occ is None:
                raise ValueError(
                    "this checkpoint uses dynamic conditioning; pass "
                    "`dynamic_occ`, the predicted [T, H, W] occupancy forecast"
                )
            facility = np.asarray(facility_occ, dtype=np.float32).squeeze()
            static = np.asarray(static_occ, dtype=np.float32).squeeze()
            isolated = dynamic_only(
                np.asarray(dynamic_occ, dtype=np.float32), facility, static
            )
            pooled = [
                _pool_max(slice_) if slice_.shape[0] == RAW_GRID else slice_
                for slice_ in isolated
            ]
            dynamic = np.stack(pooled).astype(np.float32)

        started = time.perf_counter()
        flow: list[tuple[float, np.ndarray]] | None = [] if record_flow else None
        candidates = self.generate(
            occupancy,
            goal_local,
            goal_yaw_local=goal_yaw_local,
            speed=speed,
            steering=steering,
            num_candidates=num_candidates,
            num_steps=num_steps,
            seed=seed,
            solver=solver,
            dynamic=dynamic,
            ego_yaw=ego_yaw,
            flow_snapshots=flow,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        limits = _limits_from_vehicle(self.vehicle)
        diagnostics = [diagnose(c, goal_local, limits) for c in candidates]

        return PlanResult(
            candidates=candidates,
            diagnostics=diagnostics,
            latency_ms_total=elapsed_ms,
            latency_ms_per_candidate=elapsed_ms / max(len(candidates), 1),
            diversity_m=pairwise_diversity(candidates),
            uses_dynamic=self.uses_dynamic,
            solver=solver,
            solver_steps=num_steps,
            seed=seed,
            dt=float(self.schema["dt"]),
            flow=flow,
        )

    def _decode(self, normalized: torch.Tensor) -> np.ndarray:
        trajectory = self.normalizer.denormalize_traj(normalized).cpu().numpy()
        heading = np.linalg.norm(trajectory[..., 2:4], axis=-1, keepdims=True)
        trajectory[..., 2:4] /= np.maximum(heading, 1e-6)
        return trajectory

    @torch.no_grad()
    def generate(
        self,
        occupancy: np.ndarray,
        goal_local: tuple[float, float],
        goal_yaw_local: float = 0.0,
        speed: float = 0.0,
        steering: float = 0.0,
        num_candidates: int = DEFAULT_CANDIDATES,
        num_steps: int = DEFAULT_SOLVER_STEPS,
        seed: int = 0,
        solver: str = DEFAULT_SOLVER,
        dynamic: np.ndarray | None = None,
        ego_yaw: float = 0.0,
        flow_snapshots: list | None = None,
    ) -> list[Candidate]:
        if solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
        if self.uses_dynamic and dynamic is None:
            raise ValueError(
                "this checkpoint uses dynamic conditioning; pass `dynamic` "
                "(the isolated moving-obstacle forecast from dataset.dynamic_only)"
            )

        cond = self.condition_vector(
            goal_local, goal_yaw_local, speed, steering, ego_yaw
        )
        cond = self.normalizer.normalize_cond(cond)

        horizon = self.schema["horizon"]
        features = self.schema["step_features"]

        occupancy_batch = (
            torch.from_numpy(np.ascontiguousarray(occupancy))
            .to(self.device)
            .unsqueeze(0)
            .expand(num_candidates, -1, -1, -1)
        )
        cond_batch = (
            torch.from_numpy(cond).to(self.device).unsqueeze(0).expand(num_candidates, -1)
        )

        dynamic_batch = None
        if self.uses_dynamic:
            dynamic_batch = (
                torch.from_numpy(np.ascontiguousarray(dynamic))
                .float()
                .to(self.device)
                .unsqueeze(0)
                .expand(num_candidates, -1, -1, -1)
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        z = torch.randn(
            (num_candidates, horizon, features),
            generator=generator,
            device=self.device,
        )

        step = 1.0 / num_steps
        if flow_snapshots is not None:
            flow_snapshots.append((0.0, self._decode(z).copy()))
        recorded_mid = False
        for index in range(num_steps):
            t = torch.full((num_candidates,), index * step, device=self.device)
            velocity = self.model(z, t, occupancy_batch, cond_batch, dynamic_batch)

            if solver == "euler":
                z = z + step * velocity
            else:
                t_next = torch.full(
                    (num_candidates,), (index + 1) * step, device=self.device
                )
                probe = self.model(
                    z + step * velocity, t_next, occupancy_batch, cond_batch,
                    dynamic_batch,
                )
                z = z + step * 0.5 * (velocity + probe)
            done = (index + 1) * step
            if flow_snapshots is not None and not recorded_mid and done >= 0.5:
                flow_snapshots.append((float(done), self._decode(z).copy()))
                recorded_mid = True

        decoded = self._decode(z)
        if flow_snapshots is not None:
            flow_snapshots.append((1.0, decoded.copy()))
        dt = float(self.schema.get("dt", DT))
        return [
            Candidate(trajectory=decoded[i], seed=seed, index=i, dt=dt)
            for i in range(num_candidates)
        ]

    def generate_from_sample(self, sample, **kwargs) -> list[Candidate]:
        """Convenience path for evaluation: condition on a dataset sample.

        Reads the goal and ego state back out of the sample's stored condition
        vector, so it stays consistent with whatever `dataset.py` feeds training.
        """
        cond = sample.cond
        return self.generate(
            occupancy=sample.occupancy,
            goal_local=(float(cond[0]), float(cond[1])),
            goal_yaw_local=float(np.arctan2(cond[2], cond[3])),
            speed=float(cond[4]),
            steering=float(cond[5]),
            ego_yaw=float(np.arctan2(cond[6], cond[7])),
            dynamic=sample.dynamic if self.uses_dynamic else None,
            **kwargs,
        )


def pairwise_diversity(candidates: list[Candidate]) -> float:
    """Mean pairwise distance between candidate paths, in metres.

    Near zero means the seeds collapsed to one route, which is the failure
    mode worth watching for in a generative planner.
    """
    paths = np.stack([c.positions for c in candidates])
    total, pairs = 0.0, 0
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            total += float(np.linalg.norm(paths[i] - paths[j], axis=-1).mean())
            pairs += 1
    return total / max(pairs, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--steps", type=int, default=DEFAULT_SOLVER_STEPS)
    parser.add_argument("--solver", choices=SOLVERS, default=DEFAULT_SOLVER)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from backend.ml.dataset import FlowMatchingDataset, load_canonical_splits

    planner = FlowMatchingPlanner.load(args.checkpoint)
    print(f"device: {planner.device}")

    held_out = FlowMatchingDataset(load_canonical_splits().test[:40])
    sample = held_out.samples[0]
    print(f"scenario: {sample.scenario_id}  behavior: {sample.behavior}")

    started = time.perf_counter()
    candidates = planner.generate_from_sample(
        sample,
        num_candidates=args.candidates,
        num_steps=args.steps,
        seed=args.seed,
        solver=args.solver,
    )
    elapsed = (time.perf_counter() - started) * 1000

    expert = sample.trajectory
    print(
        f"\n{args.candidates} candidates, {args.steps} {args.solver} steps: "
        f"{elapsed:.1f} ms total, {elapsed / args.candidates:.1f} ms each"
    )
    print(f"pairwise diversity: {pairwise_diversity(candidates):.3f} m")
    print(f"\nexpert endpoint:    {expert[-1, :2].round(2)}")
    for candidate in candidates:
        endpoint = candidate.trajectory[-1, :2]
        error = float(np.linalg.norm(endpoint - expert[-1, :2]))
        print(
            f"  candidate {candidate.index}: endpoint {endpoint.round(2)}  "
            f"final-displacement error {error:.2f} m  "
            f"max speed {candidate.speeds.max():.2f}"
        )


if __name__ == "__main__":
    main()


@dataclass
class CandidateDiagnostics:
    """Per-candidate vehicle-dynamics check, in physical units.

    Collision checking is deliberately absent: the safety filter owns the
    authoritative map and must re-check against it anyway, so duplicating a
    weaker check here would invite two disagreeing answers. These are the
    quantities the CBF layer needs to know before it corrects a candidate.
    """

    index: int
    max_speed: float
    speed_ok: bool
    max_abs_accel: float
    accel_ok: bool
    max_abs_steer: float
    steer_ok: bool
    max_curvature: float
    curvature_ok: bool
    path_length_m: float
    goal_distance_m: float
    heading_error_deg_mean: float
    speed_mismatch_ms: float
    dynamics_ok: bool


@dataclass
class PlanResult:
    """What the planner hands downstream.

    `candidates` are returned unfiltered and unranked, in the order generated.
    Nothing here is a guarantee: `dynamics_ok` reports the model's own
    kinematic self-consistency, not safety.
    """

    candidates: list[Candidate]
    diagnostics: list[CandidateDiagnostics]
    latency_ms_total: float
    latency_ms_per_candidate: float
    diversity_m: float
    uses_dynamic: bool
    solver: str
    solver_steps: int
    seed: int
    dt: float
    flow: list | None = None

    @property
    def dynamically_feasible(self) -> list[Candidate]:
        return [
            candidate
            for candidate, report in zip(self.candidates, self.diagnostics)
            if report.dynamics_ok
        ]

    def to_dict(self) -> dict:
        return {
            "trajectories": [c.trajectory.tolist() for c in self.candidates],
            "diagnostics": [asdict(d) for d in self.diagnostics],
            "latency_ms_total": self.latency_ms_total,
            "latency_ms_per_candidate": self.latency_ms_per_candidate,
            "diversity_m": self.diversity_m,
            "uses_dynamic": self.uses_dynamic,
            "solver": self.solver,
            "solver_steps": self.solver_steps,
            "seed": self.seed,
            "dt": self.dt,
            "frame": "rover local: x forward, y left, metres; yaw relative to ego",
            "channels": ["x", "y", "sin_yaw", "cos_yaw", "speed", "steering", "accel"],
        }


def _limits_from_vehicle(vehicle: np.ndarray) -> dict:
    order = (
        "length", "width", "wheelbase", "max_speed",
        "max_steer", "max_accel", "max_decel", "max_steer_rate",
    )
    return {name: float(value) for name, value in zip(order, vehicle)}


def diagnose(
    candidate: Candidate, goal_local: tuple[float, float], limits: dict
) -> CandidateDiagnostics:
    """Kinematic self-consistency of one generated trajectory.

    Checks the model against the vehicle envelope and against itself: the
    predicted speed channel must agree with the distance actually covered per
    step, and the predicted heading must agree with the direction of travel.
    Those two are where a trajectory that looks plausible on a plot stops being
    drivable.
    """
    positions = candidate.positions
    speeds = candidate.speeds
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    moving = steps > 0.02

    heading = candidate.headings
    travel = np.arctan2(np.diff(positions[:, 1]), np.diff(positions[:, 0]))
    error = np.arctan2(
        np.sin(travel - heading[:-1]), np.cos(travel - heading[:-1])
    )
    heading_error = np.degrees(np.abs(error[moving])) if moving.any() else np.array([0.0])

    implied = steps / candidate.dt
    midpoint = 0.5 * (speeds[1:] + speeds[:-1])
    mismatch = np.abs(implied - midpoint)[moving] if moving.any() else np.array([0.0])

    delta = np.arctan2(np.sin(np.diff(heading)), np.cos(np.diff(heading)))
    curvature = np.abs(delta) / np.maximum(steps, 1e-3)

    max_speed = float(np.abs(speeds).max())
    max_accel = float(np.abs(candidate.accel).max())
    max_steer = float(np.abs(candidate.steering).max())
    max_curvature = float(curvature.max()) if curvature.size else 0.0
    curvature_limit = np.tan(limits["max_steer"]) / limits["wheelbase"]

    speed_ok = max_speed <= limits["max_speed"] + 1e-3
    accel_ok = max_accel <= max(limits["max_accel"], limits["max_decel"]) + 1e-3
    steer_ok = max_steer <= limits["max_steer"] + 1e-3
    curvature_ok = max_curvature <= curvature_limit + 1e-3

    return CandidateDiagnostics(
        index=candidate.index,
        max_speed=max_speed,
        speed_ok=speed_ok,
        max_abs_accel=max_accel,
        accel_ok=accel_ok,
        max_abs_steer=max_steer,
        steer_ok=steer_ok,
        max_curvature=max_curvature,
        curvature_ok=curvature_ok,
        path_length_m=float(steps.sum()),
        goal_distance_m=float(
            np.linalg.norm(positions[-1] - np.asarray(goal_local, dtype=np.float64))
        ),
        heading_error_deg_mean=float(heading_error.mean()),
        speed_mismatch_ms=float(mismatch.mean()),
        dynamics_ok=bool(speed_ok and accel_ok and steer_ok and curvature_ok),
    )


@dataclass
class RankedCandidate:
    candidate: Candidate
    diagnostics: CandidateDiagnostics
    collision_free: bool | None
    clearance_m: float | None
    rejected: str | None

    @property
    def usable(self) -> bool:
        return self.rejected is None


def rank_candidates(
    result: PlanResult,
    collision_free: list[bool] | None = None,
    clearance_m: list[float] | None = None,
) -> list[RankedCandidate]:
    """Order candidates best-first and say why any were rejected.

    Generation returns candidates unranked, and taking the first one wastes the
    reason for drawing several: measured on held-out windows, a single
    candidate collides 47-61% of the time while at least one of eight is clean
    in 69% of v1 and 86% of v2 windows. The gap is entirely this step.

    Collision state is supplied by the caller rather than recomputed here. The
    safety filter owns the authoritative map and must re-check anyway, so a
    second, weaker check in this module would only create two answers that can
    disagree. Pass `collision_free` and `clearance_m` when a map is available;
    without them ranking falls back to kinematic self-consistency and goal
    distance, which is weaker but never wrong about geometry it cannot see.

    Ordering is lexicographic, not a weighted sum: a colliding candidate is not
    redeemable by being shorter, so the terms are not commensurable.

    Only collisions reject. Falling outside the vehicle envelope sorts a
    candidate down but does not disqualify it, because the follower consumes
    the position polyline through pure pursuit and the safety filter re-derives
    the commands: the raw speed, steering and curvature channels never reach
    the plant. Rejecting on them discarded collision-free geometry and drove
    the fallback rate from 24% to 76% in a closed-loop run.
    """
    ranked = []
    for index, (candidate, report) in enumerate(
        zip(result.candidates, result.diagnostics)
    ):
        clean = collision_free[index] if collision_free is not None else None
        clearance = clearance_m[index] if clearance_m is not None else None

        rejected = "collides" if clean is False else None

        ranked.append(
            RankedCandidate(
                candidate=candidate,
                diagnostics=report,
                collision_free=clean,
                clearance_m=clearance,
                rejected=rejected,
            )
        )

    def key(item: RankedCandidate):
        return (
            0 if item.collision_free is not False else 1,
            0 if item.diagnostics.dynamics_ok else 1,
            -(item.clearance_m if item.clearance_m is not None else 0.0),
            item.diagnostics.goal_distance_m,
            item.diagnostics.path_length_m,
        )

    return sorted(ranked, key=key)


def best_candidate(
    result: PlanResult,
    collision_free: list[bool] | None = None,
    clearance_m: list[float] | None = None,
) -> RankedCandidate | None:
    """Top-ranked candidate, or None when every one was rejected.

    None is the signal to fall back to the classical planner, which CLAUDE.md
    requires be logged and labelled rather than presented as the learned result.
    """
    ranked = rank_candidates(result, collision_free, clearance_m)
    if not ranked:
        return None
    return ranked[0] if ranked[0].usable else None
