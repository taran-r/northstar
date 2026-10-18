"""Conditional flow-matching training.

Objective (Handoff section 5C): with tau an expert trajectory and eps gaussian
noise of the same shape, sample t ~ U(0,1), form z_t = (1-t)*eps + t*tau, and
regress the model onto v = tau - eps. The network therefore learns a velocity
field over trajectory space, which `inference.py` integrates from t=0 to t=1.

Validation reuses a fixed noise/time seed every epoch, so the reported curve
reflects the model improving rather than the luck of the draw.

Usage:
    python -m backend.ml.train --epochs 40
    python -m backend.ml.train --no-include-oracle --out backend/artifacts/no_oracle.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from backend.ml.dataset import (
    CANONICAL_SPLITS,
    COND_FEATURES,
    DT,
    DYNAMIC_SLICES,
    HORIZON,
    OCC_CHANNELS,
    OCC_GRID,
    REAL_DATA_ROOT,
    STEP_FEATURES,
    CombinedBundle,
    DatasetBundle,
    build_combined_datasets,
    build_datasets,
    load_vehicle_params,
)
from backend.ml.model import HIDDEN_DIM, NUM_BLOCKS, FlowMatchingModel

DEFAULT_OUT = Path("backend/artifacts/flow_matching.pt")
VAL_SEED = 1234


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 25
    min_delta: float = 1e-4
    heading_weight: float = 0.0
    speed_weight: float = 0.0
    unit_weight: float = 0.0
    seed: int = 0
    include_oracle: bool = True
    use_dynamic: bool = False
    hidden_dim: int = HIDDEN_DIM
    num_blocks: int = NUM_BLOCKS
    dropout: float = 0.0
    combined: bool = False
    init_from: str | None = None
    resume_from: str | None = None
    splits_path: str = str(CANONICAL_SPLITS)
    data_root: str = str(REAL_DATA_ROOT)


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_v1: list[float] = field(default_factory=list)
    val_v2: list[float] = field(default_factory=list)
    best_val: float = float("inf")
    best_epoch: int = -1
    epochs_run: int = 0
    stopped_early: bool = False
    seconds: float = 0.0


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flow_matching_loss(
    model: FlowMatchingModel,
    batch: dict[str, torch.Tensor],
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    target = batch["trajectory"].to(device)
    occupancy = batch["occupancy"].to(device)
    cond = batch["cond"].to(device)
    dynamic = batch["dynamic"].to(device) if model.use_dynamic else None

    noise = torch.randn(target.shape, generator=generator, device=device)
    t = torch.rand(target.shape[0], generator=generator, device=device)
    noisy = (1 - t.reshape(-1, 1, 1)) * noise + t.reshape(-1, 1, 1) * target

    predicted = model(noisy, t, occupancy, cond, dynamic)
    return torch.nn.functional.mse_loss(predicted, target - noise)


MIN_MOVE_M = 0.02


def consistency_losses(
    clean: torch.Tensor,
    traj_mean: torch.Tensor,
    traj_std: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Physical coupling between the five predicted channels.

    Plain MSE treats position, heading and speed as independent, and the
    baseline exploits that: measured on held-out windows its heading sits 43
    degrees from its own direction of travel (expert: 2.3) and its step
    lengths disagree with its own speed channel. These terms tie the channels
    together in physical units.

    `clean` is the trajectory the model implies for t=1, recovered as
    z_t + (1 - t) * v_hat, which is exact when v_hat equals the true velocity.

    Returned values are per sample, so the caller can weight each one by its
    own t rather than by a batch average.
    """
    trajectory = clean * traj_std + traj_mean
    position = trajectory[..., :2]
    sin_yaw, cos_yaw = trajectory[..., 2], trajectory[..., 3]
    speed = trajectory[..., 4]

    travel = position[:, 1:] - position[:, :-1]
    distance = torch.linalg.vector_norm(travel, dim=-1)
    weight = (distance > MIN_MOVE_M).float()
    denominator = weight.sum(dim=-1).clamp(min=1.0)

    travel_unit = travel / distance.clamp(min=1e-3).unsqueeze(-1)
    heading = torch.stack([cos_yaw[:, :-1], sin_yaw[:, :-1]], dim=-1)
    heading_unit = heading / torch.linalg.vector_norm(
        heading, dim=-1, keepdim=True
    ).clamp(min=1e-3)
    alignment = (travel_unit * heading_unit).sum(dim=-1)
    heading_loss = ((1.0 - alignment) * weight).sum(dim=-1) / denominator

    midpoint_speed = 0.5 * (speed[:, 1:] + speed[:, :-1])
    speed_loss = (
        ((distance / dt) - midpoint_speed) ** 2 * weight
    ).sum(dim=-1) / denominator

    unit_loss = ((sin_yaw**2 + cos_yaw**2 - 1.0) ** 2).mean(dim=-1)
    return heading_loss, speed_loss, unit_loss


def composite_loss(
    model: FlowMatchingModel,
    batch: dict[str, torch.Tensor],
    generator: torch.Generator,
    device: torch.device,
    config: TrainConfig,
    traj_mean: torch.Tensor,
    traj_std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["trajectory"].to(device)
    occupancy = batch["occupancy"].to(device)
    cond = batch["cond"].to(device)
    dynamic = batch["dynamic"].to(device) if model.use_dynamic else None

    noise = torch.randn(target.shape, generator=generator, device=device)
    t = torch.rand(target.shape[0], generator=generator, device=device)
    scalar_t = t.reshape(-1, 1, 1)
    noisy = (1 - scalar_t) * noise + scalar_t * target

    predicted = model(noisy, t, occupancy, cond, dynamic)
    flow = torch.nn.functional.mse_loss(predicted, target - noise)

    uses_auxiliary = (
        config.heading_weight or config.speed_weight or config.unit_weight
    )
    if not uses_auxiliary:
        return flow, {"flow": flow.item()}

    clean = noisy + (1 - scalar_t) * predicted
    heading, speed, unit = consistency_losses(clean, traj_mean, traj_std, DT)

    # One-step reconstruction from a nearly pure-noise z_t carries no physical
    # meaning, so weight each sample by its own t (squared, to concentrate the
    # signal near t=1) instead of by a batch average.
    ramp = t**2
    auxiliary = (
        config.heading_weight * heading
        + config.speed_weight * speed
        + config.unit_weight * unit
    )
    total = flow + (ramp * auxiliary).mean()
    return total, {
        "flow": flow.item(),
        "heading": heading.mean().item(),
        "speed": speed.mean().item(),
        "unit": unit.mean().item(),
    }


def train_one_epoch(
    model: FlowMatchingModel,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    generator: torch.Generator,
    device: torch.device,
    config: TrainConfig,
    traj_mean: torch.Tensor,
    traj_std: torch.Tensor,
) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        loss, _ = composite_loss(
            model, batch, generator, device, config, traj_mean, traj_std
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(
    model: FlowMatchingModel, loader: DataLoader, device: torch.device
) -> float:
    """Pure flow-matching loss, so curves stay comparable across weightings."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(VAL_SEED)
    total = 0.0
    for batch in loader:
        total += flow_matching_loss(model, batch, generator, device).item()
    return total / len(loader)


@dataclass
class TransferReport:
    """What a partial weight load actually moved.

    Kept as data rather than just printed so tests and callers can assert on
    it. A silent partial load is how a half-random model ends up in a results
    table, so `summary()` names every tensor that did not transfer.
    """

    transferred: list[str] = field(default_factory=list)
    shape_mismatch: list[tuple[str, tuple, tuple]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    @property
    def fully_transferred(self) -> bool:
        return not (self.shape_mismatch or self.missing or self.unexpected)

    def summary(self) -> str:
        lines = [f"transferred {len(self.transferred)} tensors"]
        for name, source, target in self.shape_mismatch:
            lines.append(
                f"  re-initialised {name}: checkpoint {tuple(source)} "
                f"!= model {tuple(target)}"
            )
        for name in self.missing:
            lines.append(f"  fresh (absent from checkpoint) {name}")
        for name in self.unexpected:
            lines.append(f"  ignored (absent from model) {name}")
        return "\n".join(lines)


def transfer_weights(
    model: FlowMatchingModel, state: dict[str, torch.Tensor]
) -> TransferReport:
    """Copy every tensor whose name and shape agree; leave the rest fresh.

    Enabling dynamic conditioning widens `condition_encoder.fuse.0.weight`, and
    changing `--hidden-dim` changes most tensors. Strict loading refuses the
    whole checkpoint in those cases, which is wrong for fine-tuning: the
    encoder and residual blocks are still worth keeping. The untransferred
    tensors keep their constructor initialisation.
    """
    report = TransferReport()
    target = model.state_dict()

    for name, tensor in target.items():
        if name not in state:
            report.missing.append(name)
        elif state[name].shape != tensor.shape:
            report.shape_mismatch.append((name, state[name].shape, tensor.shape))
        else:
            tensor.copy_(state[name])
            report.transferred.append(name)

    report.unexpected = [name for name in state if name not in target]
    model.load_state_dict(target)
    return report


def save_latest(
    path: Path,
    model: FlowMatchingModel,
    bundle: DatasetBundle,
    config: TrainConfig,
    history: TrainHistory,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
) -> None:
    """Write the resumable checkpoint.

    Called after every epoch, not just at the end: a run that dies at epoch
    120 is exactly the case `--resume` exists for, and a file written only on
    clean exit would not be there. This is the newest weights, which is a
    different thing from `out_path` - that one holds the best val loss.
    """
    payload = build_checkpoint(model, bundle, config, history)
    payload["resume"] = {
        "epoch": history.epochs_run,
        "optimiser_state": optimiser.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "generator_state": generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(payload, path)


def restore_run_state(
    payload: dict,
    model: FlowMatchingModel,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    epochs: int,
) -> tuple[int, TrainHistory]:
    """Put an interrupted run back exactly where it stopped.

    Weights alone are not enough to continue a run: AdamW carries per-parameter
    moment estimates, the cosine schedule carries its position, and the data
    shuffle and noise draws come from RNG state. Restoring only the weights
    produces a different experiment that merely starts from the same place.

    Requires a `*_latest.pt`, the only variant that carries `resume`. Shapes
    must match exactly, because optimiser state is keyed to parameter shapes.
    """
    if "resume" not in payload:
        raise ValueError(
            "checkpoint carries no resume state - pass the *_latest.pt variant, "
            "not the best-val checkpoint"
        )

    report = transfer_weights(model, payload["model_state"])
    if not report.fully_transferred:
        raise ValueError(
            "cannot resume into a different architecture; optimiser state is "
            f"keyed to parameter shapes:\n{report.summary()}"
        )

    state = payload["resume"]
    optimiser.load_state_dict(state["optimiser_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    scheduler.T_max = epochs

    for group, base in zip(optimiser.param_groups, scheduler.base_lrs):
        group["lr"] = scheduler.eta_min + (base - scheduler.eta_min) * (
            1 + math.cos(math.pi * scheduler.last_epoch / scheduler.T_max)
        ) / 2
    scheduler._last_lr = [group["lr"] for group in optimiser.param_groups]
    generator.set_state(state["generator_state"])
    torch.set_rng_state(state["torch_rng_state"])

    history = TrainHistory(**payload["history"])
    return int(state["epoch"]), history


def build_checkpoint(
    model: FlowMatchingModel,
    bundle: DatasetBundle,
    config: TrainConfig,
    history: TrainHistory,
) -> dict:
    return {
        "model_state": model.state_dict(),
        "normalizer": bundle.normalizer.to_dict(),
        "splits": _bundle_splits(bundle),
        "vehicle": load_vehicle_params(Path(config.data_root)).tolist(),
        "config": asdict(config),
        "history": asdict(history),
        "schema": {
            "horizon": HORIZON,
            "dt": DT,
            "step_features": STEP_FEATURES,
            "occ_channels": OCC_CHANNELS,
            "occ_grid": OCC_GRID,
            "cond_features": COND_FEATURES,
            "use_dynamic": config.use_dynamic,
            "dynamic_slices": DYNAMIC_SLICES,
            "hidden_dim": config.hidden_dim,
            "num_blocks": config.num_blocks,
            "dropout": config.dropout,
        },
        "counts": _bundle_counts(bundle),
        "splits_provenance": _bundle_provenance(bundle),
    }


def _bundle_splits(bundle) -> dict:
    if isinstance(bundle, CombinedBundle):
        return bundle.splits
    return bundle.splits.to_dict()


def _bundle_counts(bundle) -> dict:
    if isinstance(bundle, CombinedBundle):
        return dict(bundle.counts)
    return {
        "train_windows": len(bundle.train),
        "val_windows": len(bundle.val),
        "test_windows": len(bundle.test),
        "train_scenarios": len(bundle.splits.scenarios("train")),
        "val_scenarios": len(bundle.splits.scenarios("val")),
        "test_scenarios": len(bundle.splits.scenarios("test")),
    }


def _bundle_provenance(bundle) -> dict:
    if isinstance(bundle, CombinedBundle):
        return {
            root: payload.get("provenance", {})
            for root, payload in bundle.splits.items()
        }
    return bundle.splits.provenance


def train(config: TrainConfig, out_path: Path) -> dict:
    torch.manual_seed(config.seed)
    device = resolve_device()
    print(f"device: {device}")

    started = time.perf_counter()
    if config.combined:
        bundle = build_combined_datasets(include_oracle=config.include_oracle)
        print(
            f"data: {bundle.counts['train']} train windows over "
            f"{bundle.counts['train_scenarios']} scenarios (v1+v2 via "
            f"CombinedTrainDataset), val {bundle.counts['val_v1']} v1 / "
            f"{bundle.counts['val_v2']} v2, test {bundle.counts['test_v1']} v1 / "
            f"{bundle.counts['test_v2']} v2, "
            f"oracle={'included' if config.include_oracle else 'excluded'}, "
            f"loaded in {time.perf_counter() - started:.1f}s"
        )
        val_sets = {"v1": bundle.val_v1, "v2": bundle.val_v2}
    else:
        bundle = build_datasets(
            root=Path(config.data_root),
            splits_path=Path(config.splits_path),
            include_oracle=config.include_oracle,
        )
        print(
            f"data: {len(bundle.train)} train / {len(bundle.val)} val / "
            f"{len(bundle.test)} test windows "
            f"({len(bundle.splits.scenarios('train'))}/"
            f"{len(bundle.splits.scenarios('val'))}/"
            f"{len(bundle.splits.scenarios('test'))} scenarios), "
            f"oracle={'included' if config.include_oracle else 'excluded'}, "
            f"loaded in {time.perf_counter() - started:.1f}s"
        )
        val_sets = {"v1": bundle.val}

    train_loader = DataLoader(bundle.train, batch_size=config.batch_size, shuffle=True)
    val_loaders = {
        name: DataLoader(data, batch_size=config.batch_size)
        for name, data in val_sets.items()
    }

    model = FlowMatchingModel(
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        use_dynamic=config.use_dynamic,
        dropout=config.dropout,
    ).to(device)
    print(
        f"parameters: {sum(p.numel() for p in model.parameters()):,}  "
        f"dynamic conditioning: {'on' if config.use_dynamic else 'off'}"
    )

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, config.epochs)
    generator = torch.Generator(device=device).manual_seed(config.seed + 1)
    traj_mean = torch.as_tensor(bundle.normalizer.traj_mean, device=device)
    traj_std = torch.as_tensor(bundle.normalizer.traj_std, device=device)

    if config.init_from and config.resume_from:
        raise ValueError("--init-from and --resume are mutually exclusive")

    history = TrainHistory()
    start_epoch = 0

    if config.init_from:
        payload = torch.load(
            config.init_from, map_location=device, weights_only=False
        )
        report = transfer_weights(model, payload["model_state"])
        print(f"init-from {config.init_from}\n{report.summary()}")

    if config.resume_from:
        payload = torch.load(
            config.resume_from, map_location=device, weights_only=False
        )
        start_epoch, history = restore_run_state(
            payload, model, optimiser, scheduler, generator, config.epochs
        )
        print(
            f"resumed {config.resume_from} at epoch {start_epoch} "
            f"(best val {history.best_val:.4f} at epoch {history.best_epoch})"
        )
        if start_epoch >= config.epochs:
            print(
                f"nothing to do: already ran {start_epoch} of {config.epochs} "
                f"epochs - raise --epochs to continue"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path = out_path.with_name(f"{out_path.stem}_latest.pt")
    clock = time.perf_counter()
    stale_epochs = (
        max(0, history.epochs_run - history.best_epoch)
        if history.best_epoch > 0
        else 0
    )

    for epoch in range(start_epoch + 1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimiser, generator, device,
            config, traj_mean, traj_std,
        )
        per_corpus = {
            name: evaluate(model, loader, device)
            for name, loader in val_loaders.items()
        }
        weights = {name: len(val_sets[name]) for name in val_loaders}
        total_weight = sum(weights.values())
        val_loss = sum(
            per_corpus[name] * weights[name] for name in per_corpus
        ) / total_weight
        scheduler.step()

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_v1.append(per_corpus.get("v1", float("nan")))
        history.val_v2.append(per_corpus.get("v2", float("nan")))
        history.epochs_run = epoch

        marker = ""
        if val_loss < history.best_val - config.min_delta:
            history.best_val = val_loss
            history.best_epoch = epoch
            history.seconds = time.perf_counter() - clock
            stale_epochs = 0
            torch.save(build_checkpoint(model, bundle, config, history), out_path)
            marker = "  <- best"
        else:
            stale_epochs += 1

        save_latest(
            latest_path, model, bundle, config, history,
            optimiser, scheduler, generator,
        )

        detail = "  ".join(f"{name} {value:.4f}" for name, value in per_corpus.items())
        print(
            f"epoch {epoch:3d}/{config.epochs}  "
            f"train {train_loss:.4f}  val {val_loss:.4f}  ({detail}){marker}"
        )

        if stale_epochs >= config.patience:
            history.stopped_early = True
            print(
                f"\nstopping early: no val improvement in {config.patience} epochs "
                f"(best {history.best_val:.4f} at epoch {history.best_epoch})"
            )
            break

    history.seconds = time.perf_counter() - clock

    save_latest(
        latest_path, model, bundle, config, history, optimiser, scheduler, generator
    )
    checkpoint = build_checkpoint(model, bundle, config, history)
    sidecar = out_path.with_suffix(".yaml")
    sidecar.write_text(
        yaml.safe_dump(
            {
                "config": checkpoint["config"],
                "schema": checkpoint["schema"],
                "counts": checkpoint["counts"],
                "best_val": history.best_val,
                "best_epoch": history.best_epoch,
                "epochs_run": history.epochs_run,
                "stopped_early": history.stopped_early,
                "seconds": round(history.seconds, 1),
                "splits_provenance": checkpoint["splits_provenance"],
            },
            sort_keys=False,
        )
    )

    print(
        f"\nbest val {history.best_val:.4f} at epoch {history.best_epoch} "
        f"of {history.epochs_run} run in {history.seconds:.1f}s"
    )
    print(f"best:    {out_path}\nlatest:  {latest_path}\nconfig:  {sidecar}")
    return checkpoint


def parse_args() -> tuple[TrainConfig, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--heading-weight", type=float, default=defaults.heading_weight)
    parser.add_argument("--speed-weight", type=float, default=defaults.speed_weight)
    parser.add_argument("--unit-weight", type=float, default=defaults.unit_weight)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--min-delta", type=float, default=defaults.min_delta)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=defaults.hidden_dim,
        help="residual block width; smaller shrinks the parameter count",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=defaults.num_blocks,
        help="number of residual blocks",
    )
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument(
        "--init-from",
        default=defaults.init_from,
        help="load weights from a checkpoint, fresh optimiser - fine-tuning",
    )
    restart.add_argument(
        "--resume",
        dest="resume_from",
        default=defaults.resume_from,
        help="continue a run: weights plus optimiser, scheduler and RNG "
             "state (requires a *_latest.pt)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=defaults.dropout,
        help="dropout inside each residual block",
    )
    parser.add_argument(
        "--combined",
        action=argparse.BooleanOptionalAction,
        default=defaults.combined,
        help="train on dataset_v1 + dataset_v2_additions via CombinedTrainDataset",
    )
    parser.add_argument("--splits", default=defaults.splits_path)
    parser.add_argument("--data-root", default=defaults.data_root)
    parser.add_argument(
        "--use-dynamic",
        action=argparse.BooleanOptionalAction,
        default=defaults.use_dynamic,
        help="add the time-aware predicted moving-obstacle branch",
    )
    parser.add_argument(
        "--include-oracle",
        action=argparse.BooleanOptionalAction,
        default=defaults.include_oracle,
        help="use windows whose expert planner saw ground-truth obstacle futures",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        heading_weight=args.heading_weight,
        speed_weight=args.speed_weight,
        unit_weight=args.unit_weight,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        include_oracle=args.include_oracle,
        use_dynamic=args.use_dynamic,
        init_from=args.init_from,
        resume_from=args.resume_from,
        dropout=args.dropout,
        combined=args.combined,
        splits_path=args.splits,
        data_root=args.data_root,
    )
    return config, args.out


def main() -> None:
    config, out_path = parse_args()
    train(config, out_path)


if __name__ == "__main__":
    main()
