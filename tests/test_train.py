"""Checkpoint plumbing: partial weight transfer and run resumption.

The unit tests here need no dataset. The integration test at the bottom does,
and it is the one that actually matters: it asserts a resumed run reproduces an
uninterrupted one epoch for epoch.
"""

import math

import pytest
import torch

from backend.ml.dataset import REAL_DATA_ROOT
from backend.ml.model import FlowMatchingModel
from backend.ml.train import (
    TrainConfig,
    TrainHistory,
    restore_run_state,
    train,
    transfer_weights,
)

FUSE = "condition_encoder.fuse.0.weight"


def _run_state(model, optimiser, scheduler, generator, epoch=2, history=None):
    """A payload shaped like the one `save_latest` writes."""
    history = history or TrainHistory(
        train_loss=[1.0, 0.9], val_loss=[1.1, 1.0], best_val=1.0,
        best_epoch=2, epochs_run=2,
    )
    return {
        "model_state": model.state_dict(),
        "history": vars(history),
        "resume": {
            "epoch": epoch,
            "optimiser_state": optimiser.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "generator_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        },
    }


def _parts(model, epochs=4):
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    generator = torch.Generator().manual_seed(0)
    return optimiser, scheduler, generator


def test_transfer_into_same_architecture_is_complete():
    source = FlowMatchingModel()
    torch.nn.init.normal_(source.output_projection.weight, std=0.02)

    target = FlowMatchingModel()
    report = transfer_weights(target, source.state_dict())

    assert report.fully_transferred
    assert not report.shape_mismatch and not report.missing
    for name, tensor in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], tensor), name


def test_transfer_static_to_dynamic_keeps_the_reusable_layers():
    """The weight-transfer table in the handoff, asserted.

    Turning on dynamic conditioning widens `fuse.0`'s input, so that tensor
    cannot carry over, and the dynamic encoder does not exist in the source.
    Everything else must survive.
    """
    source = FlowMatchingModel(use_dynamic=False)
    torch.nn.init.normal_(source.output_projection.weight, std=0.02)

    target = FlowMatchingModel(use_dynamic=True)
    report = transfer_weights(target, source.state_dict())

    assert not report.fully_transferred
    assert [name for name, _, _ in report.shape_mismatch] == [FUSE]
    assert all("dynamic_encoder" in name for name in report.missing)

    moved = target.state_dict()
    assert torch.equal(moved["blocks.0.input_layer.weight"],
                       source.state_dict()["blocks.0.input_layer.weight"])
    occupancy = "condition_encoder.occupancy_encoder.project.weight"
    assert torch.equal(moved[occupancy], source.state_dict()[occupancy])


def test_transfer_leaves_mismatched_tensors_at_their_initialisation():
    source = FlowMatchingModel(use_dynamic=False)
    target = FlowMatchingModel(use_dynamic=True)
    before = target.state_dict()[FUSE].clone()

    transfer_weights(target, source.state_dict())

    assert torch.equal(target.state_dict()[FUSE], before)


def test_transfer_reports_checkpoint_tensors_the_model_does_not_have():
    source = FlowMatchingModel(use_dynamic=True)
    report = transfer_weights(FlowMatchingModel(use_dynamic=False),
                              source.state_dict())

    assert any("dynamic_encoder" in name for name in report.unexpected)
    assert "ignored (absent from model)" in report.summary()


def test_transfer_across_widths_still_moves_what_fits():
    source = FlowMatchingModel(hidden_dim=256, num_blocks=3)
    report = transfer_weights(FlowMatchingModel(hidden_dim=96, num_blocks=2),
                              source.state_dict())

    assert report.transferred
    assert report.shape_mismatch


def test_resume_refuses_a_checkpoint_without_run_state():
    """The best-val checkpoint has no optimiser state; only *_latest.pt does."""
    model = FlowMatchingModel()
    with pytest.raises(ValueError, match="no resume state"):
        restore_run_state({"model_state": model.state_dict()},
                          model, *_parts(model), epochs=4)


def test_resume_refuses_an_architecture_change():
    source = FlowMatchingModel(hidden_dim=96, num_blocks=2)
    payload = _run_state(source, *_parts(source))

    target = FlowMatchingModel(hidden_dim=256, num_blocks=3)
    with pytest.raises(ValueError, match="different architecture"):
        restore_run_state(payload, target, *_parts(target), epochs=4)


def test_resume_restores_epoch_history_and_schedule():
    source = FlowMatchingModel(hidden_dim=96, num_blocks=2)
    optimiser, scheduler, generator = _parts(source, epochs=4)
    for _ in range(2):
        optimiser.step()
        scheduler.step()
    payload = _run_state(source, optimiser, scheduler, generator, epoch=2)

    target = FlowMatchingModel(hidden_dim=96, num_blocks=2)
    parts = _parts(target, epochs=4)
    epoch, history = restore_run_state(payload, target, *parts, epochs=10)

    assert epoch == 2
    assert history.epochs_run == 2 and history.best_epoch == 2
    assert history.val_loss == [1.1, 1.0]
    assert parts[1].T_max == 10
    assert parts[1].last_epoch == 2


def test_resume_revives_a_learning_rate_that_annealed_to_zero():
    """Extending a finished run must not train at a dead learning rate.

    A completed cosine budget leaves the optimiser at eta_min. The scheduler
    only recomputes after an epoch, so without re-deriving the rate the first
    resumed epoch moves no weights at all - the symptom is a validation loss
    that comes back bit-identical to the previous epoch's.
    """
    source = FlowMatchingModel(hidden_dim=96, num_blocks=2)
    optimiser, scheduler, generator = _parts(source, epochs=2)
    for _ in range(2):
        optimiser.step()
        scheduler.step()
    assert optimiser.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)

    payload = _run_state(source, optimiser, scheduler, generator, epoch=2)
    target = FlowMatchingModel(hidden_dim=96, num_blocks=2)
    parts = _parts(target, epochs=2)
    restore_run_state(payload, target, *parts, epochs=10)

    revived = parts[0].param_groups[0]["lr"]
    expected = 1e-3 * (1 + math.cos(math.pi * 2 / 10)) / 2
    assert revived == pytest.approx(expected)
    assert revived > 0


@pytest.mark.skipif(
    not REAL_DATA_ROOT.exists(), reason="expert dataset not present"
)
def test_resumed_run_reproduces_an_uninterrupted_one(tmp_path, monkeypatch):
    """The claim `--resume` has to earn.

    A 3-epoch run is interrupted after epoch 1 and resumed. Its remaining
    epochs must match the same run left alone - which only holds if the
    optimiser moments, the cosine position and both RNG streams all came back.
    """
    import backend.ml.train as train_module

    def config(**extra):
        return TrainConfig(
            epochs=3, hidden_dim=96, num_blocks=2, patience=99,
            batch_size=256, **extra,
        )

    reference = train(config(), tmp_path / "ref.pt")

    real_evaluate = train_module.evaluate
    calls = {"n": 0}

    def crash_after_first_epoch(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise KeyboardInterrupt("simulated crash")
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(train_module, "evaluate", crash_after_first_epoch)
    with pytest.raises(KeyboardInterrupt):
        train(config(), tmp_path / "run.pt")
    monkeypatch.setattr(train_module, "evaluate", real_evaluate)

    assert (tmp_path / "run_latest.pt").exists()

    resumed = train(
        config(resume_from=str(tmp_path / "run_latest.pt")), tmp_path / "run.pt"
    )

    assert resumed["history"]["epochs_run"] == 3
    assert resumed["history"]["val_loss"] == pytest.approx(
        reference["history"]["val_loss"], abs=1e-9
    )
