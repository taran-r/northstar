import pytest
import torch
from torch.utils.data import DataLoader

from backend.ml.dataset import (
    COND_FEATURES,
    HORIZON,
    OCC_CHANNELS,
    OCC_GRID,
    REAL_DATA_ROOT,
    STEP_FEATURES,
    FlowMatchingDataset,
    Normalizer,
    load_canonical_splits,
)
from backend.ml.model import (
    COND_DIM,
    FlowMatchingModel,
    OCC_EMBED_DIM,
    TIME_EMBED_DIM,
    ConditionEncoder,
    OccupancyEncoder,
    timestep_embedding,
)

BATCH = 4


def _inputs():
    return (
        torch.rand(BATCH, OCC_CHANNELS, OCC_GRID, OCC_GRID).round(),
        torch.randn(BATCH, COND_FEATURES),
        torch.rand(BATCH),
    )


def test_timestep_embedding_shape_and_range():
    embedded = timestep_embedding(torch.tensor([0.0, 0.25, 1.0]))
    assert embedded.shape == (3, TIME_EMBED_DIM)
    assert embedded.abs().max() <= 1.0


def test_timestep_embedding_distinguishes_endpoints():
    embedded = timestep_embedding(torch.tensor([0.0, 1.0]))
    assert not torch.allclose(embedded[0], embedded[1])


def test_occupancy_encoder_output_shape():
    encoder = OccupancyEncoder()
    occupancy = torch.rand(BATCH, OCC_CHANNELS, OCC_GRID, OCC_GRID)
    assert encoder(occupancy).shape == (BATCH, OCC_EMBED_DIM)


def test_condition_encoder_output_shape_and_finiteness():
    encoder = ConditionEncoder()
    output = encoder(*_inputs())
    assert output.shape == (BATCH, COND_DIM)
    assert torch.isfinite(output).all()


def test_condition_encoder_responds_to_time():
    encoder = ConditionEncoder().eval()
    occupancy, cond, _ = _inputs()
    with torch.no_grad():
        early = encoder(occupancy, cond, torch.full((BATCH,), 0.1))
        late = encoder(occupancy, cond, torch.full((BATCH,), 0.9))
    assert not torch.allclose(early, late)


def test_condition_encoder_responds_to_occupancy():
    encoder = ConditionEncoder().eval()
    occupancy, cond, t = _inputs()
    with torch.no_grad():
        occupied = encoder(occupancy, cond, t)
        empty = encoder(torch.zeros_like(occupancy), cond, t)
    assert not torch.allclose(occupied, empty)


def test_parameter_budget_is_small():
    total = sum(p.numel() for p in ConditionEncoder().parameters())
    assert 50_000 < total < 2_000_000


def test_gradients_reach_every_parameter():
    encoder = ConditionEncoder()
    encoder(*_inputs()).sum().backward()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def _model_inputs(batch: int = BATCH):
    return (
        torch.randn(batch, HORIZON, STEP_FEATURES),
        torch.rand(batch),
        torch.rand(batch, OCC_CHANNELS, OCC_GRID, OCC_GRID).round(),
        torch.randn(batch, COND_FEATURES),
    )


def test_velocity_matches_trajectory_shape():
    model = FlowMatchingModel()
    noisy, t, occupancy, cond = _model_inputs()
    assert model(noisy, t, occupancy, cond).shape == noisy.shape


def test_output_projection_is_zero_initialised():
    model = FlowMatchingModel().eval()
    with torch.no_grad():
        velocity = model(*_model_inputs())
    assert torch.allclose(velocity, torch.zeros_like(velocity))


def test_flow_matching_parameter_budget():
    total = sum(p.numel() for p in FlowMatchingModel().parameters())
    assert 100_000 < total < 3_000_000


def test_gradients_reach_every_flow_parameter():
    model = FlowMatchingModel()
    model(*_model_inputs()).sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.skipif(
    not REAL_DATA_ROOT.exists(), reason="expert dataset not present"
)
def test_model_learns_on_real_conditioned_data():
    """Random targets have no learnable link to the conditioning, so this uses
    real pairs: the point is that conditioning carries trajectory signal."""
    torch.manual_seed(0)
    data = FlowMatchingDataset(load_canonical_splits().val[:40])
    data.normalizer = Normalizer.fit(data.samples)
    batch = next(iter(DataLoader(data, batch_size=8)))
    target, occupancy, cond = batch["trajectory"], batch["occupancy"], batch["cond"]

    model = FlowMatchingModel()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(1)

    losses = []
    for _ in range(3000):
        noise = torch.randn(target.shape, generator=generator)
        t = torch.rand(target.shape[0], generator=generator)
        noisy = (1 - t.reshape(-1, 1, 1)) * noise + t.reshape(-1, 1, 1) * target
        loss = torch.nn.functional.mse_loss(
            model(noisy, t, occupancy, cond), target - noise
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())

    assert sum(losses[-50:]) / 50 < sum(losses[:50]) / 50 * 0.7


def test_dynamic_encoder_output_shape():
    from backend.ml.dataset import DYNAMIC_SLICES
    from backend.ml.model import DYNAMIC_EMBED_DIM, DynamicOccupancyEncoder

    encoder = DynamicOccupancyEncoder()
    forecast = torch.rand(BATCH, DYNAMIC_SLICES, OCC_GRID, OCC_GRID).round()
    assert encoder(forecast).shape == (BATCH, DYNAMIC_EMBED_DIM)


def test_dynamic_encoder_respects_slice_order():
    """A temporal encoder must distinguish an obstacle approaching from one
    receding; a plain channel-stack CNN would not."""
    from backend.ml.dataset import DYNAMIC_SLICES
    from backend.ml.model import DynamicOccupancyEncoder

    torch.manual_seed(0)
    encoder = DynamicOccupancyEncoder().eval()
    torch.nn.init.normal_(encoder.step_embedding, std=0.5)

    forecast = torch.zeros(1, DYNAMIC_SLICES, OCC_GRID, OCC_GRID)
    for step in range(DYNAMIC_SLICES):
        forecast[0, step, 20, 10 + step] = 1.0

    with torch.no_grad():
        forward = encoder(forecast)
        reversed_ = encoder(forecast.flip(1))
    assert not torch.allclose(forward, reversed_)


def test_dynamic_model_requires_the_forecast():
    from backend.ml.dataset import DYNAMIC_SLICES

    model = FlowMatchingModel(use_dynamic=True)
    noisy, t, occupancy, cond = _model_inputs()
    with pytest.raises(ValueError, match="pass `dynamic`"):
        model(noisy, t, occupancy, cond)

    forecast = torch.rand(BATCH, DYNAMIC_SLICES, OCC_GRID, OCC_GRID).round()
    assert model(noisy, t, occupancy, cond, forecast).shape == noisy.shape


def test_dynamic_branch_costs_few_parameters():
    static_only = sum(p.numel() for p in FlowMatchingModel().parameters())
    with_dynamic = sum(
        p.numel() for p in FlowMatchingModel(use_dynamic=True).parameters()
    )
    assert with_dynamic > static_only
    assert with_dynamic - static_only < 200_000


def test_dynamic_output_responds_to_the_forecast():
    from backend.ml.dataset import DYNAMIC_SLICES

    torch.manual_seed(0)
    model = FlowMatchingModel(use_dynamic=True)
    torch.nn.init.normal_(model.output_projection.weight, std=0.02)
    model.eval()

    noisy, t, occupancy, cond = _model_inputs()
    busy = torch.rand(BATCH, DYNAMIC_SLICES, OCC_GRID, OCC_GRID).round()
    with torch.no_grad():
        crowded = model(noisy, t, occupancy, cond, busy)
        clear = model(noisy, t, occupancy, cond, torch.zeros_like(busy))
    assert not torch.allclose(crowded, clear)
