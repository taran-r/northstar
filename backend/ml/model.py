"""Conditional flow-matching network.

`ConditionEncoder` summarizes the situation (occupancy raster, scalar features,
flow time t) into one vector. `FlowMatchingModel` predicts the velocity field
v_theta(z_t, t, condition) over trajectory space: given a partly-noisy
trajectory, which direction moves it toward an expert-like one.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from backend.ml.dataset import (
    COND_FEATURES,
    DYNAMIC_SLICES,
    HORIZON,
    OCC_CHANNELS,
    OCC_GRID,
    STEP_FEATURES,
)

OCC_STAGES = (16, 32, 64)
OCC_EMBED_DIM = 128
DYNAMIC_STAGES = (8, 16, 32)
DYNAMIC_SLICE_DIM = 64
DYNAMIC_EMBED_DIM = 64
TIME_EMBED_DIM = 64
COND_DIM = 256
GROUP_NORM_GROUPS = 8
TIME_EMBED_SCALE = 1000.0
HIDDEN_DIM = 256
NUM_BLOCKS = 3


def timestep_embedding(t: torch.Tensor, dim: int = TIME_EMBED_DIM) -> torch.Tensor:
    half = dim // 2
    exponents = torch.arange(half, dtype=torch.float32, device=t.device) / half
    frequencies = torch.exp(-math.log(TIME_EMBED_SCALE) * exponents)
    angles = t.reshape(-1, 1) * frequencies.reshape(1, -1) * TIME_EMBED_SCALE
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


class OccupancyEncoder(nn.Module):
    def __init__(
        self,
        grid_size: int = OCC_GRID,
        embed_dim: int = OCC_EMBED_DIM,
        in_channels: int = OCC_CHANNELS,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        for out_channels in OCC_STAGES:
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(GROUP_NORM_GROUPS, out_channels),
                nn.SiLU(),
            ]
            in_channels = out_channels

        self.backbone = nn.Sequential(*layers)
        reduced = grid_size // (2 ** len(OCC_STAGES))
        self.project = nn.Linear(in_channels * reduced * reduced, embed_dim)

    def forward(self, occupancy: torch.Tensor) -> torch.Tensor:
        features = self.backbone(occupancy)
        return self.project(features.flatten(start_dim=1))


class DynamicOccupancyEncoder(nn.Module):
    """Time-aware encoder over the predicted moving-obstacle forecast.

    One CNN is shared across all slices, so the spatial reading of a forklift
    is the same whichever forecast step it appears in; a learned per-step
    embedding plus a causal-order 1D convolution then supply the time axis.
    Shared spatial weights keep this branch small relative to the static one.

    Input is the dynamic-only forecast from `dataset.dynamic_only`, never
    ground-truth future obstacle motion.
    """

    def __init__(
        self,
        grid_size: int = OCC_GRID,
        slices: int = DYNAMIC_SLICES,
        slice_dim: int = DYNAMIC_SLICE_DIM,
        embed_dim: int = DYNAMIC_EMBED_DIM,
    ):
        super().__init__()
        self.slices = slices

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in DYNAMIC_STAGES:
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(min(GROUP_NORM_GROUPS, out_channels), out_channels),
                nn.SiLU(),
            ]
            in_channels = out_channels
        self.spatial = nn.Sequential(*layers)

        reduced = grid_size // (2 ** len(DYNAMIC_STAGES))
        self.slice_projection = nn.Linear(in_channels * reduced * reduced, slice_dim)
        self.step_embedding = nn.Parameter(torch.zeros(slices, slice_dim))

        self.temporal = nn.Sequential(
            nn.Conv1d(slice_dim, slice_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(slice_dim, embed_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, dynamic: torch.Tensor) -> torch.Tensor:
        batch, slices = dynamic.shape[:2]
        flattened = dynamic.reshape(batch * slices, 1, *dynamic.shape[2:])

        features = self.spatial(flattened).flatten(start_dim=1)
        per_slice = self.slice_projection(features).reshape(batch, slices, -1)
        per_slice = per_slice + self.step_embedding

        temporal = self.temporal(per_slice.transpose(1, 2))
        return temporal.mean(dim=-1)


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        grid_size: int = OCC_GRID,
        cond_features: int = COND_FEATURES,
        cond_dim: int = COND_DIM,
        use_dynamic: bool = False,
    ):
        super().__init__()
        self.use_dynamic = use_dynamic
        self.occupancy_encoder = OccupancyEncoder(grid_size)
        self.dynamic_encoder = (
            DynamicOccupancyEncoder(grid_size) if use_dynamic else None
        )

        fused_width = OCC_EMBED_DIM + cond_features + TIME_EMBED_DIM
        if use_dynamic:
            fused_width += DYNAMIC_EMBED_DIM

        self.fuse = nn.Sequential(
            nn.Linear(fused_width, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(
        self,
        occupancy: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        dynamic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [
            self.occupancy_encoder(occupancy),
            cond,
            timestep_embedding(t).to(cond.dtype),
        ]
        if self.dynamic_encoder is not None:
            if dynamic is None:
                raise ValueError(
                    "this model was built with dynamic conditioning; pass `dynamic`"
                )
            parts.append(self.dynamic_encoder(dynamic))
        return self.fuse(torch.cat(parts, dim=-1))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        cond_dim: int = COND_DIM,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.input_layer = nn.Linear(hidden_dim, hidden_dim)
        self.cond_layer = nn.Linear(cond_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = self.norm(hidden)
        residual = self.activation(self.input_layer(residual) + self.cond_layer(condition))
        return hidden + self.output_layer(self.dropout(residual))


class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        grid_size: int = OCC_GRID,
        k_steps: int = HORIZON,
        step_features: int = STEP_FEATURES,
        cond_features: int = COND_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
        cond_dim: int = COND_DIM,
        num_blocks: int = NUM_BLOCKS,
        use_dynamic: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.k_steps = k_steps
        self.step_features = step_features
        self.use_dynamic = use_dynamic
        trajectory_dim = k_steps * step_features

        self.condition_encoder = ConditionEncoder(
            grid_size, cond_features, cond_dim, use_dynamic
        )
        self.input_projection = nn.Linear(trajectory_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden_dim, cond_dim, dropout) for _ in range(num_blocks)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, trajectory_dim)

        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        t: torch.Tensor,
        occupancy: torch.Tensor,
        cond: torch.Tensor,
        dynamic: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self.condition_encoder(occupancy, cond, t, dynamic)

        hidden = self.input_projection(noisy_trajectory.flatten(start_dim=1))
        for block in self.blocks:
            hidden = block(hidden, condition)

        velocity = self.output_projection(self.output_norm(hidden))
        return velocity.reshape_as(noisy_trajectory)
