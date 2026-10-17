"""Coordinate conversions between dataset frame, backend frame, and Three.js.

Dataset / data-pipeline frame (existing corpus — do not change on disk):
  - Horizontal: X, Z
  - Elevation: Y
  - Heading yaw: rotation about +Y, 0 along +X, CCW toward +Z

Backend frame (CLAUDE.md):
  - Horizontal: X, Y
  - Elevation: Z
  - Heading psi: rotation about +Z, 0 along +X, CCW toward +Y

Three.js / R3F (Y-up), from CLAUDE.md:
  (x, y, z)_backend -> (x, z, -y)_frontend
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def dataset_xz_to_backend_xy(
    x: float, z: float, y: float = 0.0, yaw: float = 0.0
) -> tuple[float, float, float, float]:
    """Map dataset (x, z, y, yaw) -> backend (x, y, z, psi)."""
    return float(x), float(z), float(y), float(yaw)


def backend_xy_to_dataset_xz(
    x: float, y: float, z: float = 0.0, psi: float = 0.0
) -> tuple[float, float, float, float]:
    """Map backend (x, y, z, psi) -> dataset (x, z, y, yaw)."""
    return float(x), float(y), float(z), float(psi)


def backend_to_threejs(
    x: float, y: float, z: float
) -> tuple[float, float, float]:
    """CLAUDE.md: (x, y, z)_backend -> (x, z, -y)_frontend."""
    return float(x), float(z), float(-y)


def threejs_to_backend(
    x: float, y: float, z: float
) -> tuple[float, float, float]:
    """Inverse of backend_to_threejs."""
    return float(x), float(-z), float(y)


def dataset_to_threejs(
    x: float, z: float, y: float = 0.0
) -> tuple[float, float, float]:
    """Dataset XZ-horizontal -> Three.js Y-up via backend mapping."""
    bx, by, bz, _ = dataset_xz_to_backend_xy(x, z, y, 0.0)
    return backend_to_threejs(bx, by, bz)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def transform_path_dataset_to_backend(
    points: Sequence[Sequence[float]],
) -> np.ndarray:
    """Nx2+ array with columns x,z[,yaw,...] -> x,y[,psi,...] (elevation dropped/ignored)."""
    arr = np.asarray(points, dtype=np.float64)
    out = arr.copy()
    # x stays; former z becomes backend y
    out[:, 0] = arr[:, 0]
    out[:, 1] = arr[:, 1]
    if arr.shape[1] > 2:
        out[:, 2] = arr[:, 2]  # yaw/psi unchanged numerically under this mapping
    return out
