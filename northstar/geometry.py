"""2D geometry helpers on the XZ ground plane (Y is elevation)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class AABB:
    """Axis-aligned bounding box on the XZ plane."""

    min_x: float
    min_z: float
    max_x: float
    max_z: float

    def inflate(self, margin: float) -> "AABB":
        return AABB(
            self.min_x - margin,
            self.min_z - margin,
            self.max_x + margin,
            self.max_z + margin,
        )

    def contains(self, x: float, z: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_z <= z <= self.max_z

    def intersects(self, other: "AABB") -> bool:
        return not (
            self.max_x < other.min_x
            or self.min_x > other.max_x
            or self.max_z < other.min_z
            or self.min_z > other.max_z
        )


@dataclass(frozen=True)
class OrientedBox:
    """Oriented rectangle footprint on the XZ plane."""

    x: float
    z: float
    yaw: float
    length: float  # along local +X
    width: float  # along local +Z

    def corners(self) -> np.ndarray:
        hx, hz = 0.5 * self.length, 0.5 * self.width
        local = np.array(
            [[-hx, -hz], [hx, -hz], [hx, hz], [-hx, hz]], dtype=np.float64
        )
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        world = local @ rot.T
        world[:, 0] += self.x
        world[:, 1] += self.z
        return world

    def aabb(self) -> AABB:
        corners = self.corners()
        return AABB(
            float(corners[:, 0].min()),
            float(corners[:, 1].min()),
            float(corners[:, 0].max()),
            float(corners[:, 1].max()),
        )

    def contains_point(self, x: float, z: float, margin: float = 0.0) -> bool:
        dx, dz = x - self.x, z - self.z
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        lx = c * dx + s * dz
        lz = -s * dx + c * dz
        return abs(lx) <= 0.5 * self.length + margin and abs(lz) <= 0.5 * self.width + margin


def wrap_angle(yaw: float) -> float:
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    return wrap_angle(a - b)


def point_in_polygon(x: float, z: float, polygon: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon test. polygon is list of [x, z]."""
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, zi = polygon[i]
        xj, zj = polygon[j]
        intersects = ((zi > z) != (zj > z)) and (
            x < (xj - xi) * (z - zi) / (zj - zi + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def segments_intersect(
    a1: Sequence[float],
    a2: Sequence[float],
    b1: Sequence[float],
    b2: Sequence[float],
) -> bool:
    def orient(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)
    if o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0:
        return False
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def oriented_boxes_collide(a: OrientedBox, b: OrientedBox, margin: float = 0.0) -> bool:
    """SAT collision between two oriented boxes with optional inflation."""
    if margin > 0:
        a = OrientedBox(a.x, a.z, a.yaw, a.length + 2 * margin, a.width + 2 * margin)
        b = OrientedBox(b.x, b.z, b.yaw, b.length + 2 * margin, b.width + 2 * margin)
    if not a.aabb().intersects(b.aabb()):
        return False

    axes = []
    for box in (a, b):
        c, s = math.cos(box.yaw), math.sin(box.yaw)
        axes.append(np.array([c, s]))
        axes.append(np.array([-s, c]))

    ca = a.corners()
    cb = b.corners()
    for axis in axes:
        pa = ca @ axis
        pb = cb @ axis
        if pa.max() < pb.min() or pb.max() < pa.min():
            return False
    return True


def circle_box_collide(
    cx: float, cz: float, radius: float, box: OrientedBox
) -> bool:
    dx, dz = cx - box.x, cz - box.z
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    lx = c * dx + s * dz
    lz = -s * dx + c * dz
    closest_x = max(-0.5 * box.length, min(0.5 * box.length, lx))
    closest_z = max(-0.5 * box.width, min(0.5 * box.width, lz))
    return (lx - closest_x) ** 2 + (lz - closest_z) ** 2 <= radius**2


def polyline_length(points: Iterable[Sequence[float]]) -> float:
    pts = list(points)
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(pts)):
        total += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    return total
