"""Signed clearance between the rover footprint and obstacle geometry.

Clearance is meters of separation on the XZ plane. Positive means the shapes
do not overlap. Negative means penetration depth (minimum translation that
separates them). The rover box is the oriented footprint from
``VehicleState.footprint``, which already includes ``VehicleParams.footprint_margin``.
"""

from __future__ import annotations

import math

import numpy as np

from northstar.geometry import OrientedBox


def closest_points_on_segments(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closest points on segments p1-q1 and p2-q2 (Ericson)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    eps = 1e-12
    if a <= eps and e <= eps:
        return p1, p2
    if a <= eps:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= eps:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            if denom > eps:
                s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
    return p1 + d1 * s, p2 + d2 * t


def _box_axes(box: OrientedBox) -> tuple[np.ndarray, np.ndarray]:
    c = math.cos(box.yaw)
    s = math.sin(box.yaw)
    return np.array([c, s], dtype=np.float64), np.array([-s, c], dtype=np.float64)


def _axis_aligned(yaw: float) -> bool:
    half_pi = math.pi / 2.0
    r = abs(yaw) % half_pi
    return min(r, half_pi - r) < 1e-3


def _aabb_bounds(box: OrientedBox) -> tuple[float, float, float, float]:
    c = abs(math.cos(box.yaw))
    s = abs(math.sin(box.yaw))
    hx = 0.5 * box.length * c + 0.5 * box.width * s
    hz = 0.5 * box.length * s + 0.5 * box.width * c
    return box.x - hx, box.z - hz, box.x + hx, box.z + hz


def _unit_from_centers(a: OrientedBox, b: OrientedBox) -> np.ndarray:
    dx = a.x - b.x
    dz = a.z - b.z
    nrm = math.hypot(dx, dz)
    if nrm < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return np.array([dx / nrm, dz / nrm], dtype=np.float64)


def _aligned_clearance(a: OrientedBox, b: OrientedBox) -> tuple[float, np.ndarray]:
    """Exact signed distance for rectangles whose edges are parallel to XZ."""
    ax0, az0, ax1, az1 = _aabb_bounds(a)
    bx0, bz0, bx1, bz1 = _aabb_bounds(b)
    gap_x = max(bx0 - ax1, ax0 - bx1)
    gap_z = max(bz0 - az1, az0 - bz1)
    if gap_x > 0.0 or gap_z > 0.0:
        if gap_x > 0.0 and gap_z > 0.0:
            dist = math.hypot(gap_x, gap_z)
        else:
            dist = max(gap_x, gap_z)
        return dist, _unit_from_centers(a, b)
    overlap_x = min(ax1, bx1) - max(ax0, bx0)
    overlap_z = min(az1, bz1) - max(az0, bz0)
    if overlap_x <= overlap_z:
        sign = 1.0 if a.x >= b.x else -1.0
        return -overlap_x, np.array([sign, 0.0], dtype=np.float64)
    sign = 1.0 if a.z >= b.z else -1.0
    return -overlap_z, np.array([0.0, sign], dtype=np.float64)


def oriented_box_clearance(a: OrientedBox, b: OrientedBox) -> tuple[float, np.ndarray]:
    """Signed clearance and a unit normal that moves ``a`` away from ``b``.

    The normal is a direction in the XZ plane. When the boxes are separated it
    points from the closest point on ``b`` toward the closest point on ``a``.
    When they overlap it is the minimum-translation direction for ``a``.
    Rectangles aligned with the axes use an exact AABB test. Rotated rectangles
    use the separating-axis test.
    """
    if _axis_aligned(a.yaw) and _axis_aligned(b.yaw):
        return _aligned_clearance(a, b)

    ca = a.corners()
    cb = b.corners()
    axes = _box_axes(a) + _box_axes(b)

    min_overlap = math.inf
    best_axis: np.ndarray | None = None
    separated = False
    for axis in axes:
        pa = ca @ axis
        pb = cb @ axis
        overlap = float(min(pa.max(), pb.max()) - max(pa.min(), pb.min()))
        if overlap < 0.0:
            separated = True
            break
        if overlap < min_overlap:
            min_overlap = overlap
            center_delta = float((pa.mean() - pb.mean()))
            best_axis = axis if center_delta >= 0.0 else -axis

    if not separated and best_axis is not None and min_overlap < math.inf:
        n = best_axis
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-12:
            n = np.array([1.0, 0.0])
        else:
            n = n / n_norm
        return -float(min_overlap), n

    best_dist = math.inf
    best_pa = ca[0]
    best_pb = cb[0]
    for i in range(4):
        a1 = ca[i]
        a2 = ca[(i + 1) % 4]
        for j in range(4):
            b1 = cb[j]
            b2 = cb[(j + 1) % 4]
            pa, pb = closest_points_on_segments(a1, a2, b1, b2)
            dist = float(np.linalg.norm(pa - pb))
            if dist < best_dist:
                best_dist = dist
                best_pa = pa
                best_pb = pb
    delta = best_pa - best_pb
    n_norm = float(np.linalg.norm(delta))
    if n_norm < 1e-9:
        fallback = np.array([a.x - b.x, a.z - b.z], dtype=np.float64)
        fb = float(np.linalg.norm(fallback))
        normal = fallback / fb if fb > 1e-9 else np.array([1.0, 0.0])
    else:
        normal = delta / n_norm
    return float(best_dist), normal


def box_circle_clearance(
    box: OrientedBox, cx: float, cz: float, radius: float
) -> tuple[float, np.ndarray]:
    """Signed clearance between an oriented box and a circle, plus a unit normal.

    The normal moves the box away from the circle. Clearance is the distance
    from the circle boundary to the box: ``distance(center, box) - radius``.
    """
    dx = cx - box.x
    dz = cz - box.z
    c = math.cos(box.yaw)
    s = math.sin(box.yaw)
    lx = c * dx + s * dz
    lz = -s * dx + c * dz
    hx = 0.5 * box.length
    hz = 0.5 * box.width
    qx = float(np.clip(lx, -hx, hx))
    qz = float(np.clip(lz, -hz, hz))
    vx = lx - qx
    vz = lz - qz
    dist = math.hypot(vx, vz)

    def world_from_local(nx: float, nz: float) -> np.ndarray:
        # Inverse of the local-frame transform (columns are the box axes).
        wx = c * nx - s * nz
        wz = s * nx + c * nz
        return np.array([wx, wz], dtype=np.float64)

    if dist >= 1e-8:
        # Local vector from the circle center toward the closest box point.
        n_local = np.array([-vx / dist, -vz / dist])
        clearance = dist - radius
        return float(clearance), world_from_local(float(n_local[0]), float(n_local[1]))

    # Center is inside the rectangle. Leave through the nearest face.
    options = (
        (lx - (-hx), -1.0, 0.0),
        (hx - lx, 1.0, 0.0),
        (lz - (-hz), 0.0, -1.0),
        (hz - lz, 0.0, 1.0),
    )
    pen, nx, nz = min(options, key=lambda item: item[0])
    # ``(nx, nz)`` points from the center toward the nearest face. Moving the
    # box that way pushes the center deeper inside, so the separating direction
    # for the box is the opposite.
    clearance = -(pen + radius)
    return float(clearance), world_from_local(-nx, -nz)
