"""Rasterize facility + obstacles into local occupancy grids for NN conditioning."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from northstar.geometry import OrientedBox, point_in_polygon
from northstar.facility.schema import (
    Column,
    DynamicObject,
    FacilitySpec,
    StaticObject,
    WallSegment,
)


class OccupancyGrid:
    """Binary / float occupancy on XZ. 1 = occupied, 0 = free."""

    def __init__(
        self,
        origin_x: float,
        origin_z: float,
        resolution: float,
        width: int,
        height: int,
        data: Optional[np.ndarray] = None,
    ):
        self.origin_x = origin_x
        self.origin_z = origin_z
        self.resolution = resolution
        self.width = width
        self.height = height
        self.data = (
            np.zeros((height, width), dtype=np.float32)
            if data is None
            else data.astype(np.float32)
        )

    def world_to_grid(self, x: float, z: float) -> tuple[int, int]:
        gx = int((x - self.origin_x) / self.resolution)
        gz = int((z - self.origin_z) / self.resolution)
        return gx, gz

    def grid_to_world(self, gx: int, gz: int) -> tuple[float, float]:
        x = self.origin_x + (gx + 0.5) * self.resolution
        z = self.origin_z + (gz + 0.5) * self.resolution
        return x, z

    def in_bounds(self, gx: int, gz: int) -> bool:
        return 0 <= gx < self.width and 0 <= gz < self.height

    def set_occupied(self, gx: int, gz: int, value: float = 1.0) -> None:
        if self.in_bounds(gx, gz):
            self.data[gz, gx] = max(self.data[gz, gx], value)

    def is_occupied(self, x: float, z: float, inflate: int = 0) -> bool:
        gx, gz = self.world_to_grid(x, z)
        if not self.in_bounds(gx, gz):
            return True
        if inflate <= 0:
            return self.data[gz, gx] > 0.5
        for dz in range(-inflate, inflate + 1):
            for dx in range(-inflate, inflate + 1):
                nx, nz = gx + dx, gz + dz
                if not self.in_bounds(nx, nz) or self.data[nz, nx] > 0.5:
                    return True
        return False

    def crop_local(
        self, center_x: float, center_z: float, size_m: float
    ) -> "OccupancyGrid":
        half = size_m * 0.5
        ox = center_x - half
        oz = center_z - half
        n = max(1, int(round(size_m / self.resolution)))
        out = OccupancyGrid(ox, oz, self.resolution, n, n)
        for gz in range(n):
            for gx in range(n):
                wx, wz = out.grid_to_world(gx, gz)
                sgx, sgz = self.world_to_grid(wx, wz)
                if self.in_bounds(sgx, sgz):
                    out.data[gz, gx] = self.data[sgz, sgx]
                else:
                    out.data[gz, gx] = 1.0
        return out


def _paint_disk(grid: OccupancyGrid, x: float, z: float, radius: float) -> None:
    r_cells = int(math.ceil(radius / grid.resolution)) + 1
    gx0, gz0 = grid.world_to_grid(x, z)
    r2 = radius * radius
    for dz in range(-r_cells, r_cells + 1):
        for dx in range(-r_cells, r_cells + 1):
            gx, gz = gx0 + dx, gz0 + dz
            if not grid.in_bounds(gx, gz):
                continue
            wx, wz = grid.grid_to_world(gx, gz)
            if (wx - x) ** 2 + (wz - z) ** 2 <= r2:
                grid.set_occupied(gx, gz)


def _paint_box(grid: OccupancyGrid, box: OrientedBox) -> None:
    aabb = box.aabb()
    gx0, gz0 = grid.world_to_grid(aabb.min_x, aabb.min_z)
    gx1, gz1 = grid.world_to_grid(aabb.max_x, aabb.max_z)
    for gz in range(gz0, gz1 + 1):
        for gx in range(gx0, gx1 + 1):
            if not grid.in_bounds(gx, gz):
                continue
            wx, wz = grid.grid_to_world(gx, gz)
            if box.contains_point(wx, wz):
                grid.set_occupied(gx, gz)


def _paint_wall(grid: OccupancyGrid, wall: WallSegment) -> None:
    dx, dz = wall.x2 - wall.x1, wall.z2 - wall.z1
    length = math.hypot(dx, dz)
    if length < 1e-6:
        return
    yaw = math.atan2(dz, dx)
    mid_x = 0.5 * (wall.x1 + wall.x2)
    mid_z = 0.5 * (wall.z1 + wall.z2)
    _paint_box(
        grid,
        OrientedBox(mid_x, mid_z, yaw, length, wall.thickness),
    )


_BASE_OCC_CACHE: dict[tuple[str, float], OccupancyGrid] = {}


def build_facility_occupancy(
    facility: FacilitySpec,
    resolution: float = 0.25,
    static_obstacles: Optional[Sequence[StaticObject]] = None,
) -> OccupancyGrid:
    key = (facility.facility_id, resolution)
    base = _BASE_OCC_CACHE.get(key)
    if base is None:
        width_x = facility.dimensions["width_x"]
        depth_z = facility.dimensions["depth_z"]
        ox = facility.origin.get("x", 0.0)
        oz = facility.origin.get("z", 0.0)
        w = int(math.ceil(width_x / resolution))
        h = int(math.ceil(depth_z / resolution))
        grid = OccupancyGrid(ox, oz, resolution, w, h)

        for wall in facility.walls:
            _paint_wall(grid, wall)
        for col in facility.columns:
            _paint_disk(grid, col.x, col.z, col.radius)

        for zone in facility.zones:
            if zone.kind != "non_traversable":
                continue
            xs = [p[0] for p in zone.polygon]
            zs = [p[1] for p in zone.polygon]
            gx0, gz0 = grid.world_to_grid(min(xs), min(zs))
            gx1, gz1 = grid.world_to_grid(max(xs), max(zs))
            for gz in range(max(0, gz0), min(h, gz1 + 1)):
                for gx in range(max(0, gx0), min(w, gx1 + 1)):
                    wx, wz = grid.grid_to_world(gx, gz)
                    if point_in_polygon(wx, wz, zone.polygon):
                        grid.set_occupied(gx, gz)

        for obj in facility.permanent_objects:
            _paint_box(
                grid,
                OrientedBox(obj.x, obj.z, obj.yaw, obj.length, obj.width),
            )
        _BASE_OCC_CACHE[key] = grid
        base = grid

    out = OccupancyGrid(
        base.origin_x,
        base.origin_z,
        base.resolution,
        base.width,
        base.height,
        data=base.data.copy(),
    )
    if static_obstacles:
        for obj in static_obstacles:
            _paint_box(
                out,
                OrientedBox(obj.x, obj.z, obj.yaw, obj.length, obj.width),
            )
    return out


def dynamic_occupancy_at_time(
    base: OccupancyGrid,
    dynamics: Sequence[DynamicObject],
    t: float,
    use_prediction: bool = True,
) -> OccupancyGrid:
    """Copy base and stamp dynamic obstacles at time t."""
    out = OccupancyGrid(
        base.origin_x,
        base.origin_z,
        base.resolution,
        base.width,
        base.height,
        data=base.data.copy(),
    )
    for dyn in dynamics:
        pose = sample_dynamic_pose(dyn, t, use_prediction=use_prediction)
        if pose is None:
            continue
        x, z, yaw = pose
        _paint_box(out, OrientedBox(x, z, yaw, dyn.length, dyn.width))
    return out


def sample_dynamic_pose(
    dyn: DynamicObject, t: float, use_prediction: bool = True
) -> Optional[tuple[float, float, float]]:
    traj = dyn.prediction if use_prediction else dyn.ground_truth
    if not traj:
        return None
    if t <= traj[0].t:
        w = traj[0]
        return w.x, w.z, w.yaw
    if t >= traj[-1].t:
        w = traj[-1]
        return w.x, w.z, w.yaw
    for i in range(len(traj) - 1):
        a, b = traj[i], traj[i + 1]
        if a.t <= t <= b.t:
            u = 0.0 if b.t <= a.t else (t - a.t) / (b.t - a.t)
            x = a.x + u * (b.x - a.x)
            z = a.z + u * (b.z - a.z)
            # shortest-path yaw lerp
            dyaw = (b.yaw - a.yaw + math.pi) % (2 * math.pi) - math.pi
            yaw = a.yaw + u * dyaw
            return x, z, yaw
    return None
