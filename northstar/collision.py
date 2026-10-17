"""Collision checking against facility structure, static and dynamic obstacles."""

from __future__ import annotations

import math
from typing import Optional, Sequence

from northstar.facility.occupancy import OccupancyGrid, sample_dynamic_pose
from northstar.facility.schema import (
    Column,
    DynamicObject,
    FacilitySpec,
    StaticObject,
    WallSegment,
)
from northstar.geometry import OrientedBox, circle_box_collide, oriented_boxes_collide
from northstar.vehicle.bicycle import VehicleParams, VehicleState


def wall_as_box(wall: WallSegment) -> OrientedBox:
    dx, dz = wall.x2 - wall.x1, wall.z2 - wall.z1
    length = math.hypot(dx, dz)
    yaw = math.atan2(dz, dx) if length > 1e-9 else 0.0
    return OrientedBox(
        0.5 * (wall.x1 + wall.x2),
        0.5 * (wall.z1 + wall.z2),
        yaw,
        max(length, 1e-3),
        wall.thickness,
    )


class CollisionChecker:
    def __init__(
        self,
        facility: FacilitySpec,
        static_obstacles: Optional[Sequence[StaticObject]] = None,
        occupancy: Optional[OccupancyGrid] = None,
        vehicle_params: Optional[VehicleParams] = None,
    ):
        self.facility = facility
        self.static_obstacles = list(static_obstacles or [])
        self.occupancy = occupancy
        self.vehicle_params = vehicle_params or VehicleParams()
        # When True and occupancy is set, skip exact geometry (planning speed).
        self.fast_occupancy: bool = False
        self._wall_boxes = [wall_as_box(w) for w in facility.walls]
        self._perm_boxes = [
            OrientedBox(o.x, o.z, o.yaw, o.length, o.width)
            for o in facility.permanent_objects
        ]
        self._static_boxes = [
            OrientedBox(o.x, o.z, o.yaw, o.length, o.width)
            for o in self.static_obstacles
        ]
        self._columns = list(facility.columns)

        # Bounds
        self.min_x = facility.origin.get("x", 0.0) + 0.5
        self.min_z = facility.origin.get("z", 0.0) + 0.5
        self.max_x = self.min_x + facility.dimensions["width_x"] - 1.0
        self.max_z = self.min_z + facility.dimensions["depth_z"] - 1.0

    def in_bounds(self, state: VehicleState) -> bool:
        return self.min_x <= state.x <= self.max_x and self.min_z <= state.z <= self.max_z

    def collides_static(self, state: VehicleState) -> bool:
        if not self.in_bounds(state):
            return True
        box = state.footprint(self.vehicle_params)

        # Fast occupancy: check center + footprint corners
        if self.occupancy is not None:
            pts = [(state.x, state.z)]
            for corner in box.corners():
                pts.append((float(corner[0]), float(corner[1])))
            hit = any(self.occupancy.is_occupied(px, pz) for px, pz in pts)
            if self.fast_occupancy:
                return hit
            if not hit:
                return False
            # Occupancy positive → confirm with exact geometry below

        for w in self._wall_boxes:
            if oriented_boxes_collide(box, w):
                return True
        for col in self._columns:
            if circle_box_collide(col.x, col.z, col.radius, box):
                return True
        for b in self._perm_boxes:
            if oriented_boxes_collide(box, b):
                return True
        for b in self._static_boxes:
            if oriented_boxes_collide(box, b):
                return True
        # Occupancy said occupied (e.g. non-traversable zone) but no box hit
        if self.occupancy is not None:
            for corner in box.corners():
                if self.occupancy.is_occupied(float(corner[0]), float(corner[1])):
                    return True
        return False

    def collides_dynamic(
        self,
        state: VehicleState,
        dynamics: Sequence[DynamicObject],
        t: float,
        use_prediction: bool = True,
        margin: float = 0.15,
    ) -> bool:
        box = state.footprint(self.vehicle_params)
        for dyn in dynamics:
            pose = sample_dynamic_pose(dyn, t, use_prediction=use_prediction)
            if pose is None:
                continue
            x, z, yaw = pose
            other = OrientedBox(x, z, yaw, dyn.length, dyn.width)
            if oriented_boxes_collide(box, other, margin=margin):
                return True
        return False

    def collides(
        self,
        state: VehicleState,
        dynamics: Optional[Sequence[DynamicObject]] = None,
        t: Optional[float] = None,
        use_prediction: bool = True,
    ) -> bool:
        if self.collides_static(state):
            return True
        if dynamics:
            tt = state.t if t is None else t
            return self.collides_dynamic(state, dynamics, tt, use_prediction=use_prediction)
        return False
