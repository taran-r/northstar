"""Procedural static obstacle placement inside the industrial facility."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from northstar.collision import CollisionChecker
from northstar.facility.schema import FacilitySpec, StaticObject, Zone
from northstar.geometry import OrientedBox, point_in_polygon
from northstar.vehicle.bicycle import VehicleState


OBJECT_CATALOG = [
    # type, length, width, height
    ("storage_rack", 2.4, 1.0, 2.5),
    ("pallet", 1.2, 1.0, 0.9),
    ("crate", 0.8, 0.8, 0.9),
    ("shipping_container", 3.0, 1.5, 2.0),
    ("parked_vehicle", 2.8, 1.3, 1.6),
    ("machinery", 2.0, 1.4, 1.8),
    ("barrier", 1.5, 0.4, 1.1),
    ("equipment_cart", 1.0, 0.6, 1.0),
    ("workstation", 1.8, 0.9, 1.4),
]


def _zone_by_kind(facility: FacilitySpec, kinds: set[str]) -> list[Zone]:
    return [z for z in facility.zones if z.kind in kinds]


def _sample_in_zone(zone: Zone, rng: np.random.Generator) -> tuple[float, float]:
    xs = [p[0] for p in zone.polygon]
    zs = [p[1] for p in zone.polygon]
    for _ in range(40):
        x = float(rng.uniform(min(xs), max(xs)))
        z = float(rng.uniform(min(zs), max(zs)))
        if point_in_polygon(x, z, zone.polygon):
            return x, z
    return float(np.mean(xs)), float(np.mean(zs))


def generate_static_obstacles(
    facility: FacilitySpec,
    rng: np.random.Generator,
    n_min: int = 6,
    n_max: int = 16,
    collision: Optional[CollisionChecker] = None,
    keepout: Optional[list[VehicleState]] = None,
    block_corridor_prob: float = 0.25,
) -> list[StaticObject]:
    """Place believable temporary obstacles; optionally block a corridor segment."""
    n = int(rng.integers(n_min, n_max + 1))
    place_zones = _zone_by_kind(
        facility, {"equipment", "work", "loading", "staging", "traversable"}
    )
    if not place_zones:
        return []

    obstacles: list[StaticObject] = []
    checker = collision or CollisionChecker(facility)

    for i in range(n):
        typ, length, width, height = OBJECT_CATALOG[int(rng.integers(0, len(OBJECT_CATALOG)))]
        # Scale variation
        scale = float(rng.uniform(0.85, 1.15))
        length, width = length * scale, width * scale
        zone = place_zones[int(rng.integers(0, len(place_zones)))]
        yaw = float(rng.choice([0.0, math.pi / 2, math.pi, -math.pi / 2, float(rng.uniform(-0.3, 0.3))]))

        placed = False
        for _ in range(30):
            x, z = _sample_in_zone(zone, rng)
            # Keep away from exterior walls a bit
            if x < 1.5 or z < 1.5 or x > facility.dimensions["width_x"] - 1.5:
                continue
            if z > facility.dimensions["depth_z"] - 1.5:
                continue
            box = OrientedBox(x, z, yaw, length, width)
            # Reject if overlaps permanent structure via a probe state at center
            probe = VehicleState(x=x, z=z, yaw=yaw, speed=0.0)
            # Temporarily check using object-as-vehicle is wrong; use wall checks:
            hit = False
            from northstar.geometry import oriented_boxes_collide, circle_box_collide
            from northstar.collision import wall_as_box

            for w in facility.walls:
                if oriented_boxes_collide(box, wall_as_box(w)):
                    hit = True
                    break
            if hit:
                continue
            for col in facility.columns:
                if circle_box_collide(col.x, col.z, col.radius + 0.1, box):
                    hit = True
                    break
            if hit:
                continue
            for other in obstacles:
                ob = OrientedBox(other.x, other.z, other.yaw, other.length, other.width)
                if oriented_boxes_collide(box, ob, margin=0.2):
                    hit = True
                    break
            if hit:
                continue
            if keepout:
                for k in keepout:
                    if math.hypot(k.x - x, k.z - z) < 2.5:
                        hit = True
                        break
            if hit:
                continue

            obstacles.append(
                StaticObject(
                    id=f"static_{i}_{typ}",
                    type=typ,
                    x=x,
                    z=z,
                    yaw=yaw,
                    length=length,
                    width=width,
                    height=height,
                    permanent=False,
                )
            )
            placed = True
            break

        if not placed:
            continue

    # Occasionally block part of a corridor with a barrier (creates detours)
    if rng.random() < block_corridor_prob:
        corridors = _zone_by_kind(facility, {"corridor"})
        if corridors:
            zone = corridors[int(rng.integers(0, len(corridors)))]
            x, z = _sample_in_zone(zone, rng)
            axis = zone.metadata.get("axis", "ew")
            yaw = 0.0 if axis == "ns" else math.pi / 2
            obstacles.append(
                StaticObject(
                    id="corridor_block_barrier",
                    type="barrier",
                    x=x,
                    z=z,
                    yaw=yaw,
                    length=2.2,
                    width=0.45,
                    height=1.2,
                    permanent=False,
                    metadata={"blocks_corridor": zone.id},
                )
            )

    _ = checker  # reserved for future tighter filtering
    return obstacles
