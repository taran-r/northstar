"""Occupancy for the hall the frontend already draws.

The scene aisle runs along +Z from 0 to 96, with x in [-16, 16]. Dataset
planning uses the same x and z_dataset = -z_scene, which is what
`toSceneGround` reverses. Racks and their guards, columns and their base
slabs, the pallet, carton and tote stacks, and the outer walls are the floor
solids in `frontend/src/scene/layout.ts`, so the rover cannot drive through
them. Change both files together.
"""

from __future__ import annotations

import math

from northstar.facility.schema import (
    FACILITY_SCHEMA_VERSION,
    Column,
    DynamicObject,
    DynamicWaypoint,
    FacilitySpec,
    StaticObject,
    WallSegment,
)
from northstar.obstacles.dynamic import _interpolate_route

HALL_X0 = -16.0
HALL_X1 = 16.0
HALL_Z_SCENE = 96.0

# Scene pose at the near end of the aisle, facing +scene Z.
HALL_START_X = -1.5
HALL_START_Z = -8.0
HALL_START_YAW = -math.pi / 2
HALL_GOAL_X = 3.0
HALL_GOAL_Z = -86.0


def scene_to_dataset(x: float, z_scene: float) -> tuple[float, float]:
    return float(x), float(-z_scene)


# Floor props outside the lane, copied from `buildHallProps` in layout.ts.
# Scene (x, z_scene) centre, footprint (size_x, size_z), height.
# pallets(x, z, n): 1.24 x 1.0, n layers of 0.2 m.
HALL_PALLETS = (
    (-9.4, 31.0, 6),
    (-10.2, 31.0, 5),
    (-9.6, 32.2, 4),
    (9.4, 49.0, 5),
    (10.3, 49.2, 6),
    (-9.8, 72.0, 5),
    (-10.8, 72.2, 4),
)
# cartons(x, z, nx, ny): nx boxes 0.92 wide on a 0.95 pitch, 0.84 deep, ny high.
HALL_CARTONS = (
    (9.8, 36.5, 2, 3),
    (-9.6, 58.0, 2, 3),
    (10.0, 66.0, 2, 2),
)
HALL_TOTES = ((-10.6, 40.0, 1.6, 0.9, 1.0),)
RACK_HALF_X = 2.3
RACK_LEN = 9.0
# Each rack has two 0.6 m guards at the near end, 0.2 m past the frame.
RACK_GUARD = 0.6
# The drawn column is a 1.1 m square on a 1.4 m base slab. The base is the
# footprint a wheel meets, and its corners stick out past the 0.55 m circle.
COLUMN_BASE = 1.4


def _scene_box(
    obj_id: str, kind: str, x: float, z_scene: float, size_x: float, size_z: float, height: float
) -> StaticObject:
    """An axis-aligned scene prop. Yaw 0 puts `length` along x, `width` along z."""
    _, z = scene_to_dataset(x, z_scene)
    return StaticObject(
        id=obj_id,
        type=kind,
        x=float(x),
        z=z,
        yaw=0.0,
        length=float(size_x),
        width=float(size_z),
        height=float(height),
        permanent=True,
    )


def hall_props() -> list[StaticObject]:
    """Every drawn floor solid besides racks: column bases, guards, pallets, cartons, totes."""
    props: list[StaticObject] = []
    for k in range(8):
        z_scene = 8.0 + 12.0 * k
        for side, x in (("w", -8.0), ("e", 8.0)):
            props.append(
                _scene_box(f"col_base_{side}_{k}", "column", x, z_scene, COLUMN_BASE, COLUMN_BASE, 11.0)
            )
    for k in range(7):
        z_scene = 14.0 + 12.0 * k
        for cx in _rack_spots(k):
            guard_z = z_scene - RACK_LEN / 2 - 0.2
            for side, gx in (("a", cx - RACK_HALF_X + 0.2), ("b", cx + RACK_HALF_X - 0.2)):
                props.append(
                    _scene_box(
                        f"guard_{k}_{int(cx)}_{side}", "guard", gx, guard_z, RACK_GUARD, RACK_GUARD, 0.5
                    )
                )
    for i, (x, z_scene, n) in enumerate(HALL_PALLETS):
        props.append(_scene_box(f"pallets_{i}", "pallet", x, z_scene, 1.24, 1.0, 0.2 * n))
    for i, (x, z_scene, nx, ny) in enumerate(HALL_CARTONS):
        width = 0.95 * (nx - 1) + 0.92
        props.append(_scene_box(f"cartons_{i}", "crates", x, z_scene, width, 0.84, 0.21 + 0.46 * ny))
    for i, (x, z_scene, sx, sz, h) in enumerate(HALL_TOTES):
        props.append(_scene_box(f"tote_{i}", "tote", x, z_scene, sx, sz, h))
    return props


def _rack_spots(k: int) -> list[float]:
    spots = []
    if k not in (2, 5):
        spots.append(-12.6)
    if k != 3:
        spots.append(12.6)
    return spots


def hall_facility() -> FacilitySpec:
    walls = [
        WallSegment("west", HALL_X0, 0.0, HALL_X0, -HALL_Z_SCENE, thickness=0.4, height=11.0),
        WallSegment("east", HALL_X1, 0.0, HALL_X1, -HALL_Z_SCENE, thickness=0.4, height=11.0),
        WallSegment("near", HALL_X0, 0.0, HALL_X1, 0.0, thickness=0.4, height=11.0),
        WallSegment("far", HALL_X0, -HALL_Z_SCENE, HALL_X1, -HALL_Z_SCENE, thickness=0.4, height=11.0),
    ]
    columns = []
    for k in range(8):
        z_scene = 8.0 + 12.0 * k
        _, z = scene_to_dataset(0.0, z_scene)
        for side, x in (("w", -8.0), ("e", 8.0)):
            columns.append(Column(id=f"col_{side}_{k}", x=x, z=z, radius=0.55, height=11.0))

    racks: list[StaticObject] = []
    for k in range(7):
        z_scene = 14.0 + 12.0 * k
        for x in _rack_spots(k):
            _, z = scene_to_dataset(x, z_scene)
            racks.append(
                StaticObject(
                    id=f"rack_{k}_{int(x)}",
                    type="rack",
                    x=x,
                    z=z,
                    yaw=math.pi / 2,
                    length=RACK_LEN,
                    width=2 * RACK_HALF_X,
                    height=4.6,
                    permanent=True,
                )
            )

    return FacilitySpec(
        schema_version=FACILITY_SCHEMA_VERSION,
        facility_id="hall_aisle_v1",
        name="Hall aisle",
        dimensions={"width_x": HALL_X1 - HALL_X0, "depth_z": HALL_Z_SCENE, "height_y": 11.0},
        origin={"x": HALL_X0, "y": 0.0, "z": -HALL_Z_SCENE},
        walls=walls,
        columns=columns,
        zones=[],
        permanent_objects=racks + hall_props(),
    )


def _ping_pong(waypoints: list[tuple[float, float]], speed: float, duration: float) -> list[DynamicWaypoint]:
    cycle = list(waypoints) + list(reversed(waypoints[:-1]))
    once = _interpolate_route(cycle, [speed] * len(cycle), 0.2)
    if len(once) < 2:
        return once
    period = max(once[-1].t - once[0].t, 0.2)
    out: list[DynamicWaypoint] = []
    shift = 0.0
    while shift < duration + 1e-6:
        for sample in once:
            t = sample.t + shift
            if t > duration + 1e-6:
                break
            out.append(
                DynamicWaypoint(t=t, x=sample.x, z=sample.z, yaw=sample.yaw, speed=sample.speed)
            )
        shift += period
    return out


FORKLIFT_LENGTH = 4.4
FORKLIFT_WIDTH = 2.0
FORKLIFT_HEIGHT = 1.8


def placed_forklift_route(
    x: float,
    z: float,
    yaw: float,
    *,
    index: int = 0,
    duration: float = 600.0,
) -> list[DynamicWaypoint]:
    """Ping-pong a placed forklift along the aisle or across it.

    ``yaw`` is the bicycle heading stored for the body. Scene yaw for the
    +Z-forward mesh is ``yaw + π/2``; that matches the setup patrol direction.
    """
    scene_yaw = yaw + math.pi / 2
    across = abs(math.sin(scene_yaw)) > 0.5
    reach = 5.5
    if across:
        a = (max(HALL_X0 + 1.2, x - reach), z)
        b = (min(HALL_X1 - 1.2, x + reach), z)
    else:
        a = (x, min(-2.0, z + reach))
        b = (x, max(-HALL_Z_SCENE + 2.0, z - reach))
    speed = 0.95 + (index % 7) * 0.08
    return _ping_pong([a, b], speed=speed, duration=duration)


def placed_forklift(
    x: float,
    z: float,
    yaw: float,
    *,
    index: int,
    duration: float = 600.0,
) -> DynamicObject:
    return DynamicObject(
        id=f"user_forklift_{index}",
        type="forklift",
        length=FORKLIFT_LENGTH,
        width=FORKLIFT_WIDTH,
        height=FORKLIFT_HEIGHT,
        behavior="route_follow",
        ground_truth=placed_forklift_route(x, z, yaw, index=index, duration=duration),
        prediction=[],
        prediction_model="constant_velocity",
    )


def hall_crossing(duration: float = 600.0) -> list[DynamicObject]:
    """One forklift crossing the aisle at the old staged crossing.

    The visual route ran into the rack row. This one stays between the lane
    edges so the body that is drawn is the body the filter sees.
    """
    start = scene_to_dataset(5.8, 51.2)
    end = scene_to_dataset(-5.8, 51.2)
    route = _ping_pong([start, end], speed=1.5, duration=duration)
    return [
        DynamicObject(
            id="hall-crossing",
            type="forklift",
            length=FORKLIFT_LENGTH,
            width=FORKLIFT_WIDTH,
            height=FORKLIFT_HEIGHT,
            behavior="route_follow",
            ground_truth=route,
            prediction=[],
            prediction_model="constant_velocity",
        )
    ]
