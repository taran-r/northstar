"""Construct the persistent base industrial facility (authoritative for Python + Three.js)."""

from __future__ import annotations

from pathlib import Path

from .schema import (
    FACILITY_SCHEMA_VERSION,
    Column,
    FacilitySpec,
    StaticObject,
    WallSegment,
    Zone,
)
from .loader import save_facility


def _rect_poly(x0: float, z0: float, x1: float, z1: float) -> list[list[float]]:
    return [[x0, z0], [x1, z0], [x1, z1], [x0, z1]]


def _hwall(id_: str, x1: float, x2: float, z: float, thickness: float = 0.3) -> WallSegment:
    return WallSegment(id=id_, x1=x1, z1=z, x2=x2, z2=z, thickness=thickness)


def _vwall(id_: str, z1: float, z2: float, x: float, thickness: float = 0.3) -> WallSegment:
    return WallSegment(id=id_, x1=x, z1=z1, x2=x, z2=z2, thickness=thickness)


def build_industrial_facility_v1() -> FacilitySpec:
    """
    ~60m x 40m industrial warehouse/workshop with:
      - exterior shell
      - two long corridors + cross intersections
      - rack/equipment zones, loading bay, staging, open maneuvering
      - support columns on a grid
      - permanent equipment pads
    """
    W, D = 60.0, 40.0
    walls: list[WallSegment] = []
    # Exterior shell
    walls += [
        _hwall("ext_south", 0.0, W, 0.0, 0.4),
        _hwall("ext_north", 0.0, W, D, 0.4),
        _vwall("ext_west", 0.0, D, 0.0, 0.4),
        _vwall("ext_east", 0.0, D, W, 0.4),
    ]

    # Interior structural dividers (doorway gaps left as missing segments)
    # West equipment hall divider (gap at corridor)
    walls += [
        _vwall("div_w1", 0.0, 12.0, 18.0),
        _vwall("div_w2", 16.0, 28.0, 18.0),
        _vwall("div_w3", 32.0, 40.0, 18.0),
    ]
    # Mid workshop divider
    walls += [
        _vwall("div_m1", 0.0, 10.0, 36.0),
        _vwall("div_m2", 14.0, 26.0, 36.0),
        _vwall("div_m3", 30.0, 40.0, 36.0),
    ]
    # East loading / staging divider with wider openings
    walls += [
        _vwall("div_e1", 0.0, 8.0, 48.0),
        _vwall("div_e2", 18.0, 40.0, 48.0),
    ]

    # Horizontal corridor walls (main E-W corridor around z=14 and z=26)
    walls += [
        _hwall("corr_s1", 0.0, 16.0, 12.0),
        _hwall("corr_s2", 20.0, 34.0, 12.0),
        _hwall("corr_s3", 38.0, 48.0, 12.0),
        _hwall("corr_n1", 0.0, 16.0, 28.0),
        _hwall("corr_n2", 20.0, 34.0, 28.0),
        _hwall("corr_n3", 38.0, 60.0, 28.0),
    ]

    # Internal work-bay partitions (partial walls creating bottlenecks)
    walls += [
        _hwall("bay_a", 2.0, 10.0, 6.0, 0.25),
        _hwall("bay_b", 20.0, 30.0, 20.0, 0.25),
        _vwall("bay_c", 2.0, 8.0, 8.0, 0.25),
        _vwall("bay_d", 30.0, 38.0, 54.0, 0.25),
        _hwall("bay_e", 50.0, 58.0, 8.0, 0.25),
    ]

    columns: list[Column] = []
    cid = 0
    for x in [9.0, 27.0, 42.0, 54.0]:
        for z in [6.0, 20.0, 34.0]:
            # Skip columns that would sit in corridor centerlines awkwardly
            if 12.5 < z < 15.5 or 25.5 < z < 28.5:
                continue
            columns.append(Column(id=f"col_{cid}", x=x, z=z, radius=0.35))
            cid += 1

    zones: list[Zone] = [
        Zone("zone_loading", "loading", _rect_poly(48.5, 0.5, 59.5, 11.5)),
        Zone("zone_staging", "staging", _rect_poly(48.5, 18.5, 59.5, 27.5)),
        Zone("zone_equip_w", "equipment", _rect_poly(1.0, 1.0, 17.0, 11.0)),
        Zone("zone_work_mid", "work", _rect_poly(19.0, 13.0, 35.0, 27.0)),
        Zone("zone_open_maneuver", "traversable", _rect_poly(37.0, 13.0, 47.5, 27.0)),
        Zone(
            "corridor_main",
            "corridor",
            _rect_poly(1.0, 12.5, 59.0, 15.5),
            metadata={"axis": "ew"},
        ),
        Zone(
            "corridor_north",
            "corridor",
            _rect_poly(1.0, 26.5, 47.5, 29.5),
            metadata={"axis": "ew"},
        ),
        Zone(
            "corridor_ns_w",
            "corridor",
            _rect_poly(16.5, 1.0, 19.5, 39.0),
            metadata={"axis": "ns"},
        ),
        Zone(
            "corridor_ns_m",
            "corridor",
            _rect_poly(34.5, 1.0, 37.5, 39.0),
            metadata={"axis": "ns"},
        ),
        Zone("intersect_a", "intersection", _rect_poly(16.0, 12.0, 20.0, 16.0)),
        Zone("intersect_b", "intersection", _rect_poly(34.0, 12.0, 38.0, 16.0)),
        Zone("intersect_c", "intersection", _rect_poly(16.0, 26.0, 20.0, 30.0)),
        Zone("intersect_d", "intersection", _rect_poly(34.0, 26.0, 38.0, 30.0)),
        Zone("spawn_west", "spawn", _rect_poly(2.0, 13.0, 8.0, 15.0)),
        Zone("spawn_south", "spawn", _rect_poly(25.0, 2.0, 31.0, 6.0)),
        Zone("goal_loading", "goal", _rect_poly(52.0, 2.0, 58.0, 8.0)),
        Zone("goal_staging", "goal", _rect_poly(52.0, 20.0, 58.0, 26.0)),
        Zone("goal_north", "goal", _rect_poly(4.0, 32.0, 12.0, 38.0)),
        # Non-traversable machine pits / reserved pads
        Zone("pit_1", "non_traversable", _rect_poly(3.0, 32.0, 7.0, 36.0)),
        Zone("pit_2", "non_traversable", _rect_poly(40.0, 32.0, 45.0, 38.0)),
    ]

    permanent: list[StaticObject] = [
        StaticObject(
            id="cnc_station_1",
            type="machinery",
            x=5.0,
            z=3.5,
            yaw=0.0,
            length=3.0,
            width=1.6,
            height=2.2,
            permanent=True,
        ),
        StaticObject(
            id="cnc_station_2",
            type="machinery",
            x=12.0,
            z=3.5,
            yaw=0.0,
            length=3.0,
            width=1.6,
            height=2.2,
            permanent=True,
        ),
        StaticObject(
            id="conveyor_pad",
            type="machinery",
            x=27.0,
            z=34.0,
            yaw=0.0,
            length=6.0,
            width=1.2,
            height=1.0,
            permanent=True,
        ),
        StaticObject(
            id="dock_bumper_a",
            type="barrier",
            x=59.2,
            z=4.0,
            yaw=1.5708,
            length=4.0,
            width=0.4,
            height=1.2,
            permanent=True,
        ),
        StaticObject(
            id="dock_bumper_b",
            type="barrier",
            x=59.2,
            z=22.0,
            yaw=1.5708,
            length=4.0,
            width=0.4,
            height=1.2,
            permanent=True,
        ),
    ]

    return FacilitySpec(
        schema_version=FACILITY_SCHEMA_VERSION,
        facility_id="industrial_facility_v1",
        name="Northstar Industrial Facility v1",
        dimensions={"width_x": W, "depth_z": D, "height_y": 6.0},
        origin={"x": 0.0, "y": 0.0, "z": 0.0},
        walls=walls,
        columns=columns,
        zones=zones,
        permanent_objects=permanent,
        metadata={
            "coordinate_system": {
                "horizontal": "XZ",
                "elevation": "Y",
                "units": {"distance": "m", "time": "s", "angle": "rad"},
            },
            "description": (
                "Persistent warehouse/workshop shell with E-W and N-S corridors, "
                "work bays, loading/staging docks, and support columns."
            ),
        },
    )


def write_default_facility(out_path: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[2]
    path = out_path or (root / "facilities" / "industrial_facility_v1.json")
    facility = build_industrial_facility_v1()
    save_facility(facility, path)
    return path


if __name__ == "__main__":
    p = write_default_facility()
    print(f"Wrote {p}")
