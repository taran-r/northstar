"""Versioned, serializable industrial facility specification.

Coordinate system:
  - X, Z: horizontal plane (meters)
  - Y: elevation (meters)
  - angles: radians
  - time: seconds
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


FACILITY_SCHEMA_VERSION = "1.0.0"


@dataclass
class WallSegment:
    """Permanent structural wall as a thickened line segment on XZ."""

    id: str
    x1: float
    z1: float
    x2: float
    z2: float
    thickness: float = 0.3
    height: float = 4.0
    y: float = 0.0
    kind: str = "structural"


@dataclass
class Column:
    id: str
    x: float
    z: float
    radius: float = 0.35
    height: float = 4.5
    y: float = 0.0


@dataclass
class Zone:
    """Named polygonal region for spawn/goal sampling and semantics."""

    id: str
    kind: Literal[
        "traversable",
        "non_traversable",
        "loading",
        "staging",
        "work",
        "corridor",
        "intersection",
        "equipment",
        "spawn",
        "goal",
    ]
    polygon: list[list[float]]  # [[x,z], ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StaticObject:
    """Temporary or permanent box obstacle with oriented footprint."""

    id: str
    type: str
    x: float
    z: float
    yaw: float
    length: float
    width: float
    height: float = 1.5
    y: float = 0.0
    permanent: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamicWaypoint:
    t: float
    x: float
    z: float
    yaw: float
    speed: float = 0.0


@dataclass
class DynamicObject:
    """Moving obstacle with ground-truth trajectory and prediction metadata."""

    id: str
    type: str
    length: float
    width: float
    height: float = 1.8
    y: float = 0.0
    behavior: str = "route_follow"
    # Ground-truth future used by simulator / oracle planner
    ground_truth: list[DynamicWaypoint] = field(default_factory=list)
    # Predicted future provided to non-oracle agents (constant-velocity, etc.)
    prediction: list[DynamicWaypoint] = field(default_factory=list)
    prediction_model: str = "constant_velocity"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FacilitySpec:
    """Persistent base industrial facility shared with Three.js."""

    schema_version: str
    facility_id: str
    name: str
    dimensions: dict[str, float]  # width_x, depth_z, height_y
    origin: dict[str, float]
    walls: list[WallSegment]
    columns: list[Column]
    zones: list[Zone]
    permanent_objects: list[StaticObject] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FacilitySpec":
        return cls(
            schema_version=data["schema_version"],
            facility_id=data["facility_id"],
            name=data["name"],
            dimensions=data["dimensions"],
            origin=data.get("origin", {"x": 0.0, "y": 0.0, "z": 0.0}),
            walls=[WallSegment(**w) for w in data.get("walls", [])],
            columns=[Column(**c) for c in data.get("columns", [])],
            zones=[Zone(**z) for z in data.get("zones", [])],
            permanent_objects=[
                StaticObject(**o) for o in data.get("permanent_objects", [])
            ],
            metadata=data.get("metadata", {}),
        )


@dataclass
class RoverState:
    x: float
    z: float
    yaw: float
    speed: float = 0.0
    steering: float = 0.0
    y: float = 0.0


@dataclass
class ScenarioSpec:
    """One procedurally generated scenario inside the base facility."""

    schema_version: str
    scenario_id: str
    facility_id: str
    facility_schema_version: str
    seed: int
    start: RoverState
    goal: RoverState
    static_obstacles: list[StaticObject]
    dynamic_obstacles: list[DynamicObject]
    metadata: dict[str, Any] = field(default_factory=dict)
    # If True, expert planner used ground-truth dynamic futures (oracle).
    oracle_dynamics: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioSpec":
        return cls(
            schema_version=data["schema_version"],
            scenario_id=data["scenario_id"],
            facility_id=data["facility_id"],
            facility_schema_version=data["facility_schema_version"],
            seed=data["seed"],
            start=RoverState(**data["start"]),
            goal=RoverState(**data["goal"]),
            static_obstacles=[StaticObject(**o) for o in data.get("static_obstacles", [])],
            dynamic_obstacles=[
                DynamicObject(
                    id=d["id"],
                    type=d["type"],
                    length=d["length"],
                    width=d["width"],
                    height=d.get("height", 1.8),
                    y=d.get("y", 0.0),
                    behavior=d.get("behavior", "route_follow"),
                    ground_truth=[DynamicWaypoint(**w) for w in d.get("ground_truth", [])],
                    prediction=[DynamicWaypoint(**w) for w in d.get("prediction", [])],
                    prediction_model=d.get("prediction_model", "constant_velocity"),
                    metadata=d.get("metadata", {}),
                )
                for d in data.get("dynamic_obstacles", [])
            ],
            metadata=data.get("metadata", {}),
            oracle_dynamics=data.get("oracle_dynamics", False),
        )
