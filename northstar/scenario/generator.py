"""Procedural scenario generation inside the persistent industrial facility."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from northstar.collision import CollisionChecker
from northstar.facility.occupancy import build_facility_occupancy
from northstar.facility.schema import (
    FACILITY_SCHEMA_VERSION,
    FacilitySpec,
    RoverState,
    ScenarioSpec,
    Zone,
)
from northstar.geometry import point_in_polygon
from northstar.obstacles.dynamic import generate_dynamic_obstacles
from northstar.obstacles.static_gen import generate_static_obstacles
from northstar.vehicle.bicycle import VehicleParams, VehicleState


BEHAVIOR_TAGS = [
    "straight",
    "sharp_turn",
    "long_curve",
    "tight_corridor",
    "intersection",
    "static_avoid",
    "brake_for_obstacle",
    "wait_traffic",
    "yield_intersection",
    "accelerate_clear",
    "detour_blockage",
    "follow_vehicle",
]


class ScenarioGenerator:
    def __init__(
        self,
        facility: FacilitySpec,
        vehicle_params: Optional[VehicleParams] = None,
    ):
        self.facility = facility
        self.vehicle_params = vehicle_params or VehicleParams()
        self._base_occ = build_facility_occupancy(facility, resolution=0.25)

    def _zones(self, kinds: set[str]) -> list[Zone]:
        return [z for z in self.facility.zones if z.kind in kinds]

    def _sample_pose(
        self,
        zones: list[Zone],
        rng: np.random.Generator,
        collision: CollisionChecker,
        prefer_yaw: Optional[float] = None,
    ) -> Optional[VehicleState]:
        if not zones:
            return None
        for _ in range(60):
            zone = zones[int(rng.integers(0, len(zones)))]
            xs = [p[0] for p in zone.polygon]
            zs = [p[1] for p in zone.polygon]
            x = float(rng.uniform(min(xs), max(xs)))
            z = float(rng.uniform(min(zs), max(zs)))
            if not point_in_polygon(x, z, zone.polygon):
                continue
            if prefer_yaw is None:
                yaw = float(rng.choice([0.0, math.pi / 2, math.pi, -math.pi / 2]))
            else:
                yaw = prefer_yaw + float(rng.uniform(-0.2, 0.2))
            state = VehicleState(x=x, z=z, yaw=yaw, speed=0.0, steering=0.0, t=0.0)
            if not collision.collides_static(state):
                return state
        return None

    def generate(
        self,
        seed: int,
        scenario_id: Optional[str] = None,
        force_behavior: Optional[str] = None,
        multi_modal: bool = False,
    ) -> Optional[ScenarioSpec]:
        rng = np.random.default_rng(seed)
        behavior = force_behavior or BEHAVIOR_TAGS[int(rng.integers(0, len(BEHAVIOR_TAGS)))]

        # First place start/goal with only permanent facility
        base_collision = CollisionChecker(
            self.facility, occupancy=self._base_occ, vehicle_params=self.vehicle_params
        )
        spawn_zones = self._zones({"spawn", "corridor", "staging", "loading", "traversable", "work"})
        goal_zones = self._zones({"goal", "loading", "staging", "corridor", "work"})

        # Behavior-biased sampling
        if behavior in {"tight_corridor", "straight"}:
            spawn_zones = self._zones({"corridor", "spawn"}) or spawn_zones
            goal_zones = self._zones({"corridor", "goal"}) or goal_zones
        elif behavior in {"intersection", "yield_intersection", "wait_traffic"}:
            spawn_zones = self._zones({"corridor", "intersection", "spawn"}) or spawn_zones
            goal_zones = self._zones({"goal", "corridor", "loading"}) or goal_zones
        elif behavior in {"detour_blockage", "static_avoid"}:
            spawn_zones = self._zones({"spawn", "equipment", "work"}) or spawn_zones
            goal_zones = self._zones({"goal", "loading", "staging"}) or goal_zones

        start = self._sample_pose(spawn_zones, rng, base_collision)
        goal = self._sample_pose(goal_zones, rng, base_collision)
        if start is None or goal is None:
            return None
        if math.hypot(goal.x - start.x, goal.z - start.z) < 8.0:
            # Prefer non-trivial trips
            alt = self._sample_pose(goal_zones, rng, base_collision)
            if alt is not None and math.hypot(alt.x - start.x, alt.z - start.z) >= 8.0:
                goal = alt

        # Face roughly toward goal at start
        start.yaw = math.atan2(goal.z - start.z, goal.x - start.x)

        static: list = []
        for _ in range(5):
            static = generate_static_obstacles(
                self.facility,
                rng,
                n_min=5,
                n_max=14,
                keepout=[start, goal],
                block_corridor_prob=0.55 if behavior == "detour_blockage" else 0.2,
            )
            occ = build_facility_occupancy(
                self.facility, resolution=0.3, static_obstacles=static
            )
            chk = CollisionChecker(
                self.facility,
                static_obstacles=static,
                occupancy=occ,
                vehicle_params=self.vehicle_params,
            )
            chk.fast_occupancy = True
            if not chk.collides_static(start) and not chk.collides_static(goal):
                break
        else:
            static = [
                o
                for o in static
                if math.hypot(o.x - start.x, o.z - start.z) > 3.5
                and math.hypot(o.x - goal.x, o.z - goal.z) > 3.5
            ]

        n_dyn_min, n_dyn_max = 1, 3
        if behavior in {"wait_traffic", "yield_intersection", "follow_vehicle", "brake_for_obstacle"}:
            n_dyn_min, n_dyn_max = 2, 4
        elif behavior in {"straight", "sharp_turn"}:
            n_dyn_min, n_dyn_max = 0, 2

        dynamics, use_oracle = generate_dynamic_obstacles(
            self.facility,
            rng,
            n_min=n_dyn_min,
            n_max=n_dyn_max,
            oracle_fraction=0.35 if behavior in {"wait_traffic", "yield_intersection"} else 0.15,
        )

        sid = scenario_id or f"scn_{seed:06d}"
        return self._assemble(sid, seed, behavior, multi_modal, start, goal, static, dynamics, use_oracle)

    def sample_endpoints(
        self,
        seed: int,
        min_separation_m: float = 12.0,
    ) -> Optional[tuple[VehicleState, VehicleState]]:
        """Sample a start/goal pair on the fixed facility, far enough apart to be a real route."""
        rng = np.random.default_rng(seed)
        base_collision = CollisionChecker(
            self.facility, occupancy=self._base_occ, vehicle_params=self.vehicle_params
        )
        spawn_zones = self._zones({"spawn", "corridor", "staging", "loading", "traversable", "work"})
        goal_zones = self._zones({"goal", "loading", "staging", "corridor", "work"})
        for _ in range(12):
            start = self._sample_pose(spawn_zones, rng, base_collision)
            goal = self._sample_pose(goal_zones, rng, base_collision)
            if start is None or goal is None:
                continue
            if math.hypot(goal.x - start.x, goal.z - start.z) < min_separation_m:
                continue
            start.yaw = math.atan2(goal.z - start.z, goal.x - start.x)
            return start, goal
        return None

    def _assemble(
        self,
        sid: str,
        seed: int,
        behavior: str,
        multi_modal: bool,
        start: VehicleState,
        goal: VehicleState,
        static: list,
        dynamics: list,
        use_oracle: bool,
    ) -> ScenarioSpec:
        return ScenarioSpec(
            schema_version=FACILITY_SCHEMA_VERSION,
            scenario_id=sid,
            facility_id=self.facility.facility_id,
            facility_schema_version=self.facility.schema_version,
            seed=seed,
            start=RoverState(
                x=start.x, z=start.z, yaw=start.yaw, speed=0.0, steering=0.0, y=0.0
            ),
            goal=RoverState(
                x=goal.x, z=goal.z, yaw=goal.yaw, speed=0.0, steering=0.0, y=0.0
            ),
            static_obstacles=static,
            dynamic_obstacles=dynamics,
            metadata={
                "behavior_tag": behavior,
                "multi_modal": multi_modal,
                "requested_behaviors": [behavior],
            },
            oracle_dynamics=use_oracle,
        )
