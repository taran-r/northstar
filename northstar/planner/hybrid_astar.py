"""Kinematically constrained Hybrid A* for static industrial navigation.

States are continuous (x, z, yaw) with discrete closed-set hashing.
Motion primitives integrate the kinematic bicycle model.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from northstar.collision import CollisionChecker
from northstar.geometry import angle_diff, wrap_angle
from northstar.vehicle.bicycle import BicycleModel, ControlInput, VehicleState


@dataclass
class HybridAStarConfig:
    xy_resolution: float = 0.5
    yaw_resolution: float = math.radians(15.0)
    step_dt: float = 0.25
    steps_per_motion: int = 2
    steer_angles: tuple[float, ...] = (-0.55, -0.35, -0.15, 0.0, 0.15, 0.35, 0.55)
    speeds: tuple[float, ...] = (0.8, 1.4, 2.0)
    max_iterations: int = 80_000
    goal_xy_tol: float = 0.75
    goal_yaw_tol: float = math.radians(40.0)
    heuristic_weight: float = 1.2


class HybridAStar:
    def __init__(
        self,
        collision: CollisionChecker,
        model: Optional[BicycleModel] = None,
        config: Optional[HybridAStarConfig] = None,
    ):
        self.collision = collision
        self.model = model or BicycleModel(collision.vehicle_params)
        self.config = config or HybridAStarConfig()

    def _index(self, s: VehicleState) -> tuple[int, int, int]:
        c = self.config
        return (
            int(round(s.x / c.xy_resolution)),
            int(round(s.z / c.xy_resolution)),
            int(round(wrap_angle(s.yaw) / c.yaw_resolution)),
        )

    def _heuristic(self, s: VehicleState, goal: VehicleState) -> float:
        dist = math.hypot(goal.x - s.x, goal.z - s.z)
        heading = abs(angle_diff(goal.yaw, s.yaw))
        return dist + 0.35 * heading * max(0.0, 3.0 - dist)

    def plan(
        self,
        start: VehicleState,
        goal: VehicleState,
    ) -> Optional[list[VehicleState]]:
        c = self.config
        if self.collision.collides_static(start):
            return None

        open_heap: list[tuple[float, int]] = []
        g_score: dict[tuple[int, int, int], float] = {}
        parents: dict[int, Optional[int]] = {}
        states: list[VehicleState] = []

        start_s = start.copy()
        states.append(start_s)
        parents[0] = None
        g_score[self._index(start_s)] = 0.0
        heapq.heappush(
            open_heap, (self._heuristic(start_s, goal) * c.heuristic_weight, 0)
        )

        iterations = 0
        goal_id: Optional[int] = None
        best_id = 0
        best_h = self._heuristic(start_s, goal)

        while open_heap and iterations < c.max_iterations:
            iterations += 1
            _, nid = heapq.heappop(open_heap)
            s = states[nid]
            key = self._index(s)

            h = self._heuristic(s, goal)
            if h < best_h:
                best_h = h
                best_id = nid

            if (
                math.hypot(goal.x - s.x, goal.z - s.z) <= c.goal_xy_tol
                and abs(angle_diff(goal.yaw, s.yaw)) <= c.goal_yaw_tol
            ):
                goal_id = nid
                break

            for steer in c.steer_angles:
                for speed in c.speeds:
                    ns = s.copy()
                    ok = True
                    accel = 0.0
                    for _ in range(c.steps_per_motion):
                        accel = float(
                            np.clip(
                                (speed - ns.speed) / max(c.step_dt, 1e-3),
                                -self.model.params.max_decel,
                                self.model.params.max_accel,
                            )
                        )
                        ns = self.model.step(
                            ns, ControlInput(accel=accel, steer_cmd=steer), c.step_dt
                        )
                        if self.collision.collides_static(ns):
                            ok = False
                            break
                    if not ok:
                        continue

                    nkey = self._index(ns)
                    step_cost = c.step_dt * c.steps_per_motion + 0.05 * abs(steer)
                    tentative = g_score.get(key, float("inf")) + step_cost
                    if tentative + 1e-9 >= g_score.get(nkey, float("inf")):
                        continue

                    new_id = len(states)
                    states.append(ns)
                    parents[new_id] = nid
                    g_score[nkey] = tentative
                    f = tentative + c.heuristic_weight * self._heuristic(ns, goal)
                    heapq.heappush(open_heap, (f, new_id))

        if goal_id is None:
            return None

        path_ids: list[int] = []
        cur: Optional[int] = goal_id
        while cur is not None:
            path_ids.append(cur)
            cur = parents[cur]
        path_ids.reverse()
        return [states[i].copy() for i in path_ids]
