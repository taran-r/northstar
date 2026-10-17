"""Time-aware Hybrid A* for dynamic obstacle avoidance.

Extends Hybrid A* with a discretized time dimension. Collision checks use
predicted (or oracle ground-truth) dynamic obstacle poses at the node's time.

Waiting in place is an explicit action so the rover can yield at intersections.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from northstar.collision import CollisionChecker
from northstar.facility.schema import DynamicObject
from northstar.geometry import angle_diff, wrap_angle
from northstar.vehicle.bicycle import BicycleModel, ControlInput, VehicleState


@dataclass
class TimeAwareConfig:
    xy_resolution: float = 0.6
    yaw_resolution: float = math.radians(20.0)
    time_resolution: float = 0.4
    max_time: float = 45.0
    step_dt: float = 0.2
    steps_per_motion: int = 2
    steer_angles: tuple[float, ...] = (-0.55, -0.3, 0.0, 0.3, 0.55)
    speeds: tuple[float, ...] = (0.0, 0.9, 1.6, 2.2)  # 0 = wait
    max_iterations: int = 120_000
    goal_xy_tol: float = 0.9
    goal_yaw_tol: float = math.radians(45.0)
    heuristic_weight: float = 1.25
    wait_cost: float = 0.15  # extra cost per wait step to prefer motion


class TimeAwarePlanner:
    """
    Expert planner for dynamic scenarios.

    When `use_oracle=True`, collision checks use ground-truth dynamic futures.
    Those scenarios are flagged in ScenarioSpec.oracle_dynamics so the neural
    model is not silently trained as if it had perfect foresight.
    """

    def __init__(
        self,
        collision: CollisionChecker,
        model: Optional[BicycleModel] = None,
        config: Optional[TimeAwareConfig] = None,
    ):
        self.collision = collision
        self.model = model or BicycleModel(collision.vehicle_params)
        self.config = config or TimeAwareConfig()

    def _index(self, s: VehicleState) -> tuple[int, int, int, int]:
        c = self.config
        return (
            int(round(s.x / c.xy_resolution)),
            int(round(s.z / c.xy_resolution)),
            int(round(wrap_angle(s.yaw) / c.yaw_resolution)),
            int(round(s.t / c.time_resolution)),
        )

    def _heuristic(self, s: VehicleState, goal: VehicleState) -> float:
        dist = math.hypot(goal.x - s.x, goal.z - s.z)
        v = max(self.model.params.max_speed * 0.7, 0.5)
        return dist / v + 0.2 * abs(angle_diff(goal.yaw, s.yaw))

    def plan(
        self,
        start: VehicleState,
        goal: VehicleState,
        dynamics: Sequence[DynamicObject],
        use_oracle: bool = False,
    ) -> Optional[list[VehicleState]]:
        c = self.config
        use_prediction = not use_oracle

        if self.collision.collides(
            start, dynamics, t=start.t, use_prediction=use_prediction
        ):
            return None

        open_heap: list[tuple[float, int]] = []
        g_score: dict[tuple[int, int, int, int], float] = {}
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
        best_cost = float("inf")

        while open_heap and iterations < c.max_iterations:
            iterations += 1
            fcur, nid = heapq.heappop(open_heap)
            s = states[nid]

            if (
                math.hypot(goal.x - s.x, goal.z - s.z) <= c.goal_xy_tol
                and abs(angle_diff(goal.yaw, s.yaw)) <= c.goal_yaw_tol
            ):
                goal_id = nid
                break

            h = self._heuristic(s, goal)
            score = g_score[self._index(s)] + h
            if score < best_cost:
                best_cost = score
                best_id = nid

            if s.t > c.max_time:
                continue

            key = self._index(s)
            for steer in c.steer_angles:
                for speed in c.speeds:
                    # Waiting only makes sense at near-zero steer effort
                    if speed < 1e-3 and abs(steer) > 1e-3:
                        continue
                    ns = s.copy()
                    ok = True
                    for _ in range(c.steps_per_motion):
                        if speed < 1e-3:
                            accel = float(
                                np.clip(
                                    -ns.speed / max(c.step_dt, 1e-3),
                                    -self.model.params.max_decel,
                                    self.model.params.max_accel,
                                )
                            )
                            ns = self.model.step(
                                ns,
                                ControlInput(accel=accel, steer_cmd=0.0),
                                c.step_dt,
                            )
                            # Hold position while waiting (kinematic brake)
                            ns.x, ns.z = s.x, s.z
                            ns.yaw = s.yaw
                        else:
                            accel = float(
                                np.clip(
                                    (speed - ns.speed) / max(c.step_dt, 1e-3),
                                    -self.model.params.max_decel,
                                    self.model.params.max_accel,
                                )
                            )
                            ns = self.model.step(
                                ns,
                                ControlInput(accel=accel, steer_cmd=steer),
                                c.step_dt,
                            )
                        if self.collision.collides(
                            ns, dynamics, t=ns.t, use_prediction=use_prediction
                        ):
                            ok = False
                            break
                    if not ok:
                        continue

                    nkey = self._index(ns)
                    dt_cost = c.step_dt * c.steps_per_motion
                    extra = c.wait_cost if speed < 1e-3 else 0.05 * abs(steer)
                    tentative = g_score.get(key, float("inf")) + dt_cost + extra
                    if tentative + 1e-9 >= g_score.get(nkey, float("inf")):
                        continue

                    new_id = len(states)
                    states.append(ns)
                    parents[new_id] = nid
                    g_score[nkey] = tentative
                    f = tentative + c.heuristic_weight * self._heuristic(ns, goal)
                    heapq.heappush(open_heap, (f, new_id))

        if goal_id is None:
            if best_id == 0:
                return None
            goal_id = best_id

        path_ids: list[int] = []
        cur: Optional[int] = goal_id
        while cur is not None:
            path_ids.append(cur)
            cur = parents[cur]
        path_ids.reverse()
        path = [states[i].copy() for i in path_ids]
        if len(path) < 2:
            return None
        return path
