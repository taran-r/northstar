"""Interactive closed-loop rover: flow matching, tracking, CBF, Hybrid A*.

The dataset episode in `bridge.episode` replays a fixed scenario. This module
is the session the frontend drives: a destination, static obstacles the user
drops, and forklifts that follow physical routes. Nothing here invents a
trajectory. A usable flow-matching candidate is tracked and labelled
`learned`. When every candidate is rejected, or no checkpoint is loaded,
Hybrid A* supplies the reference and the frame is labelled `fallback`.

Replanning is event-driven. A new plan is requested when the goal or the
obstacle set changes, the safety filter holds a standing replan flag, the
reference is used up, or the reference now intersects the obstacle forecast.
A minimum gap keeps a clear corridor from calling the network every step.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from typing import Iterator, Optional

import numpy as np

from backend.ml.bridge import (
    CONTROL_DT,
    CANDIDATES,
    CRUISE_SPEED,
    DYNAMIC_SLICE_DT,
    DYNAMIC_SLICES,
    FACILITY_PATH,
    GOAL_RADIUS_M,
    MAP_RESOLUTION,
    MAP_SIZE_M,
    SOLVER_STEPS,
    TRACKER_LOOKAHEAD,
    TRACKER_STEER_GAIN,
    EpisodeMetrics,
    _candidate_collisions,
    controller_mode,
    controller_state,
    path_points,
    state_frame,
)
from backend.ml.dataset import world_to_local
from backend.ml.evaluation import _vehicle_limits
from backend.ml.inference import FlowMatchingPlanner, best_candidate
from northstar.collision import CollisionChecker
from northstar.control.cbf import CBFConfig
from northstar.control.simulate import body_clearance, closed_loop_step
from northstar.control.scene import BoxObstacle, CircleObstacle, SafetyScene, build_scene
from northstar.control.tracking import TrajectoryTracker
from northstar.facility.loader import load_facility
from northstar.facility.occupancy import (
    OccupancyGrid,
    build_facility_occupancy,
    sample_dynamic_pose,
)
from northstar.facility.schema import DynamicObject, DynamicWaypoint, StaticObject
from northstar.geometry import OrientedBox
from northstar.obstacles.dynamic import _interpolate_route
from northstar.planner.hybrid_astar import HybridAStar, HybridAStarConfig
from northstar.vehicle.bicycle import ControlInput, VehicleParams, VehicleState
from backend.ml.hall import (
    FORKLIFT_HEIGHT,
    FORKLIFT_LENGTH,
    FORKLIFT_WIDTH,
    HALL_GOAL_X,
    HALL_GOAL_Z,
    HALL_START_X,
    HALL_START_YAW,
    HALL_START_Z,
    hall_crossing,
    hall_facility,
    placed_forklift,
)

MIN_REPLAN_GAP_S = 0.8
REPLAN_HOLD_S = 0.4
PROGRESS_M = 0.25
PENETRATION_M = 0.02
FORECAST_S = 4.0
TRAIL_LIMIT = 500

OBSTACLE_SIZES = {
    "pallet": (1.2, 1.0, 1.0),
    "crates": (1.8, 1.6, 1.2),
    "barrier": (2.6, 0.45, 1.0),
    "forklift": (2.4, 1.2, 2.0),
}

# Coarser than the expert search. This runs on the control thread when the
# network has nothing usable, so it has to return before the session feels stuck.
# Each primitive has to leave its grid cell. A single short step from rest
# hashes back onto the start cell and the search ends with no path.
FALLBACK_CONFIG = HybridAStarConfig(
    xy_resolution=0.5,
    yaw_resolution=math.radians(20.0),
    step_dt=0.35,
    steps_per_motion=3,
    steer_angles=(-0.45, -0.2, 0.0, 0.2, 0.45),
    speeds=(1.3,),
    max_iterations=8_000,
    goal_xy_tol=1.3,
    goal_yaw_tol=math.pi,
    heuristic_weight=1.4,
)

START_X = 4.0
START_Z = 14.0

# Interactive queries use a shorter integration than evaluation. The samples
# are still the model's flow; fewer steps only coarsens the ODE.
INTERACTIVE_CANDIDATES = 4
INTERACTIVE_STEPS = 8
# The aisle is 96 m. 1.6 m/s reads as a walk, and the bicycle's limit is 2.5.
# A sharp heading error used to pin the tracker at 0.8 m/s, which is the
# crawl in the cluttered hall. The plant step is coarser here so one safety
# solve covers more ground; the filter itself is unchanged in kind.
HALL_PACE = 5.0
HALL_CRUISE = 2.45
HALL_TURN_SPEED = 1.8
HALL_CONTROL_DT = 0.2
HALL_REPLAN_GAP_S = 2.0
HALL_REPLAN_HOLD_S = 1.5
# The local ribbon bends inside one metre. A 1.5 m pursuit point still sits
# on the straight piece, so the tracker aims at the blockage the ribbon is
# going around.
HALL_LOOKAHEAD = 3.5
# One accepted step used to flash Watching, and the replan queued by that
# same brake flashed Replan, while the rover had not left the correction.
MODE_LATCH_S = 1.0
# Shorter backup and a tighter neighbourhood than the offline filter.
# The constraint is the same discrete-time CBF; fewer obstacles and a
# shorter rollout are what keep a step from taking most of a second.
HALL_CBF = CBFConfig(
    # Long enough to see the crossing forklift before it owns the lane.
    preview_time=0.7,
    idle_time=1.0,
    activation_clearance=2.2,
    nearby_extra=1.0,
    replan_horizon=4.0,
    replan_distances=(1.4, 2.6, 4.0),
)
# Hybrid A* only has to clear the blockage. The next replan asks the network again.
FALLBACK_HORIZON_M = 9.0
VIZ_GRID = 20

# Footprints match Obstacles.tsx at OS=1.2, plus a small pad for the chase
# rover (drawn at 1.5×). A 1.2 m pad on every axis turned the striped barrier
# (~0.5 m thick on screen) into a 2 m wall and the filter emergency-stopped
# while the mesh still looked clear.
_PAD = 0.4
HALL_SIZES = {
    "pallet": (1.49 + _PAD, 1.20 + _PAD, 1.1),
    "crates": (2.28 + _PAD, 2.04 + _PAD, 1.4),
    "barrier": (3.36 + _PAD, 0.53 + _PAD, 1.0),
    "forklift": (FORKLIFT_LENGTH, FORKLIFT_WIDTH, FORKLIFT_HEIGHT),
}


class HoldTracker:
    """Zero-speed command used before a destination exists, or after arrival."""

    def __init__(self, params: VehicleParams):
        self.params = params

    def nominal_control(self, state: VehicleState, dt: float) -> ControlInput:
        return ControlInput(accel=-self.params.max_decel, steer_cmd=state.steering)


def _loop_route(
    waypoints: list[tuple[float, float]],
    speed: float,
    duration: float,
    dt: float = 0.2,
) -> list[DynamicWaypoint]:
    """Ping-pong a polyline for `duration` seconds. The result is ground truth."""
    if len(waypoints) < 2:
        raise ValueError("a route needs two points")
    cycle = list(waypoints) + list(reversed(waypoints[:-1]))
    once = _interpolate_route(cycle, [speed] * len(cycle), dt)
    if len(once) < 2:
        return once
    period = max(once[-1].t - once[0].t, dt)
    out: list[DynamicWaypoint] = []
    shift = 0.0
    while shift < duration + 1e-6:
        for sample in once:
            t = sample.t + shift
            if t > duration + 1e-6:
                break
            out.append(
                DynamicWaypoint(
                    t=t, x=sample.x, z=sample.z, yaw=sample.yaw, speed=sample.speed
                )
            )
        shift += period
    return out


def _crossing_forklifts(duration: float = 180.0) -> list[DynamicObject]:
    """Two agents whose routes stay in open bays of industrial_facility_v1.

    The east-west forklift shuttles along the main lane and crosses x=18,
    which is the gap a rover uses to leave the west bay. The second stays in
    the north bay so a destination up there is not an empty room.
    """
    lane = _loop_route([(8.0, 14.2), (28.0, 14.2)], speed=1.1, duration=duration)
    north = _loop_route([(6.0, 22.0), (16.0, 22.0)], speed=0.8, duration=duration)
    specs = (
        ("forklift_lane", "forklift", 2.4, 1.2, 2.0, lane),
        ("forklift_north", "forklift", 2.4, 1.2, 2.0, north),
    )
    agents = []
    for obj_id, kind, length, width, height, route in specs:
        agents.append(
            DynamicObject(
                id=obj_id,
                type=kind,
                length=length,
                width=width,
                height=height,
                behavior="route_follow",
                ground_truth=route,
                prediction=[],
                prediction_model="constant_velocity",
            )
        )
    return agents


def _paint_forecast(grid: OccupancyGrid, agents: list[DynamicObject], t: float) -> None:
    """Stamp constant-velocity sweeps. This is the forecast, not future truth."""
    from northstar.facility.occupancy import _paint_box
    from northstar.control.scene import causal_kinematics

    for agent in agents:
        kin = causal_kinematics(agent, t)
        if kin is None:
            continue
        x, z, yaw, vx, vz = kin
        steps = int(FORECAST_S / 0.5) + 1
        for k in range(steps):
            _paint_box(
                grid,
                OrientedBox(
                    x + vx * 0.5 * k,
                    z + vz * 0.5 * k,
                    yaw,
                    agent.length,
                    agent.width,
                ),
            )


def _makes_progress(
    polyline: np.ndarray, state: VehicleState, goal: tuple[float, float]
) -> bool:
    """A local candidate has to end closer to the destination than it started.

    The tracker only consumes the position polyline. A collision-free squiggle
    that does not approach the goal is not a usable route, and Hybrid A*
    should take over rather than the rover following it.
    """
    if polyline.shape[0] < 2:
        return False
    start_gap = math.hypot(goal[0] - state.x, goal[1] - state.z)
    if start_gap <= GOAL_RADIUS_M:
        return True
    end = polyline[-1]
    end_gap = math.hypot(goal[0] - float(end[0]), goal[1] - float(end[1]))
    length = float(np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum())
    if length < 0.4:
        return False
    return end_gap < start_gap - PROGRESS_M


def _flow_curves(snapshots, selected: Optional[int]) -> list[dict]:
    """Subsample real ODE states into local-frame polylines for the graph.

    `t` is the flow time in [0, 1]. Coordinates are metres in the rover frame
    (x forward, y left), not a world path.
    """
    if not snapshots:
        return []
    curves = []
    for flow_t, decoded in snapshots:
        array = np.asarray(decoded)
        if array.ndim != 3:
            continue
        for index in range(array.shape[0]):
            points = []
            for point in array[index, ::4, :2]:
                points.append({"x": round(float(point[0]), 2), "y": round(float(point[1]), 2)})
            if len(points) >= 2:
                curves.append(
                    {
                        "t": round(float(flow_t), 3),
                        "index": int(index),
                        "selected": selected is not None and index == selected,
                        "points": points,
                    }
                )
    return curves


def _raster(grid: np.ndarray, n: int = VIZ_GRID) -> list[int]:
    """Max-pool a world occupancy crop into the cells the panel draws.

    These are the same rasters the encoder is fed, coarsened for the socket.
    A cell is occupied or free. Nothing here is a hidden activation.
    """
    array = np.asarray(grid, dtype=np.float32)
    if array.ndim != 2 or array.size == 0:
        return [0] * (n * n)
    height, width = array.shape
    rows = np.linspace(0, height, n + 1).astype(np.int64)
    cols = np.linspace(0, width, n + 1).astype(np.int64)
    cells: list[int] = []
    for i in range(n):
        y0 = int(rows[i])
        y1 = max(int(rows[i + 1]), y0 + 1)
        for j in range(n):
            x0 = int(cols[j])
            x1 = max(int(cols[j + 1]), x0 + 1)
            cells.append(1 if float(array[y0:y1, x0:x1].max()) >= 0.5 else 0)
    return cells


def _clip_horizon(
    points: list[tuple[float, float]], state: VehicleState, horizon_m: float
) -> list[tuple[float, float]]:
    """Keep the part of a route that gets around the blockage, not the rest of the aisle."""
    if len(points) < 2:
        return points
    nearest = min(
        range(len(points)),
        key=lambda index: math.hypot(points[index][0] - state.x, points[index][1] - state.z),
    )
    kept = [points[nearest]]
    travelled = 0.0
    previous = points[nearest]
    for point in points[nearest + 1 :]:
        step = math.hypot(point[0] - previous[0], point[1] - previous[1])
        if travelled + step >= horizon_m and travelled >= 2.0:
            remain = max(0.0, horizon_m - travelled)
            if step > 1e-6:
                share = remain / step
                kept.append(
                    (
                        previous[0] + share * (point[0] - previous[0]),
                        previous[1] + share * (point[1] - previous[1]),
                    )
                )
            break
        travelled += step
        kept.append(point)
        previous = point
    return kept if len(kept) >= 2 else list(points[:2])


def _latch_mode(shown: str, until: float, mode: str, now: float) -> tuple[str, float]:
    """Hold Correcting across a one-step gap between a brake and the next label.

    Arriving at the goal still replaces it immediately. Hall runs never latch
    an emergency stop — that froze the demo chip while a replan was still due.
    """
    if mode == "goal":
        return mode, 0.0
    if mode == "intervention":
        return "intervention", now + MODE_LATCH_S
    if shown == "intervention" and now < until and mode in ("tracking", "replanning"):
        return "intervention", until
    return mode, 0.0


def _display_mode(diagnostics, reached_goal: bool, planning: bool) -> str:
    """Keep the chip on the correction while the filter is changing the command.

    A replan request used to outrank that correction, so the label flipped
    between Replan and Correcting on consecutive steps of the same brake.
    An infeasible program is a hard brake and a new route — never an emergency
    stop. That label froze the hall demo when a pad briefly overlapped the body.
    """
    if reached_goal:
        return "goal"
    infeasible = diagnostics is not None and bool(
        getattr(diagnostics, "infeasible", False)
    )
    correcting = diagnostics is not None and (
        bool(getattr(diagnostics, "intervened", False)) or infeasible
    )
    if correcting:
        if planning and infeasible:
            return "replanning"
        return "intervention"
    # Hall must not inherit bridge.controller_mode's estop-on-infeasible path.
    if infeasible:
        return "replanning" if planning else "intervention"
    return controller_mode(diagnostics, reached_goal, planning)


def _closest_scene(
    scene: SafetyScene,
    state: VehicleState,
    limit: int = 8,
    horizon_s: float = 3.0,
) -> SafetyScene:
    """Keep the bodies the footprint can meet, not merely the nearest centres.

    Ranking by centre distance dropped long racks and wide boxes whose face was
    already on the rover while six small centres sat closer. Moving agents were
    dropped the same way, so a crossing forklift could reach the rover without
    ever entering the quadratic program.
    """
    obstacles = scene.obstacles()
    if len(obstacles) <= limit:
        return scene

    def reach(obs: BoxObstacle | CircleObstacle) -> float:
        if isinstance(obs, CircleObstacle):
            radius = obs.radius
        else:
            radius = 0.5 * math.hypot(obs.length, obs.width)
        gap = math.hypot(obs.x - state.x, obs.z - state.z) - radius
        speed = math.hypot(getattr(obs, "vx", 0.0), getattr(obs, "vz", 0.0))
        return gap - speed * horizon_s

    movers = [
        obs
        for obs in obstacles
        if math.hypot(getattr(obs, "vx", 0.0), getattr(obs, "vz", 0.0)) > 0.05
    ]
    static = [obs for obs in obstacles if obs not in movers]
    static.sort(key=reach)
    keep = list(movers) + static[: max(0, limit - len(movers))]
    return SafetyScene(
        t=scene.t,
        boxes=[obs for obs in keep if isinstance(obs, BoxObstacle)],
        circles=[obs for obs in keep if isinstance(obs, CircleObstacle)],
    )


def _horizon_consumed(tracker: TrajectoryTracker, state: VehicleState) -> bool:
    if tracker.total < 0.5:
        return True
    return tracker.closest_s(state) >= tracker.total - 1.0


def _route_blocked(
    tracker: TrajectoryTracker, state: VehicleState, grid: OccupancyGrid
) -> bool:
    s_along = tracker.closest_s(state)
    reach = min(tracker.total, s_along + FORECAST_S)
    s = s_along + 0.4
    while s <= reach + 1e-6:
        x, z = tracker.point_at(s)
        if grid.is_occupied(x, z):
            return True
        s += 0.45
    return False


class InteractiveSim:
    """One facility, one rover, commands applied between control steps."""

    def __init__(
        self,
        planner: Optional[FlowMatchingPlanner] = None,
        seed: int = 1,
        use_cbf: bool = True,
        facility_path=FACILITY_PATH,
        dynamics: Optional[list[DynamicObject]] = None,
        layout: str = "facility",
        async_plan: bool = False,
        pace: float = 1.0,
    ):
        self.planner = planner
        self.seed = int(seed)
        self.use_cbf = use_cbf
        self.layout = layout
        self.async_plan = async_plan
        self.pace = pace
        self.control_dt = HALL_CONTROL_DT if layout == "hall" else CONTROL_DT
        self.cbf_config = HALL_CBF if layout == "hall" else None
        self.sizes = HALL_SIZES if layout == "hall" else OBSTACLE_SIZES
        self.params = VehicleParams()
        self.limits = _vehicle_limits()
        self.commands: queue.Queue = queue.Queue()
        self._lock = threading.RLock()
        self.statics: list[StaticObject] = []
        self._planner_busy = False
        self._generation = 0
        self._spawn_i = 0
        if layout == "hall":
            self.facility = hall_facility()
            self.dynamics = hall_crossing() if dynamics is None else list(dynamics)
            self.state = VehicleState(
                x=HALL_START_X,
                z=HALL_START_Z,
                yaw=HALL_START_YAW,
                speed=0.0,
                steering=0.0,
                t=0.0,
            )
        else:
            self.facility = load_facility(facility_path)
            self.dynamics = _crossing_forklifts() if dynamics is None else list(dynamics)
            self.state = VehicleState(
                x=START_X, z=START_Z, yaw=0.0, speed=0.0, steering=0.0, t=0.0
            )
        self.goal: Optional[tuple[float, float]] = None
        self.metrics = EpisodeMetrics(start_gap_m=0.0)
        self.tracker: TrajectoryTracker | HoldTracker | None = None
        self.source = "fallback"
        self.candidate_paths: list[list[dict]] = []
        self.selected_path: list[dict] = []
        self.fallback_path: list[dict] = []
        self.flow: list[dict] = []
        self.conditioning: dict | None = None
        self.trail: list[dict] = []
        self.plan_latency_ms: Optional[float] = None
        self.last_plan_t = -1e9
        self.replan_hold_s = 0.0
        self._recovery_replan = False
        self._shown_mode = "tracking"
        self._shown_until = 0.0
        self._stall_anchor: Optional[tuple[float, float, float]] = None
        self.arriving = False
        self.tick = 0
        self._static_dirty = True
        self.base_facility = build_facility_occupancy(self.facility, MAP_RESOLUTION)
        self.base_static = self.base_facility
        self._rebuild_static()

    def submit(self, command: dict) -> None:
        self.commands.put(command)

    def try_goal(self, x: float, z: float) -> bool:
        with self._lock:
            free = self._point_free(x, z, body_clearance_m=1.2)
        if not free:
            return False
        self.submit({"type": "set_goal", "x": float(x), "z": float(z)})
        return True

    def try_spawn(self, kind: str, x: float, z: float, yaw: float) -> bool:
        if kind not in self.sizes:
            return False
        with self._lock:
            free = self._point_free(x, z, body_clearance_m=1.4)
        if not free:
            return False
        self.submit(
            {
                "type": "spawn_obstacle",
                "kind": kind,
                "x": float(x),
                "z": float(z),
                "yaw": float(yaw),
            }
        )
        return True

    def place_obstacle(self, kind: str, x: float, z: float, yaw: float) -> bool:
        """Insert a drawn obstacle even when it sits near another body.

        The setup screen shows every stack the user placed. `try_spawn` refuses
        a point near the forklift or another box, and the rover then drives
        through a mesh the planner never received. The hall layout uses this
        instead, and still refuses a point outside the map.
        """
        if kind not in self.sizes:
            return False
        origin_x = float(self.facility.origin.get("x", 0.0))
        origin_z = float(self.facility.origin.get("z", 0.0))
        width = self.facility.dimensions["width_x"]
        depth = self.facility.dimensions["depth_z"]
        if not (
            origin_x <= x <= origin_x + width and origin_z <= z <= origin_z + depth
        ):
            return False
        self.submit(
            {
                "type": "spawn_obstacle",
                "kind": kind,
                "x": float(x),
                "z": float(z),
                "yaw": float(yaw),
            }
        )
        return True

    def _point_free(self, x: float, z: float, body_clearance_m: float) -> bool:
        origin_x = float(self.facility.origin.get("x", 0.0))
        origin_z = float(self.facility.origin.get("z", 0.0))
        width = self.facility.dimensions["width_x"]
        depth = self.facility.dimensions["depth_z"]
        margin = 1.2
        if not (
            origin_x + margin <= x <= origin_x + width - margin
            and origin_z + margin <= z <= origin_z + depth - margin
        ):
            return False
        if math.hypot(x - self.state.x, z - self.state.z) < body_clearance_m:
            return False
        for dx in (-0.6, 0.0, 0.6):
            for dz in (-0.6, 0.0, 0.6):
                if self.base_static.is_occupied(x + dx, z + dz):
                    return False
        for agent in self.dynamics:
            pose = sample_dynamic_pose(agent, self.state.t, use_prediction=False)
            if pose is None:
                continue
            if math.hypot(pose[0] - x, pose[1] - z) < 2.2:
                return False
        for obj in self.statics:
            if math.hypot(obj.x - x, obj.z - z) < 1.6:
                return False
        return True

    def _rebuild_static(self) -> None:
        self.base_static = build_facility_occupancy(
            self.facility, MAP_RESOLUTION, static_obstacles=self.statics
        )
        self._static_dirty = False

    def _drain(self) -> tuple[bool, bool]:
        goal_changed = False
        obstacles_changed = False
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                break
            kind = command.get("type")
            if kind == "set_goal":
                self._generation += 1
                x, z = float(command["x"]), float(command["z"])
                if self._point_free(x, z, body_clearance_m=1.2) or (
                    self.goal == (x, z)
                ):
                    self.goal = (x, z)
                    self.metrics.goal_reached = False
                    self.arriving = False
                    self.metrics.start_gap_m = float(
                        math.hypot(x - self.state.x, z - self.state.z)
                    )
                    self.metrics.travelled_m = 0.0
                    goal_changed = True
            elif kind == "spawn_obstacle":
                self._generation += 1
                kind_name = str(command.get("kind", ""))
                spec = self.sizes.get(kind_name)
                if spec is None:
                    continue
                length, width, height = spec
                self._spawn_i += 1
                x = float(command["x"])
                z = float(command["z"])
                yaw = float(command.get("yaw", 0.0))
                if kind_name == "forklift":
                    # Placed forklifts patrol. A static body in the aisle is what
                    # pinned the rover with no route once the filter braked.
                    self.dynamics.append(
                        placed_forklift(x, z, yaw, index=self._spawn_i)
                    )
                else:
                    self.statics.append(
                        StaticObject(
                            id=f"user_{self._spawn_i}",
                            type=kind_name,
                            x=x,
                            z=z,
                            yaw=yaw,
                            length=length,
                            width=width,
                            height=height,
                        )
                    )
                obstacles_changed = True
            elif kind == "remove_obstacle":
                self._generation += 1
                target = str(command.get("id", ""))
                before = len(self.statics)
                self.statics = [o for o in self.statics if o.id != target]
                before_d = len(self.dynamics)
                self.dynamics = [o for o in self.dynamics if o.id != target]
                if len(self.statics) != before or len(self.dynamics) != before_d:
                    obstacles_changed = True
            elif kind == "clear_obstacles":
                self._generation += 1
                before_s = len(self.statics)
                before_d = len(self.dynamics)
                self.statics = []
                # Keep the staged hall crossing; drop every user-placed patrol.
                self.dynamics = [
                    agent for agent in self.dynamics if not agent.id.startswith("user_")
                ]
                if len(self.statics) != before_s or len(self.dynamics) != before_d:
                    obstacles_changed = True
            elif kind == "toggle_safety":
                # The hall run keeps the barrier filter on. Turning it off is
                # what let the rover drive into a body and then estop inside it.
                if self.layout != "hall":
                    self.use_cbf = bool(command.get("enabled", True))
        if obstacles_changed:
            self._rebuild_static()
        return goal_changed, obstacles_changed

    def _cv_occupancy(self, t_now: float, t_query: float) -> OccupancyGrid:
        """Static map plus agents extrapolated from the velocity sensed at `t_now`."""
        from northstar.control.scene import causal_kinematics
        from northstar.facility.occupancy import _paint_box

        grid = OccupancyGrid(
            self.base_static.origin_x,
            self.base_static.origin_z,
            self.base_static.resolution,
            self.base_static.width,
            self.base_static.height,
            data=self.base_static.data.copy(),
        )
        horizon = t_query - t_now
        for agent in self.dynamics:
            kin = causal_kinematics(agent, t_now)
            if kin is None:
                continue
            x, z, yaw, vx, vz = kin
            _paint_box(
                grid,
                OrientedBox(x + vx * horizon, z + vz * horizon, yaw, agent.length, agent.width),
            )
        return grid

    def _forecast_grid(self, t: float) -> OccupancyGrid:
        grid = OccupancyGrid(
            self.base_static.origin_x,
            self.base_static.origin_z,
            self.base_static.resolution,
            self.base_static.width,
            self.base_static.height,
            data=self.base_static.data.copy(),
        )
        _paint_forecast(grid, self.dynamics, t)
        return grid

    def _hybrid(
        self,
        state: VehicleState,
        goal: Optional[tuple[float, float]] = None,
        statics: Optional[list[StaticObject]] = None,
        dynamics: Optional[list[DynamicObject]] = None,
    ) -> Optional[list[tuple[float, float]]]:
        goal = self.goal if goal is None else goal
        statics = self.statics if statics is None else statics
        dynamics = self.dynamics if dynamics is None else dynamics
        if goal is None:
            return None
        from northstar.control.scene import causal_kinematics

        extras: list[StaticObject] = []
        for agent in dynamics:
            kin = causal_kinematics(agent, state.t)
            if kin is None:
                continue
            x, z, yaw, vx, vz = kin
            for k, horizon in enumerate((0.0, 2.0, 4.0)):
                extras.append(
                    StaticObject(
                        id=f"{agent.id}_block_{k}",
                        type="block",
                        x=x + vx * horizon,
                        z=z + vz * horizon,
                        yaw=yaw,
                        length=agent.length + 0.4,
                        width=agent.width + 0.4,
                    )
                )
        occupied = [*statics, *extras]
        # The 0.25 m map already has the placed boxes. Rebuilding at 0.5 m
        # left pallet centres in free cells, and the occupancy shortcut then
        # never tested the box, so the primitive drove through the stack.
        grid = OccupancyGrid(
            self.base_static.origin_x,
            self.base_static.origin_z,
            self.base_static.resolution,
            self.base_static.width,
            self.base_static.height,
            data=self.base_static.data.copy(),
        )
        from northstar.facility.occupancy import _paint_box

        for extra in extras:
            _paint_box(
                grid,
                OrientedBox(extra.x, extra.z, extra.yaw, extra.length, extra.width),
            )
        checker = CollisionChecker(
            self.facility, occupied, occupancy=grid, vehicle_params=self.params
        )
        checker.fast_occupancy = True
        search = HybridAStar(checker, config=FALLBACK_CONFIG)
        # Search from a moving pose so the first primitive clears one cell.
        # Only the polyline is kept; the plant keeps its real speed.
        start = state.copy()
        start.speed = max(state.speed, 1.0)
        goal_state = VehicleState(
            x=goal[0], z=goal[1], yaw=state.yaw, speed=0.0, t=state.t
        )
        path = search.plan(start, goal_state)
        if not path:
            return None
        return [(float(sample.x), float(sample.z)) for sample in path]

    def _adopt(self, polyline: list[tuple[float, float]] | np.ndarray, learned: bool) -> None:
        points = [(float(a), float(b)) for a, b in polyline]
        self.tracker = TrajectoryTracker(
            points,
            params=self.params,
            cruise_speed=HALL_CRUISE if self.layout == "hall" else CRUISE_SPEED,
            lookahead=HALL_LOOKAHEAD if self.layout == "hall" else TRACKER_LOOKAHEAD,
            steer_gain=TRACKER_STEER_GAIN,
            turn_speed=HALL_TURN_SPEED if self.layout == "hall" else 0.8,
            turn_slow_angle=1.05 if self.layout == "hall" else 0.7,
        )
        self._stall_anchor = None
        stamped = path_points(points, dt=None, t0=self.state.t)
        if learned:
            self.source = "learned"
            self.selected_path = stamped
            self.fallback_path = []
        else:
            self.source = "fallback"
            self.selected_path = []
            self.fallback_path = stamped

    def _plan(self, state: VehicleState, generation: Optional[int] = None) -> None:
        """Search off to the side, then publish under the session lock.

        `generation` drops a result when the goal or the obstacles changed
        while the network was integrating.
        """
        goal = self.goal
        facility_grid = self.base_facility
        static_grid = self.base_static
        dynamics = list(self.dynamics)
        statics = list(self.statics)
        started = time.perf_counter()
        learned = False
        candidate_paths: list[list[dict]] = []
        flow: Optional[list[dict]] = None
        polyline = None
        facility_crop = facility_grid.crop_local(state.x, state.z, MAP_SIZE_M)
        static_crop = static_grid.crop_local(state.x, state.z, MAP_SIZE_M)
        conditioning = {
            "size": VIZ_GRID,
            "facility": _raster(facility_crop.data),
            "static": _raster(static_crop.data),
        }
        if self.planner is not None and goal is not None:
            dynamic_stack = None
            if self.planner.uses_dynamic:
                dynamic_stack = np.stack(
                    [
                        self._cv_occupancy(state.t, state.t + k * DYNAMIC_SLICE_DT)
                        .crop_local(state.x, state.z, MAP_SIZE_M)
                        .data
                        for k in range(DYNAMIC_SLICES)
                    ]
                )
            ego = {"x": state.x, "z": state.z, "yaw": state.yaw}
            gx, gy, gyaw = world_to_local(
                np.array(goal[0]), np.array(goal[1]), np.array(0.0), ego
            )
            result = self.planner.plan(
                facility_occ=facility_crop.data,
                static_occ=static_crop.data,
                dynamic_occ=dynamic_stack,
                goal_local=(float(gx), float(gy)),
                goal_yaw_local=float(gyaw),
                speed=state.speed,
                steering=state.steering,
                ego_yaw=state.yaw,
                num_candidates=INTERACTIVE_CANDIDATES,
                num_steps=INTERACTIVE_STEPS,
                seed=self.seed + self.metrics.replans + 1,
                record_flow=True,
            )
            forecast = self._forecast_grid(state.t)
            flags, clearances = _candidate_collisions(
                result.candidates, state, forecast, self.limits
            )
            dt_traj = float(self.planner.schema.get("dt", 0.2))
            candidate_paths = [
                path_points(c.world_xz(ego), dt=dt_traj, t0=state.t)
                for c in result.candidates
            ]
            pick = best_candidate(result, collision_free=flags, clearance_m=clearances)
            selected = pick.candidate.index if pick is not None else None
            flow = _flow_curves(getattr(result, "flow", None), selected)
            if pick is not None:
                proposal = pick.candidate.world_xz(ego)
                if _makes_progress(proposal, state, goal):
                    polyline = proposal
                    learned = True
        if not learned:
            route = self._hybrid(state, goal, statics, dynamics)
            if route is not None and len(route) >= 2:
                polyline = _clip_horizon(route, state, FALLBACK_HORIZON_M)
            else:
                polyline = None
        latency_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            if generation is not None and generation != self._generation:
                return
            self.metrics.replans += 1
            self.last_plan_t = state.t
            self.plan_latency_ms = latency_ms
            if self.planner is not None:
                self.candidate_paths = candidate_paths
            if flow:
                self.flow = flow
            self.conditioning = conditioning
            if polyline is not None:
                self._adopt(polyline, learned=learned)
            elif not isinstance(self.tracker, TrajectoryTracker):
                # A failed search used to drop the ribbon the rover was already
                # on. Keep that reference; the filter still refuses a step that
                # would enter an obstacle.
                self.source = "fallback"
                self.selected_path = []
                self.fallback_path = []
                self.tracker = HoldTracker(self.params)

    def _plan_worker(self, generation: int, state: VehicleState) -> None:
        try:
            self._plan(state, generation=generation)
        finally:
            with self._lock:
                self._planner_busy = False

    def _needs_replan(
        self,
        state: VehicleState,
        goal_changed: bool,
        obstacles_changed: bool,
        forecast: OccupancyGrid,
    ) -> bool:
        if self.goal is None or self.metrics.goal_reached or self.arriving:
            return False
        if goal_changed or obstacles_changed or self.tracker is None:
            return True
        if self._recovery_replan:
            return True
        if isinstance(self.tracker, HoldTracker):
            return True
        hold_limit = HALL_REPLAN_HOLD_S if self.layout == "hall" else REPLAN_HOLD_S
        if self.replan_hold_s >= hold_limit:
            return True
        if _horizon_consumed(self.tracker, state):
            return True
        return _route_blocked(self.tracker, state, forecast)

    def _visible_obstacles(self, t: float) -> list[dict]:
        from northstar.control.scene import causal_kinematics
        from backend.ml.bridge import _xy

        out = []
        for obstacle in self.statics:
            x, y = _xy(obstacle.x, obstacle.z)
            out.append(
                {
                    "id": obstacle.id,
                    "position": [x, y, float(obstacle.y)],
                    "velocity": [0.0, 0.0, 0.0],
                    "shape": "box",
                    "size": [float(obstacle.length), float(obstacle.width), float(obstacle.height)],
                    "kind": obstacle.type,
                    "yaw": float(obstacle.yaw),
                }
            )
        for agent in self.dynamics:
            pose = sample_dynamic_pose(agent, t, use_prediction=False)
            if pose is None:
                continue
            ax, az, ayaw = pose
            kin = causal_kinematics(agent, t)
            vx = vz = 0.0
            if kin is not None:
                vx, vz = kin[3], kin[4]
            x, y = _xy(ax, az)
            out.append(
                {
                    "id": agent.id,
                    "position": [x, y, float(agent.y)],
                    "velocity": [float(vx), float(vz), 0.0],
                    "shape": "box",
                    "size": [float(agent.length), float(agent.width), float(agent.height)],
                    "kind": agent.type,
                    "yaw": float(ayaw),
                }
            )
        return out

    def _step_locked(self):
        goal_changed, obstacles_changed = self._drain()
        state = self.state
        if self.goal is not None and not self.metrics.goal_reached:
            gap = math.hypot(self.goal[0] - state.x, self.goal[1] - state.z)
            if gap <= GOAL_RADIUS_M:
                self.arriving = True
            if self.arriving and state.speed < 0.4:
                self.metrics.goal_reached = True

        forecast = self._forecast_grid(state.t) if self.goal is not None else self.base_static
        replanned = False
        launch = None
        if self._needs_replan(state, goal_changed, obstacles_changed, forecast):
            gap = HALL_REPLAN_GAP_S if self.layout == "hall" else MIN_REPLAN_GAP_S
            # A braked rover waiting on a forklift used the long hall gap and
            # sat forever. Ask for a new route as soon as the lane opens.
            if state.speed < 0.3 and self.replan_hold_s > 0.0:
                gap = min(gap, 0.6)
            due = state.t - self.last_plan_t >= gap
            urgent = (
                goal_changed
                or obstacles_changed
                or self.tracker is None
                or self._recovery_replan
                or (
                    state.speed < 0.15
                    and self.replan_hold_s >= (
                        HALL_REPLAN_HOLD_S if self.layout == "hall" else REPLAN_HOLD_S
                    )
                )
            )
            if due or urgent:
                self._recovery_replan = False
                background = (
                    self.async_plan
                    and self.tracker is not None
                    and not self._planner_busy
                )
                if background:
                    self._planner_busy = True
                    launch = (self._generation, state.copy())
                else:
                    self._plan(state)
                    replanned = True
                    self.replan_hold_s = 0.0
        if self.tracker is None:
            self.tracker = HoldTracker(self.params)

        if (self.arriving or self.metrics.goal_reached) and not isinstance(
            self.tracker, HoldTracker
        ):
            self.tracker = HoldTracker(self.params)

        perceived = build_scene(
            state.t,
            facility=self.facility,
            static_obstacles=self.statics,
            dynamic_obstacles=self.dynamics,
            dynamics_mode="causal_cv",
        )
        if self.layout == "hall":
            perceived = _closest_scene(perceived, state)
        nxt, nominal, applied, diagnostics = closed_loop_step(
            state,
            self.tracker,
            perceived,
            self.control_dt,
            params=self.params,
            config=self.cbf_config,
            use_cbf=True if self.layout == "hall" else self.use_cbf,
        )
        truth = build_scene(
            nxt.t,
            facility=self.facility,
            static_obstacles=self.statics,
            dynamic_obstacles=self.dynamics,
            dynamics_mode="snapshot_gt",
        )
        clearance = body_clearance(nxt, truth, self.params)
        if clearance < -PENETRATION_M:
            self.metrics.collisions += 1
            # Hall pads are soft demo geometry. Freezing the pose here left the
            # rover latched inside the box with no way to crawl onto a new route.
            if self.layout != "hall":
                held = state.copy()
                held.t = state.t + self.control_dt
                held.speed = 0.0
                nxt = held
                truth = build_scene(
                    nxt.t,
                    facility=self.facility,
                    static_obstacles=self.statics,
                    dynamic_obstacles=self.dynamics,
                    dynamics_mode="snapshot_gt",
                )
                clearance = body_clearance(nxt, truth, self.params)

        if self.metrics.min_clearance is None:
            self.metrics.min_clearance = float(clearance)
        else:
            self.metrics.min_clearance = min(self.metrics.min_clearance, float(clearance))

        if diagnostics is not None and diagnostics.intervened:
            self.metrics.interventions += 1
        # The heading probe goes true while the nose still points at the thing
        # the ribbon is steering around. Replacing that ribbon every hold
        # window is what flipped the chip through Correcting, Watching, and
        # Replan without the rover ever taking the turn. A ribbon that is
        # still clear is kept. A rover that has not moved on it is allowed
        # to ask for a different one.
        if self._stall_anchor is None:
            self._stall_anchor = (float(nxt.x), float(nxt.z), float(nxt.t))
        ax, az, at = self._stall_anchor
        if math.hypot(nxt.x - ax, nxt.z - az) >= 0.35:
            self._stall_anchor = (float(nxt.x), float(nxt.z), float(nxt.t))
            stalled = False
        else:
            stalled = float(nxt.t) - at >= 2.5
        path_clear = (
            isinstance(self.tracker, TrajectoryTracker)
            and not _horizon_consumed(self.tracker, nxt)
            and not _route_blocked(self.tracker, nxt, forecast)
        )
        recovery = (
            not self.metrics.goal_reached
            and (
                clearance < 0.0
                or (
                    diagnostics is not None
                    and bool(getattr(diagnostics, "infeasible", False))
                )
            )
        )
        if recovery:
            self.replan_hold_s += self.control_dt
            self._recovery_replan = True
        elif (
            diagnostics is not None
            and diagnostics.replan_required
            and not self.metrics.goal_reached
            and (not path_clear or stalled)
        ):
            self.replan_hold_s += self.control_dt
        else:
            self.replan_hold_s = 0.0

        self.metrics.travelled_m += float(
            math.hypot(nxt.x - state.x, nxt.z - state.z)
        )
        from backend.ml.bridge import _xy

        trail_x, trail_y = _xy(nxt.x, nxt.z)
        self.trail.append({"x": trail_x, "y": trail_y, "t": float(nxt.t)})
        if len(self.trail) > TRAIL_LIMIT:
            self.trail = self.trail[-TRAIL_LIMIT:]

        mode = _display_mode(
            diagnostics, self.metrics.goal_reached, replanned or self._planner_busy
        )
        mode, self._shown_until = _latch_mode(
            self._shown_mode, self._shown_until, mode, float(nxt.t)
        )
        self._shown_mode = mode
        latency = self.plan_latency_ms if replanned else (
            None if diagnostics is None else float(diagnostics.solve_time_s) * 1000.0
        )
        frame = state_frame(
            tick=self.tick,
            sim_time=nxt.t,
            seed=self.seed,
            state=nxt,
            goal_xz=self.goal,
            obstacles=self._visible_obstacles(nxt.t),
            candidate_paths=self.candidate_paths,
            selected_path=self.selected_path,
            fallback_path=self.fallback_path,
            trail=list(self.trail),
            controller=controller_state(
                nominal, applied, diagnostics, mode, self.source, latency
            ),
            metrics=self.metrics.to_dict(),
            flow=self.flow or None,
            conditioning=self.conditioning,
        )
        self.state = nxt
        self.tick += 1
        return frame, launch

    def step(self) -> dict:
        with self._lock:
            frame, launch = self._step_locked()
        if launch is not None:
            generation, state = launch
            threading.Thread(
                target=self._plan_worker, args=(generation, state), daemon=True
            ).start()
        return frame

    def frames(self) -> Iterator[dict]:
        while True:
            started = time.perf_counter()
            frame = self.step()
            budget = self.control_dt / max(self.pace, 0.25)
            remaining = budget - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
            yield frame
