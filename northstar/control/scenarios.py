"""Deterministic scenarios for the safety filter.

Nominal commands come from the obstacle-unaware tracker (or, where noted, a
fixed command). Dynamic agents publish a ground-truth polyline. The filter is
given only a constant-velocity forecast built from the current sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from northstar.control.scene import BoxObstacle
from northstar.control.simulate import SimWorld
from northstar.control.tracking import TrajectoryTracker
from northstar.facility.schema import DynamicObject, DynamicWaypoint
from northstar.vehicle.bicycle import ControlInput, VehicleParams, VehicleState


@dataclass
class Scenario:
    name: str
    description: str
    start: VehicleState
    goal_xy: tuple[float, float]
    tracker: TrajectoryTracker
    world: SimWorld
    dt: float
    duration: float


def straight_agent(
    obj_id: str,
    x0: float,
    z0: float,
    yaw: float,
    speed: float,
    length: float,
    width: float,
    duration: float,
    dt: float = 0.1,
    obj_type: str = "forklift",
) -> DynamicObject:
    """Constant-velocity ground truth. No privileged forecast is stored."""
    samples: list[DynamicWaypoint] = []
    t = 0.0
    x, z = x0, z0
    vx = speed * math.cos(yaw)
    vz = speed * math.sin(yaw)
    while t <= duration + 1e-9:
        samples.append(DynamicWaypoint(t=t, x=x, z=z, yaw=yaw, speed=speed))
        x += vx * dt
        z += vz * dt
        t += dt
    return DynamicObject(
        id=obj_id,
        type=obj_type,
        length=length,
        width=width,
        ground_truth=samples,
        prediction=[],
        prediction_model="withheld",
    )


def _tracker(path: list[tuple[float, float]], cruise: float, params: VehicleParams) -> TrajectoryTracker:
    return TrajectoryTracker(path, params=params, cruise_speed=cruise, lookahead=1.5, steer_gain=1.7)


def static_barrier(params: VehicleParams | None = None) -> Scenario:
    """Wide barrier across a straight route. Braking is the feasible response."""
    params = params or VehicleParams()
    goal = (14.0, 0.0)
    path = [(0.0, 0.0), goal]
    barrier = BoxObstacle(
        id="barrier",
        x=8.0,
        z=0.0,
        yaw=0.0,
        length=0.8,
        width=8.0,
        kind="barrier",
    )
    return Scenario(
        name="static_barrier",
        description="Approach a barrier that spans the route faster than a late brake can handle.",
        start=VehicleState(x=0.0, z=0.0, yaw=0.0, speed=2.0, steering=0.0, t=0.0),
        goal_xy=goal,
        tracker=_tracker(path, cruise=2.0, params=params),
        world=SimWorld(boxes=[barrier]),
        dt=0.1,
        duration=8.0,
    )


def crossing_forklift(params: VehicleParams | None = None) -> Scenario:
    """Forklift crosses the route. Prediction is constant velocity from sensing."""
    params = params or VehicleParams()
    goal = (18.0, 0.0)
    path = [(0.0, 0.0), goal]
    # Rover at 1.6 m/s reaches x=8 in 5.0 s. Forklift speed 1.2 from z=-6
    # reaches z=0 at the same time and then leaves toward +z.
    agent = straight_agent(
        "forklift_cross",
        x0=8.0,
        z0=-6.0,
        yaw=math.pi / 2.0,
        speed=1.2,
        length=2.4,
        width=1.2,
        duration=16.0,
        obj_type="forklift",
    )
    return Scenario(
        name="crossing_forklift",
        description="Yield to a forklift whose predicted path crosses the route.",
        start=VehicleState(x=0.0, z=0.0, yaw=0.0, speed=1.6, steering=0.0, t=0.0),
        goal_xy=goal,
        tracker=_tracker(path, cruise=1.6, params=params),
        world=SimWorld(dynamic_obstacles=[agent]),
        dt=0.1,
        duration=14.0,
    )


def narrow_corridor(params: VehicleParams | None = None) -> Scenario:
    """Reference path sits too close to one wall of a passable corridor."""
    params = params or VehicleParams()
    goal = (12.0, 0.42)
    path = [(0.0, 0.42), goal]
    walls = [
        BoxObstacle(id="wall_north", x=6.0, z=0.90, yaw=0.0, length=16.0, width=0.3, kind="wall"),
        BoxObstacle(id="wall_south", x=6.0, z=-0.90, yaw=0.0, length=16.0, width=0.3, kind="wall"),
    ]
    return Scenario(
        name="narrow_corridor",
        description="Nominal pure pursuit pulls the rover into a corridor wall.",
        start=VehicleState(x=0.0, z=0.0, yaw=0.0, speed=1.2, steering=0.0, t=0.0),
        goal_xy=goal,
        tracker=_tracker(path, cruise=1.2, params=params),
        world=SimWorld(boxes=walls),
        dt=0.1,
        duration=8.0,
    )


def follow_lead(params: VehicleParams | None = None) -> Scenario:
    """Slower vehicle ahead on the same line. The filter must not close the gap."""
    params = params or VehicleParams()
    goal = (22.0, 0.0)
    path = [(0.0, 0.0), goal]
    lead = straight_agent(
        "lead_agv",
        x0=6.0,
        z0=0.0,
        yaw=0.0,
        speed=0.55,
        length=1.4,
        width=0.8,
        duration=18.0,
        obj_type="agv",
    )
    return Scenario(
        name="follow_lead",
        description="Close on a slower AGV from behind.",
        start=VehicleState(x=0.0, z=0.0, yaw=0.0, speed=1.6, steering=0.0, t=0.0),
        goal_xy=goal,
        tracker=_tracker(path, cruise=1.6, params=params),
        world=SimWorld(dynamic_obstacles=[lead]),
        dt=0.1,
        duration=10.0,
    )


def unavoidable_wall(params: VehicleParams | None = None) -> Scenario:
    """Barrier closer than the stopping distance at the initial speed."""
    params = params or VehicleParams()
    goal = (10.0, 0.0)
    path = [(0.0, 0.0), goal]
    # Bumper is at x=0.65. Face of a 0.5 m thick barrier at x=1.00 (0.35 m of gap).
    barrier = BoxObstacle(
        id="sudden_wall",
        x=1.25,
        z=0.0,
        yaw=0.0,
        length=0.5,
        width=8.0,
        kind="barrier",
    )
    return Scenario(
        name="unavoidable_wall",
        description="Contact is already inside the braking distance.",
        start=VehicleState(x=0.0, z=0.0, yaw=0.0, speed=2.4, steering=0.0, t=0.0),
        goal_xy=goal,
        tracker=_tracker(path, cruise=2.4, params=params),
        world=SimWorld(boxes=[barrier]),
        dt=0.05,
        duration=1.2,
    )


def open_floor(params: VehicleParams | None = None) -> Scenario:
    """Empty floor. The filter should leave the tracker alone."""
    params = params or VehicleParams()
    goal = (12.0, 0.0)
    path = [(0.0, 0.0), goal]
    return Scenario(
        name="open_floor",
        description="No obstacles. Nominal tracking should be unchanged.",
        start=VehicleState(x=0.0, z=0.0, yaw=0.0, speed=0.4, steering=0.0, t=0.0),
        goal_xy=goal,
        tracker=_tracker(path, cruise=1.5, params=params),
        world=SimWorld(),
        dt=0.1,
        duration=12.0,
    )


class AggressiveTracker:
    """Reference that deliberately exceeds actuator limits."""

    def nominal_control(self, state: VehicleState, dt: float) -> ControlInput:
        del state, dt
        return ControlInput(accel=8.0, steer_cmd=2.0)


def all_scenarios(params: VehicleParams | None = None) -> list[Scenario]:
    params = params or VehicleParams()
    return [
        static_barrier(params),
        crossing_forklift(params),
        narrow_corridor(params),
        follow_lead(params),
        unavoidable_wall(params),
        open_floor(params),
    ]
