"""FastAPI seam between the flow-matching planner and the frontend.

The frontend's `SimSource` is `{live, subscribe(onFrame), send(intent)}`, so the
socket carries `StateFrame`s down and `Intent`s up, in the schema
`frontend/src/types/sim.ts` defines. Nothing here reshapes that schema:
`backend.ml.bridge` already emits it and validates against it.

Two ways to drive the UI:

- `live`: run the model, ranking, safety filter and plant now. Honest, but the
  CBF solve measures ~280 ms against a 100 ms control interval on CPU, so it
  streams slower than real time.
- `replay`: stream a recorded frame log at wall-clock pace. The frames are
  measured output from an earlier live run, so the visuals are real while the
  pacing is right.

Run it with:
    python -m uvicorn backend.api.server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.ml.bridge import FACILITY_PATH, SCHEMA_VERSION, episode, validate_frame
from backend.ml.inference import FlowMatchingPlanner
from backend.ml.hall import HALL_GOAL_X, HALL_GOAL_Z
from backend.ml.interactive import HALL_PACE, InteractiveSim
from northstar.facility.loader import load_facility

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "backend" / "artifacts"
FRAMES_DIR = ARTIFACTS / "frames"
DEFAULT_CHECKPOINT = ARTIFACTS / "v2_static_control.pt"
SCENARIO_DIRS = {
    "v2": ROOT / "data" / "dataset_v2_additions" / "scenarios",
    "v1": ROOT / "data" / "dataset_v1" / "scenarios",
}
CONTROL_DT_S = 0.1
# Deliberately tiny. `pump` paces a replay at CONTROL_DT_S per frame while the
# producer reads the log as fast as it can, so a deep queue lets the producer
# race seconds ahead of the socket - and then `pause` only stops production,
# leaving the whole backlog still to drain before the viewer sees any effect.
# At 2 the producer blocks on `put` and stays in lockstep with the send.
QUEUE_LIMIT = 2

app = FastAPI(title="Northstar planner bridge", version=str(SCHEMA_VERSION))
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

_planner: FlowMatchingPlanner | None = None
_planner_lock = threading.Lock()


def planner(checkpoint: Path = DEFAULT_CHECKPOINT) -> FlowMatchingPlanner:
    """Load the checkpoint once and share it across connections.

    Loading costs seconds and the planner is read-only under inference, so a
    per-connection load would make every reconnect feel like a hang.
    """
    global _planner
    with _planner_lock:
        if _planner is None:
            _planner = FlowMatchingPlanner.load(checkpoint)
        return _planner


def scenario_path(name: str) -> Path | None:
    for directory in SCENARIO_DIRS.values():
        candidate = directory / f"{name}.json"
        if candidate.exists():
            return candidate
    return None


def list_scenarios(limit: int = 40) -> list[str]:
    names: list[str] = []
    for directory in SCENARIO_DIRS.values():
        if directory.exists():
            names.extend(sorted(p.stem for p in directory.glob("*.json")))
    return names[:limit]


def list_replays() -> list[dict]:
    if not FRAMES_DIR.exists():
        return []
    out = []
    for path in sorted(FRAMES_DIR.glob("*.jsonl")):
        out.append(
            {
                "name": path.stem,
                "policy_mode": "safety_off" if "safety_off" in path.stem else "safety_on",
                "size_bytes": path.stat().st_size,
            }
        )
    return out


def replay_frames(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


@dataclass
class Session:
    """One socket's run: a worker thread producing frames, the socket draining.

    `episode()` is synchronous and CPU-bound, so it cannot run on the event
    loop without stalling every other connection. The thread owns the
    simulation, the queue is the handoff, and the events are the only shared
    mutable state.
    """

    frames: queue.Queue = field(default_factory=lambda: queue.Queue(QUEUE_LIMIT))
    stop: threading.Event = field(default_factory=threading.Event)
    resume: threading.Event = field(default_factory=threading.Event)
    steps: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(0))
    worker: threading.Thread | None = None
    use_cbf: bool = True
    seed: int = 0
    paced: bool = False
    sim: InteractiveSim | None = None

    def running(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def wait_turn(self) -> bool:
        """Block until this frame may be produced.

        Running means `resume` is set. Paused means it is not, and each `step`
        intent releases exactly one token. Toggling an event for a single step
        would race the worker, letting through zero frames or several.
        """
        while not self.stop.is_set():
            if self.resume.is_set():
                return True
            if self.steps.acquire(timeout=0.05):
                return True
        return False

    def drain(self) -> None:
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                return

    def halt(self) -> None:
        self.stop.set()
        self.resume.set()
        self.steps.release()
        # Drain while joining, not after. A producer blocked on `put` to a full
        # queue never reaches its next `stop` check, so a plain join would time
        # out and the thread would then wake and push a frame from the old run
        # into the next one.
        if self.worker is not None:
            deadline = time.monotonic() + 5.0
            while self.worker.is_alive() and time.monotonic() < deadline:
                self.drain()
                self.worker.join(timeout=0.05)
        self.worker = None
        self.drain()
        self.stop.clear()

    def launch(self, produce) -> None:
        self.halt()
        self.resume.set()

        def run() -> None:
            try:
                for frame in produce():
                    if not self.wait_turn():
                        break
                    self.frames.put(frame)
            except Exception as error:
                self.frames.put({"type": "error", "message": repr(error)})
            finally:
                self.frames.put(None)

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()


@app.get("/health")
def health() -> dict:
    checkpoint = DEFAULT_CHECKPOINT
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "checkpoint": checkpoint.name if checkpoint.exists() else None,
        "checkpoint_present": checkpoint.exists(),
        "scenarios": len(list_scenarios(limit=100000)),
        "replays": len(list_replays()),
    }


@app.get("/scenarios")
def scenarios() -> dict:
    return {"scenarios": list_scenarios()}


@app.get("/replays")
def replays() -> dict:
    return {"replays": list_replays()}


def facility_payload() -> dict:
    """Walls, columns and machinery in the backend frame.

    Dataset horizontal Z is backend Y. The frontend converts with the one
    helper in `scene/coords`, so this payload does not pre-transform.
    """
    spec = load_facility(FACILITY_PATH)
    return {
        "facility_id": spec.facility_id,
        "width_x": spec.dimensions["width_x"],
        "depth_y": spec.dimensions["depth_z"],
        "height_z": spec.dimensions.get("height_y", 6.0),
        "walls": [
            {
                "id": wall.id,
                "x1": wall.x1,
                "y1": wall.z1,
                "x2": wall.x2,
                "y2": wall.z2,
                "thickness": wall.thickness,
                "height": wall.height,
            }
            for wall in spec.walls
        ],
        "columns": [
            {
                "id": column.id,
                "x": column.x,
                "y": column.z,
                "radius": column.radius,
                "height": column.height,
            }
            for column in spec.columns
        ],
        "permanent": [
            {
                "id": obj.id,
                "kind": obj.type,
                "x": obj.x,
                "y": obj.z,
                "yaw": obj.yaw,
                "length": obj.length,
                "width": obj.width,
                "height": obj.height,
            }
            for obj in spec.permanent_objects
        ],
    }


@app.get("/facility")
def facility() -> dict:
    return facility_payload()


@app.get("/metrics")
def metrics() -> dict:
    """Measured safety-off/on comparison, or nulls when no runs are logged.

    The frontend's replay table must never show invented numbers, so an absent
    summary returns nulls rather than plausible-looking defaults.
    """
    summary = FRAMES_DIR / "summary.json"
    if not summary.exists():
        return {"measured": False, "rows": None}
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return {"measured": True, "rows": payload}


async def pump(socket: WebSocket, session: Session) -> None:
    loop = asyncio.get_running_loop()
    while True:
        frame = await loop.run_in_executor(None, session.frames.get)
        if frame is None:
            await socket.send_json({"type": "end"})
            return
        await socket.send_json(frame)
        if session.paced:
            await asyncio.sleep(CONTROL_DT_S)


def start_live(session: Session, name: str, duration_s: float):
    path = scenario_path(name)
    if path is None:
        raise FileNotFoundError(f"unknown scenario {name!r}")
    loaded = planner()

    def produce():
        return episode(
            path, loaded, use_cbf=session.use_cbf,
            duration_s=duration_s, seed=session.seed,
        )

    session.paced = False
    session.launch(produce)


def start_interactive(session: Session, seed: int):
    """Live facility session. A missing checkpoint is reported, not faked."""
    loaded = None
    if DEFAULT_CHECKPOINT.exists():
        loaded = planner()
    sim = InteractiveSim(
        planner=loaded, seed=seed, use_cbf=session.use_cbf, async_plan=True, pace=HALL_PACE
    )
    session.sim = sim

    def produce():
        return sim.frames()

    session.paced = False
    session.launch(produce)
    return loaded is not None


def start_hall(session: Session, seed: int, obstacles: list, goal: dict | None):
    """The drawn aisle, with the same model and safety filter as the facility."""
    loaded = None
    if DEFAULT_CHECKPOINT.exists():
        loaded = planner()
    sim = InteractiveSim(
        planner=loaded,
        seed=seed,
        layout="hall",
        async_plan=True,
        pace=HALL_PACE,
        use_cbf=True,
    )
    sim.use_cbf = True
    session.sim = sim
    target = goal or {}
    sim.try_goal(
        float(target.get("x", HALL_GOAL_X)),
        float(target.get("y", HALL_GOAL_Z)),
    )
    for item in obstacles:
        if not isinstance(item, dict):
            continue
        sim.place_obstacle(
            str(item.get("kind", "")),
            float(item.get("x", 0.0)),
            float(item.get("y", 0.0)),
            float(item.get("yaw", 0.0)),
        )

    def produce():
        return sim.frames()

    session.paced = False
    session.launch(produce)
    return loaded is not None


def start_replay(session: Session, policy_mode: str):
    matches = [r for r in list_replays() if r["policy_mode"] == policy_mode]
    if not matches:
        raise FileNotFoundError(f"no recorded run for {policy_mode!r}")
    path = FRAMES_DIR / f"{matches[0]['name']}.jsonl"

    def produce():
        return replay_frames(path)

    session.paced = True
    session.launch(produce)


@app.websocket("/ws")
async def socket_endpoint(socket: WebSocket) -> None:
    await socket.accept()
    session = Session()
    pumping: asyncio.Task | None = None

    await socket.send_json(
        {
            "type": "hello",
            "schema_version": SCHEMA_VERSION,
            "scenarios": list_scenarios(limit=12),
            "replays": list_replays(),
            "intents": [
                "start", "start_interactive", "start_hall", "pause", "resume", "step", "reset",
                "toggle_safety", "start_replay", "set_goal", "spawn_obstacle",
                "remove_obstacle", "clear_obstacles",
            ],
            "checkpoint_present": DEFAULT_CHECKPOINT.exists(),
            "checkpoint": DEFAULT_CHECKPOINT.name if DEFAULT_CHECKPOINT.exists() else None,
        }
    )

    async def restart(coro_factory) -> None:
        nonlocal pumping
        if pumping is not None:
            pumping.cancel()
        coro_factory()
        pumping = asyncio.create_task(pump(socket, session))

    try:
        while True:
            intent = await socket.receive_json()
            kind = intent.get("type")

            if kind == "start":
                name = intent.get("scenario") or list_scenarios(limit=1)[0]
                duration = float(intent.get("duration_s", 45.0))
                session.sim = None
                await restart(lambda: start_live(session, name, duration))

            elif kind == "start_interactive":
                session.seed = int(intent.get("seed", session.seed))
                ready = {"neural": False}

                def launch_interactive() -> None:
                    ready["neural"] = start_interactive(session, session.seed)

                await restart(launch_interactive)
                await socket.send_json(
                    {
                        "type": "ack",
                        "intent": kind,
                        "neural": ready["neural"],
                        "checkpoint": DEFAULT_CHECKPOINT.name if ready["neural"] else None,
                    }
                )

            elif kind == "start_hall":
                session.seed = int(intent.get("seed", session.seed))
                ready = {"neural": False}
                obstacles = intent.get("obstacles") or []
                goal = intent.get("goal") if isinstance(intent.get("goal"), dict) else None

                def launch_hall() -> None:
                    ready["neural"] = start_hall(session, session.seed, obstacles, goal)

                await restart(launch_hall)
                await socket.send_json(
                    {
                        "type": "ack",
                        "intent": kind,
                        "neural": ready["neural"],
                        "checkpoint": DEFAULT_CHECKPOINT.name if ready["neural"] else None,
                    }
                )

            elif kind == "set_goal":
                if session.sim is None:
                    await socket.send_json(
                        {"type": "ack", "intent": kind, "accepted": False, "handled": False}
                    )
                else:
                    accepted = session.sim.try_goal(
                        float(intent.get("x", 0.0)), float(intent.get("y", 0.0))
                    )
                    await socket.send_json(
                        {"type": "ack", "intent": kind, "accepted": accepted}
                    )

            elif kind == "spawn_obstacle":
                if session.sim is None:
                    await socket.send_json(
                        {"type": "ack", "intent": kind, "accepted": False, "handled": False}
                    )
                else:
                    accepted = session.sim.try_spawn(
                        str(intent.get("kind", "")),
                        float(intent.get("x", 0.0)),
                        float(intent.get("y", 0.0)),
                        float(intent.get("yaw", 0.0)),
                    )
                    await socket.send_json(
                        {"type": "ack", "intent": kind, "accepted": accepted}
                    )

            elif kind == "remove_obstacle":
                if session.sim is not None:
                    session.sim.submit(
                        {"type": "remove_obstacle", "id": str(intent.get("id", ""))}
                    )
                await socket.send_json({"type": "ack", "intent": kind})

            elif kind == "clear_obstacles":
                if session.sim is not None:
                    session.sim.submit({"type": "clear_obstacles"})
                await socket.send_json({"type": "ack", "intent": kind})

            elif kind == "start_replay":
                session.seed = int(intent.get("seed", session.seed))
                mode = intent.get("policy_mode", "safety_on")
                session.use_cbf = mode == "safety_on"
                await restart(lambda: start_replay(session, mode))

            elif kind == "pause":
                session.resume.clear()

            elif kind == "resume":
                # The counterpart to `pause`. Without it the only way back to
                # running is a relaunch, which would restart the run from zero.
                session.resume.set()

            elif kind == "step":
                session.resume.clear()
                session.steps.release()

            elif kind == "toggle_safety":
                session.use_cbf = bool(intent.get("enabled", True))
                if session.sim is not None:
                    session.sim.submit(
                        {"type": "toggle_safety", "enabled": session.use_cbf}
                    )
                await socket.send_json(
                    {"type": "ack", "intent": kind, "use_cbf": session.use_cbf}
                )

            elif kind == "reset":
                session.seed = int(intent.get("seed", 0))
                session.halt()
                await socket.send_json({"type": "ack", "intent": kind, "seed": session.seed})

            else:
                await socket.send_json({"type": "ack", "intent": kind, "handled": False})

    except WebSocketDisconnect:
        pass
    finally:
        if pumping is not None:
            pumping.cancel()
        session.halt()


__all__ = ["app", "validate_frame"]
