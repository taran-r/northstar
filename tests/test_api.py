"""Contract tests for the frontend-facing API seam.

These assert the socket speaks the schema `frontend/src/types/sim.ts` declares
and that the run controls behave, so a change on either side fails here rather
than showing up as a blank panel in the UI.
"""

import json
import threading
import time
from pathlib import Path

import pytest

from backend.api.server import (
    Session,
    app,
    list_replays,
    list_scenarios,
    replay_frames,
    scenario_path,
)
from backend.ml.bridge import SCHEMA_VERSION, validate_frame

try:
    from starlette.testclient import TestClient
except Exception as error:
    pytest.skip(
        f"starlette test client unavailable: {error}", allow_module_level=True
    )

FRAMES_DIR = Path("backend/artifacts/frames")
SCENARIO = "v2_scn_00001"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as instance:
        yield instance


def test_health_reports_the_checkpoint_and_schema(client):
    payload = client.get("/health").json()
    assert payload["ok"] is True
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["scenarios"] > 0


def test_scenarios_are_listed_and_resolvable(client):
    names = client.get("/scenarios").json()["scenarios"]
    assert names
    assert scenario_path(names[0]) is not None


def test_unknown_scenario_resolves_to_none():
    assert scenario_path("definitely_not_a_scenario") is None


def test_metrics_never_invents_numbers(client):
    payload = client.get("/metrics").json()
    if payload["measured"]:
        assert payload["rows"] is not None
    else:
        assert payload["rows"] is None


def test_socket_announces_the_schema_on_connect(client):
    with client.websocket_connect("/ws") as socket:
        hello = socket.receive_json()
    assert hello["type"] == "hello"
    assert hello["schema_version"] == SCHEMA_VERSION
    assert "start" in hello["intents"]


def test_unknown_intent_is_acked_as_unhandled(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        socket.send_json({"type": "nonsense"})
        ack = socket.receive_json()
    assert ack["type"] == "ack"
    assert ack["handled"] is False


def test_toggle_safety_is_acknowledged(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        socket.send_json({"type": "toggle_safety", "enabled": False})
        ack = socket.receive_json()
    assert ack["use_cbf"] is False


@pytest.mark.skipif(
    not (FRAMES_DIR / f"{SCENARIO}_safety_off.jsonl").exists(),
    reason="no recorded frame log",
)
def test_replay_streams_schema_valid_frames(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        socket.send_json({"type": "start_replay", "seed": 0, "policy_mode": "safety_off"})
        frames = []
        while len(frames) < 10:
            message = socket.receive_json()
            if message.get("type") in ("end", "error"):
                break
            frames.append(message)

    assert len(frames) == 10
    for frame in frames:
        assert validate_frame(frame) == []
        assert frame["measured"] is True
        assert frame["version"] == SCHEMA_VERSION


@pytest.mark.skipif(
    not (FRAMES_DIR / f"{SCENARIO}_safety_off.jsonl").exists(),
    reason="no recorded frame log",
)
def test_recorded_logs_are_schema_valid_on_disk():
    path = FRAMES_DIR / f"{SCENARIO}_safety_off.jsonl"
    for index, frame in enumerate(replay_frames(path)):
        assert validate_frame(frame) == [], f"frame {index}"
        if index >= 25:
            break


def test_replay_listing_labels_the_policy_mode():
    for entry in list_replays():
        assert entry["policy_mode"] in ("safety_on", "safety_off")


def _drain(session, sink):
    while True:
        frame = session.frames.get()
        if frame is None:
            return
        sink.append(frame)


def test_pause_stops_production_and_each_step_releases_exactly_one():
    """The `step` intent must advance one frame, not zero and not a burst.

    Toggling a single event for this races the worker thread, which is why the
    session gates on a semaphore while paused.
    """
    session = Session()
    seen = []
    threading.Thread(target=_drain, args=(session, seen), daemon=True).start()

    def forever():
        index = 0
        while True:
            yield {"n": index}
            index += 1
            time.sleep(0.005)

    session.launch(forever)
    try:
        time.sleep(0.4)
        assert seen, "nothing produced while running"

        session.resume.clear()
        time.sleep(0.3)
        settled = len(seen)
        time.sleep(0.4)
        assert len(seen) == settled

        session.steps.release()
        time.sleep(0.25)
        assert len(seen) == settled + 1

        session.resume.set()
        time.sleep(0.25)
        assert len(seen) > settled + 1
    finally:
        session.halt()

    assert not session.running()


def test_halt_is_idempotent_and_clears_the_queue():
    session = Session()

    def once():
        yield {"n": 0}

    session.launch(once)
    time.sleep(0.2)
    session.halt()
    session.halt()
    assert not session.running()
    assert session.frames.empty()
