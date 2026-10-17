#!/usr/bin/env python
"""Quick smoke test: one scenario through the expert pipeline."""

from __future__ import annotations

from northstar.facility.build_base import build_industrial_facility_v1, write_default_facility
from northstar.pipeline.expert import plan_expert_trajectory
from northstar.scenario.generator import ScenarioGenerator
from northstar.vehicle.bicycle import VehicleParams


def main() -> None:
    write_default_facility()
    facility = build_industrial_facility_v1()
    gen = ScenarioGenerator(facility, VehicleParams())
    for seed in range(50):
        scn = gen.generate(seed=1000 + seed, force_behavior="intersection")
        if scn is None:
            continue
        rec = plan_expert_trajectory(facility, scn, dt=0.2)
        if rec is not None:
            print(
                f"OK seed={seed} id={scn.scenario_id} "
                f"states={len(rec.states)} dur={rec.metadata['duration']:.2f}s "
                f"planner={rec.metadata['planner']}"
            )
            return
    raise SystemExit("Smoke test failed: no successful plan in 50 attempts")


if __name__ == "__main__":
    main()
