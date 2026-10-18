"""Compare the safety filter against the same tracker with the filter disabled.

Writes measured metrics and diagnostic plots. Nothing here reads or writes
``data/dataset_v1`` or ``data/dataset_v2_additions``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from northstar.control.scenarios import all_scenarios
from northstar.control.simulate import SimResult, simulate_scenario
from northstar.facility.occupancy import sample_dynamic_pose
from northstar.geometry import OrientedBox
from northstar.vehicle.bicycle import VehicleParams


def _run_pair(scenario, params: VehicleParams) -> tuple[SimResult, SimResult]:
    kwargs = dict(
        name=scenario.name,
        start=scenario.start,
        goal_xy=scenario.goal_xy,
        tracker=scenario.tracker,
        world=scenario.world,
        dt=scenario.dt,
        duration=scenario.duration,
        params=params,
    )
    blind = simulate_scenario(**kwargs, use_cbf=False)
    guarded = simulate_scenario(**kwargs, use_cbf=True)
    return blind, guarded


def _box_polygon(box: OrientedBox):
    corners = box.corners()
    return Polygon(corners, closed=True, fill=False, lw=1.2, edgecolor="#444444")


def _draw_static(ax, world) -> None:
    for box in world.boxes:
        shape = OrientedBox(box.x, box.z, box.yaw, box.length, box.width)
        ax.add_patch(_box_polygon(shape))
        ax.text(box.x, box.z, box.id, fontsize=7, ha="center", va="center", color="#333333")


def _draw_dynamic_samples(ax, world, times: list[float]) -> None:
    for agent in world.dynamic_obstacles:
        for t in times:
            pose = sample_dynamic_pose(agent, t, use_prediction=False)
            if pose is None:
                continue
            x, z, yaw = pose
            shape = OrientedBox(x, z, yaw, agent.length, agent.width)
            patch = _box_polygon(shape)
            patch.set_linestyle("--")
            patch.set_edgecolor("#b35c00")
            ax.add_patch(patch)


def _plot_scenario(folder: Path, scenario, blind: SimResult, guarded: SimResult) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    ax = axes[0]
    _draw_static(ax, scenario.world)
    if scenario.world.dynamic_obstacles:
        t_end = max(guarded.steps[-1].t if guarded.steps else 0.0, 0.1)
        _draw_dynamic_samples(ax, scenario.world, [0.0, 0.5 * t_end, t_end])
    if blind.steps:
        ax.plot([s.x for s in blind.steps], [s.z for s in blind.steps], color="#c0392b", lw=1.5, label="filter off")
    if guarded.steps:
        ax.plot(
            [s.x for s in guarded.steps],
            [s.z for s in guarded.steps],
            color="#1f6aa5",
            lw=1.5,
            label="filter on",
        )
    ax.scatter([scenario.start.x], [scenario.start.z], c="black", s=18, zorder=3)
    ax.scatter([scenario.goal_xy[0]], [scenario.goal_xy[1]], c="#1e8449", s=22, zorder=3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("path")
    ax.legend(fontsize=8, loc="best")

    def series(result: SimResult, attr: str) -> tuple[list[float], list[float]]:
        return [s.t for s in result.steps], [getattr(s, attr) for s in result.steps]

    ax = axes[1]
    for result, color, label in ((blind, "#c0392b", "off"), (guarded, "#1f6aa5", "on")):
        ts, cs = series(result, "clearance_m")
        ax.plot(ts, cs, color=color, lw=1.3, label=label)
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("body clearance (m)")
    ax.set_title("clearance")
    ax.legend(fontsize=8)

    ax = axes[2]
    if guarded.steps:
        ts = [s.t for s in guarded.steps]
        ax.plot(ts, [s.nominal_accel for s in guarded.steps], color="#888888", lw=1.0, label="nominal a")
        ax.plot(ts, [s.applied_accel for s in guarded.steps], color="#1f6aa5", lw=1.3, label="safe a")
        ax.plot(ts, [s.applied_steer for s in guarded.steps], color="#1e8449", lw=1.0, label="safe steer")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("command")
    ax.set_title("filter on")
    ax.legend(fontsize=7)
    fig.suptitle(scenario.name.replace("_", " "))
    fig.tight_layout()
    fig.savefig(folder / f"{scenario.name}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    params = VehicleParams()
    out = Path("docs/cbf_eval")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    solve_times = []
    for scenario in all_scenarios(params):
        blind, guarded = _run_pair(scenario, params)
        rows.append(
            {
                "name": scenario.name,
                "description": scenario.description,
                "dt_s": scenario.dt,
                "duration_s": scenario.duration,
                "without_cbf": blind.metrics(),
                "with_cbf": guarded.metrics(),
            }
        )
        solve_times.append(guarded.mean_solve_time_s)
        _plot_scenario(out, scenario, blind, guarded)
        print(
            f"{scenario.name}: collide off/on {blind.collided}/{guarded.collided} "
            f"clearance {blind.min_clearance_m:.3f}/{guarded.min_clearance_m:.3f} "
            f"interventions {guarded.intervention_count} "
            f"infeasible {guarded.infeasible_count} "
            f"replan {guarded.replan_count} "
            f"progress {guarded.goal_progress:.2f} "
            f"solve_ms {1000.0 * guarded.mean_solve_time_s:.2f}"
        )
    payload = {
        "vehicle": params.to_dict(),
        "note": (
            "Collision clearance is the physical body against ground-truth obstacle "
            "poses. The filter sees constant-velocity forecasts from the current sample."
        ),
        "mean_solve_time_s_across_scenarios": float(sum(solve_times) / max(len(solve_times), 1)),
        "scenarios": rows,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
