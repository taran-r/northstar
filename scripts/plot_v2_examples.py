#!/usr/bin/env python
"""Save a few trajectory plots from dataset_v2_additions. Read-only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    root = Path("data/dataset_v2_additions")
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    chosen = []
    for path in sorted((root / "trajectories").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        realized = data.get("metadata", {}).get("realized", {})
        if realized.get("detoured") or realized.get("yielded"):
            chosen.append((path, data, realized))
        if len(chosen) >= 4:
            break
    if not chosen:
        print("No yielded or detour trajectories to plot yet.")
        return
    for path, data, realized in chosen:
        xs = [s["x"] for s in data["states"]]
        zs = [s["z"] for s in data["states"]]
        speeds = [s["speed"] for s in data["states"]]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        axes[0].plot(xs, zs, color="C0")
        axes[0].scatter([xs[0]], [zs[0]], c="green", s=20, label="start")
        axes[0].scatter([xs[-1]], [zs[-1]], c="red", s=20, label="end")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("z (m)")
        axes[0].legend(fontsize=8)
        axes[1].plot([s["t"] for s in data["states"]], speeds, color="C1")
        axes[1].set_xlabel("t (s)")
        axes[1].set_ylabel("speed (m/s)")
        kind = "detour" if realized.get("detoured") else "yield"
        fig.suptitle(f"{data['trajectory_id']} ({kind})")
        fig.tight_layout()
        out = fig_dir / f"{data['trajectory_id']}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
