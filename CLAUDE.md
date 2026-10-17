# CLAUDE.md

Shared project guide for everyone (and every Claude) working in this repo. Keep it short. Personal or task-specific notes go in `CLAUDE.local.md` (gitignored). Full design brief: `Handoff.md` (add it to the repo root if it isn't there yet).

## What we're building

A 3D autonomous **ground-rover** simulation in an industrial environment. A small **conditional flow-matching** network (PyTorch, trained from scratch) generates timed trajectory candidates. A **CBF-style safety filter** (small QP plus emergency-stop fallback) corrects steering and acceleration online. The backend simulator is authoritative, and a Three.js frontend renders it.

Not a drone, not 2D pathfinding, not A* with an AI wrapper.

## Ground rules

- **Project name is undecided.** Don't invent or add a name or branding in code, UI, docs, or slides.
- Keep flow matching as the core ML technique. A classical planner (Hybrid A*) is the baseline and the logged fallback, never presented as the learned result.
- The CBF must be derived for the real dynamics (mind relative degree). A plain `distance < threshold -> brake` is only an emergency fallback, not "CBF". Call it "CBF-style" or "discrete-time CBF filter" and make no guarantee claims.
- Metrics come from measured runs only. No placeholder numbers.
- Backend owns physics, time, and collisions. The frontend renders snapshots and never invents outcomes.
- Don't assume CUDA. Default to CPU PyTorch (the team has AMD iGPUs).
- Don't add ROS, Isaac Sim, Unity, or a second physics engine.

## Conventions

- Backend coordinates: `(x, y, z)`, z up, meters, seconds, heading `psi` in the XY plane.
- Three.js is Y-up: `(x, y, z)_backend -> (x, z, -y)_frontend`. Use the single tested helper for all conversions, including yaw and paths.
- Model local frame: x forward, y left, normalized consistently.
- Vehicle model: kinematic bicycle, state `(x, y, psi, v)`, control `(a, delta)`, with bounds on speed, acceleration, and steering.
- One simulation clock plus a deterministic seed for every scenario and replay.
- Train/test splits are by **map layout/seed**, not by individual examples.
- Message schemas (state, intents) are in Handoff.md section 10. Change them only in agreement with the whole team, and version them.

## Layout (planned, create only what's needed)

`backend/` (FastAPI, `sim/`, `planning/`, `ml/`, `evaluation/`), `frontend/` (React + TypeScript + React Three Fiber), `tests/`, `scripts/`, `docs/`.

## Commands

```bash
pip install -e .
python -m pytest tests/test_dataset_loader.py -q
# Data owner only (do not overwrite data/dataset_v1 from training work):
# python scripts/generate_dataset.py --out data/dataset_v1 --min-windows 5000
```

Training handoff: `docs/TRAINING_HANDOFF.md`. Dataset coords are XZ-horizontal (see `northstar/coords.py` for backend/Three.js mapping).

## Working agreements

- Branch per task, small PRs, and merge to main often so integration stays early.
- Don't reformat or rewrite files you don't own. Ownership: ML/data, control/sim, frontend, integration (see Handoff.md section 12).
- Keep a runnable demo on main at all times.
- When reporting status, say what actually works versus what is planned.
