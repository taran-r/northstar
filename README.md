# Autonomous industrial rover — data generation

Synthetic expert-trajectory generation for a ground rover in a shared industrial facility.

This repo is a multi-part project (see `CLAUDE.md`). **What is implemented today** is the ML/data side: facility specification, vehicle model, expert planner, scenario generation, and a PyTorch-ready dataset. Flow-matching training, CBF control, FastAPI sim, and Three.js rendering still live in the planned `backend/` / `frontend/` layout as placeholders.

> Project display name is undecided — do not add branding. The Python package folder is `northstar/` only because that is the current repo directory name.

## Quick start (data pipeline)

```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows
pip install -e .

python scripts/smoke_test.py
python scripts/generate_dataset.py --out data/dataset_v1 --n-scenarios 360 --min-windows 5000
python scripts/validate_dataset.py --root data/dataset_v1
```

A validated corpus is already at `data/dataset_v1` (**5021** windows).

## Repository layout

```
CLAUDE.md                 Shared team guide (conventions, ownership, goals)
facilities/               Authoritative base facility JSON (shared with future Three.js)
data/dataset_v1/          Generated training corpus
docs/DATA_PIPELINE.md     Planner choices, tensors, teammate handoff
northstar/                Python package: facility, vehicle, planner, dataset, pipeline
scripts/                  generate / validate / smoke_test entrypoints
backend/                  Planned FastAPI sim / planning / ml / evaluation (placeholders)
frontend/                 Planned React + R3F Three.js app (placeholders)
tests/                    Planned tests (placeholder)
pyproject.toml            Package metadata
requirements.txt          Pip dependencies
```

### `northstar/` package (implemented)

| Module | Role |
|--------|------|
| `facility/` | Versioned facility schema, JSON I/O, occupancy grids, base builder |
| `vehicle/` | Configurable kinematic bicycle model |
| `obstacles/` | Procedural static obstacles + time-indexed dynamic agents |
| `planner/` | Hybrid A*, timed follow, smoother, time-aware search |
| `scenario/` | Procedural scenarios inside the persistent facility |
| `trajectory/` | Full routes + fixed-horizon (H=32, dt=0.2s) windows |
| `dataset/` | PyTorch `Dataset` + collate helper |
| `pipeline/` | Generate / validate orchestration |
| `geometry.py`, `collision.py` | Shared geometry and collision checks |

### `data/dataset_v1/`

| Path | Contents |
|------|----------|
| `facility/` | Facility JSON used for this build |
| `scenarios/` | Obstacles, start/goal, oracle flag per scenario |
| `trajectories/` | Full expert routes |
| `samples/` | NN windows (`*.npz` + `*.json` sidecars) |
| `splits/` | `train.json` / `val.json` / `test.json` |
| `dataset_meta.json` | Schema, vehicle params, stats |
| `validation_report.json` | Last validation summary |

## Coordinates (important)

Team convention in `CLAUDE.md`: backend `(x, y, z)` with **z up**, heading in the **XY** plane.

This data pipeline currently uses the generator brief’s ground plane as **X/Z** with **Y** elevation (Three.js-like). When wiring into `backend/sim`, map `(x, z, yaw)_data → (x, y, psi)_backend` (and keep a single tested helper for Three.js Y-up conversion as described in `CLAUDE.md`).

## Expert planner (brief)

1. Hybrid A* (bicycle primitives) on static geometry  
2. B-spline smooth + curvature-aware speed profile  
3. Timed following with brake/wait vs **predicted** dynamics  
4. Validate against **ground-truth** dynamic trajectories  

Oracle scenarios are flagged; NN `dynamic_occ` still uses constant-velocity predictions. Full detail: [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md).

## PyTorch loading

```python
from torch.utils.data import DataLoader
from northstar.dataset import NorthstarTrajectoryDataset, collate_trajectory_batch

ds = NorthstarTrajectoryDataset("data/dataset_v1", split="train")
loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_trajectory_batch)
```

Default to **CPU** PyTorch (team AMD iGPUs). See `CLAUDE.md`.

## Out of scope here

- Flow-matching architecture / training loop (`backend/ml/`)
- CBF-style safety filter (`backend/planning/` / control)
- Authoritative realtime sim API (`backend/sim/`, `backend/api/`)
- Three.js frontend (`frontend/`)
