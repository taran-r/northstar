# Training handoff — conditional flow matching

This document is for the teammate who will train the conditional flow-matching model.

## Immutable initial corpus

**`data/dataset_v1` is the immutable initial dataset.**

- Do **not** overwrite, delete, or regenerate into `data/dataset_v1`.
- The data owner may expand the corpus later (new versioned dirs or additive updates).
- For local experiments that regenerate data, write to a **different** path (e.g. `data/dataset_exp/`).
- Do **not** modify `northstar/pipeline/`, `northstar/planner/`, `northstar/scenario/`, or `northstar/obstacles/` without coordination.

Your ownership: model architecture, training loop, metrics, checkpoints — preferably under `backend/ml/` (currently a placeholder).

## Setup (fresh clone)

```bash
git clone <repo-url>
cd northstar
python -m venv .venv
# Windows
.\.venv\Scripts\activate
pip install -e .
pip install pytest   # optional, for tests
```

Verify:

```bash
python scripts/smoke_test.py   # planner smoke (optional)
python -m pytest tests/test_dataset_loader.py -q
```

## Load the dataset

```python
from torch.utils.data import DataLoader
from northstar.dataset import NorthstarTrajectoryDataset, collate_trajectory_batch

train_ds = NorthstarTrajectoryDataset("data/dataset_v1", split="train")
val_ds = NorthstarTrajectoryDataset("data/dataset_v1", split="val")
test_ds = NorthstarTrajectoryDataset("data/dataset_v1", split="test")

train_loader = DataLoader(
    train_ds, batch_size=16, shuffle=True, collate_fn=collate_trajectory_batch
)
batch = next(iter(train_loader))
```

### Tensor contract (per batch)

| Key | Shape | Dtype | Notes |
|-----|-------|-------|--------|
| `future` | `[B, 32, 7]` | float32 | World frame: `x,z,yaw,speed,steering,accel,t_rel` |
| `future_rel` | `[B, 32, 7]` | float32 | **Ego frame** — recommended flow-matching target |
| `ego0` | `[B, 5]` | float32 | `x,z,yaw,speed,steering` |
| `subgoal` | `[B, 3]` | float32 | `x,z,yaw` (see leakage note below) |
| `global_goal` | `[B, 3]` | float32 | Scenario destination |
| `facility_occ` | `[B, 1, 80, 80]` | float32 | Permanent structure, ego-centered |
| `static_occ` | `[B, 1, 80, 80]` | float32 | Facility + temporary static obstacles |
| `dynamic_occ` | `[B, 8, 80, 80]` | float32 | **Predicted** dynamic occupancy (not GT) |
| `oracle_dynamics` | `[B]` | float32 | 1 if expert used GT futures |
| `mode_id` | `[B]` | int64 | Multi-modal expert index |
| `dt` | `[B]` | float32 | 0.2 s |

Horizon: **32** steps × **0.2 s** ≈ **6.4 s**. Map: **20 m** @ **0.25 m** → 80×80.

Splits are by **scenario** (no scenario appears in more than one of train/val/test).

Current sizes (after scenario split fix): see `data/dataset_v1/dataset_meta.json` → `splits` / `split_policy`.

## Coordinates

Dataset horizontal plane is **X/Z**, elevation **Y**.  
Backend (`CLAUDE.md`) is **X/Y** horizontal, **Z** up.

Use `northstar.coords`:

```python
from northstar.coords import dataset_xz_to_backend_xy, backend_to_threejs
```

Do not rewrite on-disk samples; convert at integration boundaries.

## Conditioning guidance for flow matching

**Safe inputs (available at inference):**
- `ego0`, `facility_occ`, `static_occ`, `dynamic_occ` (predictions), `global_goal`
- Optionally recent ego history if you add it later from live state

**Use with care:**
- `subgoal`: when the global goal is beyond the horizon, the dataset sets `subgoal` to the **expert window endpoint** (`is_local=1`). That pose is **not** automatically available at inference unless your stack provides a planner/heuristic subgoal. Prefer conditioning primarily on `global_goal` + maps, or train with a subgoal policy you can reproduce at test time.
- `oracle_dynamics`: for analysis/ablations only; do not assume GT dynamic futures in the main training objective.

**Targets:**
- Prefer `future_rel[:, :, :3]` (and optionally speed/accel channels).
- Default to **CPU** PyTorch (team AMD iGPUs).

## Fine-tuning on an expanded dataset

1. Owner publishes a new directory (e.g. `data/dataset_v2`) or documents additive files — **not** a silent overwrite of `dataset_v1`.
2. Point `NorthstarTrajectoryDataset(root=...)` at the new root (same schema).
3. Fine-tune from the `dataset_v1` checkpoint with a lower LR; keep the same feature layout.
4. Re-run `pytest tests/test_dataset_loader.py` against the new root (update `ROOT` in tests or parametrize).

## What `scripts/validate_dataset.py` already checks

Heuristic QA on NPZ files: finite futures, speed bounds, non-frozen windows, step continuity, map ranks. It does **not** guarantee scenario-level split isolation, inference-time subgoal availability, or that dynamic obstacles interact with the ego path.

## Minimal training sketch (you implement)

```python
# Pseudocode only — architecture is your responsibility
for batch in train_loader:
    cond = {
        "ego0": batch["ego0"],
        "global_goal": batch["global_goal"],
        "facility_occ": batch["facility_occ"],
        "static_occ": batch["static_occ"],
        "dynamic_occ": batch["dynamic_occ"],
    }
    target = batch["future_rel"]  # [B, 32, 7]
    loss = flow_matching_loss(model, target, cond, t=sample_t(batch_size))
    loss.backward(); optim.step()
```
