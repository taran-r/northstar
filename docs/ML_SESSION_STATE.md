# ML workstream — session state

Handoff for picking up `backend/ml/` work in a fresh session. Last updated 2026-10-17.

## Does the model work?

Flow-matching training converges. **Generated trajectories are not yet drivable** — zero
candidates pass the kinematic feasibility check. Flow loss improves while physical quality
stays poor, which is why the evaluation suite in `evaluation.py` exists.

Measured on 40 held-out windows / 26 scenarios, 8 candidates, 16 Heun steps:

| Metric | Baseline | Aux v4 | Expert |
|---|---|---|---|
| Val loss (best) | 0.3676 | 0.4367 | — |
| Heading error, mean | 42.9° | 29.2° | 1.85° |
| Heading error, max | 149.3° | 118.9° | 5.01° |
| Speed mismatch | 1.295 m/s | 0.906 | 0.011 |
| Collision rate | 0.656 | 0.603 | 0.250 |
| Feasible rate | 0.000 | 0.000 | 0.625 |
| Curvature within limit | 0.219 | 0.250 | 0.625 |
| Path length | 17.06 m | 14.12 m | 11.50 m |
| Goal progress | 9.415 m | 9.500 m | 9.996 m |
| Diversity | 1.671 m | 1.385 m | — |
| Latency / candidate | 8.3 ms | 6.4 ms | — |

Auxiliary physical losses are the lever that works: heading error -32%, speed mismatch -30%.
Note that flow loss got *worse* while physical quality got *better* — ranking on loss alone
would have rejected the better model.

Expert numbers are not 100% because the rasters are footprint-inflated and mark
outside-facility as occupied. Only model-vs-expert comparison on the identical checker is
meaningful; absolute rates are not.

## Known-good commands

```bash
python -m pytest tests/ -q                      # 75 passing
python -m backend.ml.train --help
python -m backend.ml.evaluation backend/artifacts/baseline_v2_static_canonical.pt \
    backend/artifacts/aux_v4_canonical.pt --windows 40
```

## Splits — read this before touching data loading

`data/dataset_v1/splits/*.json` **in the working tree are leaky** — 235 scenarios appear in
both train and test. They predate the audit fix.

The canonical manifests were extracted read-only from `origin/main` (commit `4c30a182`) into
`backend/ml/splits_v1_canonical.json`: **4203 / 420 / 398 windows, 290 / 36 / 37 scenarios,
zero overlap**. `load_canonical_splits()` uses that file and `assert_no_scenario_overlap`
hard-fails on the leaky ones.

Per the user's decision this was fixed locally only and **not** reported upstream to the data
owner. The working tree is behind `origin/main`; pull for `northstar/coords.py` and
`docs/TRAINING_HANDOFF.md`.

## Hard constraints

- `data/dataset_v1/` is **read-only**. Do not modify or regenerate it.
- Read-only: shared facility files, `northstar/pipeline/`, `northstar/planner/`,
  `northstar/scenario/`, `northstar/obstacles/`.
- Owned: `backend/ml/{model,dataset,train,inference,evaluation}.py` only.
- No ground-truth future obstacle motion in conditioning.
- No expert-derived subgoals. `subgoal == future[-1]` within 5cm in 4414/5021 windows
  (87.9%) — it is a label leak. Two regression tests fail loudly if a future dataset
  reintroduces it. `global_goal` is safe (0/800 match).
- Do not push, force-push, or merge to `main` without explicit permission. Nothing is
  committed yet by the user's choice.

## Next steps, in the user's priority order

### On hold: frame mismatch fix

Paused pending dataset expansion. This is the top technical recommendation.

`OccupancyGrid.crop_local` is a pure translation, so rasters are world-axis-aligned while
trajectories are rover-frame, and ego yaw never enters the 14-feature condition vector —
`dataset.py:154` uses it to rotate the goal, then discards it. The model cannot align the
map to the path it predicts.

- Option 1 (cheap): add `sin/cos(ego_yaw)`, `COND_FEATURES` 14 → 16. Widens `fuse.0`.
- Option 2 (correct): rotate the raster into the ego frame. No shape change, but resampling
  blurs thin walls — check occupied-cell count before/after.

### Proposed next: checkpoint plumbing (Sprint 7b)

Data-independent, and the prerequisite for both the architecture sweep and the frame fix.

**Live bug:** `inference.py:106` builds `FlowMatchingModel(use_dynamic=...)` without
`hidden_dim` / `num_blocks`, so they fall back to module defaults (256, 3). A smaller model
trains fine and then fails to load with a shape mismatch. Must be fixed before any
architecture experiment.

To build:

| Piece | Purpose |
|---|---|
| `--init-from <ckpt>` | load weights, fresh optimizer — fine-tuning |
| `--resume <ckpt>` | weights + optimizer/scheduler/RNG — continue a run |
| Partial-load helper | transfer shape-matching tensors, report the rest, don't crash |
| `--hidden-dim` / `--num-blocks` | wire existing constructor args to the CLI |
| Schema fields | persist sizes so `inference.py` rebuilds correctly |

`*_latest.pt` already carries `resume{epoch, optimiser_state, scheduler_state,
generator_state, torch_rng_state}` — the state is saved, there is just no flag to consume it.

### Then: smaller architecture sweep (Sprint 8b)

1M params against ~1,017 effective samples. Current 256/3 ≈ 1.02M; small 128/3 ≈ 320k;
tiny 96/2 ≈ 150k. Three seeds per arm.

### Also pending

- Mirror augmentation (`y → −y`, negate `sin(yaw)`, flip rasters). Blocked in practice by
  the frame question: the flip axis is ambiguous while raster and trajectory disagree.
- Diagnostic plots — unmet handoff deliverable, and what judges actually look at.
- `docs/ML_DATA_NEEDS.md` §9 still lists the dynamic-coverage figure as an open
  discrepancy. It is resolved: 45% / 12% after isolation reconciles with the handoff's
  41% / 31%.
- Oracle on/off comparison (`--no-include-oracle`). User chose "train both, compare";
  never run.

## Weight transfer

Enabling dynamic conditioning widens `fuse.0.weight` (adds `DYNAMIC_EMBED_DIM` to its input).

| Tensor group | Transfers? |
|---|---|
| `occupancy_encoder.*` | yes |
| `blocks.*` | yes |
| `input_projection`, `output_projection`, `output_norm` | yes |
| `condition_encoder.fuse.0` | **no** — re-initialize |
| `dynamic_encoder.*` | **no** — new, initialize fresh |

## Measurement methodology — three traps already hit

1. **Single-seed comparisons are uninterpretable.** Seed 0 said dynamic conditioning hurt by
   0.0067; seed 1 said it helped by 0.0069. Three seeds per arm gave static 0.3717
   (std 0.0028) vs dynamic 0.3738 (std 0.0053) — difference +0.0021 against pooled std
   0.0044, a definitive null. Minimum 3 seeds on this corpus.
2. **Binary metrics saturate and hide progress.** `heading_consistent` demands *every* moving
   step within tolerance, so it read 0.000 for baseline, aux_v3 and aux_v4 alike while mean
   heading error dropped 42.3° → 35.9° → 27.0°. Always read the continuous companions
   (`heading_error_deg_mean/max`, `speed_mismatch_ms`).
3. **Per-sample vs batch-mean ramps.** The auxiliary ramp was `t.mean()` — one batch-average
   scalar — while the comment claimed per-sample fade-in. Samples at t≈0.05, where the
   reconstructed trajectory is essentially noise, contributed full-strength gradients. Fixed
   to per-sample `t**2`; that fix alone moved heading error 35.9° → 27.0°.

## Dataset facts worth not re-deriving

- 5,021 windows ≈ **1,017 effective independent samples** (88% window overlap;
  `scn_00004_m0_w0000` and `w0004` share 28/32 steps bit-identically).
- One facility only. This is the single largest limitation.
- `dynamic_occ` is the **union** of building + static + movers. Covers 100% of
  facility/static cells; 76% of windows have all 8 slices bit-identical; the moving part is
  ~7% of occupied cells. `dynamic_only()` isolates it (0.42 → 0.00103 occupied fraction).
- **46% of expert paths leave the 20m conditioning window** — the conditioning often does
  not cover the horizon. Likely the next explanation to chase if the frame fix does not move
  heading error, and it is a dataset-geometry issue for the data workstream.

## Checkpoints

All preserved in `backend/artifacts/` (18 `.pt` files, ~141 MB). Best val loss in parens.

| Checkpoint | Notes |
|---|---|
| `baseline_v1_static_only.pt` (0.3500) | ad-hoc split, pre-canonical — not comparable |
| `baseline_v2_static_canonical.pt` (0.3676) | **the canonical control** |
| `dynamic_v2_canonical.pt` (0.3743) | dynamic conditioning |
| `seed1_static` (0.3739) / `seed1_dynamic` (0.3670) | variance study |
| `seed2_static` (0.3735) / `seed2_dynamic` (0.3800) | variance study |
| `aux_v3_canonical.pt` (0.4287) | buggy batch-mean ramp |
| `aux_v4_canonical.pt` (0.4367) | **best physical quality**, fixed ramp |

`*_latest.pt` variants carry resumable optimizer/scheduler/RNG state.
