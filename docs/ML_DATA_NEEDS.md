# ML data needs — measured from `data/dataset_v1`

**Audience:** data-generation owner, integration owner
**Author:** ML / training workstream
**Date:** 2026-10-17
**Status:** request for discussion, not a blocker — v1 trains end to end today

## Summary

The v1 corpus is enough to train a working conditional flow-matching model, and it
does: a checkpoint exists and generates candidates in 3.5 ms each. But the model
overfits after ~49 epochs, and the cause is the amount of *independent* data rather
than anything in the architecture.

The headline number: **5,021 windows is approximately 1,017 independent samples.**

Everything below is measured from the committed corpus, not estimated.

## 1. Window count overstates the data by ~5x

Windows are cut from each expert trajectory every 4 steps with a 32-step horizon.
Consecutive windows therefore share 28 of 32 waypoints.

| Measurement | Value |
| --- | --- |
| Windows on disk | 5,021 |
| Overlap between consecutive windows | **88%** (28 of 32 steps identical) |
| Distinct expert trajectories | 407 |
| Distinct scenarios | 363 |
| Mean windows per trajectory | 12.3 |
| Mean trajectory length | 79.6 states (median 74, min 26, max 240) |
| **Non-overlapping windows available** | **~1,017** |

Verify with:

```bash
python - <<'EOF'
import json, numpy as np
from pathlib import Path
d = Path('data/dataset_v1')
lens = np.array([len(json.load(open(p))['states']) for p in (d/'trajectories').glob('*.json')])
print('windows on disk:', len(list((d/'samples').glob('*.npz'))))
print('non-overlapping (len/32):', int((lens/32).clip(min=1).sum()))
EOF
```

## 2. The model is oversized relative to that

| | |
| --- | --- |
| Model parameters | 1,025,168 |
| Effective independent samples | ~1,017 |
| Ratio | ~1,000 parameters per sample |

## 3. Overfitting is measurable

Training run: 300 epochs, batch 64, AdamW lr 1e-3, weight decay 1e-4, CPU, 355 s.
Splits are scenario-level (291 / 36 / 36 scenarios; 3,969 / 437 / 615 windows) with
zero scenario overlap between splits.

| Epoch | Train loss | Val loss |
| --- | --- | --- |
| 1 | 1.389 | 1.216 |
| 10 | 0.471 | 0.499 |
| 25 | 0.376 | 0.397 |
| **49** | — | **0.350 (best)** |
| 300 | **0.086** | **0.738** |

Train loss falls to 0.086 while validation loss more than doubles from its best. The
model stops generalizing at roughly epoch 49 and memorizes after that.

## 4. The largest single limitation: one facility

All 363 scenarios are generated inside `industrial_facility_v1`.

This means the honest claim we can currently make is *"generalizes to unseen obstacle
placements and unseen start/goal pairs within a familiar building"* — not
*"generalizes to unseen maps."* That distinction is likely to be probed, and we cannot
currently answer it with evidence.

**Two or three additional facility layouts would be worth more than several thousand
extra scenarios in the existing one.**

## 5. Behavior coverage is thin per class

The generator produced ~30 trajectories per behavior tag across 12 tags.

| Behavior | Trajectories | Windows |
| --- | --- | --- |
| straight | 31 | 427 |
| sharp_turn | 31 | 304 |
| long_curve | 31 | 398 |
| tight_corridor | 30 | 345 |
| intersection | 30 | 480 |
| static_avoid | 30 | 650 |
| brake_for_obstacle | 30 | 301 |
| wait_traffic | 30 | 411 |
| yield_intersection | 30 | 359 |
| accelerate_clear | 30 | 305 |
| detour_blockage | 30 | 737 |
| follow_vehicle | 30 | 304 |

After a scenario-level split, a held-out set of 36 scenarios contains roughly 3
scenarios per behavior. Per-behavior evaluation on v1 will have very wide error bars,
so treat any per-behavior metric from v1 as indicative only.

## 6. Generation cost is understood

From `dataset_meta.json`: 409 successful trajectories against 2,221 failures, in
2,313 s. Roughly a 15% success rate, ~38 minutes per batch of this size.

So a request for 3x the data implies on the order of 7,000 additional planning
attempts. This request is written with that cost in mind — hence the emphasis on
*more facilities* over *more scenarios*, since facility diversity buys more
generalization per unit of compute.

## 7. What the ML side is doing about it (no data needed)

These are in progress and do not depend on anyone else:

| Change | Expected effect |
| --- | --- |
| Mirror augmentation (reflect y and occupancy) | Doubles effective data; valid because the rover is left-right symmetric |
| Early stopping on validation loss | Stops the run near epoch 49 instead of 300 |
| Reduce model capacity | Better matched to ~1k independent samples |
| Scenario-level splits | Already implemented; see note in `backend/ml/dataset.py` |

Conditioning currently uses `global_goal`, `ego0` state, vehicle parameters, and the
`facility_occ` + `static_occ` rasters. The `subgoal` field is deliberately excluded:
in 4,414 of 5,021 windows it equals the expert trajectory's final waypoint to within
5 cm (median distance 0.000 m), which would hand the model its own target. This
matches the constraint already documented in the training handoff.

## 8. Requests, in priority order

1. **2–3 additional facility layouts.** Highest value per unit of generation time.
   Even a modest number of scenarios in each would let us report a real unseen-map
   result.
2. **More trajectories in the thin behaviors** — `brake_for_obstacle`,
   `follow_vehicle`, `sharp_turn`, `accelerate_clear` are at ~30 each.
3. **A larger `window_stride`, or more trajectories rather than more windows.**
   Stride 4 inflates the sample count without adding information. Stride 16 or 32
   would give nearly independent windows; we can also subsample on the ML side if
   that is easier.
4. **Longer trajectories where feasible.** Mean length is 79.6 states (~15.9 s), and
   the median is 74, so a typical plan yields only 2 non-overlapping windows. Longer
   runs yield more independent windows per successful plan.

## 9. One discrepancy to reconcile

The training handoff reports dynamic obstacles present in ~41% of windows and within
~8 m of the rover in ~31%. A crude check on 500 randomly sampled windows found
non-zero `dynamic_occ` mass in 500/500 and within 8 m in 500/500.

We are clearly measuring different quantities — plausibly a threshold on occupancy
mass, or ground-truth versus predicted occupancy. Worth settling before either figure
is quoted externally, since dynamic-interaction coverage affects what we can claim
about moving-obstacle behavior.

## Reproducing the numbers here

```bash
python -m pytest tests/ -q                       # 49 tests, includes split and leak checks
python -m backend.ml.train --epochs 300          # the run tabulated in section 3
python -m backend.ml.inference backend/artifacts/flow_matching_v1_best.pt
```

Checkpoint and config: `backend/artifacts/flow_matching_v1_best.pt` and
`flow_matching_v1_best.yaml`.
