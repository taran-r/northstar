# ML workstream — session state

Handoff for picking up `backend/ml/`, `backend/api/` and the frontend seam in a fresh
session. Last updated 2026-10-18. Supersedes the 2026-10-17 version, which covered only
v1 training.

## Read this first: the environment trap

`.venv` shipped with **torch and nothing else** — no numpy, scipy, fastapi, uvicorn or
pytest. Everything measured on 2026-10-17 ran on the system `python3`, which happened to
have them, so documented commands worked for one interpreter and failed for the other with
`No module named uvicorn`. Fixed by installing `requirements.txt` plus pytest into `.venv`.

If anything reports a missing module, run `.venv/Scripts/python -m pip install -r
requirements.txt pytest` before debugging anything else. `requirements.txt` previously
declared none of fastapi, uvicorn, websockets, httpx2 or pytest; it now declares all of them.

## Does it work?

Flow-matching training converges and the **full pipeline runs end to end**: model proposes
→ ranking selects → CBF filter certifies → bicycle plant integrates → frames reach the
frontend schema. 136 tests pass.

The model alone is **not drivable**. The system is usable because the safety filter covers
it, which is the architecture's point.

### Closed loop, 12 held-out v2 scenarios

| | Safety off | Safety on |
|---|---|---|
| Collided | **9 / 12** | **0 / 12** |
| Worst min clearance | −0.80 m | +0.16 m |
| Goal reached | 4 / 12 | 4 / 12 |
| Path efficiency (arrivals only) | 0.92 | 0.98 |
| Interventions per run | 0 | 213 |
| Fallback replans | 275/412 = 67% | 210/419 = 50% |
| Plan latency | 45 ms | 82 ms |
| CBF solve | — | 280 ms |

The filter removes every collision without costing a single goal arrival. Two findings worth
keeping: fallback rate is *lower* with the filter on, because it keeps the rover in states
the planner can still solve from; and goal arrival at 4/12 is a planner limitation, not a
safety cost.

### Open-loop physical quality, 150 windows per set

| Metric | primary_dynamic | dropout | **static_control** | Expert |
|---|---|---|---|---|
| Heading error mean, v1 | 23.60° | 26.96° | **22.84°** | 1.75° |
| Heading error mean, v2 | 26.82° | 31.30° | **24.73°** | 2.65° |
| Speed mismatch, v1 | 0.726 | 0.861 | 0.729 | 0.010 |
| Collision rate, v1 | 0.592 | 0.608 | 0.612 | 0.273 |
| Collision rate, v2 | 0.487 | 0.493 | 0.468 | 0.000 |
| Feasible (all gates) | 0.000 | 0.000 | 0.000 | 0.767 / 1.000 |
| Latency per candidate | 8.0 ms | 8.5 ms | **3.8 ms** | — |

Heading error improved **42.9° → 22.8°** against the previous v1 baseline, a 47% cut, from
the frame fix plus 3.9× more data. `feasible` stays 0.000 because it ANDs over
`heading_consistent`, which demands *every* moving step within 0.6 rad.

### Best-of-8 availability — the number that made the demo viable

| | v1 test | v2 test |
|---|---|---|
| Windows with ≥1 collision-free candidate | 69.3% | 86.0% |
| Mean clean candidates per window | 3.11 / 8 | 4.25 / 8 |
| Best clean candidate goal progress | 10.93 m | 10.18 m |
| Expert goal progress | 10.31 m | 9.82 m |

A single candidate collides 47–61% of the time; best-of-8 is clean in 69–86% of windows.
The whole difference is the ranking step.

## Architecture

```
data/dataset_v1 + v2_additions
  -> CombinedTrainDataset            (train only; refuses val/test by design)
  -> backend.ml.dataset              (7-feature target, 16-feature condition, 8 dynamic slices)
  -> backend.ml.train                (flow matching + physical auxiliary losses)
  -> backend.ml.inference            plan() -> rank_candidates() -> best_candidate()
  -> Candidate.world_xz(ego)         the world XZ polyline boundary
  -> northstar.control               TrajectoryTracker -> filter_control -> BicycleModel
  -> backend.ml.bridge               StateFrame (schema 1) + validate_frame()
  -> backend.api.server              WebSocket /ws + REST /health /scenarios /replays /metrics
  -> frontend socketSource/useLiveRun -> RunScreen   (opt-in via ?live=1)
```

## Known-good commands

```bash
python -m pytest tests/ -q                                   # 136 passing, ~100s
python -m uvicorn backend.api.server:app --port 8000
cd frontend && npm run dev                                   # open /?live=1
python -m backend.ml.train --combined --no-include-oracle --use-dynamic \
    --heading-weight 2.0 --speed-weight 0.3 --unit-weight 0.3 --out backend/artifacts/<n>.pt
python -m backend.ml.evaluation backend/artifacts/v2_static_control.pt --windows 150 --steps 16
python -m backend.ml.bridge                                  # records frame logs
```

Use the **replay** path to demo (`start_replay` intent). The live path is real but streams
~2.8× slower than real time: 280 ms CBF solve against a 100 ms control interval on CPU.

## Dataset

Combined 1,663 scenarios / 22,464 windows. **Zero scenario overlap across all six splits**,
verified pairwise. Shipped manifests are clean as of `b32b87a7`; the earlier leaky
working-tree copies are gone.

| | v1 | v2_additions |
|---|---|---|
| Scenarios | 363 | 1,300 |
| Windows | 5,021 | 17,443 |
| Train windows | 4,203 | 13,722 |
| `oracle_dynamics=1` | 23.7% | 3.6% |
| `subgoal == future[-1]` (<5 cm) | 88.8% | 90.0% |
| Steps over the bicycle curvature limit | **21.6% test / 25.1% train** | **0.0%** |

Training uses 17,925 windows, **16,495 after excluding oracle**, over 1,250 scenarios.

**v1 contains kinematically impossible expert trajectories.** One turns 67° while travelling
0.112 m, where 3.9° is the bicycle maximum. A quarter of v1 training targets contain such a
step; v2 is clean. This is the leading suspect for remaining heading error and the cheapest
experiment left.

`subgoal` is a label leak in both corpora and `is_local` is never set, so it cannot be used
to filter. `subgoal` stays out of conditioning; two regression tests enforce that.

v2 does **not** fix the single-facility limitation. Its own metadata: "Every scenario uses
industrial_facility_v1. Held-out splits measure unseen obstacle configurations and routes
inside this building, not a new building."

## Model

`STEP_FEATURES = 7`: `x, y, sin(yaw), cos(yaw), speed, steering, accel` from `future_rel`.
The last two are the control targets the CBF corrects rather than re-derives; the old
5-feature target gave it nothing to correct.

`COND_FEATURES = 16`: goal (4) + ego speed/steering + **sin/cos of world ego yaw** (4) +
vehicle limits (8). That ego-yaw pair is the frame fix. Rasters are world-axis-aligned while
trajectories are rover-frame, and ego yaw was previously used to rotate the goal and then
discarded, so the model could not align the map to the path it predicted. Note `future_rel`'s
yaw is already ego-relative, so its sin/cos would be a constant `(0, 1)` — the fix needs the
*world* yaw from `ego0`.

1,057,232 params static / 1,156,096 with dynamic conditioning, 256 wide / 3 blocks.
`v2_static_control.pt` is the static one.

## Measurement traps already hit — six

1. **Single-seed comparisons are uninterpretable.** Seed 0 said dynamic conditioning hurt;
   seed 1 said it helped. Three seeds gave a definitive null. Minimum 3 seeds.
2. **Binary metrics saturate.** `heading_consistent` read 0.000 across baseline, aux_v3 and
   aux_v4 while mean heading error fell 42.3° → 27.0°. Always read the continuous companions.
3. **Per-sample vs batch-mean ramps.** The auxiliary ramp was `t.mean()`, one batch scalar,
   while the comment claimed per-sample fade-in. Fixing it to `t**2` per sample moved heading
   error 35.9° → 27.0°.
4. **Chord-vs-arc curvature.** The gate measured yaw change against the chord while yaw is
   gained along the arc, overestimating by `1 + θ²/24`. Only 0.12% at this spacing, but the
   expert drives at full steering lock, so the bias alone failed **61% of v2 expert
   trajectories** whose true curvature sits exactly on the limit. Now `k = 2·sin(θ/2)/chord`,
   exact for a circular arc. v2 expert feasibility went 0.393 → 1.000.
5. **Curvature at zero speed is undefined.** A hair of yaw over a near-zero step read up to
   17 per metre against a 0.613 limit, and the gate took the per-trajectory maximum. Masked
   to moving steps.
6. **Ranking must not reject on the model's raw kinematics.** The follower consumes only the
   position polyline through pure pursuit and the filter re-derives commands, so speed,
   steering and curvature channels never reach the plant. Rejecting on them discarded
   collision-free geometry and drove the fallback rate from 24% to 76%. Only collisions
   reject; the envelope sorts.

Flow loss and physical quality disagree: aux_v4 had *worse* loss and *better* physical
quality, and here `static_control` won on both. Never rank on loss alone.

## Training results

All three arms: combined corpus, oracle excluded, aux weights 2.0/0.3/0.3, seed 0, CPU,
~35 s/epoch, early stopping patience 25.

| Arm | Best val | Best epoch | Stopped | val v1 | val v2 | Wall |
|---|---|---|---|---|---|---|
| `v2_static_control` | **0.4502** | 101 | 126 | 0.4562 | 0.4489 | 522 s |
| `v2_primary_dynamic` | 0.4621 | 68 | 93 | 0.4654 | 0.4613 | 1186 s |
| `v2_dynamic_dropout` | 0.5040 | 67 | 92 | 0.4944 | 0.5063 | 772 s |

Best epoch moved from 35 (v1 only) to 68–101, so the extra data pushed overfitting out.
Train and val loss are **not comparable** in these runs: train is the composite including
auxiliary terms, val is pure flow loss.

**Dynamic conditioning is a second independent null.** It was never missing — it was
implemented and measured as a null on v1, and on v2 static matches or beats it on every
metric at half the latency. **Dropout 0.1 hurt everything.**

**Reducing capacity is measured as harmful.** The v1 sweep (3 widths × 3 seeds): 256/3 →
0.3717, 128/3 → 0.5619, 96/2 → 0.7421, gaps of 59× and 93× pooled std, and the small models
*underfit* (train 0.51/0.68 vs 0.29). `--hidden-dim` cannot shrink much anyway: the
condition encoder is a fixed floor that `--hidden-dim` never touches — 347,888 static,
446,752 with dynamic, `occupancy_encoder.project` alone being 204,928. To build a
genuinely small model you have to shrink `OCC_EMBED_DIM` / `OCC_STAGES`, not the blocks.

## Checkpoints

42 `.pt` files in `backend/artifacts/` (gitignored). Current best: **`v2_static_control.pt`**,
the default in `backend/api/server.py`. Each has a `_latest.pt` carrying resumable
optimiser/scheduler/RNG state written every epoch, and a `.yaml` sidecar with config, counts
and splits provenance.

Frame logs: `backend/artifacts/frames/v2_scn_00001_{safety_off,safety_on}.jsonl` plus
`summary.json`. Sample cache: `backend/artifacts/cache/samples_step7_cond16_dyn8_oracle0.pt`
(552 MB) — rebuilds in 57 s if deleted, and makes runs start in 2.7 s.

## Git state

`b80acf71` on `main`, **ahead 1 of origin, not pushed**. It carries the whole ML workstream
and the bridge.

Uncommitted: `CLAUDE.md`, `README.md`, `requirements.txt` modified; untracked
`backend/api/`, `tests/test_api.py`, `frontend/src/sim/socketSource.ts`,
`frontend/src/hooks/useLiveRun.ts`. **All of `frontend/` is untracked** — its owner has never
committed it. Nothing has been pushed; do not push without asking.

## Hard constraints

- `data/dataset_v1/` and `data/dataset_v2_additions/` are **read-only**. Never regenerate.
- Read-only (other owners): `northstar/control/` (CBF), `northstar/pipeline/`,
  `northstar/planner/`, `northstar/scenario/`, `northstar/obstacles/`, `northstar/facility/`.
- Owned: `backend/ml/`, `backend/api/`, and the two frontend files listed above.
- No ground-truth future obstacle motion in conditioning. No expert-derived subgoals.
- Fallbacks must be logged and labelled, never presented as the learned result.
- Do not push, force-push or merge to `main` without explicit permission.

## Next steps, in priority order

1. **Train on v2 only, or filter the infeasible v1 windows.** 25% of v1 training targets are
   kinematically impossible; v2 is clean. Cheapest high-value experiment left.
2. **Add a curvature-limit auxiliary loss.** Heading and speed consistency terms cut heading
   error 42.9° → 29.2° historically; curvature now passes only 0.166 and gates feasibility.
3. **Raise the fallback rate.** 50–67% in live rollout vs 14% on dataset windows. Two causes:
   distribution shift at poses never trained on, and goals up to 22 m against a 20 m
   conditioning window. A deployment should call Hybrid A* on fallback and label it — the
   bridge currently holds the previous reference and reports `source: "fallback"`.
4. Three seeds per arm to settle static vs dynamic properly. Current arms are single-seed.
5. **Speed up the CBF path or keep demoing from replay.** 280 ms solve vs 100 ms interval.
6. `src/overlays/networkGraph.ts` has no real source; the policy panel stays illustrative.
   The bridge exposes candidate paths and diagnostics, not layer activations.

## Still-open items from earlier sessions

- Mirror augmentation (`y → −y`, negate `sin(yaw)`, flip rasters) — never run.
- Diagnostic plots — unmet handoff deliverable.
- Oracle on/off comparison — oracle is now excluded by default, but the two have never been
  trained and compared head to head.
- `OccupancyGrid.crop_local` is a per-cell Python loop; it cost 245 ms per replan until the
  unused dynamic rasters were skipped for the static model. It is shared code, not ours, and
  would need optimising for a real-time live path.
