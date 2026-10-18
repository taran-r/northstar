# Autonomous Ground-Rover Project — Claude Training-Agent Handoff

> **Your assignment:** Implement, train, evaluate, and checkpoint a **conditional flow-matching trajectory generator in PyTorch** using the existing, committed `data/dataset_v1` corpus. You own the ML workstream. Another teammate owns procedural data generation and is expanding the corpus concurrently.
>
> **Act on the repository.** Inspect the actual implementation, write working code, run it, train a real model, and report measured outcomes. Do not stop at architectural suggestions or pseudocode. Don't claim a feature or benchmark works unless you've run it.

## 1. Project and objective

We are building an autonomous four-wheel ground rover that moves through a **single persistent 3D industrial facility**. Temporary static obstacles (e.g., pallets, barriers, parked equipment) and moving obstacles (e.g., forklifts, AGVs, carts, workers) are procedurally placed within that facility. The frontend will ultimately use Three.js; a separate safety workstream will implement a Control Barrier Function (CBF) controller.

Your contribution is a **learned generative motion planner**: given the rover's state, goal, and an observation/prediction of static and moving obstacles, generate diverse candidate *time-indexed* future driving trajectories. The training examples came from a classical expert-planning pipeline; the scientific question is whether learned generation can produce useful, smooth, feasible candidates on new obstacle configurations. **Do not assume it will outperform Hybrid A\*.** Measure performance and preserve classical baselines.

This is **conditional flow matching**, not PPO, not an LLM wrapper, and not ordinary supervised next-state regression. It should learn a vector field that transforms noise into samples from a distribution of expert-like trajectories.

## 2. Start by reading the live repo

Read these files before changing anything:

- `CLAUDE.md`, `README.md`, `docs/DATA_PIPELINE.md`, and especially `docs/TRAINING_HANDOFF.md` (added in the latest readiness audit).
- `northstar/dataset/pytorch.py` and the trajectory/vehicle/coordinate modules it relies on.
- `northstar/coords.py` and related tests.
- `data/dataset_v1/dataset_meta.json`, `validation_report.json`, and the actual split manifests.
- Representative `data/dataset_v1/scenarios/*.json`, `trajectories/*.json`, and `.npz` samples.

**The audit has superseded the older repository handoff's split counts.** The current reported corpus is:

| Item | Count / shape |
| --- | --- |
| Successful scenarios | 363 |
| Full expert trajectories | 409 |
| Total windows | 5,021 |
| Train / validation / test windows | **4,203 / 420 / 398** |
| Train / validation / test scenarios | **290 / 36 / 37** |
| Future horizon | 32 steps at 0.2 s intervals (~6.4 s) |
| `future`, `future_rel` | `[B,32,7]` float32 |
| `facility_occ`, `static_occ` | `[B,1,80,80]` |
| `dynamic_occ` | `[B,8,80,80]` — *predicted*, not ground-truth future occupancy |
| Other fields | `ego0`, `subgoal`, `global_goal`, `behavior_tag`, `mode_id`, `oracle_dynamics`, `dt` |

The audit reported that the revised split is **grouped by parent scenario with zero cross-split scenario overlap**, the loader can load all samples, and a small pytest suite passed. Independently verify this before training. `validation_report.json` may not verify full forward vehicle execution or behavior diversity; **zero “bad samples” does not mean all trajectories are collision-free under high-fidelity simulation**.

Use the repo's actual APIs and documented shapes over this summary if they differ; report discrepancies instead of guessing.

## 3. Two critical data-leakage constraints

### A. Never condition the production model on expert-derived local subgoals

The audit found approximately **4,414 / 5,021** windows have `subgoal` values derived from the *expert trajectory's future window endpoint*. That is not necessarily available when the real rover begins planning, so using it uncritically would leak information about the desired answer and produce misleading validation performance.

**For the first model, condition on `global_goal`, `ego0`, and occupancy observations/predictions—not the expert-derived `subgoal`.** Verify goal coordinate semantics and transform global goals into a consistent ego-relative frame if appropriate.

If you later design an inference-time available local-subgoal generator, it must compute the subgoal independently of the expert target. Train and evaluate it as a separate, explicit configuration, not as a silent shortcut.

### B. No oracle future obstacle states in inference features

The dynamic occupancy maps are reported to use constant-velocity forecasts, which is acceptable as a first approximation. Some expert labels carry an `oracle_dynamics` flag because the *expert* could use ground-truth obstacle futures in planning. **Do not pass the flag or actual future obstacles as privileged inputs to the neural network.** Inspect how these labels are produced and analyze performance separately for oracle-flagged and non-oracle samples. Predicted future obstacle occupancy is not a safety guarantee.

Also note that the audit found moving-obstacle predictions somewhere in the local map in ~41% of windows, but within ~8 meters of the rover in only ~31%. Expect limited coverage of meaningful dynamic interaction in v1; another teammate is generating richer examples.

## 4. Coordinate conventions and model target

The stored dataset uses **X/Z horizontal, Y up**. The backend convention in `CLAUDE.md` is **X/Y horizontal, Z up**. The audit added `northstar/coords.py` with tested conversions; use them rather than inventing conflicting transformations. Confirm `future_rel` is genuinely ego-relative and inspect precisely how yaw and relative time are encoded. The loader now wraps some yaw values; the underlying `.npz` files were not rewritten.

Don't blindly generate all seven values in each sample independently:

- Time is deterministic; it must **not** be a generative target.
- Angles wrap around; choose an explicit periodic or unwrapped yaw representation.
- Position, speed, steering, and acceleration must be dynamically consistent.

Implement a principled first trajectory target (for example, ego-relative 2D positions + heading representation + speed, deriving timestamps and using a vehicle-model feasibility check). It is acceptable to begin with a simpler 2D-position sequence **as a baseline**, but the final interface must expose enough information for a real tracking controller and collision/kinematic evaluation. Document trade-offs and conversions back to physical trajectories.

Normalize inputs and targets using **training-split statistics only**. Save all normalization parameters with every checkpoint. Verify round-trip decoding and physical units.

## 5. Implement the conditional flow-matching model

Research starting points:

- Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747
- FlowMP / learned trajectory generation: https://github.com/mkhangg/flow_mp

Use PyTorch; train your own small model rather than relying on external model inference APIs. A reasonable first architecture is:

1. Lightweight CNN for `facility_occ` and `static_occ` (combine or encode separately, but don't conflate channels).
2. Lightweight temporal encoder for the **eight** forecast `dynamic_occ` slices, preserving their time order.
3. MLP for current rover state, goal, and vehicle constraints where available.
4. Conditional trajectory vector-field network (small MLP/1D temporal model with flow-time embedding) predicting updates to a trajectory-shaped sample.

For training, with a normalized expert target `x1` and independent Gaussian noise `x0`:

```text
t ~ Uniform(0, 1)
x_t = (1 - t) * x0 + t * x1
v_target = x1 - x0
loss = mean_squared_error(v_theta(x_t, t, conditioning), v_target)
```

This is a baseline conditional flow-matching objective; add physically motivated auxiliary losses only after the baseline actually trains. The learned *flow velocity* is velocity through generative sample-space, **not the rover's travel velocity**.

At inference, sample Gaussian noise and integrate the learned vector field (Euler or Heun ODE integration) for a configurable number of function evaluations. Support multiple samples per scenario so the planner can rank alternative paths; report generation latency including all candidates and feasibility checks.

Do not pass target paths into conditioning, evaluation-time future observations, or source identifiers that leak expert solutions.

## 6. Training procedure — begin now on v1

Implement the following in order, executing each stage:

**Stage 0 — environment and data sanity:** Create an isolated Python environment, install the project, confirm the exact loader API and collate behavior, inspect several batches, and run the existing tests. Don't modify the shared dataset.

**Stage 1 — smoke test:** Confirm one forward/backward pass on real batches, stable finite gradients, conditioning shape compatibility, numerical round trips, and valid decoded samples. Overfit a small **training-only** subset (e.g., 64–256 windows) to make sure the objective is implemented correctly.

**Stage 2 — throughput benchmark:** Benchmark ~100–200 steps on the actual device after warm-up and report seconds/step and estimated time for a proposed full run. The teammate's laptop has an **AMD Radeon 860M**. Do not assume NVIDIA CUDA. Test compatible supported acceleration only if it works on the actual OS/drivers/operator set; keep a fully working CPU path and choose by measured stability/performance. Avoid CUDA-only extensions.

**Stage 3 — initial full training:** Train on the **4,203 training windows**, monitor the **420 validation windows** and avoid touching the **398 test windows** for model selection. Start with a modest model and configurable steps/epochs, batch size, learning rate, weight decay, and random seed. Save best and latest checkpoints throughout; choose stopping criteria based on actual validation results and compute budget rather than an arbitrary promise that it has converged.

**Stage 4 — evaluate:** Evaluate held-out performance only after selecting the model based on validation data. Generate multiple samples for a scenario, decode to world coordinates, check feasibility and collision proxies, and visualize predicted versus expert trajectories on actual maps. Record failures as well as successes.

## 7. Evaluation: build honest robotics metrics

Implement metrics for:

- Static collision rate (including **the rover footprint**, not point-center-only checks).
- Dynamic collision rate at aligned timestamps; distinguish estimated occupancy from ground-truth scenario replay.
- Goal progress / final displacement to the appropriate reachable goal; don't rely only on ADE/FDE when multiple valid routes exist.
- Kinematic feasibility: speed, steering/curvature, acceleration, and heading/position consistency.
- Trajectory smoothness and candidate diversity; avoid counting nearly identical samples as diverse.
- Inference time per candidate and per scenario, including ODE steps and filtering.
- Comparison with the **expert planner** where meaningful, while noting that “generates path faster” is not proven until measured on identical held-out scenarios.

Leverage existing dynamics/collision modules through **read-only imports** when possible. If those utilities cannot accurately validate generated paths, document the limitation and create new ML-side evaluation utilities without silently changing the dataset generator.

The downstream CBF safety filter is separate, under construction, and does **not** magically make a globally invalid trajectory feasible.

Include plots showing static obstacles, predicted dynamic occupancy, generated candidate routes, expert routes, heading/speed profiles, and invalid examples. These are diagnostics, not the Three.js frontend.

## 8. Checkpointing and fine-tuning contract

The owner is generating additional **new** scenarios in a separate dataset directory while you train. Do not wait for those examples. The **committed `data/dataset_v1` is immutable** for this first round.

Save at minimum:

- `flow_matching_v1_best.pt` — best validation checkpoint.
- `flow_matching_v1_latest.pt` — fully resumable latest checkpoint.

Each checkpoint should include model weights, optimizer state, scheduler state (if used), training step/epoch, architecture and feature specifications, preprocessing/normalization statistics, random seed/RNG states where practical, current metrics, dataset schema/version, and a dataset/split fingerprint. Retain the initial checkpoint permanently for comparison.

Build a separate `finetune` / `resume` execution path that:

1. Loads the v1 checkpoint and normalization configuration without incompatible reinterpretation.
2. Accepts a **validated new dataset directory** or a manifest of additional training examples, plus the original v1 **training** split.
3. Never incorporates v1 validation/test examples into training.
4. Checks schema, channels, units, horizon, and normalization compatibility before continuing.
5. Uses a configurable smaller LR and mixes old training windows with new data (potentially with stratified sampling to emphasize challenging dynamic interactions).
6. Saves new versioned outputs, e.g. `flow_matching_v2_best.pt`, without replacing v1.
7. Compares v1 and v2 on the **same held-out v1 test scenarios**, plus a separately held-out hard-dynamic test split.

**Fine-tuning is not automatic.** New files appearing in Git do not update an already trained checkpoint. Do not initiate it until the owner provides and approves the validated expanded corpus. If feature semantics or architecture changed, stop and explain whether weight transfer is possible.

## 9. Collaboration, Git, and ownership

- The owner will push a **new audit-fix commit** on `main` after the previously reported `c4cf550`; use the **current** remote `main`, not old commit IDs or old split counts.
- If your local repository has diverged because of an earlier history rewrite, **do not blindly merge or hard-reset** with uncommitted work. Coordinate with the owner or use a clean clone/new feature branch; protect your work.
- Put the ML implementation primarily under `backend/ml/` or a clearly isolated ML package; place run logs/checkpoints under a documented artifacts path ignored by Git as appropriate.
- **Read-only:** `data/dataset_v1/`, shared facility files, `northstar/pipeline/`, `northstar/planner/`, `northstar/scenario/`, `northstar/obstacles/`. Don't regenerate, overwrite, or commit modifications to them.
- Do not implement the full CBF, Three.js UI, or unrelated backend services.
- Do not push, force-push, merge into `main`, or rewrite shared history without explicit permission. Work in a separate feature branch and coordinate the eventual integration.
- Keep CLI scripts simple, reproducible, and runnable without a GPU.

## 10. Required deliverables

Implement and actually run:

1. Model + conditioning encoders + flow time embedding.
2. Flow-matching loss and sample-generation ODE solver.
3. Dataset-to-model adapters, preprocessing, and normalization.
4. Training, resume, evaluation, sampling, and fine-tuning CLIs/configs.
5. Tests for shapes, NaNs, input leakage checks, round-trip transforms, and checkpoint restoration.
6. Actual trained **v1 checkpoint** and diagnostic plots on the existing corpus.
7. Physical feasibility/collision metrics, failure analyses, and at least one meaningful comparison baseline.
8. A documented, callable inference interface: scenario observation + rover state + goal -> **multiple time-indexed candidate trajectories** in a stated coordinate frame.
9. An explicit fine-tuning procedure that will work with the owner's future dataset expansion.
10. A concise README explaining exact commands and dependencies.

In the final report state the **actual** device, training steps/duration, number of parameters, train/validation loss, held-out metrics, collision/kinematic validity, candidate generation latency, file paths, checkpoint versions, implementation limitations, and commands for another developer to reproduce your results. If you cannot complete training, save a working resumable state and report the blocker rather than claiming success.

**Begin with repository inspection and a first successful real batch. Then implement and run training.**
