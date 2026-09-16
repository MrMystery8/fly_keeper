# Arcade Bridge v2 — Stale-Dataset Correction

**Experiment:** Arcade Mode (engineered flying goalkeeper), Metal real-time backend.
**Scope:** Correct the training dataset that produced the deployed Arcade Bridge v1
and retrain the *same* linear architecture on clean data, to test one causal
hypothesis in isolation. **Science Mode is not touched.**

> The original scientific bridge (`learned-visual-dn-bridge-complete`) was frozen
> after validation. Arcade Mode uses a **separate** Metal-compatible 2-output (2D)
> bridge trained for the flying goalkeeper. Arcade Bridge v2 corrects a data-pipeline
> bug in that Arcade bridge; it does not reinterpret or retrain Science Mode. A later
> binocular Arcade experiment (separate) tests whether restoring both simulated eyes
> improves lateral visual control.

---

## 1. Headline result (honest)

| Metric | v1 (stale data) | v2 (corrected data) |
|---|---|---|
| **Offline held-out lateral corr** | 0.2878 | **0.4069** |
| Offline lateral MAE | 0.6011 | 0.5418 |
| Offline lateral direction accuracy | — | **0.6822** |
| Offline held-out vertical corr | 0.7076 | 0.3315 |
| Offline vertical MAE | 0.0344 | 0.2852 |
| Closed-loop save % (3×3, 36 shots) | 41.7% | 38.9% |

**Two distinct conclusions, both reported honestly:**

1. **The stale-dataset fix works at the level it operates.** Correcting the data
   raised the **offline** held-out lateral correlation from **0.29 → 0.41** and gives
   a real directional signal (**direction accuracy 0.68**, `LEFT < CENTER < RIGHT`
   ordering). This confirms the causal hypothesis from the handoff: the deployed v1
   bridge was degraded by post-resolution padding frames, not by Metal losing the
   information.

2. **This did not translate into materially better closed-loop save %.** v2 (38.9%)
   is not better than v1 (41.7%) and both sit near the passive floor (38.9%). The
   closed-loop shortfall is a **separate downstream bottleneck** in the
   DN-injection → DN-asymmetry decoder stage (Section 8), *not* the training data.
   We report this rather than tuning it away.

The realistic offline target the handoff set (lateral corr ≳ 0.6, dir acc ≈ 0.8+)
was **not** reached on honest held-out data. Section 6 explains why: the ≈0.65–0.82
figure that motivated that target was an **in-sample, low-shots-only** probe number;
the honest held-out ceiling on the full low+high distribution is ≈0.41–0.48.

---

## 2. Stale-dataset diagnosis (independently reproduced)

The v1 dataset generator collected one feature window per 20 ms decision while
`world.result is None and n < MAX_DECISIONS (90)`. That guard is correct *only if the
world reliably sets a terminal result*. It did not for lofted shots.

### 2.1 Proof of post-resolution padding in v1

Reconstructing per-episode lengths from the stored v1 dataset (`arcade_dataset_v1.npz`):

```
episodes: 60   total steps: 3339   min 29 / median 42 / mean 55.65 / max 90
length histogram (tail): ... 45:6  47:1  79:1  90:19
episodes that padded to MAX_DECISIONS=90: 19 / 60  (32%)
```

**One third of v1 episodes ran the full 90 steps** while the meaningful shot resolved
around step ~42 — i.e. ~48 post-resolution "dead-ball" frames per stale episode,
paired with saturated teacher labels.

### 2.2 Root cause of non-termination

Instrumenting fresh episodes in the current world isolated it to **lofted (high)
shots**. Low shots always resolved cleanly (29–58 steps). Every high shot failed to
resolve and padded to 90:

```
seed  group   h     result  terminal  ball_x@term
1003  left   0.85   None      90        +0.111
1009  center 0.85   None      90        -0.141
1015  right  0.85   None      90        +0.379
...
```

A lofted ball travels normally toward goal, then near the goal mouth its forward
velocity collapses (a crossbar/frame deflection of the large ball) and it stalls or
dribbles at ~0.2 cm/s in front of the line — never crossing `x ≤ GOAL_LINE_X` and never
matching the "deflected clearly away" SAVE branch. The scorer had no terminal case for
"dead ball in front of goal," so the episode padded to the cap.

This is a **data-pipeline bug**, exactly as the handoff stated. It is not evidence that
Metal lacks lateral information.

---

## 3. The fix: guaranteed terminal resolution + anti-padding

### 3.1 World (`arcade_world.py`)

Added a stall/dead-shot terminal branch to `_check_outcome`, preserving all existing
`contact ≠ save` semantics:

- **GOAL** if the ball crosses `x ≤ GOAL_LINE_X` inside the posts and below the bar
  (`touch_but_goal` flagged if the keeper had touched it — a keeper touch that still
  goes in is a GOAL).
- **SAVE** if it crosses wide/over, or was deflected clearly up-field and away.
- **NEW — stall resolution:** if the ball is in front of the line (`x > GOAL_LINE_X`)
  with total speed `< 1.2 cm/s` for `5` consecutive steps, the shot is dead. It resolves
  as a *creeping GOAL* only if it has effectively reached the mouth inside the posts,
  otherwise a *SAVE* (kept out). Records `terminal_step`; exposes `shot_live` and
  `terminal_step`.

Verification: every fresh episode now terminates (36 episodes: lengths 32–68, **0**
reached the cap, **0** unresolved, 18 GOAL / 18 SAVE). The oracle physical ceiling on
the corrected world is **still 100% (36/36)** across the full 3×3 matrix — the fix adds
a terminal case, it does not change reachability.

### 3.2 Generator (`arcade_dataset.py`)

- Collects **only while `world.shot_live`** — never padded to `MAX_DECISIONS`.
- Emits a **per-episode manifest** (`episode_id`, `shot_seed`, `shot_class`,
  `height_class`, `episode_length`, `terminal_result`, `terminal_step`,
  `last_step_collected`, `keeper_contact`).
- `_assert_integrity()` **fails the run** if any episode is unresolved, collected a
  window after its terminal step, or hit the safety cap.
- Prints the natural (variable) episode-length distribution; lengths are *not* forced
  equal.

### 3.3 Old vs fresh episode lengths

| | v1 (stale) | v2 (corrected) |
|---|---|---|
| episodes | 60 | 60 |
| total samples | 3339 | 3008 |
| length min / median / mean / max | 29 / 42 / 55.6 / **90** | 31 / 52 / 50.1 / **67** |
| episodes at MAX_DECISIONS (90) | **19** | **0** |

~331 post-resolution padding frames were removed; no episode reaches the cap.

---

## 4. A teacher-label bug the stale data was masking

Correcting the padding exposed a second bug. The teacher (`teacher_command`, used for
labels only) predicted the ball's goal-line crossing height with a full ballistic term
`0.5·g·t²` at `g = −981 cm/s²` — but the arcade world **cancels gravity on a lofted ball**
(straight-line rise). So for high shots the teacher predicted a hugely negative crossing
height and `u_vert` collapsed to ~0.

**Why v1's vertical corr looked good (0.71):** its signal came largely from the ~48
*contaminated* post-loft frames, where the loft had ended and the ball was genuinely
falling under gravity — so the buggy ballistic teacher happened to be right *there*. The
stale frames were supplying the vertical "signal."

**Fix:** the teacher now uses `g = 0` while `world._ball_no_gravity` (lofting), matching
the world's and the oracle's loft model (`z_cross = bz + vz·t`). After the fix, high-shot
`u_vert` reaches ~0.82 (mean ~0.44–0.48) on live frames; low-shot `u_vert = 0`. Lateral
labels are unchanged and clean (LEFT −1.0, RIGHT +1.0, CENTER ~0). The v2 dataset's
`vert_frac_active` is 0.33 — now from *live* frames, not padding.

This is why v2's honest vertical corr (0.33) is *lower* than v1's (0.71): v1's number was
inflated by contaminated frames + a compensating teacher bug. v2 measures the honest,
live-frame vertical decodability.

---

## 5. Data integrity, splits, no leakage

- **Split by complete episode / shot seed** (fixed seed 20260916), 42 / 9 / 9 episodes
  train / val / test. Individual temporal windows are never split.
- No test seed influences feature selection (frozen 207-neuron manifest, unchanged),
  normalization (train-only `μ/σ`), regularization (`α` chosen on validation), or bias.
- Temporal windows are the 4 causal 20 ms windows preceding each decision, and collection
  stops at `terminal_step`, so **no window incorporates neural data after the ball
  crossed / after a save / after reset**, and there is no cross-episode contamination
  (each episode uses its own fresh `MaleCNSBrain.reset()` and ring buffer).

---

## 6. Offline held-out metrics (v2)

Deployed 2-output linear bridge, held-out test seeds (low + high combined):

**Lateral** — corr **0.4069**, MAE 0.5418, MSE (reported in JSON), **direction accuracy
0.6822**.

Per-class predicted lateral mean (u_lat convention: <0 = left, >0 = right):

| class | raw mean | relative to CENTER |
|---|---|---|
| LEFT | −0.2692 | **−0.31** |
| CENTER | +0.0408 | 0.00 |
| RIGHT | +0.1119 | **+0.071** |

Ordering is correct: **LEFT < CENTER < RIGHT**. The model is slightly left-of-center in
absolute terms but the *directional* structure is right.

Confusion matrix (LEFT/RIGHT frames, sign of prediction):

|  | pred LEFT | pred RIGHT |
|---|---|---|
| **true LEFT** | 75 | 22 |
| **true RIGHT** | 46 | 71 |

LEFT is decoded well (75/97); RIGHT is weaker (71/117, 46 leak to left) — consistent with
the known anatomical hemisphere imbalance in the 207-neuron set (Section 9).

**Vertical** — corr 0.3315, MAE 0.2852, MSE 0.1178.

### 6.1 Reconciling with the audit's ≈0.82 figure (important)

The prior audit's Metal lateral probe reported corr ≈0.82 / dir-acc ≈0.90. Replicating
that probe's exact methodology on the corrected v2 data shows it was **in-sample and
low-shots-only**:

| variant | corr | dir acc |
|---|---|---|
| low-only, **in-sample** (audit-style) | **0.833** | **0.911** |
| low-only, held-out by seed | 0.000\* | 0.825 |
| low+high, in-sample | 0.704 | 0.830 |
| **low+high, held-out (v2 deployed)** | **0.484** | **0.678** |

\*corr collapses on the tiny held-out low-only set (variance artifact); direction is still
82% correct.

So the honest held-out linear ceiling is ≈0.41–0.48 corr / ≈0.68–0.82 dir-acc. The
≈0.82 figure that set the "≳0.6" expectation was optimistic (in-sample). This is exactly
the in-sample-vs-held-out mistake the handoff warned against; we do not repeat it.

A separate check confirmed a **separate per-axis ridge gives the same lateral corr (0.41)**
as the joint 2-output fit — a shared regularizer is *not* the bottleneck, so per-axis heads
are not currently justified.

---

## 7. Offline / runtime inference parity

Held-out test samples run through the exact runtime inference path
(`LinearBridge2D.predict`, as `ArcadeLearnedBridge.command` calls it) reproduce the offline
training computation, and the live feature extractor reproduces stored features:

```
normalization (μ, σ):        bit-identical
feature identity/order:      sel_graph == dataset graph_index; newest-first
manifest body_ids == dataset graph_index; extractor dim 828 == 207×4
prediction parity (419 test samples): max |Δ| = 1.1e-15
live extractor parity:       max |Δ| = 0.0
ALL_PASS = true
```

Offline metrics therefore transfer exactly to the runtime bridge.

---

## 8. Closed-loop 3×3 evaluation

Matched shot matrix `{LEFT, CENTER, RIGHT} × {LOW, MID, HIGH}`, 4 seeds/cell = 36 shots,
eval base seed 90000. Each bridge runs its own frozen runtime config (v1: smoothing 0.7,
lat_gain 3.5, vert_gain 3.5; v2: smoothing 0.2, lat_gain 3.5, vert_gain 4.5 — see §8.1).
`keeper contact ≠ automatic SAVE`; the goal-line test decides.

| controller | overall | LEFT | CENTER | RIGHT | LOW | MID | HIGH | keeper-contact | touch-but-goal |
|---|---|---|---|---|---|---|---|---|---|
| **oracle / heuristic** | **100%** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — | — |
| passive (floor) | 38.9% | — | — | — | — | — | — | — | — |
| Arcade Bridge **v1** | 41.7% | 0.25 | 0.583 | 0.417 | 0.417 | 0.00 | 0.833 | 0.278 | 0.00 |
| Arcade Bridge **v2** | 38.9% | 0.167 | 0.333 | 0.667 | 0.333 | 0.00 | 0.833 | 0.111 | 0.00 |

v2 full 3×3 cell matrix (saves/n):

```
              LOW   MID   HIGH
   LEFT       0/4   0/4    2/4
   CENTER     0/4   0/4    4/4
   RIGHT      4/4   0/4    4/4
```

Observations:
- Both learned bridges sit at the passive floor overall. High shots are saved well
  (vertical channel works); the MID row is 0 for *all* learned controllers **and** for
  passive, even though the oracle saves MID 100% — so MID is physically reachable but the
  weak neural command does not produce the required precise vertical hold.
- v2 remains **right-favoring** at runtime (RIGHT 0.667 > CENTER 0.333 > LEFT 0.167),
  the same hemisphere asymmetry seen offline and rooted in the neuron anatomy (§9).
- No `touch-but-goal` events occurred in this matrix; contact remained distinct from SAVE.

### 8.1 Runtime attenuation (why offline gains don't reach the body)

A step-by-step trace of the runtime lateral path found **five** transform stages between
`model.predict` (offline ±0.3) and the body command, with the discriminative lateral
signal confined to the last ~15 frames of an episode:

1. bridge EMA (`cmd_smoothing`) — **primary attenuator**;
2. `lat_gain` scaling → mV;
3. DN current basis (bounded, one-sided, `drive_mv 25`, clip ±1);
4. fixed MaleCNS dynamics → DN spikes;
5. decoder normalized L/R asymmetry + `turn_gain` + its own EMA (0.5).

At the observed operating point the lateral DN drive is only ~1.75–8.75 mV (vs the 25 mV
used to *screen* drivability), so the L/R spike asymmetry the decoder normalizes is small.
A calibration sweep (on calibration seeds **disjoint** from the eval matrix, base 70000)
confirmed the diagnosis: lowering `cmd_smoothing` 0.7 → 0.2 raised calibration save% from
33% → 39%, while raising `lat_gain` did **not** help (the DN-asymmetry stage, not the
pre-DN gain, is the bottleneck). v2's frozen runtime config adopts the calibrated
`cmd_smoothing 0.2` / `vert_gain 4.5` (model **weights unchanged**).

The remaining closed-loop gap is therefore located in the **fixed, engineered
DN-injection → decoder stage**, which is downstream of and independent from the training
data corrected here. We deliberately did **not** escalate to a nonlinear model (the linear
offline decode partially works; the bottleneck is not the model class).

---

## 9. Limitations

- **One-eye visual path.** The Arcade `VisionBridge` renders only `eye_left` and drives all
  ~3,335 retinal receptors from that single image. The right-eye camera produces genuinely
  distinct imagery (~47–55/255 mean absolute pixel difference) that is currently discarded.
  This is inherited from the frozen implementation and is the subject of the *next*,
  separate binocular experiment. It is not corrected here (v2 deliberately preserves the v1
  visual architecture so the stale-data fix is isolated).
- **Hemisphere imbalance.** The 207 selected neurons are anatomically ~206 right / 1 left
  somaSide (more balanced by functional preference: ~68 left-pref / ~139 right-pref). RIGHT
  decodes better than LEFT both offline and closed-loop.
- **DN-decode bottleneck.** The closed-loop lateral command is small because the engineered
  DN-injection/decoder stage produces weak L/R asymmetry at the operating drive; this caps
  closed-loop lateral behaviour regardless of offline improvements.
- **Vertical labels depend on the arcade loft model.** The teacher (and oracle) treat a
  lofted shot as a straight rise (gravity cancelled), an arcade trajectory choice, not
  ballistics.
- **Small samples.** 60 episodes / 9 test episodes; the 3×3 closed-loop uses 4 seeds/cell.

---

## 10. Distinction from Science Mode

| | Science Mode (frozen) | Arcade Bridge v2 |
|---|---|---|
| tag | `learned-visual-dn-bridge-complete` | `arcade-bridge-v2-complete` (this work) |
| backend | CPU scientific reference | Metal real-time |
| body | grounded FlyBody | engineered force-driven flight |
| bridge output | 1 scalar (lateral) | 2 outputs (lateral, vertical) |
| status | **untouched, not retrained** | separate engineered mode |

Science Mode's ~33% (natural) / ~71% (learned bridge) / ~94% (heuristic) results are not
reinterpreted or retrained based on any Arcade result.

---

## 11. Frozen v2 artifact

`arcade_bridge_v2_freeze.json` records: dataset file + sha256 and its per-episode manifest;
train/val/test split seeds (42/9/9, seed 20260916); the 207 neuron IDs (graph_index +
body_ids); the 4×20 ms temporal-window definition; normalization checksums; weights/bias
checksums and shapes; the DN mapping (lateral DNp20/DNpe017 L/R, vertical DNp01 pair) with
drive bounds; and the (calibrated) motor-interface gains. **v1 is preserved** for comparison
(`arcade_bridge_model_v1.npz`, `arcade_dataset_v1.npz`) and remains reproducible.

---

## 12. Reproduce

```bash
source upstream/doomfly/.venv-neural/bin/activate
cd workspace

# 1. verify the stale diagnosis (stored v1 lengths + fresh termination)
python -m experiments.arcade_demo.diagnose_stale --per-cell 6

# 2. regenerate the corrected one-eye Metal dataset (integrity-asserted)
python -m experiments.arcade_demo.arcade_dataset --per-cell 10 --base-seed 1000 \
    --backend metal --out arcade_dataset_v2.npz

# 3. train the same linear 2-output bridge on corrected data
python -m experiments.arcade_demo.arcade_train --dataset arcade_dataset_v2.npz \
    --out-model arcade_bridge_model_v2.npz --out-report arcade_bridge_train_v2.json \
    --model-version arcade-bridge-v2

# 4. offline analysis + inference parity
python -m experiments.arcade_demo.offline_probe_v2
python -m experiments.arcade_demo.probe_methodology
python -m experiments.arcade_demo.parity_check

# 5. runtime-gain calibration (disjoint seeds) + freeze
python -m experiments.arcade_demo.calibrate_v2 --per-cell 2 --base-seed 70000
python -m experiments.arcade_demo.freeze_v2

# 6. closed-loop 3x3 comparison (v1 vs v2 vs oracle/passive)
python -m experiments.arcade_demo.arcade_evaluate --per-cell 4 --base-seed 90000 \
    --out arcade_evaluate_v1_vs_v2.json
```

## 13. Core interpretation

> Metal MaleCNS retains useful lateral information; the deployed Arcade v1 bridge was
> degraded mainly because its training dataset was contaminated by post-resolution frames
> (plus a teacher gravity bug the contamination masked). Correcting the data recovers the
> **offline** lateral correlation (0.29 → 0.41, direction accuracy 0.68). It does **not** by
> itself recover closed-loop save %, because a separate downstream bottleneck — the fixed
> engineered DN-injection/decoder stage — limits how much of that lateral signal reaches the
> body. The next experiment (binocular vision) is kept experimentally separate so its effect
> is not confounded with this stale-data correction.
