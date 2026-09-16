# Arcade Binocular Vision Experiment

**Branch:** `arcade-binocular-vision` (built on `arcade-bridge-v2-complete`).
**Question:** Does restoring genuine bilateral visual input improve lateral
goalkeeper behaviour, versus the effectively one-eyed Arcade Bridge v2?

**Answer (honest):** No — not in this configuration. Genuine two-eye input did
**not** improve lateral decoding; offline held-out lateral correlation and
closed-loop save % are both modestly *lower* than one-eye v2. Blind-eye ablations
confirm the binocular model genuinely *uses* both eyes (blinding either eye
collapses its lateral signal toward chance), so this is a real bilateral
controller — the added right-camera information simply did not help here.

This experiment is kept separate from Arcade Bridge v2 and from Science Mode; none
of those artifacts were modified.

---

## 1. Eye / receptor mapping (audit)

The laterality field for R1-R6 receptors is the connectome **`rootSide`**, not
`somaSide` (which is ~all `None` for retina). From `binocular_audit.py`:

- **Left eye:** 1107 receptors, image viewport uv_x ∈ [0.00, 0.60] (mean 0.299).
- **Right eye:** 2228 receptors, uv_x ∈ [0.40, 1.00] (mean 0.754).
- Overlapping viewports by design (upstream `doom/prepare.py`:
  `left uv_x = 0.60·z`, `right uv_x = 0.40 + 0.60·(1−z)`).
- The two eye cameras are genuinely distinct (image mean-abs-diff 47–84/255).
- The **current one-eye path renders only `eye_left`** and samples all 3335
  receptors from it, so 2228 right-eye receptors read left-camera pixels. Feeding
  them the correct right camera changes their luminance by 0.08–0.32 (center 0.32).

Explicit manifest written to `binocular_eye_manifest.npz`
(`retina_body_id`, `rootSide`, `uv`, `left_mask`, `right_mask`).

## 2. True two-eye vision + sanity

`BinocularVisionBridge` renders both cameras and routes left-eye receptors ←
`eye_left`, right-eye receptors ← `eye_right`, each via its own uv. Ablation
conditions: `both`, `left_only`, `right_only`, `left_blind`, `right_blind`,
`both_blind`, `mono_left` (= current one-eye).

Sanity (`binocular_sanity.py`): switching the right receptors to the right camera
changes right-receptor luminance by up to 0.49 (center) and moves **58–82 of the
207 bridge-feature neurons** by >0.5 spikes/shot — so binocular input measurably
reaches the optic-lobe features.

## 3. Feature selection (training data only)

The frozen 207-neuron set is anatomically 206 R / 1 L — a poor basis for bilateral
control. For the binocular bridge we built a **600-neuron bilateral candidate
pool** (206 R + 394 L, spanning the 22 cell types of the frozen 207;
`binocular_pool.py`) and selected 207 features **from training episodes only**
(|corr(lateral, summed-window activity)|, top-k with a ≥20/side floor;
`binocular_train.py`). The selected set is genuinely bilateral: **144 R / 63 L**.
Split by episode/seed, same seed (20260916) and fractions as v2; no test leakage.

## 4. Results (held-out, honest)

| model | vision | neurons | lateral corr | lateral dir-acc | vertical corr |
|---|---|---|---|---|---|
| **Arcade Bridge v2** | one-eye | frozen 207 (206R/1L) | **0.407** | **0.682** | 0.332 |
| Binocular (selected) | two-eye | 207 selected (144R/63L) | 0.341 | 0.654 | 0.345 |
| Binocular (frozen 207) | two-eye | frozen 207 (206R/1L) | 0.219 | 0.589 | 0.341 |

**Blind-eye ablation** of the binocular selected model (fresh held-out seeds):

| condition | lateral corr | lateral dir-acc |
|---|---|---|
| both | 0.276 | 0.606 |
| left_only (right eye blind) | 0.097 | 0.525 |
| right_only (left eye blind) | 0.030 | 0.517 |

Blinding either eye collapses the lateral signal toward chance — the binocular
model truly depends on both eyes.

**Closed-loop 3×3** (27 matched shots, no privileged state to the controller):

| controller | overall | LEFT | CENTER | RIGHT |
|---|---|---|---|---|
| oracle / heuristic | 100% | 1.00 | 1.00 | 1.00 |
| v2 one-eye | 33.3% | 0.333 | 0.222 | 0.444 |
| binocular | 25.9% | 0.333 | 0.222 | 0.222 |

## 5. Interpretation

- Restoring the second eye is a **real** change (ablations prove both eyes are
  used), but it did **not** improve lateral goalkeeping here; it slightly hurt.
- The frozen right-hemisphere-heavy 207 set degrades markedly under two-eye input
  (0.41 → 0.22): those neurons were selected under the one-eye (all-left-camera)
  regime, and changing 2228 right-eye receptors' input distribution disturbs them.
- Data-driven bilateral feature selection recovers most of that loss (0.22 → 0.34)
  but still does not beat one-eye v2 (0.41).
- The closed-loop ceiling remains dominated by the same downstream DN-decode
  bottleneck documented for v2, so vision changes have limited closed-loop leverage.

This negative result is scientifically stronger than quietly adding a camera: it
isolates that, with this MaleCNS feature set and engineered DN decoder, the
one-eye Arcade bridge decodes lateral direction at least as well as the bilateral
one. A productive follow-up would target the DN-injection/decoder stage (the
established bottleneck) rather than the eye count.

## 6. Reproduce

```bash
source upstream/doomfly/.venv-neural/bin/activate
cd workspace
python -m experiments.arcade_demo.binocular_audit
python -m experiments.arcade_demo.binocular_sanity --backend metal
python -m experiments.arcade_demo.binocular_pool --max-pool 600
python -m experiments.arcade_demo.binocular_dataset --per-cell 10 --base-seed 1000 \
    --backend metal --out binocular_dataset.npz
python -m experiments.arcade_demo.binocular_train --dataset binocular_dataset.npz --k 207
python -m experiments.arcade_demo.binocular_compare
python -m experiments.arcade_demo.binocular_evaluate --per-cell 3 --base-seed 90000
```

Science Mode and Arcade Bridge v2 are untouched by this experiment.
