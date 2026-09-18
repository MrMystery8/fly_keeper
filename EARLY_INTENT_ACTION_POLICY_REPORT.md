# Early-Intent RL Goalkeeper: Final Architecture & Evaluation Report

## Executive Summary

The Arcade fruit fly goalkeeper project has completed its development and reached the final frozen stop condition. By transitioning from continuous closed-loop direction decoding (which degrades under physical self-motion to chance levels) to a decoupled **WHERE / HOW Early-Intent architecture**, the project achieved an embodied goalkeeper that actively defends goal-bound shots through real biophysical descending neurons (DNs) in MuJoCo.

The frozen project champion (`arcade_early_intent_vert_refined_rl.npz`, commit `aead9e8`, tag `arcade-early-intent-rl-70pct`) was evaluated across two rigorous batteries:

1. **Final Untouched Generalization Test (90 Clean-Goal Shots, Seeds 600000+)**:
   - **Overall Save Rate**: **67.8% (61/90 saves)** — independently validated on completely unseen seeds.
   - **Flank Balance**: `LEFT` = **80.0% (24/30)**, `CENTER` = **63.3% (19/30)**, `RIGHT` = **60.0% (18/30)**.
   - **Height Coverage**: `LOW` = **56.7% (17/30)**, `MID` = **76.7% (23/30)**, `HIGH` = **70.0% (21/30)**.
   - **Head-to-Head**: Decisively outperforms the continuous-decoding baseline (**v2 = 23.3%**) by nearly $3\times$, while approaching the physical **Oracle ceiling (94.4%)**.
   - **Causal Eye Ablations**: Intact bilateral save rate (**67.8%**) collapses to **10.0%** (`left_blind`), **25.6%** (`right_blind`), and **8.9%** (`both_blind`).

2. **Development Benchmark (54 Clean-Goal Shots, Seeds 500000+)**:
   - **Overall Save Rate**: **70.4% (38/54 saves)** (used to guide differential vertical refinement).

All shots in both test batteries passed the counterfactual absent-keeper validator: **natural-miss saves = 0**, guaranteeing that every evaluated shot was a genuine clean goal.

---

## Final Untouched Test Battery (90 Clean-Goal Shots, Seeds 600000+)

To obtain an untouched generalization claim, the frozen champion and fixed baselines were evaluated on 90 shots (10 shots per cell $\times$ 9 cells in the $3 \times 3$ grid) drawn from `seed = 600000+`. This seed range was verified to have never been touched in feature collection, encoder training, supervised demonstrations, DAgger, RL, or development evaluation.

### Matched 4-Way Baseline Comparison (90 Identical Untouched Shots)

| Metric / Slice | Oracle Ceiling | v2 Baseline (Monocular) | 53.7% Early-Intent Baseline | **Final Champion (Vertical-Refined)** |
|:---|---:|---:|---:|---:|
| **Overall Save %** | **94.4% (85/90)** | 23.3% (21/90) | 56.7% (51/90) | **67.8% (61/90)** 🏆 |
| **LEFT Shots** | 100.0% (30/30) | 26.7% (8/30) | 46.7% (14/30) | **80.0% (24/30)** 🔥 |
| **CENTER Shots** | 86.7% (26/30) | 36.7% (11/30) | 66.7% (20/30) | **63.3% (19/30)** |
| **RIGHT Shots** | 96.7% (29/30) | 6.7% (2/30) | 56.7% (17/30) | **60.0% (18/30)** 🔥 |
| **LOW Shots** | 83.3% (25/30) | 30.0% (9/30) | 50.0% (15/30) | **56.7% (17/30)** |
| **MID Shots** | 100.0% (30/30) | 33.3% (10/30) | 73.3% (22/30) | **76.7% (23/30)** 🎯 |
| **HIGH Shots** | 100.0% (30/30) | 6.7% (2/30) | 46.7% (14/30) | **70.0% (21/30)** 🚀 |
| **Peak Lateral (cm)** | — | — | 0.534 | **0.663** |
| **Peak Vertical (cm)** | — | — | 0.585 | **0.619** |
| **Diagonal Jumps** | — | — | 72.2% | **82.2%** |
| **Movement Onset** | — | — | step 6.68 | **step 7.14** |

### Complete $3 \times 3$ Matrix for Final Champion (90 Untouched Shots)

| Height \ Side | LEFT | CENTER | RIGHT | Height Total |
|:---|:---:|:---:|:---:|:---:|
| **HIGH** | 8/10 (80.0%) | 5/10 (50.0%) | 8/10 (80.0%) | **21/30 (70.0%)** |
| **MID** | 10/10 (100.0%) | 8/10 (80.0%) | 5/10 (50.0%) | **23/30 (76.7%)** |
| **LOW** | 6/10 (60.0%) | 6/10 (60.0%) | 5/10 (50.0%) | **17/30 (56.7%)** |
| **Side Total** | **24/30 (80.0%)** | **19/30 (63.3%)** | **18/30 (60.0%)** | **61/90 (67.8%)** |

### Outcome Diagnostics & Movement Mechanics

- **Keeper Contact**: **61/90 (67.8%)** — 100% of saves occurred through physical keeper body deflection.
- **Deflected Saves**: **61/90 (67.8%)** — every save was actively redirected away from the goal line.
- **Touch-but-Goal**: **0/90 (0.0%)** — no shots entered after deflecting off the keeper.
- **Natural Misses**: **0/90 (0.0%)** — strict clean-goal validation confirmed.
- **Mean Peak Lateral Displacement**: **0.663 cm** (vs 0.424 cm for v2).
- **Mean Peak Vertical Displacement**: **0.619 cm**.
- **Movement Onset Latency**: **step 7.14** (precisely following the step 0–7 sensing freeze).
- **Mean Airborne Duration**: **0.554 s** across all shots.
- **Vertical Behavior by Height**:
  - `LOW`: mean $u_{\text{vert}} = 0.270$, peak $z = 0.610\text{ cm}$, takeoff = 100%
  - `MID`: mean $u_{\text{vert}} = 0.262$, peak $z = 0.574\text{ cm}$, takeoff = 100%
  - `HIGH`: mean $u_{\text{vert}} = 0.299$, peak $z = 0.675\text{ cm}$, takeoff = 100%

### Final Eye Ablation Verification (90 Untouched Shots)

| Condition | Overall Save % | LEFT Saves | CENTER Saves | RIGHT Saves | Causal Finding |
|:---|---:|---:|---:|---:|:---|
| **Both Eyes Open (`both`)** | **67.8% (61/90)** | **80.0% (24/30)** | **63.3% (19/30)** | **60.0% (18/30)** | Intact bilateral sensory loop |
| **Left Eye Blind (`left_blind`)** | **10.0% (9/90)** | 30.0% (9/30) | **0.0% (0/30)** | **0.0% (0/30)** | Contralateral and center collapse to 0% |
| **Right Eye Blind (`right_blind`)**| **25.6% (23/90)** | **0.0% (0/30)** | **0.0% (0/30)** | 76.7% (23/30) | Contralateral and center collapse to 0% |
| **Both Eyes Blind (`both_blind`)** | **8.9% (8/90)** | 26.7% (8/30) | **0.0% (0/30)** | **0.0% (0/30)** | Center and right collapse; chance floor |

> **Scientific Interpretation**: The learned goalkeeper depends decisively on the intact bilateral visual configuration. Monocular blindness acts as a severe out-of-distribution optical intervention that completely eliminates contralateral and central interception ability.

---

## Development Benchmark (54 Clean-Goal Shots, Seeds 500000+)

The development benchmark served to guide model selection and differential residual refinement:

| Metric / Slice | 53.7% Early-Intent Baseline | **Vertical-Refined RL (Final Champion)** |
|:---|---:|---:|
| **Overall Save %** | 53.7% (29/54) | **70.4% (38/54)** 🏆 |
| **LEFT Shots** | 38.9% (7/18) | **83.3% (15/18)** |
| **CENTER Shots** | 61.1% (11/18) | **61.1% (11/18)** |
| **RIGHT Shots** | 61.1% (11/18) | **66.7% (12/18)** |
| **LOW Shots** | 33.3% (6/18) | **38.9% (7/18)** |
| **MID Shots** | 72.2% (13/18) | **94.4% (17/18)** |
| **HIGH Shots** | 55.6% (10/18) | **77.8% (14/18)** |
| **Peak Lateral (cm)** | 0.593 | **0.702** |
| **Peak Vertical (cm)** | 0.575 | **0.609** |
| **Diagonal Jumps** | 74.1% | **81.5%** |
| **Movement Onset** | step 6.59 | **step 7.24** |

### Complete $3 \times 3$ Matrix (54 Development Shots)

| Height \ Side | LEFT | CENTER | RIGHT | Height Total |
|:---|:---:|:---:|:---:|:---:|
| **HIGH** | 6/6 (100.0%) | 4/6 (66.7%) | 4/6 (66.7%) | **14/18 (77.8%)** |
| **MID** | 6/6 (100.0%) | 5/6 (83.3%) | 6/6 (100.0%) | **17/18 (94.4%)** |
| **LOW** | 3/6 (50.0%) | 2/6 (33.3%) | 2/6 (33.3%) | **7/18 (38.9%)** |
| **Side Total** | **15/18 (83.3%)** | **11/18 (61.1%)** | **12/18 (66.7%)** | **38/54 (70.4%)** |

---

## Causal Architecture & System Delineation

```text
LEFT CAMERA ──┐
              ├─► Fullframe Retinal Mosaics ─► MaleCNS Connectome (Metal Backend)
RIGHT CAMERA ─┘
                                                       │
                     ┌─────────────────────────────────┴─────────────────────────────────┐
                     ▼ [Pre-Motion Window: Steps 0–8]                                    ▼ [Execution Phase: Steps 8+]
         Bilateral Neural Streams (6-dim)                                    Ongoing Neural Streams (6-dim)
                     │                                                                   │
                     ▼                                                                   │
         Early-Intent Estimator (Softmax)                                                │
                     │                                                                   │
                     ▼                                                                   │
         Directional Intent & Confidence ───────────────────────────────────────────────►│
                                                                                         ▼
                                                                             Execution Policy (ExecPolicy)
                                                                               + Fly Proprioception (5-dim)
                                                                               + Previous Action (2-dim)
                                                                                         │
                                                                                         ▼
                                                                             Continuous Actions (u_lat, u_vert)
                                                                                         │
                                                                                         ▼
                                                                             Real Descending Neurons (ArcadeDNBasis)
                                                                                         │
                                                                                         ▼
                                                                             MuJoCo Physical Body (ArcadeFlyBody)
                                                                                         │
                                                                                         ▼
                                                                               Hop / Jump / Dive Parry
```

### Biological vs. Engineered Components

To maintain scientific integrity, the biological and engineered components are strictly delineated:

#### Biological Components (Faithfully Simulated):
1. **MaleCNS Connectome**: Full biophysical adult male Drosophila connectome (Metal backend), with per-episode membrane voltage and spike activity reset.
2. **Retinal Receptor Populations**: Left and right compound eye ommatidia arrays receiving physical optical projections.
3. **Spiking Dynamics**: Biological spike propagation across intermediate neuropils (lobula, lobula plate, central complex).
4. **Descending Neurons (DNs)**: Real identified descending neurons (`DNa01`, `DNa02`, `DNb01`, etc.) driving somatic motor circuits.

#### Engineered Components:
1. **Camera-to-Retina Interface**: Perspective projection of MuJoCo 3D arena onto ommatidial coordinate frames.
2. **Neural-Stream Readout**: Linear pooling of bilateral optic lobe soma spike rates into a compact 6-dimensional embedding.
3. **Early-Intent Estimator**: 3-class softmax classifier mapping pre-motion embeddings to signed directional intent.
4. **Execution Policy (`ExecPolicy`)**: Tiny tanh MLP mapping latched intent, ongoing embedding, and proprioception to DN drive.
5. **Reward Objective**: Simulator-truth terminal and shaping signals utilized during training only.
6. **Arcade Locomotor Physics**: Bounded physical forces representing aerial jump and dive mechanics.

> **Crucial Note**: We do NOT claim that Drosophila naturally understands football or ball interception. The architecture demonstrates that a biological connectome can provide rich, sensory-conditioned internal representations that an engineered execution layer can learn to steer.

---

## The Key Scientific Finding: WHERE vs. HOW Decoupling

A central insight emerged from the failure of earlier continuous-decoding models:

### Why Continuous Decoding Failed:
In continuous closed-loop setups (such as v2), the fly begins moving immediately. Once physical locomotion starts, three degrading factors compound:
1. **Self-Motion Visual Blur**: Keeper movement shifts the retinal image across ommatidia, confounding ball trajectory estimation with ego-motion optic flow.
2. **Distributional Drift**: Motor feedback and changing camera angles shift MaleCNS internal activity away from the passive sensory manifold on which linear decoders were fit.
3. **Lateral Asymmetry & Collapse**: The decoder rapidly latches onto spurious single-eye cues, collapsing one flank (RIGHT saves fell to 0.0%–6.7%).

### The WHERE / HOW Solution:
Biological ball-catchers and visual predators frequently decouple visual trajectory estimation from motor execution:
1. **WHERE (Pre-Motion Sensing Window, Steps 0–8)**:
   The fly keeper holds physically stationary (`gait_on = 0`, zero lateral drive, zero DN injection). The retinal view remains pristine, perfectly matching the stationary sensory distribution. Over 8 decision steps ($\approx 160\text{ ms}$), the `EarlyIntentEstimator` accumulates evidence to determine shot direction and confidence.
2. **HOW (Execution Phase, Steps 8+)**:
   At step 8, the directional intent is latched as an input to the execution policy. The policy now only solves the motor control problem: translating intent and proprioception into continuous DN commands, producing decisive lateral dives ($\text{peak\_lat} = 0.663\text{ cm}$) and diagonal aerial parries without second-guessing shot direction under self-motion noise.

---

## Reproducibility & Frozen Specifications

### Environment & Execution
- **Python Environment**: `upstream/doomfly/.venv-neural/bin/python`
- **Python Path**: `PYTHONPATH=workspace`
- **Platform**: macOS Apple Silicon (Metal backend for MaleCNS)

### Artifact Paths & Checksums (SHA-256)

| Artifact | Path | SHA-256 Checksum |
|:---|:---|:---|
| **Champion Policy** | `workspace/outputs/arcade_demo/arcade_early_intent_vert_refined_rl.npz` | `710b703c01915c26494461e65e13a370674103ae1bc5ca56189f64e6a0627599` |
| **53.7% Champion** | `workspace/outputs/arcade_demo/arcade_early_intent_ei_rl_53pct_champion.npz` | `106f9303fafb2d8e1433e9505dde104bc0cf6fd9804f3a5198b3dd57cf95be22` |
| **Intent Estimator** | `workspace/outputs/arcade_demo/early_intent_estimator.npz` | `4761b863c156406aacd974cd88ea4ba5a9a49fbb00e273c03228e3b5bca366b6` |
| **Neural Encoder** | `workspace/outputs/arcade_demo/binocular_rich_encoder_early.npz` | `f159364517f64f07f7f5384be65ffe6a77daa9aafcc9adceb2cada5c647fd082` |
| **Final Untouched Eval** | `workspace/outputs/arcade_demo/arcade_early_intent_final_untouched_eval.json` | `eddb96a3fa997bf1884c199d47fd13cf8f6c5118262c1e1bf8249f4c212308c6` |

### Runtime Observation Vector (15 Dimensions)
Deployed runtime execution policies receive **zero privileged ball information**:
- `obs[0]`: Latched lateral early intent $P(\text{right}) - P(\text{left}) \in [-1, +1]$
- `obs[1]`: Intent confidence margin $\max(P) - \frac{1}{3}$
- `obs[2:8]`: 6-dimensional bilateral MaleCNS neural stream embedding
- `obs[8:13]`: Fly proprioception: $[y, z - z_{\text{stand}}, v_y, v_z, \text{airborne}]$
- `obs[13:15]`: Previous motor action: $[u_{\text{lat}}, u_{\text{vert}}]$

### Automated Regression Test Suite
Run the automated verification suite:
```bash
PYTHONPATH=workspace upstream/doomfly/.venv-neural/bin/python \
  workspace/experiments/arcade_demo/test_champion_regressions.py
```
Passes all 4 tests in $\approx 10.8\text{ s}$:
- Lateral sign convention (LEFT $\to +y$, RIGHT $\to -y$)
- Sensing freeze enforcement (steps 0–7: `gait_on = 0`, zero DN injection)
- Inter-episode reset hygiene
- Fullframe binocular mapping preservation

### Interactive Playable Goalkeeper
Launch the interactive game:
```bash
PYTHONPATH=workspace upstream/doomfly/.venv-neural/bin/python \
  workspace/experiments/arcade_demo/play_arcade_goalkeeper.py
```
- **Controls**: `LEFT`/`RIGHT` aim direction, `UP`/`DOWN` aim height, `SPACE` fires, `C` or `1–7` change camera angle, `R` resets score, `Q`/`ESC` quits.
- Defaults directly to the frozen champion policy `arcade_early_intent_vert_refined_rl.npz`.
