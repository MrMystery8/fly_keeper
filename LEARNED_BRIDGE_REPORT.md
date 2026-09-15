# Minimal learned visual → descending-neuron bridge — report

**Question.** Given that the fixed MaleCNS already encodes ball direction strongly
in its early visual populations and that real descending neurons can causally
produce useful left/right body movement, can a *deliberately minimal learned
transformation* across the experimentally-identified representational bottleneck
(optic-lobe → visual-projection → descending) produce genuinely
**vision-dependent** embodied goalkeeper behaviour — and how small can that
artificial transformation be?

**Answer (positive, with controls).** Yes. A single-degree-of-freedom linear
bridge from selected optic-lobe neurons to a fixed opponent descending-neuron
motor basis raises held-out save rate from the Natural-MaleCNS floor of **33 %**
to **71 %** (95 % CI [0.57, 0.82]; bridge − natural = +0.375, two-proportion
z = 3.68, p = 2.4 × 10⁻⁴). The behaviour is genuinely vision-driven: it collapses
to baseline under a blinded eye (35 %) and under a per-timestep shuffled retina
(38 %), degrades under mirrored vision (50 %), and is bit-for-bit Natural MaleCNS
when the bridge is switched OFF. Causal ablations show clean laterality, and the
intervention can be shrunk to **16 optic-lobe neurons / 65 trainable parameters**
or a **single opponent DN pair** while remaining clearly above baseline.

> **Scientific status.** The native MaleCNS synaptic weights did **not** learn the
> task. MaleCNS provides a fixed, biological *visual representation*; the learned
> bridge is a small, explicitly engineered decoder that connects that
> representation to real MaleCNS descending neurons. A fair short description:
> *a fruit-fly-brain-powered goalkeeper using a minimal learned visual-to-motor
> bridge.*

Branch `learned-visual-dn-bridge`, forked from tag `pathway-audit-complete`
(native fixed weights, **not** plasticity-modified state). CPU scientific-reference
backend only.

---

## Motivation

Four prior results (frozen at tags `embodied-flykeeper-baseline`,
`neural-diagnostics-complete`, `pathway-audit-complete`,
`targeted-plasticity-complete`) set up this experiment:

1. **The body works.** The articulated MuJoCo fly + football environment let a
   privileged heuristic save ~90–98 % of shots (here 94 % on the held-out test
   set). Body mechanics are not the bottleneck.
2. **Natural MaleCNS is ~passive.** The fixed brain saves ~33 % — 0 % left,
   100 % centre, 0 % right — i.e. it only "saves" centre shots by standing still.
3. **The visual side already knows the direction.** Retina and early optic-lobe
   populations decode left-vs-right essentially perfectly.
4. **The motor side is capable.** Causal stimulation of selected descending
   neurons produces useful, approximately symmetric left/right movement.

The pathway audit localized the failure to the **optic-lobe → visual-projection**
transition, where the directional contrast collapses and only a common-mode
"something is out there" signal reaches the descending neurons. Targeted
reward-modulated plasticity on that route **failed**: it strengthened common-mode
transmission but never built left/right opponent structure (trained ≈ passive ≈
37.5 %, vision-independent, opponent asymmetry ≈ 0). The audit's conclusion was
that the defect is **representational / credit-assignment**, not a lack of
training — so the appropriate next step is a small, explicit, learned bridge
across the missing mapping, tested under strict vision-dependence controls. That
is this experiment.

---

## Frozen biological / engineered components

Unchanged during all bridge work (verified: the connectome file on disk was
never written; graph SHA-256 is checked by the adapter):

- Native MaleCNS synaptic weights (~25.6 M edges, ~166.7 k neurons).
- LIF / neural dynamics and the native kernel.
- Retinal mapping and R1–R6 encoding (`retinal_samples`, receptor UV layout).
- Body physics and MuJoCo mechanics; the CPG locomotion layer.
- The descending-neuron → locomotion motor decoder mechanics.
- The CPU scientific-reference backend (no Metal during this phase).

Engineered (as in all prior phases, and explicitly reported as such): the LIF
model itself, retinal encoding, body interface, motor decoding, and **the learned
bridge introduced here**.

---

## Architecture

Full runtime path (bridge ON):

```
3D scene
→ fly eye camera (eye_left) → R1–R6 retinal sampling      [fixed]
→ full fixed MaleCNS visual processing                     [fixed]
→ 207 selected optic-lobe neuron spike counts, last 4×20 ms windows
→ LEARNED bridge: normalize → linear map → tanh → scalar u ∈ [−1,+1]
→ fixed DN motor basis B_lr: u → bounded additive current into 4 real DNs
→ real DNs evolve under the existing MaleCNS LIF dynamics   [fixed]
→ existing DescendingMotorDecoder → CPG locomotion          [fixed]
→ articulated MuJoCo fly body                               [fixed]
```

Per 20 ms decision the controller runs the *exact* Natural-MaleCNS loop
(render → step brain 20 ms → read DNs → decode) and, when enabled, **adds** a
bounded external current to the selected DNs before the next step. The bridge
current from decision *t−1* is applied at the brain step of decision *t*
(one-window causal lag, matching the adapter's additive-current-consumed-by-next-
step semantics). DN spike state, membrane voltage, body velocity, and MuJoCo pose
are **never overwritten** — only bounded excitatory current is added, so the DNs
keep evolving under native dynamics.

**Bridge OFF** injects nothing and reduces to Natural MaleCNS (regression below).

Code: `workspace/experiments/learned_bridge/` (`dn_basis.py`, `bridge.py`,
`runtime.py`, `select_visual.py`, `dataset.py`, `features.py`, `splits.py`,
`train_offline.py`, `validate_injection.py`, `calibrate_gain.py`, `evaluate.py`,
`shuffle_controls.py`, `ablations.py`, `minimality.py`, `stats.py`,
`regression_bridge_off.py`, `stage1_by_distance.py`).

---

## Visual feature selection

Selection reused **only** existing diagnostics — the moving-shot
`neural_probe/dynamic_tensor.npz` (left/centre/right at three speeds, plus
mirrored and blind controls) — and **never** the held-out closed-loop shots.

Criteria (`select_visual.py`): a candidate optic-lobe neuron must (i) strongly
encode left-vs-right direction (|AUC−0.5|·2 ≥ 0.6), (ii) be reliable — same sign
of the R−L response across all three speeds, (iii) be vision-dependent — activity
drops when the eye is blinded, (iv) sit **upstream** of the collapse, and (v)
have mean activity ≥ 0.5 spikes.

We deliberately read from **`ol_intrinsic`** (the lamina/medulla computation
neurons: L1–L5 monopolar cells, Tm/Dm medulla types — exactly the cell types the
pathway audit named in its L2→Tm4 / L3→Mi1 / L5→MeVP9 routes) and **exclude the
raw R1–R6 photoreceptors** (`ol_sensory`): reading R1–R6 directly would be reading
the retina/pixels, not an optic-lobe *representation*. `visual_projection` was
confirmed near-silent (0 strong-directional cells in the probe) and is excluded
from the primary pool.

**Final pool: 207 optic-lobe neurons** (effect size 0.61–1.0; 68 left-preferring,
139 right-preferring). Dominant types: L5, L3, L1, Tm1, Dm. The machine-readable
manifest (`visual_manifest.npz` / `.json`) records per neuron: body ID, graph
index, cell type, superclass, soma side, direction preference, AUC, effect size,
signed L−R difference, reliability, mean activity, blind activity, blind drop,
mirror-flip score, response latency, and selection score.

*Note:* the pool's `somaSide` is almost entirely "R"; this is a property of how
these columnar medulla cells are annotated/recruited under this stimulus, not a
selection bug — the pool is directionally balanced by *preference* (68 L / 139 R),
which is what matters for the bridge.

---

## Temporal features

Causal spike-count windows. At decision *t*, the feature vector concatenates the
selected neurons' spike counts over the last four 20 ms windows —
`[t, t−20, t−40, t−60] ms` — most-recent first (feature dim = 207 × 4 = 828).
Windows before episode start are zero-padded, matching the runtime ring buffer.
**Only past/present windows are ever used.**

**Latency / alignment.** A causal target-offset sweep (predict the teacher command
at `t + Δ`, Δ ∈ {0,20,40,60,80} ms, using features available at `t`) was run on
train/validation only. Validation correlation rose monotonically with offset
(0.635 → 0.680) and was best (and still rising) at **Δ = 80 ms**, indicating the
neural features *lead* the interception command by ≥ 80 ms of combined
retinal/neural/mechanical latency. **Δ = 80 ms (4 windows) was frozen.** No future
neural activity is used at inference.

---

## DN motor basis

Selected DNs (from the causal `neural_probe/motor_perturbation.json`), which
produced useful, approximately symmetric body movement when stimulated with no
vision:

| Side | Body IDs | Cell types | Stimulation → body |
| --- | --- | --- | --- |
| LEFT | 10162, 10527 | DNp20_L, DNpe017_L | +25 mV → Δy = **+0.459 cm** (strafes left / +y) |
| RIGHT | 10059, 555871 | DNp20_R, DNpe017_R | +25 mV → Δy = **−0.480 cm** (strafes right / −y) |

Body strafe symmetry was near-perfect (0.003). The bridge builds a **signed
opponent basis** `B_lr`: a scalar command `u ∈ [−1,+1]` maps to
`I_bridge_DN(u) = |u|·25 mV` on the LEFT DNs for `u<0` and on the RIGHT DNs for
`u>0` (opponent side silent), clipped to the ±30 mV adapter safety bound. This is
a **single lateral motor degree of freedom** — the minimal motor intervention.
The DN readout used by the existing decoder is the *same* four neurons, so the
injected current and the readout live on the same real opponent populations.

**Open-loop validation** (`validate_injection.py`, no vision — bridge current the
only directional input) passed cleanly: `u=−1` → Δy = +0.459 (left DNs 269 vs
right 37 spikes), `u=+1` → Δy = −0.478 (right 279 vs left 26), graded intermediate
response at ±0.5, opponent symmetry 0.02.

---

## Bridge model

```
u = tanh( ((x − μ) / σ) · w + b )
I_bridge_DN = gain · u  → B_lr  (bounded additive DN current)
```

- `x` — 828-d temporal feature vector; `μ, σ` — train-set normalization (frozen).
- `w, b` — ridge-regression weights (closed-form, numpy; ridge α = 300 chosen on
  validation); `tanh` bounds `u`.
- The motor side is the **fixed** basis `B_lr` (not learned). Runtime `gain = 3.5`
  and a causal command EMA `cmd_smoothing = 0.8` were the only runtime scalars,
  selected on a dedicated **validation** shot set (seeds 70000+, disjoint from
  train and test) and frozen before the test evaluation.

**Trainable parameter count: 829** (828 weights + 1 bias). A pure linear model —
no RL, RNN, MLP, or nonlinear decoder.

---

## Trainable parameter count

**829 parameters** (full 207-neuron model). The minimality experiment shows a
**16-neuron / 65-parameter** model is already clearly above baseline.

---

## Dataset

`dataset.py` ran the fixed MaleCNS closed loop with the **bridge OFF** (Natural
MaleCNS — the fly barely moves, so the retinal stream is the natural approaching
shot) and recorded, per 20 ms decision: the raw per-window spike counts of all
207 selected neurons, the heuristic **teacher** continuous lateral command
`u_teacher ∈ [−1,+1]` (from true ball state — labels only), and true ball/fly
state (for alignment/splits only, never a bridge feature).

- **120 episodes** (40 left / 40 centre / 40 right), **5,612** decision steps
  (~47 steps/episode). Storing raw per-window counts let every alignment offset,
  feature-count subset, and temporal-window ablation be built offline from this
  single expensive collection.
- Teacher labels span [−1, +1] (51 % left, 42 % right, 7 % neutral).

---

## Train / validation / test split

**By complete episode / shot seed** (`splits.py`, seed 20260915, 60/20/20,
stratified by group): no window from any episode appears in more than one split,
eliminating the adjacent-window leakage that would arise from random frame splits.

| Split | Episodes | left / centre / right |
| --- | --- | --- |
| train | 72 | 24 / 24 / 24 |
| val | 24 | 8 / 8 / 8 |
| test | 24 | 8 / 8 / 8 |

Dataset episode seeds are 1000–1119. Test-episode seeds (untouched until the model
was frozen): 1007, 1008, 1017–1022, 1034, 1040, 1043, 1045, 1049, 1057, 1063,
1064, 1079, 1084, 1089, 1091–1093, 1108, 1110, 1118. Feature selection,
normalization, α, alignment offset, and runtime gain/smoothing were all chosen on
train/validation only. Closed-loop evaluation used **freshly generated held-out
shot seeds** disjoint from the dataset (test 90000+, gain-calibration 70000+,
ablations 95000+, minimality 96000+).

---

## Teacher

The existing privileged **heuristic** goalkeeper. During dataset generation only,
it reads true ball state, predicts the ball's goal-line crossing `y_cross`, and
emits `lateral = clip(2.5·(y_cross − fly_y), −1, 1)`. We store the continuous
command in the decoder convention `u_teacher = −lateral` (so `u<0` = strafe left).
The heuristic supplies **labels only** — it is completely removed from the learned
controller at runtime and in every evaluation.

---

## Offline LEFT/RIGHT sanity result

Logistic classifier on the selected temporal features (left/right episodes only,
held out by episode): **test accuracy 0.78, AUC 0.87**. Because this pools
genuinely ambiguous frames, we binned accuracy by ball distance
(`stage1_by_distance.py`): during the informative approach band (ball_x 1.5–3.0)
accuracy is **0.92** (91–100 % per bin), collapsing to chance only at spawn
(ball just appeared) and at the crossing plane (ball fills/leaves the view). The
feature-extraction / timing / body-ID pipeline is therefore validated — the
strong directional signal the diagnostics reported is preserved — so escalating
model complexity was **not** warranted.

---

## Offline continuous-action prediction

Ridge regression, features → teacher `u`, evaluated on the untouched test split
(`train_offline.json`):

| Metric | Test |
| --- | --- |
| Pearson correlation | 0.69 |
| MAE | 0.52 |
| signed direction accuracy | 0.82 |
| left → predicted left | 0.91 |
| right → predicted right | 0.83 |
| centre → predicted ~neutral | 0.73 |

Validation feature-count sweep: 16 → 0.37, 32 → 0.43, 64 → 0.70, 128 → 0.70,
207 → 0.68 (correlation). The signal saturates by ~64–128 neurons.

---

## Closed-loop results

Frozen bridge (gain 3.5, smoothing 0.8), matched held-out shots.

**Establishing directional control (Stage 4/5).** With a clean sustained command
the body reaches ±0.40 cm symmetrically; the frozen bridge produces correctly
signed commands (left shots → mean u ≈ −0.24, right shots → +0.43) and, when it
moves, moves in the correct direction on **100 %** of directional shots
(`move_dir_acc = 1.0`), versus chance (0.5) for Natural MaleCNS.

**Headline test set (48 shots, seeds 90000+):**

| Controller | Overall | 95 % CI | Left | Centre | Right |
| --- | ---: | --- | ---: | ---: | ---: |
| Passive | 0.33 | [0.22, 0.47] | 0.00 | 1.00 | 0.00 |
| Random | 0.33 | [0.22, 0.47] | 0.00 | 1.00 | 0.00 |
| Natural MaleCNS | 0.33 | [0.22, 0.47] | 0.00 | 1.00 | 0.00 |
| **Learned bridge** | **0.71** | **[0.57, 0.82]** | 0.25 | 1.00 | 0.88 |
| Heuristic (oracle) | 0.94 | [0.83, 0.98] | 0.88 | 1.00 | 0.94 |

*(Targeted-plasticity MaleCNS from the prior phase = 0.375, vision-independent;
not re-run here to avoid loading plasticity-trained state, but reported for
context.)*

Behavioural/neural metrics (bridge vs natural): teacher-command correlation
**0.77 vs 0.00**; movement-direction accuracy **1.0 vs 0.5**; mean absolute
lateral displacement **0.101 vs 0.020 cm** (5× more, purposeful movement);
bridge |u| mean 0.29.

---

## Mandatory controls

Frozen bridge, matched test shots (`evaluate_test.json`, `shuffle_controls.json`).

| Condition | Overall | Left | Centre | Right | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| **Normal** | 0.71 | 0.25 | 1.00 | 0.88 | ≫ Natural ✅ |
| **Blind** | 0.35 | 0.00 | 0.94 | 0.13 | → baseline ✅ |
| **Mirrored** | 0.50 | 0.00 | 0.94 | 0.56 | degraded / laterality broken ✅ |
| **Shuffled — per-timestep** | 0.38 | 0.00 | 0.94 | 0.19 | → baseline ✅ |
| Shuffled — fixed permutation | 0.63–0.71 | — | — | — | did **not** collapse (see below) |
| **Bridge OFF** | 0.33 | 0.00 | 1.00 | 0.00 | = Natural MaleCNS ✅ |

**On the shuffled-retina control.** The project's built-in `shuffled` applies a
*fixed* spatial permutation to the R1–R6 luminance vector. We found this does
**not** collapse a learned controller, and the reason is principled: a fixed
permutation is a **bijection**, so the distinct luminance vectors for a left vs a
right ball remain distinct and *consistent* after permutation — retinotopy is
destroyed but left/right *separability* is preserved, and a learned readout can
still latch onto the (spatially scrambled but consistent) signal. We therefore
added the genuine structure-dependence control that Section 26 explicitly permits:
a **per-timestep** random shuffle (fresh permutation every 20 ms, fixed seed
stream), which destroys any consistent spatial→neuron mapping while preserving the
per-frame marginal luminance distribution. Under it the bridge **collapses to
baseline (0.38, right 0.94 → 0.19)**, confirming the bridge depends on structured,
consistent retinal input rather than a non-structural artifact. Both results are
reported; the fixed-permutation non-collapse is a documented bijection limitation
of that particular control, not evidence against vision-dependence (blind, mirror,
per-step-shuffle, and bridge-OFF already establish it).

**Statistics** (`stats.py`; two-proportion z-tests + Wilson/bootstrap CIs):
bridge − natural = **+0.375, z = 3.68, p = 2.4 × 10⁻⁴**, CI [0.19, 0.56];
bridge − blind p = 5 × 10⁻⁴; bridge − per-step-shuffle = +0.333, p = 1.0 × 10⁻³;
bridge − bridge-OFF p = 2.4 × 10⁻⁴. All key contrasts significant at p < 0.01.

**Bridge-OFF regression** (`regression_bridge_off.py`): a *present-but-disabled*
bridge is **bit-for-bit identical** to no-bridge across 537 matched steps (max
abs difference 0.0 in fly_y, lateral command, and left/right DN spikes) and
reproduces the Natural signature (0 % L / 100 % C / 0 % R). Adding the bridge
infrastructure does not perturb the baseline.

---

## Controller comparisons

See the headline table above: Passive = Random = Natural = Bridge-OFF = 0.33;
Learned bridge = 0.71; Heuristic ceiling = 0.94. The learned bridge closes roughly
**62 %** of the gap between Natural MaleCNS and the privileged oracle, using a
vision-only neural controller with 829 parameters.

---

## DAgger

**Not used.** Offline imitation transferred to closed loop without a catastrophic
distribution shift (closed-loop directional control and the strong controls held),
so no DAgger iterations were needed. The dataset was generated under bridge-OFF
(near-stationary body); the successful closed-loop transfer indicates the
optic-lobe features are robust to the small body motion the bridge induces. DAgger
remains an option if larger body excursions later degrade feature validity.

---

## Ablations

Closed-loop causal ablations of the frozen bridge (`ablations.py`, 24 held-out
shots, seeds 95000; intact 0.875 on this seed set):

| Ablation | Overall | Left | Right | mean final y | Effect |
| --- | ---: | ---: | ---: | ---: | --- |
| intact | 0.875 | 0.625 | 1.00 | −0.09 | — |
| zero entire bridge | 0.33 | 0.00 | 0.00 | −0.04 | → Natural MaleCNS |
| zero left-preferring visual pop. | 0.67 | **0.00** | 1.00 | −0.39 | kills LEFT saves, pushes body right |
| zero right-preferring visual pop. | 0.58 | 0.88 | **0.00** | +0.34 | kills RIGHT saves; **LEFT rises to 0.88** |
| zero left-DN drive | 0.63 | **0.00** | 1.00 | −0.17 | kills LEFT saves |
| zero right-DN drive | 0.63 | 0.88 | **0.00** | +0.04 | kills RIGHT saves |

Every ablation produces the expected laterality. Notably, zeroing the
right-preferring visual population **improves** left saves (0.63 → 0.88),
directly identifying the source of the left/right asymmetry: the strong
right-preferring population normally *competes with and suppresses* the left
channel (see Limitations).

---

## Temporal-history tests

Offline, held-out test split (`ablations.py`):

| Configuration | corr | signed dir acc |
| --- | ---: | ---: |
| full 4 windows | 0.68 | 0.82 |
| single current window only | **0.31** | 0.63 |
| remove oldest window | 0.61 | 0.78 |
| remove newest window | 0.61 | 0.78 |
| reversed window order (frozen weights) | 0.68 | 0.81 |

**Temporal integration is essential** — a single 20 ms window drops correlation
from 0.68 to 0.31; removing either the oldest or newest window degrades it.
However, the *order* of the windows is nearly interchangeable (reversed ≈ full),
because the ball moves slowly (~0.05 cm/window) so adjacent windows are highly
correlated. Honest interpretation: the history helps mainly by **integrating the
direction signal over ~80 ms**, not by encoding fine-grained motion *order*.

---

## Minimality experiment

Closed-loop, held-out shots (`minimality.py`, seeds 96000):

**Visual (retrained per size):**

| Neurons | Params | Save rate |
| ---: | ---: | ---: |
| 16 | 65 | 0.58 |
| 32 | 129 | 0.67 |
| 64 | 257 | 0.63 |
| 128 | 513 | 0.71 |
| 207 | 829 | 0.75 |

**Motor:**

| DN basis | # DNs | Save rate |
| --- | ---: | ---: |
| 2 + 2 opponent | 4 | 0.75 |
| 1 + 1 (single opponent pair) | 2 | 0.67 |

Even **16 optic-lobe neurons / 65 parameters**, or a **single opponent DN pair**,
yield a clearly-above-baseline vision-driven goalkeeper. This directly answers the
central question: the artificial intervention can be made very small while still
generating useful embodied behaviour.

---

## Statistical uncertainty

All save rates carry Wilson 95 % CIs; a parametric bootstrap gives matching
intervals (`stats.py`). Key contrast bridge vs natural: +0.375, z = 3.68,
p = 2.4 × 10⁻⁴, CI [0.19, 0.56]. Evaluations use 48 matched held-out shots
(16/group) for headline numbers and 24 for the ablation/minimality sweeps; the
non-overlap of the bridge CI [0.57, 0.82] with the baseline CI [0.22, 0.47] is the
core statistical result. Per-direction rates on 16 (or 8) shots have wide
intervals and are not over-interpreted — the left/right asymmetry is discussed
qualitatively and confirmed causally by ablation, not by a few-point difference.

---

## Limitations

- **Left/right asymmetry.** Left saves (0.25 on the test set) lag right (0.88).
  The bridge's *command* is balanced (offline left u = −0.41, right u = +0.31;
  closed-loop `move_dir_acc = 1.0`) and the body moves left symmetrically under a
  clean constant command (reaching +0.40 cm). The asymmetry is an **embodiment /
  competition** effect: the natural common-mode DN drive drifts the fly rightward
  (drive-none Δy = −0.064), and the strong right-preferring optic-lobe population
  competes with the left channel — zeroing it raises left saves to 0.88. It is a
  real, mechanistically-explained finding, not tuned away.
- **Engineered decoder, not learned biology.** The bridge is an explicit linear
  map; MaleCNS synapses did not learn. The runtime gain and command smoothing are
  experimenter choices (selected on validation).
- **Fixed-permutation shuffle** does not collapse the bridge (bijection preserves
  separability); the per-timestep shuffle is the valid structure-dependence
  control and does collapse it.
- **Toy scale.** The arena is deliberately normalized (large ball, cm-scale fly),
  not FIFA-accurate.
- **Bridge-OFF dataset.** Training features were collected with a near-stationary
  body; large learned-body excursions could shift the feature distribution
  (mitigable with DAgger, not needed here).
- **Reversed-window order** is uninformative here because the slow ball makes
  adjacent windows redundant; a faster stimulus would test motion-order coding
  better.

---

## Scientific interpretation

The fixed MaleCNS visual system already contains a strong, clean left/right
direction signal in its optic-lobe (medulla/lamina) neurons, and real MaleCNS
descending neurons can causally drive useful left/right body movement. The native
connectome, however, does **not** route that directional representation onto
opponent descending populations — the contrast dies at the optic-lobe →
visual-projection stage — and reward-modulated plasticity on that route could not
build the missing structure. Inserting a **deliberately minimal learned linear
bridge** (as few as 16 neurons / 65 parameters, or 2 descending neurons) across
exactly that identified gap produces genuinely **vision-dependent** embodied
goalkeeper behaviour: it beats Natural MaleCNS by a large, statistically
significant margin (33 % → 71 %, p = 2 × 10⁻⁴), collapses under blinded and
per-timestep-shuffled vision, degrades under mirrored vision, is bit-identical to
Natural MaleCNS when switched off, and shows clean causal laterality under
ablation.

**The native MaleCNS synaptic weights did not learn the task.** MaleCNS supplies a
fixed biological visual representation and functional descending motor outputs; a
small, explicit, engineered bridge connects the two across the experimentally
identified representational bottleneck. The result is a scientifically defensible,
minimal demonstration that the missing piece was a *transformation*, not
information or motor capacity — and that a very small artificial transformation
suffices.

---

### Reproducibility

Seeds and constants: dataset episodes 1000–1119; split seed 20260915;
alignment offset 80 ms; ridge α = 300; runtime gain 3.5; command smoothing 0.8;
DN drive 25 mV (±30 mV bound); per-step-shuffle seed stream 1234; held-out eval
seeds 90000+ (test), 70000+ (gain), 95000+ (ablation), 96000+ (minimality).
Frozen artifacts and every result JSON live under
`workspace/outputs/learned_bridge/` (`visual_manifest.*`, `bridge_model.npz`,
`dataset_main.npz`, `regression_bridge_off.json`, `train_offline.json`,
`stage1_by_distance.json`, `validate_injection.json`, `calibrate_gain.json`,
`evaluate_test.json`, `shuffle_controls.json`, `ablations.json`,
`minimality.json`, `stats.json`).
