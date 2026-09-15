# Neural diagnostics report — where does shot direction live in MaleCNS?

This phase did not touch the MaleCNS scientific core, the physics, or the body.
The embodied goalkeeper was frozen (`git tag embodied-flykeeper-baseline`) and
used as a fixed benchmark. All experiments live under
`workspace/experiments/neural_probe/`; artifacts under
`workspace/outputs/neural_probe/`. No plasticity was added.

**Central question:** is information about the incoming football represented
inside MaleCNS, where does it exist, and can it be connected to useful body
movement?

**Short answer:** shot direction is represented *strongly and cleanly* in the
retina and optic lobe, an anatomical pathway to descending neurons exists, and
the descending neurons can drive the body correctly. But under fair,
causally-usable, closed-loop conditions the fixed network does **not** deliver
usable directional drive to the descending neurons. The information is lost (as
usable signal) between the optic lobe and the descending output. **Decision-tree
outcome: B.**

---

## 0. Passive baseline — MaleCNS is standing still

A true never-moves controller was added and run on the same deterministic shots.

| Controller | Save rate | Left | Center | Right |
| --- | ---: | ---: | ---: | ---: |
| passive (never moves) | 30 % | 0 % | 100 % | 0 % |
| random | 30 % | 0 % | 100 % | 0 % |
| MaleCNS (DNp20, original) | 30 % | 0 % | 100 % | 0 % |
| heuristic (true ball state) | 100 % | 100 % | 100 % | 100 % |

MaleCNS is *numerically identical* to passive: every "save" is a centre shot
blocked by the body simply being there. Confirmed: the DNp20 decoder extracts
nothing useful. (The heuristic proves the body/environment can save everything.)

## 1. Method — probing perception with a fixed body

`neural_probe/probe.py` presents controlled visual stimuli with the body held
fixed (open loop) and records sparse per-neuron spike counts for all 166,700
neurons across 5 ms windows, preserving biological body IDs. Two stimulus sets:

- **static**: a bright ball frozen at left / centre / right / strong-L / strong-R
  positions, plus mirrored, blind, and shuffled controls.
- **dynamic**: the *actual moving-ball retinal sequence* the embodied fly would
  see (ball approaching from 3.2 cm at 3–8 cm/s), left / centre / right, fast /
  slow, mirrored, blind. This is the realistic case.

Data are stored as compact tensors (`static_tensor.npz`, `dynamic_tensor.npz`).

**Metric correction.** The first screen used a naive AUC that mislabels a
*constant* response as perfectly discriminative (AUC 0/1). `neural_probe/metrics.py`
fixes this: AUC is Mann-Whitney with tie mid-ranks (constant → 0.5), and a neuron
is "directional" only if `|Cohen's d| ≥ 0.5` AND `|AUC−0.5| ≥ 0.1` AND it is
active. All tables below use the corrected metric.

## 2. Sensory hierarchy — where the signal lives (and dies)

Cross-validated left-vs-right temporal decoding accuracy per stage
(`neural_probe/temporal.py`; leave-one-trial-out, analysis-only):

| Stage | static (≤120 ms) | dynamic / moving (≤120 ms) |
| --- | --- | --- |
| retina (R1–R6, ol_sensory) | **1.00** at every window | **1.00** at every window |
| optic lobe (ol_intrinsic) | **1.00** at every window | **1.00** at every window |
| visual projection neurons | 0.58–0.67 | 0.50 (chance) |
| central brain | 0.50 | 1.00 at ≥50 ms* |
| descending neurons | 0.92 at ≥50 ms* | 1.00 at ≥100 ms* |
| VNC / motor | 0.50 | 0.50 |

By the corrected *summed-count* metric, directional neurons number 3,038
(retina) + 6,153 (optic lobe) but **0** among descending neurons — the total
spike count of every DN is at chance (AUC 0.5).

\*The apparent 0.92–1.00 decode at central/descending stages is a **timing**
effect on aligned trials, and it does not survive the closed-loop test (§6).
Treat it as an upper bound from idealised, trial-aligned analysis, not as usable
signal.

**First stage where usable directional information disappears: between the optic
lobe and the descending neurons.** The retina and optic lobe carry it perfectly;
by the descending stage only 4 neurons respond at all and their rate code is at
chance.

## 3. Descending-neuron screen (all 1,332 DNs, not just DNp20)

`neural_probe/broad_dn_screen.py`: of the **1,332** descending neurons in the
graph, only **4** fire at all under visual drive — DNp20 (10059 R, 10162 L) and
DNpe017 (10527 L, 555871 R). The other 1,328 are silent. By total count all four
have AUC ≈ 0.5 (no direction).

## 4. Connectivity — the pathway exists

`neural_probe/trace_connectivity.py` (BFS over the prepared CSR graph): from the
40 strongest directional visual neurons, **all 1,332 descending neurons are
reachable** — 273 at 2 hops, 1,051 at 3, 8 at 4 (median 3). There are **zero
monosynaptic** visual→DN edges; the route is polysynaptic. The 4 responsive DNs
sit exactly **2 hops** from directional visual neurons.

So this is **not** "no anatomical pathway" (which would be Outcome C/A-no-path).
The wiring exists; the fixed dynamics simply do not propagate the directional
signal through it as usable drive.

## 5. Motor side — descending neurons *can* drive the body

`neural_probe/motor_perturbation.py`:

- **Locomotion-bridge symmetry** (direct commands, no brain): strafe left
  +0.503 cm vs right −0.506 cm (asymmetry 0.003); turn left +1.60 vs right
  −1.65 rad/s (asymmetry 0.015). The body is unbiased.
- **Descending perturbation** (inject 25 mV current into DN populations, no
  vision): driving the **left** DNs moves the fly **+0.46 cm (left)** with the
  decoder emitting LEFT on every frame; driving the **right** DNs moves it
  **−0.48 cm (right)**, RIGHT every frame; no drive → stays put.

So the DN→decoder→locomotion→body chain works correctly and symmetrically. The
motor side is not the problem.

## 6. Candidate decoder — justified by analysis, rejected by the control

The open-loop analysis suggested the DN signal is temporal, so an interpretable
causal readout was tested (`neural_probe/candidate_decoder_test.py`): a leaky
integral of the (right−left) DN spike contrast separated left from right at
AUC 0/1, p = 0.0005 on 20 aligned trials/side — while the non-leaky cumulative
count was at chance. That passed the pre-registered gate, so a
`LeakyIntegratorDecoder` was built (`embodiment/motor_decoder.py`, fully
documented, no ball state, no trained classifier) and A/B tested in the closed
loop (`neural_probe/closed_loop_suite.py`, 30 episodes):

| Controller / condition | Save rate | Left | Centre | Right |
| --- | ---: | ---: | ---: | ---: |
| MaleCNS DNp20 / normal | 30 % | 0 % | 100 % | 0 % |
| MaleCNS DNp20 / blind | 30 % | 0 % | 100 % | 0 % |
| MaleCNS leaky / normal | 50 % | 0 % | 89 % | **54 %** |
| MaleCNS leaky / **blind** | 53 % | 0 % | 100 % | **54 %** |

The leaky decoder appears to lift the save rate — but its right-shot performance
is **identical with the eye blinded**. Its movement (≈ 720 RIGHT vs ≈ 25 LEFT
commands) is a constant rightward strafe produced by a fixed baseline firing
asymmetry (DNp20_R fires slightly more than DNp20_L even in the dark), which
happens to intercept right shots. Adding baseline subtraction removes the lift
and leaves normal ≈ blind. **The "improvement" is not vision-driven.** The
decoder was therefore rejected; DNp20 remains the documented default.

This is the crucial result: the descending timing "signal" from §2 exists only
under trial-aligned open-loop analysis and is dominated by baseline asymmetry.
It is **not extractable by any interpretable causal readout in the closed loop**,
where shot onset is not aligned to a fixed analysis window.

## 7. Success criterion (from the brief)

The brief set a scientific success bar independent of save rate: left shot →
reliably more leftward response, right → rightward, mirror → reverse, blind →
disappears. **This bar is met at the retina and optic lobe** (100 % L/R decode,
and mirroring flips the responding cells) but is **not met at the descending /
motor stage** (blind = normal in closed loop). So genuine visually-driven
sensorimotor *behaviour* does not yet exist; genuine visually-driven *early
neural representation* clearly does.

## 8. Diagnosis — Outcome B

- **A. No anatomical pathway** — ruled out (all DNs reachable in 2–3 hops).
- **B. Direction reaches central/intermediate neurons but not usably to
  descending neurons despite an existing pathway** — **this is the outcome.**
  Retina/optic lobe: perfect. Visual projection neurons: already weak
  (0.5–0.67, and 0.5 for moving shots). Descending: 4/1,332 active, rate code at
  chance, any residual signal is timing/baseline and not causally decodable.
- **C. Only near the visual input** — partially true (signal is strongest and
  only robust there) but the pathway does exist, so B is the precise label.
- **D. No directional info even in retina** — firmly ruled out.
- **E. Good DN signal, body fails** — ruled out; the body works (§5).

**Exact location where usable information is lost: the optic-lobe →
visual-projection → descending transformation.** The connectome route exists but
the fixed LIF dynamics collapse the directional code there, leaving only 4
weakly, non-directionally active descending neurons.

## 9. Recommendation for the next step

Not "plasticity by default." The diagnosis pinpoints a specific, narrow target:

1. **First, sensory-mapping / drive check (cheap, no learning).** Only 4 of
   1,332 DNs fire at all — the visual drive barely reaches the descending stage.
   Before any learning, verify the retinal encoding and drive levels actually
   recruit the visual-projection and pre-descending populations (the probe shows
   visual-projection decoding is already near chance for moving shots). If a
   modest, documented change to the sensory encoding restores a directional
   signal at the visual-projection stage, re-run the DN screen. This may resolve
   the bottleneck without plasticity.

2. **If the signal still dies before the DNs, then targeted plasticity is
   justified — but scoped to the visual→descending mapping, not whole-brain.**
   Because (a) the information is present and clean upstream, (b) an anatomical
   2–3 hop route exists, and (c) the DNs can drive the body correctly, the
   missing piece is specifically the transformation that should route
   optic-lobe direction onto opponent descending populations. A reward-modulated
   plasticity rule restricted to the edges on those 2–3 hop paths (SAVE→reward,
   GOAL→negative) is the appropriate, minimal intervention. Keep the retinal
   pathway, LIF core, and body fixed.

3. **Do not** pursue better motor decoding or body work — those are already
   sufficient (heuristic 98–100 %, DN→body verified).

---

### Files
- Experiments: `workspace/experiments/neural_probe/{probe,collect_data,metrics,temporal,hierarchy,trace_connectivity,broad_dn_screen,motor_perturbation,validate_descending,candidate_decoder_test,closed_loop_suite}.py`
- Candidate decoder: `workspace/embodiment/motor_decoder.py::LeakyIntegratorDecoder` (rejected; `--decoder leaky` in the runner)
- Passive controller: `workspace/experiments/embodied_flykeeper/controllers.py::PassiveController`
- Artifacts: `workspace/outputs/neural_probe/*.json`, `*_tensor.npz`
