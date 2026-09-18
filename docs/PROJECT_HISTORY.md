# Fruit Fly Goalkeeper — end-of-project history

**Repository:** `MrMystery8/fly_keeper`  
**Project window represented by Git:** 15–19 September 2026 (PKT)  
**Final project commit before this documentation update:** `133a3db` (`Polish arcade goalkeeper presentation`)  
**Final frozen champion:** `arcade-early-intent-final-champion` / commit `91ac824`

This is the canonical chronological account of how the project evolved. It is
grounded in the 47 commits on the repository's final line of development,
its preserved side branch, Git tags, frozen artifacts, and the linked phase
reports. The phase reports are historical evidence and are intentionally not
rewritten here.

## Executive conclusion

The project began as an attempt to make a fixed, full MaleCNS fruit-fly
connectome play goalkeeper through its own visual input and real descending
neurons (DNs). That natural controller did not solve the task: it saved only
centre shots by standing still. The body and motor side worked, and early
visual populations contained direction, but directional information collapsed
before it reached the useful DNs.

Two direct remedies were tested and rejected with controls: a causal decoder
was not vision-driven in closed loop, and targeted reward-modulated plasticity
strengthened common-mode activity without creating left/right structure. A
small learned visual-to-DN bridge then demonstrated genuine vision-dependent
goalkeeping in the original embodied setting (33% to 71% on a held-out
48-shot test). The subsequent arcade setting exposed data-integrity and
continuous-control limits, leading to the final **WHERE/HOW early-intent
architecture**: an initial stationary sensory window decides *where* the shot
is headed; an engineered action policy decides *how* to intercept it through
the same real-DN/MuJoCo pathway.

The final untouched 90-shot battery saved **61/90 (67.8%)**, versus **21/90
(23.3%)** for the continuous monocular v2 baseline and **85/90 (94.4%)** for
the privileged physical oracle. Bilateral eye ablations reduced the champion
to 10.0% (left blind), 25.6% (right blind), and 8.9% (both blind), supporting
the claim that its behavior depends on the intended visual configuration.

> This is not a digital fly or a validated account of fly cognition. The
> connectome is measured MaleCNS connectivity; the retinal encoder, LIF
> integration interface, body, locomotion, motor decoder, learned bridge, and
> final policy are engineered and are described as such throughout.

## Architecture at the finish line

```
two simulated eyes → retinal mosaics → fixed 166,700-neuron MaleCNS graph
                                         │
                                         └→ real DN activity
                                                │
stationary 160-ms sensing window → EarlyIntentEstimator (WHERE)
                                                │
15-dimensional non-privileged observation → frozen RL/action policy (HOW)
                                                │
                                         DN injection → 2-axis fly body → MuJoCo save
```

The project deliberately retained the immutable biological substrate while
moving learning to explicitly bounded engineered interfaces. Earlier science
mode used a 829-parameter linear learned bridge from selected optic-lobe
features to a fixed DN opponent basis. The final arcade champion instead uses
bilateral retinal evidence during an eight-decision (about 160 ms) sensing
window and an engineered early-intent controller. Neither result claims that
the native connectome's synapses learned football.

| Component | Status at project end |
| --- | --- |
| MaleCNS connectivity and native fixed weights | Preserved; never overwritten by training |
| CPU backend | Scientific reference / authoritative for exact-parity work |
| Metal backend | Optional, relaxed throughput experiment; not exact closed-loop parity |
| MuJoCo body, vision bridge, motor decoder | Engineered embodiment interface |
| Learned visual→DN bridge | Frozen, validated science-mode intervention (829 parameters) |
| Early-intent estimator and action policy | Engineered final arcade controller |

## Chronological development record

### 1. Deterministic reference and Metal exploration — 15 September

**Goal.** Preserve a trustworthy full-graph reference while investigating a
faster Apple Metal implementation.

**What changed.** Commit `978f57c` froze the deterministic Metal v3
radix-sort foundation. `b55962b` documented serial bridge/toolchain identity,
and `dce319b` added deterministic work-list primitives. `8c93153`
checkpointed the continuing Metal v4–v6 work on branch
`metal-deterministic-v3` before the project moved into embodiment.

**What worked.** The CPU/native reference remained reproducible; serial Metal
served as a test oracle. The selected relaxed v5 Metal path achieved a
100-episode FlyKeeper run in 26.16 s versus 48.43 s on CPU (1.85× end-to-end)
on the recorded M4 benchmark.

**What did not become the scientific backend.** Relaxed parallel propagation
changed closed-loop behavior: in a 100-episode comparison, CPU saved 26 and
Metal saved 25, with 9 terminal outcomes and 728/1,400 actions differing.
Metal's left-action collapse was behaviorally material. The project therefore
kept CPU as the reference and treated Metal as a throughput variant, not a
drop-in truth-preserving replacement.

**Evidence.** Tag `metal-v3-foundation-2026-09-15`; branch
`metal-deterministic-v3`; [Metal backend report](../workspace/gpu/METAL_BACKEND_REPORT.md).

### 2. Embodied fixed-connectome baseline — 15 September

**Goal.** Put the real graph in a physically interpretable goalkeeper loop
without changing its weights.

**What changed.** `fa4c2ea` added the fly-body CPG, MuJoCo goalkeeper world,
retinal vision bridge, and motor decoder. `9f87f4e` added stable strafe control
and a runner; `5adeb2d` added an MP4 demo and evaluation suite; `1074221`
recorded the measured baseline. The phase was frozen by `0c152ee`.

**What worked.** The body could move, contact the ball, and run headlessly.
A privileged heuristic demonstrated that the physical setup was capable of
strong performance (reported heuristic ceiling 83% at this stage; later
arcade-oracle measurements differ because the environment changed).

**What failed.** Natural MaleCNS saved about 33%: 0% left, 100% centre, 0%
right. The apparent centre success was simply standing still, not visual
interception. This established the baseline against which all later claims
were evaluated.

**Evidence.** Tag `embodied-flykeeper-baseline`; [embodied report](reports/EMBODIED_REPORT.md).

### 3. Neural diagnostics — 15 September

**Goal.** Determine whether failure was sensory, motor, or an internal routing
problem before adding a learned controller.

**What changed.** `8fdb8e8` added a passive baseline; `a5c237f` added static
and dynamic probes, temporal decoding, connectivity tracing, and DN
validation. `63b4eb1` tested DN-to-body perturbations and broad DN screening.
`d1c0f4a` built an interpretable leaky-integrator candidate; `8b0e286` tested
it in closed loop. `27d6de2` froze the diagnosis.

**What worked.** Retinal/early visual activity carried left/right information;
all 1,332 DNs were considered; and DNs could causally produce useful,
symmetric body movement. Offline directional separation by the candidate
readout was excellent (reported AUC 0/1, p < 0.001).

**What failed.** The closed-loop candidate failed the decisive blind control:
its apparent improvement was a baseline asymmetry artifact, not visual
control. Offline decoding alone was therefore not accepted as evidence of a
vision-driven controller.

**Evidence.** Tag `neural-diagnostics-complete`; [neural diagnostics report](reports/NEURAL_DIAGNOSTICS_REPORT.md).

### 4. Pathway audit — 15 September

**Goal.** Locate the failure point in the real visual-to-DN route and decide
whether changing only selected biological synapses was justified.

**What changed.** `6d77f66` ranked 663 candidate visual sources to 664 DNs and
performed subthreshold probes. `27fb532` added causal transition stimulation;
`17cdd12` decomposed common-mode versus directional signal and swept gain;
`5c6f74f` swept time windows and laterality. `64a88d0` froze the finding.

**What worked.** Visual activity reached the pathway and individual
transitions could transmit. The audit gave a causal, measurable basis for a
targeted intervention rather than arbitrary whole-network training.

**What failed.** Directional modulation collapsed at the optic-lobe →
visual-projection transition. Neither gain, timing, mirror/laterality, nor
different observation windows restored meaningful deep opponent direction.

**Why the project changed direction.** A targeted visual→projection→DN
plasticity experiment was the smallest faithful attempt to repair the found
bottleneck.

**Evidence.** Tag `pathway-audit-complete`; [pathway activation report](reports/PATHWAY_ACTIVATION_REPORT.md).

### 5. Targeted reward-modulated plasticity — side branch, 15 September

**Goal.** Learn directional structure only on the audited route while keeping
the rest of the connectome intact.

**What changed.** On branch `targeted-visual-motor-plasticity`, `778b607`
built a 1,211-synapse mask (0.0047% of the graph). `38b5c44` added a bounded,
subthreshold-eligibility, reward-modulated learning engine and curriculum.
The remaining commits added boundedness tests, an eligibility diagnosis,
checkpointing, action-side differentiated credit, and blind/mirrored/shuffled
and frozen controls. `dbf42fe` froze the result.

**What worked.** Training ran cleanly, weights were checkpointed without
overwriting the connectome, and the experiment produced a strong negative
result with the controls needed to trust it.

**What failed.** Scalar reward changed transmission but not direction. About
87% of eligibility was common-mode; source→middle had limited direction and
middle→DN was tiny and nondirectional. The trained controller remained about
passive (about 37.5%) and vision-independent.

**Why the project changed direction.** The failure was representational, not
a missing reward signal or insufficient training. The next intervention moved
learning to a transparent, frozen readout bridge instead of asserting that the
native synapses could solve the task.

**Evidence.** Tag `targeted-plasticity-complete`; branch
`targeted-visual-motor-plasticity`; commits `778b607`–`dbf42fe`.

### 6. Minimal learned visual→DN bridge — 15–16 September

**Goal.** Test the smallest explicit learned interface that could turn real
visual activity into a causal DN opponent command.

**What changed.** `208d99d` selected 207 direction-selective optic-lobe
neurons and a causal DN motor basis. `28b9f0f` added the linear bridge,
injector, controller, and bit-identical bridge-off regression. `5c303e3`
collected 120 episodes / 5,612 steps and froze an 829-parameter model after
training-only feature selection. Subsequent commits validated injection,
calibrated gain/smoothing, added causal ablations and minimality tests, then
completed statistical testing in `ac88413`. `f7f0f04` froze the phase.

**What worked.** On the held-out 48-shot test, save rate improved from 0.33 to
0.71 (p = 0.0002). Blinding collapsed it to 0.35; mirroring reduced it to
0.50; per-step retinal shuffling also failed; bridge-off exactly reproduced
Natural MaleCNS. Side ablations removed the corresponding saves. The bridge
remained useful with 16 neurons / 65 parameters or a pair of DNs.

**What did not work.** Fixed native synapses still did not learn the task, and
the body retained a rightward common-mode asymmetry. These limits are part of
the result, not hidden implementation details.

**Evidence.** Tag `learned-visual-dn-bridge-complete`; [learned bridge report](reports/LEARNED_BRIDGE_REPORT.md).

### 7. Interactive science demo — 16 September

**Goal.** Make the frozen learned-bridge result inspectable and playable
without silently changing the scientific loop.

**What changed.** `ac31188` added a 3D penalty-game interface over the exact
frozen bridge, controller modes, visual conditions, integrity checks, and
regressions.

**What worked.** The interactive mode retained no privileged ball-state path
to the neural controller and exposed bridge, vision, and debug controls. It
made the scientific result easier to demonstrate while preserving the phase
tag.

**Known limit.** CPU reference simulation runs at about 0.12× real time in the
interactive loop; it is a science demo, not a real-time game.

**Evidence.** Tag `flykeeper-interactive-complete`; [interactive demo report](reports/INTERACTIVE_DEMO_REPORT.md).

### 8. Arcade redesign and stale-dataset correction — 16 September

**Goal.** Build a faster, two-axis arcade goalkeeper and verify that its
offline learning data represented genuine complete shots.

**What changed.** `7be8516` introduced Arcade Bridge v1: force-flight body,
two-axis DN bridge, corrected interception/scoring, and a 100% oracle. `35e52e1`
then identified post-resolution padding and a teacher-label issue, guaranteed
terminal resolution, rebuilt the dataset pipeline, and froze v2.

**What worked.** The correction eliminated padded terminal episodes (19/60 at
the 90-step cap in v1 to 0/36 fresh verification episodes) and raised the
honest offline lateral metric from 0.29 to 0.41. Offline/runtime inference
parity was exact for the repaired path.

**What failed.** Better offline lateral decoding did not solve the downstream
DN/body bottleneck: v2 remained right-favoring and produced only 38.9% in its
reported 3×3 closed-loop evaluation, close to v1's 41.7%.

**Evidence.** Tag `arcade-bridge-v2-complete`; [Arcade Bridge v2 report](reports/ARCADE_BRIDGE_V2_REPORT.md).

### 9. True binocular bridge — 16 September

**Goal.** Determine whether retaining both eyes could overcome the arcade
decoder bottleneck.

**What changed.** `26036bf` audited eye/receptor mappings and added a true
two-eye bridge and sanity tests. `e300260` completed matched bilateral,
monocular, and blind-eye experiments.

**What worked.** The implementation genuinely used both eyes: eye ablations
produced the expected degradation, so the result was not a disguised
one-eye controller.

**What failed.** Bilateral features did not improve the relevant outcome:
reported lateral performance was 0.34 versus v2's 0.41 and closed-loop save
rate was 25.9% versus 33.3%. More visual input did not repair the DN decoding
bottleneck.

**Why the project changed direction.** Continuous visual-to-action decoding
was too vulnerable once self-motion changed the retinal distribution.

**Evidence.** Tag `arcade-binocular-vision`; [binocular report](reports/ARCADE_BINOCULAR_REPORT.md).

### 10. Early-intent action policy and vertical refinement — 18 September

**Goal.** Decouple early visual localization from later movement control while
keeping the deployed policy free of privileged ball variables.

**What changed.** Branch `arcade-binocular-action-policy` froze a 53.7%
early-intent RL checkpoint in `0befa9a`. The final branch then used an
eight-step, stationary pre-motion retinal window to infer direction and a
15-dimensional runtime observation vector for the action policy. `aead9e8`
introduced differential vertical refinement (70.4% on the development
54-shot set). `91ac824` froze the untouched 90-shot validation and regression
suite; `133a3db` polished the presentation.

**What worked.** The final champion saved 61/90 (67.8%) on untouched seeds
600000+, with 24/30 left, 19/30 centre, and 18/30 right saves. It improved
substantially over the 23.3% v2 continuous baseline and retained physical
contact on every save. The 90-shot bilateral ablations provided strong
causal evidence that the frozen controller depends on its visual input.

**What remains limited.** The physical oracle still reached 94.4%, low shots
were the weakest class (56.7%), and eye-blind conditions are severe
out-of-distribution interventions rather than ordinary robustness tests.

**Evidence.** Tags `arcade-early-intent-rl-53pct`,
`arcade-early-intent-rl-70pct`, and `arcade-early-intent-final-champion`;
[final early-intent report](reports/EARLY_INTENT_ACTION_POLICY_REPORT.md).

## Full Git ledger

The table records every commit in chronological order. The plasticity row range
is a side branch from the pathway-audit tip; the learned-bridge line continued
from that shared history. All other phase branches preserve named checkpoints
on the final project ancestry.

| UTC+05:00 date | Commit | Milestone |
| --- | --- | --- |
| Sep 15 | `978f57c` | Deterministic Metal v3 radix-sort foundation |
| Sep 15 | `b55962b`, `dce319b`, `8c93153` | Toolchain identity, v3 primitives, v4–v6 checkpoint |
| Sep 15 | `fa4c2ea`, `9f87f4e`, `5adeb2d`, `1074221`, `0c152ee` | Embodiment, baseline, visual demo, report/tag |
| Sep 15 | `8fdb8e8`, `a5c237f`, `63b4eb1`, `d1c0f4a`, `8b0e286`, `27d6de2` | Passive baseline, probing, causal tests, rejected decoder, diagnostics tag |
| Sep 15 | `6d77f66`, `27fb532`, `17cdd12`, `5c6f74f`, `64a88d0` | Pathway ranking, stimulation, decomposition, sweeps, audit tag |
| Sep 15 | `778b607`, `38b5c44`, `ff8578b`, `3ee9f31`, `f018c8c`, `d34bae7`, `be9a4e0`, `82518d8`, `dbf42fe` | Targeted-plasticity branch and negative-result tag |
| Sep 15–16 | `208d99d`, `28b9f0f`, `5c303e3`, `4260b52`, `ce12b9b`, `d36e397`, `8e29c8e`, `ac88413`, `f7f0f04` | Learned bridge through final tag |
| Sep 16 | `ac31188` | Interactive science demo |
| Sep 16 | `7be8516`, `35e52e1` | Arcade v1 and corrected v2 |
| Sep 16 | `26036bf`, `e300260` | Binocular experiment and negative-result tag |
| Sep 18 | `0befa9a`, `aead9e8`, `91ac824` | 53.7% checkpoint, 70.4% refinement, 67.8% final champion |
| Sep 19 | `133a3db` | Final presentation polish |

## Compute accounting

No complete command-run ledger was recorded, so this section does **not**
claim an exact project-total bill. It separates directly measured numbers,
calculable lower bounds, and a clearly labelled development estimate.

### Measured rates and runs

| Workload | Backend | Recorded measurement |
| --- | --- | --- |
| Full-graph baseline benchmark | CPU / Apple M4 | 29.2857 simulated seconds in 56.017 wall seconds; about 686 MiB RSS; about 916% CPU across cores |
| Embodied 60-episode MaleCNS condition | CPU / Apple M4 | about 28 s neural time in about 350 s wall |
| Metal sustained retinal sequence | Metal / Apple M4 | 280,000 ticks at 12,994 steps/s (1.299× biological real time) |
| FlyKeeper 100-episode comparison | Metal / Apple M4 | 26.16 s wall, versus 48.43 s CPU |
| Arcade matched 27-shot run | Metal / Apple M4 | about 9.6 minutes wall |
| Interactive reference loop | CPU | about 0.12× real time; a penalty takes several wall-clock seconds |

Sources: [setup report](reports/SETUP_REPORT.md), [embodied report](reports/EMBODIED_REPORT.md),
[Metal report](../workspace/gpu/METAL_BACKEND_REPORT.md),
[Arcade v3 report](reports/ARCADE_BINOCULAR_BRIDGE_V3_REPORT.md), and
[interactive report](reports/INTERACTIVE_DEMO_REPORT.md).

### Reconstructable lower bounds

- The explicitly timed 60-episode CPU condition (350 s), 100-episode Metal
  comparison (26.16 s), and 27-shot Metal run (about 576 s) alone account for
  **at least about 16 minutes of recorded wall time**. This is intentionally
  only a lower bound: it excludes nearly all dataset generation, sweeps,
  development evaluations, test reruns, compilation, and failed experiments.
- At the recorded 27-shot Arcade pace, a comparable 90-shot battery would take
  roughly **32 minutes**. This conversion is illustrative only; different
  arcade paths, warm-up behavior, and evaluation configurations make it
  unsuitable for summing every reported battery as though they were identical.

### Project-scale estimate, with uncertainty

The timeline contains repeated dataset collections, calibration sweeps,
ablation ladders, training runs, and final matrices that were not individually
timed. Based on the measured rates and the number of recorded experimental
stages, a conservative reconstruction is **roughly 8–25 CPU wall-hours and
4–12 Metal/GPU wall-hours of experiment execution**, plus unlogged interactive
debugging, builds, and analysis. Treat this as an order-of-magnitude
development estimate—not resource telemetry and not a cost-accounting figure.
The exact total is unknowable from repository evidence alone.

## Reproduce and inspect

The frozen final report is the source of truth for champion commands, hashes,
and artifact paths. The final playable Arcade champion is:

```sh
PYTHONPATH=workspace upstream/doomfly/.venv-neural/bin/python \
  workspace/experiments/arcade_demo/play_arcade_goalkeeper.py
```

The earlier interactive science demo and its regressions remain available as
historical validation tooling:

```sh
upstream/doomfly/.venv-neural/bin/python -m workspace.experiments.interactive_demo.game
upstream/doomfly/.venv-neural/bin/python -m workspace.experiments.interactive_demo.run_regressions
```

For the final Arcade architecture, start with the command and checksum section
in [EARLY_INTENT_ACTION_POLICY_REPORT.md](reports/EARLY_INTENT_ACTION_POLICY_REPORT.md).
For the prior learned-bridge experiment, use the reproducibility section in
[LEARNED_BRIDGE_REPORT.md](reports/LEARNED_BRIDGE_REPORT.md). Do not mix metrics across
science mode and arcade mode: they use different bodies, evaluators, and
benchmarks.

## Report index

- [Setup and reference baseline](reports/SETUP_REPORT.md)
- [Metal backend investigation](../workspace/gpu/METAL_BACKEND_REPORT.md)
- [Embodied fixed-connectome baseline](reports/EMBODIED_REPORT.md)
- [Neural diagnostics](reports/NEURAL_DIAGNOSTICS_REPORT.md)
- [Pathway activation audit](reports/PATHWAY_ACTIVATION_REPORT.md)
- [Minimal learned visual→DN bridge](reports/LEARNED_BRIDGE_REPORT.md)
- [Interactive science demo](reports/INTERACTIVE_DEMO_REPORT.md)
- [Arcade Bridge v2 correction](reports/ARCADE_BRIDGE_V2_REPORT.md)
- [Arcade binocular vision experiment](reports/ARCADE_BINOCULAR_REPORT.md)
- [Arcade action-policy evolution](reports/ARCADE_BINOCULAR_ACTION_POLICY_REPORT.md)
- [Final early-intent champion](reports/EARLY_INTENT_ACTION_POLICY_REPORT.md)

## What the project established—and did not establish

**Established:** the retained MaleCNS graph can be embodied; early visual
direction is measurable; DNs can move the body; the fixed route loses the
needed direction; targeted native plasticity did not restore it; a small,
explicit learned bridge can produce controlled vision-dependent saves; and a
separate early-intent/action policy can attain 67.8% on a protected arcade
test battery with bilateral-vision dependence.

**Not established:** natural football competence in the connectome, biological
plausibility of the engineered readouts/policies, exact parity of the relaxed
Metal backend, real-time CPU play, or generalization beyond the documented
simulated environments and control conditions.
