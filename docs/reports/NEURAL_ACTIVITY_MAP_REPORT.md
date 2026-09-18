# Neural activity map — binocular Arcade diagnostic

## Scope and integrity

This is an Arcade-only diagnostic on branch `arcade-binocular-neural-remap`.
Science Mode, its training data, and the frozen scientific bridge are not
modified. The analysis reads the corrected Metal binocular dataset, whose
manifest records complete episodes only and rejects post-terminal samples.
Splits are by whole shot seed (train/validation/test), never by frame.

## Verified binocular input

`BinocularVisionBridge` renders `eye_left` and `eye_right` at every decision.
It samples each receptor at its own retinal UV coordinate, then sends left
`rootSide == L` receptors only pixels from `eye_left` and right `rootSide == R`
receptors only pixels from `eye_right`. The audited manifest has 1,107 left and
2,228 right R1–R6 receptors (3,335 total). In representative deterministic
shots the two camera images differ by 47.07–84.37 mean RGB units; substituting
the proper right-eye image changes right-receptor luminance by 0.0819–0.3176.
That rules out duplicate left-camera pixels at the retinal input.

The machine-readable mapping, receptor IDs, sides, and UVs is
`workspace/outputs/arcade_demo/binocular_eye_manifest.npz`; the current audit
is `workspace/outputs/arcade_demo/binocular_audit.json`.

## Where task information is measured

The first map is deliberately conservative: raw retina is verified directly by
the audit above; the 600-neuron bilateral candidate optic-lobe pool is recorded
from the real Metal MaleCNS at every causal 20-ms decision; the current bridge
features are a training-only selection from that pool; DNs are read directly
from the real MaleCNS.

`workspace/outputs/arcade_demo/neural_activity_map.json` contains the top 50
directional and height-responsive neurons with body ID, type, soma/root side,
effect size, response difference, and reliability. It also contains the
complete episode split and held-out diagnostic-probe results. The current data
shows weak *single-bin* L/R decoding (held-out accuracy 0.542), while the
existing causal four-bin bridge reports 0.341 lateral correlation and 0.654
L/R direction accuracy. This establishes that the task signal is temporally
distributed and weak in the present candidate basis rather than evidence for a
late direct body controller.

Important limitation: the legacy binocular dataset has LOW/HIGH examples only;
its height probe is explicitly reported as two observed classes, not fabricated
as a three-class result. A full V3 data collection must add MID before any
height model selection.

## DN and motor causal check

Directly injecting the real lateral basis at 25 mV, without the visual bridge,
produced +0.37950 and -0.39758 lateral fly displacement in matched trials
(magnitude ratio 0.9545). Direct vertical-DN stimulation produced +1.22791
vertical displacement. The exact IDs are lateral-left `[10162, 10527]`,
lateral-right `[10059, 555871]`, and vertical `[10001, 10010]`.

The raw synchronized traces are in
`workspace/outputs/arcade_demo/dn_causal_audit.json`. Thus there is no gross
left/right DN-to-motor polarity or magnitude failure requiring a substitute
motor path. The remaining causal question is whether early bilateral visual
features can be selected to drive this symmetric basis reliably and early
enough.

## Interpretation

Both eyes genuinely reach the MaleCNS, and direct DNs can drive the articulated
fly symmetrically. That initial evidence did **not** justify a bridge rewrite;
it justified the balanced temporal collection below, which became the controlled
V3 decision gate.

## Balanced temporal foundation (V3 development)

The follow-up collection is now complete: **72 Metal episodes / 3,274 samples**
with exactly eight complete episodes in every LEFT/CENTER/RIGHT × LOW/MID/HIGH
cell. It stores eight 20-ms causal bins and resets the history at each episode.
The manifest asserts resolved terminals, no post-terminal samples, exact 3×3
balance, unique `(seed, step)` rows, and manifest/sample-boundary agreement.
The complete-episode stratified split is 45 train / 9 validation / 18 test.

Validation selected an eight-bin (160 ms), 207-feature lateral probe. Its one
untouched test estimate is lateral correlation **0.4044**, L/R accuracy
**0.6894**, and AUC **0.7184**. A separate vertical head selected 64 features
and reached vertical correlation **0.4292**. This establishes that the weak
single-bin result was not the whole story: useful binocular laterality appears
under causal temporal integration.

The complete temporal-window/budget validation table, selected body IDs, and
balanced selectivity map are in
`workspace/outputs/arcade_demo/binocular_balanced_temporal_analysis.json`; the
dataset and episode manifest are `binocular_balanced_dataset.npz` and
`binocular_balanced_dataset_manifest.json` in the same output directory.

The synchronized MID trace, vertical command-response curve, and tiny nonlinear
ceiling probe are documented in [ARCADE_BINOCULAR_BOTTLENECK_AUDIT.md](ARCADE_BINOCULAR_BOTTLENECK_AUDIT.md).

Matched left-only/right-only/both Metal data shows the current both-eye fusion
underperforms left-only decoding; the exact held-out comparison is in that
bottleneck report.
