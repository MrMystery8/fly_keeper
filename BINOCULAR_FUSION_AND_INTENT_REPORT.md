# Binocular fusion and interception-intent report

## Reset-corrected baseline

The controller evaluator now resets MaleCNS before every episode. The matched
27-shot Metal baseline is Oracle 100.0%, v2 one-eye 74.1%, first binocular
29.6%, and frozen V3 binocular 37.0%. All earlier closed-loop percentages are
pre-fix historical values, not valid rankings.

## Geometry update (supersedes the earlier retinotopy anomaly)

The earlier "no rendered ball pixels in the right camera" finding was overturned
by a systematic calibration (`BINOCULAR_RETINOTOPY_AUDIT.md`). The right camera
DOES see the ball (35/45 grid positions, equal to the left). The real defect was
a retinal-viewport coverage gap: the upstream overlapping-viewport UV map
(left uv_x in [0,0.6], right uv_x in [0.4,1.0]) leaves a frontal blind wedge
because each 140-degree eye camera points 67 degrees laterally, so a near-frontal
(CENTER) ball lands near each image's inner edge, outside both receptor windows.
The fix (`retina_map="fullframe"` in `BinocularVisionBridge`) rescales each eye
population's uv_x to span its own camera frame — no camera move, no pose change,
no FOV widening. The binocular dataset was regenerated under the fix
(`*_fullframe.npz`).

## Current evidence (geometry-corrected, fullframe)

Matched balanced Metal decoding, same complete-episode split, on the
geometry-corrected datasets. Numbers in parentheses are the pre-fix (stale
viewport) values for reference.

| stream | lateral corr | L/R accuracy | AUC |
|---|---:|---:|---:|
| left only | 0.657 (0.579) | 0.739 | 0.838 |
| right only | 0.513 (0.488) | 0.693 | 0.781 |
| naïve both-eye pooling | 0.471 (0.404) | 0.655 | 0.719 |

The right stream has nonzero task correlation but is still damaged by naïve
pooled fusion. Preserving stream identity until prediction gives:

| analysis-only lateral decoder | held-out corr | L/R accuracy | AUC |
|---|---:|---:|---:|
| naïve pooled binocular | 0.471 (0.404) | 0.655 | 0.719 |
| late fusion (validation-selected 65% left / 35% right) | 0.711 (0.655) | 0.766 | 0.875 |
| common/opponent stream-prediction basis | 0.687 (0.639) | 0.764 | 0.862 |
| left plus right residual | 0.709 (0.648) | 0.754 | 0.873 |

The geometry fix lifts every stream, and the late-fusion / residual results
strengthen: the right stream still predicts meaningful error remaining after the
left-stream prediction (residual stream test corr 0.37), and late fusion
(0.711) now exceeds even the left-only monocular decode (0.657). This is offline
evidence and it is now trustworthy — the right-eye geometry/coverage anomaly is
resolved, so the both-eye stream is no longer contaminated by a frontal blind
wedge.

## Target semantics

V3's motor targets are a poor vertical abstraction: held-out output means do
not order LOW/MID/HIGH, while the DN-flight interface has a deadband. The first
intent audit exposed that its dynamic stored `z_cross` labels could become
post-impact extrapolations. The generator now records immutable shot-plane
labels; for the existing balanced feature artifact, the corrected audit
reconstructs `z*` from the explicitly stored shot height fraction instead.

The corrected held-out linear comparison is not yet a win for intent labels:

| target | corr | selected history / features | class accuracy | class-mean prediction |
|---|---:|---|---:|---|
| motor lateral `u_lat` | 0.404 | 160 ms / 207 | 33.7% | left -0.101, center -0.004, right 0.143 |
| destination `y*` | 0.419 | 160 ms / 207 | 34.1% | left 0.077, center -0.004, right -0.136 |
| motor vertical `u_vert` | 0.429 | 160 ms / 64 | 35.8% | low 0.297, mid 0.343, high 0.279 |
| destination `z*` | 0.299 | 120 ms / 64 | 26.9% | low 0.083, mid 0.117, high 0.156 |

`z*` produces the desired LOW < MID < HIGH ordering, but its classification
still collapses to MID and its correlation is lower. Neither lateral nor
vertical intent has established a superior decoding target yet. At 80 and 160
ms the held-out fixed-time probes are chance-level; useful correlation appears
by 240 ms, which makes timing a continuing constraint.

## Closed-loop outcome of the structured lateral hybrid

The offline late-fusion win was carried into a closed-loop development hybrid
(separate left/right hemisphere lateral estimators -> late linear fusion ->
real lateral DN opponent pathway, with the existing V3 vertical head frozen).
Full details in `STRUCTURED_BINOCULAR_LATERAL_REPORT.md`. On 90 matched
reset-corrected shots:

| controller | overall | LEFT | CENTER | RIGHT |
|---|---:|---:|---:|---:|
| Oracle | 100.0% | 1.00 | 1.00 | 1.00 |
| v2 one-eye | 70.0% | 0.80 | 0.63 | 0.67 |
| V3 (stale viewport) | 38.9% | 0.63 | 0.30 | 0.23 |
| structured hybrid (both eyes) | 41.1% | 0.30 | 0.33 | 0.60 |

The fly causally uses both eyes (both 41.1% > left-blind 30.0%, right-blind
35.6%, both-blind 26.7%), and the right eye specifically rescues right-side
saves (right/low 8/10 with both eyes, 0/10 with either eye blinded). But the
hybrid is substantially WORSE than v2 (41.1% vs 70.0%): it collapses on LOW/MID
lateral shots where v2 succeeds. The improved sensory representation did not
translate into a better embodied lateral controller.

## Decision

No V4 is justified. The right-eye geometry is validated and structured fusion is
reproducibly useful OFFLINE, but the closed-loop lateral behaviour does not beat
(or match) the 74.1%/70% v2 baseline — the third V4 condition fails, so no V4 is
manufactured. The clean, causal binocular sensory architecture is the real
deliverable and is worth keeping as the foundation. The controller gap (decode
quality and lateral commitment/timing on LOW/MID shots), not the sensory system,
is now the bottleneck. Real DNs stay in the path; there is no RL or direct
visual-to-MuJoCo control.
