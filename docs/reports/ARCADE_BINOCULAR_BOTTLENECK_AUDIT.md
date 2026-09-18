# Arcade binocular V3 bottleneck audit

## Scope

This is Arcade-only. V3 weights, Science Mode, historical artifacts, and the
real DN basis were not changed. Simulator truth appears only in diagnostics.

## Critical evaluation-integrity correction

The dataset generator reset MaleCNS every episode, but historical closed-loop
`run_episode()` did not. It now resets MaleCNS before each shot. Earlier
closed-loop percentages, including V3 29.6%, are pre-fix references and must
be regenerated before headline model ranking or a tag.

## MID end-to-end trace

Nine fresh deterministic shots (LEFT/CENTER/RIGHT × LOW/MID/HIGH) were logged
at 20 ms: teacher target, V3 command, actual prior-step DN drive, DN spikes,
motor output, fly position/velocity/state, ball position, envelope distance,
and terminal semantics. All three MID shots were GOAL.

LOW, MID, and HIGH had the same mean V3 vertical sequence: 0.34 at launch,
0.41 at 20 ms, ~0.42 at 40–100 ms, and 0.50 at 160 ms. DN activity, decoded
vertical command, and takeoff therefore also began together (20–40 ms), before
the output decayed. MID intent is already ambiguous at the bridge output; it
does not disappear solely in the DN/motor path.

Raw trace: `workspace/outputs/arcade_demo/arcade_binocular_bottleneck_audit.json`.

## Vertical command-to-height response

Direct real-DN stimulation exposes an effective deadband:

| `u_vert` | decoded vertical | takeoff | peak height | rise latency |
|---:|---:|---|---:|---:|
| 0.0–0.2 | 0.000 | no | 0.000 cm | — |
| 0.3 | 0.252 | yes | 0.446 cm | 60 ms |
| 0.4 | 0.564 | yes | 0.920 cm | 40 ms |
| 0.5 | 0.811 | yes | 1.341 cm | 20 ms |
| 0.6–0.9 | 0.913–0.975 | yes | 1.426–1.466 cm | 20 ms |

`ArcadeFlyBody` uses `TAKEOFF_VERT = 0.15`; the real DN decoder emits no
vertical command through 5 mV drive (`u_vert <= 0.2`). The 1.0 response is
non-monotonic because sustained strong drive reduces DN spikes. The downstream
mapping is thus not a calibrated continuous height channel.

## Does V3 represent MID?

Untouched balanced-test V3 output means are LOW 0.297, MID 0.342, HIGH 0.279
(SD 0.131, 0.104, 0.146): not monotonic. A train-centroid LOW/MID/HIGH readout
is only **38.7%** accurate; most LOW and HIGH frames are classified MID. V3
therefore lacks a reliable three-level height representation.

`workspace/outputs/arcade_demo/v3_vertical_class_audit.json` holds the full
confusion matrix and distributions.

## Representation and temporal ceiling

An analysis-only tiny 32-ReLU MLP used the same V3 selected features/splits:

| task | linear test | tiny-MLP test |
|---|---:|---:|
| lateral correlation | 0.404 | 0.459 |
| L/R accuracy | 0.689 | 0.669 |
| L/R AUC | 0.718 | 0.698 |
| vertical correlation | 0.429 | 0.405 |

It does not reveal a compelling nonlinear reserve. Causal multiscale
mean/slope summaries also did not beat raw 160-ms history on validation.
Details are in `workspace/outputs/arcade_demo/binocular_representation_audit.json`.

## Decision

No V4 is created. The smallest justified next experiment is a calibration-only,
monotonic real-DN-to-flight vertical mapping, replaying frozen V3 outputs on
development seeds. It cannot solve the non-monotonic V3 vertical prediction;
after calibration, vertical target/feature redesign can be considered
separately. Both eyes remain in the architecture.

## Matched binocular-fusion result

The required three-way offline comparison is complete on identically seeded,
balanced Metal datasets. TRAIN/VALIDATION selected 160 ms / 207 features for
each condition; held-out lateral decoding is:

| retinal condition | corr | L/R accuracy | AUC |
|---|---:|---:|---:|
| both | 0.404 | 0.689 | 0.718 |
| left-only | **0.579** | **0.754** | **0.824** |
| right-only | 0.488 | 0.669 | 0.748 |

Current both-eye fusion does not demonstrate complementary laterality; it is
worse than left-only. This does not authorize reverting the final architecture
to one eye. It identifies fusion-aware feature selection as the representation
problem before any further bridge retraining. Raw output:
`workspace/outputs/arcade_demo/binocular_balanced_fusion_audit.json`.
