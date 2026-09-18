# Arcade binocular action-policy report

## Status

This is a new engineered Arcade controller.  Science Mode, the frozen
connectome, camera geometry, retinal identity, and historical controller
artifacts are not modified.

## Causal architecture

```text
left fullframe retina  -> frozen MaleCNS -> left stream prediction  \
                                                                 tiny action policy -> real DN basis -> DN decoder -> force-driven ArcadeFlyBody
right fullframe retina -> frozen MaleCNS -> right stream prediction /
```

The policy preserves separate left/right stream predictions until its final
fusion.  Its ten causal inputs are those streams, the vertical stream,
previous 2-D action, fly lateral/vertical position and velocity, and airborne
state.  It receives no ball position, velocity, shot class, player aim, or
interception target.  Teacher truth is used only offline to create labels.

## Policies tested

`BinocularActionPolicyBridge` is a 10 -> 12 -> 2 tanh MLP.  It has three
controlled deployment modes: bounded residual, full action, and convex
base/full blend.  In all three cases its action is written into the existing
bridge EMA and reaches the body only through the real `ArcadeDNBasis` DN
injector and DN decoder.  It never writes MuJoCo body position or velocity.

Three supervised datasets were collected with fullframe binocular retinas and
episode resets: 1,231 residual-labelled steps / 27 episodes, 1,195 full-action
steps / 27 episodes, and the selected 4,115-step residual set / 90 episodes.
Splits are by episode seed.  The selected 90-episode model reached held-out
imitation correlations of 0.725 lateral and 0.604 vertical; those are teacher
action metrics, not save-rate metrics.

## Results

### Authoritative matched 90-shot matrix

Fresh balanced LEFT/CENTER/RIGHT × LOW/MID/HIGH evaluation, ten reset-corrected
shots per cell, fullframe binocular retinas:

| controller | overall | left | center | right | low | mid | high |
|---|---:|---:|---:|---:|---:|---:|---:|
| fullframe binocular supervised base | 45.6% | 30.0% | 43.3% | 63.3% | 43.3% | 20.0% | 73.3% |
| binocular residual action policy (90-episode pretraining) | **53.3%** | 80.0% | 33.3% | 46.7% | 53.3% | 30.0% | 76.7% |

The policy improves the matched fullframe binocular base by 7.7 percentage
points, principally on lateral low shots, while preserving physical simulation.  It
does **not** meet the requested 60–70% result or the historical one-eye v2
70.0% benchmark, so it is not represented as the final successful controller.

Full policy matrix (saves/attempts):

| | low | mid | high |
|---|---:|---:|---:|
| left | 10/10 | 6/10 | 8/10 |
| center | 2/10 | 0/10 | 8/10 |
| right | 4/10 | 3/10 | 7/10 |

Movement averaged over those 90 shots: mean absolute lateral displacement
0.309 cm, peak lateral displacement 0.728 cm, mean vertical displacement
0.587 cm, peak vertical displacement 1.301 cm, takeoff rate 100%, mean
airborne duration 41.9 20-ms decisions, and one substantial flight event per
episode.

### Movement comparison with v2

On a separate matched fresh 3×3 trace set, v2 one-eye measured mean absolute
lateral displacement 0.234 cm and peak lateral displacement 0.474 cm.  The
selected binocular action policy's corresponding 90-shot values are 0.309 cm
and 0.728 cm: +32% mean and +54% peak lateral movement.  It also remained
airborne for 41.9 decisions on average versus v2's 36.3.  The new controller
therefore objectively moves more, while retaining physical force-driven motion.

### Development 27-shot matrix, residual policy (scale 0.20)

| controller | save rate | mean abs lateral | peak lateral | mean vertical | peak vertical |
|---|---:|---:|---:|---:|---:|
| fullframe binocular supervised base | 51.9% | 0.244 cm | 0.664 cm | 0.574 cm | 1.317 cm |
| residual action policy | 48.1% | 0.269 cm | 0.712 cm | 0.525 cm | 1.307 cm |

The residual controller increases lateral commitment but did not improve saves
on this held-out matrix.  It is therefore not selected as a final successful
policy on the basis of this result.

### Full-action policy smoke matrix, 9 shots

The full policy was intentionally much more actionous (peak lateral movement
1.58 cm versus 0.72 cm for the base), but saved 33.3% versus 44.4% on its fresh
smoke matrix.  It overcommits; this is retained as a negative controlled result.

### Offline calibration does not substitute for closed-loop selection

The 90-episode policy's episode-held-out imitation data favored residual scale
1.0 (MAE 0.250 versus 0.447 at scale 0.20).  A fresh 27-shot physical check
showed the opposite behavior: scale 1.0 saved 29.6% and reached 1.51 cm peak
lateral displacement, indicating overcommitment.  The retained scale 0.20 is
therefore chosen by closed-loop behavior, not offline action MAE.

### Causal eye ablation ladder, selected 90-episode policy, matched 9 shots

| condition | residual policy save rate |
|---|---:|
| both eyes | 55.6% |
| left eye blinded | 33.3% |
| right eye blinded | 33.3% |
| both eyes blinded | 33.3% |

Both individual eye removals reduce the selected policy by 22.3 points to the
blind floor on these matched seeds.  The reproducible protocol and outputs are
in `run_action_policy_ablations.py` and
`workspace/outputs/arcade_demo/arcade_binocular_action_policy_eye_ablations.json`.

## Movement semantics

The Arcade body uses bounded applied forces, damping, speed limits and genuine
MuJoCo collision response.  It exposes READY, GROUND_CORRECTION, TAKEOFF,
FLIGHT, DIVE and RECOVER states.  A contact is not a save: the ball is deflected
and must still fail to enter the valid goal volume to count as a save.

## Scientific scope

Biological/data-derived components are the MaleCNS connectome simulation,
retinal receptor populations and real descending neurons.  The retina camera
interface, stream readout, tiny policy, teacher labels and Arcade locomotor
mapping are engineered.  This is an embodied task controller, not a claim that
a fly naturally learned football.

## Current conclusion

The controller is causally binocular and physically actionous, but the present
supervised alternatives have not yet met the requested 60–70% held-out target.
The full 90-shot result is the authoritative measurement; the next iteration
needs timing/vertical-control improvement or bounded RL, not a performance claim
from this controller.
