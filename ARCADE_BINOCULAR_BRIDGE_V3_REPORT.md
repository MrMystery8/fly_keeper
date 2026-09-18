# Arcade binocular neural-remap decision report

## Decision

**Arcade Binocular Bridge V3 was justified and built as a separate artifact.**
The prior dataset was discarded for V3 development because it omitted MID.
The fresh balanced data showed that 160 ms of causal binocular MaleCNS history
contains held-out lateral information (0.4044 correlation; 68.94% L/R accuracy;
0.7184 AUC). The model remains a small, interpretable supervised linear bridge;
it is not RL, an RNN, a transformer, or a direct visual-to-body controller.

Science Mode remains frozen. This report concerns only the engineered Arcade
path:

`two eye cameras → R1–R6 → MaleCNS → candidate visual features → bridge → real DNs → DN decoder → articulated fly`.

## Historical pre-reset baseline (Metal, 27 shots — invalid for ranking)

All controllers were evaluated on the same seeds (90000 onward), same 3×3
shot matrix, timing, ball physics, motor parameters, and interception semantics.

| controller | overall | LEFT | CENTER | RIGHT | LOW | MID | HIGH |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Arcade Bridge v2, one eye | 37.0% | 33.3% | 22.2% | 55.6% | 22.2% | 11.1% | 77.8% |
| current binocular bridge | 25.9% | 33.3% | 22.2% | 22.2% | 0.0% | 0.0% | 77.8% |

Full 3×3 cells are preserved in
`workspace/outputs/arcade_demo/binocular_evaluate.json`. The terminal-event
breakdown is Oracle contact/deflected-save/touch-but-goal =
100.0%/100.0%/0.0%; v2 = 11.1%/11.1%/0.0%; binocular = 0.0%/0.0%/0.0%.
The evaluator records those separately and preserves the required
`CONTACT != SAVE` semantics.

One immediately preceding identical Metal repetition yielded 33.3% for v2
rather than 37.0% (a single boundary shot); binocular remained 25.9%. This is
reported rather than hidden: fixed shot seeds alone do not currently guarantee
bit-identical Metal closed-loop terminal outcomes. No value from either run was
used for fitting or model selection.

## Authoritative reset-corrected baseline (Metal, 27 shots)

The evaluator now resets `controller.brain` before every shot. Re-running the
same 3×3, three-shot-per-cell matrix with seeds 90000 onward yields:

| controller | overall | LEFT | CENTER | RIGHT | LOW | MID | HIGH | contact / deflected / touch-goal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0 / 100.0 / 0.0% |
| v2 one-eye | 74.1% | 77.8% | 77.8% | 66.7% | 66.7% | 55.6% | 100.0% | 63.0 / 63.0 / 0.0% |
| first binocular | 29.6% | 44.4% | 22.2% | 22.2% | 11.1% | 0.0% | 77.8% | 3.7 / 3.7 / 0.0% |
| frozen V3 binocular | 37.0% | 66.7% | 22.2% | 22.2% | 33.3% | 0.0% | 77.8% | 22.2 / 22.2 / 0.0% |

Pre-reset percentages above are historical only. The full reset-corrected
matrices are in `workspace/outputs/arcade_demo/binocular_evaluate.json`.
Ten shots per cell would be preferable but was not practical in this pass: the
matched 27-shot run took about 9.6 minutes. This was not used for fitting.

## Receptor and feature audit

- Input mapping is biological `rootSide`, not missing retinal `somaSide`.
- Both visual streams are available: 1,107 L and 2,228 R receptor IDs, each at
  its own UV coordinate; eye-image differences and per-eye receptor changes are
  recorded in `binocular_audit.json`.
- The historical 207 set is 206 R / 1 L and was selected under effectively
  monocular input. The current binocular model instead selected 144 R / 63 L
  from a 600-neuron bilateral pool, using training frames only.
- Held-out offline metrics remain below v2: v2 lateral corr/accuracy 0.407/0.682
  vs binocular 0.341/0.654; vertical corr 0.332 vs 0.345.
- Existing blind-eye controls show real dependency: both 0.276/0.606 lateral
  corr/accuracy, left-only 0.097/0.525, right-only 0.030/0.517.

The exact selected IDs, sides, split seed lists, normalization, weights, and
regularization are stored in
`workspace/outputs/arcade_demo/binocular_bridge_train.json` and
`binocular_bridge_model.npz`; top selectivity maps are in
`neural_activity_map.json`.

## Downstream causal audit

The real DN basis is `[10162,10527]` (left), `[10059,555871]` (right), and
`[10001,10010]` (vertical). Matched direct stimulation, with no bridge or ball
state used for control, yielded opposite lateral body displacement with a 0.955
magnitude ratio and robust vertical lift. This favors neither a DN-basis
replacement nor direct optic-lobe-to-body control.

## Smallest justified next change

Collect a new **Arcade-only V3 development dataset** before model fitting:

1. include LEFT/CENTER/RIGHT × LOW/MID/HIGH complete episodes;
2. record retina, bilateral candidate visual pool, bridge command, per-DN
   activity, decoder command, body state, and terminal semantics at each 20-ms
   decision;
3. select separate lateral and vertical feature sets on TRAIN episodes only,
   scoring selectivity, reliability, early latency, and redundancy;
4. compare 16/32/64/128/207-feature linear heads with validation-only choices;
5. only then perform one held-out matched closed-loop comparison and the four
   blind-eye causal controls.

This preserves both eyes, real DNs, Metal runtime, and embodied interception
while avoiding a test-tuned or incomplete-data “v3”.

## Implemented V3 and final held-out closed loop

V3 has two independent linear heads, both reading true bilateral MaleCNS
activity and injecting the existing real DN basis. The lateral head uses 207
features over 160 ms (ridge alpha 300); the vertical head uses 64 features over
160 ms (alpha 100). Feature selection, normalisation, window/budget selection,
and alpha selection are TRAIN/VALIDATION only; all splits are whole episodes.
Exact IDs, weights, normalisers, and split seeds are in
`workspace/outputs/arcade_demo/arcade_binocular_bridge_v3.npz` and its
`_train.json` companion.

On the matched 27-shot Metal matrix: Oracle 100.0%, v2 one-eye 33.3%, first
binocular 25.9%, and **V3 29.6%**. V3 therefore improves the first binocular
bridge but does not yet beat v2. Its 3×3 result is LEFT 44.4% / CENTER 22.2% /
RIGHT 22.2%; LOW 11.1% / MID 0.0% / HIGH 77.8%; contact 3.7%, deflected save
3.7%, touch-but-goal 0.0%. It is consequently an honest competent supervised
binocular baseline, not a success claim over the one-eye bridge.

Fresh V3 eye controls (nine held-out shots per condition) were: binocular
22.2%, left-blind 22.2%, right-blind 22.2%, both-blind 11.1%. Both-blind
degrades, but single-eye ablations did not on this small matrix. The final
model's single-eye contribution is therefore unresolved; do not interpret V3
as demonstrating a robust binocular advantage. Results are recorded in
`arcade_binocular_v3_eye_controls.json`.

## Post-V3 causal audit

The historical closed-loop values are superseded by the reset-corrected matrix
above. The focused audit finds both non-monotonic V3 height output
and a real vertical deadband below `u_vert≈0.3`; V3 is frozen and no V4 is
justified. See [ARCADE_BINOCULAR_BOTTLENECK_AUDIT.md](ARCADE_BINOCULAR_BOTTLENECK_AUDIT.md).
