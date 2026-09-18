# Binocular controller-interface audit

## Status and scope

The sensory solution is frozen: camera geometry/FOV, `retina_map="fullframe"`,
separate eye streams and late fusion were not changed. The V3 vertical path is
also unchanged. No RL and no V4 were introduced.

The established 90-shot, reset-corrected matched baseline (10 shots in every
LEFT/CENTER/RIGHT × LOW/MID/HIGH cell; seed base 90000) is:

| controller | saves | contact | touch-but-goal | deflected save |
|---|---:|---:|---:|---:|
| Oracle | 90/90 (100.0%) | 100.0% | 0.0% | 100.0% |
| v2 one-eye | 63/90 (70.0%) | 62.2% | 0.0% | 62.2% |
| fullframe structured binocular | 37/90 (41.1%) | 18.9% | 0.0% | 18.9% |

The binocular matrix was left low 0/10, left mid 0/10, center low 0/10,
center mid 2/10, right low 8/10, right mid 2/10; high was 25/30. Thus this is a
lateral closed-loop failure concentrated in low/mid, not a vertical redesign
request. Its causal ablation remains strong: both eyes 41.1%, left blind 30.0%,
right blind 35.6%, both blind 26.7%.

## New synchronized interface trace

`workspace/experiments/arcade_demo/binocular_controller_interface_audit.py`
writes a causal row per 20 ms decision into
`workspace/outputs/arcade_demo/binocular_controller_interface_audit.json`.
Each row records raw/fused stream values, the *previous* output actually
injected into DNs, current DN L/R spikes and opponent difference, decoded
motor command, fly position/velocity, ball position/time-to-cross, and outcome.
This corrects a potentially misleading alignment: a command calculated at tick
`t` is injected at `t+1`, then decoded after the neural step.

The completed 1-shot-per-cell stratified smoke holdout is diagnostic only (not
the 90-shot confirmation): v2 6/9, uncalibrated binocular 4/9, calibrated
binocular 7/9. It reproduces the historical ordering and isolates the failure.

### What the trace shows

| measure, smoke holdout | v2 | uncalibrated binocular |
|---|---:|---:|
| first correct bridge sign, left/right | 20/20 ms | 0/40 ms |
| first sustained correct sign, left/right | 20/20 ms | 80/93 ms |
| first DN asymmetry, left/right | 40/40 ms | 120/80 ms |
| injected-output → DN-difference slope | 6.92 | 2.60 |
| injected-output → motor slope | -1.58 | -1.14 |
| DN near-zero frequency | 17.1% | 19.3% |

The binocular output is not merely too small. In the same trace, raw means by
direction were LEFT `+0.233`, CENTER `+0.358`, RIGHT `+0.313`, whereas desired
bridge ordering is LEFT < CENTER < RIGHT. LEFT sign accuracy was only 21% for
low, 42% for mid, and 32% for high; RIGHT was much better (82%, 63%, 83%). The
bridge consequently commits early rightward/bias motion, then obtains a
sustained corrective signal 60–100 ms later than v2. That is especially costly
for the low/mid shots that resolve quickly.

This answers the apparent contradiction: offline correlation rewards ranking
across samples; it does not require zero intercept, correct sign on every early
frame, persistence, nor a DN-compatible temporal distribution. The new model
contains useful binocular information (shown by ablation/offline decoding), but
its online intercept and transient dynamics are incompatible with the v2
lateral action interface.

## DN transfer and controls

There is no hidden alternate downstream path. Both production variants use the
same `ArcadeDNBasis`, real DN readout, and `Arcade2AxisDecoder`. The control
runs make that explicit:

- v2 output through the hybrid downstream interface: 6/9, exactly v2's 6/9.
- binocular output through the v2 downstream interface: 4/9, exactly the
  uncalibrated binocular 4/9.

So the primary bottleneck is bridge-output dynamics at the interface, not a
hybrid-only downstream integration bug. The smaller binocular transfer slopes
are a *consequence* of that signal spending more time in biased/transient or
weakly effective regimes, not evidence of a different DN implementation.

## Development-only calibration

A pure scalar, monotonic calibration was fit from a disjoint, stratified
development trace only. It uses no ball state or shot class at runtime:

```text
u_DN = clip(0.8326 * u_fused - 0.15, -1, 1)
```

The bias centers the development CENTER output; gain then matches v2's
left/right absolute output magnitude. On the diagnostic holdout it raised saves
from 4/9 to 7/9 and contact from 2/9 to 6/9. This is promising but **not a
claim of 90-shot restoration**: feedback changes future visual frames, so the
calibrated closed-loop distribution must be measured on the final held-out
matrix rather than inferred from the uncalibrated trace.

## Failure classification and conclusion

On the smoke misses, binocular failures were 40% early reversal/stop and 60%
center/interception failures; v2 was 33% reversal/stop and 67%
center/interception. The small sample is not suitable for final percentages,
but its direction, bias, and latency measures are decisive enough to reject the
"scale only" hypothesis.

The required 90-shot confirmation command is:

```sh
upstream/doomfly/.venv-neural/bin/python \
  workspace/experiments/arcade_demo/binocular_controller_interface_audit.py \
  --per-cell 10 --dev-per-cell 10 --base-seed 90000
```

For the final, leakage-free calibration run, use the already selected values
explicitly (the runner skips development fitting when these flags are present):

```sh
upstream/doomfly/.venv-neural/bin/python \
  workspace/experiments/arcade_demo/binocular_controller_interface_audit.py \
  --per-cell 10 --dev-per-cell 0 --base-seed 90000 \
  --fixed-gain 0.8326 --fixed-bias -0.15
```

This freezes the calibration from the earlier disjoint 9-shot development
episodes (seed base `193100`; one shot in each 3×3 cell). The final test starts
at seed `90000`, so no episode contributes to both parameter selection and
evaluation. It preserves all frozen artifacts and writes the full matrices, traces,
conditioned distributions, latency, transfer, calibration, swaps, and failure
breakdown to `workspace/outputs/arcade_demo/binocular_controller_interface_audit.json`.
The native neural backend takes about 7.5 minutes even for the 9-shot smoke
matrix, so the full multi-controller confirmation is a long run.

V4 is **not justified yet**. First freeze and evaluate this minimal calibration
on the 90-shot held-out matrix. If it confirms recovery while eye ablation still
shows a binocular advantage, that is the evidence threshold for a true V4. If
it does not, the next work should target early bridge bias/persistence—not
retinal geometry, vertical control, or RL.
