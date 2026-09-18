# Structured binocular lateral report

Does resolving the right-eye geometry and building a clean separate-stream,
late-fused binocular LATERAL pathway produce a genuinely better embodied
goalkeeper than the reset-corrected v2 one-eye baseline (74.1%)?

**Short answer:** the right-eye sensory defect is real and fixable, structured
late fusion is reproducibly better than naïve pooling and the fly demonstrably
uses both eyes closed-loop — but the resulting lateral controller (41.1% on 90
shots) is substantially worse than v2 (70.0%). The sensory architecture is
sound and worth keeping; a V4 is **not** justified.

All work is analysis/experiment only. Science Mode, the frozen `VisionBridge`,
MaleCNS dynamics, the neural graph, LIF constants, and the upstream retinal UV
map are untouched. Frozen tags were not altered. No RL, no vertical redesign.

## Artifacts

| script | output |
|---|---|
| `binocular_calibration.py` | `binocular_calibration.json` (45-position visibility grid) |
| `binocular_overlay.py` | `calib_overlays/*.png` (receptor-coverage overlays) |
| `binocular_neural_trace.py` | `binocular_neural_trace_{viewport,fullframe}.json` |
| `binocular_dataset.py --retina-map fullframe` | `binocular_balanced_{dataset,left_only,right_only}_fullframe.npz` |
| `binocular_structured_fusion.py --suffix _fullframe` | `binocular_structured_fusion_fullframe.json` |
| `binocular_lateral_hybrid.py` | `arcade_binocular_lateral_hybrid.npz` (development hybrid) |
| `binocular_lateral_eval.py` | `binocular_lateral_eval.json`, `binocular_lateral_hybrid_traces.json` |

The fix lives in `binocular_vision.py`: `BinocularVisionBridge(..., retina_map=)`
with `"viewport"` (original, default) and `"fullframe"` (corrected). Backward
compatible; nothing downstream changed unless it opts into `fullframe`.

---

## The ten questions

### 1. Can the ball actually be seen by the right camera across the task envelope?

Yes. A deterministic 45-position grid (x in {2.4, 1.2, 0.4}, y in
{+1.2,+0.6,0,-0.6,-1.2}, z in {0.45, 0.7, 1.25}) rendered both eye cameras with
the keeper at its standing goal-line pose. The right camera saw the ball in
**35/45** positions, equal to the left (35/45); both saw it in 25/45; neither in
0/45. The prior "right camera is blind" claim was an artifact of sampling only
three fired-shot frames.

### 2. Is right-eye camera geometry correct?

Yes. Both eye cameras sit on the `head` body of the vendor flybody MJCF as
genuine mirror counterparts (local pos ±0.0219 in x, mirror-signed quaternions,
fovy 140 each). Live world optical axes: left points world-left, right points
world-right, each 67 degrees off the +x incoming-ball axis, interocular angle
132.7 degrees. This is biologically-appropriate lateral compound-eye placement.
The cameras were **not** re-pointed and the FOV was **not** widened — doing so
would be wrong, because the geometry is already correct.

### 3. Is right retinal sampling correctly oriented?

This is where the real defect was. The upstream UV projection lays both eye
receptor populations onto overlapping viewports of one shared image
(left uv_x in [0,0.6] = image px 0-95; right uv_x in [0.4,1.0] = px 64-159, with
the right eye horizontally mirrored via `0.40 + 0.60*(1-z)`). Because each eye
camera axis points 67 degrees laterally, a near-frontal (CENTER) ball lands near
the inner edge of each camera image (left cam px ~130, right cam px ~30) — both
outside the corresponding receptor window. Result: a **frontal blind wedge**
where CENTER shots produce ~zero receptor signal in BOTH eyes even though both
cameras see the ball (overlays: `calib_overlays/CENTER_MID_*.png`).

Fix (`retina_map="fullframe"`): rescale each eye population's uv_x to span its
own camera frame [0,1], preserving retinotopic order and the right eye's
mirrored orientation; uv_y unchanged. This corrects which pixels each receptor
reads without touching camera pose or FOV. Receptor ball-signal (MID height,
near frame): CENTER goes from 0.0/0.0 (viewport) to 201/269 (fullframe); a real
binocular overlap band appears at LEFT/CENTER/RIGHT; the far periphery correctly
stays monocular.

### 4. Does right-eye MaleCNS activity contain direction information?

At the retina level, the fix restores strong, spatially-appropriate drive to
both eye populations including the previously silent CENTER zone. The
authoritative neural-level answer is the decoding (Q5-Q6): a right-only linear
decoder reaches lateral test corr 0.51 (L/R accuracy 0.69, AUC 0.78), i.e. the
right stream carries genuine, decodable direction information. A static-position
optic-lobe spike probe is a weak instrument for direction selectivity and is not
relied on.

### 5. Does right-eye information add value conditioned on the left eye?

Yes. On the geometry-corrected data, the left-plus-right-residual decoder
(right stream predicting the error left over after the left prediction) reaches
test corr 0.709, above left-only (0.657); the residual stream alone scores 0.37.
The validation-selected fusion places a stable non-trivial weight on the right
stream (0.35). The right eye adds complementary information beyond the left.

### 6. Does late fusion still outperform naïve pooling after geometry validation?

Yes, and more strongly. Held-out lateral corr (pre-fix -> post-fix):

| stream | corr | L/R acc | AUC |
|---|---:|---:|---:|
| A left-only | 0.657 (0.579) | 0.739 | 0.838 |
| B right-only | 0.513 (0.488) | 0.693 | 0.781 |
| C naïve pooling | 0.471 (0.404) | 0.655 | 0.719 |
| D late fusion | **0.711** (0.655) | 0.766 | 0.875 |
| E left+right residual | 0.709 (0.648) | 0.754 | 0.873 |

Late fusion (0.711) far exceeds naïve pooling (0.471) and now also exceeds
left-only monocular (0.657). Early/naïve feature pooling still destroys useful
structure; preserving streams and fusing late still wins.

### 7. Does this improvement survive closed-loop control?

No. A development hybrid (separate left/right hemisphere lateral estimators ->
late linear fusion -> real lateral DN opponent pathway; existing V3 vertical head
frozen) was evaluated on 90 matched reset-corrected shots:

| controller | overall | LEFT | CENTER | RIGHT | LOW | MID | HIGH | keeper contact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Oracle | 100.0% | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| v2 one-eye | 70.0% | 0.80 | 0.63 | 0.67 | 0.67 | 0.50 | 0.93 | 0.62 |
| V3 (stale viewport) | 38.9% | 0.63 | 0.30 | 0.23 | 0.37 | 0.00 | 0.80 | 0.17 |
| structured hybrid (both) | 41.1% | 0.30 | 0.33 | 0.60 | 0.27 | 0.13 | 0.83 | 0.19 |

The hybrid edges past the stale V3 (41.1% vs 38.9%) and is far below v2 (70.0%).
The offline decode gain did not translate into a better embodied controller.

Full 3x3 matrix, both eyes (saves/attempts):

```
            LOW     MID     HIGH
LEFT        0/10    0/10    9/10
CENTER      0/10    2/10    8/10
RIGHT       8/10    2/10    8/10
```

The hybrid collapses on LOW/MID lateral shots (LEFT/low 0/10, CENTER/low 0/10)
that v2 handles (LEFT/low 8/10). HIGH is saved everywhere via the frozen vertical
loft. The lateral commitment/timing on grounded low shots is the failure mode.

### 8. Does the fly causally use both eyes?

Yes — this is the cleanest positive result. Eye-blinding ladder (90 shots):

| condition | overall | LEFT | CENTER | RIGHT | keeper contact |
|---|---:|---:|---:|---:|---:|
| both | 41.1% | 0.30 | 0.33 | 0.60 | 0.19 |
| right_blind (left eye only) | 35.6% | 0.53 | 0.30 | 0.23 | 0.18 |
| left_blind (right eye only) | 30.0% | 0.40 | 0.27 | 0.23 | 0.04 |
| both_blind | 26.7% | 0.30 | 0.27 | 0.23 | 0.00 |

both > right_blind and both > left_blind: removing either eye degrades
performance, so both eyes are causally used. both_blind keeper-contact is 0.0 vs
0.19 with vision, so vision causally produces interceptions (the ~27% floor is
HIGH shots caught by the frozen vertical loft plus reach envelope). The RIGHT
side is where binocular input matters most: RIGHT saves are 0.60 with both eyes
and collapse toward 0.23 when either eye is blinded (right/low 8/10 -> 0/10).
The right eye helps mostly on the right side, as anticipated.

### 9. Does structured binocular lateral control beat or complement the 74.1% v2 baseline?

It does not beat it, and only partially complements it. Overall 41.1% is well
below v2's 70%/74.1%. The one qualitative gain is a more symmetric and stronger
RIGHT side (0.60 vs v2 0.67 and vs stale-V3 0.23) with clean causal two-eye use.
By the report's own success criterion — "similar overall performance + more
symmetric L/R + clear causal use of both eyes" — the causal and right-side
criteria are met but overall performance is not similar; it is materially worse.
Reported honestly rather than framed as a win.

### 10. Is a real V4 justified?

No. The three V4 conditions were: (a) right-eye geometry validated — YES;
(b) structured fusion reproducibly useful — YES, offline; (c) closed-loop lateral
behaviour benefits — NO. Because (c) fails, per the brief's rule a V4 is **not**
manufactured. The hybrid remains an explicitly-labelled development hybrid, not
V4.

---

## What is worth keeping

- The geometry/coverage fix (`retina_map="fullframe"`) is a genuine correctness
  improvement to the binocular sensory pathway and should be the basis of any
  future binocular work. It resolves a real bug (frontal blind wedge) without any
  task-privileged camera tuning.
- The geometry-corrected, integrity-checked binocular datasets (`*_fullframe.npz`)
  and the structured-fusion evidence (right eye adds conditional information;
  late fusion >> naïve pooling) are trustworthy foundations.

## Why the controller underperformed (honest diagnosis)

- Closed-loop lateral decode quality (~0.49 corr for the deployed hemisphere-split
  fusion head) is below v2's mature monocular lateral decoder. The hemisphere
  split gives each stream fewer neurons (394 L / 206 R) than v2's tuned 207-neuron
  monocular feature set, and the fused command still misses the LOW/MID
  commitment window.
- The failure is in the controller layer (decode strength, lateral commitment and
  timing on grounded low shots), not the sensory system. That is the right place
  to look next.

## Recommended next steps (not done here, by instruction)

1. Improve the lateral controller on the corrected sensory stream (feature
   budget, temporal window, gain/smoothing, commitment timing) before any V4.
2. Only then revisit vertical representation (Q of MaleCNS height coding).
3. RL comes after a trustworthy binocular + vertical sensory-motor stack, to
   refine timing/commitment/takeoff — not to paper over a sensory bug. That bug
   is now fixed.
