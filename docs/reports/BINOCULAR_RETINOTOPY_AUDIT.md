# Binocular retinotopy and camera-geometry audit (corrected)

This supersedes the earlier version of this report. A systematic
visibility/geometry calibration overturned the previous conclusion that "the
right camera renders did not visibly contain ball pixels." The right camera CAN
see the ball. The real defect is a **retinal-viewport / receptor-coverage
mismatch**, not a camera-geometry defect.

Experiments (analysis only; Science Mode, the frozen `VisionBridge`, MaleCNS,
the neural graph, LIF constants, and the upstream retinal UV map are all
untouched):

- `experiments/arcade_demo/binocular_calibration.py`
  → `workspace/outputs/arcade_demo/binocular_calibration.json`
- `experiments/arcade_demo/binocular_overlay.py`
  → `workspace/outputs/arcade_demo/calib_overlays/*.png`
- `experiments/arcade_demo/binocular_neural_trace.py`
  → `workspace/outputs/arcade_demo/binocular_neural_trace_{viewport,fullframe}.json`

---

## 1. Can the ball actually be seen by the right camera? (PHASE 1)

Yes. A deterministic 45-position grid was rendered with the fly held at its
standing goal-line pose (xy=(0,0), yaw=0, facing +x toward incoming shots):

```
x (depth) in {2.4, 1.2, 0.4} cm
y (lateral) in {+1.2, +0.6, 0.0, -0.6, -1.2} cm   (+y = world-left)
z (height) in {0.45, 0.7, 1.25} cm                 (LOW / MID / HIGH)
```

For each position both eye cameras were rendered and the ball's visibility was
measured by a ball-removal render difference (centroid, pixel area).

```
left  camera sees the ball : 35 / 45 positions
right camera sees the ball : 35 / 45 positions
both cameras see it        : 25 / 45   (binocular overlap band)
neither                    :  0 / 45
```

The left camera covers world-LEFT through RIGHT and misses FARRIGHT; the right
camera covers world-RIGHT through LEFT and misses FARLEFT. This is symmetric and
expected. The prior "right eye is blind" finding was an artifact of sampling
only three representative fired-shot frames.

## 2. Is the right-eye camera geometry correct? (PHASE 2)

Yes. Both eye cameras are defined on the `head` body of the vendor flybody MJCF
(`fruitfly.xml`), as genuine mirror counterparts:

```
eye_left   local pos (-0.0219, 0.0131, 0)  quat (0.474, 0.688,  0.344,  0.429)  fovy 140
eye_right  local pos (+0.0219, 0.0131, 0)  quat (0.474, 0.688, -0.344, -0.429)  fovy 140
```

Live world optical axes when the keeper stands on the goal line:

```
left  optical axis  ~ (+0.39, +0.92, +0.09)   -> points world-LEFT
right optical axis  ~ (+0.39, -0.92, +0.09)   -> points world-RIGHT
each axis vs the +x (incoming-ball) axis : 67 deg
interocular axis angle                   : 132.7 deg
```

These are laterally-pointing compound eyes with a narrow frontal binocular
overlap — biologically appropriate. The cameras were **not** re-pointed at the
penalty spot and the FOV was **not** widened. Correcting geometry here would be
wrong: the geometry is already correct.

## 3. Is the right retinal sampling correctly oriented? (PHASE 3) — THE DEFECT

The upstream retinal UV projection (`doom/prepare.py`) lays both eye receptor
populations onto **overlapping viewports of a single shared image**:

```
left  receptors : uv_x = 0.60 * z          -> uv_x in [0.00, 0.60]  (image px 0-95 of 160)
right receptors : uv_x = 0.40 + 0.60*(1-z) -> uv_x in [0.40, 1.00]  (image px 64-159)
                  (the (1-z) term horizontally MIRRORS the right eye vs the left)
```

That layout assumes each eye's useful field maps onto opposite halves of one
forward image. But each eye camera points 67 deg laterally, so a near-**frontal
(CENTER) ball lands near the INNER edge of each camera image** — precisely where
the receptor coverage is absent:

```
world position   left-cam ball px   right-cam ball px
CENTER (mid x)   ~130  (> 95 -> outside left receptors' 0-95 window)
                 ~30   (< 64 -> outside right receptors' 64-159 window)
```

Consequence: for CENTER shots **both** receptor populations receive ~zero ball
signal even though both cameras clearly see the ball. The two eyes hand off
(left covers world-left, right covers world-right) with a **blind wedge in the
middle**, instead of overlapping in the frontal zone. Overlays in
`calib_overlays/CENTER_MID_*.png` show the ball's red centroid sitting outside
the green receptor sample cloud in both eyes.

Receptor-level ball signal (sum of positive luminance delta) confirms it:

```
position   L recept (viewport)   R recept (viewport)
FARLEFT      18.1                  0.0
LEFT         17.3                  0.0
CENTER        0.0                  0.0     <-- frontal blind wedge
RIGHT         0.0                 36.1
FARRIGHT      0.0                 29.9
```

## The fix (receptor→pixel sampling only; no camera change)

The upstream shared-image viewport layout is a frozen Science-Mode artifact and
is left untouched. The **arcade binocular bridge** instead corrects how each
eye's receptors sample **that eye's own camera image**: each population's `uv_x`
is stretched to span the full frame `[0,1]` of its own eye camera, preserving
retinotopic order and the right eye's mirrored orientation; `uv_y` is unchanged.
This moves no camera, changes no pose, and widens no FOV — it only fixes which
pixels of the (unchanged) rendered eye image each receptor reads.

Implemented as `BinocularVisionBridge(..., retina_map=...)` in
`binocular_vision.py`:
- `retina_map="viewport"` (default) — original behaviour, preserved.
- `retina_map="fullframe"` — the correction above.

Receptor-level result (same grid, MID height, near frame):

```
position   L recept: viewport -> fullframe    R recept: viewport -> fullframe
LEFT         17.3  ->  66.9                     0.0  ->   7.4
CENTER        0.0  -> 201.0                     0.0  -> 268.8   <-- gap closed
RIGHT         0.0  ->   3.9                    36.1  ->  84.2
```

The frontal blind wedge closes and a genuine binocular overlap band appears at
LEFT/CENTER/RIGHT; the far periphery correctly remains monocular.

## 4. Does right-eye MaleCNS activity contain direction information? (PHASE 4)

Retina-level drive into the unchanged MaleCNS confirms the fix restores strong,
spatially-appropriate input to both eye populations, including the previously
silent CENTER zone. A static-position optic-lobe probe is a weak instrument
(single-digit, noisy spike deltas over a short settle) and is not the right
measurement for direction selectivity, which depends on the full temporal
window dynamics of a moving approach. The authoritative neural-level test is the
offline decoding on real reset-corrected episodes
(`STRUCTURED_BINOCULAR_LATERAL_REPORT.md`, Phase 6), which measures whether the
per-eye streams carry *decodable* lateral/direction information.

## Consequence for the dataset

Because the fix changes retinal activation (and therefore downstream MaleCNS
visual activity), any binocular dataset generated under `retina_map="viewport"`
is stale. Corrected datasets are regenerated with `retina_map="fullframe"`
(preserving every safeguard: per-episode reset, terminal resolution, no padding,
no post-terminal samples, no cross-episode temporal leakage, train/val/test
episode separation, balanced 3x3). See the structured-lateral report for the
decoding and closed-loop consequences.
