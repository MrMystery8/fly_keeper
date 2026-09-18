"""PHASE 4: trace calibration stimuli into MaleCNS, per eye, before/after fix.

For representative world ball positions we drive the UNCHANGED MaleCNS with the
per-eye luminance produced by a chosen retinal sampling MODE, then read:
    receptor activation      (retina spike/drive proxy: luminance sum per eye)
    optic-lobe activation     (selected OL neuron spikes per eye side)
    direction-selective delta (how OL activity shifts as the ball moves L->R)

Two sampling modes are compared to make the coverage bug and its fix explicit:

  mode="viewport"  (CURRENT):  each eye's receptors sample their OWN camera image
                   at the upstream overlapping-viewport UV (left uv_x in [0,0.6],
                   right uv_x in [0.4,1.0]). This is what BinocularVisionBridge
                   does today.

  mode="fullframe" (FIX):      each eye's receptors sample their OWN camera image
                   after REMAPPING that population's uv_x to span the eye's FULL
                   camera frame [0,1]. This removes the frontal coverage gap
                   WITHOUT moving the camera, changing its pose, or widening FOV.
                   It only corrects which pixels of the (unchanged) rendered eye
                   image each receptor reads.

Nothing in MaleCNS, the neural graph, LIF constants, or camera geometry is
changed. Only the receptor->pixel sampling is varied, and only for the arcade
binocular bridge experiment (Science Mode + frozen VisionBridge untouched).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from adapters.brain import MaleCNSBrain
from doom.game import retinal_samples
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.binocular_vision import load_eye_manifest
from experiments.arcade_demo.binocular_calibration import place_ball, hide_ball, W, H

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"

CASES = [
    ("FARLEFT", 1.2, 1.2), ("LEFT", 1.2, 0.6), ("CENTER", 1.2, 0.0),
    ("RIGHT", 1.2, -0.6), ("FARRIGHT", 1.2, -1.2),
]
Z = 0.7
SETTLE_STEPS = 6          # decisions to settle MaleCNS on a static stimulus
DECISION_MS = 20.0
# A short APPROACH sequence per case (ball sweeps in along -x through the target
# lateral position). Motion drives the graded/direction-selective visual path,
# which a single static frame does not, so optic-lobe neurons actually respond.
APPROACH_X = [2.4, 1.8, 1.2, 0.7, 0.4]


def remap_fullframe(uv, mask):
    """Return a uv copy where the masked population's uv_x is min-max stretched
    to [0,1] (span the eye's full camera frame). uv_y unchanged."""
    out = uv.copy()
    x = uv[mask, 0]
    lo, hi = x.min(), x.max()
    out[mask, 0] = (x - lo) / (hi - lo)
    return out


def build_uv(mode, uv, lm, rm):
    if mode == "viewport":
        return uv, uv
    if mode == "fullframe":
        uv_l = remap_fullframe(uv, lm)
        uv_r = remap_fullframe(uv, rm)
        return uv_l, uv_r
    raise ValueError(mode)


def ol_side_ids(brain):
    """Bilateral optic-lobe candidate pool IDs split by connectome side.

    Uses the SAME 600-neuron bilateral pool (binocular_pool.npz) the binocular
    dataset records, split L/R by connectome rootSide -- these are the real
    optic-lobe neurons the decoder consumes, so their per-side response is the
    meaningful Phase-4 optic-lobe readout."""
    m = np.load(OUT / "binocular_pool.npz", allow_pickle=False)
    pool = [int(x) for x in m["body_ids"]]
    side = m["somaSide"].astype(str)
    left = [int(i) for i, s in zip(pool, side) if s == "L"]
    right = [int(i) for i, s in zip(pool, side) if s == "R"]
    return left, right, pool


def drive_and_read(brain, retina_ids, luminance, read_ids):
    brain.reset()
    for _ in range(SETTLE_STEPS):
        brain.stimulate_retinal_luminance(retina_ids, luminance)
        brain.step(DECISION_MS)
    act = brain.read(read_ids)
    return {int(i): float(act[i]["spikes"]) for i in read_ids}


def run(mode="viewport"):
    man = load_eye_manifest()
    uv = man["uv"]; lm = man["left_mask"].astype(bool); rm = man["right_mask"].astype(bool)
    uv_l, uv_r = build_uv(mode, uv, lm, rm)

    brain = MaleCNSBrain(backend="cpu")
    retina_ids = [int(brain._brain.ids[i]) for i in brain._brain.retina]
    ol_left, ol_right, _ = ol_side_ids(brain)
    read_ids = ol_left + ol_right
    k = min(len(ol_left), len(ol_right))
    print(f"optic-lobe pool: {len(ol_left)} L + {len(ol_right)} R neurons")

    world = ArcadeGoalkeeperWorld(seed=770000)
    world.fly.set_pose(xy=(0.0, 0.0), yaw=0.0)
    world.fly.set_command(0, 0, 0, lateral=0.0, vertical=0.0)
    mujoco.mj_forward(world.model, world.data)
    r = mujoco.Renderer(world.model, height=H, width=W)

    def render(cam):
        r.update_scene(world.data, camera=cam); return r.render().copy()

    def per_eye_luminance(img_l, img_r):
        lum = retinal_samples(img_l, uv_l).copy()
        lum[rm] = retinal_samples(img_r, uv_r)[rm]
        return lum

    # background (ball hidden), for the retina-level ball signal reference
    hide_ball(world)
    bg_l = render("eye_left"); bg_r = render("eye_right")
    base = per_eye_luminance(bg_l, bg_r)

    rows = []
    for name, x_end, y in CASES:
        # retina-level ball signal at the near frame (strongest, clearest)
        place_ball(world, APPROACH_X[-1], y, Z)
        lum_near = per_eye_luminance(render("eye_left"), render("eye_right"))
        dret = np.maximum(lum_near - base, 0.0)
        left_ret = float(dret[lm].sum()); right_ret = float(dret[rm].sum())

        # drive MaleCNS through the moving approach and read OL response
        brain.reset()
        for xi in APPROACH_X:
            place_ball(world, xi, y, Z)
            lum = per_eye_luminance(render("eye_left"), render("eye_right"))
            brain.stimulate_retinal_luminance(retina_ids, lum)
            brain.step(DECISION_MS)
        act_ball = brain.read(read_ids)
        # baseline: same number of steps on background
        brain.reset()
        for _ in APPROACH_X:
            brain.stimulate_retinal_luminance(retina_ids, base)
            brain.step(DECISION_MS)
        act_bg = brain.read(read_ids)
        olL = sum(float(act_ball[i]["spikes"] - act_bg[i]["spikes"]) for i in ol_left)
        olR = sum(float(act_ball[i]["spikes"] - act_bg[i]["spikes"]) for i in ol_right)
        rows.append(dict(pos=name, world=[x_end, y, Z],
                         retina_left=round(left_ret, 4), retina_right=round(right_ret, 4),
                         ol_left_delta=round(olL, 3), ol_right_delta=round(olR, 3),
                         ol_lateral_delta=round(olL - olR, 3)))
        print(f"{name:>8} | retinaL={left_ret:8.3f} retinaR={right_ret:8.3f} "
              f"| OL_L={olL:8.2f} OL_R={olR:8.2f} OL(L-R)={olL-olR:8.2f}")
    brain.close()
    out = dict(mode=mode, settle_steps=SETTLE_STEPS, n_ol_per_side=k, rows=rows)
    (OUT / f"binocular_neural_trace_{mode}.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "viewport"
    print(f"=== PHASE 4 neural trace  mode={mode} ===")
    run(mode)
