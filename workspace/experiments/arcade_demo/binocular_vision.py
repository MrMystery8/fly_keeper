"""True two-eye Arcade vision bridge (separate experiment; not Science Mode).

The frozen `embodiment.vision_bridge.VisionBridge` renders ONE camera (eye_left)
and drives ALL ~3335 R1-R6 receptors from it -- so the 2228 right-eye receptors
sample left-camera pixels and the right camera is discarded. This module keeps
that frozen bridge untouched and adds a genuinely BILATERAL Arcade bridge:

    left-eye receptors  (rootSide == 'L') <- eye_left  camera image
    right-eye receptors (rootSide == 'R') <- eye_right camera image

using each receptor's OWN uv coordinate (the upstream retinal layout). The eye
assignment comes from the audited manifest (binocular_eye_manifest.npz, built
from the connectome `rootSide` field). Nothing in the neural graph, retinal uv
layout, or MaleCNS dynamics is modified; we only feed each receptor the pixels
from the correct eye's camera.

Ablation conditions (for the binocular sanity study):
    both        : left<-left cam, right<-right cam            (true binocular)
    left_only   : right-eye receptors blinded (zeroed)        (left eye only)
    right_only  : left-eye receptors blinded (zeroed)         (right eye only)
    left_blind  : left camera blacked out (right cam still on) (lose left view)
    right_blind : right camera blacked out (left cam still on) (lose right view)
    both_blind  : both cameras blacked out                    (no vision floor)
    mono_left   : the CURRENT one-eye behaviour (all receptors <- left cam),
                  provided for apples-to-apples parity with Arcade Bridge v2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from doom.game import retinal_samples

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def load_eye_manifest():
    """Load the audited receptor->eye manifest (built by binocular_audit.py)."""
    p = OUT / "binocular_eye_manifest.npz"
    if not p.exists():
        raise FileNotFoundError(
            "binocular_eye_manifest.npz missing; run "
            "`python -m experiments.arcade_demo.binocular_audit` first.")
    m = np.load(p, allow_pickle=False)
    return dict(uv=m["uv"], left_mask=m["left_mask"], right_mask=m["right_mask"],
                retina_body_id=m["retina_body_id"])


class BinocularVisionBridge:
    """Bilateral fly vision: each eye's receptors sample its own camera.

    Renders eye_left and eye_right, samples R1-R6 luminance per eye using each
    receptor's uv, assembles the full luminance vector (left receptors from the
    left image, right receptors from the right image), and drives MaleCNS via the
    unchanged retinal luminance pathway. `condition` selects an ablation.
    """

    CONDITIONS = ("both", "left_only", "right_only", "left_blind", "right_blind",
                  "both_blind", "mono_left")

    def __init__(self, fly_body, brain, width=160, height=96, condition="both"):
        assert condition in self.CONDITIONS, condition
        self.fb = fly_body
        self.brain = brain
        self.width = width
        self.height = height
        self.condition = condition
        self._renderer = mujoco.Renderer(fly_body.model, height=height, width=width)
        # retinal uv layout + biological IDs (unchanged upstream mapping)
        self.uv = np.asarray(brain._brain.uv)
        self.retina_ids = [int(brain._brain.ids[i]) for i in brain._brain.retina]
        man = load_eye_manifest()
        self.left_mask = man["left_mask"].astype(bool)
        self.right_mask = man["right_mask"].astype(bool)
        # sanity: manifest receptor order must match the brain's retina order
        if not np.array_equal(man["retina_body_id"].astype(np.int64),
                              np.asarray(self.retina_ids, dtype=np.int64)):
            raise ValueError("eye manifest receptor order != brain retina order")

    def _render(self, camera):
        self._renderer.update_scene(self.fb.data, camera=camera)
        return self._renderer.render()

    def perceive(self):
        """Render both eyes, route each eye's receptors to its own camera image,
        apply the ablation condition, and drive MaleCNS. Returns (luminance, info).
        """
        cond = self.condition
        img_l = self._render("eye_left")
        img_r = self._render("eye_right")
        black = np.zeros_like(img_l)

        if cond == "left_blind":
            img_l = black
        elif cond == "right_blind":
            img_r = black
        elif cond == "both_blind":
            img_l = black; img_r = black

        lum_l = retinal_samples(img_l, self.uv)
        if cond == "mono_left":
            # current one-eye behaviour: ALL receptors sample the left image
            luminance = lum_l.copy()
        else:
            lum_r = retinal_samples(img_r, self.uv)
            luminance = lum_l.copy()
            luminance[self.right_mask] = lum_r[self.right_mask]

        # receptor-population ablations (blind an eye's receptors, not the camera)
        if cond == "left_only":
            luminance[self.right_mask] = 0.0
        elif cond == "right_only":
            luminance[self.left_mask] = 0.0

        self.brain.stimulate_retinal_luminance(self.retina_ids, luminance)
        info = dict(
            left_lum_mean=float(luminance[self.left_mask].mean()),
            right_lum_mean=float(luminance[self.right_mask].mean()),
            condition=cond)
        return luminance, info
