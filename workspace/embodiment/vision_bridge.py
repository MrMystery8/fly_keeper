"""Body-relative vision -> existing MaleCNS R1-R6 retinal pathway.

The embodied fly perceives the world only through a rendered image taken from
its own head/eye pose. That image is fed into DoomFly's *unchanged*
`retinal_samples(rgb, uv)` using the model's mapped R1-R6 receptor UV layout,
then delivered as luminance drive via `MaleCNSBrain.stimulate_retinal_luminance`.

No privileged state (ball coordinates, velocity, target point) is ever given to
the brain. Only pixels reach MaleCNS, exactly as in the original FlyKeeper.

The compound eye is approximated by a single wide-FOV camera on the fly's head.
Using the flybody `eye_left`/`eye_right` cameras (fovy 140 deg) or a synthetic
forward camera is a modeling choice; the R1-R6 UV sampling is the reused,
biologically-derived part.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "upstream" / "doomfly"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
from doom.game import retinal_samples  # unchanged upstream retinal sampler


class VisionBridge:
    """Renders a fly-relative RGB frame and maps it to R1-R6 luminance."""

    def __init__(self, fly_body, brain, width=160, height=96, camera="eye_left",
                 condition="normal"):
        self.fb = fly_body
        self.brain = brain
        self.width = width
        self.height = height
        self.condition = condition
        self._renderer = mujoco.Renderer(fly_body.model, height=height, width=width)
        # Resolve camera: named model camera, or a free tracking camera id.
        self._camera = camera
        # The mapped R1-R6 receptor UV layout and their biological IDs, from the
        # prepared MaleCNS graph (unchanged).
        self.uv = brain._brain.uv
        self.retina_ids = [int(brain._brain.ids[i]) for i in brain._brain.retina]
        self._static_frame = None  # for the static-ball control

    def set_static_reference(self, rgb):
        self._static_frame = rgb.copy()

    def render(self) -> np.ndarray:
        """Return the current fly-relative RGB frame (H, W, 3) uint8."""
        self._renderer.update_scene(self.fb.data, camera=self._camera)
        return self._renderer.render()

    def _apply_condition(self, rgb):
        c = self.condition
        if c == "blind":
            return np.zeros_like(rgb)
        if c == "mirrored":
            return rgb[:, ::-1].copy()
        if c == "static_ball" and self._static_frame is not None:
            return self._static_frame
        return rgb

    def perceive(self, rgb=None):
        """Render (or accept) a frame, sample R1-R6, and drive the brain.

        Returns (luminance_vector, rgb_used) for diagnostics. The luminance is
        queued into MaleCNS; the caller then advances `brain.step(...)`.
        """
        if rgb is None:
            rgb = self.render()
        used = self._apply_condition(rgb)
        luminance = retinal_samples(used, self.uv)
        self.brain.stimulate_retinal_luminance(self.retina_ids, luminance)
        return luminance, used
