"""Early-latch binocular bridge: read the vision/DN lateral commitment during a
brief STATIONARY sensing window (before the keeper's own motion corrupts the eye
view), latch it, then execute the committed lateral + a continuous vertical.

WHY (measured): the fused binocular lateral prediction separates LEFT/CENTER/
RIGHT correctly ONLY while the fly is stationary (held-still first-15-steps:
left -0.151 / center +0.024 / right +0.001, P(L<R)=1.00). Once the fly acts on
its own command the eye view is corrupted and the signal collapses/inverts
(closed-loop: left +0.061 / right -0.051, P=0.00). So the honest fix is not to
manufacture a signal but to READ the real MaleCNS/DN signal at the moment it is
valid, latch it, and commit -- exactly like a real keeper reads the shot then
dives. MaleCNS and the real DNs stay fully in the causal path; this only changes
WHEN the lateral decision is frozen.

The vertical channel is NOT latched (the loft is stable and vertical stays valid
through the shot); it tracks the frozen V3 vertical head continuously so LOW/MID/
HIGH still differentiate.
"""
from __future__ import annotations

import numpy as np


class EarlyLatchBridge:
    """Wrap the structured hybrid; latch lateral from a stationary sensing window.

    Protocol per decision (matches the existing bridge interface):
      inject(brain) / observe(brain) delegate to the wrapped hybrid so the real
      DNs are driven every step. command(world) returns (u_lat, u_vert):
        * during the first `sense_steps` decisions -> command lateral 0 (stay
          still, keep the eye view clean) while accumulating the hybrid's fused
          lateral estimate; vertical is issued normally.
        * at latch time -> freeze the accumulated lateral estimate (gained), then
          hold it for the rest of the shot.
    """

    uses_proprioception = True

    def __init__(self, hybrid, sense_steps=8, commit_gain=3.2, vert_gain=1.0):
        self.hybrid = hybrid
        self.sense_steps = int(sense_steps)
        self.commit_gain = float(commit_gain)
        self.vert_gain = float(vert_gain)
        self.enabled = True
        self.body_ids = hybrid.body_ids
        self.n_windows = hybrid.n_windows
        self.reset()

    def reset(self):
        self.hybrid.reset()
        self._t = 0
        self._sense = []
        self._latched = None

    def observe(self, brain):
        self.hybrid.observe(brain)

    def inject(self, brain):
        return self.hybrid.inject(brain)

    def command(self, world=None):
        base = np.asarray(self.hybrid.command(), float)   # (u_lat, u_vert)
        base_lat, base_vert = float(base[0]), float(base[1])
        if self._t < self.sense_steps:
            # stationary sensing: keep still laterally so the eye view stays clean
            self._sense.append(base_lat)
            u_lat = 0.0
        else:
            if self._latched is None:
                raw = float(np.mean(self._sense)) if self._sense else base_lat
                # commit: amplify the (small) clean early estimate to a decisive
                # lateral command, bounded.
                self._latched = float(np.clip(self.commit_gain * raw, -1.0, 1.0))
            u_lat = self._latched
        u_vert = float(np.clip(self.vert_gain * base_vert, 0.0, 1.0))
        self._t += 1
        # hand the executed command to the wrapped hybrid's EMA/DN injector
        self.hybrid._ema = np.array([u_lat, u_vert], float)
        return u_lat, u_vert
