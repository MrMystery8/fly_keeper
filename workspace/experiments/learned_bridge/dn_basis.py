"""Fixed left/right descending-neuron motor basis for the learned bridge.

WHY THESE DNs (causal evidence, not a guess)
--------------------------------------------
The neural-probe motor-perturbation experiment
(`workspace/outputs/neural_probe/motor_perturbation.json`) injected bounded
external current into two candidate descending populations *with no vision* and
measured the resulting body displacement through the UNCHANGED hierarchical
locomotion layer:

    drive LEFT DNs  [10162 DNp20_L, 10527 DNpe017_L]  -> dy = +0.459 cm  (moves +y = fly LEFT)
    drive RIGHT DNs [10059 DNp20_R, 555871 DNpe017_R] -> dy = -0.480 cm  (moves -y = fly RIGHT)
    drive none                                        -> dy = -0.064 cm  (drift)

So these four real MaleCNS descending neurons can causally and approximately
symmetrically produce useful left/right goalkeeper movement. The body-symmetry
control gave strafe_symmetry = 0.003 (near-perfect L/R mechanical symmetry).

THE BASIS
---------
We build a SIGNED OPPONENT basis B_lr over these DNs. A scalar lateral command
u in [-1, +1] maps to a bounded additive current injected into the DNs:

    I_bridge_DN(u) = u * B_lr        (units: mV-equivalent external current)

with the sign convention matched to the body:
    u < 0  -> drive LEFT DNs  -> fly strafes LEFT  (+y)
    u > 0  -> drive RIGHT DNs -> fly strafes RIGHT (-y)

This is the convention used by the teacher/decoder elsewhere in the project
(DescendingMotorDecoder: asym>0 == right-DNs dominate == lateral<0 == strafe
right). Here `u` is the DECODER-style command: u<0 = strafe left. We therefore
put positive current on LEFT DNs for u<0 and on RIGHT DNs for u>0.

The injection is a bounded ADDITIVE external current (adapters.brain.stimulate),
consumed by the next brain.step. The selected DNs keep evolving under the
existing MaleCNS LIF dynamics; their spike state / voltage are never overwritten.
The native visual->DN pathway is left fully intact and running; the bridge
current is ADDED on top of it (bridge ON) or omitted entirely (bridge OFF).
"""
from __future__ import annotations
import numpy as np

# Real MaleCNS descending-neuron body IDs, established by causal stimulation.
LEFT_DN = (10162, 10527)     # DNp20_L, DNpe017_L   -> body +y (fly's left)
RIGHT_DN = (10059, 555871)   # DNp20_R, DNpe017_R   -> body -y (fly's right)

# Per-DN drive magnitude (mV-equivalent). 25 mV reproduced the causal
# perturbation result and is within the adapter's +/-30 mV safety bound. The
# runtime gain that scales |u| is applied separately (bridge gain), so this is
# the reference amplitude at |u| = 1.
DRIVE_MV = 25.0


class DNMotorBasis:
    """Maps a scalar lateral command u in [-1,1] to bounded DN currents.

    left_ids / right_ids: the causal opponent populations (defaults above).
    drive_mv: reference per-neuron current magnitude at |u| = 1.
    gain: runtime scaling on |u| (bridge gain); the frozen value is stored with
          the bridge, not here, so this class stays a pure fixed basis.
    """

    def __init__(self, left_ids=LEFT_DN, right_ids=RIGHT_DN, drive_mv=DRIVE_MV,
                 max_abs_mv=30.0):
        self.left_ids = tuple(int(i) for i in left_ids)
        self.right_ids = tuple(int(i) for i in right_ids)
        self.drive_mv = float(drive_mv)
        self.max_abs_mv = float(max_abs_mv)
        # A stacked opponent basis vector over [left_ids..., right_ids...]:
        # B_lr[left] = -drive (fires for u<0), B_lr[right] = +drive (fires u>0).
        # The current for command u on neuron k is: -u*drive on left, +u*drive
        # on right, then rectified to >=0 (a neuron only receives positive
        # excitatory drive; the opponent is silent).
        self.ids = list(self.left_ids + self.right_ids)

    def currents(self, u):
        """Return (ids, currents) for command u in [-1,1] as a bounded, signed
        opponent injection. Left DNs get current for u<0, right DNs for u>0.
        Each neuron receives a non-negative excitatory current (opponent side
        is 0), magnitude |u|*drive_mv, clipped to the safety bound."""
        u = float(np.clip(u, -1.0, 1.0))
        mag = min(abs(u) * self.drive_mv, self.max_abs_mv)
        vals = []
        if u < 0:      # strafe left -> excite LEFT DNs
            vals = [mag] * len(self.left_ids) + [0.0] * len(self.right_ids)
        elif u > 0:    # strafe right -> excite RIGHT DNs
            vals = [0.0] * len(self.left_ids) + [mag] * len(self.right_ids)
        else:
            vals = [0.0] * len(self.ids)
        return list(self.ids), vals

    def inject(self, brain, u):
        """Queue the bridge current for command u into the brain (additive,
        bounded, one-step). Returns the injected (ids, currents) for logging."""
        ids, vals = self.currents(u)
        nz_ids = [i for i, v in zip(ids, vals) if v != 0.0]
        nz_vals = [v for v in vals if v != 0.0]
        if nz_ids:
            brain.stimulate(nz_ids, nz_vals)
        return ids, vals

    def readout_ids(self):
        """The DN body IDs whose spikes/voltage we monitor (all basis DNs)."""
        return list(self.ids)

    def describe(self):
        return dict(left_dn=list(self.left_ids), right_dn=list(self.right_ids),
                    drive_mv=self.drive_mv, max_abs_mv=self.max_abs_mv,
                    convention="u<0 -> excite LEFT DNs -> strafe +y (left); "
                               "u>0 -> excite RIGHT DNs -> strafe -y (right)",
                    causal_evidence="neural_probe/motor_perturbation.json: "
                               "L-drive dy=+0.459, R-drive dy=-0.480")
