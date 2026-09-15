"""Engineered decoder: MaleCNS descending-neuron activity -> locomotion commands.

BIOLOGICAL (reused): the neuron identities are real MaleCNS descending neurons.
We reuse the same left/right DNp20 body IDs the existing FlyKeeper used
(10162 = left, 10059 = right), which trace to the DoomFly experimental BCI
mapping. Optionally, a forward-drive population can be added.

ENGINEERED (our modeling): the mapping from spike-count asymmetry to a turn /
forward / gait command, the activity window, normalization, thresholds, and
gains. The fly does not natively "know" goalkeeping; these are experimenter
choices that turn descending activity into intent for the CPG locomotion layer.

The decoder receives ONLY neural readouts (spike counts / voltages) from the
brain. It never sees ball position, velocity, or any game state.
"""
from __future__ import annotations
import math
import numpy as np


class DescendingMotorDecoder:
    """Turn/forward/gait command from descending-neuron spike counts.

    left_ids / right_ids: MaleCNS body IDs whose relative activity votes for a
    left vs right lateral move. forward_ids (optional): activity that gates
    forward creeping. Everything is a documented engineered gain/threshold.
    """

    # Defaults inherited from the existing FlyKeeper DNp20 assignment.
    LEFT_IDS = (10162,)
    RIGHT_IDS = (10059,)

    def __init__(self, left_ids=None, right_ids=None, forward_ids=None,
                 turn_gain=1.0, forward_bias=0.35, deadband=0.0,
                 smoothing=0.4):
        self.left_ids = tuple(left_ids) if left_ids else self.LEFT_IDS
        self.right_ids = tuple(right_ids) if right_ids else self.RIGHT_IDS
        self.forward_ids = tuple(forward_ids) if forward_ids else ()
        self.turn_gain = float(turn_gain)
        self.forward_bias = float(forward_bias)
        self.deadband = float(deadband)
        self.smoothing = float(smoothing)
        self._turn = 0.0

    def readout_ids(self):
        return list(dict.fromkeys(self.left_ids + self.right_ids + self.forward_ids))

    def decode(self, activity: dict) -> tuple[dict, dict]:
        """activity: {body_id: {'spikes': int, 'voltage_mv': float}}.

        Returns (command, diagnostics). command has keys forward, turn, gait_on,
        plus a discrete `move` label (LEFT / RIGHT / STAY) for logging and for
        parity with the existing 2D goalkeeper.
        """
        left = float(sum(activity[i]["spikes"] for i in self.left_ids))
        right = float(sum(activity[i]["spikes"] for i in self.right_ids))
        total = left + right
        # Normalized asymmetry in [-1, 1]: +1 = fully right, -1 = fully left.
        asym = 0.0 if total <= 0 else (right - left) / total
        # Discrete label matches the original decoder's tie/threshold logic.
        if left > right and left:
            move = "LEFT"
        elif right > left and right:
            move = "RIGHT"
        else:
            move = "STAY"

        # Map the left/right descending asymmetry to a LATERAL strafe command.
        # move==LEFT (left DNs win) => strafe toward the fly's +y (its left).
        # Sign convention: lateral > 0 strafes toward body +y.
        lateral_raw = 0.0
        if abs(asym) > self.deadband:
            # asym > 0 means RIGHT-side DNs dominate => strafe right (lateral<0).
            lateral_raw = float(np.clip(-self.turn_gain * asym, -1, 1))
        # Exponential smoothing keeps the body from jittering frame to frame.
        self._turn = (1 - self.smoothing) * self._turn + self.smoothing * lateral_raw

        gait_on = 1.0 if abs(self._turn) > 1e-3 else 0.0
        command = {"forward": 0.0, "lateral": self._turn, "turn": 0.0,
                   "gait_on": gait_on, "move": move}
        diagnostics = {"left_spikes": left, "right_spikes": right,
                       "asymmetry": asym, "lateral": self._turn}
        return command, diagnostics

    def reset(self):
        self._turn = 0.0


class LeakyIntegratorDecoder:
    """Timing-aware descending-neuron readout (diagnostics-justified).

    The neural probe showed the fixed connectome carries left/right shot
    direction to the descending neurons (DNp20, DNpe017) ONLY in the TIMING of
    their spikes across the ~120 ms approach, not in a 20 ms total spike count
    (which is at chance). A leaky temporal integral of the right-vs-left DN
    activity contrast separates left from right shots (permutation p < 0.001).

    This decoder implements exactly that interpretable, causal, online feature:
    a leaky integrator of (sum right DNs - sum left DNs), mapped to a lateral
    strafe command. It uses ONLY descending spike counts - no ball state, no
    trained classifier, no privileged information. The single calibrated
    quantity is `sign`, which fixes the polarity found in the probe (left shots
    produce a higher integrated contrast).

    ENGINEERED: the time constant `tau`, gain, deadband, saturation, and sign.
    BIOLOGICAL: the neuron identities and their spikes.
    """

    LEFT_IDS = (10162, 10527)     # DNp20_L, DNpe017_L
    RIGHT_IDS = (10059, 555871)   # DNp20_R, DNpe017_R

    def __init__(self, left_ids=None, right_ids=None, tau=5.0, gain=0.6,
                 deadband=0.15, sign=-1.0, baseline_subtract=False):
        self.left_ids = tuple(left_ids) if left_ids else self.LEFT_IDS
        self.right_ids = tuple(right_ids) if right_ids else self.RIGHT_IDS
        self.tau = float(tau)          # leak time constant, in decision windows
        self.gain = float(gain)
        self.deadband = float(deadband)
        self.sign = float(sign)        # polarity calibrated from the probe
        # If set, subtract a slow running mean of the contrast so a constant
        # baseline firing asymmetry (present even when blind) cannot drive a
        # standing strafe bias; only DEVIATIONS from baseline move the fly.
        self.baseline_subtract = bool(baseline_subtract)
        self._integral = 0.0
        self._baseline = 0.0

    def readout_ids(self):
        return list(dict.fromkeys(self.left_ids + self.right_ids))

    def decode(self, activity: dict):
        left = float(sum(activity[i]["spikes"] for i in self.left_ids))
        right = float(sum(activity[i]["spikes"] for i in self.right_ids))
        inst = right - left
        if self.baseline_subtract:
            # slow running mean (tau ~ 10x the integrator) removes constant bias
            self._baseline = 0.98 * self._baseline + 0.02 * inst
            inst = inst - self._baseline
        a = np.exp(-1.0 / self.tau)
        self._integral = a * self._integral + inst
        signal = self.sign * self.gain * self._integral
        lateral = 0.0
        if abs(signal) > self.deadband:
            lateral = float(np.clip(signal, -1, 1))
        move = "LEFT" if lateral > 0.05 else "RIGHT" if lateral < -0.05 else "STAY"
        gait_on = 1.0 if abs(lateral) > 1e-3 else 0.0
        command = {"forward": 0.0, "lateral": lateral, "turn": 0.0,
                   "gait_on": gait_on, "move": move}
        diagnostics = {"left_spikes": left, "right_spikes": right,
                       "integral": round(self._integral, 3), "lateral": lateral}
        return command, diagnostics

    def reset(self):
        self._integral = 0.0
        self._baseline = 0.0
