"""Early-intent + RL execution bridge (WHERE / HOW separation).

WHERE: during a brief PRE-MOTION sensing window the keeper holds still while the
EarlyIntentEstimator (fit on corrected-env early-window data; 98% episode-level
L/C/R incl. right, held-out) accumulates a CONFIDENCE-AWARE lateral intent:
    intent_lat = P(right) - P(left)   (signed, small when uncertain/center)
    confidence = max P - 1/3          (margin above chance)
The intent is NOT a hard argmax command; it is an OBSERVATION to the execution
policy, which decides how hard/when to commit.

HOW: after the sensing window an execution policy (tiny MLP) maps
    [intent_lat, confidence, current 6-dim neural-stream embedding,
     fly_y, fly_z-stand, vy, vz, airborne, prev_u_lat, prev_u_vert]  (13 dims)
-> (u_lat, u_vert), both continuous and bounded. u_vert stays CONTINUOUS (never a
latched class) so LOW/MID/HIGH commitment is graded from ongoing MaleCNS activity.
Real DNs stay in the causal path (ArcadeDNBasis inject via the hybrid vertical
head + lateral opponent). During sensing the keeper is commanded to hold still
(tiny/zero lateral) but the vertical head may already begin a small rise so a
very fast high ball is not missed.

No ball truth in the deployed observation. The estimator's supervised L/C/R
labels are training-only (engineered discriminative readout).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))

from experiments.arcade_demo.early_intent import EarlyIntentEstimator, IntentState

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"

EXEC_OBS_NAMES = (
    "intent_lat", "confidence",
    "emb0", "emb1", "emb2", "emb3", "emb4", "emb5",
    "fly_y", "fly_z", "fly_vy", "fly_vz", "is_airborne",
    "prev_u_lat", "prev_u_vert",
)
EXEC_OBS_DIM = len(EXEC_OBS_NAMES)


@dataclass
class ExecPolicy:
    """Tiny tanh MLP: exec obs -> (u_lat, u_vert)."""
    w1: np.ndarray; b1: np.ndarray; w2: np.ndarray; b2: np.ndarray
    mean: np.ndarray; scale: np.ndarray

    def predict(self, obs):
        x = (np.asarray(obs, float) - self.mean) / self.scale
        h = np.tanh(x @ self.w1 + self.b1)
        return np.tanh(h @ self.w2 + self.b2)

    @classmethod
    def load(cls, path):
        d = np.load(OUT / path if not str(path).startswith("/") else path, allow_pickle=False)
        meta = json.loads(str(d["metadata"][0]))
        return cls(d["w1"], d["b1"], d["w2"], d["b2"], d["mean"], d["scale"]), meta

    def save(self, path, meta):
        np.savez(OUT / path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                 mean=self.mean, scale=self.scale, metadata=np.array([json.dumps(meta)]))


class EarlyIntentBridge:
    """Sensing-window early intent + execution policy over real DNs.

    `policy` None -> a simple hand-wired execution (intent*confidence lateral +
    hybrid vertical head) used to SEED supervised data. Otherwise an ExecPolicy.
    """

    uses_proprioception = True

    def __init__(self, hybrid, estimator: EarlyIntentEstimator, policy=None,
                 sense_steps=8, commit_gain=1.6):
        self.hybrid = hybrid
        self.est = estimator
        self.policy = policy
        self.sense_steps = int(sense_steps)
        self.commit_gain = float(commit_gain)
        self.enabled = True
        self.body_ids = list(hybrid.body_ids)
        self.n_windows = hybrid.n_windows
        self._pool_ids = list(estimator.encoder.pool_body_ids)
        self._pool_nw = estimator.encoder.n_windows
        self.reset()

    def reset(self):
        self.hybrid.reset()
        self._pool_buf = [np.zeros(len(self._pool_ids), np.float32) for _ in range(self._pool_nw)]
        self.intent = IntentState(self.est, sense_steps=self.sense_steps)
        self.previous = np.zeros(2)
        self.last_obs = np.zeros(EXEC_OBS_DIM)
        self.last_embed = np.zeros(self.est.encoder.embed_dim)
        self._t = 0

    @property
    def sensing(self):
        """True during the pre-motion sensing window (keeper should hold still,
        gait OFF, no DN drive) so the MaleCNS view matches the PASSIVE stationary
        distribution the early-intent estimator was fit on."""
        return self._t < self.sense_steps

    def gait_on(self):
        """Gait flag for the caller: 0 while sensing (hold still) else 1."""
        return 0.0 if self.sensing else 1.0

    def inject(self, brain):
        # CRITICAL: during the sensing window we must NOT drive the DNs. The
        # early-intent estimator was fit on PASSIVE stationary MaleCNS activity
        # (no injected current); injecting during sensing shifts the activity
        # off that distribution and corrupts the intent read. So inject only in
        # the execution phase (after the sensing window).
        if self._t >= self.sense_steps:
            return self.hybrid.inject(brain)
        return [], []

    def observe(self, brain):
        self.hybrid.observe(brain)
        rd = brain.read(self._pool_ids)
        self._pool_buf.append(np.array([rd[i]["spikes"] for i in self._pool_ids], np.float32))
        self._pool_buf.pop(0)
        # accumulate the early-intent evidence during the sensing window
        # (latches after the window).
        self.intent.update(self._pool_buf)

    def _proprio(self, world):
        fly = world.fly; vel = fly.data.qvel[fly.root_dofadr:fly.root_dofadr + 3]
        stand = float(getattr(fly, "_stand_z", None) or fly.position[2])
        return np.array([float(fly.position[1]), float(fly.position[2]) - stand,
                         float(vel[1]), float(vel[2]),
                         float(getattr(fly, "_airborne", False))], float)

    def observation(self, world):
        embed = self.est.encoder.encode(self._pool_buf)
        self.last_embed = embed
        return np.r_[self.intent.intent_lat, self.intent.confidence, embed,
                     self._proprio(world), self.previous].astype(float)

    def command(self, world):
        base_vert = float(np.asarray(self.hybrid.command(), float)[1])
        sensing = self._t < self.sense_steps
        obs = self.observation(world)
        if self.policy is not None:
            pred = self.policy.predict(obs)
            u_lat = float(pred[0]); u_vert = float(np.clip(pred[1], 0.0, 1.0))
        else:
            # hand-wired seed controller: lateral = confidence-scaled intent;
            # vertical = hybrid vertical head (continuous). Used only to seed
            # supervised data before RL.
            u_lat = float(np.clip(self.commit_gain * self.intent.intent_lat, -1.0, 1.0))
            u_vert = float(np.clip(base_vert, 0.0, 1.0))
        if sensing:
            # hold still laterally to keep the sensing view clean; allow a small
            # vertical readiness so a fast high ball is not missed outright.
            u_lat = 0.0
            u_vert = min(u_vert, 0.25)
        action = np.array([np.clip(u_lat, -1.0, 1.0), np.clip(u_vert, 0.0, 1.0)])
        self.hybrid._ema = action
        self.last_obs = obs; self.previous = action; self._t += 1
        return float(action[0]), float(action[1])
