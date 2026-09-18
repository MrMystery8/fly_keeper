"""Compact richer-observation binocular action policy (Iteration 2).

Fixes the two failures of RL v1:
  A. RIGHT-side control (was 0%): the observation now carries a LEFT/RIGHT
     NEURAL-STREAM DISCRIMINATIVE MaleCNS embedding (RichBinocularEncoder,
     3 dims/stream) that recovers right-side direction (offline recall
     0.17 -> 0.62), instead of the two collapsed scalars that inverted under
     self-motion. (Streams = connectome hemisphere groups, not a proven 1:1 eye
     mapping; input is still genuinely binocular.)
  B. always-takeoff (was 100%): the policy outputs a continuous u_vert and is
     trained (teacher + graded RL shaping) so LOW center stays low; it no longer
     jumps maximally on every shot.

OBSERVATION (14 dims, causal, NO ball truth):
    left_embed  (3)   discriminative LEFT neural-stream MaleCNS features
    right_embed (3)   discriminative RIGHT neural-stream MaleCNS features
                       (streams NOT mixed before their per-stream encoders)
    base_vert   (1)   frozen V3 vertical-head scalar (real-DN vertical signal)
    prev_lat, prev_vert (2)   previous action (self-motion context)
    fly_y, fly_z-stand, vy, vz, airborne (5)   proprioception (self-motion)

The stream embeddings + proprioception + previous action let the policy infer
ball-motion vs keeper-self-motion visual change WITHOUT any privileged camera
subtraction. Output (u_lat,u_vert) drives the REAL ArcadeDNBasis DNs (vertical
head still injected); no direct visual->body path, no MuJoCo pose/velocity write.
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

from experiments.arcade_demo.binocular_rich_encoder import RichBinocularEncoder

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"

RICH_OBS_NAMES = (
    "left_embed_0", "left_embed_1", "left_embed_2",
    "right_embed_0", "right_embed_1", "right_embed_2",
    "base_vert", "prev_lat", "prev_vert",
    "fly_y", "fly_z", "fly_vy", "fly_vz", "is_airborne",
)
OBS_DIM = len(RICH_OBS_NAMES)


@dataclass
class RichPolicy:
    """Two-output tanh MLP over the 14-dim rich observation (audit-small)."""
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def predict(self, obs):
        x = (np.asarray(obs, float) - self.mean) / self.scale
        h = np.tanh(x @ self.w1 + self.b1)
        return np.tanh(h @ self.w2 + self.b2)

    @classmethod
    def load(cls, path):
        d = np.load(OUT / path if not str(path).startswith("/") else path,
                    allow_pickle=False)
        meta = json.loads(str(d["metadata"][0]))
        return cls(d["w1"], d["b1"], d["w2"], d["b2"], d["mean"], d["scale"]), meta

    def save(self, path, metadata):
        np.savez(OUT / path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                 mean=self.mean, scale=self.scale,
                 metadata=np.array([json.dumps(metadata)]))


class RichActionBridge:
    """Rich-observation binocular action bridge (drop-in for ArcadeController).

    Maintains its OWN rolling buffer over the encoder's 600-neuron pool (so the
    discriminative embedding matches the offline dataset construction) AND keeps
    the frozen hybrid's vertical head + DN basis so the real DNs stay in the loop.
    `policy` may be None (returns the frozen hybrid base command) for round-0
    seeding, or a RichPolicy in 'full' or 'residual' mode.
    """

    uses_proprioception = True

    def __init__(self, hybrid, encoder: RichBinocularEncoder, policy=None,
                 mode="full", residual_scale=0.35):
        self.hybrid = hybrid
        self.encoder = encoder
        self.policy = policy
        self.mode = mode
        self.residual_scale = float(residual_scale)
        self.enabled = True
        # runtime needs these attributes on a bridge
        self.body_ids = list(hybrid.body_ids)
        self.n_windows = hybrid.n_windows
        # pool buffer for the encoder (separate neuron set from the hybrid heads)
        self._pool_ids = list(encoder.pool_body_ids)
        self._pool_nw = encoder.n_windows
        self.reset()

    def reset(self):
        self.hybrid.reset()
        self._pool_buf = [np.zeros(len(self._pool_ids), np.float32)
                          for _ in range(self._pool_nw)]
        self.previous = np.zeros(2, dtype=float)
        self.last_obs = np.zeros(OBS_DIM, dtype=float)
        self.last_base = np.zeros(2, dtype=float)
        self.last_embed = np.zeros(self.encoder.embed_dim, dtype=float)

    def inject(self, brain):
        return self.hybrid.inject(brain)

    def observe(self, brain):
        self.hybrid.observe(brain)
        rd = brain.read(self._pool_ids)
        self._pool_buf.append(np.array([rd[i]["spikes"] for i in self._pool_ids],
                                       np.float32))
        self._pool_buf.pop(0)

    def _proprioception(self, world):
        fly = world.fly
        vel = fly.data.qvel[fly.root_dofadr:fly.root_dofadr + 3]
        stand = float(getattr(fly, "_stand_z", None) or fly.position[2])
        return np.asarray((float(fly.position[1]), float(fly.position[2]) - stand,
                           float(vel[1]), float(vel[2]),
                           float(getattr(fly, "_airborne", False))), float)

    def observation(self, world, base):
        embed = self.encoder.encode(self._pool_buf)
        self.last_embed = embed
        return np.r_[embed, float(base[1]), self.previous,
                     self._proprioception(world)].astype(float)

    def command(self, world):
        base = np.asarray(self.hybrid.command(), float)   # frozen hybrid (u_lat,u_vert)
        obs = self.observation(world, base)
        if self.policy is None or not self.enabled:
            action = base.copy()
        else:
            pred = self.policy.predict(obs)
            if self.mode == "full":
                action = pred
            else:  # residual
                action = base + self.residual_scale * pred
        action = np.clip(action, (-1.0, 0.0), (1.0, 1.0))
        # hand command to the frozen hybrid's EMA/DN injector (real DNs)
        self.hybrid._ema = np.asarray(action, float)
        self.last_obs = obs
        self.last_base = base
        self.previous = action
        return float(action[0]), float(action[1])
