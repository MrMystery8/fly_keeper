"""Small binocular residual action policy for the engineered Arcade keeper.

The policy deliberately sits *between* frozen MaleCNS stream readouts and the
existing :class:`ArcadeDNBasis`, rather than driving MuJoCo forces.  Its only
visual inputs are the separate left/right late-fusion stream predictions made
from MaleCNS activity.  The remaining inputs are causal keeper proprioception
and the preceding action.  Ball truth is used only by the dataset teacher.

At runtime the path is therefore:

    two retinas -> frozen MaleCNS -> L/R stream heads -> tiny MLP residual
    -> real descending neurons -> DN decoder -> force-driven ArcadeFlyBody.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


OBS_NAMES = (
    "left_stream", "right_stream", "vertical_stream",
    "previous_lat", "previous_vert", "fly_y", "fly_z",
    "fly_vy", "fly_vz", "is_airborne",
)


@dataclass
class TinyPolicy:
    """Two-output tanh MLP, intentionally small enough to audit exactly."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def predict(self, observation: np.ndarray) -> np.ndarray:
        x = (np.asarray(observation, float) - self.mean) / self.scale
        hidden = np.tanh(x @ self.w1 + self.b1)
        return np.tanh(hidden @ self.w2 + self.b2)

    @classmethod
    def load(cls, path: Path) -> tuple["TinyPolicy", dict]:
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["metadata"][0]))
        return cls(data["w1"], data["b1"], data["w2"], data["b2"],
                   data["mean"], data["scale"]), metadata

    def save(self, path: Path, metadata: dict) -> None:
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                 mean=self.mean, scale=self.scale,
                 metadata=np.array([json.dumps(metadata)]))


class BinocularActionPolicyBridge:
    """Wrap a binocular neural bridge with a bounded learned action residual.

    ``visual_bridge`` must expose the historical ``observe``, ``command`` and
    ``inject`` protocol.  In particular the structured late-fusion bridge
    exposes separate ``last_pL``/``last_pR`` stream predictions.  The policy
    alters high-level intents only; the wrapped bridge still injects those
    commands through its real DN basis.
    """

    uses_proprioception = True

    def __init__(self, visual_bridge, policy: TinyPolicy, residual_scale=.35,
                 mode="residual"):
        self.visual_bridge = visual_bridge
        self.policy = policy
        self.residual_scale = float(residual_scale)
        if mode not in ("residual", "full", "blend"):
            raise ValueError("mode must be residual, full, or blend")
        self.mode = mode
        self.enabled = True
        self.body_ids = visual_bridge.body_ids
        self.n_windows = visual_bridge.n_windows
        self.reset()

    def reset(self):
        self.visual_bridge.reset()
        self.previous = np.zeros(2, dtype=float)
        self.last_observation = np.zeros(len(OBS_NAMES), dtype=float)
        self.last_base = np.zeros(2, dtype=float)
        self.last_residual = np.zeros(2, dtype=float)

    def observe(self, brain):
        self.visual_bridge.observe(brain)

    def _proprioception(self, world) -> np.ndarray:
        fly = world.fly
        vel = fly.data.qvel[fly.root_dofadr:fly.root_dofadr + 3]
        stand_z = float(getattr(fly, "_stand_z", None) or fly.position[2])
        return np.asarray((float(fly.position[1]), float(fly.position[2]) - stand_z,
                           float(vel[1]), float(vel[2]),
                           float(getattr(fly, "_airborne", False))), float)

    def observation(self, world, base: np.ndarray) -> np.ndarray:
        # Separate L/R values are intentionally retained until this final,
        # learned fusion point.  V3 has no stream-specific lateral fields, so
        # fail closed to its single lateral prediction rather than fabricate an
        # eye identity.
        left = float(getattr(self.visual_bridge, "last_pL", base[0]))
        right = float(getattr(self.visual_bridge, "last_pR", base[0]))
        return np.r_[left, right, float(base[1]), self.previous,
                     self._proprioception(world)]

    def command(self, world):
        base = np.asarray(self.visual_bridge.command(), dtype=float)
        obs = self.observation(world, base)
        predicted = self.policy.predict(obs) if self.enabled else np.zeros(2)
        residual = predicted * self.residual_scale
        if self.mode == "full":
            action = predicted
        elif self.mode == "blend":
            # ``residual_scale`` is the full-policy mixture coefficient here:
            # zero is the frozen safe base, one is the actionous full policy.
            action = (1.0 - self.residual_scale) * base + self.residual_scale * predicted
        else:
            action = base + residual
        action = np.clip(action, (-1.0, 0.0), (1.0, 1.0))
        # The next 20 ms neural step consumes this command through the wrapped
        # bridge's existing DN injector.  This is the only hand-off; no body
        # command is ever written here.
        self.visual_bridge._ema = np.asarray(action, dtype=float)
        self.last_observation = obs
        self.last_base, self.last_residual = base, np.asarray(residual, float)
        self.previous = action
        return float(action[0]), float(action[1])

    def inject(self, brain):
        # The policy never injects directly: use the frozen bridge's DN basis.
        # Its EMA is the command consumed by this step, as in the existing loop.
        return self.visual_bridge.inject(brain)
