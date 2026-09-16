"""2-output (lateral + vertical) learned bridge for the arcade fly.

Same minimal philosophy as the frozen bridge, extended to two commands:

    x (207 optic-lobe features x 4 windows)
      -> normalize -> linear map W (feature_dim -> 2) + b -> tanh
      -> u = [u_lateral in (-1,1), u_vertical_raw in (-1,1)]
    u_vertical = max(0, u_vertical_raw)         # vertical is one-sided (lift up)
      -> ArcadeDNBasis.inject(brain, u_lat, u_vert)  (bounded current on real DNs)
      -> real DNs evolve under fixed MaleCNS dynamics
      -> arcade decoder reads those DNs -> lateral strafe + vertical hop/flight
      -> ArcadeFlyBody

The feature extractor is IDENTICAL to the frozen one (same 207 neurons, same
4x20ms causal windows). Only the linear head has 2 outputs instead of 1. Trained
and run on the Metal backend so model and runtime match.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class MetalSafeFeatureExtractor:
    """Causal temporal spike-count features read by BODY ID (CPU + Metal safe).

    Same 4x20ms causal ring buffer and newest-first ordering as the frozen
    VisualFeatureExtractor, but reads spike counts via brain.read(body_ids)
    instead of brain._brain.counts (which is GPU-resident / empty on Metal).
    The body_ids are the frozen manifest's 207 selected neurons, in order, so
    the feature vector matches the frozen layout exactly.
    """

    def __init__(self, body_ids, n_windows=4):
        self.body_ids = [int(x) for x in body_ids]
        self.n_neurons = len(self.body_ids)
        self.n_windows = int(n_windows)
        self.dim = self.n_neurons * self.n_windows
        self._buf = None
        self.reset()

    def reset(self):
        self._buf = [np.zeros(self.n_neurons, dtype=np.float32)
                     for _ in range(self.n_windows)]

    def observe(self, brain):
        rd = brain.read(self.body_ids)
        counts = np.array([rd[i]["spikes"] for i in self.body_ids],
                          dtype=np.float32)
        self._buf.append(counts)
        if len(self._buf) > self.n_windows:
            self._buf.pop(0)

    def features(self, order="newest_first"):
        wins = self._buf[-self.n_windows:]
        if order == "newest_first":
            wins = wins[::-1]
        return np.concatenate(wins).astype(np.float64)


class LinearBridge2D:
    """Normalized linear map from temporal features to 2 commands (lat, vert).

    u = tanh( ((x - mu) / sd) @ W + b )   with W: (feature_dim, 2), b: (2,)
    """

    def __init__(self, W, b, mu, sd, feature_order="newest_first"):
        self.W = np.asarray(W, dtype=np.float64).reshape(-1, 2)
        self.b = np.asarray(b, dtype=np.float64).reshape(2)
        self.mu = np.asarray(mu, dtype=np.float64).ravel()
        self.sd = np.asarray(sd, dtype=np.float64).ravel()
        self.sd = np.where(self.sd < 1e-6, 1.0, self.sd)
        self.feature_order = feature_order

    @property
    def n_params(self):
        return int(self.W.size + self.b.size)

    def predict(self, x):
        xn = (np.asarray(x, dtype=np.float64) - self.mu) / self.sd
        raw = xn @ self.W + self.b
        return np.tanh(raw)          # shape (2,)

    def save(self, path, metadata=None):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.W, b=self.b, mu=self.mu, sd=self.sd,
                 feature_order=np.array([self.feature_order]),
                 metadata=np.array([json.dumps(metadata or {})]))
        return str(path)

    @classmethod
    def load(cls, path):
        a = np.load(path, allow_pickle=False)
        m = cls(a["W"], a["b"], a["mu"], a["sd"],
                feature_order=str(a["feature_order"][0]))
        m.loaded_metadata = json.loads(str(a["metadata"][0])) if "metadata" in a else {}
        return m


class ArcadeLearnedBridge:
    """Feature extractor + 2D linear model + ArcadeDNBasis, with runtime gains.

    Produces bounded additive DN current for (lateral, vertical) each decision;
    the arcade decoder reads those DNs downstream. `enabled` toggles the bridge.
    """

    def __init__(self, extractor: VisualFeatureExtractor, model: LinearBridge2D,
                 basis, lat_gain=1.0, vert_gain=1.0, enabled=True,
                 cmd_smoothing=0.0):
        self.extractor = extractor
        self.model = model
        self.basis = basis
        self.lat_gain = float(lat_gain)
        self.vert_gain = float(vert_gain)
        self.enabled = bool(enabled)
        self.cmd_smoothing = float(cmd_smoothing)
        self._u = np.zeros(2)
        self._ema = np.zeros(2)

    def reset(self):
        self.extractor.reset()
        self._u = np.zeros(2)
        self._ema = np.zeros(2)

    def observe(self, brain):
        self.extractor.observe(brain)

    def command(self):
        """Return (u_lat, u_vert) in the arcade convention (vert one-sided)."""
        if not self.enabled:
            self._u = np.zeros(2)
            self._ema = np.zeros(2)
            return 0.0, 0.0
        x = self.extractor.features(order=self.model.feature_order)
        raw = self.model.predict(x)      # (2,) in (-1,1)
        if self.cmd_smoothing > 0.0:
            a = self.cmd_smoothing
            self._ema = a * self._ema + (1.0 - a) * raw
            self._u = self._ema
        else:
            self._u = raw
        u_lat = float(self._u[0])
        u_vert = float(max(0.0, self._u[1]))   # vertical is lift-only
        return u_lat, u_vert

    def inject(self, brain):
        if not self.enabled:
            return [], []
        u_lat = float(self._u[0]) * self.lat_gain
        u_vert = float(max(0.0, self._u[1])) * self.vert_gain
        return self.basis.inject(brain, u_lat, u_vert)

    @property
    def last_u(self):
        return (float(self._u[0]), float(max(0.0, self._u[1])))
