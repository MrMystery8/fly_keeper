"""The learned visual -> descending-neuron bridge (EXPERIMENTAL, minimal).

Runtime path (bridge ON):

    retina -> fixed MaleCNS visual processing
           -> selected optic-lobe neuron spike counts  (causal temporal windows)
           -> LEARNED bridge (feature normalize -> linear map -> scalar u in [-1,1])
           -> fixed DN motor basis B_lr  (bounded additive current into real DNs)
           -> real DNs evolve under the existing MaleCNS LIF dynamics
           -> existing DescendingMotorDecoder -> CPG locomotion -> body

Bridge OFF: the visual-feature -> DN-current injection is simply not applied.
The native MaleCNS visual->DN pathway and the DN readout run unchanged, so the
system reduces EXACTLY to Natural MaleCNS.

The learned part is ONLY: feature normalization (train-set mean/std) and a linear
map W (n_features -> 1) producing the scalar lateral command u. The motor side is
a FIXED opponent basis (dn_basis.DNMotorBasis) -- a single lateral degree of
freedom, the minimal intervention (Section 18). No per-DN outputs are learned
initially.

This is an engineered decoder, transparently reported as such. It is NOT a claim
that the fixed connectome learned to route direction; the native synapses are
untouched.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from experiments.learned_bridge.dn_basis import DNMotorBasis

DECISION_MS = 20.0
DEFAULT_WINDOWS = 4          # 4 x 20 ms causal windows = last 80 ms
V_REST = -52.0


class VisualFeatureExtractor:
    """Causal temporal spike-count features for the selected optic-lobe neurons.

    Maintains a ring buffer of per-window spike counts for the selected graph
    indices. At each decision it returns the concatenation of the last
    `n_windows` windows (most-recent first), i.e. feature dim = n_neurons *
    n_windows. Only PAST/PRESENT activity is used (windows before/at t).

    The extractor reads spike counts directly from the fixed brain's per-neuron
    `counts` buffer (the last completed 20 ms step) via graph indices; it never
    sees ball state.
    """

    def __init__(self, graph_index, n_windows=DEFAULT_WINDOWS):
        self.graph_index = np.asarray(graph_index, dtype=np.int64)
        self.n_neurons = len(self.graph_index)
        self.n_windows = int(n_windows)
        self.dim = self.n_neurons * self.n_windows
        self._buf = None
        self.reset()

    def reset(self):
        # ring buffer of the last n_windows spike-count vectors (oldest..newest)
        self._buf = [np.zeros(self.n_neurons, dtype=np.float32)
                     for _ in range(self.n_windows)]

    def observe(self, brain):
        """Record the just-completed 20 ms window's spike counts (call after
        brain.step). Uses the fixed brain's raw counts by graph index."""
        counts = brain._brain.counts[self.graph_index].astype(np.float32)
        self._buf.append(counts)
        if len(self._buf) > self.n_windows:
            self._buf.pop(0)

    def features(self, order="newest_first"):
        """Return the current temporal feature vector.

        order: 'newest_first' (default), 'oldest_first', or 'reversed' /
        window-subset variants are handled by the caller for ablations.
        """
        wins = self._buf[-self.n_windows:]
        if order == "newest_first":
            wins = wins[::-1]
        # oldest_first == buffer order
        return np.concatenate(wins).astype(np.float64)


class LinearBridgeModel:
    """Normalized linear map from temporal features to a scalar command u.

    u = tanh( ((x - mu) / sd) @ w + b )

    - mu, sd: train-set feature normalization (frozen).
    - w, b: learned linear weights (frozen after selection).
    - tanh bounds u to (-1, 1) so the DN drive is inherently bounded.

    Trainable parameter count = n_features (w) + 1 (b). Reported prominently.
    """

    def __init__(self, w, b, mu, sd, feature_order="newest_first"):
        self.w = np.asarray(w, dtype=np.float64).ravel()
        self.b = float(b)
        self.mu = np.asarray(mu, dtype=np.float64).ravel()
        self.sd = np.asarray(sd, dtype=np.float64).ravel()
        self.sd = np.where(self.sd < 1e-6, 1.0, self.sd)
        self.feature_order = feature_order

    @property
    def n_params(self):
        return int(self.w.size + 1)

    def predict_raw(self, x):
        """Linear pre-activation (before tanh), for diagnostics."""
        xn = (np.asarray(x, dtype=np.float64) - self.mu) / self.sd
        return float(xn @ self.w + self.b)

    def predict(self, x):
        return float(np.tanh(self.predict_raw(x)))

    # ---------------------------------------------------------------- I/O
    def save(self, path, metadata=None):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w=self.w, b=np.array([self.b]), mu=self.mu, sd=self.sd,
                 feature_order=np.array([self.feature_order]),
                 metadata=np.array([json.dumps(metadata or {})]))
        return str(path)

    @classmethod
    def load(cls, path):
        a = np.load(path, allow_pickle=False)
        m = cls(a["w"], float(a["b"][0]), a["mu"], a["sd"],
                feature_order=str(a["feature_order"][0]))
        m.loaded_metadata = json.loads(str(a["metadata"][0])) if "metadata" in a else {}
        return m


class LearnedBridge:
    """Composes the feature extractor, linear model, and DN motor basis.

    A single object the controller queries each decision:
      1. extractor.observe(brain)         (after the brain step)
      2. u = model.predict(extractor.features())
      3. basis.inject(brain, gain * u)    (BEFORE the next brain step; bridge ON)

    `enabled` toggles bridge ON/OFF at runtime. When OFF, no current is injected
    and the system is Natural MaleCNS.
    """

    def __init__(self, extractor: VisualFeatureExtractor, model: LinearBridgeModel,
                 basis: DNMotorBasis, gain=1.0, enabled=True, cmd_smoothing=0.0):
        self.extractor = extractor
        self.model = model
        self.basis = basis
        self.gain = float(gain)
        self.enabled = bool(enabled)
        # Optional causal EMA on the scalar command u. The offline diagnostic
        # showed the per-window model output is directionally correct but noisy
        # (occasional sign flips), which dilutes sustained lateral drive; a
        # short EMA gives the fixed DNs a cleaner sustained command WITHOUT
        # touching any downstream mechanics. 0.0 = no smoothing (raw per-window).
        self.cmd_smoothing = float(cmd_smoothing)
        self._last_u = 0.0
        self._u_ema = 0.0

    def reset(self):
        self.extractor.reset()
        self._last_u = 0.0
        self._u_ema = 0.0

    def observe(self, brain):
        self.extractor.observe(brain)

    def command(self):
        """Scalar lateral command u in [-1,1] from current features (bridge ON).
        Returns 0.0 when disabled. Applies optional causal EMA smoothing."""
        if not self.enabled:
            self._last_u = 0.0
            self._u_ema = 0.0
            return 0.0
        x = self.extractor.features(order=self.model.feature_order)
        u_raw = self.model.predict(x)
        if self.cmd_smoothing > 0.0:
            a = self.cmd_smoothing
            self._u_ema = a * self._u_ema + (1.0 - a) * u_raw
            self._last_u = self._u_ema
        else:
            self._last_u = u_raw
        return self._last_u

    def inject(self, brain, u=None):
        """Inject the bounded DN current for command u (defaults to last u).
        No-op when disabled."""
        if not self.enabled:
            return [], []
        if u is None:
            u = self._last_u
        return self.basis.inject(brain, self.gain * u)

    @property
    def last_u(self):
        return self._last_u
