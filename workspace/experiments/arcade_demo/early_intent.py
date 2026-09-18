"""EARLY-INTENT architecture: separate WHERE (early binocular direction) from
HOW (RL execution).

Motivation (measured): continuous closed-loop directional decoding fails --
proprioception/action cannot rescue direction once the keeper moves (motion-
conditioned probe: acc 0.376-0.383, ~chance). But direction IS more informative
BEFORE the keeper moves (corrected-env stationary L/C/R acc ~0.45, right ~0.42).
So we read direction during a brief PRE-MOTION sensing window, maintain a
CONFIDENCE-AWARE lateral intent state, and hand that intent to an RL execution
controller as an OBSERVATION (never a forced max-lateral command).

Components:
  * EarlyIntentEstimator: a 3-class (L/C/R) softmax over the discriminative
    left/right neural-stream embedding (RichBinocularEncoder), fit ONCE offline
    on corrected-env stationary data. Produces P(left),P(center),P(right).
  * IntentState: accumulates estimator log-evidence over the sensing window,
    yielding a signed continuous lateral intent (P(right)-P(left)) and a
    confidence (margin). Uncertain/center -> small intent -> less lateral
    commitment (NOT an argmax hard command).

Vertical is NOT latched here -- it stays continuous and is produced by the RL
execution controller from ongoing MaleCNS + proprioception + flight state.
No ball truth anywhere in the deployed path; supervised L/C/R labels are used
training-only to fit the estimator (engineered discriminative readout).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))

from experiments.arcade_demo.binocular_rich_encoder import RichBinocularEncoder

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
CLASSES = ("left", "center", "right")
ESTIMATOR_FILE = "early_intent_estimator.npz"


def train_estimator(features_file="binocular_rich_features_corrected.npz",
                    encoder_file="binocular_rich_encoder_corrected.npz",
                    out=ESTIMATOR_FILE, split_seed=7, epochs=600, lr=0.15, l2=2e-3):
    """Fit the 3-class L/C/R softmax over the discriminative embedding on
    corrected-env STATIONARY data (episode-disjoint report)."""
    d = np.load(OUT / features_file, allow_pickle=False)
    X = d["X_stationary"].astype(float); groups = d["groups_stationary"]
    seeds = d["seeds_stationary"].astype(int)
    enc, _ = RichBinocularEncoder.load(encoder_file)
    # embed via the fixed encoder (same neuron pool / windows)
    nw = enc.n_windows; npool = len(enc.pool_body_ids)
    Xr = X.reshape(len(X), nw, npool)
    XL = Xr[:, :, enc.left_idx].reshape(len(X), -1)
    XR = Xr[:, :, enc.right_idx].reshape(len(X), -1)
    eL = ((XL - enc.muL) / enc.sdL) @ enc.WL
    eR = ((XR - enc.muR) / enc.sdR) @ enc.WR
    E = np.c_[eL, eR]
    uids = np.unique(seeds); rng = np.random.default_rng(split_seed); rng.shuffle(uids)
    ntr = int(0.7 * len(uids)); tr_ids = set(uids[:ntr].tolist())
    tr = np.array([s in tr_ids for s in seeds]); te = ~tr
    mu = E[tr].mean(0); sd = E[tr].std(0); sd[sd < 1e-6] = 1
    Z = (E - mu) / sd
    yi = np.array([CLASSES.index(v) for v in groups]); Y = np.eye(3)[yi]
    W = np.zeros((Z.shape[1], 3)); b = np.zeros(3)
    for _ in range(epochs):
        z = Z[tr] @ W + b; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        W -= lr * (Z[tr].T @ (p - Y[tr]) / tr.sum() + l2 * W); b -= lr * (p - Y[tr]).mean(0)
    zt = Z[te] @ W + b; zt -= zt.max(1, keepdims=True); pt = np.exp(zt); pt /= pt.sum(1, keepdims=True)
    pred = pt.argmax(1); yti = yi[te]
    acc = float((pred == yti).mean())
    rec = {c: round(float((pred[yti == i] == i).mean()), 3) if (yti == i).any() else None
           for i, c in enumerate(CLASSES)}
    meta = dict(features=features_file, encoder=encoder_file, epochs=epochs,
                heldout_acc=round(acc, 3), heldout_recall=rec,
                note="early-intent L/C/R softmax over discriminative embedding "
                     "(corrected-env stationary; training-only labels)")
    np.savez(OUT / out, W=W, b=b, mu=mu, sd=sd, metadata=np.array([json.dumps(meta)]))
    print(f"[early-intent] heldout acc={acc:.3f} recall={rec}")
    print(f"[early-intent] saved {out}")
    return meta


class EarlyIntentEstimator:
    """Fixed 3-class softmax over the discriminative embedding -> P(L),P(C),P(R)."""

    def __init__(self, encoder: RichBinocularEncoder, W, b, mu, sd):
        self.encoder = encoder
        self.W, self.b, self.mu, self.sd = W, b, mu, sd

    @classmethod
    def load(cls, path=ESTIMATOR_FILE, encoder_file="binocular_rich_encoder_early.npz"):
        enc, _ = RichBinocularEncoder.load(encoder_file)
        d = np.load(OUT / path if not str(path).startswith("/") else path,
                    allow_pickle=False)
        return cls(enc, d["W"], d["b"], d["mu"], d["sd"]), json.loads(str(d["metadata"][0]))

    def probs(self, buf):
        e = self.encoder.encode(buf)
        z = ((e - self.mu) / self.sd) @ self.W + self.b
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        return p            # [P(left), P(center), P(right)]


class IntentState:
    """Confidence-aware latched lateral intent from a pre-motion sensing window.

    During the first `sense_steps` decisions the keeper holds still and we
    accumulate estimator log-probabilities (evidence). After the window we latch:
        intent_lat = P(right) - P(left)   in [-1,1], signed, small when uncertain
        confidence = max(P) - 1/3         separation above chance (0..~0.67)
    intent_lat drives the SIGN/strength of lateral demand only as an OBSERVATION
    to the RL controller; center/uncertain naturally yields small intent.
    """

    def __init__(self, estimator: EarlyIntentEstimator, sense_steps=3):
        # Direction is most decodable in the FIRST ~3 approach steps (measured:
        # steps[0,3) acc 0.545, right 0.61; decays later as the ball nears head
        # -on). So the sensing window is short. We AVERAGE per-step probabilities
        # over the window (NOT sum log-probs, which over-saturates to a hard
        # one-hot and destroys the confidence signal).
        self.est = estimator
        self.sense_steps = int(sense_steps)
        self.reset()

    def reset(self):
        self._t = 0
        self._psum = np.zeros(3)
        self._nseen = 0
        self.intent_lat = 0.0
        self.confidence = 0.0
        self.p = np.array([1/3, 1/3, 1/3])

    def update(self, buf):
        """Call every decision step with the current pool buffer.

        Accumulates a MEAN probability over the sensing window; after the window
        the intent is latched (stops updating). intent_lat = P(R)-P(L) is signed
        and small when uncertain/center; confidence = margin above chance.
        """
        if self._t < self.sense_steps:
            self._psum += self.est.probs(buf)
            self._nseen += 1
            pa = self._psum / max(1, self._nseen)
            self.p = pa
            self.intent_lat = float(pa[2] - pa[0])
            self.confidence = float(pa.max() - 1/3)
        self._t += 1
        return self.intent_lat, self.confidence


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="binocular_rich_features_corrected.npz")
    p.add_argument("--encoder", default="binocular_rich_encoder_corrected.npz")
    p.add_argument("--out", default=ESTIMATOR_FILE)
    a = p.parse_args()
    train_estimator(a.features, a.encoder, a.out)
