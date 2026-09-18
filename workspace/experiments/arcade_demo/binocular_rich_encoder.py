"""Richer binocular observation: LEFT/RIGHT NEURAL-STREAM discriminative encoders.

Naming note: `pool_side = L/R` is the connectome soma-side (neural hemisphere) of
each pooled neuron, NOT a demonstrated one-to-one physical-eye mapping. We
therefore call these "left/right neural-stream encoders", an engineered
discriminative readout of hemisphere-grouped MaleCNS activity. The sensory input
is still genuinely binocular (two cameras -> fullframe retinae -> MaleCNS); we
just do not claim each stream corresponds uniquely to one eye.

WHY. The prior action policy observed only two collapsed scalars (last_pL,
last_pR). Offline analysis (episode-disjoint) showed a single scalar per neural
stream recovers RIGHT-side direction at only ~0.17 recall (essentially lost),
and even the full 4800-dim feature ceiling gives L/C/R recall 0.73/0.46/0.41. A
variance-based PCA encoder ALSO loses right (recall 0.00-0.17) because variance
is dominated by non-directional signal. A small class-SUPERVISED (discriminative)
per-stream projection recovers right to ~0.63 recall (overall 3-way acc 0.57 vs
0.32 for the collapsed scalar). So the fix is a compact discriminative encoder,
one per neural stream, that preserves stream identity and is fit ONCE offline on
the balanced binocular dataset. It reads the same 600-neuron bilateral pool /
8-window features the runtime produces.

The encoder is a FIXED, ENGINEERED discriminative readout (no ball truth; the
supervised L/C/R labels are training-only). The downstream policy still learns
the control; the encoder just stops the observation from throwing the signal
away. The two streams are NOT mixed before their per-stream encoders.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
ENCODER_FILE = "binocular_rich_encoder.npz"
CLASSES = ("left", "center", "right")


def _fit_ridge_logits(Z, y, k, ridge=5.0):
    """Ridge regression to one-hot class indicators -> [d,3] weight; keep k cols.

    Discriminative (LDA-like) low-dim embedding: unlike PCA it keeps the
    DIRECTION-separating axes, not the total-variance axes.
    """
    yi = np.array([CLASSES.index(v) for v in y])
    Y = np.eye(len(CLASSES))[yi]
    A = Z.T @ Z + ridge * np.eye(Z.shape[1])
    W = np.linalg.solve(A, Z.T @ Y)
    return W[:, :k]


def train_encoder(dataset="binocular_balanced_dataset_fullframe.npz",
                  k_per_eye=3, out=ENCODER_FILE, split_seed=7):
    d = np.load(OUT / dataset, allow_pickle=False)
    X = d["features"].astype(float)
    nw = int(d["n_windows"]); pool = [int(x) for x in d["pool_body_ids"]]
    side = d["pool_side"]; groups = d["groups"]; seeds = d["seeds"].astype(int)
    npool = len(pool)
    Xr = X.reshape(len(X), nw, npool)
    left_idx = np.where(side == "L")[0]; right_idx = np.where(side == "R")[0]
    XL = Xr[:, :, left_idx].reshape(len(X), -1)
    XR = Xr[:, :, right_idx].reshape(len(X), -1)

    uids = np.unique(seeds); rng = np.random.default_rng(split_seed); rng.shuffle(uids)
    ntr = int(0.7 * len(uids)); tr_ids = set(uids[:ntr].tolist())
    tr = np.array([s in tr_ids for s in seeds]); te = ~tr

    def fit_eye(Xe):
        mu = Xe[tr].mean(0); sd = Xe[tr].std(0); sd[sd < 1e-6] = 1
        Ztr = (Xe[tr] - mu) / sd
        W = _fit_ridge_logits(Ztr, groups[tr], k_per_eye)
        return mu, sd, W

    muL, sdL, WL = fit_eye(XL)
    muR, sdR, WR = fit_eye(XR)

    # report separability of the fused embedding on held-out episodes
    def embed(Xe, mu, sd, W):
        return ((Xe - mu) / sd) @ W
    EL = embed(XL, muL, sdL, WL); ER = embed(XR, muR, sdR, WR)
    Z = np.c_[EL, ER]
    acc, rec = _softmax_eval(Z[tr], groups[tr], Z[te], groups[te])

    meta = dict(dataset=dataset, k_per_eye=int(k_per_eye), n_windows=int(nw),
                n_left_neurons=int(len(left_idx)),
                n_right_neurons=int(len(right_idx)),
                embed_dim=int(2 * k_per_eye),
                heldout_group_acc=round(acc, 3),
                heldout_recall={c: round(float(r), 3) for c, r in rec.items()},
                note="discriminative per-eye ridge-logit encoder; fixed extractor")
    np.savez(OUT / out,
             pool_body_ids=np.asarray(pool, np.int64),
             pool_side=side, n_windows=np.array([nw], np.int32),
             left_idx=np.asarray(left_idx, np.int64),
             right_idx=np.asarray(right_idx, np.int64),
             muL=muL, sdL=sdL, WL=WL, muR=muR, sdR=sdR, WR=WR,
             metadata=np.array([json.dumps(meta)]))
    print(f"[rich-encoder] embed_dim={2*k_per_eye} heldout 3-way acc={acc:.3f} "
          f"recall L/C/R={rec['left']:.2f}/{rec['center']:.2f}/{rec['right']:.2f}")
    print(f"[rich-encoder] saved {out}")
    return meta


def _softmax_eval(Xtr, ytr, Xte, yte, epochs=400, lr=0.15, l2=1e-3):
    cls = list(CLASSES); K = len(cls)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd < 1e-6] = 1
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    yi = np.array([cls.index(v) for v in ytr]); yti = np.array([cls.index(v) for v in yte])
    W = np.zeros((Xtr.shape[1], K)); b = np.zeros(K); Y = np.eye(K)[yi]
    for _ in range(epochs):
        z = Xtr @ W + b; z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        W -= lr * (Xtr.T @ (p - Y) / len(Xtr) + l2 * W); b -= lr * (p - Y).mean(0)
    pt = (Xte @ W + b).argmax(1)
    rec = {c: float((pt[yti == i] == i).mean()) if (yti == i).any() else 0.0
           for i, c in enumerate(cls)}
    return float((pt == yti).mean()), rec


class RichBinocularEncoder:
    """Fixed left/right NEURAL-STREAM discriminative encoder over the 600-pool
    / n_windows buffer.

    encode(buf) -> concatenated [left_stream_embed(k), right_stream_embed(k)]
    directional features. `buf` is a list of n_windows spike-count vectors over
    pool_body_ids (newest last), matching the dataset/runtime construction
    x=concat(buf[::-1]). Streams = connectome soma-side (hemisphere) groups.
    """

    def __init__(self, pool_body_ids, n_windows, left_idx, right_idx,
                 muL, sdL, WL, muR, sdR, WR):
        self.pool_body_ids = [int(x) for x in pool_body_ids]
        self.n_windows = int(n_windows)
        self.left_idx = np.asarray(left_idx, int)
        self.right_idx = np.asarray(right_idx, int)
        self.muL, self.sdL, self.WL = muL, sdL, WL
        self.muR, self.sdR, self.WR = muR, sdR, WR
        self.embed_dim = WL.shape[1] + WR.shape[1]

    @classmethod
    def load(cls, path=ENCODER_FILE):
        d = np.load(OUT / path if not str(path).startswith("/") else path,
                    allow_pickle=False)
        return cls(d["pool_body_ids"], int(d["n_windows"][0]),
                   d["left_idx"], d["right_idx"],
                   d["muL"], d["sdL"], d["WL"], d["muR"], d["sdR"], d["WR"]), \
            json.loads(str(d["metadata"][0]))

    def encode(self, buf):
        # buf: list of n_windows arrays over pool_body_ids (newest last)
        arr = np.asarray(buf[-self.n_windows:][::-1])          # [nw, npool] newest-first
        XL = arr[:, self.left_idx].reshape(-1)
        XR = arr[:, self.right_idx].reshape(-1)
        eL = ((XL - self.muL) / self.sdL) @ self.WL
        eR = ((XR - self.muR) / self.sdR) @ self.WR
        return np.concatenate([eL, eR]).astype(np.float32)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="binocular_balanced_dataset_fullframe.npz")
    p.add_argument("--k-per-eye", type=int, default=3)
    p.add_argument("--out", default=ENCODER_FILE)
    a = p.parse_args()
    train_encoder(a.dataset, a.k_per_eye, a.out)
