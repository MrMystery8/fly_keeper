"""Train + freeze the 2-output (lateral+vertical) arcade bridge.

Ridge regression (closed-form, same method as the frozen bridge) from the 828-d
temporal features to the 2-axis teacher command, split by episode/seed to avoid
adjacent-window leakage. Freezes W (828x2), b (2), and train-set normalization,
plus runtime gains, into arcade_bridge_model.npz with metadata + provenance.

No RL / MLP / DAgger -- a pure linear head, trained on the Metal-collected
dataset so it matches the Metal runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))

from experiments.arcade_demo.arcade_bridge import LinearBridge2D

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
ARCADE_MODEL_VERSION = "arcade-bridge-v1"


def _split_by_seed(seeds, rng, fracs=(0.7, 0.15, 0.15)):
    uniq = np.unique(seeds)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    tr = set(uniq[:n_tr].tolist())
    va = set(uniq[n_tr:n_tr + n_va].tolist())
    te = set(uniq[n_tr + n_va:].tolist())
    return tr, va, te


def _ridge(X, Y, alpha):
    """Closed-form ridge: W = (X^T X + alpha I)^-1 X^T Y, with bias via augmentation."""
    n, d = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    A = Xa.T @ Xa
    A[:-1, :-1] += alpha * np.eye(d)      # do not regularize the bias
    W_full = np.linalg.solve(A, Xa.T @ Y)  # (d+1, 2)
    return W_full[:-1], W_full[-1]         # W (d,2), b (2,)


def train(dataset="arcade_dataset.npz", alphas=(30, 100, 300, 1000),
          lat_gain=3.5, vert_gain=3.5, cmd_smoothing=0.7, seed=20260916):
    data = np.load(OUT / dataset, allow_pickle=False)
    X = data["features"].astype(np.float64)
    Y = np.stack([data["u_lat"], data["u_vert"]], axis=1).astype(np.float64)
    seeds = data["seeds"]
    gi = data["graph_index"].astype(np.int64)
    n_windows = int(data["n_windows"])
    n_neurons = len(gi)

    rng = np.random.default_rng(seed)
    tr, va, te = _split_by_seed(seeds, rng)
    m_tr = np.array([s in tr for s in seeds])
    m_va = np.array([s in va for s in seeds])
    m_te = np.array([s in te for s in seeds])

    mu = X[m_tr].mean(axis=0)
    sd = X[m_tr].std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xn = (X - mu) / sd

    best = None
    for alpha in alphas:
        W, b = _ridge(Xn[m_tr], Y[m_tr], alpha)
        pred_va = np.tanh(Xn[m_va] @ W + b)
        # score: mean per-axis correlation on validation
        cs = []
        for j in range(2):
            a, p = Y[m_va][:, j], pred_va[:, j]
            if a.std() > 1e-9 and p.std() > 1e-9:
                cs.append(float(np.corrcoef(a, p)[0, 1]))
        score = float(np.mean(cs)) if cs else -1
        if best is None or score > best[0]:
            best = (score, alpha, W, b)
    score, alpha, W, b = best

    # test metrics
    pred_te = np.tanh(Xn[m_te] @ W + b)
    metrics = {}
    for j, name in enumerate(("lateral", "vertical")):
        a, p = Y[m_te][:, j], pred_te[:, j]
        corr = (float(np.corrcoef(a, p)[0, 1])
                if a.std() > 1e-9 and p.std() > 1e-9 else 0.0)
        mae = float(np.mean(np.abs(a - p)))
        metrics[name] = dict(corr=round(corr, 4), mae=round(mae, 4))

    model = LinearBridge2D(W=W, b=b, mu=mu, sd=sd, feature_order="newest_first")
    meta = dict(model_version=ARCADE_MODEL_VERSION, n_windows=n_windows,
                n_neurons=n_neurons, n_params=int(model.n_params),
                best_alpha=int(alpha), lat_gain=lat_gain, vert_gain=vert_gain,
                cmd_smoothing=cmd_smoothing, drive_mv=25.0,
                sel_graph=gi.tolist(), backend_trained="metal",
                dataset=dataset, val_score=round(score, 4), test=metrics)
    path = model.save(OUT / "arcade_bridge_model.npz", metadata=meta)

    # checksum the frozen artifact
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    meta["sha256"] = digest
    (OUT / "arcade_bridge_train.json").write_text(json.dumps(
        dict(meta=meta, split=dict(train=len(tr), val=len(va), test=len(te)),
             test_metrics=metrics), indent=2))
    print(json.dumps(dict(alpha=alpha, val_score=round(score, 4),
                          test=metrics, n_params=model.n_params,
                          sha256=digest[:16]), indent=2))
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="arcade_dataset.npz")
    p.add_argument("--lat-gain", type=float, default=3.5)
    p.add_argument("--vert-gain", type=float, default=3.5)
    p.add_argument("--cmd-smoothing", type=float, default=0.7)
    a = p.parse_args()
    train(dataset=a.dataset, lat_gain=a.lat_gain, vert_gain=a.vert_gain,
          cmd_smoothing=a.cmd_smoothing)


if __name__ == "__main__":
    main()
