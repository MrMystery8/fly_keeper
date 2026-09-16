"""Offline decodability probe on the corrected v2 dataset (analysis only).

Answers: is the shortfall of the joint 2-output linear bridge (lateral corr
~0.41) a MODEL-SELECTION artifact (single shared ridge alpha across two very
differently-scaled heads) or a genuine ceiling of the linear decode from the
corrected Metal features?

All splits are BY EPISODE/SEED (same 70/15/15 seed split as the trainer). No
test seed influences normalization, alpha, or fitting. We compare, on the SAME
held-out test seeds:

  A. joint 2-output ridge (as deployed), shared alpha  -> lateral/vertical corr
  B. separate per-axis ridge, alpha swept independently -> lateral/vertical corr
  C. lateral-only ridge on LEFT/RIGHT frames (audit-style focused probe)

This does not modify any frozen artifact; it decides whether per-axis heads /
per-axis regularization are justified for v2.
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


def _split_by_seed(seeds, seed=20260916, fracs=(0.7, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    uniq = np.unique(seeds); rng.shuffle(uniq)
    n = len(uniq); n_tr = int(round(fracs[0] * n)); n_va = int(round(fracs[1] * n))
    tr = set(uniq[:n_tr].tolist()); va = set(uniq[n_tr:n_tr + n_va].tolist())
    te = set(uniq[n_tr + n_va:].tolist())
    return tr, va, te


def _ridge_1d(X, y, alpha):
    d = X.shape[1]
    Xa = np.hstack([X, np.ones((len(X), 1))])
    A = Xa.T @ Xa
    A[:-1, :-1] += alpha * np.eye(d)
    wf = np.linalg.solve(A, Xa.T @ y)
    return wf[:-1], wf[-1]


def _corr(a, p):
    return (float(np.corrcoef(a, p)[0, 1])
            if a.std() > 1e-9 and p.std() > 1e-9 else 0.0)


def main(dataset="arcade_dataset_v2.npz"):
    d = np.load(OUT / dataset, allow_pickle=False)
    X = d["features"].astype(np.float64)
    ul = d["u_lat"].astype(np.float64); uv = d["u_vert"].astype(np.float64)
    seeds = d["seeds"]
    groups = np.array([g.decode() if isinstance(g, bytes) else str(g)
                       for g in d["groups"]])
    tr, va, te = _split_by_seed(seeds)
    m_tr = np.array([s in tr for s in seeds]); m_te = np.array([s in te for s in seeds])
    mu = X[m_tr].mean(0); sd = X[m_tr].std(0); sd = np.where(sd < 1e-6, 1, sd)
    Xn = (X - mu) / sd
    alphas = [10, 30, 100, 300, 1000, 3000]
    report = {}

    # --- B. separate per-axis ridge, alpha chosen per axis on TRAIN via 5-fold-ish
    #        (here: pick by train-seed subsplit == use val seeds) ---
    m_va = np.array([s in va for s in seeds])
    def best_axis(y):
        best = None
        for al in alphas:
            w, b = _ridge_1d(Xn[m_tr], y[m_tr], al)
            pv = Xn[m_va] @ w + b
            c = _corr(y[m_va], pv)
            if best is None or c > best[0]:
                best = (c, al, w, b)
        _, al, w, b = best
        pt = Xn[m_te] @ w + b
        return dict(alpha=al, corr=round(_corr(y[m_te], pt), 4),
                    mae=round(float(np.mean(np.abs(y[m_te] - pt))), 4))
    report["separate_per_axis"] = dict(lateral=best_axis(ul), vertical=best_axis(uv))

    # --- C. lateral-only on LEFT/RIGHT frames (focused, audit-style) ---
    lr_tr = m_tr & (groups != "center"); lr_te = m_te & (groups != "center")
    best = None
    for al in alphas:
        w, b = _ridge_1d(Xn[lr_tr], ul[lr_tr], al)
        pv = Xn[m_va & (groups != "center")] @ w + b
        c = _corr(ul[m_va & (groups != "center")], pv)
        if best is None or c > best[0]:
            best = (c, al, w, b)
    _, al, w, b = best
    pt = Xn[lr_te] @ w + b
    dir_acc = float(np.mean(np.sign(pt) == np.sign(ul[lr_te])))
    report["lateral_only_LR"] = dict(
        alpha=al, corr=round(_corr(ul[lr_te], pt), 4),
        direction_accuracy=round(dir_acc, 4), n_test_lr=int(lr_te.sum()))

    # --- also: full-set lateral direction accuracy from the separate head ---
    w, b = _ridge_1d(Xn[m_tr], ul[m_tr], report["separate_per_axis"]["lateral"]["alpha"])
    pt = Xn[m_te] @ w + b
    lr = groups[m_te] != "center"
    report["separate_lateral_dir_acc_LRframes"] = round(
        float(np.mean(np.sign(pt[lr]) == np.sign(ul[m_te][lr]))), 4)

    print(json.dumps(report, indent=2))
    (OUT / "offline_probe_v2.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
