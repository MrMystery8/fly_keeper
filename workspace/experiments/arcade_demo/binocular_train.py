"""Train the binocular Arcade bridge with feature selection from TRAINING DATA ONLY.

Pipeline (no test leakage anywhere):
  1. split by complete episode/seed (same seed + fractions as the v2 trainer, so
     the same shots land in train/val/test);
  2. FEATURE SELECTION on TRAIN episodes only: score each of the 600 candidate-pool
     neurons by |Pearson(lateral label, summed-window activity)| on train frames,
     select the top-K, keeping representation from BOTH hemispheres (no artificial
     50/50 forced, but we record the composition and require both sides to appear
     if the data supports it);
  3. fit the same closed-form ridge 2-output linear head on the selected features
     (4 windows each), alpha chosen on validation;
  4. report held-out lateral corr / dir-acc / confusion / class means and vertical
     corr/MAE, plus the selected set's hemisphere composition.

Selection uses only train frames; val picks alpha; test is untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))

from experiments.arcade_demo.arcade_bridge import LinearBridge2D

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def _split_by_seed(seeds, seed=20260916, fracs=(0.7, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    uniq = np.unique(seeds); rng.shuffle(uniq)
    n = len(uniq); n_tr = int(round(fracs[0] * n)); n_va = int(round(fracs[1] * n))
    return (set(uniq[:n_tr].tolist()), set(uniq[n_tr:n_tr + n_va].tolist()),
            set(uniq[n_tr + n_va:].tolist()))


def _ridge(X, Y, alpha):
    n, d = X.shape
    Xa = np.hstack([X, np.ones((n, 1))]); A = Xa.T @ Xa
    A[:-1, :-1] += alpha * np.eye(d)
    Wf = np.linalg.solve(A, Xa.T @ Y); return Wf[:-1], Wf[-1]


def _corr(a, p):
    return float(np.corrcoef(a, p)[0, 1]) if a.std() > 1e-9 and p.std() > 1e-9 else 0.0


def select_features(Xtr_pool, ul_tr, sides, k=207, min_per_side=20):
    """Score each pool neuron by |corr(lateral, summed-window activity)| on TRAIN.

    Xtr_pool: (n_tr, pool*W) newest-first blocks. We sum a neuron's windows to a
    per-neuron activity, correlate with the lateral label, and take the top-k by
    |corr|, ensuring at least min_per_side from each hemisphere if available.
    Returns the selected pool indices (into 0..pool-1).
    """
    pool = len(sides)
    W = Xtr_pool.shape[1] // pool
    # per-neuron summed activity across its windows -> (n_tr, pool)
    act = Xtr_pool.reshape(-1, W, pool).sum(axis=1)   # newest-first blocks sum ok
    scores = np.array([abs(_corr(ul_tr, act[:, j])) for j in range(pool)])
    order = np.argsort(-scores)
    left_idx = [j for j in order if sides[j] == "L"]
    right_idx = [j for j in order if sides[j] == "R"]
    sel = []
    # guarantee minimum representation from each side where available
    sel += left_idx[:min(min_per_side, len(left_idx))]
    sel += right_idx[:min(min_per_side, len(right_idx))]
    # fill the rest by global score order
    for j in order:
        if len(sel) >= k:
            break
        if j not in sel:
            sel.append(int(j))
    sel = sel[:k]
    return np.array(sorted(sel), dtype=int), scores


def _expand(Xpool, sel, pool, W):
    """Select the (window x neuron) columns for the chosen pool indices, keeping
    newest-first block layout: for each window block, take the sel columns."""
    Xr = Xpool.reshape(-1, W, pool)[:, :, sel]      # (n, W, k)
    return Xr.reshape(Xpool.shape[0], -1)           # (n, W*k), newest-first blocks


def train(dataset="binocular_dataset.npz", k=207, alphas=(30, 100, 300, 1000),
          lat_gain=3.5, vert_gain=4.5, cmd_smoothing=0.2, seed=20260916,
          out_model="binocular_bridge_model.npz",
          out_report="binocular_bridge_train.json",
          model_version="arcade-binocular-vision"):
    data = np.load(OUT / dataset, allow_pickle=False)
    Xpool = data["features"].astype(np.float64)
    ul = data["u_lat"].astype(np.float64); uv = data["u_vert"].astype(np.float64)
    Y = np.stack([ul, uv], axis=1)
    seeds = data["seeds"]; n_windows = int(data["n_windows"])
    pool_ids = data["pool_body_ids"].astype(np.int64)
    sides = data["pool_side"].astype(str)
    pool = len(pool_ids); W = n_windows
    groups = np.array([g.decode() if isinstance(g, bytes) else str(g)
                       for g in data["groups"]])

    tr, va, te = _split_by_seed(seeds, seed=seed)
    m_tr = np.array([s in tr for s in seeds]); m_va = np.array([s in va for s in seeds])
    m_te = np.array([s in te for s in seeds])

    # 1. feature selection on TRAIN only
    sel, scores = select_features(Xpool[m_tr], ul[m_tr], sides, k=k)
    sel_ids = pool_ids[sel]; sel_sides = sides[sel]
    X = _expand(Xpool, sel, pool, W)                # (N, k*W) newest-first

    # 2. normalization on TRAIN only
    mu = X[m_tr].mean(0); sd = X[m_tr].std(0); sd = np.where(sd < 1e-6, 1.0, sd)
    Xn = (X - mu) / sd

    # 3. ridge, alpha on VAL
    best = None
    for al in alphas:
        Wm, b = _ridge(Xn[m_tr], Y[m_tr], al)
        pv = np.tanh(Xn[m_va] @ Wm + b)
        cs = [_corr(Y[m_va][:, j], pv[:, j]) for j in range(2)]
        sc = float(np.mean(cs))
        if best is None or sc > best[0]:
            best = (sc, al, Wm, b)
    score, alpha, Wm, b = best

    # 4. held-out test metrics
    pt = np.tanh(Xn[m_te] @ Wm + b)
    metrics = {}
    for j, name in enumerate(("lateral", "vertical")):
        a, p = Y[m_te][:, j], pt[:, j]
        metrics[name] = dict(corr=round(_corr(a, p), 4),
                             mae=round(float(np.mean(np.abs(a - p))), 4),
                             mse=round(float(np.mean((a - p) ** 2)), 4))
    gte = groups[m_te]; lt = Y[m_te][:, 0]; lp = pt[:, 0]; lr = gte != "center"
    dir_acc = float(np.mean(np.sign(lp[lr]) == np.sign(lt[lr]))) if lr.any() else 0.0
    def _c(v): return np.where(v >= 0, 1, -1)
    tt, pp = _c(lt[lr]), _c(lp[lr])
    metrics["lateral"].update(dict(
        direction_accuracy=round(dir_acc, 4),
        confusion=dict(
            true_left={"pred_left": int(((tt < 0) & (pp < 0)).sum()),
                       "pred_right": int(((tt < 0) & (pp > 0)).sum())},
            true_right={"pred_left": int(((tt > 0) & (pp < 0)).sum()),
                        "pred_right": int(((tt > 0) & (pp > 0)).sum())}),
        class_means={c: round(float(lp[gte == c].mean()), 4) for c in
                     ("left", "center", "right")}))
    c0 = metrics["lateral"]["class_means"]["center"]
    metrics["lateral"]["class_means_rel_center"] = {
        c: round(metrics["lateral"]["class_means"][c] - c0, 4)
        for c in ("left", "center", "right")}

    model = LinearBridge2D(W=Wm, b=b, mu=mu, sd=sd, feature_order="newest_first")
    sel_side_counts = dict(Counter(sel_sides.tolist()))
    meta = dict(model_version=model_version, n_windows=n_windows,
                n_neurons=int(k), n_params=int(model.n_params),
                best_alpha=int(alpha), lat_gain=lat_gain, vert_gain=vert_gain,
                cmd_smoothing=cmd_smoothing, drive_mv=25.0,
                selected_body_ids=sel_ids.tolist(),
                selected_sides=sel_sides.tolist(),
                selected_side_counts=sel_side_counts,
                pool_size=int(pool), backend_trained="metal", dataset=dataset,
                vision="binocular", val_score=round(score, 4), test=metrics,
                split_seed=int(seed),
                split_seeds=dict(train=sorted(int(s) for s in tr),
                                 val=sorted(int(s) for s in va),
                                 test=sorted(int(s) for s in te)),
                feature_selection="|corr(lateral, summed-window act)| on TRAIN only, "
                                  "top-k with >=20/side floor")
    path = model.save(OUT / out_model, metadata=meta)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest(); meta["sha256"] = digest
    (OUT / out_report).write_text(json.dumps(
        dict(meta=meta, split=dict(train=len(tr), val=len(va), test=len(te))), indent=2))
    print(json.dumps(dict(alpha=alpha, val_score=round(score, 4),
                          selected_sides=sel_side_counts,
                          test={k2: {kk: vv for kk, vv in v.items()
                                     if kk in ("corr", "mae", "direction_accuracy")}
                                for k2, v in metrics.items()},
                          n_params=model.n_params, sha256=digest[:16]), indent=2))
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="binocular_dataset.npz")
    p.add_argument("--k", type=int, default=207)
    p.add_argument("--out-model", default="binocular_bridge_model.npz")
    p.add_argument("--out-report", default="binocular_bridge_train.json")
    a = p.parse_args()
    train(dataset=a.dataset, k=a.k, out_model=a.out_model, out_report=a.out_report)


if __name__ == "__main__":
    main()
