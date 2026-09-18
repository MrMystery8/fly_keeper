"""Validation-controlled, analysis-only binocular late-fusion probes.

This deliberately keeps left and right neural recordings separate until each
stream has made a prediction.  It is not a bridge and never touches MuJoCo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
from experiments.arcade_demo.binocular_temporal_foundation import (
    ALPHAS, BUDGETS, WINDOWS, _corr, _design, _ridge, _select,
    stratified_episode_split,
)

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def _load(name):
    d = np.load(OUT / name, allow_pickle=False)
    return d["features"].astype(float), d["u_lat"].astype(float), d["seeds"].astype(int), d["groups"].astype(str), d["height_classes"].astype(str), len(d["pool_body_ids"]), int(d["n_windows"][0])


def _metrics(y, p):
    use = np.abs(y) > .05
    pos, neg = p[use & (y > 0)], p[use & (y < 0)]
    auc = (pos[:, None] > neg[None, :]).mean() + .5 * (pos[:, None] == neg[None, :]).mean()
    return {"corr": round(_corr(y, p), 4), "direction_accuracy": round(float((np.sign(p[use]) == np.sign(y[use])).mean()), 4), "auc_right_vs_left": round(float(auc), 4)}


def _stream_model(raw, y, masks, pool, max_w):
    """Fit entirely on TRAIN; choose window/budget/ridge on VAL."""
    candidates = []
    for window in WINDOWS:
        if window > max_w:
            continue
        X = _design(raw, window, pool)
        activity = X.reshape(len(X), window, pool).sum(1)
        for budget in BUDGETS:
            chosen, _ = _select(activity, y, masks["train"], budget)
            xs = X.reshape(len(X), window, pool)[:, :, chosen].reshape(len(X), -1)
            mu = xs[masks["train"]].mean(0)
            sd = np.where(xs[masks["train"]].std(0) < 1e-6, 1., xs[masks["train"]].std(0))
            z = (xs - mu) / sd
            for alpha in ALPHAS:
                w, b = _ridge(z[masks["train"]], y[masks["train"]], alpha)
                p = np.tanh(z @ w + b)
                candidates.append((_corr(y[masks["val"]], p[masks["val"]]), window, budget, alpha, p))
    _, window, budget, alpha, p = max(candidates, key=lambda r: r[0])
    return p, {"window_bins": window, "budget": budget, "alpha": alpha,
               "validation": _metrics(y[masks["val"]], p[masks["val"]]),
               "test": _metrics(y[masks["test"]], p[masks["test"]])}


def main(suffix="", out_name="binocular_structured_fusion_audit.json"):
    left, y, seeds, groups, heights, pool, max_w = _load(f"binocular_balanced_left_only{suffix}.npz")
    right, yr, rs, rg, rh, rpool, rmax = _load(f"binocular_balanced_right_only{suffix}.npz")
    both, yb, bs, bg, bh, bpool, bmax = _load(f"binocular_balanced_dataset{suffix}.npz")
    if not (np.array_equal(seeds, rs) and np.array_equal(seeds, bs) and np.allclose(y, yr) and np.allclose(y, yb)):
        raise AssertionError("matched eye-condition datasets are not aligned")
    parts, _ = stratified_episode_split(seeds, groups, heights)
    masks = {k: np.isin(seeds, list(v)) for k, v in parts.items()}
    pl, left_info = _stream_model(left, y, masks, pool, max_w)
    pr, right_info = _stream_model(right, y, masks, rpool, rmax)
    pb, both_info = _stream_model(both, y, masks, bpool, bmax)

    # Late fusion: choose one interpretable blend on validation only.
    alphas = np.linspace(0., 1., 21)
    alpha = max(alphas, key=lambda a: _corr(y[masks["val"]], ((1-a)*pl + a*pr)[masks["val"]]))
    late = (1-alpha)*pl + alpha*pr

    # Stream-level common/opponent basis.  This is appropriate even though
    # individual neurons cannot be falsely paired across streams.
    Z = np.c_[pl + pr, pl - pr]
    mu = Z[masks["train"]].mean(0); sd = np.where(Z[masks["train"]].std(0)<1e-6,1.,Z[masks["train"]].std(0)); Z=(Z-mu)/sd
    common_candidates=[]
    for alpha_ridge in ALPHAS:
        w,b=_ridge(Z[masks["train"]],y[masks["train"]],alpha_ridge); p=np.tanh(Z@w+b)
        common_candidates.append((_corr(y[masks["val"]],p[masks["val"]]),alpha_ridge,p))
    _, common_alpha, common = max(common_candidates,key=lambda r:r[0])

    # Right residual uses left's fixed visual prediction.  It asks whether
    # right activity predicts what the left stream has not already explained.
    residual_target = y - pl
    rr, residual_info = _stream_model(right, residual_target, masks, rpool, rmax)
    gains=np.linspace(0.,1.5,31)
    gain=max(gains,key=lambda g:_corr(y[masks["val"]],np.clip(pl+g*rr,-1,1)[masks["val"]]))
    residual=np.clip(pl+gain*rr,-1,1)
    def record(p, extra=None):
        o={"validation":_metrics(y[masks["val"]],p[masks["val"]]),"test":_metrics(y[masks["test"]],p[masks["test"]])}
        if extra:o.update(extra)
        return o
    out={"analysis_only":True,"split_episode_counts":{k:len(v) for k,v in parts.items()},
         "left_stream":left_info,"right_stream":right_info,"naive_both":both_info,
         "late_fusion":record(late,{"right_weight_chosen_on_validation":round(float(alpha),2)}),
         "common_opponent":record(common,{"ridge_alpha_chosen_on_validation":common_alpha}),
         "left_plus_right_residual":record(residual,{"residual_stream":residual_info,"residual_gain_chosen_on_validation":round(float(gain),2)}),
         "dataset_suffix": suffix or "(stale viewport)",
         "limitations":["Eye streams are defined by matched eye-condition recordings, not invented neuron-pair subtraction.","This compares offline lateral decoding only; it is not a deployable bridge or a V4 selection."]}
    (OUT/out_name).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="",
                   help="dataset suffix, e.g. '_fullframe' for the geometry-fixed triplet")
    p.add_argument("--out", default="binocular_structured_fusion_audit.json")
    a = p.parse_args()
    main(suffix=a.suffix, out_name=a.out)
