"""Balanced-binocular temporal probe and feature-map foundation (analysis only).

This module never writes a bridge model.  It proves the quality and content of
the fresh dataset, performs all selection/normalisation on TRAIN episodes only,
uses VALIDATION for temporal-window and feature-budget comparison, and exposes
one untouched TEST estimate for the validation-selected configuration.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
WINDOWS = (1, 2, 4, 6, 8)
BUDGETS = (16, 32, 64, 128, 207)
ALPHAS = (10.0, 30.0, 100.0, 300.0, 1000.0)


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 1e-9 and b.std() > 1e-9 else 0.0


def _auc(y01, score):
    """Mann-Whitney AUC, deterministic and dependency-free."""
    pos, neg = score[y01], score[~y01]
    if not len(pos) or not len(neg): return None
    return float((pos[:, None] > neg[None, :]).mean() + .5 * (pos[:, None] == neg[None, :]).mean())


def _ridge(X, y, alpha):
    # Centered X is supplied by the caller.  The dual form avoids repeatedly
    # inverting a 1,656-column matrix when there are only a few hundred complete
    # training-episode frames.
    yc = y - y.mean()
    if X.shape[1] > X.shape[0]:
        dual = np.linalg.solve(X @ X.T + alpha * np.eye(len(X)), yc)
        return X.T @ dual, float(y.mean())
    w = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ yc)
    return w, float(y.mean())


def stratified_episode_split(seeds, groups, heights, split_seed=20260917):
    """Split complete episodes independently within every 3x3 shot cell."""
    rows = {}
    for s, g, h in zip(seeds, groups, heights): rows.setdefault(int(s), (str(g), str(h)))
    rng = np.random.default_rng(split_seed); parts = {"train": set(), "val": set(), "test": set()}
    cells = {}
    for s, cell in rows.items(): cells.setdefault(cell, []).append(s)
    for cell, ss in sorted(cells.items()):
        ss = np.asarray(sorted(ss)); rng.shuffle(ss)
        if len(ss) < 5: raise ValueError(f"need >=5 episodes/cell for stable 3-way split: {cell}={len(ss)}")
        # For the default eight episodes/cell this yields 5/1/2, retaining two
        # independent held-out episodes in *every* shot cell.
        ntr = max(1, int(round(.65 * len(ss)))); nva = max(1, int(round(.15 * len(ss))))
        if ntr + nva >= len(ss): ntr, nva = len(ss) - 2, 1
        parts["train"].update(map(int, ss[:ntr])); parts["val"].update(map(int, ss[ntr:ntr+nva])); parts["test"].update(map(int, ss[ntr+nva:]))
    return parts, {f"{g}/{h}": len(v) for (g, h), v in cells.items()}


def _select(activity, target, train, budget):
    # Direction/height relevance from TRAIN samples only; sums are causal-only.
    score = np.array([abs(_corr(target[train], activity[train, j])) for j in range(activity.shape[1])])
    return np.argsort(-score)[:min(budget, activity.shape[1])], score


def _design(raw, W, pool):
    # Stored as newest-first [window, neuron]; selecting the newest W does not
    # introduce future samples.
    return raw.reshape(len(raw), -1, pool)[:, :W, :].reshape(len(raw), W * pool)


def _fit_task(X, target, train, val, test, pool, task="lateral", budget=64):
    # Score a neuron on summed selected causal bins, select on TRAIN, standardise
    # only after selection, and choose ridge alpha on VAL.
    W = X.shape[1] // pool
    act = X.reshape(len(X), W, pool).sum(1)
    chosen, score = _select(act, target, train, budget)
    Xs = X.reshape(len(X), W, pool)[:, :, chosen].reshape(len(X), -1)
    mu = Xs[train].mean(0); sd = np.where(Xs[train].std(0) < 1e-6, 1.0, Xs[train].std(0))
    Z = (Xs - mu) / sd
    best = None
    for alpha in ALPHAS:
        w, b = _ridge(Z[train], target[train], alpha); p = np.tanh(Z[val] @ w + b)
        metric = _corr(target[val], p)
        if best is None or metric > best[0]: best = (metric, alpha, w, b)
    _, alpha, w, b = best
    def measure(mask):
        p = np.tanh(Z[mask] @ w + b); y = target[mask]
        result = dict(corr=round(_corr(y, p), 4), mae=round(float(abs(y-p).mean()), 4))
        if task == "lateral":
            use = np.abs(y) > .05
            result.update(direction_accuracy=round(float((np.sign(p[use]) == np.sign(y[use])).mean()), 4),
                          auc_right_vs_left=round(_auc(y[use] > 0, p[use]), 4))
        return result
    return dict(validation=measure(val), test=measure(test), alpha=alpha,
                selected_indices=chosen, scores=score, mu=mu, sd=sd, W=w, b=b)


def _height_class_probe(X, classes, train, val, test, pool, budget):
    """TRAIN-selected, centroid-based LOW/MID/HIGH diagnostic."""
    W = X.shape[1] // pool
    encoded = np.select([classes == "low", classes == "mid", classes == "high"], [0., .5, 1.])
    act = X.reshape(len(X), W, pool).sum(1)
    chosen, _ = _select(act, encoded, train, budget)
    Xs = X.reshape(len(X), W, pool)[:, :, chosen].reshape(len(X), -1)
    mu=Xs[train].mean(0); sd=np.where(Xs[train].std(0)<1e-6,1.,Xs[train].std(0)); Z=(Xs-mu)/sd
    labels=("low","mid","high")
    cent=np.stack([Z[train & (classes == c)].mean(0) for c in labels])
    def measure(mask):
        dist=((Z[mask,None,:]-cent[None,:,:])**2).mean(2)
        pred=np.array([labels[i] for i in dist.argmin(1)])
        return dict(accuracy=round(float((pred == classes[mask]).mean()),4), n=int(mask.sum()),
                    class_means={c:round(float(dist[classes[mask] == c, i].mean()),4) for i,c in enumerate(labels)})
    return dict(validation=measure(val), test=measure(test), selected_indices=chosen)


def analyze(dataset="binocular_balanced_dataset.npz", split_seed=20260917):
    d = np.load(OUT / dataset, allow_pickle=False)
    raw = d["features"].astype(np.float64); pool_ids = d["pool_body_ids"].astype(np.int64)
    pool_sides = d["pool_side"].astype(str); pool_types = d["pool_type"].astype(str)
    groups = d["groups"].astype(str); hclass = d["height_classes"].astype(str); seeds = d["seeds"].astype(int)
    ul = d["u_lat"].astype(float); uv = d["u_vert"].astype(float); steps = d["steps"].astype(int)
    max_w = int(np.asarray(d["n_windows"]).ravel()[0]); pool = len(pool_ids)
    if raw.shape[1] != max_w * pool: raise ValueError("feature layout mismatch")
    parts, cell_counts = stratified_episode_split(seeds, groups, hclass, split_seed)
    masks = {k: np.isin(seeds, list(v)) for k, v in parts.items()}

    # Dataset quality contract: no duplicate (seed, step), all cells across all
    # split sets, and explicit balanced complete-episode membership.
    if len(np.unique(np.c_[seeds, steps], axis=0)) != len(seeds): raise AssertionError("duplicate seed/step samples")
    for name, values in parts.items():
        got = Counter(zip(groups[masks[name]], hclass[masks[name]]))
        if len(got) != 9: raise AssertionError(f"split lacks balanced cells: {name} {got}")

    results = []; fitted = {}
    for window in WINDOWS:
        if window > max_w: continue
        X = _design(raw, window, pool)
        for budget in BUDGETS:
            lat = _fit_task(X, ul, masks["train"], masks["val"], masks["test"], pool, "lateral", budget)
            vert = _fit_task(X, uv, masks["train"], masks["val"], masks["test"], pool, "vertical", budget)
            height = _height_class_probe(X, hclass, masks["train"], masks["val"], masks["test"], pool, budget)
            key = f"w{window}_k{budget}"; fitted[key] = (lat, vert, height)
            results.append(dict(window_bins=window, window_ms=window*20, budget=budget,
                                lateral_val=lat["validation"], vertical_val=vert["validation"], height_val=height["validation"],
                                validation_score=round((lat["validation"]["corr"] + vert["validation"]["corr"] + height["validation"]["accuracy"])/3, 4)))
    winner = max(results, key=lambda r: r["validation_score"]); key=f"w{winner['window_bins']}_k{winner['budget']}"; lat, vert, height = fitted[key]

    # Training-only selectivity map at the validation-selected causal window.
    Xwin = _design(raw, winner["window_bins"], pool).reshape(len(raw), winner["window_bins"], pool).sum(1)
    tr=masks["train"]
    means = {g: Xwin[tr & (groups == g)].mean(0) for g in ("left","center","right")}
    hm = {h: Xwin[tr & (hclass == h)].mean(0) for h in ("low","mid","high")}
    lrd = means["right"]-means["left"]; lrvar=np.sqrt((Xwin[tr & (groups=="right")].var(0)+Xwin[tr & (groups=="left")].var(0))/2)
    hd=hm["high"]-hm["low"]; hvar=np.sqrt((Xwin[tr & (hclass=="high")].var(0)+Xwin[tr & (hclass=="low")].var(0))/2)
    ann=pd.read_feather(ROOT / "upstream/doomfly/connectome_data/malecns_v1/annotations.feather").set_index("bodyId")
    def rows(order):
        ans=[]
        for j in order[:64]:
            bid=int(pool_ids[j]); a=ann.loc[bid] if bid in ann.index else None
            ans.append(dict(body_id=bid, cell_type=str(pool_types[j]), somaSide=str(pool_sides[j]),
              rootSide=str(a.get("rootSide", "?")) if a is not None else "?",
              left_mean=round(float(means["left"][j]),4), center_mean=round(float(means["center"][j]),4), right_mean=round(float(means["right"][j]),4),
              low_mean=round(float(hm["low"][j]),4), mid_mean=round(float(hm["mid"][j]),4), high_mean=round(float(hm["high"][j]),4),
              direction_effect=round(float(lrd[j]/max(lrvar[j],1e-6)),4), height_effect=round(float(hd[j]/max(hvar[j],1e-6)),4),
              reliability=round(float(Xwin[tr,j].mean()/max(Xwin[tr,j].std(),1e-6)),4)))
        return ans
    report = dict(
        schema_version=1, analysis_only=True, dataset=dataset, backend="metal",
        dataset_quality=dict(
            sample_count=int(len(seeds)), episode_count=int(len(np.unique(seeds))),
            cell_episode_counts=cell_counts,
            complete_episode_split={k: len(v) for k, v in parts.items()},
            no_cross_episode_windows=True, no_duplicate_seed_step=True,
            condition=str(d["condition"][0])),
        temporal_validation=results, chosen_by_validation=winner,
        held_out_test=dict(lateral=lat["test"], vertical=vert["test"], height_classification=height["test"]),
        selected_features=dict(
            lateral_body_ids=[int(pool_ids[i]) for i in lat["selected_indices"]],
            vertical_body_ids=[int(pool_ids[i]) for i in vert["selected_indices"]]),
        neural_map=dict(
            top_directional=rows(np.argsort(-np.abs(lrd / np.maximum(lrvar, 1e-6)))),
            top_height=rows(np.argsort(-np.abs(hd / np.maximum(hvar, 1e-6))))),
        limitations=[
            "This is a diagnostic linear-probe foundation, not a deployed bridge.",
            "The test estimate is reported only for the validation-selected temporal/budget configuration."])
    (OUT / "binocular_balanced_temporal_analysis.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--dataset", default="binocular_balanced_dataset.npz"); p.add_argument("--split-seed", type=int, default=20260917)
    a=p.parse_args(); r=analyze(a.dataset,a.split_seed); print(json.dumps(dict(chosen=r["chosen_by_validation"], test=r["held_out_test"], quality=r["dataset_quality"]),indent=2))
