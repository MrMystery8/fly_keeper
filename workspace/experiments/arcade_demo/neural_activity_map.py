"""Analysis-only map of task information in the bilateral Arcade dataset.

This intentionally does *not* train or overwrite a controller.  It uses complete
episode splits, fits every diagnostic probe on TRAIN only, and writes a compact
machine-readable map of where direction and height are decodable in the retinal-
to-candidate-visual-to-bridge path.  The raw candidate activity is the Metal
MaleCNS activity captured by ``binocular_dataset.py``.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
from experiments.arcade_demo.binocular_train import _split_by_seed, _corr

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def _ridge(X, Y, alpha=100.0):
    xa = np.c_[X, np.ones(len(X))]
    a = xa.T @ xa
    a[:-1, :-1] += alpha * np.eye(X.shape[1])
    w = np.linalg.solve(a, xa.T @ Y)
    return w[:-1], w[-1]


def _standardize(X, train):
    mu = X[train].mean(0); sd = X[train].std(0)
    return (X - mu) / np.where(sd < 1e-6, 1.0, sd)


def _binary_accuracy(X, signed_y, train, test):
    """Held-out L/R accuracy; excludes center frames by construction."""
    use = signed_y != 0
    tr = train & use; te = test & use
    z = _standardize(X.astype(np.float64), tr)
    w, b = _ridge(z[tr], signed_y[tr])
    p = z[te] @ w + b
    return dict(corr=round(_corr(signed_y[te], p), 4),
                accuracy=round(float((np.sign(p) == np.sign(signed_y[te])).mean()), 4),
                n_test=int(te.sum()))


def _threeway_accuracy(X, labels, train, test):
    """Nearest-centroid diagnostic classifier, with centroids fit on train only."""
    z = _standardize(X.astype(np.float64), train)
    classes = ("low", "mid", "high")
    present = [c for c in classes if np.any(labels[train] == c)]
    cent = np.stack([z[train & (labels == c)].mean(0) for c in present])
    pred = np.array([present[i] for i in ((z[test, None, :] - cent[None]) ** 2).sum(2).argmin(1)])
    return dict(accuracy=round(float((pred == labels[test]).mean()), 4),
                n_test=int(test.sum()), classes_present=present)


def _latency(activity, groups, steps, signed, train, n_steps=90):
    """First 20-ms bin whose train-only population separation has stable sign.

    This is a representation latency, not an online decision claim: it records
    when the candidate pool's mean L/R difference becomes reliably nonzero in
    the training episodes.  It is reported separately from motor latency.
    """
    # Mean over the top 32 train-selected directional units makes a stable pool.
    l = activity[train & (signed < 0)].mean(0)
    r = activity[train & (signed > 0)].mean(0)
    score = np.abs(r - l)
    top = np.argsort(-score)[:min(32, len(score))]
    rows = []
    for t in range(n_steps):
        ml = train & (signed < 0) & (steps == t)
        mr = train & (signed > 0) & (steps == t)
        if ml.sum() < 2 or mr.sum() < 2:
            continue
        dl = float(activity[ml][:, top].mean())
        dr = float(activity[mr][:, top].mean())
        pooled = float(np.sqrt((activity[ml][:, top].var() + activity[mr][:, top].var()).mean() / 2))
        rows.append(dict(step=int(t), ms=int(t * 20), left_mean=round(dl, 5),
                         right_mean=round(dr, 5), effect=round((dr-dl) / max(pooled, 1e-6), 5)))
    onset = next((r["ms"] for i, r in enumerate(rows)
                  if abs(r["effect"]) >= 0.5 and i + 1 < len(rows)
                  and abs(rows[i + 1]["effect"]) >= 0.5), None)
    return dict(pool_size=int(len(top)), onset_ms=onset, timecourse=rows)


def analyze(dataset="binocular_dataset.npz", split_seed=20260916):
    d = np.load(OUT / dataset, allow_pickle=False)
    pool = d["pool_body_ids"].astype(np.int64)
    sides = d["pool_side"].astype(str); types = d["pool_type"].astype(str)
    W = int(d["n_windows"])
    # newest window is the actual sample at this decision; do not mix future bins.
    act = d["features"].reshape(-1, W, len(pool))[:, 0, :].astype(np.float64)
    seeds = d["seeds"].astype(int); groups = d["groups"].astype(str)
    heights = d["heights"].astype(float); steps = d["steps"].astype(int)
    signed = np.where(groups == "left", -1.0, np.where(groups == "right", 1.0, 0.0))
    hlabel = np.where(heights < .34, "low", np.where(heights < .68, "mid", "high"))
    tr_s, va_s, te_s = _split_by_seed(seeds, seed=split_seed)
    train = np.isin(seeds, list(tr_s)); val = np.isin(seeds, list(va_s)); test = np.isin(seeds, list(te_s))

    # Per-neuron effects and reliability use train episodes only.
    left = act[train & (signed < 0)]; right = act[train & (signed > 0)]
    low = act[train & (hlabel == "low")]; high = act[train & (hlabel == "high")]
    lr_delta = right.mean(0) - left.mean(0)
    lr_sd = np.sqrt((right.var(0) + left.var(0)) / 2)
    h_delta = high.mean(0) - low.mean(0)
    h_sd = np.sqrt((high.var(0) + low.var(0)) / 2)
    lr_eff = lr_delta / np.maximum(lr_sd, 1e-6); h_eff = h_delta / np.maximum(h_sd, 1e-6)
    reliability = act[train].mean(0) / np.maximum(act[train].std(0), 1e-6)

    ann = pd.read_feather(ROOT / "upstream/doomfly/connectome_data/malecns_v1/annotations.feather")
    byid = ann.set_index("bodyId")
    def rows(order, effect, delta):
        ans = []
        for j in order[:50]:
            bid = int(pool[j]); a = byid.loc[bid] if bid in byid.index else None
            ans.append(dict(body_id=bid, type=str(types[j]), somaSide=str(sides[j]),
                            rootSide=(str(a.get("rootSide", "?")) if a is not None else "?"),
                            effect_size=round(float(effect[j]), 5), mean_difference=round(float(delta[j]), 5),
                            reliability=round(float(reliability[j]), 5)))
        return ans
    top_dir = rows(np.argsort(-np.abs(lr_eff)), lr_eff, lr_delta)
    top_height = rows(np.argsort(-np.abs(h_eff)), h_eff, h_delta)

    # Layer labels are intentionally conservative: retina is verified separately;
    # this dataset directly captured the bilateral candidate optic-lobe pool.
    direction_probe = _binary_accuracy(act, signed, train, test)
    height_probe = _threeway_accuracy(act, hlabel, train, test)
    latency = _latency(act, groups, steps, signed, train)
    report = dict(schema_version=1, analysis_only=True, dataset=dataset,
                  backend="metal (as recorded in dataset summary)",
                  split=dict(seed=split_seed, train_episodes=len(tr_s), val_episodes=len(va_s), test_episodes=len(te_s),
                             complete_episode_split=True),
                  data_integrity="Uses pre-existing corrected no-post-terminal dataset manifest; no controller weights changed.",
                  layers=dict(retina=dict(status="separate receptor/camera audit required; raw retinal traces are not present in this feature dataset"),
                              candidate_optic_lobe=dict(n_neurons=int(len(pool)), side_counts=dict(Counter(sides.tolist())),
                                                        direction_probe=direction_probe, height_probe=height_probe,
                                                        direction_latency=latency),
                              existing_feature_pool=dict(status="audited separately against current bridge metadata"),
                              descending_neurons=dict(status="requires synchronized closed-loop trace; generated by downstream audit")),
                  top_directional_neurons=top_dir, top_height_neurons=top_height,
                  limitations=["Existing binocular training data contains low/high but no mid-height episodes; three-way height probe therefore reports only classes present.",
                               "Frame observations within an episode are correlated; splits, feature scoring, and probes are nevertheless by complete episode/seed.",
                               "Representation onset is not motor latency; downstream trace is required for causal timing."])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "neural_activity_map.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--dataset", default="binocular_dataset.npz")
    a = p.parse_args(); r = analyze(a.dataset)
    print(json.dumps({"direction": r["layers"]["candidate_optic_lobe"]["direction_probe"],
                      "height": r["layers"]["candidate_optic_lobe"]["height_probe"],
                      "latency_ms": r["layers"]["candidate_optic_lobe"]["direction_latency"]["onset_ms"]}, indent=2))


if __name__ == "__main__":
    main()
