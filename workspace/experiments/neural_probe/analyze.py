"""Whole-brain and descending-neuron direction screens from probe data.

Runs the NeuralProbe across conditions/trials, aggregates sparse per-neuron
spike responses into dense per-neuron feature vectors, and computes:

  - mean response per condition (baseline-subtracted vs blind)
  - left vs right difference and standardized effect size (Cohen's d)
  - ROC AUC for left-vs-right single-neuron discriminability (analysis only)
  - mirror consistency (does the response swap under mirrored input?)
  - first-spike latency (first window with a spike)

Then it ranks all active neurons, annotates the top ones, and produces a
descending-neuron screen. Classifiers are diagnostic only and never become the
controller.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from experiments.neural_probe.probe import NeuralProbe, STIMULI

OUT = ROOT / "workspace" / "outputs" / "neural_probe"


def _trial_window_matrix(trial, n):
    """Dense (n_windows, n_neurons) spike-count matrix for one trial."""
    W = len(trial["windows"])
    mat = np.zeros((W, n), dtype=np.int32)
    for w, (idx, cnt) in enumerate(trial["windows"]):
        mat[w, idx] = cnt
    return mat


def _trial_total(trial, n, window_slice=None):
    """Total per-neuron spike counts over a window slice (default: all)."""
    total = np.zeros(n, dtype=np.float64)
    windows = trial["windows"]
    sel = range(len(windows)) if window_slice is None else window_slice
    for w in sel:
        idx, cnt = windows[w]
        total[idx] += cnt
    return total


def _cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(1, na + nb - 2))
    if pooled < 1e-9:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def _auc(pos, neg):
    """ROC AUC that `pos` scores exceed `neg` (Mann-Whitney U / (n*m))."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def collect_matrices(probe, trials=6, window_ms=5.0, n_windows=24, warmup_ms=25.0):
    """Return per-condition list of per-trial total-count vectors + window mats."""
    data = probe.collect(conditions=list(STIMULI.keys()), trials=trials,
                         window_ms=window_ms, n_windows=n_windows, warmup_ms=warmup_ms)
    n = probe.n
    totals = {c: np.stack([_trial_total(t, n) for t in trs]) for c, trs in data.items()}
    # keep window matrices only for a few conditions (memory) for temporal analysis
    winmats = {c: np.stack([_trial_window_matrix(t, n) for t in data[c]])
               for c in ("left", "right", "center", "blind")}
    return data, totals, winmats


def whole_brain_screen(probe, totals):
    """Per-neuron direction metrics over the full presentation window."""
    n = probe.n
    left = totals["left"]; right = totals["right"]
    blind = totals["blind"]; center = totals["center"]
    ml = left.mean(0); mr = right.mean(0); mc = center.mean(0); mb = blind.mean(0)
    # baseline-subtract with the blind (dark) condition
    ml_b = ml - mb; mr_b = mr - mb; mc_b = mc - mb
    lr_diff = mr_b - ml_b
    # effect size and AUC per neuron (left vs right trials)
    d = np.zeros(n); auc = np.full(n, 0.5)
    active = np.flatnonzero((left.sum(0) + right.sum(0)) > 0)
    for i in active:
        d[i] = _cohens_d(right[:, i], left[:, i])
        auc[i] = _auc(right[:, i], left[:, i])
    # mirror consistency: a right-preferring neuron under mirrored-left should
    # respond like it does to a true right ball. Score = correlation of the
    # neuron's (right-left) sign with its (mirrored_left - mirrored_right) sign.
    mlft = totals["mirrored_left"].mean(0) - mb
    mrgt = totals["mirrored_right"].mean(0) - mb
    mirror_diff = mrgt - mlft   # note: mirrored_right image looks like a left ball
    # consistency: true lr_diff should be OPPOSITE to mirror_diff if the neuron
    # tracks retinal side (mirror flips the image). Use signed agreement.
    mirror_consistency = -np.sign(lr_diff) * np.sign(mirror_diff)
    return dict(active=active, ml=ml, mr=mr, mc=mc, mb=mb,
                ml_b=ml_b, mr_b=mr_b, mc_b=mc_b, lr_diff=lr_diff,
                cohens_d=d, auc=auc, mirror_consistency=mirror_consistency,
                mirrored_left=mlft, mirrored_right=mrgt)


def first_spike_latency(winmat_stack, window_ms):
    """Mean first-spike window index per neuron over trials -> latency (ms)."""
    # winmat_stack: (trials, windows, neurons)
    trials, W, n = winmat_stack.shape
    lat = np.full(n, np.nan)
    fired = winmat_stack > 0
    for i in range(n):
        firsts = []
        for t in range(trials):
            w = np.argmax(fired[t, :, i]) if fired[t, :, i].any() else -1
            if w >= 0:
                firsts.append(w)
        if firsts:
            lat[i] = np.mean(firsts) * window_ms
    return lat


def load_annotations():
    import pandas as pd
    a = pd.read_feather(ROOT / "upstream/doomfly/connectome_data/malecns_v1/annotations.feather")
    return a.set_index("bodyId")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    probe = NeuralProbe(seed=7)
    print("collecting probe data ...")
    data, totals, winmats = collect_matrices(probe, trials=6, window_ms=5.0,
                                             n_windows=24, warmup_ms=25.0)
    screen = whole_brain_screen(probe, totals)
    ids = probe.ids

    # rank all active neurons by |effect size| gated by minimal responsiveness
    active = screen["active"]
    score = np.abs(screen["cohens_d"]) * (np.abs(screen["auc"] - 0.5) * 2)
    order = active[np.argsort(-score[active])]
    ann = load_annotations()

    def describe(i):
        bid = int(ids[i])
        row = ann.loc[bid] if bid in ann.index else None
        typ = str(row["type"]) if row is not None else "?"
        side = str(row["somaSide"]) if row is not None else "?"
        sup = str(row["superclass"]) if row is not None else "?"
        return bid, typ, side, sup

    top = []
    for i in order[:40]:
        bid, typ, side, sup = describe(i)
        top.append(dict(body_id=bid, type=typ, soma_side=side, superclass=sup,
                        left=round(float(screen["ml_b"][i]), 2),
                        right=round(float(screen["mr_b"][i]), 2),
                        center=round(float(screen["mc_b"][i]), 2),
                        lr_diff=round(float(screen["lr_diff"][i]), 2),
                        cohens_d=round(float(screen["cohens_d"][i]), 2),
                        auc=round(float(screen["auc"][i]), 3),
                        mirror_consistent=int(screen["mirror_consistency"][i]) > 0))
    (OUT / "whole_brain_top.json").write_text(json.dumps(top, indent=2))
    print(f"top {len(top)} direction-selective neurons written")
    print(json.dumps(top[:12], indent=2))

    # save the raw totals for the descending screen / temporal analysis
    np.savez_compressed(OUT / "totals.npz",
                        ids=ids,
                        **{f"tot_{c}": totals[c] for c in totals})
    for c in winmats:
        np.savez_compressed(OUT / f"winmat_{c}.npz", mat=winmats[c])
    print("saved totals + window matrices")


if __name__ == "__main__":
    main()
