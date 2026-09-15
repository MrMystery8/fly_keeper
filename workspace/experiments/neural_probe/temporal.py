"""Temporal directional analysis + interpretable temporal decoding.

Directional information may be transient (present in a few windows) or carried
by spike TIMING rather than total count. This analysis, per stage:

  1) per-window directional strength: for each 5 ms window index, how many
     neurons are directional (corrected metric) and the best |AUC-0.5|.
  2) sliding-window AUC of a simple population signal.
  3) cross-validated temporal decoding: use the per-window spike-count vector of
     a stage's neurons as features and a simple nearest-centroid / logistic
     classifier to decode left vs right (analysis only, never the controller),
     across cumulative windows 5/10/20/50/100/200 ms-equivalent.

Also reports response onset and peak-information window per stage.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
OUT = ROOT / "workspace" / "outputs" / "neural_probe"

from experiments.neural_probe.metrics import load, auc, cohens_d, MIN_SPIKES, D_MIN, AUC_MARGIN


def stage_masks(ids_active, ann):
    sup = []
    for bid in ids_active:
        bid = int(bid)
        sup.append(str(ann.loc[bid]["superclass"]) if bid in ann.index else "unknown")
    sup = np.array(sup)
    stages = {
        "retina(ol_sensory)": sup == "ol_sensory",
        "optic_lobe(ol_intrinsic)": sup == "ol_intrinsic",
        "visual_projection": sup == "visual_projection",
        "central_brain": np.array([s.startswith("cb") or s in ("central_brain",) for s in sup]),
        "descending": np.array(["descend" in s.lower() for s in sup]),
        "vnc": np.array([s.startswith("vnc") for s in sup]),
    }
    return stages


def per_window_directional(T, ci_left, ci_right, mask):
    """For each window, count directional neurons within mask (corrected)."""
    # T: (n_cond, n_trials, n_win, n_active)
    L = T[ci_left][:, :, mask].astype(float)   # (trials, win, m)
    R = T[ci_right][:, :, mask].astype(float)
    n_win = L.shape[1]; m = L.shape[2]
    counts = []
    best_margin = []
    for w in range(n_win):
        lw = L[:, w, :]; rw = R[:, w, :]
        act = (lw.mean(0) + rw.mean(0)) / 2.0
        ndir = 0; bm = 0.0
        for j in range(m):
            if act[j] < 0.2:
                continue
            a = auc(rw[:, j], lw[:, j]); d = cohens_d(rw[:, j], lw[:, j])
            if abs(a - 0.5) > bm:
                bm = abs(a - 0.5)
            if abs(d) >= D_MIN and abs(a - 0.5) >= AUC_MARGIN and act[j] >= 0.2:
                ndir += 1
        counts.append(ndir); best_margin.append(round(bm, 3))
    return counts, best_margin


def temporal_decode(T, ci_left, ci_right, mask, cum_windows):
    """Cross-validated (leave-one-trial-out) nearest-centroid decoding of L vs R
    using per-window spike-count features of masked neurons, for each cumulative
    window budget. Returns accuracy per budget. Analysis only."""
    L = T[ci_left][:, :, mask].astype(float)   # (trials, win, m)
    R = T[ci_right][:, :, mask].astype(float)
    n_trials = L.shape[0]
    accs = {}
    for cw in cum_windows:
        cw = min(cw, L.shape[1])
        # feature = concatenated per-window counts up to cw
        Lf = L[:, :cw, :].reshape(n_trials, -1)
        Rf = R[:, :cw, :].reshape(n_trials, -1)
        # z-score features across all trials for stability
        allf = np.vstack([Lf, Rf])
        mu = allf.mean(0); sd = allf.std(0) + 1e-6
        Lz = (Lf - mu) / sd; Rz = (Rf - mu) / sd
        correct = 0; total = 0
        for held in range(n_trials):
            trainL = np.delete(Lz, held, 0); trainR = np.delete(Rz, held, 0)
            cL = trainL.mean(0); cR = trainR.mean(0)
            for testvec, truth in [(Lz[held], "L"), (Rz[held], "R")]:
                dL = np.linalg.norm(testvec - cL); dR = np.linalg.norm(testvec - cR)
                pred = "L" if dL < dR else "R"
                correct += int(pred == truth); total += 1
        accs[cw] = round(correct / total, 3)
    return accs


def main():
    import pandas as pd
    ann = pd.read_feather(
        ROOT / "upstream/doomfly/connectome_data/malecns_v1/annotations.feather"
    ).set_index("bodyId")

    result = {}
    for label, tfile, lk, rk in [
        ("static", "static_tensor.npz", "left", "right"),
        ("dynamic", "dynamic_tensor.npz", "left_5_none", "right_5_none"),
    ]:
        data = load(tfile)
        T = data["tensor"]
        conds = data["conditions"]
        ids_active = data["ids"][data["active"]]
        stages = stage_masks(ids_active, ann)
        ci_left = conds.index(lk); ci_right = conds.index(rk)
        wms = data["window_ms"]
        cum = [int(round(x / wms)) for x in (5, 10, 20, 50, 100, 200)]
        cum = [max(1, c) for c in cum]

        result[label] = {}
        print(f"\n===== {label.upper()} temporal =====  (window={wms}ms)")
        for name, mask in stages.items():
            if mask.sum() == 0:
                continue
            counts, bm = per_window_directional(T, ci_left, ci_right, mask)
            onset = next((w for w, c in enumerate(counts) if c > 0), None)
            peak = int(np.argmax(counts)) if any(counts) else None
            accs = temporal_decode(T, ci_left, ci_right, mask, cum)
            result[label][name] = dict(
                n_neurons=int(mask.sum()),
                per_window_directional=counts,
                per_window_best_auc_margin=bm,
                onset_window=onset, onset_ms=None if onset is None else onset * wms,
                peak_window=peak,
                decode_acc_by_ms={int(c * wms): a for c, a in accs.items()},
            )
            print(f"  {name:26s} n={int(mask.sum()):5d} onset={onset} peak={peak} "
                  f"decode(ms->acc)={result[label][name]['decode_acc_by_ms']}")

    (OUT / "temporal_summary.json").write_text(json.dumps(result, indent=2, default=str))
    print("\nsaved temporal_summary.json")


if __name__ == "__main__":
    main()
