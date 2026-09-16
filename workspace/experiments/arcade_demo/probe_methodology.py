"""Reconcile the audit's 0.82 lateral number with the v2 held-out 0.41.

The lateral_audit Q5 probe reported corr/dir_acc IN-SAMPLE (fit and scored on the
same LEFT/RIGHT frames) using LOW shots only. This probe replicates each choice
on the corrected v2 dataset to show exactly which methodological differences
account for the gap, so v2's honest held-out number is interpreted correctly.

Variants (all lateral-only, LEFT/RIGHT frames):
  1. low-only, IN-SAMPLE      (== audit methodology)
  2. low-only, held-out-by-seed
  3. low+high, IN-SAMPLE
  4. low+high, held-out-by-seed   (== v2 deployed condition)
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


def _ridge(X, y, alpha=300):
    Xa = np.hstack([X, np.ones((len(X), 1))]); d = X.shape[1]
    A = Xa.T @ Xa; A[:-1, :-1] += alpha * np.eye(d)
    wf = np.linalg.solve(A, Xa.T @ y); return wf[:-1], wf[-1]


def _corr(a, p):
    return float(np.corrcoef(a, p)[0, 1]) if a.std() > 1e-9 and p.std() > 1e-9 else 0.0


def main(dataset="arcade_dataset_v2.npz"):
    d = np.load(OUT / dataset, allow_pickle=False)
    X = d["features"].astype(np.float64); ul = d["u_lat"].astype(np.float64)
    seeds = d["seeds"]; heights = d["heights"].astype(float)
    groups = np.array([g.decode() if isinstance(g, bytes) else str(g)
                       for g in d["groups"]])
    lr_all = groups != "center"

    rng = np.random.default_rng(20260916)
    uniq = np.unique(seeds); rng.shuffle(uniq)
    n = len(uniq); tr = set(uniq[:int(.7 * n)].tolist())
    te = set(uniq[int(.85 * n):].tolist())
    is_tr = np.array([s in tr for s in seeds]); is_te = np.array([s in te for s in seeds])

    def run(height_mask, held_out):
        sel = lr_all & height_mask
        mu = X[sel].mean(0); sd = X[sel].std(0); sd = np.where(sd < 1e-6, 1, sd)
        Xn = (X - mu) / sd
        if held_out:
            f_tr = sel & is_tr; f_te = sel & is_te
        else:
            f_tr = sel; f_te = sel                     # in-sample
        w, b = _ridge(Xn[f_tr], ul[f_tr])
        p = Xn[f_te] @ w + b
        return dict(corr=round(_corr(ul[f_te], p), 4),
                    dir_acc=round(float(np.mean(np.sign(p) == np.sign(ul[f_te]))), 4),
                    n_train=int(f_tr.sum()), n_eval=int(f_te.sum()))

    low = heights < 0.34
    allh = np.ones_like(low, dtype=bool)
    report = {
        "1_low_only_IN_SAMPLE (audit-style)": run(low, held_out=False),
        "2_low_only_held_out": run(low, held_out=True),
        "3_low+high_IN_SAMPLE": run(allh, held_out=False),
        "4_low+high_held_out (v2 deployed)": run(allh, held_out=True),
    }
    print(json.dumps(report, indent=2))
    (OUT / "probe_methodology.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
