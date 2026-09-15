"""Test whether an INTERPRETABLE, causal timing-aware readout separates left vs
right shots from the descending neurons - before wiring it into the closed loop.

The validation showed the L/R signal is in DN spike TIMING over ~120 ms, not in
total counts, and is not time-locked to a fixed window in a way a fixed-window
count could exploit. Here we test simple, causal, interpretable features that a
real controller could compute online (no trained classifier, no ball state, no
label alignment):

  A. leaky temporal integral of each DN's activity (slow + fast integrator);
     directional signal = difference of a rate-of-change contrast.
  B. cumulative spike-count difference R-side vs L-side over the trajectory.
  C. peak-window-agnostic: max over sliding 40 ms windows of (R - L) contrast.

For each we compute left-vs-right AUC with a permutation null over 20 trials/side.
If none of these causal interpretable features separate L/R above chance, then a
deployable decoder cannot extract the signal without trial alignment/learning,
and the closed-loop MaleCNS cannot be expected to improve. This gates task 9.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))
OUT = ROOT / "workspace" / "outputs" / "neural_probe"

from experiments.neural_probe.metrics import auc
from experiments.neural_probe.validate_descending import collect_dn
LEFT_DN = [10162, 10527]     # DNp20_L, DNpe017_L
RIGHT_DN = [10059, 555871]   # DNp20_R, DNpe017_R


def perm_p(pos, neg, n_perm=2000, seed=0):
    obs = auc(pos, neg); allv = np.concatenate([pos, neg]); n = len(pos)
    rng = np.random.default_rng(seed); null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(allv); null[k] = auc(p[:n], p[n:])
    return float(obs), float((np.sum(np.abs(null - 0.5) >= abs(obs - 0.5)) + 1) / (n_perm + 1))


def leaky_contrast(prof, right_cols, left_cols, tau_windows=20.0):
    """Causal leaky integral of (sum right DNs - sum left DNs). Returns the final
    integrated contrast. prof: (frames, n_dn)."""
    a = np.exp(-1.0 / tau_windows)
    s = 0.0; out = 0.0
    for w in range(prof.shape[0]):
        inst = prof[w, right_cols].sum() - prof[w, left_cols].sum()
        s = a * s + inst
        out = s
    return out


def sliding_peak_contrast(prof, right_cols, left_cols, win=8):
    """Max absolute (R-L) summed over any causal sliding window of `win` frames."""
    r = prof[:, right_cols].sum(1); l = prof[:, left_cols].sum(1)
    diff = r - l
    best = 0.0
    for start in range(len(diff) - win + 1):
        s = diff[start:start + win].sum()
        if abs(s) > abs(best):
            best = s
    return best


def main():
    print("collecting 20 trials/side dynamic L/R on the 4 DNs (again, for decoder test) ...")
    dn_ids, L, R = collect_dn(trials=20)
    # column indices of right/left DNs within the collected [dn_ids] order
    right_cols = [dn_ids.index(i) for i in RIGHT_DN]
    left_cols = [dn_ids.index(i) for i in LEFT_DN]

    result = {}
    # A. leaky integral contrast, several time constants
    for tau in (5.0, 10.0, 20.0, 40.0):
        cl = np.array([leaky_contrast(L[t], right_cols, left_cols, tau) for t in range(L.shape[0])])
        cr = np.array([leaky_contrast(R[t], right_cols, left_cols, tau) for t in range(R.shape[0])])
        a, p = perm_p(cr, cl)
        result[f"leaky_contrast_tau{int(tau)}"] = dict(auc=round(a, 3), p=round(p, 4),
                                                        meanL=round(float(cl.mean()), 3),
                                                        meanR=round(float(cr.mean()), 3))
    # B. cumulative total contrast R-L
    cl = np.array([L[t][:, right_cols].sum() - L[t][:, left_cols].sum() for t in range(L.shape[0])])
    cr = np.array([R[t][:, right_cols].sum() - R[t][:, left_cols].sum() for t in range(R.shape[0])])
    a, p = perm_p(cr, cl)
    result["cumulative_contrast"] = dict(auc=round(a, 3), p=round(p, 4),
                                         meanL=round(float(cl.mean()), 3), meanR=round(float(cr.mean()), 3))
    # C. sliding-peak contrast
    for win in (4, 8, 12):
        cl = np.array([sliding_peak_contrast(L[t], right_cols, left_cols, win) for t in range(L.shape[0])])
        cr = np.array([sliding_peak_contrast(R[t], right_cols, left_cols, win) for t in range(R.shape[0])])
        a, p = perm_p(cr, cl)
        result[f"sliding_peak_win{win}"] = dict(auc=round(a, 3), p=round(p, 4))

    (OUT / "candidate_decoder_test.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    best = max(result.items(), key=lambda kv: abs(kv[1]["auc"] - 0.5))
    print(f"\nbest interpretable causal feature: {best[0]} AUC={best[1]['auc']} p={best[1]['p']}")
    print("JUSTIFIED to build closed-loop decoder?" ,
          "YES" if abs(best[1]["auc"] - 0.5) >= 0.15 and best[1]["p"] < 0.05 else
          "NO (signal not extractable by an interpretable causal readout)")


if __name__ == "__main__":
    main()
