"""Rigorously validate whether descending neurons carry decodable left/right
information, guarding against overfitting.

The temporal-decode result (descending -> 100% at >=100ms under dynamic shots)
used leave-one-trial-out on only 6 trials with many features, which can overfit.
Here we:

  1) collect MORE trials (20 per side) of dynamic left vs right shots, recording
     only the 4 responsive descending neurons (fast).
  2) compute, per DN, an interpretable per-window firing profile and its
     left-vs-right AUC with a permutation null (shuffle labels) to get a p-value.
  3) run a leave-one-trial-out decode using ONLY simple features (per-DN total
     count; per-DN window-of-peak count) and compare to a label-shuffled null.
  4) identify exactly which DN + which time window (if any) drives discrimination.

If decoding collapses to chance under more trials / against the null, the
earlier 100% was overfitting and the honest conclusion is that DNs do NOT carry
usable direction. If it survives, we have a genuine (if subtle) signal.
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

# The only descending neurons that respond at all (from the screen).
DN_IDS = {10059: "DNp20_R", 10162: "DNp20_L", 10527: "DNpe017_L", 555871: "DNpe017_R"}


def collect_dn(trials=20, frames=24, window_ms=5.0):
    from experiments.neural_probe.probe import NeuralProbe
    probe = NeuralProbe(seed=7)
    b = probe.brain
    dn_ids = list(DN_IDS.keys())
    # index of each DN in the full graph
    id_to_idx = {int(probe.ids[i]): i for i in range(len(probe.ids))}
    dn_idx = np.array([id_to_idx[i] for i in dn_ids])
    rng = np.random.default_rng(2024)

    def one(group):
        sp = 5.0 * float(rng.uniform(0.9, 1.1))
        seq = probe._dynamic_luminance_sequence(group, sp, "none", frames, window_ms / 1000.0)
        b.reset(); bb = b._brain
        for _ in range(3):  # short warmup on first frame
            b.stimulate_retinal_luminance(probe.retina_ids, seq[0]); b.step(window_ms)
        prof = np.zeros((frames, len(dn_idx)), dtype=np.int32)
        for w, lum in enumerate(seq):
            b.stimulate_retinal_luminance(probe.retina_ids, lum); b.step(window_ms)
            prof[w] = bb.counts[dn_idx]
        return prof

    L = np.stack([one("left") for _ in range(trials)])   # (trials, frames, 4)
    R = np.stack([one("right") for _ in range(trials)])
    return dn_ids, L, R


def perm_auc_pvalue(pos, neg, n_perm=2000, seed=0):
    obs = auc(pos, neg)
    allv = np.concatenate([pos, neg]); n = len(pos)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(allv)
        null[k] = auc(p[:n], p[n:])
    # two-sided p on |AUC-0.5|
    p = (np.sum(np.abs(null - 0.5) >= abs(obs - 0.5)) + 1) / (n_perm + 1)
    return float(obs), float(p)


def loo_decode(featL, featR, shuffle=False, seed=0):
    """Leave-one-trial-out nearest-centroid on given feature matrices."""
    rng = np.random.default_rng(seed)
    nt = featL.shape[0]
    X = np.vstack([featL, featR]); y = np.array([0] * nt + [1] * nt)
    if shuffle:
        y = rng.permutation(y)
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xz = (X - mu) / sd
    correct = 0
    for h in range(len(y)):
        mask = np.ones(len(y), bool); mask[h] = False
        c0 = Xz[mask][y[mask] == 0].mean(0); c1 = Xz[mask][y[mask] == 1].mean(0)
        d0 = np.linalg.norm(Xz[h] - c0); d1 = np.linalg.norm(Xz[h] - c1)
        correct += int((0 if d0 < d1 else 1) == y[h])
    return correct / len(y)


def main():
    print("collecting 20 trials/side of dynamic L/R shots on the 4 responsive DNs ...")
    dn_ids, L, R = collect_dn(trials=20)
    frames = L.shape[1]
    result = {"n_trials_per_side": int(L.shape[0]), "frames": int(frames),
              "window_ms": 5.0, "per_dn": {}, "decoding": {}}

    # per-DN total-count AUC with permutation null
    print("\n=== per-DN left-vs-right (total count over trajectory) ===")
    for j, bid in enumerate(dn_ids):
        totL = L[:, :, j].sum(1); totR = R[:, :, j].sum(1)
        a, p = perm_auc_pvalue(totR, totL, n_perm=2000, seed=j)
        # best single window
        best_w, best_a = 0, 0.5
        for w in range(frames):
            aw = auc(R[:, w, j], L[:, w, j])
            if abs(aw - 0.5) > abs(best_a - 0.5):
                best_w, best_a = w, aw
        _, best_p = perm_auc_pvalue(R[:, int(best_w), j], L[:, int(best_w), j], n_perm=2000, seed=100 + j)
        result["per_dn"][DN_IDS[bid]] = dict(
            body_id=bid, total_auc=round(a, 3), total_p=round(p, 4),
            meanL=round(float(totL.mean()), 2), meanR=round(float(totR.mean()), 2),
            best_window=best_w, best_window_auc=round(best_a, 3),
            best_window_p=round(best_p, 4))
        print(f"  {DN_IDS[bid]:12s} totalAUC={a:.3f} p={p:.3f} | "
              f"L={totL.mean():.1f} R={totR.mean():.1f} | "
              f"best win {best_w} AUC={best_a:.3f} p={best_p:.3f}")

    # decoding with simple interpretable features vs shuffle null
    print("\n=== leave-one-trial-out decoding (real vs label-shuffle null) ===")
    feats = {
        "per_dn_total": (L.sum(1), R.sum(1)),                       # (trials, 4)
        "per_dn_first_half": (L[:, :frames // 2].sum(1), R[:, :frames // 2].sum(1)),
        "per_dn_second_half": (L[:, frames // 2:].sum(1), R[:, frames // 2:].sum(1)),
        "full_spatiotemporal": (L.reshape(L.shape[0], -1), R.reshape(R.shape[0], -1)),
    }
    for name, (fL, fR) in feats.items():
        real = loo_decode(fL, fR)
        null = np.mean([loo_decode(fL, fR, shuffle=True, seed=s) for s in range(50)])
        result["decoding"][name] = dict(real_acc=round(real, 3),
                                        shuffle_acc=round(float(null), 3),
                                        n_features=int(fL.shape[1]))
        print(f"  {name:22s} real={real:.3f} shuffle_null={null:.3f} "
              f"(features={fL.shape[1]})")

    (OUT / "descending_validation.json").write_text(json.dumps(result, indent=2))
    print("\nsaved descending_validation.json")


if __name__ == "__main__":
    main()
