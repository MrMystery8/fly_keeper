"""Sensory-drive gain sweep (no connectivity/weight change).

The retinal adapter may be under- or over-driving the deeper network. We scale
the retinal luminance input by several multipliers and, per gain, measure
left-vs-right decoding accuracy at each stage (retina / optic lobe / visual
projection / central / descending), plus the blind control. We look for a gain
regime where a directional DESCENDING response appears for normal vision but NOT
for blind input.

Luminance is clipped to [0,1] by the model, so gains >1 mainly push more
receptors toward saturation; gains <1 reduce drive. This sweeps sensory drive
strength within the existing model - it does not modify neural dynamics or
weights. The baseline is never permanently changed.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))
OUT = ROOT / "workspace" / "outputs" / "pathway_audit"

GAINS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
STAGES = ["ol_sensory", "ol_intrinsic", "visual_projection", "central", "descending"]


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    allv = np.concatenate([pos, neg])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); start = csum - counts
    ranks = ((start + csum + 1) / 2.0)[inv]
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def loo_decode(featL, featR):
    """Leave-one-trial-out nearest-centroid L vs R accuracy."""
    nt = featL.shape[0]
    X = np.vstack([featL, featR]); y = np.array([0] * nt + [1] * nt)
    mu = X.mean(0); sd = X.std(0) + 1e-9; Xz = (X - mu) / sd
    correct = 0
    for h in range(len(y)):
        m = np.ones(len(y), bool); m[h] = False
        c0 = Xz[m][y[m] == 0].mean(0); c1 = Xz[m][y[m] == 1].mean(0)
        pred = 0 if np.linalg.norm(Xz[h] - c0) < np.linalg.norm(Xz[h] - c1) else 1
        correct += int(pred == y[h])
    return correct / len(y)


class GainProbe:
    def __init__(self, seed=7):
        from experiments.neural_probe.probe import NeuralProbe
        self.p = NeuralProbe(seed=seed)
        self.b = self.p.brain
        import pandas as pd
        ann = pd.read_feather(
            ROOT / "upstream/doomfly/connectome_data/malecns_v1/annotations.feather"
        ).set_index("bodyId")
        ids = self.p.ids
        are = ann.reindex(ids)
        sup = are["superclass"].astype(str).to_numpy()
        self.stage_idx = {
            "ol_sensory": np.flatnonzero(sup == "ol_sensory"),
            "ol_intrinsic": np.flatnonzero(sup == "ol_intrinsic"),
            "visual_projection": np.flatnonzero(sup == "visual_projection"),
            "central": np.flatnonzero(np.char.startswith(sup.astype(str), "cb")),
            "descending": np.flatnonzero(np.char.find(sup.astype(str), "descend") >= 0),
        }

    def trial_stage_counts(self, group, gain, transform, frames=24, window_ms=5.0,
                           warmup_ms=15.0):
        seq = self.p._dynamic_luminance_sequence(group, 5.0, transform, frames,
                                                 window_ms / 1000.0)
        seq = [np.clip(s * gain, 0, 1) for s in seq]
        self.b.reset(); bb = self.b._brain
        for _ in range(int(round(warmup_ms / window_ms))):
            self.b.stimulate_retinal_luminance(self.p.retina_ids, seq[0]); self.b.step(window_ms)
        # accumulate per-stage total spike vector (sum over stage neurons per window)
        stage_series = {s: [] for s in self.stage_idx}
        for lum in seq:
            self.b.stimulate_retinal_luminance(self.p.retina_ids, lum); self.b.step(window_ms)
            for s, idx in self.stage_idx.items():
                stage_series[s].append(int(bb.counts[idx].sum()))
        return {s: np.array(v) for s, v in stage_series.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gp = GainProbe(seed=7)
    trials = 6
    result = {}
    for gain in GAINS:
        # collect L/R/blind trials at this gain; feature = per-window stage series
        data = {"left": [], "right": [], "blind": []}
        for cond in data:
            tf = "blind" if cond == "blind" else "none"
            grp = "center" if cond == "blind" else cond
            for _ in range(trials):
                data[cond].append(gp.trial_stage_counts(grp, gain, tf))
        row = {}
        for s in STAGES:
            L = np.stack([d[s] for d in data["left"]])   # (trials, frames)
            R = np.stack([d[s] for d in data["right"]])
            B = np.stack([d[s] for d in data["blind"]])
            # decode L vs R using the per-window stage series
            acc = loo_decode(L, R)
            # activity + directional vs common-mode on totals
            totL, totR, totB = L.sum(1), R.sum(1), B.sum(1)
            row[s] = dict(
                decode_acc=round(acc, 3),
                mean_total_L=round(float(totL.mean()), 1),
                mean_total_R=round(float(totR.mean()), 1),
                mean_total_blind=round(float(totB.mean()), 1),
                lr_diff=round(float(totR.mean() - totL.mean()), 2),
                common_mode=round(float((totR.mean() + totL.mean()) / 2), 2),
            )
        result[f"gain_{gain}"] = row
        print(f"\n=== gain {gain}x ===")
        for s in STAGES:
            r = row[s]
            print(f"  {s:18s} decodeLR={r['decode_acc']:.2f} "
                  f"L={r['mean_total_L']:.0f} R={r['mean_total_R']:.0f} "
                  f"blind={r['mean_total_blind']:.0f} lrDiff={r['lr_diff']:+.1f} "
                  f"common={r['common_mode']:.0f}")

    (OUT / "gain_sweep.json").write_text(json.dumps(result, indent=2))
    print("\nsaved gain_sweep.json")
    # highlight: any gain where descending decode > 0.7?
    print("\nDESCENDING decode accuracy by gain:")
    for gain in GAINS:
        print(f"  {gain}x: {result[f'gain_{gain}']['descending']['decode_acc']}")


if __name__ == "__main__":
    main()
