"""Temporal / integration-window sweep per stage.

Does giving the network more integration time restore directional transmission
to the deeper populations? Reuses the dynamic probe tensor and, for each stage,
measures the left-vs-right directional SIGNAL (|R-L| total spikes) and a
cross-validated decode as a function of the cumulative integration window
(5/10/20/50/100/120 ms of the ~120 ms trajectory).

This distinguishes an analysis-window choice (how long we accumulate spikes)
from the neural dynamics themselves. If a stage carries no direction at ANY
window, more integration time does not help.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
OUT = ROOT / "workspace" / "outputs" / "pathway_audit"

WINDOWS_MS = [5, 10, 20, 50, 100, 120]


def loo_decode(L, R):
    nt = L.shape[0]
    X = np.vstack([L, R]); y = np.array([0] * nt + [1] * nt)
    mu = X.mean(0); sd = X.std(0) + 1e-9; Xz = (X - mu) / sd
    ok = 0
    for h in range(len(y)):
        m = np.ones(len(y), bool); m[h] = False
        c0 = Xz[m][y[m] == 0].mean(0); c1 = Xz[m][y[m] == 1].mean(0)
        pred = 0 if np.linalg.norm(Xz[h] - c0) < np.linalg.norm(Xz[h] - c1) else 1
        ok += int(pred == y[h])
    return ok / len(y)


def main():
    import pandas as pd
    ann = pd.read_feather(
        ROOT / "upstream/doomfly/connectome_data/malecns_v1/annotations.feather"
    ).set_index("bodyId")
    d = np.load(ROOT / "workspace/outputs/neural_probe/dynamic_tensor.npz")
    conds = [str(c) for c in d["conditions"]]
    T = d["tensor"]  # (cond, trials, win, active)
    wms = float(d["window_ms"])
    ids_active = d["ids"][d["active"]]
    are = ann.reindex(ids_active)
    sup = are["superclass"].astype(str).to_numpy().astype("U32")
    stages = {
        "retina": sup == "ol_sensory",
        "optic_lobe": sup == "ol_intrinsic",
        "visual_projection": sup == "visual_projection",
        "central": np.char.startswith(sup, "cb"),
        "descending": np.char.find(sup, "descend") >= 0,
    }
    ciL = conds.index("left_5_none"); ciR = conds.index("right_5_none")

    result = {}
    print(f"{'stage':18s} " + " ".join(f"{w}ms" for w in WINDOWS_MS))
    for name, mask in stages.items():
        if mask.sum() == 0:
            continue
        row = {}
        line = []
        for w in WINDOWS_MS:
            nw = max(1, int(round(w / wms)))
            # per-window stage-summed series up to nw windows -> feature vector
            L = T[ciL][:, :nw, mask].sum(2)  # (trials, nw)
            R = T[ciR][:, :nw, mask].sum(2)
            acc = loo_decode(L, R)
            lr = float(R.sum(1).mean() - L.sum(1).mean())
            row[w] = dict(decode=round(acc, 3), lr_signal=round(lr, 2))
            line.append(f"{acc:.2f}")
        result[name] = row
        print(f"{name:18s} " + "  ".join(line))

    (OUT / "temporal_sweep.json").write_text(json.dumps(result, indent=2))
    print("\nlr_signal (|R-L| total spikes) at 120 ms per stage:")
    for name in stages:
        if name in result:
            print(f"  {name:18s} {result[name][120]['lr_signal']:+.2f}")
    print("\nsaved temporal_sweep.json")


if __name__ == "__main__":
    main()
