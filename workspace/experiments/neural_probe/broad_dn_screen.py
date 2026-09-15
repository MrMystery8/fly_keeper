"""Broad descending-neuron screen across ALL annotated DNs.

Does not restrict to DNp20/DNpe017. From the probe tensors, reports, for every
descending neuron in the graph: whether it is active at all under visual drive,
its best per-window left/right AUC (dynamic shots), and groups results by DN
functional type where annotations allow (turning/locomotion/escape/etc. inferred
from type prefixes). The central question: does ANY descending neuron carry
usable directional information under realistic (moving) shots?
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
OUT = ROOT / "workspace" / "outputs" / "neural_probe"

from experiments.neural_probe.metrics import load, auc


def main():
    import pandas as pd
    ann = pd.read_feather(
        ROOT / "upstream/doomfly/connectome_data/malecns_v1/annotations.feather"
    ).set_index("bodyId")

    data = load("dynamic_tensor.npz")
    T = data["tensor"]  # (cond, trials, win, active)
    conds = data["conditions"]
    ids_active = data["ids"][data["active"]]
    ci_l = conds.index("left_5_none"); ci_r = conds.index("right_5_none")

    # which active neurons are descending
    is_dn = []
    for bid in ids_active:
        bid = int(bid)
        is_dn.append(bid in ann.index and "descend" in str(ann.loc[bid]["superclass"]).lower())
    is_dn = np.array(is_dn)

    L = T[ci_l]; R = T[ci_r]  # (trials, win, active)
    frames = L.shape[1]

    rows = []
    for j in np.flatnonzero(is_dn):
        bid = int(ids_active[j])
        row = ann.loc[bid]
        totL = L[:, :, j].sum(1); totR = R[:, :, j].sum(1)
        activity = float((totL.mean() + totR.mean()) / 2)
        # best single-window AUC across the trajectory
        best_a = 0.5
        for w in range(frames):
            a = auc(R[:, w, j], L[:, w, j])
            if abs(a - 0.5) > abs(best_a - 0.5):
                best_a = a
        rows.append(dict(body_id=bid, type=str(row["type"]), soma_side=str(row["somaSide"]),
                         activity=round(activity, 2),
                         total_auc=round(auc(totR, totL), 3),
                         best_window_auc=round(best_a, 3)))
    responsive = [r for r in rows if r["activity"] >= 0.5]
    responsive.sort(key=lambda r: -abs(r["best_window_auc"] - 0.5))

    # group DN types by inferred function (rough, annotation-based)
    def fam(t):
        t = t or ""
        if t.startswith("DNp"):
            return "DNp (posterior/visual)"
        if t.startswith("DNa"):
            return "DNa (anterior/steering)"
        if t.startswith("DNb"):
            return "DNb"
        if t.startswith("DNg"):
            return "DNg (gnathal/misc)"
        if t.startswith("DNd") or t.startswith("DNc"):
            return "DNc/DNd"
        if t.startswith("MDN"):
            return "MDN (backward)"
        return "other"
    from collections import defaultdict
    by_fam = defaultdict(lambda: {"n": 0, "responsive": 0, "best_auc_margin": 0.0})
    for r in rows:
        f = fam(r["type"]); by_fam[f]["n"] += 1
        if r["activity"] >= 0.5:
            by_fam[f]["responsive"] += 1
        by_fam[f]["best_auc_margin"] = max(by_fam[f]["best_auc_margin"],
                                           abs(r["best_window_auc"] - 0.5))
    fam_summary = {k: {**v, "best_auc_margin": round(v["best_auc_margin"], 3)}
                   for k, v in by_fam.items()}

    result = dict(
        total_dn=len(rows),
        responsive_dn=len(responsive),
        by_family=fam_summary,
        top_responsive=responsive[:20],
    )
    (OUT / "broad_dn_screen.json").write_text(json.dumps(result, indent=2))
    print(f"total DNs (active in tensor union): {len(rows)}")
    print(f"responsive (activity>=0.5 spk): {len(responsive)}")
    print("by family (n / responsive / best|AUC-0.5|):")
    for f, v in sorted(fam_summary.items(), key=lambda kv: -kv[1]["responsive"]):
        print(f"  {f:26s} n={v['n']:4d} responsive={v['responsive']:3d} "
              f"best|AUC-.5|={v['best_auc_margin']:.3f}")
    print("top responsive DNs by best-window discriminability:")
    for r in responsive[:12]:
        print(f"  {r['body_id']:>8} {r['type']:>10} {r['soma_side']} "
              f"act={r['activity']:.1f} totalAUC={r['total_auc']:.3f} "
              f"bestwinAUC={r['best_window_auc']:.3f}")


if __name__ == "__main__":
    main()
