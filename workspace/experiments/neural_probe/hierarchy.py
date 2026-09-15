"""Sensory-pathway + descending-neuron diagnosis from saved probe totals.

Loads workspace/outputs/neural_probe/totals.npz (per-condition, per-trial
per-neuron spike totals) and answers, stage by stage:

  retina (ol_sensory) -> optic lobe (ol_intrinsic/ol_*) -> central brain ->
  descending neurons

At each stage: how much left/right directional information survives? Where does
it disappear? Produces the descending-neuron ranked table and classifies the
sensory-pathway outcome (A/B/C/D).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
OUT = ROOT / "workspace" / "outputs" / "neural_probe"


def _cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(1, na + nb - 2))
    return 0.0 if pooled < 1e-9 else float((a.mean() - b.mean()) / pooled)


def _auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def main():
    d = np.load(OUT / "totals.npz")
    ids = d["ids"].astype(np.int64)
    totals = {k[4:]: d[k] for k in d.files if k.startswith("tot_")}
    n = len(ids)

    import pandas as pd
    ann = pd.read_feather(
        ROOT / "upstream/doomfly/connectome_data/malecns_v1/annotations.feather"
    ).set_index("bodyId")
    id_to_row = {int(b): i for i, b in enumerate(ids)}

    # per-neuron left/right/center means (baseline-subtracted vs blind) + stats
    left = totals["left"]; right = totals["right"]; blind = totals["blind"]
    mb = blind.mean(0)
    lr_diff = (right.mean(0) - mb) - (left.mean(0) - mb)
    cohd = np.zeros(n); auc = np.full(n, 0.5)
    active = np.flatnonzero((left.sum(0) + right.sum(0)) > 0)
    for i in active:
        cohd[i] = _cohens_d(right[:, i], left[:, i])
        auc[i] = _auc(right[:, i], left[:, i])
    dir_strength = np.abs(auc - 0.5) * 2  # 0..1

    # map graph index -> superclass
    def superclass(i):
        bid = int(ids[i])
        if bid in ann.index:
            return str(ann.loc[bid]["superclass"])
        return "unknown"

    sup = np.array([superclass(i) for i in range(n)])

    # ---- stage summary: strong directional info per superclass ----
    stage_rows = []
    for s in sorted(set(sup[active])):
        mask = active[sup[active] == s]
        if len(mask) == 0:
            continue
        strong = mask[dir_strength[mask] > 0.9]         # near-perfect L/R sep
        good = mask[dir_strength[mask] > 0.6]
        stage_rows.append(dict(
            superclass=s, n_active=int(len(mask)),
            n_strong_auc_gt_0_9=int(len(strong)),
            n_good_auc_gt_0_6=int(len(good)),
            max_abs_d=round(float(np.abs(cohd[mask]).max()), 2),
            median_dir_strength=round(float(np.median(dir_strength[mask])), 3),
        ))
    stage_rows.sort(key=lambda r: -r["n_strong_auc_gt_0_9"])
    (OUT / "pathway_by_superclass.json").write_text(json.dumps(stage_rows, indent=2))
    print("=== directional info by superclass (stage) ===")
    for r in stage_rows:
        print(f"  {r['superclass']:20s} active={r['n_active']:5d} "
              f"strong(AUC>0.9)={r['n_strong_auc_gt_0_9']:4d} "
              f"good(>0.6)={r['n_good_auc_gt_0_6']:4d} max|d|={r['max_abs_d']}")

    # ---- descending-neuron screen ----
    dn = ann[ann["superclass"].astype(str).str.contains("descend", case=False, na=False)]
    ml = totals["mirrored_left"].mean(0) - mb
    mr = totals["mirrored_right"].mean(0) - mb
    mirror_diff = mr - ml
    static = totals["static"].mean(0) - mb
    rows = []
    for bid in dn.index:
        bid = int(bid)
        if bid not in id_to_row:
            continue
        i = id_to_row[bid]
        if (left[:, i].sum() + right[:, i].sum()) == 0:
            responsive = False
        else:
            responsive = True
        row = ann.loc[bid]
        rows.append(dict(
            body_id=bid, type=str(row["type"]), soma_side=str(row["somaSide"]),
            left=round(float(left[:, i].mean() - mb[i]), 2),
            center=round(float(totals["center"][:, i].mean() - mb[i]), 2),
            right=round(float(right[:, i].mean() - mb[i]), 2),
            lr_diff=round(float(lr_diff[i]), 2),
            cohens_d=round(float(cohd[i]), 2),
            auc=round(float(auc[i]), 3),
            dir_strength=round(float(dir_strength[i]), 3),
            mirror_diff=round(float(mirror_diff[i]), 2),
            blind=round(float(blind[:, i].mean()), 2),
            static=round(float(static[i]), 2),
            responsive=responsive,
        ))
    rows.sort(key=lambda r: -r["dir_strength"])
    (OUT / "descending_screen.json").write_text(json.dumps(rows, indent=2))

    n_dn = len(rows)
    n_resp = sum(r["responsive"] for r in rows)
    n_strong = sum(r["dir_strength"] > 0.9 for r in rows)
    n_good = sum(r["dir_strength"] > 0.6 for r in rows)
    print(f"\n=== descending-neuron screen ===")
    print(f"  {n_dn} DNs in graph, {n_resp} responsive, "
          f"{n_strong} strong (AUC>0.9), {n_good} good (>0.6)")
    print("  top 15 by directional strength:")
    for r in rows[:15]:
        print(f"   {r['body_id']:>8} {r['type']:>10} {r['soma_side']} "
              f"L={r['left']:+.2f} C={r['center']:+.2f} R={r['right']:+.2f} "
              f"AUC={r['auc']:.2f} d={r['cohens_d']:+.2f} mirrorΔ={r['mirror_diff']:+.2f}")

    # ---- diagnosis A/B/C/D ----
    retina_strong = next((r["n_strong_auc_gt_0_9"] for r in stage_rows
                          if r["superclass"] == "ol_sensory"), 0)
    ol_strong = sum(r["n_strong_auc_gt_0_9"] for r in stage_rows
                    if r["superclass"].startswith("ol_"))
    central_strong = sum(r["n_strong_auc_gt_0_9"] for r in stage_rows
                         if not r["superclass"].startswith("ol_")
                         and "descend" not in r["superclass"].lower())
    if retina_strong == 0:
        diagnosis = "D: retina does not encode shot direction adequately"
    elif n_strong > 0:
        diagnosis = "C/D: direction reaches descending neurons; current DNp20 readout misses it / motor mapping wrong"
    elif central_strong > 0:
        diagnosis = "B(partial): direction survives into central brain but not to descending neurons"
    elif ol_strong > 0:
        diagnosis = "C: direction present in optic lobe but lost before central/descending stages"
    else:
        diagnosis = "B: direction present only at retina; lost in CNS dynamics"
    summary = dict(retina_strong=int(retina_strong), optic_lobe_strong=int(ol_strong),
                   central_strong=int(central_strong),
                   descending_responsive=int(n_resp), descending_strong=int(n_strong),
                   descending_good=int(n_good), diagnosis=diagnosis)
    (OUT / "pathway_diagnosis.json").write_text(json.dumps(summary, indent=2))
    print("\n=== DIAGNOSIS ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
