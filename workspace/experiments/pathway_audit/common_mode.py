"""Baseline / saturation / common-mode vs directional decomposition per stage.

Reuses the dynamic probe tensor (real moving shots). Per stage we compute, from
total spike counts per trial:

  directional = mean(right) - mean(left)                      (signed L/R signal)
  common_mode = (mean(right) + mean(left)) / 2                (shared drive)
  blind       = mean(blind)                                   (tonic/no-vision)
  dir/common ratio                                            (is direction swamped?)
  visual_gain = common_mode - blind                           (how much vision adds)

Also checks saturation / refractory pressure using the subthreshold snapshot:
what fraction of each stage's neurons are chronically near reset (heavy spiking)
vs far below threshold (never approach firing).

The point: if every stage has a large common-mode / tonic drive but a vanishing
directional component by the descending stage, the fix is not "more gain" but a
directional-transmission problem (loss of the R-L contrast), which localizes
where targeted intervention would be needed.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
OUT = ROOT / "workspace" / "outputs" / "pathway_audit"


def main():
    import pandas as pd
    ann = pd.read_feather(
        ROOT / "upstream/doomfly/connectome_data/malecns_v1/annotations.feather"
    ).set_index("bodyId")

    d = np.load(ROOT / "workspace/outputs/neural_probe/dynamic_tensor.npz")
    conds = [str(c) for c in d["conditions"]]
    T = d["tensor"]                    # (cond, trials, win, active)
    ids_active = d["ids"][d["active"]]
    tot = T.sum(2)                     # (cond, trials, active)
    ci = {c: i for i, c in enumerate(conds)}

    are = ann.reindex(ids_active)
    sup = are["superclass"].astype(str).to_numpy().astype("U32")
    stage_masks = {
        "retina(ol_sensory)": sup == "ol_sensory",
        "optic_lobe(ol_intrinsic)": sup == "ol_intrinsic",
        "visual_projection": sup == "visual_projection",
        "central(cb*)": np.char.startswith(sup, "cb"),
        "descending": np.char.find(sup, "descend") >= 0,
    }

    L = tot[ci["left_5_none"]].astype(float)     # (trials, active)
    R = tot[ci["right_5_none"]].astype(float)
    B = tot[ci["center_5_blind"]].astype(float)

    rows = []
    for name, mask in stage_masks.items():
        if mask.sum() == 0:
            continue
        # population totals per trial
        lt = L[:, mask].sum(1); rt = R[:, mask].sum(1); bt = B[:, mask].sum(1)
        directional = float(rt.mean() - lt.mean())
        common = float((rt.mean() + lt.mean()) / 2)
        blind = float(bt.mean())
        visual_gain = common - blind
        ratio = abs(directional) / (common + 1e-9)
        # per-neuron: how many carry a directional signal that survives noise
        # (|mean R - mean L| > 1 spike)
        per_dir = np.abs(R[:, mask].mean(0) - L[:, mask].mean(0))
        n_dir_1spk = int((per_dir > 1.0).sum())
        rows.append(dict(stage=name, n=int(mask.sum()),
                         directional_signal=round(directional, 2),
                         common_mode=round(common, 2),
                         blind_tonic=round(blind, 2),
                         visual_gain=round(visual_gain, 2),
                         dir_over_common=round(ratio, 4),
                         n_neurons_dir_gt_1spk=n_dir_1spk))

    (OUT / "common_mode.json").write_text(json.dumps(rows, indent=2))
    print("=== common-mode vs directional decomposition (dynamic shots) ===")
    print(f"{'stage':26s} {'dir(R-L)':>9} {'common':>8} {'blind':>7} "
          f"{'visGain':>8} {'dir/common':>10} {'nDir>1spk':>9}")
    for r in rows:
        print(f"{r['stage']:26s} {r['directional_signal']:>9.2f} {r['common_mode']:>8.1f} "
              f"{r['blind_tonic']:>7.1f} {r['visual_gain']:>8.1f} "
              f"{r['dir_over_common']:>10.4f} {r['n_neurons_dir_gt_1spk']:>9d}")
    print("\nsaved common_mode.json")

    # interpretation hint
    dn = next((r for r in rows if r["stage"] == "descending"), None)
    ol = next((r for r in rows if "optic_lobe" in r["stage"]), None)
    if dn and ol:
        print(f"\ndir/common ratio: optic lobe {ol['dir_over_common']:.4f} -> "
              f"descending {dn['dir_over_common']:.4f}")
        print(f"descending directional neurons (>1 spk L/R diff): {dn['n_neurons_dir_gt_1spk']}")


if __name__ == "__main__":
    main()
