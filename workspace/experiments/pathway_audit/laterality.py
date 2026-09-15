"""Explicit laterality test of the candidate visual->DN pathway.

Two complementary tests:

1) NATURAL VISION (open loop): for genuinely lateral shots (left ball at +y,
   right ball at -y) and their mirror images, does a LEFT-preferring vs
   RIGHT-preferring descending population respond preferentially, and does the
   preference reverse under mirroring and vanish under blind? This is stronger
   than a firing-rate correlation because it demands sign consistency.

2) CAUSAL LATERALITY (stimulation): drive the LEFT-hemisphere directional
   visual neurons vs the RIGHT-hemisphere ones (external current) and measure
   whether left-side vs right-side descending neurons respond preferentially.
   This asks whether the ANATOMY supports lateralized routing at all, decoupled
   from whether natural vision recruits it.
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

RESPONSIVE_DN = {10059: ("DNp20", "R"), 10162: ("DNp20", "L"),
                 10527: ("DNpe017", "L"), 555871: ("DNpe017", "R")}


def natural_vision_laterality():
    """Compare responsive-DN L vs R spikes across left/right/mirror/blind."""
    from experiments.neural_probe.probe import NeuralProbe
    p = NeuralProbe(seed=7)
    b = p.brain
    id_to_idx = {int(p.ids[i]): i for i in range(len(p.ids))}
    left_dn = [10162, 10527]; right_dn = [10059, 555871]
    lidx = np.array([id_to_idx[i] for i in left_dn])
    ridx = np.array([id_to_idx[i] for i in right_dn])

    def run(group, transform, trials=6):
        Ls, Rs = [], []
        for _ in range(trials):
            seq = p._dynamic_luminance_sequence(group, 5.0, transform, 24, 0.005)
            b.reset(); bb = b._brain
            for _ in range(3):
                b.stimulate_retinal_luminance(p.retina_ids, seq[0]); b.step(5.0)
            lsum = rsum = 0
            for lum in seq:
                b.stimulate_retinal_luminance(p.retina_ids, lum); b.step(5.0)
                lsum += int(bb.counts[lidx].sum()); rsum += int(bb.counts[ridx].sum())
            Ls.append(lsum); Rs.append(rsum)
        return np.array(Ls), np.array(Rs)

    conds = {"left": ("left", "none"), "right": ("right", "none"),
             "mirror_left": ("left", "mirror"), "mirror_right": ("right", "mirror"),
             "blind": ("center", "blind")}
    out = {}
    for name, (grp, tf) in conds.items():
        L, R = run(grp, tf)
        out[name] = dict(left_dn_spikes=round(float(L.mean()), 2),
                         right_dn_spikes=round(float(R.mean()), 2),
                         lr_pref=round(float(R.mean() - L.mean()), 2))
    return out


def causal_laterality():
    """Drive left- vs right-hemisphere directional visual neurons; measure L/R DN."""
    from adapters.brain import MaleCNSBrain
    import pandas as pd
    ann = pd.read_feather(
        ROOT / "upstream/doomfly/connectome_data/malecns_v1/annotations.feather"
    ).set_index("bodyId")
    pw = json.loads((OUT / "candidate_pathways.json").read_text())
    # split strong directional source neurons by soma side
    left_src = [r["source"]["body_id"] for r in pw["top_routes"]
                if r["source"]["side"] == "L"]
    right_src = [r["source"]["body_id"] for r in pw["top_routes"]
                 if r["source"]["side"] == "R"]
    left_src = list(dict.fromkeys(left_src)); right_src = list(dict.fromkeys(right_src))

    brain = MaleCNSBrain(backend="cpu"); bb = brain._brain
    id_to_idx = {int(bb.ids[i]): i for i in range(len(bb.ids))}
    left_dn = [10162, 10527]; right_dn = [10059, 555871]
    lidx = np.array([id_to_idx[i] for i in left_dn])
    ridx = np.array([id_to_idx[i] for i in right_dn])

    def drive(src_ids):
        brain.reset(); b2 = brain._brain
        lsum = rsum = 0
        for _ in range(12):
            if src_ids:
                brain.stimulate(src_ids, [25.0] * len(src_ids))
            brain.step(5.0)
            lsum += int(b2.counts[lidx].sum()); rsum += int(b2.counts[ridx].sum())
        return lsum, rsum

    res = {}
    for name, ids in [("drive_left_visual", left_src),
                      ("drive_right_visual", right_src), ("drive_none", [])]:
        l, r = drive(ids)
        res[name] = dict(n_driven=len(ids), left_dn_spikes=l, right_dn_spikes=r,
                         lr_pref=r - l)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== natural-vision laterality (responsive DN L vs R spikes) ===")
    nat = natural_vision_laterality()
    for k, v in nat.items():
        print(f"  {k:14s} L={v['left_dn_spikes']:.1f} R={v['right_dn_spikes']:.1f} "
              f"R-L={v['lr_pref']:+.2f}")
    print("\n=== causal laterality (drive L vs R visual sources) ===")
    caus = causal_laterality()
    for k, v in caus.items():
        print(f"  {k:20s} n={v['n_driven']:3d} L_DN={v['left_dn_spikes']} "
              f"R_DN={v['right_dn_spikes']} R-L={v['lr_pref']:+d}")
    (OUT / "laterality.json").write_text(json.dumps(
        {"natural_vision": nat, "causal": caus}, indent=2))
    print("\nsaved laterality.json")


if __name__ == "__main__":
    main()
