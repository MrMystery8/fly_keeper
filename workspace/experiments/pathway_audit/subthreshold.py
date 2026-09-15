"""Subthreshold pathway audit: is direction-dependent INPUT reaching the
descending neurons (and their intermediates) even when they do not spike?

Reuses the neural_probe dynamic-shot presentation (fly fixed, real moving-ball
retinal sequence). At the end of each control window we snapshot, for a set of
candidate neurons (visual sources, intermediates, descending), read-only:

  - membrane potential v           (rest -52 mV, threshold -45 mV)
  - conductance / synaptic drive g (excitatory synaptic input accumulator)
  - external+tonic drive           (drive[])
  - distance to threshold          (-45 - v)
  - refractory counter
  - spike count                    (counts[])

We then compare left vs right vs center vs blind and ask, per stage:
  * is there direction-dependent MEMBRANE modulation (even without spikes)?
  * how far below threshold does the useful signal sit?

The MaleCNS core is not modified; we only read its state arrays.
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

V_THRESHOLD = -45.0
V_REST = -52.0

RESPONSIVE_DN = {10059: "DNp20_R", 10162: "DNp20_L", 10527: "DNpe017_L", 555871: "DNpe017_R"}


def candidate_neurons():
    """Return {stage: [body_ids]} for the audit, from the ranked pathways."""
    pathways = json.loads((OUT / "candidate_pathways.json").read_text())
    sources, mids, dns = set(), set(), set()
    for r in pathways["top_routes"]:
        sources.add(r["source"]["body_id"])
        mids.add(r["intermediate"]["body_id"])
        dns.add(r["descending"]["body_id"])
    # ensure the responsive DNs are included
    dns |= set(RESPONSIVE_DN.keys())
    return {"source": sorted(sources), "intermediate": sorted(mids),
            "descending": sorted(dns)}


class SubthresholdProbe:
    def __init__(self, seed=7):
        from experiments.neural_probe.probe import NeuralProbe
        self.p = NeuralProbe(seed=seed)
        self.b = self.p.brain
        self.id_to_idx = {int(self.p.ids[i]): i for i in range(len(self.p.ids))}

    def trial(self, group, ids, speed=5.0, transform="none", frames=24,
              window_ms=5.0, warmup_ms=15.0):
        idx = np.array([self.id_to_idx[i] for i in ids])
        seq = self.p._dynamic_luminance_sequence(group, speed, transform, frames,
                                                 window_ms / 1000.0)
        self.b.reset()
        bb = self.b._brain
        for _ in range(int(round(warmup_ms / window_ms))):
            self.b.stimulate_retinal_luminance(self.p.retina_ids, seq[0])
            self.b.step(window_ms)
        # record v, g, drive, refractory, counts at each window end
        V = np.zeros((frames, len(idx)), np.float32)
        G = np.zeros((frames, len(idx)), np.float32)
        D = np.zeros((frames, len(idx)), np.float32)
        Rf = np.zeros((frames, len(idx)), np.float32)
        C = np.zeros((frames, len(idx)), np.int32)
        for w, lum in enumerate(seq):
            self.b.stimulate_retinal_luminance(self.p.retina_ids, lum)
            self.b.step(window_ms)
            V[w] = bb.v[idx]; G[w] = bb.g[idx]; D[w] = bb.drive[idx]
            Rf[w] = bb.refractory[idx]; C[w] = bb.counts[idx]
        return dict(v=V, g=G, drive=D, refractory=Rf, counts=C)


def summarize(stage_ids, per_cond, id_names=None):
    """Per-neuron direction-dependent membrane analysis over conditions."""
    rows = []
    for j, bid in enumerate(stage_ids):
        # mean over windows+trials of v/g for each condition
        def mv(cond, key):
            arr = np.stack([t[key][:, j] for t in per_cond[cond]])  # (trials,frames)
            return float(arr.mean())
        vL, vR, vC, vB = (mv("left", "v"), mv("right", "v"), mv("center", "v"), mv("blind", "v"))
        gL, gR, gB = (mv("left", "g"), mv("right", "g"), mv("blind", "g"))
        # peak (max over windows, mean over trials) membrane, and closest approach
        def peak_v(cond):
            arr = np.stack([t["v"][:, j] for t in per_cond[cond]])
            return float(arr.max())
        pkL, pkR = peak_v("left"), peak_v("right")
        spikesL = float(np.mean([t["counts"][:, j].sum() for t in per_cond["left"]]))
        spikesR = float(np.mean([t["counts"][:, j].sum() for t in per_cond["right"]]))
        rows.append(dict(
            body_id=int(bid), name=(id_names or {}).get(int(bid), ""),
            v_left=round(vL, 3), v_right=round(vR, 3), v_center=round(vC, 3), v_blind=round(vB, 3),
            v_lr_diff=round(vR - vL, 4),
            v_dir_vs_blind=round(((vR + vL) / 2) - vB, 4),
            dist_to_thresh_left=round(V_THRESHOLD - vL, 3),
            dist_to_thresh_right=round(V_THRESHOLD - vR, 3),
            peak_v_left=round(pkL, 3), peak_v_right=round(pkR, 3),
            g_left=round(gL, 4), g_right=round(gR, 4), g_lr_diff=round(gR - gL, 4),
            spikes_left=round(spikesL, 2), spikes_right=round(spikesR, 2),
        ))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cand = candidate_neurons()
    print("candidate counts:", {k: len(v) for k, v in cand.items()})
    probe = SubthresholdProbe(seed=7)

    # cap intermediates/sources to a manageable number by strongest routes order
    for stage in ("source", "intermediate", "descending"):
        cand[stage] = cand[stage][:60]

    trials = 5
    result = {}
    for stage in ("source", "intermediate", "descending"):
        ids = cand[stage]
        per_cond = {c: [] for c in ("left", "right", "center", "blind")}
        for cond in per_cond:
            tf = "blind" if cond == "blind" else "none"
            grp = "center" if cond == "blind" else cond
            for _ in range(trials):
                per_cond[cond].append(probe.trial(grp, ids, transform=tf))
        names = RESPONSIVE_DN if stage == "descending" else {}
        rows = summarize(ids, per_cond, names)
        # rank by |v_lr_diff|
        rows.sort(key=lambda r: -abs(r["v_lr_diff"]))
        result[stage] = rows
        print(f"\n=== {stage.upper()} (n={len(ids)}) top by |v left-right diff| ===")
        for r in rows[:8]:
            print(f"  {r['body_id']:>8}{(' '+r['name']) if r['name'] else '':<12} "
                  f"vL={r['v_left']:+.2f} vR={r['v_right']:+.2f} "
                  f"vLRdiff={r['v_lr_diff']:+.4f} dist2thr(L/R)={r['dist_to_thresh_left']:.2f}/"
                  f"{r['dist_to_thresh_right']:.2f} spk L/R={r['spikes_left']}/{r['spikes_right']}")

    (OUT / "subthreshold.json").write_text(json.dumps(result, indent=2))
    # focused summary on responsive DNs
    dn_rows = [r for r in result["descending"] if r["body_id"] in RESPONSIVE_DN]
    print("\n=== RESPONSIVE DNs subthreshold detail ===")
    for r in sorted(dn_rows, key=lambda r: r["name"]):
        print(f"  {r['name']:12} vL={r['v_left']:+.3f} vR={r['v_right']:+.3f} "
              f"vC={r['v_center']:+.3f} vBlind={r['v_blind']:+.3f} "
              f"vLRdiff={r['v_lr_diff']:+.4f} dirVsBlind={r['v_dir_vs_blind']:+.4f} "
              f"gLRdiff={r['g_lr_diff']:+.4f}")
    print("\nsaved subthreshold.json")


if __name__ == "__main__":
    main()
