"""Causal pathway stimulation A/B/C (no weight changes).

Injects external current (MaleCNSBrain.stimulate, <=30 mV, read-only w.r.t. the
graph) into one stage of a candidate route and measures whether the next stage
responds. This isolates exactly which transition transmits and which fails:

  A. stimulate directional visual (source) neurons  -> measure intermediates
  B. stimulate candidate intermediate neurons       -> measure descending neurons
  C. stimulate candidate descending neurons         -> (body already verified)

For each transition we report, for the downstream population: fraction that
spike, mean membrane depolarization vs an unstimulated baseline, and how many
cross threshold. A transition "works" if driving the upstream stage reliably
depolarizes / fires the downstream stage.
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


def load_stage_ids():
    pw = json.loads((OUT / "candidate_pathways.json").read_text())
    src, mid, dn = [], [], []
    for r in pw["top_routes"]:
        src.append(r["source"]["body_id"])
        mid.append(r["intermediate"]["body_id"])
        dn.append(r["descending"]["body_id"])
    # dedupe, preserve order
    def uniq(x):
        return list(dict.fromkeys(x))
    return uniq(src), uniq(mid), uniq(dn), pw["top_routes"]


class CausalStim:
    def __init__(self, seed=7):
        from adapters.brain import MaleCNSBrain
        self.brain = MaleCNSBrain(backend="cpu")
        self.b = self.brain._brain
        self.id_to_idx = {int(self.b.ids[i]): i for i in range(len(self.b.ids))}

    def stimulate_and_measure(self, drive_ids, measure_ids, current=25.0,
                              settle_ms=40.0, window_ms=5.0):
        """Reset, drive `drive_ids` with `current` mV each window for settle_ms,
        then report downstream `measure_ids` state."""
        self.brain.reset(); self.b = self.brain._brain
        midx = np.array([self.id_to_idx[i] for i in measure_ids])
        steps = int(round(settle_ms / window_ms))
        for _ in range(steps):
            if drive_ids:
                self.brain.stimulate(drive_ids, [current] * len(drive_ids))
            self.brain.step(window_ms)
        v = self.b.v[midx].copy()
        counts = self.b.counts[midx].copy()  # spikes in the last window
        # also accumulate spikes over a few extra windows
        total_spk = counts.astype(np.int64).copy()
        for _ in range(4):
            if drive_ids:
                self.brain.stimulate(drive_ids, [current] * len(drive_ids))
            self.brain.step(window_ms)
            total_spk += self.b.counts[midx]
        return dict(v=v, spikes_last=counts, spikes_5win=total_spk)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src, mid, dn, routes = load_stage_ids()
    cs = CausalStim(seed=7)

    def summ(stim, meas, tag):
        driven = cs.stimulate_and_measure(stim, meas)
        base = cs.stimulate_and_measure([], meas)  # no drive baseline
        dv = driven["v"] - base["v"]
        frac_spk = float((driven["spikes_5win"] > 0).mean())
        frac_spk_base = float((base["spikes_5win"] > 0).mean())
        return dict(
            transition=tag, n_driven=len(stim), n_measured=len(meas),
            downstream_frac_spiking=round(frac_spk, 3),
            baseline_frac_spiking=round(frac_spk_base, 3),
            mean_depol_mv=round(float(dv.mean()), 3),
            max_depol_mv=round(float(dv.max()), 3),
            n_crossed_threshold=int((driven["v"] > V_THRESHOLD).sum()),
            mean_downstream_v_driven=round(float(driven["v"].mean()), 2),
            mean_downstream_v_base=round(float(base["v"].mean()), 2),
        )

    result = {}
    # A: sources -> intermediates
    result["A_source_to_intermediate"] = summ(src, mid, "source->intermediate")
    # B: intermediates -> descending
    result["B_intermediate_to_descending"] = summ(mid, dn, "intermediate->descending")
    # B2: sources directly -> descending (2-hop end-to-end drive)
    result["B2_source_to_descending"] = summ(src, dn, "source->descending(2hop)")
    # C: descending -> (self reference: confirm they spike when driven)
    result["C_descending_selfdrive"] = summ(dn, dn, "descending->descending(ref)")

    (OUT / "causal_stim.json").write_text(json.dumps(result, indent=2))
    print("=== causal pathway stimulation (drive upstream, measure downstream) ===")
    for k, r in result.items():
        print(f"\n{r['transition']}: drove {r['n_driven']} -> measured {r['n_measured']}")
        print(f"  downstream frac spiking: {r['downstream_frac_spiking']} "
              f"(baseline {r['baseline_frac_spiking']})")
        print(f"  mean depol vs baseline: {r['mean_depol_mv']:+.3f} mV "
              f"(max {r['max_depol_mv']:+.3f})")
        print(f"  downstream crossed threshold: {r['n_crossed_threshold']}/{r['n_measured']}")
    print("\nsaved causal_stim.json")


if __name__ == "__main__":
    main()
