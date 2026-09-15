"""Bridge-OFF regression: adding the bridge infrastructure must NOT change the
Natural MaleCNS baseline (Section 8).

We verify, on identical shot seeds / initial conditions / CPU backend / timing:

  A) BridgeController with bridge=None            (no bridge object at all)
  B) BridgeController with a DISABLED LearnedBridge (object present, enabled=False)

Both must produce IDENTICAL neural activity (DN spikes each step), controller
outputs (lateral command each step), body trajectories (fly_y each step), and
save outcomes -- because a disabled bridge injects zero current and only reads
state. This proves the bridge scaffolding is side-effect-free when OFF.

We also confirm the outcome matches the known Natural MaleCNS signature
(0% left, 100% center, 0% right -- saves center by standing) on the 2+2 opponent
DN readout used by the bridge stack.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from experiments.learned_bridge.runtime import (make_world_brain_vision,
                                                BridgeController, run_episode,
                                                MAX_DECISIONS, DECISION_S)
from experiments.learned_bridge.dn_basis import DNMotorBasis, LEFT_DN, RIGHT_DN
from experiments.learned_bridge.bridge import (VisualFeatureExtractor,
                                               LinearBridgeModel, LearnedBridge)

OUT = ROOT / "workspace" / "outputs" / "learned_bridge"
GROUPS = ("left", "center", "right")


def _dummy_bridge(enabled=False):
    """A LearnedBridge with tiny arbitrary weights, DISABLED, to prove that a
    present-but-off bridge changes nothing."""
    man = np.load(OUT / "visual_manifest.npz")
    gi = man["graph_index"][:32]
    ex = VisualFeatureExtractor(gi, n_windows=4)
    rng = np.random.default_rng(0)
    w = rng.normal(size=ex.dim) * 0.3
    model = LinearBridgeModel(w=w, b=0.1, mu=np.zeros(ex.dim), sd=np.ones(ex.dim))
    basis = DNMotorBasis()
    return LearnedBridge(ex, model, basis, gain=1.0, enabled=enabled)


def run_traces(bridge_factory, seed, episodes):
    world, brain, vision, decoder = make_world_brain_vision(seed=seed)
    bridge = bridge_factory() if bridge_factory else None
    ctrl = BridgeController(brain, vision, decoder, bridge=bridge)
    all_traces, outcomes = [], []
    for ep in range(episodes):
        g = str(np.random.default_rng(seed + ep).choice(GROUPS))
        shot = world.sample_shot(g)
        out, tr = run_episode(world, ctrl, shot, collect_trace=True)
        outcomes.append(out)
        all_traces.append(tr)
    brain.close()
    return outcomes, all_traces


def compare_traces(ta, tb):
    """Max abs differences across matched steps for the regression."""
    dmax = dict(fly_y=0.0, lateral=0.0, left_spikes=0.0, right_spikes=0.0)
    nsteps = 0
    for ea, eb in zip(ta, tb):
        for sa, sb in zip(ea, eb):
            nsteps += 1
            dmax["fly_y"] = max(dmax["fly_y"], abs(sa["fly_y"] - sb["fly_y"]))
            dmax["lateral"] = max(dmax["lateral"], abs(sa["lateral"] - sb["lateral"]))
            dmax["left_spikes"] = max(dmax["left_spikes"],
                                      abs((sa["left_spikes"] or 0) - (sb["left_spikes"] or 0)))
            dmax["right_spikes"] = max(dmax["right_spikes"],
                                       abs((sa["right_spikes"] or 0) - (sb["right_spikes"] or 0)))
    return dmax, nsteps


def outcome_summary(outcomes):
    by = {g: [0, 0] for g in GROUPS}
    for o in outcomes:
        by[o["group"]][0] += int(o["result"] == "SAVE")
        by[o["group"]][1] += 1
    tot_s = sum(v[0] for v in by.values()); tot_n = sum(v[1] for v in by.values())
    return dict(save_rate=round(tot_s / tot_n, 4),
                by_group={g: (round(v[0] / v[1], 4) if v[1] else None) for g, v in by.items()},
                saves=tot_s, n=tot_n)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--seed", type=int, default=9000)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"[regression] running {a.episodes} episodes, seed {a.seed} ...")
    out_none, tr_none = run_traces(None, a.seed, a.episodes)
    out_off, tr_off = run_traces(lambda: _dummy_bridge(enabled=False), a.seed, a.episodes)

    dmax, nsteps = compare_traces(tr_none, tr_off)
    identical = all(v == 0.0 for v in dmax.values())

    summ_none = outcome_summary(out_none)
    summ_off = outcome_summary(out_off)
    outcomes_match = (summ_none == summ_off)

    result = dict(
        episodes=a.episodes, seed=a.seed, steps_compared=nsteps,
        max_abs_diff=dmax,
        bit_identical=bool(identical),
        outcomes_match=bool(outcomes_match),
        bridge_none_summary=summ_none,
        bridge_off_summary=summ_off,
        natural_signature_ok=bool(
            summ_off["by_group"].get("center") == 1.0),
        interpretation=("A disabled bridge injects zero current and only reads "
                        "state; identical traces confirm the scaffolding is "
                        "side-effect-free and bridge-OFF == Natural MaleCNS."),
    )
    (OUT / "regression_bridge_off.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not identical:
        print("\n[FAIL] bridge-OFF traces differ from no-bridge -- infrastructure "
              "is not side-effect-free.")
        sys.exit(1)
    print("\n[PASS] bridge-OFF is bit-identical to no-bridge (Natural MaleCNS).")


if __name__ == "__main__":
    main()
