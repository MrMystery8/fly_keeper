"""Final held-out evaluation: DAgger + residual-RL policy, movement, eye ablations.

Held-out CLEAN_GOAL 3x3 (seeds disjoint from training), matched across policies,
full MaleCNS reset per episode. Reports save%, natural-miss% (~0), 3x3 matrix,
keeper-contact/deflect/touch-but-goal, full movement metrics, and eye ablations
(both / left_blind / right_blind / both_blind) for the final policy to confirm
the saves are genuinely binocular-vision-dependent.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import (ArcadeController,
                                                    Arcade2AxisDecoder,
                                                    run_episode)
from experiments.arcade_demo import shot_validity as SV
from experiments.arcade_demo.arcade_baselines import (movement_metrics,
                                                      _agg_movement, summarize,
                                                      build_valid_cells)

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def _load_rl(path):
    """Load a residual-RL policy saved by train_arcade_residual_rl."""
    from experiments.arcade_demo.arcade_action_policy import TinyPolicy
    from experiments.arcade_demo.train_arcade_residual_rl import (RLResidualHead,
                                                                  RLResidualBridge)
    d = np.load(OUT / path, allow_pickle=False)
    meta = json.loads(str(d["metadata"][0]))
    head = RLResidualHead(d["w1"], d["b1"], d["w2"], d["b2"], d["mean"], d["scale"])
    alpha = float(d["alpha"][0]); base_scale = float(d["base_scale"][0])
    base_mode = str(d["base_mode"][0]) if "base_mode" in d else "residual"
    base_policy_file = str(d["base_policy"][0])
    base_policy, _ = TinyPolicy.load(OUT / base_policy_file)
    return head, alpha, base_scale, base_policy, base_mode, meta


def eval_policy(brain, cells, kind, policy_file=None, residual_scale=0.35,
               condition="both", rl_file=None):
    from embodiment.vision_bridge import VisionBridge
    from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
    from experiments.arcade_demo.binocular_lateral_hybrid import load as load_hybrid
    from experiments.arcade_demo.arcade_bridge import (LinearBridge2D,
                                                       ArcadeLearnedBridge,
                                                       MetalSafeFeatureExtractor)
    from experiments.arcade_demo.arcade_action_policy import (TinyPolicy,
                                                BinocularActionPolicyBridge)
    from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
    from experiments.arcade_demo.train_arcade_residual_rl import RLResidualBridge

    res = defaultdict(lambda: [0, 0]); contact = tbg = defl = 0; ms = []
    rl_pack = _load_rl(rl_file) if kind == "rl" else None
    for seed, g, hname, hf in cells:
        w = ArcadeGoalkeeperWorld(seed=seed)
        if kind == "v2":
            man = np.load(ROOT / "workspace/outputs/learned_bridge/visual_manifest.npz")
            body = [int(x) for x in man["body_ids"][:207]]
            model = LinearBridge2D.load(OUT / policy_file)
            fm = getattr(model, "loaded_metadata", {})
            ex = MetalSafeFeatureExtractor(body, n_windows=int(fm.get("n_windows", 4)))
            bridge = ArcadeLearnedBridge(ex, model, ArcadeDNBasis(),
                                         lat_gain=fm.get("lat_gain", 3.5),
                                         vert_gain=fm.get("vert_gain", 3.5),
                                         enabled=True,
                                         cmd_smoothing=fm.get("cmd_smoothing", 0.7))
            vision = VisionBridge(w.fly, brain, camera="eye_left", condition="normal")
        else:
            visual, _ = load_hybrid("arcade_binocular_lateral_hybrid.npz", ArcadeDNBasis())
            if kind == "dagger":
                pol, _ = TinyPolicy.load(OUT / policy_file)
                bridge = BinocularActionPolicyBridge(visual, pol,
                                                     residual_scale=residual_scale,
                                                     mode="residual")
            elif kind == "rl":
                head, alpha, base_scale, base_policy, base_mode, _ = rl_pack
                bridge = RLResidualBridge(visual, base_policy, head, alpha,
                                          base_scale, base_mode=base_mode)
            else:
                raise ValueError(kind)
            vision = BinocularVisionBridge(w.fly, brain, condition=condition,
                                           retina_map="fullframe")
        ctrl = ArcadeController(brain, vision, Arcade2AxisDecoder(), bridge)
        out, tr = run_episode(w, ctrl, w.sample_shot(g, height_frac=hf),
                              collect_trace=True)
        res[(g, hname)][0] += int(out["result"] == "SAVE"); res[(g, hname)][1] += 1
        contact += int(out.get("keeper_contact", False))
        d = w.outcome_diagnostics()
        tbg += int(bool(d.get("touch_but_goal", False)))
        defl += int(bool(d.get("deflected_save", False)))
        ms.append(movement_metrics(tr))
    return res, contact, tbg, defl, ms


def _absent(cells):
    s = 0
    for seed, g, hname, hf in cells:
        w = ArcadeGoalkeeperWorld(seed=seed)
        result, _ = SV.simulate_absent_keeper(w, w.sample_shot(g, height_frac=hf))
        s += int(result == "SAVE")
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=6)
    p.add_argument("--base-seed", type=int, default=200000, help="HELD-OUT seeds")
    p.add_argument("--dagger", default="arcade_dagger_v3_policy_r2.npz")
    p.add_argument("--rl", default="arcade_residual_rl.npz")
    p.add_argument("--v2", default="arcade_bridge_model_v2.npz")
    p.add_argument("--residual-scale", type=float, default=0.35)
    p.add_argument("--ablations", action="store_true")
    p.add_argument("--out", default="arcade_final_eval.json")
    a = p.parse_args()

    t0 = time.perf_counter()
    cells, _ = build_valid_cells(a.per_cell, a.base_seed)
    n = len(cells)
    absent = _absent(cells)
    print(f"[final] {n} HELD-OUT CLEAN_GOAL shots; natural-miss saves {absent}/{n}\n")
    result = dict(protocol=dict(held_out=True, base_seed=a.base_seed, n_shots=n,
                                clean_goal_only=True, retina_map="fullframe"),
                  natural_miss_saves=f"{absent}/{n}")

    from adapters.brain import MaleCNSBrain
    brain = MaleCNSBrain(backend="metal")
    try:
        for kind, kw in (("v2", dict(policy_file=a.v2)),
                         ("dagger", dict(policy_file=a.dagger,
                                         residual_scale=a.residual_scale)),
                         ("rl", dict(rl_file=a.rl))):
            rr = eval_policy(brain, cells, kind, **kw)
            s = summarize(*rr[:4], absent, rr[4])
            result[kind] = s
            m = s["movement"]
            print(f"  {kind:8s}: {s['overall']:.1%}  "
                  f"(L {s['by_group']['left']['save_pct']} "
                  f"C {s['by_group']['center']['save_pct']} "
                  f"R {s['by_group']['right']['save_pct']} | "
                  f"low {s['by_height']['low']['save_pct']} "
                  f"mid {s['by_height']['mid']['save_pct']} "
                  f"high {s['by_height']['high']['save_pct']}) "
                  f"touch {s['keeper_contact_rate']} | peakLat {m.get('peak_lat')} "
                  f"peakVert {m.get('peak_vert')} takeoff {m.get('takeoff')} "
                  f"hop {m.get('hop')} diag {m.get('diagonal_jump')} "
                  f"onset {m.get('onset_latency')}")

        if a.ablations:
            print("\n  eye ablations (final RL policy):")
            abl = {}
            for cond in ("both", "left_blind", "right_blind", "both_blind"):
                rr = eval_policy(brain, cells, "rl", rl_file=a.rl, condition=cond)
                s = summarize(*rr[:4], absent, rr[4])
                abl[cond] = dict(overall=s["overall"], by_group=s["by_group"],
                                 movement=s["movement"])
                print(f"    {cond:12s}: {s['overall']:.1%} "
                      f"(L {s['by_group']['left']['save_pct']} "
                      f"C {s['by_group']['center']['save_pct']} "
                      f"R {s['by_group']['right']['save_pct']})")
            result["eye_ablations"] = abl
    finally:
        brain.close()

    result["wall_seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / a.out).write_text(json.dumps(result, indent=2))
    print(f"\nsaved {a.out} ({result['wall_seconds']}s)")


if __name__ == "__main__":
    main()
