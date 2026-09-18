"""Phase 6: clean matched reset-corrected baselines on the CORRECTED environment.

After the shot-validity + geometry + loft + deadband fixes, old save rates on the
previous (broken) shot distribution are no longer authoritative. This harness
re-establishes the Arcade baseline honestly:

  * every evaluation cell is filtered through the shot-validity oracle
    (shot_validity.accept_cells) so ONLY CLEAN_GOAL shots are scored -- natural
    misses/posts/crossbars/stalls are excluded by construction.
  * matched seeds across all controllers (same shots for everyone).
  * full MaleCNS reset per episode (run_episode resets brain + controller).

Controllers:
  oracle    : privileged interception (physical ceiling)
  passive   : never moves (residual chance floor -- fly body still present)
  absent    : genuinely absent keeper (natural-miss sanity: should be ~0% saves)
  v2        : frozen arcade 2-output bridge (arcade_bridge_model_v2.npz)
  supervised: frozen binocular late-fusion hybrid (vision-only, both eyes)
  residual  : binocular action policy (residual mode) over the supervised base

Reported per controller: overall save%, 3x3 matrix, by-group, by-height,
keeper-contact/deflected/touch-but-goal rates, and movement metrics
(peak/mean lateral + vertical displacement, takeoff rate, airborne steps,
hop rate, movement onset latency, diagonal-jump frequency).
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
from experiments.arcade_demo.arcade_oracle import ArcadeOracleController
from experiments.arcade_demo import shot_validity as SV

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
GROUPS = ("left", "center", "right")
HEIGHTS = (("low", 0.0), ("mid", 0.5), ("high", 0.85))
AIRBORNE = ("TAKEOFF", "FLIGHT", "DIVE")
HOP_STATES = ("LOW_HOP", "TAKEOFF", "FLIGHT", "DIVE")


def build_valid_cells(per_cell, base_seed, verbose=True):
    """Build per_cell CLEAN_GOAL cells per (group,height), resampling as needed.

    Returns (cells, reject_summary). Guarantees natural-miss% == 0 on the eval.
    """
    cells = []
    rejects = defaultdict(lambda: defaultdict(int))
    seed = base_seed
    for g in GROUPS:
        for hname, hf in HEIGHTS:
            got = 0
            while got < per_cell:
                cls, _, _ = SV.classify_shot(seed, g, hf)
                if cls == SV.CLEAN_GOAL:
                    cells.append((seed, g, hname, hf)); got += 1
                else:
                    rejects[(g, hname)][cls] += 1
                seed += 1
    rej = {f"{g}/{h}": dict(c) for (g, h), c in sorted(rejects.items())}
    n_rej = sum(sum(c.values()) for c in rejects.values())
    if verbose:
        print(f"[valid-cells] built {len(cells)} CLEAN_GOAL cells "
              f"({per_cell}/cell); resampled past {n_rej} non-clean shots")
        if rej:
            print(f"[valid-cells] rejects: {rej}")
    return cells, rej


def movement_metrics(trace):
    """Movement descriptors from a controller episode trace."""
    if not trace:
        return {}
    y = np.array([r["fly_y"] for r in trace])
    z = np.array([r["fly_z"] for r in trace])
    states = [r.get("movement_state") for r in trace]
    uv = np.array([r.get("u_vert", 0.0) for r in trace])
    ul = np.array([r.get("u_lat", 0.0) for r in trace])
    lat_disp = np.abs(y - y[0])
    vert_disp = np.abs(z - z[0])
    airborne = [s in AIRBORNE for s in states]
    hop = [s in HOP_STATES for s in states]
    # movement onset: first step with >0.05 cm lateral OR >0.05 cm vertical disp
    onset = next((i for i in range(len(trace))
                  if lat_disp[i] > 0.05 or vert_disp[i] > 0.05), None)
    # diagonal jump: airborne AND meaningful lateral displacement simultaneously
    diagonal = any(airborne[i] and lat_disp[i] > 0.3 for i in range(len(trace)))
    return dict(
        peak_lat=float(lat_disp.max()), mean_lat=float(lat_disp.mean()),
        peak_vert=float(vert_disp.max()), mean_vert=float(vert_disp.mean()),
        takeoff=float(any(airborne)), airborne_steps=int(sum(airborne)),
        hop=float(any(hop)), onset_latency=(int(onset) if onset is not None else -1),
        diagonal_jump=float(diagonal),
    )


def _agg_movement(ms):
    if not ms:
        return {}
    keys = ms[0].keys()
    return {k: round(float(np.mean([m[k] for m in ms])), 4) for k in keys}


def summarize(res, contact, tbg, defl, absent_saves, ms):
    tot_s = sum(v[0] for v in res.values())
    tot_n = sum(v[1] for v in res.values())

    def axis(idx, keys):
        out = {}
        for key in keys:
            s = sum(v[0] for (g, h), v in res.items()
                    if (g if idx == 0 else h) == key)
            n = sum(v[1] for (g, h), v in res.items()
                    if (g if idx == 0 else h) == key)
            out[key] = dict(save_pct=round(s / n, 3) if n else None, saves=s, n=n)
        return out

    return dict(
        overall=round(tot_s / tot_n, 3), saves=tot_s, n=tot_n,
        by_group=axis(0, GROUPS),
        by_height=axis(1, ("low", "mid", "high")),
        by_cell={f"{g}/{h}": f"{v[0]}/{v[1]}" for (g, h), v in sorted(res.items())},
        keeper_contact_rate=round(contact / tot_n, 3),
        touch_but_goal_rate=round(tbg / tot_n, 3),
        deflected_save_rate=round(defl / tot_n, 3),
        natural_miss_pct=round(absent_saves / tot_n, 3),
        movement=_agg_movement(ms),
    )


def run_controller(cells, make_ctrl, collect_movement=True, brain=None):
    """Generic evaluation loop. make_ctrl(world, brain) -> controller."""
    res = defaultdict(lambda: [0, 0])
    contact = tbg = defl = 0
    ms = []
    for seed, g, hname, hf in cells:
        w = ArcadeGoalkeeperWorld(seed=seed)
        ctrl = make_ctrl(w, brain)
        out, tr = run_episode(w, ctrl, w.sample_shot(g, height_frac=hf),
                              collect_trace=collect_movement)
        res[(g, hname)][0] += int(out["result"] == "SAVE")
        res[(g, hname)][1] += 1
        contact += int(out.get("keeper_contact", False))
        d = w.outcome_diagnostics() if hasattr(w, "outcome_diagnostics") else {}
        tbg += int(bool(d.get("touch_but_goal", False)))
        defl += int(bool(d.get("deflected_save", False)))
        if collect_movement:
            ms.append(movement_metrics(tr))
    return res, contact, tbg, defl, ms


class _Passive:
    def reset(self): pass
    def act(self, world):
        return ({"forward": 0, "lateral": 0, "turn": 0, "gait_on": 0,
                 "vertical": 0, "move": "STAY"}, {})


def eval_absent(cells):
    """Genuinely-absent keeper natural-miss sanity: saves should be ~0."""
    saves = 0
    for seed, g, hname, hf in cells:
        w = ArcadeGoalkeeperWorld(seed=seed)
        shot = w.sample_shot(g, height_frac=hf)
        result, _ = SV.simulate_absent_keeper(w, shot)
        saves += int(result == "SAVE")
    return saves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=4)
    p.add_argument("--base-seed", type=int, default=90000)
    p.add_argument("--v2-model", default="arcade_bridge_model_v2.npz")
    p.add_argument("--policy", default="arcade_binocular_action_policy_90.npz")
    p.add_argument("--residual-scale", type=float, default=0.35)
    p.add_argument("--out", default="arcade_clean_baselines.json")
    p.add_argument("--only", default="", help="comma list to restrict controllers")
    a = p.parse_args()

    only = set(x for x in a.only.split(",") if x)
    t0 = time.perf_counter()
    cells, rejects = build_valid_cells(a.per_cell, a.base_seed)
    n = len(cells)
    print(f"[baselines] {n} matched CLEAN_GOAL shots "
          f"({a.per_cell}/cell, 3x3 cells)\n")
    result = dict(protocol=dict(
        retina_map="fullframe", eye_condition="both", matched_reset_corrected=True,
        clean_goal_only=True, n_shots=n, per_cell=a.per_cell,
        base_seed=a.base_seed, residual_scale=a.residual_scale),
        rejects=rejects)

    # natural-miss sanity (absent keeper) -- shared across all cells
    absent_saves = eval_absent(cells)
    result["natural_miss_saves_absent_keeper"] = f"{absent_saves}/{n}"
    print(f"  absent-keeper natural-miss saves: {absent_saves}/{n} "
          f"(must be ~0 for a valid eval)\n")

    def want(name):
        return not only or name in only

    # ---- oracle ----
    if want("oracle"):
        oc = ArcadeOracleController(lat_gain=3.0)
        rr = run_controller(cells, lambda w, b: oc)
        result["oracle"] = summarize(*rr[:4], absent_saves, rr[4])
        print(f"  oracle    : {result['oracle']['overall']:.1%}  "
              f"move(peakLat={result['oracle']['movement'].get('peak_lat')} "
              f"peakVert={result['oracle']['movement'].get('peak_vert')} "
              f"takeoff={result['oracle']['movement'].get('takeoff')})")

    # ---- passive (present fly, chance floor) ----
    if want("passive"):
        rr = run_controller(cells, lambda w, b: _Passive())
        result["passive"] = summarize(*rr[:4], absent_saves, rr[4])
        print(f"  passive   : {result['passive']['overall']:.1%} (present-fly floor)")

    # ---- neural controllers (share one brain) ----
    need_brain = want("v2") or want("supervised") or want("residual")
    if need_brain:
        from adapters.brain import MaleCNSBrain
        from embodiment.vision_bridge import VisionBridge
        from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
        from experiments.arcade_demo.binocular_lateral_hybrid import load as load_hybrid
        from experiments.arcade_demo.arcade_bridge import (LinearBridge2D,
                                                           ArcadeLearnedBridge,
                                                           MetalSafeFeatureExtractor)
        from experiments.arcade_demo.arcade_action_policy import (TinyPolicy,
                                                    BinocularActionPolicyBridge)
        from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
        brain = MaleCNSBrain(backend="metal")
        try:
            # v2 frozen arcade bridge (monocular eye_left, as historically frozen)
            if want("v2"):
                man = np.load(ROOT / "workspace/outputs/learned_bridge/visual_manifest.npz")
                body = [int(x) for x in man["body_ids"][:207]]

                def make_v2(w, b):
                    model = LinearBridge2D.load(OUT / a.v2_model)
                    fm = getattr(model, "loaded_metadata", {})
                    ex = MetalSafeFeatureExtractor(body,
                                                   n_windows=int(fm.get("n_windows", 4)))
                    bridge = ArcadeLearnedBridge(ex, model, ArcadeDNBasis(),
                                                 lat_gain=fm.get("lat_gain", 3.5),
                                                 vert_gain=fm.get("vert_gain", 3.5),
                                                 enabled=True,
                                                 cmd_smoothing=fm.get("cmd_smoothing", 0.7))
                    return ArcadeController(
                        b, VisionBridge(w.fly, b, camera="eye_left",
                                        condition="normal"),
                        Arcade2AxisDecoder(), bridge=bridge)
                rr = run_controller(cells, make_v2, brain=brain)
                result["v2"] = summarize(*rr[:4], absent_saves, rr[4])
                _print_neural("v2", result["v2"])

            # supervised binocular late-fusion hybrid (both eyes, fullframe)
            if want("supervised"):
                def make_sup(w, b):
                    visual, _ = load_hybrid("arcade_binocular_lateral_hybrid.npz",
                                            ArcadeDNBasis())
                    return ArcadeController(
                        b, BinocularVisionBridge(w.fly, b, condition="both",
                                                 retina_map="fullframe"),
                        Arcade2AxisDecoder(), visual)
                rr = run_controller(cells, make_sup, brain=brain)
                result["supervised"] = summarize(*rr[:4], absent_saves, rr[4])
                _print_neural("supervised", result["supervised"])

            # residual binocular action policy over the supervised base
            if want("residual"):
                def make_res(w, b):
                    visual, _ = load_hybrid("arcade_binocular_lateral_hybrid.npz",
                                            ArcadeDNBasis())
                    pol, _ = TinyPolicy.load(OUT / a.policy)
                    bridge = BinocularActionPolicyBridge(
                        visual, pol, residual_scale=a.residual_scale,
                        mode="residual")
                    return ArcadeController(
                        b, BinocularVisionBridge(w.fly, b, condition="both",
                                                 retina_map="fullframe"),
                        Arcade2AxisDecoder(), bridge)
                rr = run_controller(cells, make_res, brain=brain)
                result["residual"] = summarize(*rr[:4], absent_saves, rr[4])
                _print_neural("residual", result["residual"])
        finally:
            brain.close()

    result["wall_seconds"] = round(time.perf_counter() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / a.out).write_text(json.dumps(result, indent=2))
    print(f"\nsaved {a.out} ({result['wall_seconds']}s)")


def _print_neural(name, s):
    m = s["movement"]
    print(f"  {name:10s}: {s['overall']:.1%}  "
          f"(L {s['by_group']['left']['save_pct']} "
          f"C {s['by_group']['center']['save_pct']} "
          f"R {s['by_group']['right']['save_pct']} | "
          f"low {s['by_height']['low']['save_pct']} "
          f"mid {s['by_height']['mid']['save_pct']} "
          f"high {s['by_height']['high']['save_pct']}) "
          f"touch {s['keeper_contact_rate']} tbg {s['touch_but_goal_rate']} "
          f"| move peakLat {m.get('peak_lat')} peakVert {m.get('peak_vert')} "
          f"takeoff {m.get('takeoff')} hop {m.get('hop')} "
          f"diag {m.get('diagonal_jump')}")


if __name__ == "__main__":
    main()
