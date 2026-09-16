"""Evaluate the arcade Learned Bridge (2-output) against oracle + baselines.

Runs the frozen arcade bridge through the SAME neural chain as Science Mode
(eye -> retina -> full MaleCNS -> learned bridge -> real DNs) but with the
enhanced flight motor + non-contact interception model. NO privileged ball state
reaches the neural controller; NO retraining. Metal backend.

Controllers compared on matched arcade shots (L/C/R x low/mid/high):
  oracle    : privileged interception (physical ceiling)
  passive   : never moves (chance floor)
  learned   : the frozen 2-output bridge (vision-only)
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

from embodiment.vision_bridge import VisionBridge
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import (ArcadeController,
                                                    Arcade2AxisDecoder,
                                                    run_episode, DECISION_S,
                                                    MAX_DECISIONS)
from experiments.arcade_demo.arcade_bridge import (LinearBridge2D,
                                                   ArcadeLearnedBridge,
                                                   MetalSafeFeatureExtractor)
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo.arcade_oracle import ArcadeOracleController

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
GROUPS = ("left", "center", "right")
HEIGHTS = (("low", 0.0), ("mid", 0.5), ("high", 0.85))


def _cells(per_cell, base_seed):
    cells, seed = [], base_seed
    for g in GROUPS:
        for hname, hf in HEIGHTS:
            for _ in range(per_cell):
                cells.append((seed, g, hname, hf)); seed += 1
    return cells


def build_learned_bridge(brain, lat_gain=None, vert_gain=None, cmd_smoothing=None,
                         model_file="arcade_bridge_model.npz"):
    """Load a FROZEN arcade 2-output bridge (no retrain). Metal-safe features."""
    man = np.load(ROOT / "workspace/outputs/learned_bridge/visual_manifest.npz")
    body = [int(x) for x in man["body_ids"][:207]]
    model = LinearBridge2D.load(OUT / model_file)
    fm = getattr(model, "loaded_metadata", {})
    ex = MetalSafeFeatureExtractor(body, n_windows=int(fm.get("n_windows", 4)))
    basis = ArcadeDNBasis()
    lg = lat_gain if lat_gain is not None else fm.get("lat_gain", 3.5)
    vg = vert_gain if vert_gain is not None else fm.get("vert_gain", 3.5)
    sm = cmd_smoothing if cmd_smoothing is not None else fm.get("cmd_smoothing", 0.7)
    bridge = ArcadeLearnedBridge(ex, model, basis, lat_gain=lg, vert_gain=vg,
                                 enabled=True, cmd_smoothing=sm)
    return bridge, dict(lat_gain=lg, vert_gain=vg, cmd_smoothing=sm, meta=fm)


def eval_oracle(cells):
    ctrl = ArcadeOracleController(lat_gain=3.0)
    res = defaultdict(lambda: [0, 0])
    for seed, g, hname, hf in cells:
        w = ArcadeGoalkeeperWorld(seed=seed)
        out, _ = run_episode(w, ctrl, w.sample_shot(g, height_frac=hf))
        res[(g, hname)][0] += int(out["result"] == "SAVE")
        res[(g, hname)][1] += 1
    return res


def eval_passive(cells):
    class Passive:
        def reset(self): pass
        def act(self, world):
            return ({"forward": 0, "lateral": 0, "turn": 0, "gait_on": 0,
                     "vertical": 0, "move": "STAY"}, {})
    ctrl = Passive()
    res = defaultdict(lambda: [0, 0])
    for seed, g, hname, hf in cells:
        w = ArcadeGoalkeeperWorld(seed=seed)
        out, _ = run_episode(w, ctrl, w.sample_shot(g, height_frac=hf))
        res[(g, hname)][0] += int(out["result"] == "SAVE")
        res[(g, hname)][1] += 1
    return res


def eval_learned(cells, lat_gain=None, vert_gain=None,
                 model_file="arcade_bridge_model.npz"):
    from adapters.brain import MaleCNSBrain
    brain = MaleCNSBrain(backend="metal")
    bridge, cfg = build_learned_bridge(brain, lat_gain=lat_gain,
                                       vert_gain=vert_gain, model_file=model_file)
    res = defaultdict(lambda: [0, 0])
    touched = touch_but_goal = deflected_save = 0
    # command diagnostics: u_lat by shot class, u_vert by shot height
    u_lat_by_group = defaultdict(list)
    u_vert_by_height = defaultdict(list)
    for seed, g, hname, hf in cells:
        w = ArcadeGoalkeeperWorld(seed=seed)
        vision = VisionBridge(w.fly, brain, camera="eye_left", condition="normal")
        dec = Arcade2AxisDecoder()
        ctrl = ArcadeController(brain, vision, dec, bridge=bridge)
        out, trace = run_episode(w, ctrl, w.sample_shot(g, height_frac=hf),
                                 collect_trace=True)
        res[(g, hname)][0] += int(out["result"] == "SAVE")
        res[(g, hname)][1] += 1
        touched += int(out.get("keeper_contact", False))
        touch_but_goal += int(bool(out.get("touch_but_goal", False)))
        diag = w.outcome_diagnostics() if hasattr(w, "outcome_diagnostics") else {}
        deflected_save += int(bool(diag.get("deflected_save", False)))
        if trace:
            u_lat_by_group[g].append(float(np.mean([t["u_lat"] for t in trace])))
            u_vert_by_height[hname].append(
                float(np.mean([t["u_vert"] for t in trace])))
    brain.close()
    diagnostics = dict(
        keeper_contact_rate=round(touched / len(cells), 3),
        touch_but_goal_rate=round(touch_but_goal / len(cells), 3),
        deflected_save_rate=round(deflected_save / len(cells), 3),
        u_lat_by_group={g: round(float(np.mean(v)), 4)
                        for g, v in sorted(u_lat_by_group.items())},
        u_vert_by_height={h: round(float(np.mean(v)), 4)
                          for h, v in sorted(u_vert_by_height.items())},
    )
    return res, cfg, diagnostics


def summarize(res):
    tot_s = sum(v[0] for v in res.values())
    tot_n = sum(v[1] for v in res.values())

    def _axis(idx, keys):
        out = {}
        for key in keys:
            s = sum(v[0] for (g, h), v in res.items()
                    if (g if idx == 0 else h) == key)
            n = sum(v[1] for (g, h), v in res.items()
                    if (g if idx == 0 else h) == key)
            out[key] = dict(save_pct=round(s / n, 3) if n else None,
                            saves=s, n=n)
        return out

    return dict(overall=round(tot_s / tot_n, 3), saves=tot_s, n=tot_n,
                by_group=_axis(0, ("left", "center", "right")),
                by_height=_axis(1, ("low", "mid", "high")),
                by_cell={f"{g}/{h}": f"{v[0]}/{v[1]}"
                         for (g, h), v in sorted(res.items())})


def _eval_one_bridge(name, cells, model_file, lat_gain, vert_gain):
    lres, cfg, diag = eval_learned(cells, lat_gain=lat_gain, vert_gain=vert_gain,
                                   model_file=model_file)
    summary = summarize(lres)
    summary.update(diag)
    summary["cfg"] = {k: cfg[k] for k in ("lat_gain", "vert_gain", "cmd_smoothing")}
    summary["model_file"] = model_file
    print(f"  {name:10s}: {summary['overall']:.1%}  "
          f"(L {summary['by_group']['left']['save_pct']} "
          f"C {summary['by_group']['center']['save_pct']} "
          f"R {summary['by_group']['right']['save_pct']} | "
          f"low {summary['by_height']['low']['save_pct']} "
          f"mid {summary['by_height']['mid']['save_pct']} "
          f"high {summary['by_height']['high']['save_pct']}) "
          f"touch {summary['keeper_contact_rate']} "
          f"touch-goal {summary['touch_but_goal_rate']}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=4)
    p.add_argument("--base-seed", type=int, default=90000)
    p.add_argument("--lat-gain", type=float, default=None)
    p.add_argument("--vert-gain", type=float, default=None)
    p.add_argument("--skip-oracle", action="store_true")
    p.add_argument("--v1-model", default="arcade_bridge_model_v1.npz")
    p.add_argument("--v2-model", default="arcade_bridge_model_v2.npz")
    p.add_argument("--out", default="arcade_evaluate_v1_vs_v2.json")
    a = p.parse_args()
    cells = _cells(a.per_cell, a.base_seed)
    print(f"[arcade-eval] {len(cells)} matched arcade shots "
          f"({a.per_cell}/cell, {len(GROUPS)}x{len(HEIGHTS)} cells)")
    t0 = time.perf_counter()
    result = dict(per_cell=a.per_cell, base_seed=a.base_seed, n_shots=len(cells))

    if not a.skip_oracle:
        result["oracle"] = summarize(eval_oracle(cells))
        print(f"  oracle    : {result['oracle']['overall']:.1%}")
    result["passive"] = summarize(eval_passive(cells))
    print(f"  passive   : {result['passive']['overall']:.1%}")

    result["v1"] = _eval_one_bridge("v1", cells, a.v1_model,
                                    a.lat_gain, a.vert_gain)
    result["v2"] = _eval_one_bridge("v2", cells, a.v2_model,
                                    a.lat_gain, a.vert_gain)

    print("\n  v2 3x3 cell matrix (saves/n):")
    for cell, v in result["v2"]["by_cell"].items():
        print(f"      {cell:14s} {v}")

    result["wall_seconds"] = round(time.perf_counter() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / a.out).write_text(json.dumps(result, indent=2))
    print(f"saved {a.out} ({result['wall_seconds']}s)")


if __name__ == "__main__":
    main()
