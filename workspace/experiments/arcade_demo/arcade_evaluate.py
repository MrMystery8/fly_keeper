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


def build_learned_bridge(brain, lat_gain=None, vert_gain=None, cmd_smoothing=None):
    """Load the FROZEN arcade 2-output bridge (no retrain). Metal-safe features."""
    man = np.load(ROOT / "workspace/outputs/learned_bridge/visual_manifest.npz")
    body = [int(x) for x in man["body_ids"][:207]]
    model = LinearBridge2D.load(OUT / "arcade_bridge_model.npz")
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


def eval_learned(cells, lat_gain=None, vert_gain=None):
    from adapters.brain import MaleCNSBrain
    brain = MaleCNSBrain(backend="metal")
    bridge, cfg = build_learned_bridge(brain, lat_gain=lat_gain, vert_gain=vert_gain)
    res = defaultdict(lambda: [0, 0])
    touched = 0
    for seed, g, hname, hf in cells:
        w = ArcadeGoalkeeperWorld(seed=seed)
        vision = VisionBridge(w.fly, brain, camera="eye_left", condition="normal")
        dec = Arcade2AxisDecoder()
        ctrl = ArcadeController(brain, vision, dec, bridge=bridge)
        out, _ = run_episode(w, ctrl, w.sample_shot(g, height_frac=hf))
        res[(g, hname)][0] += int(out["result"] == "SAVE")
        res[(g, hname)][1] += 1
        touched += int(out.get("keeper_contact", False))
    brain.close()
    return res, cfg, touched


def summarize(res):
    tot_s = sum(v[0] for v in res.values())
    tot_n = sum(v[1] for v in res.values())
    return dict(overall=round(tot_s / tot_n, 3), saves=tot_s, n=tot_n,
                by_cell={f"{g}/{h}": f"{v[0]}/{v[1]}"
                         for (g, h), v in sorted(res.items())})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=4)
    p.add_argument("--base-seed", type=int, default=90000)
    p.add_argument("--lat-gain", type=float, default=None)
    p.add_argument("--vert-gain", type=float, default=None)
    p.add_argument("--skip-oracle", action="store_true")
    a = p.parse_args()
    cells = _cells(a.per_cell, a.base_seed)
    print(f"[arcade-eval] {len(cells)} matched arcade shots "
          f"({a.per_cell}/cell, {len(GROUPS)}x{len(HEIGHTS)} cells)")
    t0 = time.perf_counter()
    result = {}

    if not a.skip_oracle:
        result["oracle"] = summarize(eval_oracle(cells))
        print(f"  oracle : {result['oracle']['overall']:.1%}")
    result["passive"] = summarize(eval_passive(cells))
    print(f"  passive: {result['passive']['overall']:.1%}")

    lres, cfg, touched = eval_learned(cells, lat_gain=a.lat_gain,
                                      vert_gain=a.vert_gain)
    result["learned"] = summarize(lres)
    result["learned"]["keeper_contact_rate"] = round(touched / len(cells), 3)
    result["learned_cfg"] = {k: cfg[k] for k in ("lat_gain", "vert_gain",
                                                 "cmd_smoothing")}
    print(f"  learned: {result['learned']['overall']:.1%} "
          f"(gains lat={cfg['lat_gain']} vert={cfg['vert_gain']}, "
          f"touch rate {result['learned']['keeper_contact_rate']})")
    for cell, v in result["learned"]["by_cell"].items():
        print(f"      {cell:14s} {v}")

    result["wall_seconds"] = round(time.perf_counter() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "arcade_evaluate.json").write_text(json.dumps(result, indent=2))
    print(f"saved arcade_evaluate.json ({result['wall_seconds']}s)")


if __name__ == "__main__":
    main()
