"""Closed-loop 3x3 evaluation: one-eye Arcade Bridge v2 vs binocular bridge.

Same arcade world, motor layer, DN basis, decoder, and 2-axis bridge machinery;
the ONLY differences between the two controllers are (a) the vision bridge
(one-eye VisionBridge vs BinocularVisionBridge condition="both") and (b) the
frozen linear model + its selected neurons. Oracle/passive included as ceiling/
floor. No privileged state reaches either neural controller.
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
                                                    Arcade2AxisDecoder, run_episode)
from experiments.arcade_demo.arcade_bridge import (LinearBridge2D,
                                                   ArcadeLearnedBridge,
                                                   MetalSafeFeatureExtractor)
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
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


def _summ(res):
    s = sum(v[0] for v in res.values()); n = sum(v[1] for v in res.values())
    def axis(idx, keys):
        return {k: round(sum(v[0] for (g, h), v in res.items()
                             if (g if idx == 0 else h) == k)
                         / max(1, sum(v[1] for (g, h), v in res.items()
                                      if (g if idx == 0 else h) == k)), 3)
                for k in keys}
    return dict(overall=round(s / n, 3), saves=s, n=n,
                by_group=axis(0, GROUPS),
                by_height=axis(1, ("low", "mid", "high")),
                by_cell={f"{g}/{h}": f"{v[0]}/{v[1]}" for (g, h), v in sorted(res.items())})


def _make_bridge(brain, model_file, body_ids, meta):
    model = LinearBridge2D.load(OUT / model_file)
    fm = getattr(model, "loaded_metadata", {})
    ex = MetalSafeFeatureExtractor(body_ids, n_windows=int(fm.get("n_windows", 4)))
    return ArcadeLearnedBridge(ex, model, ArcadeDNBasis(),
                               lat_gain=fm.get("lat_gain", 3.5),
                               vert_gain=fm.get("vert_gain", 4.5), enabled=True,
                               cmd_smoothing=fm.get("cmd_smoothing", 0.2))


def eval_neural(cells, kind):
    from adapters.brain import MaleCNSBrain
    brain = MaleCNSBrain(backend="metal")
    if kind == "v2":
        man = np.load(ROOT / "workspace/outputs/learned_bridge/visual_manifest.npz")
        body = [int(x) for x in man["body_ids"][:207]]
        bridge = _make_bridge(brain, "arcade_bridge_model_v2.npz", body, None)
        make_vision = lambda w: VisionBridge(w.fly, brain, camera="eye_left",
                                             condition="normal")
    else:  # binocular
        meta = json.load(open(OUT / "binocular_bridge_train.json"))["meta"]
        body = [int(x) for x in meta["selected_body_ids"]]
        bridge = _make_bridge(brain, "binocular_bridge_model.npz", body, meta)
        make_vision = lambda w: BinocularVisionBridge(w.fly, brain, condition="both")
    res = defaultdict(lambda: [0, 0]); touched = 0
    for seed, g, hname, hf in cells:
        w = ArcadeGoalkeeperWorld(seed=seed)
        ctrl = ArcadeController(brain, make_vision(w), Arcade2AxisDecoder(), bridge=bridge)
        out, _ = run_episode(w, ctrl, w.sample_shot(g, height_frac=hf))
        res[(g, hname)][0] += int(out["result"] == "SAVE"); res[(g, hname)][1] += 1
        touched += int(out.get("keeper_contact", False))
    brain.close()
    s = _summ(res); s["keeper_contact_rate"] = round(touched / len(cells), 3)
    return s


def eval_oracle(cells):
    ctrl = ArcadeOracleController(lat_gain=3.0)
    res = defaultdict(lambda: [0, 0])
    for seed, g, hname, hf in cells:
        w = ArcadeGoalkeeperWorld(seed=seed)
        out, _ = run_episode(w, ctrl, w.sample_shot(g, height_frac=hf))
        res[(g, hname)][0] += int(out["result"] == "SAVE"); res[(g, hname)][1] += 1
    return _summ(res)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=90000)
    a = p.parse_args()
    cells = _cells(a.per_cell, a.base_seed)
    print(f"[binocular-eval] {len(cells)} shots")
    t0 = time.perf_counter(); result = {"n_shots": len(cells)}
    result["oracle"] = eval_oracle(cells); print(f"  oracle     {result['oracle']['overall']:.1%}")
    result["v2_one_eye"] = eval_neural(cells, "v2")
    print(f"  v2_one_eye {result['v2_one_eye']['overall']:.1%} {result['v2_one_eye']['by_group']}")
    result["binocular"] = eval_neural(cells, "binocular")
    print(f"  binocular  {result['binocular']['overall']:.1%} {result['binocular']['by_group']}")
    result["wall_seconds"] = round(time.perf_counter() - t0, 1)
    (OUT / "binocular_evaluate.json").write_text(json.dumps(result, indent=2))
    print(f"saved binocular_evaluate.json ({result['wall_seconds']}s)")


if __name__ == "__main__":
    main()
