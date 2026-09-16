"""Runtime-gain calibration for Arcade Bridge v2 (closed-loop config, not retrain).

The frozen v2 linear model is unchanged. Only the RUNTIME knobs are swept:
cmd_smoothing (the bridge EMA that most attenuates the short lateral signal),
lat_gain, and vert_gain. Calibration runs on a CALIBRATION seed set that is
DISJOINT from the held-out evaluation matrix (base_seed 90000), so no test shot
influences the chosen config. The winner is reported; the final closed-loop
comparison then runs on the untouched evaluation matrix.
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
                                                    run_episode)
from experiments.arcade_demo.arcade_bridge import (LinearBridge2D,
                                                   ArcadeLearnedBridge,
                                                   MetalSafeFeatureExtractor)
from experiments.arcade_demo.vertical_dn import ArcadeDNBasis

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


def eval_config(brain, cells, model_file, lat_gain, vert_gain, cmd_smoothing):
    man = np.load(ROOT / "workspace/outputs/learned_bridge/visual_manifest.npz")
    body = [int(x) for x in man["body_ids"][:207]]
    model = LinearBridge2D.load(OUT / model_file)
    fm = getattr(model, "loaded_metadata", {})
    ex = MetalSafeFeatureExtractor(body, n_windows=int(fm.get("n_windows", 4)))
    bridge = ArcadeLearnedBridge(ex, model, ArcadeDNBasis(), lat_gain=lat_gain,
                                 vert_gain=vert_gain, enabled=True,
                                 cmd_smoothing=cmd_smoothing)
    res = defaultdict(lambda: [0, 0])
    u_lat_by_group = defaultdict(list)
    for seed, g, hname, hf in cells:
        w = ArcadeGoalkeeperWorld(seed=seed)
        vision = VisionBridge(w.fly, brain, camera="eye_left", condition="normal")
        ctrl = ArcadeController(brain, vision, Arcade2AxisDecoder(), bridge=bridge)
        out, trace = run_episode(w, ctrl, w.sample_shot(g, height_frac=hf),
                                 collect_trace=True)
        res[g][0] += int(out["result"] == "SAVE"); res[g][1] += 1
        if trace:
            u_lat_by_group[g].append(float(np.mean([t["u_lat"] for t in trace])))
    saves = sum(v[0] for v in res.values()); n = sum(v[1] for v in res.values())
    left_sep = (np.mean(u_lat_by_group.get("right", [0]))
                - np.mean(u_lat_by_group.get("left", [0])))
    return dict(overall=round(saves / n, 3),
                by_group={g: f"{v[0]}/{v[1]}" for g, v in res.items()},
                u_lat_sep_right_minus_left=round(float(left_sep), 4),
                u_lat_by_group={g: round(float(np.mean(v)), 4)
                                for g, v in u_lat_by_group.items()})


def main():
    from adapters.brain import MaleCNSBrain
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=2)
    p.add_argument("--base-seed", type=int, default=70000,
                   help="calibration seeds (DISJOINT from eval base_seed 90000)")
    p.add_argument("--model", default="arcade_bridge_model_v2.npz")
    a = p.parse_args()
    cells = _cells(a.per_cell, a.base_seed)
    # sweep: lower smoothing (less lateral lag) + higher lat_gain
    grid = [(lg, vg, sm)
            for sm in (0.7, 0.2, 0.0)
            for lg in (3.5, 8.0)
            for vg in (4.5,)]
    brain = MaleCNSBrain(backend="metal")
    t0 = time.perf_counter()
    results = []
    print(f"[calibrate v2] {len(cells)} calib shots x {len(grid)} configs")
    for lg, vg, sm in grid:
        r = eval_config(brain, cells, a.model, lg, vg, sm)
        r.update(lat_gain=lg, vert_gain=vg, cmd_smoothing=sm)
        results.append(r)
        print(f"  lat={lg} vert={vg} sm={sm}: {r['overall']:.1%} "
              f"sep(R-L)={r['u_lat_sep_right_minus_left']:+.3f} {r['by_group']}",
              flush=True)
        # incremental save so partial progress survives
        (OUT / "calibrate_v2_partial.json").write_text(json.dumps(results, indent=2))
    brain.close()
    results.sort(key=lambda r: (r["overall"], r["u_lat_sep_right_minus_left"]),
                 reverse=True)
    best = results[0]
    report = dict(calibration_base_seed=a.base_seed, n_calib_shots=len(cells),
                  disjoint_from_eval_seed=90000, best=best, all=results,
                  wall_seconds=round(time.perf_counter() - t0, 1))
    (OUT / "calibrate_v2.json").write_text(json.dumps(report, indent=2))
    print(f"\nBEST: lat={best['lat_gain']} vert={best['vert_gain']} "
          f"sm={best['cmd_smoothing']} -> {best['overall']:.1%}")
    print(f"saved calibrate_v2.json ({report['wall_seconds']}s)")


if __name__ == "__main__":
    main()
