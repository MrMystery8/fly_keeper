"""Phase 3: same-seed determinism + Phase 2 oracle-reachability.

(A) DETERMINISM: for a representative set of shot seeds, run the EXACT SAME
    fully-reset episode several times with the same controller and check that
    outcomes and trajectories are stable. Large SAVE<->GOAL flips from an
    identical fully-reset setup must be understood before RL.

    Two controllers are checked:
      * oracle  (privileged, no MaleCNS) -> isolates MuJoCo/RNG/body determinism
      * neural  (full binocular MaleCNS bridge) -> adds brain-reset determinism

(B) ORACLE REACHABILITY: confirm the privileged oracle keeper can SAVE every
    nominal cell on the recalibrated (CLEAN_GOAL) shot distribution -- i.e. the
    valid shots are all physically reachable.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))

from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import (ArcadeController,
                                                    Arcade2AxisDecoder,
                                                    run_episode)
from experiments.arcade_demo.arcade_oracle import ArcadeOracleController

GROUPS = ("left", "center", "right")
HEIGHTS = (("low", 0.0), ("mid", 0.5), ("high", 0.85))


def _cells(per_cell, base_seed):
    cells, seed = [], base_seed
    for g in GROUPS:
        for hname, hf in HEIGHTS:
            for _ in range(per_cell):
                cells.append((seed, g, hname, hf)); seed += 1
    return cells


def _final_state(world):
    b = world._observe_ball()
    return (world.result, round(float(world.fly.position[1]), 4),
            round(float(world.fly.position[2]), 4),
            round(float(b["pos"][0]), 4), round(float(b["pos"][1]), 4),
            round(float(b["pos"][2]), 4))


def determinism_oracle(cells, repeats):
    ctrl = ArcadeOracleController(lat_gain=3.0)
    flips, traj_var, rows = 0, [], []
    save_by_cell = defaultdict(lambda: [0, 0])
    for seed, g, hname, hf in cells:
        results, states = [], []
        for _ in range(repeats):
            w = ArcadeGoalkeeperWorld(seed=seed)
            out, _ = run_episode(w, ctrl, w.sample_shot(g, height_frac=hf))
            results.append(out["result"]); states.append(_final_state(w))
            save_by_cell[(g, hname)][0] += int(out["result"] == "SAVE")
            save_by_cell[(g, hname)][1] += 1
        n_save = sum(r == "SAVE" for r in results)
        flipped = 0 < n_save < repeats
        flips += int(flipped)
        # trajectory spread: stdev of final fly-y across repeats
        fy = [s[1] for s in states]
        traj_var.append(float(np.std(fy)))
        if flipped:
            rows.append((seed, g, hname, results))
    return dict(flips=flips, n=len(cells),
                max_final_fly_y_std=round(max(traj_var), 5),
                mean_final_fly_y_std=round(float(np.mean(traj_var)), 5),
                flipped_cells=rows[:20],
                oracle_save_by_cell={f"{g}/{h}": f"{v[0]}/{v[1]}"
                                     for (g, h), v in sorted(save_by_cell.items())})


def determinism_neural(cells, repeats):
    from adapters.brain import MaleCNSBrain
    from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
    from experiments.arcade_demo.binocular_lateral_hybrid import load as load_visual
    from experiments.arcade_demo.vertical_dn import ArcadeDNBasis
    brain = MaleCNSBrain(backend="metal")
    flips, traj_var, rows = 0, [], []
    try:
        for seed, g, hname, hf in cells:
            results, states = [], []
            for _ in range(repeats):
                w = ArcadeGoalkeeperWorld(seed=seed)
                visual, _ = load_visual("arcade_binocular_lateral_hybrid.npz",
                                        ArcadeDNBasis())
                ctrl = ArcadeController(
                    brain,
                    BinocularVisionBridge(w.fly, brain, condition="both",
                                          retina_map="fullframe"),
                    Arcade2AxisDecoder(), visual)
                out, _ = run_episode(w, ctrl, w.sample_shot(g, height_frac=hf))
                results.append(out["result"]); states.append(_final_state(w))
            n_save = sum(r == "SAVE" for r in results)
            flipped = 0 < n_save < repeats
            flips += int(flipped)
            fy = [s[1] for s in states]
            traj_var.append(float(np.std(fy)))
            if flipped:
                rows.append((seed, g, hname, results))
    finally:
        brain.close()
    return dict(flips=flips, n=len(cells),
                max_final_fly_y_std=round(max(traj_var), 5),
                mean_final_fly_y_std=round(float(np.mean(traj_var)), 5),
                flipped_cells=rows[:20])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=90000)
    p.add_argument("--skip-neural", action="store_true")
    a = p.parse_args()
    cells = _cells(a.per_cell, a.base_seed)
    print(f"[determinism] {len(cells)} cells x {a.repeats} repeats each")

    orep = determinism_oracle(cells, a.repeats)
    print(f"\n=== ORACLE determinism + reachability ===")
    print(f"SAVE<->GOAL flips: {orep['flips']}/{orep['n']} cells  "
          f"max_final_fly_y_std={orep['max_final_fly_y_std']}")
    print("oracle save by cell (reachability):")
    for k, v in orep["oracle_save_by_cell"].items():
        print(f"  {k:14s} {v}")
    if orep["flipped_cells"]:
        print("  flipped:", orep["flipped_cells"])

    result = dict(per_cell=a.per_cell, repeats=a.repeats, oracle=orep)
    if not a.skip_neural:
        nrep = determinism_neural(cells, a.repeats)
        print(f"\n=== NEURAL (binocular MaleCNS) determinism ===")
        print(f"SAVE<->GOAL flips: {nrep['flips']}/{nrep['n']} cells  "
              f"max_final_fly_y_std={nrep['max_final_fly_y_std']}  "
              f"mean={nrep['mean_final_fly_y_std']}")
        if nrep["flipped_cells"]:
            print("  flipped:", nrep["flipped_cells"])
        result["neural"] = nrep

    out = ROOT / "workspace" / "outputs" / "arcade_demo"
    out.mkdir(parents=True, exist_ok=True)
    (out / "determinism_check.json").write_text(json.dumps(result, indent=2))
    print("\nsaved determinism_check.json")


if __name__ == "__main__":
    main()
