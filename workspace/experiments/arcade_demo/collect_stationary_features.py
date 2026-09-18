"""Collect a larger stratified STATIONARY corrected-env feature set for the
WHERE component (encoder + early-intent estimator), with per-step ball_x so we
can select the early window and measure held-out generalization honestly.

Stationary = keeper holds still (clean, uncorrupted view); this is exactly the
pre-motion sensing regime the early-intent estimator operates in.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path: sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path: sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))
from adapters.brain import MaleCNSBrain
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import DECISION_MS, DECISION_S, MAX_DECISIONS
from experiments.arcade_demo.binocular_vision import BinocularVisionBridge
from experiments.arcade_demo import shot_validity as SV
OUT = ROOT / "workspace" / "outputs" / "arcade_demo"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=20)   # 20x9 = 180 episodes
    p.add_argument("--base-seed", type=int, default=340000)
    p.add_argument("--out", default="stationary_features_big.npz")
    a = p.parse_args()
    t0 = time.perf_counter()
    m = np.load(OUT / "binocular_pool.npz", allow_pickle=False)
    pool_ids = [int(x) for x in m["body_ids"]]; side = m["somaSide"].astype(str)
    nw = 8
    print(f"[collect] building stratified CLEAN_GOAL shots ({a.per_cell}/cell x9)...")
    shots = SV.stratified_valid_shots(a.per_cell, a.base_seed)
    from collections import Counter
    print(f"[collect] {len(shots)} shots groups={dict(Counter(s.group for _,s in shots))}")
    brain = MaleCNSBrain(backend="metal")
    feats, groups, seeds, ballx, stepix = [], [], [], [], []
    try:
        for si, (seed, shot) in enumerate(shots):
            world = ArcadeGoalkeeperWorld(seed=seed)
            vision = BinocularVisionBridge(world.fly, brain, condition="both", retina_map="fullframe")
            world.reset(shot); brain.reset()
            buf = [np.zeros(len(pool_ids), np.float32) for _ in range(nw)]
            n = 0
            while world.shot_live and n < MAX_DECISIONS:
                vision.perceive(); brain.step(DECISION_MS)
                rd = brain.read(pool_ids)
                buf.append(np.array([rd[i]["spikes"] for i in pool_ids], np.float32)); buf.pop(0)
                feats.append(np.concatenate(buf[::-1]).astype(np.float32))
                groups.append(shot.group); seeds.append(seed)
                ballx.append(float(world._observe_ball()["pos"][0])); stepix.append(n)
                world.fly.set_command(0, 0, 0, lateral=0.0, vertical=0.0); world.step(DECISION_S); n += 1
            if (si + 1) % 20 == 0:
                print(f"  {si+1}/{len(shots)} episodes, {len(feats)} steps", flush=True)
    finally:
        brain.close()
    np.savez(OUT / a.out, features=np.asarray(feats, np.float32),
             groups=np.asarray(groups), seeds=np.asarray(seeds, np.int64),
             ball_x=np.asarray(ballx, np.float32), step_idx=np.asarray(stepix, np.int32),
             pool_body_ids=np.asarray(pool_ids, np.int64), pool_side=side,
             n_windows=np.array([nw], np.int32))
    print(f"[collect] saved {a.out}: {len(feats)} steps from {len(shots)} episodes "
          f"({round(time.perf_counter()-t0,1)}s)")


if __name__ == "__main__":
    main()
