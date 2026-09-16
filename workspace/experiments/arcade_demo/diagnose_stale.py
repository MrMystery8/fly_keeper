"""Independent verification of the stale-dataset diagnosis (Arcade Bridge v2).

Two independent checks, no model changes:

  A. STORED v1 dataset: reconstruct per-episode length from the stored `seeds`
     array and report the length distribution actually baked into v1.

  B. FRESH episodes: re-run the CURRENT arcade world (bridge OFF, exactly as the
     generator does) and, for each step, record ball x / result, then find:
       * terminal_step  = first step at which world.result != None
       * collected_steps = how many steps the generator's loop would collect
       * post_terminal   = collected_steps - terminal_step  (should be 0)
     This tells us whether the CURRENT generator still over-collects, and where
     the ~90-step v1 episodes came from.

Run on CPU for speed (episode length is backend-independent: the ball physics
and termination test do not depend on the neural backend).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "workspace") not in sys.path:
    sys.path.insert(0, str(ROOT / "workspace"))
if str(ROOT / "upstream" / "doomfly") not in sys.path:
    sys.path.insert(0, str(ROOT / "upstream" / "doomfly"))

from embodiment.mujoco_world import GOAL_LINE_X
from experiments.arcade_demo.arcade_world import ArcadeGoalkeeperWorld
from experiments.arcade_demo.arcade_runtime import DECISION_S, MAX_DECISIONS

OUT = ROOT / "workspace" / "outputs" / "arcade_demo"
GROUPS = ("left", "center", "right")
HEIGHTS = (0.0, 0.85)


def check_stored_v1(path="arcade_dataset_v1.npz"):
    p = OUT / path
    if not p.exists():
        p = OUT / "arcade_dataset.npz"
    d = np.load(p, allow_pickle=False)
    seeds = d["seeds"]
    steps = d["steps"]            # 0-based decision index within each episode
    # length of an episode = (max stored step index for that seed) + 1
    lens = []
    for s in np.unique(seeds):
        m = seeds == s
        lens.append(int(steps[m].max()) + 1)
    lens = np.array(lens)
    return dict(
        source=str(p.name),
        episodes=int(len(lens)),
        total_steps=int(len(seeds)),
        min=int(lens.min()), median=float(np.median(lens)),
        mean=round(float(lens.mean()), 2), max=int(lens.max()),
        hist=dict(sorted(Counter(lens.tolist()).items())),
        n_at_max_decisions=int((lens >= MAX_DECISIONS).sum()),
    )


def run_fresh(per_cell=3, base_seed=1000):
    """Reproduce the generator loop exactly (bridge OFF) but INSTRUMENTED.

    Records, per episode, the first step where result!=None (terminal_step) and
    the number of steps the current `while result is None and n<MAX` loop would
    collect. Also records what the result is and the ball x at terminal.
    """
    rows = []
    seed = base_seed
    for group in GROUPS:
        for hf in HEIGHTS:
            for _ in range(per_cell):
                world = ArcadeGoalkeeperWorld(seed=seed)
                shot = world.sample_shot(group, height_frac=hf)
                world.reset(shot)
                terminal_step = None
                collected = 0
                n = 0
                ball_x_at_terminal = None
                # mirror generator: collect a sample, THEN step, while result None
                while world.result is None and n < MAX_DECISIONS:
                    collected += 1               # a feature window is recorded
                    world.fly.set_command(0, 0, 0, lateral=0.0, vertical=0.0)
                    world.step(DECISION_S)
                    n += 1
                    if world.result is not None and terminal_step is None:
                        terminal_step = n
                        ball_x_at_terminal = float(world._observe_ball()["pos"][0])
                if terminal_step is None:
                    terminal_step = MAX_DECISIONS
                    ball_x_at_terminal = float(world._observe_ball()["pos"][0])
                rows.append(dict(
                    seed=int(seed), group=group, height=float(hf),
                    result=world.result, terminal_step=int(terminal_step),
                    collected=int(collected),
                    post_terminal=int(collected - terminal_step),
                    ball_x_at_terminal=round(ball_x_at_terminal, 3),
                    reached_max=bool(collected >= MAX_DECISIONS),
                ))
                seed += 1
    lens = np.array([r["collected"] for r in rows])
    post = np.array([r["post_terminal"] for r in rows])
    return dict(
        episodes=len(rows),
        collected_len=dict(min=int(lens.min()), median=float(np.median(lens)),
                           mean=round(float(lens.mean()), 2), max=int(lens.max())),
        post_terminal=dict(max=int(post.max()), mean=round(float(post.mean()), 3),
                           n_nonzero=int((post > 0).sum())),
        n_reached_max=int(sum(r["reached_max"] for r in rows)),
        results=dict(Counter(r["result"] for r in rows)),
        rows=rows,
    )


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=3)
    a = p.parse_args()
    report = {}
    print("[A] stored v1 dataset episode lengths ...")
    report["stored_v1"] = check_stored_v1()
    print(json.dumps(report["stored_v1"], indent=2))
    print("\n[B] fresh current-world episode termination ...")
    report["fresh_current"] = run_fresh(per_cell=a.per_cell)
    fc = {k: v for k, v in report["fresh_current"].items() if k != "rows"}
    print(json.dumps(fc, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnose_stale.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved {OUT/'diagnose_stale.json'}")


if __name__ == "__main__":
    main()
